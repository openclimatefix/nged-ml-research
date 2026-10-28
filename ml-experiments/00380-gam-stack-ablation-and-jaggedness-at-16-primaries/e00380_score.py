"""Score the ablation arms of experiment 380 under demand-v1 and print every number quoted.

Reads the arms of `e00380_arms.py` and the references of experiment 87 (as experiment 377 does).
Writes the per-Primary stack-minus-BG2 gap, used by `e00380_jaggedness.py`.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    uv run --no-sync python ml-experiments/00380-*/e00380_score.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OWN = ML_EXPERIMENTS_DATA_DIR / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries" / "work"
ARM_NAMES = ("full", "drop_k", "drop_o", "build3", "drop_f", "build0", "build1", "build2")
REFS = ("bg2", "manual_heuristic", "climatology")
BANDS = (base.HEADLINE_BAND, "lead_day_1", base.LATE_BAND, base.BEYOND_BAND)
# planned contrasts: treatment, reference, label
CONTRASTS = (
    ("drop_k", "full", "remove Kalman filter"),
    ("drop_o", "full", "remove winter offset"),
    ("build3", "full", "remove recipe R"),
    ("drop_f", "full", "remove per-fold spread"),
    ("build1", "build0", "add per-fold spread to plain GAM"),
    ("build2", "build1", "add Kalman filter"),
    ("build3", "build2", "add winter offset"),
    ("full", "build3", "add recipe R"),
)
GROUPS = {
    "signed-MW": pl.col("group") == "signed-MW",
    "MVA-only with D8": pl.col("group") == "MVA-only",
    "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
    "short-history (report only)": pl.col("group") == "short-history",
}


def arm_frame(*, name: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return an ablation arm's forecasts for all 16 Primaries in MW."""
    pieces = []
    for label in base.STACK_SITES:
        one = series.filter(pl.col("label") == label)
        stored = pl.read_parquet(OWN / "arms" / f"{name}_{label}.parquet").filter(
            pl.col("power_fcst_init_time") >= base.SCORED_FIRST
        )
        pieces.append(adapter.forecast_frame(stored=stored, series=one, unit="points"))
    return pl.concat(pieces)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the scores."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )  # fmt: skip
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=base.SCORED_FIRST
    )
    arms = {n: arm_frame(name=n, series=series) for n in ARM_NAMES}
    arms.update(base.reference_frames(series=series))
    table = metrics.shared_slice_table(
        forecasts=arms, actuals=actuals, context=context, events=events,
        first_readings=first_readings,
    )  # fmt: skip
    print(f"shared half-hours of all {len(arms)} arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    for band in BANDS:
        parts = [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == band).select(
                "label", "group", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group"), on="label")
        print(f"\n== CRPS (points) per Primary, slice {band} ==")
        print(joined.sort("group", "label"))
        print(
            joined.group_by("group").agg(pl.exclude("label").mean()).sort("group"),
            "<- mean over Primaries per group",
        )
    print("\n== Planned contrasts (99%, 20,000 resamples), treatment - reference, CRPS ==")
    for band in BANDS:
        for treat, ref, text in CONTRASTS:
            for gname, where in GROUPS.items():
                if band != base.HEADLINE_BAND and (
                    (treat, ref, text) not in CONTRASTS[:4]
                    or "MVA-only without" in gname
                    or "short" in gname
                ):
                    continue  # other bands: the four stage contrasts, signed-MW and MVA-only
                r = metrics.headline_month_bootstrap_contrast(
                    treatment=rows[treat], reference=rows[ref], metric="crps",
                    where=where & pl.col(band), slice_table=table,
                )  # fmt: skip
                print(
                    f"{band:15s} {text:36s} {gname:28s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
    print("\n== Full stack minus each reference (99%), days 1 to 10 ==")
    for ref in REFS:
        for gname, where in GROUPS.items():
            r = metrics.headline_month_bootstrap_contrast(
                treatment=rows["full"], reference=rows[ref], metric="crps",
                where=where & pl.col(base.HEADLINE_BAND), slice_table=table,
            )  # fmt: skip
            print(
                f"full - {ref:16s} {gname:28s} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f})"
            )
    print("\n== Plain GAM (build0) minus references (99%), days 1 to 10 ==")
    for ref in REFS:
        for gname, where in GROUPS.items():
            r = metrics.headline_month_bootstrap_contrast(
                treatment=rows["build0"], reference=rows[ref], metric="crps",
                where=where & pl.col(base.HEADLINE_BAND), slice_table=table,
            )  # fmt: skip
            print(
                f"build0 - {ref:16s} {gname:28s} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f})"
            )
    print("\n== Per-Primary contrasts (95%, 2,000 resamples), days 1 to 10 ==")
    for treat, ref, text in CONTRASTS:
        print(text, f"({treat} - {ref})")
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows[treat], reference=rows[ref], metric="crps",
                where=pl.col(base.HEADLINE_BAND), slice_table=table,
            )
        )  # fmt: skip
    print("\n== Leave-one-out, planned stage contrasts (95%), days 1 to 10 ==")
    for treat, ref, text in CONTRASTS[:4]:
        for gname, labels in (
            ("signed-MW", ["D5", "D9", "D10", "D11", "D14", "D16"]),
            ("MVA-only", ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
        ):
            for left_out in labels:
                keep = pl.col("label").is_in([x for x in labels if x != left_out])
                r = metrics.month_bootstrap_contrast(
                    treatment=rows[treat], reference=rows[ref], metric="crps",
                    where=keep & pl.col(base.HEADLINE_BAND), slice_table=table,
                )  # fmt: skip
                print(
                    f"{text:28s} {gname:10s} without {left_out:4s} {r['difference']:+.3f} "
                    f"({r['lower']:+.3f} to {r['upper']:+.3f})"
                )
    print("\n== Tail measures, slice lead_days_1_10, mean over Primaries per group ==")
    tails = ["pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99", "coverage_q01",
             "coverage_q05", "coverage_q95", "coverage_q99", "twcrps", "mae_median"]  # fmt: skip
    for name in (*ARM_NAMES, "bg2"):
        t = scored[name].filter(
            pl.col("slice") == base.HEADLINE_BAND, pl.col("metric").is_in(tails)
        )
        print(name)
        print(
            t.group_by("group", "metric").agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value").sort("group")
        )  # fmt: skip
    print("\n== Slice CRPS pooled by group, days 1 to 10: full, drop_k, drop_o, build3, bg2 ==")
    for sl in ("sunny_unit_shape", "sunny_april_june", "winter_nights", "cold_nights",
               "economy_7_step", "christmas", "event_rows"):  # fmt: skip
        parts = []
        for name in ("full", "drop_k", "drop_o", "build3", "drop_f", "build0", "bg2"):
            t = rows[name].filter(pl.col(sl), pl.col(base.HEADLINE_BAND))
            parts.append(t.group_by("group").agg(pl.col("crps").mean().alias(name)))
        out = parts[0]
        for p in parts[1:]:
            out = out.join(p, on="group")
        print(sl)
        print(out.sort("group"))
    print("\n== Median minus mean of each arm per site (points), scored year ==")
    diffs = []
    for name in ARM_NAMES:
        for site in base.STACK_SITES:
            f = pl.read_parquet(OWN / "arms" / f"{name}_{site}.parquet").filter(
                pl.col("power_fcst_init_time") >= base.SCORED_FIRST
            )
            diffs.append(
                f.select(
                    arm=pl.lit(name),
                    site=pl.lit(site),
                    diff=(pl.col("q50") - pl.col("mean")).mean(),
                )
            )
    print(pl.concat(diffs).pivot(on="arm", index="site", values="diff").sort("site"))
    print("\n== Control: full vs experiment 377's stored stack, max abs quantile difference ==")
    for site in base.STACK_SITES:
        mine = pl.read_parquet(OWN / "arms" / f"full_{site}.parquet").filter(
            pl.col("power_fcst_init_time") >= base.SCORED_FIRST
        )
        stored = pl.read_parquet(base.stage_2_path(site=site))
        j = mine.join(
            stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_s"
        )
        print(
            site,
            mine.height,
            stored.height,
            j.height,
            float((j["q50"] - j["q50_s"]).abs().max()),  # ty: ignore[invalid-argument-type]
            float((j["q95"] - j["q95_s"]).abs().max()),  # ty: ignore[invalid-argument-type]
        )
    gap = (
        scored["full"].filter(pl.col("slice") == base.HEADLINE_BAND, pl.col("metric") == "crps")
        .select("label", "group", full=pl.col("value"))
        .join(
            scored["bg2"].filter(pl.col("slice") == base.HEADLINE_BAND, pl.col("metric") == "crps")
            .select("label", bg2=pl.col("value")), on="label",
        )
        .with_columns(gap=pl.col("full") - pl.col("bg2"))
        .sort("label")
    )  # fmt: skip
    gap.write_parquet(OWN / "gap_stack_minus_bg2.parquet")
    print("\n== Gap (full minus BG2, days 1 to 10) per Primary ==")
    print(gap)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
