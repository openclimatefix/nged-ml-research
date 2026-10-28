# ruff: noqa: E501, PLR0915
"""Score Primary demand research default 3 at all 16 Primaries under demand-v1.

Arms, each with leads 0.5 to 240 hours:

- `default_3`: the solar summary S below 168 hours, and at 168 to 240 hours S plus the heuristic
  blend on daylight half-hours.
- `default_1`: S below 168 hours and experiment 377's stage 2 at 168 to 240 hours (Primary demand
  research default 1 extended to 240 hours, the faithful default 1).
- `stack_without_s`: experiment 377's stage 2, which has no S at any lead.
- The stored `bg2`, `climatology`, and `manual_heuristic` of experiment 87.

D8 uses the donor spread in every arm (the protocol's labelled exception). MVA-only Primaries use
experiment 377's naive fit, "reactive power assumed zero". D15 and D18 are report only.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00378-*/e00378_score.py
"""

from datetime import UTC, datetime

import e00377_adapter as adapter
import e00378_arms as arms
import e00378_common as common
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
Q = list(common.QUANTILES)
SCORED_FIRST = datetime(2025, 7, 1, tzinfo=UTC)
CACHE_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
HEADLINE_BAND = "lead_days_1_10"
LATE_BAND = "lead_days_8_10"
LABELS = {"signed-MW": "signed-MW", "MVA-only": "MVA-only"}


def stored_arms(*, site: str) -> dict[str, pl.DataFrame]:
    """Return the three arms' stored frames (points, `q01` to `q99`) of one site, leads to 240 hours."""
    path = common.stage_path(site=site, stage="dp_preds_s2")
    if site == "D8":
        path = path.with_name("stack_d_donor_D8.parquet")
    base = pl.read_parquet(path).filter(
        (pl.col("seed") == 0) & (pl.col("lead_hours") <= common.HORIZON_HOURS)
    )
    source = arms.arms_source(site)
    low = pl.read_parquet(common.ARMS_DIR / f"low_lead_{source}.parquet").filter(pl.col("site") == site)
    long = pl.read_parquet(common.ARMS_DIR / f"long_lead_{source}.parquet").filter(pl.col("site") == site)
    time_keys = KEYS[1:]
    low_x1 = low.select(*time_keys, *[pl.col(f"x1_{q}").alias(f"low_{q}") for q in Q])
    long_d3 = long.select(*time_keys, *[pl.col(f"d3_{q}") for q in Q])
    # Experiment 377's frames carry `time_series_id`; the arms' files do not, so join on the two
    # time keys, site by site.
    joined = base.join(low_x1, on=time_keys, how="left").join(long_d3, on=time_keys, how="left")
    short = pl.col("lead_hours") <= common.STORED_HORIZON_HOURS
    out = {"stack_without_s": base.select(*KEYS, *Q)}
    out["default_1"] = joined.select(
        *KEYS, *[pl.when(short).then(pl.col(f"low_{q}")).otherwise(pl.col(q)).alias(q) for q in Q]
    )
    out["default_3"] = joined.select(
        *KEYS,
        *[
            pl.when(short).then(pl.col(f"low_{q}")).otherwise(pl.col(f"d3_{q}")).alias(q)
            for q in Q
        ],
    )
    for name, frame in out.items():
        if frame.select(Q).null_count().sum_horizontal().item() > 0:
            msg = f"{site} {name}: forecast rows without the arm's quantiles"
            raise ValueError(msg)
    return out


def main() -> None:
    """Print every number the write-up quotes and write the scores."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=datetime(2024, 4, 1, tzinfo=UTC), last=datetime(2026, 7, 1, tzinfo=UTC)
    )
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p)
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=SCORED_FIRST
    )
    per_arm: dict[str, list[pl.DataFrame]] = {"default_3": [], "default_1": [], "stack_without_s": []}
    for site in arms.SITES:
        one = series.filter(pl.col("label") == site)
        for name, stored in stored_arms(site=site).items():
            per_arm[name].append(adapter.forecast_frame(stored=stored, series=one, unit="points"))
    forecasts = {name: pl.concat(frames) for name, frames in per_arm.items()}
    for arm, stored_name in (("bg2", "p_s6_bg2"), ("climatology", "ref_climatology"), ("manual_heuristic", "ref_manual_heuristic")):
        stored = pl.read_parquet(
            adapter.stored_path(root=common.ML_EXPERIMENTS_DATA_DIR, pattern=f"00087*/predictions/{stored_name}/predictions.parquet")
        ).filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        forecasts[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    table = metrics.shared_slice_table(
        forecasts=forecasts, actuals=actuals, context=context, events=events, first_readings=first_readings
    )
    print(f"shared half-hours of the six arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    out = common.WORK_DIR / "scores"
    out.mkdir(exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(out / "scores.parquet")

    def crps_table(slice_name: str) -> pl.DataFrame:
        parts = [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == slice_name).select(
                "label", "group", "n_half_hours", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group", "n_half_hours"), on="label")
        return joined.sort("group", "label")

    for band in (HEADLINE_BAND, "lead_day_1", LATE_BAND, "all"):
        print(f"\n== CRPS, points, slice {band} ==")
        print(crps_table(band))
    groups = {
        "signed-MW": pl.col("group") == "signed-MW",
        "MVA-only with D8": pl.col("group") == "MVA-only",
        "MVA-only without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
        "short-history (D15, D18, report only)": pl.col("group") == "short-history",
    }
    print("\n== HEADLINE contrasts (99% interval, 20,000 resamples), treatment minus reference, CRPS ==")
    contrasts = [
        ("default_3", "bg2"),
        ("default_3", "default_1"),
        ("default_3", "stack_without_s"),
        ("default_3", "manual_heuristic"),
        ("default_3", "climatology"),
        ("default_1", "bg2"),
    ]
    for band in (HEADLINE_BAND, LATE_BAND):
        for treatment, ref in contrasts:
            for gname, where in groups.items():
                result = metrics.headline_month_bootstrap_contrast(
                    treatment=rows[treatment], reference=rows[ref], metric="crps",
                    where=where & pl.col(band), slice_table=table,
                )
                print(
                    f"{band:15s} {treatment} - {ref:18s} {gname:38s} {result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f}) n={result['n_half_hours']}, months={result['n_months']}"
                )
    print("\n== Per-Primary contrasts default_3 - bg2 and default_3 - default_1, 95%, 2,000 resamples, lead_days_1_10 ==")
    for ref in ("bg2", "default_1"):
        print(ref)
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows["default_3"], reference=rows[ref], metric="crps",
                where=pl.col(HEADLINE_BAND), slice_table=table,
            )
        )
    print("\n== Tail measures and twCRPS, slice lead_days_1_10, mean over Primaries per group ==")
    tails = ["pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99", "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "twcrps"]
    for name in ("default_3", "default_1", "bg2", "manual_heuristic", "climatology"):
        t = scored[name].filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(tails))
        print(name)
        print(t.group_by("group", "metric").agg(pl.col("value").mean()).pivot(on="metric", index="group", values="value").sort("group"))
    print("\n== Tail measures per Primary, default_3, slice lead_days_1_10 ==")
    print(scored["default_3"].filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(tails)).pivot(on="metric", index=["label", "group"], values="value").sort("group", "label"))
    print("\n== MAE of the median beside the CRPS contrasts (99%, 20,000 resamples), lead_days_1_10 ==")
    for treatment, ref in (("default_3", "bg2"), ("default_3", "default_1"), ("default_3", "manual_heuristic")):
        for gname, where in groups.items():
            result = metrics.headline_month_bootstrap_contrast(
                treatment=rows[treatment], reference=rows[ref], metric="mae_median",
                where=where & pl.col(HEADLINE_BAND), slice_table=table,
            )
            print(
                f"{treatment} - {ref:18s} {gname:38s} {result['difference']:+.3f} "
                f"({result['lower']:+.3f} to {result['upper']:+.3f})"
            )
    print("\n== Leave-one-out default_3 - bg2 and default_3 - default_1 (95%), lead_days_1_10 ==")
    for gname, labels in (
        ("signed-MW", ["D5", "D9", "D10", "D11", "D14", "D16"]),
        ("MVA-only", ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
    ):
        for ref in ("bg2", "default_1"):
            for left_out in labels:
                keep = pl.col("label").is_in([x for x in labels if x != left_out])
                result = metrics.month_bootstrap_contrast(
                    treatment=rows["default_3"], reference=rows[ref], metric="crps",
                    where=keep & pl.col(HEADLINE_BAND), slice_table=table,
                )
                print(
                    f"{gname:10s} vs {ref:10s} without {left_out:4s} {result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f})"
                )
    print("\n== Median minus mean of default_3's stored stage 2, per site, in points ==")
    print(
        pl.concat(
            [
                pl.read_parquet(common.stage_path(site=s, stage="dp_preds_s2"))
                .filter(pl.col("power_fcst_init_time") >= SCORED_FIRST)
                .select(site=pl.lit(s), diff=(pl.col("q50") - pl.col("mean")).mean())
                for s in arms.SITES
            ]
        )
    )


if __name__ == "__main__":
    main()
