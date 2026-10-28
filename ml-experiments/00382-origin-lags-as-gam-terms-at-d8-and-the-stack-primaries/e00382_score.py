"""Score experiment 382: the stack with three origin lags as stage 2 terms, against the stack.

Prints every number the write-up quotes. Arms: `default` (experiment 377's stack as stored),
`spline` (default plus origin lags, the pre-registered arm), `linear` (sensitivity), and BG2,
climatology, and the manual heuristic of experiment 87. D8's rows are its own-residual arm; its
donor-spread arm is printed separately. D15 and D18 are report only.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python e00382_score.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as base  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics

WORK = base.DATA / "00382-origin-lags-as-gam-terms-at-d8-and-the-stack-primaries" / "work"
TREATED = ("spline", "linear")
BANDS = {
    "days 1 to 10": pl.col("lead_days_1_10"),
    "days 1 to 2": pl.col("lead_day_1") | pl.col("lead_day_2"),
    "days 3 to 10": pl.col("lead_days_1_10") & ~(pl.col("lead_day_1") | pl.col("lead_day_2")),
    "days 8 to 10": pl.col("lead_days_8_10"),
}
GROUPS = {
    "signed-MW six": pl.col("group") == "signed-MW",
    "MVA-only eight with D8": pl.col("group") == "MVA-only",
    "MVA-only seven without D8": (pl.col("group") == "MVA-only") & (pl.col("label") != "D8"),
    "D8 alone": pl.col("label") == "D8",
    "D2 and D4 (placebos)": pl.col("label").is_in(["D2", "D4"]),
    "short-history (report only)": pl.col("group") == "short-history",
}
COUNTED = [s for s in base.STACK_SITES if s not in ("D15", "D18")]


def treated_path(*, arm: str, site: str, donor: bool = False) -> Path:
    """Return the stored treated forecast of a site."""
    stem = "stack_d_donor" if donor else "stack_d"
    return WORK / "arms" / arm / f"{stem}_{site}.parquet"


def treated_frames(*, series: pl.DataFrame, arm: str) -> dict[str, pl.DataFrame]:
    """Return an arm's forecast frame (and D8's donor frame) in MW."""
    pieces = [
        adapter.forecast_frame(
            stored=pl.read_parquet(treated_path(arm=arm, site=site)),
            series=series.filter(pl.col("label") == site),
            unit="points",
        )
        for site in base.STACK_SITES
    ]
    d8 = series.filter(pl.col("label") == "D8")
    donor = adapter.forecast_frame(
        stored=pl.read_parquet(treated_path(arm=arm, site="D8", donor=True)),
        series=d8,
        unit="points",
    )
    return {arm: pl.concat(pieces), f"{arm}_d8_donor": donor}


def contrast_line(*, name: str, result: metrics.ContrastInterval) -> str:
    """Format a contrast."""
    return (
        f"{name:42s} {result['difference']:+.3f} "
        f"({result['lower']:+.3f} to {result['upper']:+.3f}) "
        f"[{result['level_percent']:.0f}%] n={result['n_half_hours']}, months={result['n_months']}"
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print the scores."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=base.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
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
    default = base.stack_frames(series=series)
    arms = {"default": default["stack"], **base.reference_frames(series=series)}
    for arm in TREATED:
        arms[arm] = treated_frames(series=series, arm=arm)[arm]
    fitted = series.filter(pl.col("label").is_in(base.STACK_SITES))
    table = metrics.shared_slice_table(
        forecasts=arms, actuals=actuals, context=context, events=events,
        first_readings=first_readings,
    )  # fmt: skip
    print(f"shared half-hours of the six arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=fitted, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=fitted, thresholds=thresholds, slice_table=table
        )
    for metric in ("crps", "mae_median", "twcrps"):
        print(f"\n== {metric}, points, slice lead_days_1_10, per Primary and arm ==")
        parts = [
            t.filter(pl.col("metric") == metric, pl.col("slice") == base.HEADLINE_BAND).select(
                "label", "group", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group"), on="label")
        print(joined.sort("group", "label"))
    for arm in TREATED:
        print(f"\n== HEADLINE contrasts, {arm} minus default (99%, 20,000 resamples) ==")
        for band, band_where in BANDS.items():
            for gname, where in GROUPS.items():
                for metric in ("crps", "mae_median", "twcrps"):
                    if metric == "twcrps" and gname.startswith("short-history"):
                        continue  # D15 and D18 have no twCRPS threshold yet
                    result = metrics.headline_month_bootstrap_contrast(
                        treatment=rows[arm], reference=rows["default"], metric=metric,
                        where=where & band_where, slice_table=table,
                    )  # fmt: skip
                    print(contrast_line(name=f"{band} | {gname} | {metric}", result=result))
    print("\n== Per-Primary contrasts spline - default (95%, 2,000), CRPS ==")
    for band in ("days 1 to 10", "days 1 to 2", "days 3 to 10"):
        print(band)
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows["spline"], reference=rows["default"], metric="crps",
                where=BANDS[band], slice_table=table,
            )
        )  # fmt: skip
    print("\n== Leave-one-out spline - default (95%), days 1 to 10, CRPS ==")
    for gname, labels in (
        ("signed-MW", ["D5", "D9", "D10", "D11", "D14", "D16"]),
        ("MVA-only", ["D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]),
        ("all 14 counted", COUNTED),
    ):
        for left_out in labels:
            keep = pl.col("label").is_in([x for x in labels if x != left_out])
            result = metrics.month_bootstrap_contrast(
                treatment=rows["spline"], reference=rows["default"], metric="crps",
                where=keep & BANDS["days 1 to 10"], slice_table=table,
            )  # fmt: skip
            print(contrast_line(name=f"{gname} without {left_out}", result=result))
    print("\n== Primaries where spline beats default (days 1 to 10, counted 14) ==")
    countable = lambda t: t.filter(pl.col("label").is_in(COUNTED))  # noqa: E731
    print(
        metrics.primaries_won(
            treatment=countable(scored["spline"]),
            reference=countable(scored["default"]),
            slice_name=base.HEADLINE_BAND,
        )
    )
    print("\n== Month-by-month CRPS difference spline - default, days 1 to 10 ==")
    for label in ("D8", "D2", "D4"):
        a, b = (
            rows[x].filter(pl.col("label") == label, BANDS["days 1 to 10"])
            .with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
            .group_by("month").agg(pl.col("crps").mean()).sort("month")
            for x in ("spline", "default")
        )  # fmt: skip
        print(label, a.join(b, on="month", suffix="_default").with_columns(
            diff=pl.col("crps") - pl.col("crps_default")
        ).rows())  # fmt: skip
    print("\n== D8 leave-one-month-out spline - default (days 1 to 10 CRPS) ==")
    d8 = (
        rows["spline"].filter(pl.col("label") == "D8", BANDS["days 1 to 10"])
        .join(
            rows["default"].filter(pl.col("label") == "D8", BANDS["days 1 to 10"]).select(
                "power_fcst_init_time", "valid_time", crps_default="crps"
            ),
            on=["power_fcst_init_time", "valid_time"],
        )
        .with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            diff=pl.col("crps") - pl.col("crps_default"),
        )
    )  # fmt: skip
    for month in sorted(d8["month"].unique()):
        print(month, round(float(d8.filter(pl.col("month") != month)["diff"].mean()), 3))
    print("\n== Tail measures per group and arm, slice lead_days_1_10 (mean over Primaries) ==")
    tails = [
        "pinball_q01", "pinball_q05", "pinball_q95", "pinball_q99",
        "coverage_q01", "coverage_q05", "coverage_q95", "coverage_q99", "twcrps",
    ]  # fmt: skip
    for name in ("default", "spline", "linear", "bg2", "manual_heuristic", "climatology"):
        t = scored[name].filter(
            pl.col("slice") == base.HEADLINE_BAND, pl.col("metric").is_in(tails)
        )
        print(name)
        print(
            t.group_by("group", "metric").agg(pl.col("value").mean())
            .pivot(on="metric", index="group", values="value").sort("group")
        )  # fmt: skip
        print(
            t.filter(pl.col("label").is_in(["D8", "D2", "D4"]))
            .pivot(on="metric", index="label", values="value").sort("label")
        )  # fmt: skip
    print("\n== Slice CRPS, spline vs default, pooled by group, days 1 to 10 ==")
    for sl in (
        "sunny_unit_shape", "sunny_april_june", "winter_nights", "cold_nights", "economy_7_step",
        "christmas", "bank_holidays", "event_rows", "event_following_7_days", "cold_start_months",
    ):  # fmt: skip
        parts = []
        for name in ("spline", "default"):
            t = rows[name].filter(pl.col(sl), BANDS["days 1 to 10"])
            parts.append(t.group_by("group").agg(pl.col("crps").mean().alias(name), n=pl.len()))
        print(sl)
        print(parts[0].join(parts[1].drop("n"), on="group").sort("group"))
    print("\n== D8: own-residual arm vs donor-spread arm, on shared rows, CRPS ==")
    d8s = series.filter(pl.col("label") == "D8")
    d8_id = d8s["time_series_id"][0]
    donors = {arm: treated_frames(series=series, arm=arm)[f"{arm}_d8_donor"] for arm in TREATED}
    d8_arms = {
        "default": arms["default"].filter(pl.col("time_series_id") == d8_id),
        "default_donor": default["stack_d8_donor"],
        "spline": arms["spline"].filter(pl.col("time_series_id") == d8_id),
        "spline_donor": donors["spline"],
        "bg2": arms["bg2"].filter(pl.col("time_series_id") == d8_id),
    }
    table8 = metrics.shared_slice_table(forecasts=d8_arms, actuals=actuals, context=context)
    for name, frame in d8_arms.items():
        s = metrics.score_forecast(
            forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table8),
            actuals=actuals, series=d8s, slice_table=table8, thresholds=thresholds,
            slices=[base.HEADLINE_BAND, "lead_day_1", "lead_day_2", "all"],
        )  # fmt: skip
        print(
            name,
            s.filter(pl.col("metric") == "crps").select("slice", "n_half_hours", "value").rows(),
        )
    print("\n== Nulled rows: width (q95-q05) and coverage of [q05, q95], spline, days 1 to 10 ==")
    parts = []
    for site in base.STACK_SITES:
        f = pl.read_parquet(treated_path(arm="spline", site=site)).filter(
            pl.col("power_fcst_init_time") >= base.SCORED_FIRST,
            pl.col("lead_hours") <= 240,
            pl.col("actual").is_finite(),
        )
        inside = (pl.col("actual") >= pl.col("q05")) & (pl.col("actual") <= pl.col("q95"))
        parts.append(
            f.select(
                site=pl.lit(site),
                lags_missing="lags_missing",
                lead_hours="lead_hours",
                width=pl.col("q95") - pl.col("q05"),
                inside=inside.cast(pl.Float64),
            )
        )
    nulled = pl.concat(parts).with_columns(missing=pl.col("lags_missing") > 0)
    print(
        nulled.group_by("site", "missing").agg(
            n=pl.len(), width=pl.col("width").mean(), coverage_90=pl.col("inside").mean()
        ).sort("site", "missing")
    )  # fmt: skip
    print("\n== Median minus mean per site, points (spline), and mean-forecast error ==")
    stats = []
    for arm in ("default", "spline"):
        for site in base.STACK_SITES:
            path = (
                base.stage_2_path(site=site)
                if arm == "default"
                else treated_path(arm=arm, site=site)
            )
            f = pl.read_parquet(path).filter(
                pl.col("power_fcst_init_time") >= base.SCORED_FIRST,
                pl.col("lead_hours") <= 240,
                pl.col("actual").is_finite(),
            )
            stats.append(f.select(
                arm=pl.lit(arm), site=pl.lit(site),
                q50_minus_mean=(pl.col("q50") - pl.col("mean")).mean(),
                mean_abs_error=(pl.col("mean") - pl.col("actual")).abs().mean(),
                mean_bias=(pl.col("mean") - pl.col("actual")).mean(),
            ))  # fmt: skip
    print(pl.concat(stats).sort("site", "arm"))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
