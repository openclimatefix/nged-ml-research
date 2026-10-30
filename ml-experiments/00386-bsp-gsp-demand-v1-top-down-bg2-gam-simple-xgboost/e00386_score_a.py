"""Stage A of experiment 386: score BG2 and the references at the four BSP and GSP series.

Arms: `bg2` (experiment 87's `b_weekly`, the weekly-refitted XGBoost model alone, which is BG2 at
these series), `climatology`, and `manual_heuristic`, all on `fold_a_daily_gap_v1`. Prints every
number the write-up quotes.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00386-*/e00386_score_a.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00386_adapter as adapter
import e00386_common as common
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

REFERENCES: dict[str, str] = {
    "bg2": "b_weekly",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
HEADLINE_BAND = "lead_days_1_10"
LATE_BAND = "lead_days_8_10"
BEYOND_BAND = "lead_days_11_14"
TAILS = [
    "pinball_q01",
    "pinball_q05",
    "pinball_q95",
    "pinball_q99",
    "coverage_q01",
    "coverage_q05",
    "coverage_q95",
    "coverage_q99",
    "twcrps",
]


def reference_frames(*, series: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return BG2, climatology, and the manual heuristic of experiment 87 at the four series."""
    out = {}
    for arm, name in REFERENCES.items():
        stored = pl.read_parquet(
            adapter.stored_path(
                root=ML_EXPERIMENTS_DATA_DIR,
                pattern=f"00087*/predictions/{name}/predictions.parquet",
            )
        ).filter(pl.col("time_series_id").is_in(series["time_series_id"].to_list()))
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def main() -> None:
    """Print the scores."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    adapter.register_groups()
    series = adapter.series_table()
    cache = common.load_cache()
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
        history=actuals.rename({"actual": "power_mw"}), series=series, before=common.SCORED_FIRST
    )
    arms = reference_frames(series=series)
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"shared half-hours of the arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    out = common.WORK_DIR / "scores_a"
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "scores.parquet"
    )
    for band in (HEADLINE_BAND, "lead_day_1", LATE_BAND, BEYOND_BAND, "all"):
        print(f"\n== CRPS, points of effective capacity, slice {band} ==")
        parts = [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == band).select(
                "label", "group", "n_half_hours", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group", "n_half_hours"), on="label")
        print(joined.sort("group", "label"))
    print("\n== MAE of the median, days 1 to 10 ==")
    print(
        pl.concat(
            [
                t.filter(pl.col("metric") == "mae_median", pl.col("slice") == HEADLINE_BAND).select(
                    "label", pl.lit(n).alias("arm"), "value"
                )
                for n, t in scored.items()
            ]
        )
        .pivot(on="arm", index="label", values="value")
        .sort("label")
    )
    print("\n== Tail measures, days 1 to 10, per series ==")
    for name, t in scored.items():
        print(name)
        print(
            t.filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(TAILS))
            .pivot(on="metric", index=["label", "group"], values="value")
            .sort("group", "label")
        )
    print("\n== HEADLINE contrasts bg2 - reference (99%, 20,000 resamples), by group ==")
    for band in (HEADLINE_BAND, LATE_BAND, BEYOND_BAND):
        for ref in ("manual_heuristic", "climatology"):
            for group in ("BSP", "GSP"):
                result = metrics.headline_month_bootstrap_contrast(
                    treatment=rows["bg2"],
                    reference=rows[ref],
                    metric="crps",
                    where=(pl.col("group") == group) & pl.col(band),
                    slice_table=table,
                )
                print(
                    f"{band:15s} bg2 - {ref:16s} {group} {result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f}) "
                    f"n={result['n_half_hours']}, months={result['n_months']}"
                )
    print("\n== Per-series contrasts bg2 - manual_heuristic (95%, 2,000), days 1 to 10 ==")
    print(
        metrics.reported_contrast_by_primary(
            treatment=rows["bg2"],
            reference=rows["manual_heuristic"],
            metric="crps",
            where=pl.col(HEADLINE_BAND),
            slice_table=table,
        )
    )
    print("\n== Median minus mean is not stored for BG2; q05 and q95 coverage are above ==")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
