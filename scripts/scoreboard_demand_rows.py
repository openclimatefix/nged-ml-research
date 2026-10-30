"""Score every stored Primary demand forecast on one shared set of half-hours, and store the rows.

This is the first half of the scoreboard (`scoreboard_demand_page.py` is the second). It reads
stored forecasts and fits nothing. Two shared row sets are scored, because the stored forecasts do
not all reach the same lead:

- `to_240h`: Primary demand research defaults 1, 2, and 3 (experiment 378's stored arms; default 2
  is default 1 with the two-number solar summary S in place of the stage-2 forecast above 168
  hours), the stack without S (experiment 377's stage 2), BG2, `climatology`,
  `manual_heuristic`, and the production-style `xgboost_baseline`. D8 uses the donor spread in
  every arm, as in experiment 378.
- `to_336h`: the stack without S (experiment 377, D8 with its own-residual spread), BG2,
  `climatology`, and `manual_heuristic`, with leads to 336 hours so that days 11 to 14 are scored.

The output is one parquet of `studies.eval_protocol.metrics.scored_rows` per row set, in the
scoreboard's private work folder (never in git), plus the shared slice table.

    PYTHONPATH=<folders of experiments 184, 202, 317, 319, 320, 377, 378> \
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python scripts/scoreboard_demand_rows.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00378_arms as arms  # ty: ignore[unresolved-import]
import e00378_common as common  # ty: ignore[unresolved-import]
import e00378_score as score378  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "scoreboard-demand" / "work"
WORK_377: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00377-demand-v1-first-run-feasibility" / "work"
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
SOLAR_SUMMARY_FROM_HOURS: Final[float] = common.STORED_HORIZON_HOURS
KEYS: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
Q: Final[list[str]] = list(common.QUANTILES)
MAX_LEAD_336: Final[float] = 336.0
CONTROL_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18")
STORED_87_ARMS: Final[dict[str, str]] = {
    "bg2": "p_s6_bg2",
    "climatology": "ref_climatology",
    "manual_heuristic": "ref_manual_heuristic",
}
PRODUCTION_BASELINE: Final[dict[str, str]] = {"xgboost_baseline": "ref_xgboost_baseline"}


def default_2_frame(*, site: str, default_1: pl.DataFrame) -> pl.DataFrame:
    """Return default 2: default 1 below 168 hours and the solar summary S above it.

    Default 3 differs from default 2 only by the heuristic blend on daylight half-hours, so
    default 2 is experiment 378's `x1` column above 168 hours.
    """
    source = arms.arms_source(site)
    long = pl.read_parquet(common.ARMS_DIR / f"long_lead_{source}.parquet").filter(
        pl.col("site") == site
    )
    x1 = long.select(*KEYS, *[pl.col(f"x1_{q}").alias(f"x1_{q}") for q in Q])
    joined = default_1.join(x1, on=KEYS, how="left")
    long_lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(
        fractional=True
    ) > SOLAR_SUMMARY_FROM_HOURS
    return joined.select(
        *[c for c in default_1.columns if c not in Q],
        *[pl.when(long_lead).then(pl.col(f"x1_{q}")).otherwise(pl.col(q)).alias(q) for q in Q],
    )


def references(
    *, series: pl.DataFrame, max_lead: float, with_production_baseline: bool
) -> dict[str, pl.DataFrame]:
    """Return BG2, `climatology`, and `manual_heuristic` as stored by experiment 87."""
    out = {}
    stored_arms = STORED_87_ARMS | (PRODUCTION_BASELINE if with_production_baseline else {})
    for arm, stored_name in stored_arms.items():
        stored = pl.read_parquet(
            adapter.stored_path(
                root=common.ML_EXPERIMENTS_DATA_DIR,
                pattern=f"00087*/predictions/{stored_name}/predictions.parquet",
            )
        ).filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
            pl.col("lead_hours") <= max_lead,
        )
        out[arm] = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
    return out


def stack_336(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 377's stack (D8's own-residual arm) in MW, leads to 336 hours."""
    pieces = []
    for site in adapter.PRIMARY_LABELS:
        source = "d15" if site == "D15" else "control" if site in CONTROL_SITES else "mva"
        path = WORK_377 / source / "arms" / "a0" / "dp_preds_s2" / f"stack_d_{site}.parquet"
        pieces.append(
            adapter.forecast_frame(
                stored=pl.read_parquet(path),
                series=series.filter(pl.col("label") == site),
                unit="points",
            ).filter(
                pl.col("valid_time") - pl.col("power_fcst_init_time")
                <= pl.duration(hours=MAX_LEAD_336)
            )
        )
    return pl.concat(pieces)


def score_row_set(
    *,
    name: str,
    forecasts: dict[str, pl.DataFrame],
    series: pl.DataFrame,
    actuals: pl.DataFrame,
    context: pl.DataFrame,
    events: pl.DataFrame,
    first_readings: dict[int, datetime],
    thresholds: pl.DataFrame,
) -> None:
    """Score every forecast on the half-hours that all of them share, and write the rows."""
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"{name}: arms {list(forecasts)}; shared half-hours {table.height:,}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "intervals.parquet").unlink(missing_ok=True)
    table.write_parquet(WORK / f"slice_table_{name}.parquet")
    pieces = []
    for arm, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
        pieces.append(rows.with_columns(arm=pl.lit(arm)))
        print(f"  scored {arm}: {rows.height:,} rows", flush=True)
    pl.concat(pieces).write_parquet(WORK / f"rows_{name}.parquet")


def main() -> None:
    """Score both row sets."""
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=score378.CACHE_DIR)
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
        history=actuals.rename({"actual": "power_mw"}), series=series, before=SCORED_FIRST
    )
    common_args = {
        "series": series,
        "actuals": actuals,
        "context": context,
        "events": events,
        "first_readings": first_readings,
        "thresholds": thresholds,
    }

    per_arm: dict[str, list[pl.DataFrame]] = {
        "default_1": [],
        "default_2": [],
        "default_3": [],
        "stack_without_s": [],
    }
    for site in arms.SITES:
        one = series.filter(pl.col("label") == site)
        stored = score378.stored_arms(site=site)
        stored["default_2"] = default_2_frame(site=site, default_1=stored["default_1"])
        for name, frame in stored.items():
            per_arm[name].append(adapter.forecast_frame(stored=frame, series=one, unit="points"))
    forecasts = {name: pl.concat(frames) for name, frames in per_arm.items()}
    forecasts.update(
        references(series=series, max_lead=common.HORIZON_HOURS, with_production_baseline=True)
    )
    score_row_set(name="to_240h", forecasts=forecasts, **common_args)

    forecasts_336 = {
        "stack_without_s": stack_336(series=series),
        **references(series=series, max_lead=MAX_LEAD_336, with_production_baseline=False),
    }
    score_row_set(name="to_336h", forecasts=forecasts_336, **common_args)


if __name__ == "__main__":
    main()
