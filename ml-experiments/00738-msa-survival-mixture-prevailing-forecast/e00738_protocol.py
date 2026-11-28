"""The four load-shift scenarios and the evaluation-protocol tables for the real record.

This module adapts `main` of `e00713_protocol.py` (not edited). It scores the `real` panel of every
arm (the stored `M1`, `W0`, `LP`, `LD`, `LM`, and `LR`) on the half-hours that all arms share, with
`studies.eval_protocol.metrics`, and writes the tables of experiment 713's protocol: the mean
absolute error of the median, CRPS, threshold-weighted CRPS, and the coverage and pinball loss at
q01, q05, q95, and q99, by the protocol's slices, with a 95% month-block interval. The four
scenarios are all rows, outside events (more than 7 days after one), inside events, and the 7
days after an event.

The real record carries no injected shift, so these tables show whether an arm harms the forecast
when no shift has been added. The climatology and the manual heuristic of experiment 87 are added as
report-only rows. The stored `M1` forecast comes from this experiment's own `real` panel (the
reproduction of the stored file is checked separately), so every arm is read the same way.

Writes `protocol/{rows.parquet,scores.csv,scenarios.csv}` under the experiment's output folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import e00738_common as common
import e00738_predict as pred
import polars as pl

SCENARIO_FILE: Final[str] = "scenarios.csv"
SCENARIO_SLICES: Final[tuple[str, ...]] = (
    "all",
    "outside_events",
    "event_rows",
    "event_following_7_days",
)


def arm_frame(
    *, arm: str, months: tuple[str, ...], seeds: tuple[int, ...], series: pl.DataFrame
) -> pl.DataFrame:
    """Return an arm's seed mean on the real record, in megawatts, from the contract origin.

    Args:
        arm: The arm.
        months: The fold months.
        seeds: The seeds.
        series: The series table of experiment 377's adapter.

    Returns:
        The forecast frame in the layout that the protocol's metrics read.
    """
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
    import e00713_score as score713  # ty: ignore[unresolved-import]

    parts = []
    for month in months:
        for seed in seeds:
            path = pred.panel_path(arm=arm, month=month, seed=seed, name=pred.PANELS_REAL)
            frame = pl.read_parquet(path, columns=[*score713.KEYS, *score713.QCOLS])
            parts.append(
                score713.to_contract_shares(
                    frame=frame.with_columns(seed=pl.lit(seed, dtype=pl.Int64))
                )
            )
    mean = score713.seed_average(frame=pl.concat(parts))
    return adapter.forecast_frame(stored=mean, series=series, unit="fraction").filter(
        pl.col("valid_time") >= cp.FIRST_VALID
    )


def scenario_table(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Return the four-scenario table: the protocol's scores of the four scenario slices.

    Args:
        scores: The `scores.csv` frame (group, slice, metric, arm, mean, lo, hi, half_hours).

    Returns:
        The rows of the four scenario slices, with the scenario's name in `scenario`.
    """
    names = {
        "all": "all rows",
        "outside_events": "outside events",
        "event_rows": "inside events",
        "event_following_7_days": "7 days after an event",
    }
    return (
        scores.filter(pl.col("slice").is_in(list(SCENARIO_SLICES)))
        .with_columns(scenario=pl.col("slice").replace(names))
        .sort("group", "scenario", "metric", "arm")
    )


def run(
    *,
    months: tuple[str, ...] = common.FOLDS,
    seeds: tuple[int, ...] = common.SEEDS,
    arms: tuple[str, ...] = common.ALL_ARMS,
    out: Path | None = None,
) -> Path:
    """Score the real record of every arm by the protocol and write the tables.

    Args:
        months: The fold months.
        seeds: The seeds.
        arms: The arms.
        out: The output folder (`OUT/protocol` by default).

    Returns:
        The folder written.
    """
    from datetime import UTC, datetime

    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00668_compare_primaries as cp  # ty: ignore[unresolved-import]
    import e00713_protocol as proto  # ty: ignore[unresolved-import]
    import e00713_score as score713  # ty: ignore[unresolved-import]
    from studies.dp_features import default_input_paths, load_or_build_feature_cache
    from studies.eval_protocol import metrics
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    folder = out if out is not None else common.OUT / "protocol"
    folder.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=cp.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=metrics.HOLDOUT_START,
    )
    events = adapter.events_from_masks(cache=cache, series=series)
    first_readings = {
        int(sid): next(
            t for t, p in zip(cache.times_series(), cache.sites[lab].present, strict=True) if p
        )
        for sid, lab, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}),
        series=series,
        before=datetime(2025, 7, 1, tzinfo=UTC),
    )
    forecasts = {arm: arm_frame(arm=arm, months=months, seeds=seeds, series=series) for arm in arms}
    ids = series["time_series_id"].to_list()
    for name, stored_name in proto.REFERENCES.items():
        stored = pl.read_parquet(
            adapter.stored_path(
                root=ML_EXPERIMENTS_DATA_DIR,
                pattern=f"00087*/predictions/{stored_name}/predictions.parquet",
            )
        ).filter(pl.col("time_series_id").is_in(ids))
        frame = cp.relabel(
            frame=adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        )
        forecasts[name] = frame.filter(pl.col("valid_time") >= cp.FIRST_VALID)
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    ).with_columns(outside_events=~pl.col("event_rows") & ~pl.col("event_following_7_days"))
    kept_rows = []
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        ).with_columns(
            arm=pl.lit(name),
            outside_events=~pl.col("event_rows") & ~pl.col("event_following_7_days"),
        )
        kept_rows.append(
            rows.select(*score713.KEYS, "label", "arm", *proto.SCORED_METRICS, *proto.SLICES)
        )
    all_rows = pl.concat(kept_rows).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    )
    all_rows.write_parquet(folder / "rows.parquet")
    proto.write_scores(all_rows=all_rows, out=folder)
    scenario_table(scores=pl.read_csv(folder / "scores.csv")).write_csv(folder / SCENARIO_FILE)
    return folder
