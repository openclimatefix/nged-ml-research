"""Build the shared rows of experiment 393 from experiment 377's and 87's stored forecasts.

Writes the shared slice table, the per-arm scored rows, the shared-row forecasts (MW), the actuals,
and the series table to this experiment's private `work/` folder. Fits nothing.

    DATA_PATH_INTERNAL=... PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python ml-experiments/00393-gam-mva-fold-data-analysis/e00393_build.py
"""

from datetime import UTC, datetime

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00377_score as s377  # ty: ignore[unresolved-import]
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00393-gam-mva-fold-data-analysis" / "work"
ARMS = ("stack", "bg2", "climatology", "manual_heuristic")


def main() -> None:
    """Score the four arms on their shared rows and store the inputs of the analysis."""
    WORK.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=s377.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2019, 1, 1, tzinfo=UTC),
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
        history=actuals.rename({"actual": "power_mw"}), series=series, before=s377.SCORED_FIRST
    )
    frames = {**s377.stack_frames(series=series), **s377.reference_frames(series=series)}
    forecasts = {name: frames[name] for name in ARMS}
    scored_context = context.filter(pl.col("valid_time") >= datetime(2024, 4, 1, tzinfo=UTC))
    table = metrics.shared_slice_table(
        forecasts=forecasts,
        actuals=actuals,
        context=scored_context,
        events=events,
        first_readings=first_readings,
    )
    print("shared half-hours:", table.height)
    fitted = series.filter(pl.col("label").is_in(s377.STACK_SITES))
    for name, frame in forecasts.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        kept.write_parquet(WORK / f"forecast_{name}.parquet")
        rows = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=fitted, thresholds=thresholds, slice_table=table
        )
        rows.write_parquet(WORK / f"rows_{name}.parquet")
        print(name, kept.height, rows.height, rows.columns)
    table.write_parquet(WORK / "slice_table.parquet")
    actuals.write_parquet(WORK / "actuals.parquet")
    series.write_parquet(WORK / "series.parquet")
    thresholds.write_parquet(WORK / "thresholds.parquet")


if __name__ == "__main__":
    main()
