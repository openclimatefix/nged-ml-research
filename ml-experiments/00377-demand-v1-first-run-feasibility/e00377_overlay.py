"""Overlay C: Primary demand research default 3's stored forecasts beside default 1 extended.

Experiment 322 stored default 3's quantiles (`d1_q01` to `d1_q99`, in points of effective capacity)
for 168 to 240 hours only, at the six signed-MW Primaries, on its own older folds (one fit, not
monthly folds). The overlay is therefore days 8 to 10 alone.
This script scores those rows and the stack of this experiment on the rows both hold, and BG2 as
a shared reference. Nothing is fitted. Hours 240 to 336 do not exist for default 3. The result is a
descriptive comparison across a change of training regime, so the two arms are not a fair test of
the daylight blend.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00377-*/e00377_overlay.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter
import e00377_score as score
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.eval_protocol.metrics import QUANTILE_COLUMNS

SIGNED_MW = ("D5", "D9", "D10", "D11", "D14", "D16")


def default_3_frame(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 322's default-3 quantiles as a stored frame in the adapter's layout."""
    stored = pl.read_parquet(
        adapter.stored_path(root=score.DATA, pattern="00322*/work/arms/scored_long_lead.parquet")
    )
    renamed = stored.rename({f"d1_{name}": name for name in QUANTILE_COLUMNS}).join(
        series.select("label", "time_series_id"), left_on="site", right_on="label"
    )
    return adapter.forecast_frame(stored=renamed, series=series, unit="points")


def main() -> None:
    """Print default 3 against default 1 extended and BG2 on shared rows."""
    pl.Config.set_tbl_rows(100)
    series = adapter.series_table(labels=SIGNED_MW)
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=score.CACHE_DIR)
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=score.SCORED_FIRST
    )
    arms = {
        "default_1_extended": pl.concat(
            adapter.forecast_frame(
                stored=pl.read_parquet(score.stage_2_path(site=site)),
                series=series.filter(pl.col("label") == site),
                unit="points",
            )
            for site in SIGNED_MW
        ),
        "default_3_old_folds": default_3_frame(series=series),
        "bg2": score.reference_frames(series=series)["bg2"],
    }
    table = metrics.shared_slice_table(forecasts=arms, actuals=actuals, context=context)
    print(f"shared half-hours: {table.height}")
    for name, frame in arms.items():
        scored = metrics.score_forecast(
            forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table),
            actuals=actuals,
            series=series,
            slice_table=table,
            thresholds=thresholds,
            slices=[score.HEADLINE_BAND, score.LATE_BAND, "lead_days_5_7"],
        )
        crps = scored.filter(pl.col("metric") == "crps")
        print(f"\n== {name}: CRPS, points, pooled over the six ==")
        print(crps.group_by("slice").agg(pl.col("value").mean(), pl.col("n_half_hours").sum()))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
