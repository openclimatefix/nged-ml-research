"""Absolute scores by lead band, and the share of reverse-flow half-hours, at the four series.

Stored forecasts only, no fits: the stack of stage B, BG2, `manual_heuristic`, and `climatology`
on the shared half-hours of the scored year, CRPS in points of effective capacity by lead band.
The reverse-flow share is the fraction of half-hours of the scored year with a negative reading.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00386-*/e00386_lead_table.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00386_adapter as adapter
import e00386_common as common
import e00386_score_a as stage_a
import e00386_score_b as stage_b
import polars as pl
from studies.eval_protocol import metrics

BANDS = ((0.0, 24.0), (24.0, 72.0), (72.0, 168.0), (168.0, 240.0))


def main() -> None:
    """Print the table."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    adapter.register_groups()
    series = adapter.series_table()
    cache = common.load_cache()
    actuals, context = adapter.actuals_and_context(
        cache=cache,
        series=series,
        first=datetime(2024, 4, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}), series=series, before=common.SCORED_FIRST
    )
    arms = {"stack": stage_b.stack_frame(series=series), **stage_a.reference_frames(series=series)}
    table = metrics.shared_slice_table(forecasts=arms, actuals=actuals, context=context)
    pieces = []
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
        for low, high in BANDS:
            pieces.append(
                scored.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
                .group_by("label")
                .agg(pl.col("crps").mean())
                .with_columns(arm=pl.lit(name), band=pl.lit(f"{low:.0f}-{high:.0f} h"))
            )
    result = pl.concat(pieces)
    print(result.pivot(on="arm", index=["label", "band"], values="crps").sort("label", "band"))
    print("\n== Reverse-flow share and range of readings, scored year (points of capacity) ==")
    scored_year = actuals.filter(pl.col("valid_time") >= common.SCORED_FIRST).join(
        series.select("time_series_id", "label", "capacity_mw"), on="time_series_id"
    )
    print(
        scored_year.group_by("label")
        .agg(
            negative_share=(pl.col("actual") < 0).mean(),
            min_points=(pl.col("actual") / pl.col("capacity_mw") * 100).min(),
            capacity_mw=pl.col("capacity_mw").first(),
        )
        .sort("label")
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
