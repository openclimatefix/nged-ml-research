"""Stage D of experiment 386: monthly-refit XGBoost arms against BG2 and the stack.

Arms: `control` (experiment 87's `n1h` recipe refitted monthly) and `bundle` (experiment 381's
bundle of the daily solar summary, the sunny flag, and the 1-year half-life), each raw and widened
about the median by 1.15 (`_w`), plus the stack of stage B, the stored BG2 (`b_weekly`, weekly
refits), `manual_heuristic`, and `climatology`. The widening is applied here to a refit's raw
quantiles, so the arms differ from the stored forecasts of experiments 379 and 381 in nothing but
the stated factor.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00386-*/e00386d_score.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00386_adapter as adapter
import e00386_common as common
import e00386_score_a as stage_a
import e00386_score_b as stage_b
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.eval_protocol.metrics import QUANTILE_COLUMNS

ARM_DIR = common.WORK_DIR / "arm_preds" / "monthly"
ARMS = {"control": "control", "bundle": "bundle_solar+sunny+hl1"}
WIDENING = 1.15
HEADLINE_BAND = stage_a.HEADLINE_BAND


def widen(*, frame: pl.DataFrame, factor: float) -> pl.DataFrame:
    """Return the quantiles scaled about the median by `factor`, kept sorted."""
    median = pl.col("q50")
    scaled = [
        (median + (pl.col(q) - median) * factor).alias(q) for q in QUANTILE_COLUMNS if q != "q50"
    ]
    out = frame.with_columns(scaled)
    values = np.sort(out.select(QUANTILE_COLUMNS).to_numpy(), axis=1)
    return out.with_columns([pl.Series(q, values[:, i]) for i, q in enumerate(QUANTILE_COLUMNS)])


def xgboost_frame(*, arm: str, series: pl.DataFrame) -> pl.DataFrame:
    """Return a monthly-refit arm's forecasts in MW (stored as fractions of capacity)."""
    pieces = []
    for site in stage_b.SITES:
        files = sorted((ARM_DIR / ARMS[arm]).glob(f"{site}_s0_*.parquet"))
        stored = pl.concat([pl.read_parquet(f) for f in files])
        pieces.append(
            adapter.forecast_frame(
                stored=stored, series=series.filter(pl.col("label") == site), unit="fraction"
            )
        )
    return pl.concat(pieces)


def width(*, frame: pl.DataFrame) -> float:
    """Return the mean q95 minus q05 over the frame, in MW."""
    return float((frame["q95"] - frame["q05"]).to_numpy().mean())


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
    raw = {name: xgboost_frame(arm=name, series=series) for name in ARMS}
    arms = {
        **{f"xgb_{name}": frame for name, frame in raw.items()},
        **{f"xgb_{name}_w": widen(frame=frame, factor=WIDENING) for name, frame in raw.items()},
        "stack": stage_b.stack_frame(series=series),
        **stage_a.reference_frames(series=series),
    }
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"shared half-hours of the {len(arms)} arms: {table.height}")
    print("\n== Mean q95 - q05 width in MW, scored year, per arm ==")
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        print(f"{name:20s} {width(frame=kept):.3f}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    out = common.WORK_DIR / "scores_d"
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "scores.parquet"
    )
    for band in (HEADLINE_BAND, "lead_day_1", stage_a.LATE_BAND, stage_a.BEYOND_BAND):
        print(f"\n== CRPS, points of effective capacity, slice {band} ==")
        parts = [
            t.filter(pl.col("metric") == "crps", pl.col("slice") == band).select(
                "label", "group", pl.col("value").alias(n)
            )
            for n, t in scored.items()
        ]
        joined = parts[0]
        for p in parts[1:]:
            joined = joined.join(p.drop("group"), on="label")
        print(joined.sort("group", "label"))
    print("\n== Tail measures, days 1 to 10, per arm and series ==")
    for name, t in scored.items():
        print(name)
        print(
            t.filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(stage_a.TAILS))
            .pivot(on="metric", index=["label", "group"], values="value")
            .sort("group", "label")
        )
    contrasts = [
        ("xgb_control", "bg2"),
        ("xgb_control_w", "bg2"),
        ("xgb_bundle", "xgb_control"),
        ("xgb_bundle_w", "xgb_control_w"),
        ("xgb_bundle", "stack"),
        ("xgb_bundle_w", "stack"),
        ("xgb_control_w", "stack"),
        ("xgb_control_w", "manual_heuristic"),
        ("xgb_bundle_w", "manual_heuristic"),
    ]
    print("\n== HEADLINE contrasts treatment - reference (99%, 20,000 resamples), by group ==")
    for band in (HEADLINE_BAND, stage_a.BEYOND_BAND):
        for treatment, reference in contrasts:
            for group in ("BSP", "GSP"):
                result = metrics.headline_month_bootstrap_contrast(
                    treatment=rows[treatment],
                    reference=rows[reference],
                    metric="crps",
                    where=(pl.col("group") == group) & pl.col(band),
                    slice_table=table,
                )
                print(
                    f"{band:15s} {treatment:14s} - {reference:16s} {group} "
                    f"{result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f})"
                )
    print("\n== Per-series contrasts (95%, 2,000), days 1 to 10 ==")
    for treatment, reference in contrasts[:5]:
        print(treatment, "-", reference)
        print(
            metrics.reported_contrast_by_primary(
                treatment=rows[treatment],
                reference=rows[reference],
                metric="crps",
                where=pl.col(HEADLINE_BAND),
                slice_table=table,
            ).select("label", "difference", "lower", "upper")
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
