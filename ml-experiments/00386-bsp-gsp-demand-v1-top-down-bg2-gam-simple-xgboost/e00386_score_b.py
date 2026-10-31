"""Stage B of experiment 386: the GAM stack against BG2 at the four BSP and GSP series.

Arms: `stack` (experiment 377's stack without S below 168 h, fitted top-down on each aggregate
series, from `e00386bs3_stack.py`), and the stage A references. Prints every number the write-up
quotes.

    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00386-*/e00386_score_b.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00386_adapter as adapter
import e00386_common as common
import e00386_score_a as stage_a
import polars as pl
from studies.eval_protocol import metrics

SITES = ("D13", "D17", "D1", "D6")
HEADLINE_BAND = stage_a.HEADLINE_BAND
STACK_DIR = common.WORK_DIR / "bg" / "arms" / "a0" / "dp_preds_s2"
SLICES = (
    "sunny_unit_shape",
    "winter_nights",
    "cold_nights",
    "economy_7_step",
    "christmas",
    "bank_holidays",
    "event_rows",
    "event_following_7_days",
)


def stack_frame(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the stack's forecast frame in MW (stored in points of effective capacity)."""
    pieces = []
    for site in SITES:
        stored = pl.read_parquet(STACK_DIR / f"stack_d_{site}.parquet")
        pieces.append(
            adapter.forecast_frame(
                stored=stored, series=series.filter(pl.col("label") == site), unit="points"
            )
        )
    return pl.concat(pieces)


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
    arms = {"stack": stack_frame(series=series), **stage_a.reference_frames(series=series)}
    table = metrics.shared_slice_table(
        forecasts=arms,
        actuals=actuals,
        context=context,
        events=events,
        first_readings=first_readings,
    )
    print(f"shared half-hours of the four arms: {table.height}")
    scored, rows = {}, {}
    for name, frame in arms.items():
        kept = metrics.restrict_to_slice_table(forecast=frame, slice_table=table)
        scored[name] = metrics.score_forecast(
            forecast=kept, actuals=actuals, series=series, slice_table=table, thresholds=thresholds
        )
        rows[name] = metrics.scored_rows(
            forecast=kept, actuals=actuals, series=series, thresholds=thresholds, slice_table=table
        )
    out = common.WORK_DIR / "scores_b"
    out.mkdir(parents=True, exist_ok=True)
    pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()]).write_parquet(
        out / "scores.parquet"
    )
    for band in (HEADLINE_BAND, "lead_day_1", stage_a.LATE_BAND, stage_a.BEYOND_BAND, "all"):
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
    print("\n== Tail measures, days 1 to 10, stack and bg2, per series ==")
    for name in ("stack", "bg2"):
        print(name)
        print(
            scored[name]
            .filter(pl.col("slice") == HEADLINE_BAND, pl.col("metric").is_in(stage_a.TAILS))
            .pivot(on="metric", index=["label", "group"], values="value")
            .sort("group", "label")
        )
    print("\n== HEADLINE contrasts stack - reference (99%, 20,000 resamples), by group ==")
    for band in (HEADLINE_BAND, stage_a.LATE_BAND, stage_a.BEYOND_BAND):
        for ref in ("bg2", "manual_heuristic", "climatology"):
            for group in ("BSP", "GSP"):
                result = metrics.headline_month_bootstrap_contrast(
                    treatment=rows["stack"],
                    reference=rows[ref],
                    metric="crps",
                    where=(pl.col("group") == group) & pl.col(band),
                    slice_table=table,
                )
                print(
                    f"{band:15s} stack - {ref:16s} {group} {result['difference']:+.3f} "
                    f"({result['lower']:+.3f} to {result['upper']:+.3f}) "
                    f"n={result['n_half_hours']}, months={result['n_months']}"
                )
    print("\n== Per-series contrasts stack - bg2 (95%, 2,000), days 1 to 10 ==")
    print(
        metrics.reported_contrast_by_primary(
            treatment=rows["stack"],
            reference=rows["bg2"],
            metric="crps",
            where=pl.col(HEADLINE_BAND),
            slice_table=table,
        )
    )
    print("\n== Leave-one-out stack - bg2 (95%), days 1 to 10 ==")
    for group, labels in (("BSP", ["D13", "D17"]), ("GSP", ["D1", "D6"])):
        for left_out in labels:
            keep = pl.col("label").is_in([x for x in labels if x != left_out])
            result = metrics.month_bootstrap_contrast(
                treatment=rows["stack"],
                reference=rows["bg2"],
                metric="crps",
                where=keep & pl.col(HEADLINE_BAND),
                slice_table=table,
            )
            print(
                f"{group} without {left_out:4s} {result['difference']:+.3f} "
                f"({result['lower']:+.3f} to {result['upper']:+.3f})"
            )
    print("\n== Series won by the stack against each reference (CRPS, days 1 to 10) ==")
    for ref in ("bg2", "manual_heuristic", "climatology"):
        print(
            ref,
            metrics.primaries_won(
                treatment=scored["stack"], reference=scored[ref], slice_name=HEADLINE_BAND
            ),
        )
    print("\n== Slice CRPS, stack vs bg2, by group, days 1 to 10 ==")
    for sl in SLICES:
        parts = []
        for name in ("stack", "bg2"):
            t = rows[name].filter(pl.col(sl), pl.col(HEADLINE_BAND))
            parts.append(t.group_by("group").agg(pl.col("crps").mean().alias(name), n=pl.len()))
        print(sl)
        print(parts[0].join(parts[1].drop("n"), on="group").sort("group"))
    print("\n== Median minus mean of the stack, per site, in points ==")
    print(
        pl.concat(
            [
                pl.read_parquet(STACK_DIR / f"stack_d_{site}.parquet")
                .filter(pl.col("power_fcst_init_time") >= common.SCORED_FIRST)
                .select(site=pl.lit(site), diff=(pl.col("q50") - pl.col("mean")).mean())
                for site in SITES
            ]
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
