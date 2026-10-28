"""Step 0 checks of the first demand-v1 run: feature inputs, the adapter, and a scoring pilot.

Prints three reports: the per-Primary table of reading and ENS coverage, the finiteness of the
slice inputs at all 16 Primaries over the scored period, and a timed scoring of one signed-MW
and one MVA-only Primary through `studies.eval_protocol`, from stored forecasts.
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import e00377_adapter as adapter
import polars as pl
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CACHE_DIR = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SCORED_FIRST = datetime(2025, 7, 1, tzinfo=UTC)
ENS_FIRST_RUN = datetime(2024, 4, 1, tzinfo=UTC)


def main() -> None:
    """Print the step 0 reports."""
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    series = adapter.series_table()
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    print("first grid half-hour", cache.times_series()[0], "last", cache.times_series()[-1])

    first = datetime(2024, 4, 1, tzinfo=UTC)
    last = datetime(2026, 7, 1, tzinfo=UTC)
    actuals, context = adapter.actuals_and_context(
        cache=cache, series=series, first=first, last=last
    )
    print("\n== slice inputs finite over 2025-07-01 to 2026-06-30 (share of half-hours) ==")
    scored = context.filter(pl.col("valid_time") >= SCORED_FIRST).join(
        actuals, on=["time_series_id", "valid_time"]
    )
    print(
        scored.group_by("time_series_id")
        .agg(
            temperature_c=pl.col("temperature_c").is_finite().mean(),
            elevation_deg=pl.col("elevation_deg").is_finite().mean(),
            pv_unit_shape=pl.col("pv_unit_shape").is_finite().mean(),
            actual=pl.col("actual").is_finite().mean(),
        )
        .join(series, on="time_series_id")
        .sort("label")
        .select("label", "temperature_c", "elevation_deg", "pv_unit_shape", "actual")
    )

    events = adapter.events_from_masks(cache=cache, series=series)
    print("\n== hand-filtered event runs per Primary (padded mask runs) ==")
    print(
        events.join(series, on="time_series_id")
        .group_by("label")
        .agg(n_events=pl.len(), first_start=pl.col("start").min(), last_end=pl.col("end").max())
        .sort("label")
    )

    first_readings = {
        int(series_id): next(
            t for t, p in zip(cache.times_series(), cache.sites[label].present, strict=True) if p
        )
        for series_id, label, _ in series.iter_rows()
    }
    thresholds = metrics.twcrps_thresholds(
        history=actuals.rename({"actual": "power_mw"}),
        series=series,
        before=SCORED_FIRST,
    )
    print("\n== twCRPS threshold status ==")
    print(
        thresholds.join(series, on="time_series_id")
        .sort("label")
        .select("label", "threshold_points", "status")
    )

    # Pilot: one signed-MW Primary (D10, experiment 376's stack) and one MVA-only Primary (D12,
    # experiment 374's arms), each beside the stored BG2.
    data = ML_EXPERIMENTS_DATA_DIR
    bg2 = pl.read_parquet(
        adapter.stored_path(root=data, pattern="00087*/predictions/p_s6_bg2/predictions.parquet")
    )
    pilots = {
        "D10": (
            pl.read_parquet(
                adapter.stored_path(
                    root=data,
                    pattern="00377*/work/control/arms/a0/dp_preds_s2/a_scored_D10.parquet",
                )
            ),
            "a",
        ),
        "D12": (
            pl.read_parquet(
                adapter.stored_path(
                    root=data, pattern="00377*/work/mva/arms/a0/dp_preds_s2/a_scored_D12.parquet"
                )
            ),
            "a",
        ),
    }
    for label, (stored, arm) in pilots.items():
        start = time.perf_counter()
        one = series.filter(pl.col("label") == label)
        sid = one["time_series_id"][0]
        mine = adapter.forecast_frame(stored=stored, arm=arm, series=one, unit="points")
        ref = adapter.forecast_frame(
            stored=bg2.filter(pl.col("time_series_id") == sid), series=one, unit="fraction"
        )
        # Row check: the stored `actual` equals the cache's reading.
        check = (
            stored.filter(pl.col("seed") == 0)
            .select("time_series_id", "valid_time", "actual")
            .unique(subset=["time_series_id", "valid_time"])
            .join(actuals, on=["time_series_id", "valid_time"], suffix="_cache")
        )
        diff = (check["actual"] * one["capacity_mw"][0] / 100.0 - check["actual_cache"]).abs().max()
        print(
            f"\n== pilot {label}: rows model {mine.height}, BG2 {ref.height}; "
            f"max |actual - cache| = {diff:.3g} MW =="
        )
        table = metrics.shared_slice_table(
            forecasts={"model": mine, "bg2": ref},
            actuals=actuals,
            context=context,
            events=events,
            first_readings=first_readings,
        )
        thr = thresholds.filter(pl.col("time_series_id") == sid)
        outputs = []
        for name, frame in {"model": mine, "bg2": ref}.items():
            outputs.append(
                metrics.score_forecast(
                    forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table),
                    actuals=actuals,
                    series=one,
                    slice_table=table,
                    thresholds=thr,
                ).with_columns(arm=pl.lit(name))
            )
        result = pl.concat(outputs)
        headline = result.filter(
            pl.col("metric") == "crps",
            pl.col("slice").is_in(["all", "lead_days_1_10", "lead_days_5_7", "lead_days_8_10"]),
        ).select("arm", "slice", "n_half_hours", "value", "value_mw", "status")
        print(headline)
        print(f"scored in {time.perf_counter() - start:.1f} s; shared half-hours {table.height}")
        rows = {
            name: metrics.scored_rows(
                forecast=metrics.restrict_to_slice_table(forecast=frame, slice_table=table),
                actuals=actuals,
                series=one,
                thresholds=thr,
                slice_table=table,
            )
            for name, frame in {"model": mine, "bg2": ref}.items()
        }
        began = time.perf_counter()
        contrast = metrics.headline_month_bootstrap_contrast(
            treatment=rows["model"],
            reference=rows["bg2"],
            metric="crps",
            where=pl.col("lead_days_1_10"),
            slice_table=table,
        )
        print(
            f"headline contrast (99%, 20,000 resamples): {contrast} "
            f"in {time.perf_counter() - began:.1f} s"
        )
        lead_table = (
            metrics.score_forecast(
                forecast=ref,
                actuals=actuals,
                series=one,
                slice_table=metrics.shared_slice_table(
                    forecasts={"bg2": ref}, actuals=actuals, context=context
                ),
                thresholds=thr,
                slices=[
                    "lead_day_1",
                    "lead_day_2",
                    "lead_days_3_4",
                    "lead_days_5_7",
                    "lead_days_8_10",
                    "lead_days_11_14",
                ],
            )
            .filter(pl.col("metric") == "crps")
            .select("slice", "n_half_hours", "value")
        )
        print("BG2 alone, own shared rows (all leads it holds):")
        print(lead_table)
        print(
            "max lead in model frame (hours):",
            (mine["valid_time"] - mine["power_fcst_init_time"]).dt.total_hours().max(),
        )
        for slice_name in ("event_rows", "event_following_7_days", "cold_start_months"):
            n = int(table[slice_name].sum()) if slice_name in table.columns else None
            print(f"slice {slice_name}: {n} half-hours")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
