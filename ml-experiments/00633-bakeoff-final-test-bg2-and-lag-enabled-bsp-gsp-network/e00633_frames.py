# ruff: noqa: E501, D103
"""Experiment 633: build the 51-member weather frames of the 123 window runs for the 20 demand series.

Each run is read from the ECMWF ensemble archive, cut to the series' H3 cells, resampled to the
half-hourly grid by `_upsample_nwp_to_half_hourly`, and cut to valid times from the origin (the run
plus 9 hours) to 351 hours later, by experiment 48's `_half_hourly_rows`. Only weather and calendar
columns are written; no power reading is touched, so the frames can be built before the go.

Usage: python e00633_frames.py [--workers N] [--limit N]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import e00048_build_frames as bf  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00633_common as k
import polars as pl

COLUMNS = ["time_series_id", "h3_res_5", "latitude", "longitude"]
_SERIES: dict[str, pl.DataFrame] = {}


def build(*, run: datetime) -> pl.DataFrame:
    """Return one run's frame: weather, hour and weekday, and lead for the 20 series and 51 members."""
    series = _SERIES.setdefault("s", common.series_table())
    rows, _init = bf._half_hourly_rows(run=run, targets=series.select(COLUMNS))
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    return (
        rows.with_columns(
            hour_local=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0,
            day_of_week_local=local.dt.weekday().cast(pl.Float64),
        )
        .select(
            *bf.RUN_KEY_COLUMNS, *bf.DEMAND_WEATHER, "hour_local", "day_of_week_local", "lead_hours"
        )
        .sort("time_series_id", "valid_time", "ensemble_member")
    )


def build_and_write(*, run: datetime) -> str:
    target = k.FRAMES / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    frame = build(run=run)
    assert frame["ensemble_member"].n_unique() == 51, (run, frame["ensemble_member"].n_unique())
    assert frame["time_series_id"].n_unique() == 20, run
    assert frame["power_fcst_init_time"].unique().to_list() == [run + common.PUBLICATION_DELAY], run
    common.atomic_write(frame=frame, target=target)
    return f"{run:%Y-%m-%d}: {frame.height:,} rows, {time.monotonic() - started:.0f} s"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    os.environ["POLARS_MAX_THREADS"] = "2"
    todo = k.runs()[: a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, f in enumerate([pool.submit(build_and_write, run=r) for r in todo], 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)
