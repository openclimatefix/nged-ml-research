"""Rebuild the ensemble summaries of experiment 397 with the wind components.

The summaries hold the mean and the (population) standard deviation over the 51 members of each
weather column, for the 16 Primaries, one file per run. This copy of experiment 397's build adds
`wind_u_10m`, `wind_v_10m`, `wind_u_100m`, and `wind_v_100m` to the weather columns. The stored
51-member demand frames carry no wind components, so every run is read from the ENS archive.

Run with `PYTHONPATH` set to the experiment 87, 48, and 397 folders and this folder:

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00561-wind-components-era5-and-ens-summary-data-prep \
OMP_NUM_THREADS=1 uv run --no-sync python e00561_ens_summary_uv.py build --workers 3
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Final

import e00048_build_frames as build  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import polars as pl

WIND_COMPONENTS: Final[tuple[str, ...]] = (
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)
WEATHER: Final[tuple[str, ...]] = (*prep.WEATHER, *WIND_COMPONENTS)
SUMMARY_DIR: Final = prep.WORK_DIR / "work" / "ens_summary_uv"


def assert_origin_rule(*, frame: pl.DataFrame, run: datetime) -> pl.DataFrame:
    """Check row by row that the weather comes from the run issued at or before the origin.

    The origin (`power_fcst_init_time`) is the run's initialisation time plus 9 h, and every row's
    `valid_time` is after the origin. Returns one example row.
    """
    origin = common.origin_of(run=run)
    bad_run = frame.filter(pl.col("nwp_init_time") != run).height
    bad_origin = frame.filter(pl.col("power_fcst_init_time") != origin).height
    bad_gap = frame.filter(
        pl.col("power_fcst_init_time") - pl.col("nwp_init_time") != pl.duration(hours=9)
    ).height
    not_after = frame.filter(pl.col("valid_time") <= pl.col("power_fcst_init_time")).height
    if bad_run or bad_origin or bad_gap or not_after:
        msg = (
            f"{run:%Y-%m-%d}: origin rule broken: {bad_run} rows with another run, "
            f"{bad_origin} with another origin, {bad_gap} with a gap other than 9 h, "
            f"{not_after} with valid time at or before the origin"
        )
        raise ValueError(msg)
    return frame.select(
        "time_series_id", "nwp_init_time", "power_fcst_init_time", "valid_time", "ensemble_member"
    ).head(1)


def summarise(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the mean and (population) standard deviation over members of each weather column."""
    return frame.group_by("time_series_id", "power_fcst_init_time", "valid_time").agg(
        *[pl.col(w).mean().alias(f"mean_{w}") for w in WEATHER],
        *[pl.col(w).std(ddof=0).alias(f"std_{w}") for w in WEATHER],
    )


def write_summary(*, run: datetime) -> str:
    """Write one run's summary file if absent; return a progress line."""
    target = SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    targets = prep.primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    frame, _ = build._half_hourly_rows(run=run, targets=targets)
    example = assert_origin_rule(frame=frame, run=run)
    table = (
        summarise(frame=frame)
        .sort("time_series_id", "valid_time")
        .cast(dict.fromkeys([f"{k}_{w}" for k in ("mean", "std") for w in WEATHER], pl.Float32))
    )
    common.atomic_write(frame=table, target=target)
    if run == prep.FIRST_RUN:
        print(f"example row (origin-time rule holds for all {frame.height:,} rows):\n{example}")
    return f"{run:%Y-%m-%d}: {table.height:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build the summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N runs")
    args = parser.parse_args()
    runs = prep.run_times()[: args.limit]
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        for line in pool.map(_one, runs):
            print(line, flush=True)


def _one(run: datetime) -> str:
    return write_summary(run=run)


if __name__ == "__main__":
    main()
