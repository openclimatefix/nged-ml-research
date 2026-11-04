"""Fetch the ENS total cloud cover from Dynamical.org and summarise it per run.

The production Delta table of ENS runs holds no cloud cover, so this script reads
`total_cloud_cover_atmosphere` (`tcc`, a percentage in 0 to 100, all 51 members, every lead) from
the Dynamical.org store `ecmwf-ifs-ens-forecast-15-day-0-25-degree` (CC-BY-4.0; attribution "ECMWF
IFS ENS forecast data processed by dynamical.org from ECMWF Open Data"). The variable is all-NaN
for the runs before 2025-11-21, so the build starts there.

Per run the script (1) reads the grid points of the 8 H3 cells that hold the 16 Primaries, (2)
takes the area-weighted mean over each cell's grid points with the production weights (the weights
of the points that supplied a value are renormalised), writing that native frame as the
checkpoint `native/init=YYYY-MM-DD.parquet`, then (3) interpolates it to half-hours with the
production resampler and writes the mean and the (population) standard deviation over the members
to `ens_summary_tcc/init=YYYY-MM-DD.parquet`. Both writes are atomic, and a run whose files exist
is skipped, so a crashed run costs one run.

Run with the `PYTHONPATH` that `e00561_ens_summary_uv.py` shows, then:

    uv run --no-sync python e00561_ens_tcc.py build --workers 2
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Final

import dynamical_catalog
import e00048_build_frames as build  # ty: ignore[unresolved-import]
import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

DATASET_ID: Final[str] = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
VARIABLE: Final[str] = "total_cloud_cover_atmosphere"
FIRST_RUN: Final[datetime] = datetime(2025, 11, 21, tzinfo=UTC)
"""Runs before this date hold only NaN for the variable."""
LAST_RUN: Final[datetime] = prep.LAST_RUN
WEIGHTS_PATH: Final = common.REPO_DATA_DIR / "h3_grid_weights.parquet"
NATIVE_DIR: Final = prep.WORK_DIR / "work" / "tcc_native"
SUMMARY_DIR: Final = prep.WORK_DIR / "work" / "ens_summary_tcc"
PLACEHOLDER: Final[str] = "wind_u_10m"
"""A continuous instant of the production resampler, which interpolates it linearly in time. The
resampler knows no cloud-cover column, so the cloud cover travels under this name and is renamed
back afterwards."""
SNAP_TOLERANCE_DEGREES: Final[float] = 1e-3


def run_times() -> list[datetime]:
    """Return the daily 00:00 UTC runs from `FIRST_RUN` to `LAST_RUN`."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


def cell_weights() -> pl.DataFrame:
    """Return the production weights of the Primaries' H3 cells: cell, grid point, proportion."""
    cells = prep.primaries()["h3_res_5"].unique().to_list()
    return pl.read_parquet(WEIGHTS_PATH).filter(pl.col("h3_index").is_in(cells))


def native_cell_frame(*, run: datetime) -> pl.DataFrame:
    """Return one run's cell-level cloud cover: `h3_index`, `ensemble_member`, `valid_time`, `tcc`.

    Raises:
        ValueError: The store's run is not `run`, a grid point does not sit on a store point, or
            the run holds no value for a member and lead (the run is then not usable).
    """
    weights = cell_weights()
    store = dynamical_catalog.open(DATASET_ID, chunks=None)[VARIABLE]
    lats = np.sort(weights["nwp_lat"].unique().to_numpy()).astype(np.float64)
    lons = np.sort(weights["nwp_lon"].unique().to_numpy()).astype(np.float64)
    chosen = store.sel(init_time=np.datetime64(run.replace(tzinfo=None), "ns"))
    if np.datetime64(chosen["init_time"].values, "ns") != np.datetime64(
        run.replace(tzinfo=None), "ns"
    ):
        msg = f"store returned run {chosen['init_time'].values}, expected {run}"
        raise ValueError(msg)
    block = chosen.sel(latitude=lats, longitude=lons, method="nearest").load()
    if (
        np.abs(block["latitude"].values - lats).max() > SNAP_TOLERANCE_DEGREES
        or np.abs(block["longitude"].values - lons).max() > SNAP_TOLERANCE_DEGREES
    ):
        msg = "a production grid point does not sit on a store grid point"
        raise ValueError(msg)
    if not np.isfinite(block.values).all():
        msg = f"{run:%Y-%m-%d}: cloud cover holds NaN or infinite values"
        raise ValueError(msg)
    values = block.transpose("lead_time", "ensemble_member", "latitude", "longitude")
    lead = values["lead_time"].values.astype("timedelta64[s]").astype(np.int64)
    members = values["ensemble_member"].values
    grid = pl.DataFrame(
        {
            "lead_s": np.repeat(lead, len(members) * len(lats) * len(lons)),
            "ensemble_member": np.tile(np.repeat(members, len(lats) * len(lons)), len(lead)).astype(
                np.int8
            ),
            "nwp_lat": np.tile(np.repeat(lats, len(lons)), len(lead) * len(members)).astype(
                np.float32
            ),
            "nwp_lon": np.tile(lons, len(lead) * len(members) * len(lats)).astype(np.float32),
            "tcc": values.values.astype(np.float64).ravel(),
        }
    )
    joined = grid.join(
        weights.select("h3_index", "nwp_lat", "nwp_lon", "proportion"),
        on=["nwp_lat", "nwp_lon"],
        how="inner",
    )
    cells = (
        joined.group_by("h3_index", "ensemble_member", "lead_s")
        .agg(tcc=(pl.col("tcc") * pl.col("proportion")).sum() / pl.col("proportion").sum())
        .with_columns(
            valid_time=pl.lit(run).cast(pl.Datetime("us", "UTC"))
            + pl.duration(seconds=pl.col("lead_s")),
            tcc=pl.col("tcc").cast(pl.Float32),
        )
        .drop("lead_s")
    )
    expected = len(weights["h3_index"].unique()) * len(members) * len(lead)
    if cells.height != expected:
        msg = f"{run:%Y-%m-%d}: {cells.height} cell rows, expected {expected}"
        raise ValueError(msg)
    return cells.sort("h3_index", "ensemble_member", "valid_time")


def summary_from_native(*, run: datetime, native: pl.DataFrame) -> pl.DataFrame:
    """Return the per-Primary mean and standard deviation over members, on the half-hourly grid.

    The weather comes from `run`, and the origin is `run` plus 9 h. Every row is asserted to
    follow that rule.
    """
    targets = prep.primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    cells = native.rename({"tcc": PLACEHOLDER}).with_columns(
        nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC"))
    )
    cells = cells.join(
        targets.select("time_series_id", "h3_res_5"),
        left_on="h3_index",
        right_on="h3_res_5",
        how="inner",
    ).drop("h3_index")
    half_hourly = (
        _upsample_nwp_to_half_hourly(
            cells.lazy(),
            solar_coordinates=targets.lazy().select("time_series_id", "latitude", "longitude"),
        )
        .collect()
        .rename({PLACEHOLDER: "total_cloud_cover"})
    )
    origin = common.origin_of(run=run)
    rows = half_hourly.with_columns(
        power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC"))
    ).filter(
        (pl.col("valid_time") > pl.col("power_fcst_init_time"))
        & (
            pl.col("valid_time")
            <= pl.col("power_fcst_init_time") + pl.duration(hours=int(build.HORIZON_HOURS) + 9)
        )
    )
    example = rows.select(
        "time_series_id", "nwp_init_time", "power_fcst_init_time", "valid_time", "ensemble_member"
    ).head(1)
    wrong = rows.filter(
        (pl.col("nwp_init_time") != run)
        | (pl.col("power_fcst_init_time") - pl.col("nwp_init_time") != pl.duration(hours=9))
        | (pl.col("valid_time") <= pl.col("power_fcst_init_time"))
    ).height
    if wrong:
        msg = f"{run:%Y-%m-%d}: {wrong} rows break the origin rule"
        raise ValueError(msg)
    if run == FIRST_RUN:
        print(f"example row (origin rule holds for all {rows.height:,} rows):\n{example}")
    return (
        rows.group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(
            mean_total_cloud_cover=pl.col("total_cloud_cover").mean(),
            std_total_cloud_cover=pl.col("total_cloud_cover").std(ddof=0),
        )
        .sort("time_series_id", "valid_time")
        .cast({"mean_total_cloud_cover": pl.Float32, "std_total_cloud_cover": pl.Float32})
    )


def write_run(*, run: datetime) -> str:
    """Fetch and summarise one run if absent; return a progress line."""
    native_path = NATIVE_DIR / f"init={run:%Y-%m-%d}.parquet"
    summary_path = SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if summary_path.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    if native_path.exists():
        native = pl.read_parquet(native_path)
    else:
        native = native_cell_frame(run=run)
        common.atomic_write(frame=native, target=native_path)
    fetched = time.monotonic() - started
    table = summary_from_native(run=run, native=native)
    common.atomic_write(frame=table, target=summary_path)
    total = time.monotonic() - started
    return f"{run:%Y-%m-%d}: {table.height:,} rows, fetch {fetched:.0f} s, total {total:.0f} s"


def _one(run: datetime) -> str:
    return write_run(run=run)


def main() -> None:
    """Build the files for the requested runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N runs")
    args = parser.parse_args()
    runs = run_times()[: args.limit]
    with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        for line in pool.map(_one, runs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
