"""Build D15's ensemble arrays from the local ECMWF ENS archive and fit its calibration.

D15 has no stored arrays because no earlier experiment fitted a stack there. The arrays are cut
from the archive that the production `ecmwf_ens` asset writes (read only, no fetch), at D15's H3
cell, with the extraction of experiment 152 (`read_run`), then compared with an existing Primary's
stored arrays for the same runs. The calibration is fitted on the tuning months with experiment
111's function.

    E00377_SOURCE=d15 uv run --no-sync python ml-experiments/00377-*/e00377_ens_inputs.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Final

import e00377_common as common
import e00377_ens as ens
import h3
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.dp_features import default_input_paths
from studies.sources import REPO_DATA_DIR

N_MEMBERS: Final[int] = 51
N_TARGETS: Final[int] = 702
"""Half-hours in a frame: 351 hours after the origin."""
ARCHIVE_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
H3_RESOLUTION: Final[int] = 5
HORIZON_HOURS: Final[int] = 351
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    "temperature_2m",
    "downward_short_wave_radiation_flux_surface",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return ARCHIVE_DIR / f"init_time={stamp}"


def site_table(*, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return `site`, `time_series_id`, `latitude`, `longitude`, and `h3_res_5` of some sites.

    The coordinates stay inside the process; only the arrays leave.
    """
    series = pl.read_parquet(default_input_paths().series_table)
    kept = series.filter(pl.col("site").is_in(list(sites))).select(
        "site", "time_series_id", "latitude", "longitude"
    )
    return kept.with_columns(
        h3_res_5=pl.struct("latitude", "longitude").map_elements(
            lambda row: h3.str_to_int(
                h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION)
            ),
            return_dtype=pl.Int64,
        ),
        time_series_id=pl.col("time_series_id").cast(pl.Int32),
    )


def read_run(*, run: datetime, sites: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return one run's temperature and shortwave per site, shape (members, targets) each.

    The run is cut to the sites' H3 cells and resampled to the half-hourly grid by production's
    `_upsample_nwp_to_half_hourly`, as experiment 111's frame builder did, and cut to valid times
    from the run's `power_fcst_init_time` to 351 hours later.

    Args:
        run: The run's date.
        sites: One row per site, with `time_series_id`, `h3_res_5`, `latitude`, and `longitude`.

    Returns:
        Each site's label mapped to its temperature and shortwave arrays.

    Raises:
        ValueError: If a site lacks a member or a half-hour, or a key repeats.
    """
    cells = sites["h3_res_5"].unique().to_list()
    targets = sites.select("time_series_id", "h3_res_5", "latitude", "longitude")
    native = (
        pl.scan_parquet(partition_path(run=run) / "*.parquet")
        .select(SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = targets.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates)
        .collect()
        .rename({"downward_short_wave_radiation_flux_surface": "shortwave"})
    )
    origin = run + common.PUBLICATION_DELAY
    frame = half_hourly.with_columns(
        lead_hours=(
            pl.col("valid_time") - pl.lit(origin).cast(pl.Datetime("us", "UTC"))
        ).dt.total_minutes()
        / 60.0
    ).filter((pl.col("lead_hours") > 0.0) & (pl.col("lead_hours") <= HORIZON_HOURS))
    if frame.select("time_series_id", "ensemble_member", "valid_time").is_duplicated().any():
        msg = f"the {run:%Y-%m-%d} frame repeats a key"
        raise ValueError(msg)
    out = {}
    for row in sites.iter_rows(named=True):
        rows = frame.filter(pl.col("time_series_id") == row["time_series_id"]).sort(
            "ensemble_member", "valid_time"
        )
        if rows.height != N_MEMBERS * N_TARGETS:
            msg = (
                f"{row['site']} on {run:%Y-%m-%d} has {rows.height} rows, "
                f"not {N_MEMBERS * N_TARGETS}"
            )
            raise ValueError(msg)
        out[row["site"]] = (
            rows["temperature_2m"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
            rows["shortwave"].to_numpy().reshape(N_MEMBERS, N_TARGETS),
        )
    return out


def build(*, sites: tuple[str, ...], out_dir: Path) -> None:
    """Read every run and write one array file per site under `out_dir`."""
    table = site_table(sites=sites)
    runs = common.run_dates()
    started = time.monotonic()
    results = [read_run(run=run, sites=table) for run in runs]
    out_dir.mkdir(parents=True, exist_ok=True)
    origins = np.array(
        [np.datetime64(common.origin_of(run=run).replace(tzinfo=None), "us") for run in runs]
    )
    for site in sites:
        temperature = np.stack([r[site][0] for r in results])
        shortwave = np.stack([r[site][1] for r in results])
        np.savez(
            out_dir / f"{site}.npz", origins=origins, temperature=temperature, shortwave=shortwave
        )
        elapsed = time.monotonic() - started
        print(f"{site}: {temperature.shape}, first run {runs[0]:%Y-%m-%d}, {elapsed:.0f} s")


def check_against_stored_d10() -> None:
    """Rebuild D10's arrays and compare them with experiment 111's stored arrays."""
    check = common.WORK_DIR / "check_store"
    build(sites=("D10",), out_dir=check)
    with (
        np.load(check / "D10.npz") as mine,
        np.load(common.SOURCE_WORK_DIR / "ens_store" / "D10.npz") as stored,
    ):
        for name in ("temperature", "shortwave"):
            both = np.isfinite(mine[name]) & np.isfinite(stored[name])
            gap = np.abs(mine[name][both] - stored[name][both]).max()
            same_gaps = bool((np.isnan(mine[name]) == np.isnan(stored[name])).all())
            print(f"D10 {name}: max |rebuilt - stored| = {gap:.3g}, gaps equal {same_gaps}")
        print("origins equal:", bool((mine["origins"] == stored["origins"]).all()))


def main() -> None:
    """Build D15's store, check the extraction against D10's stored arrays, fit the calibration."""
    if "--calibration-only" not in sys.argv:
        build(sites=("D15",), out_dir=common.STORE_DIR)
        check_against_stored_d10()
    cache = common.variant_cache(cache=common.load_cache(), site="D15")
    setup = ens.load_setup(cache=cache, site="D15")
    calibration = ens.fit_calibration(cache=cache, setup=setup)
    common.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        common.CALIBRATION_DIR / "D15.npz",
        shift=calibration.temperature_shift_c,
        ratio=calibration.irradiance_ratio,
        shrink=calibration.solar_shrink,
    )
    print("calibration written")


if __name__ == "__main__":
    main()
