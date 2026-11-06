# ruff: noqa: E501
"""Fetch 13 of the 14 stage-1 ERA5 variables for 2026-07-28 to 2026-10-31 from the Climate Data Store.

The stage-1 ERA5 box was read from Google's ARCO-ERA5 store, which holds one global field per hour;
extending it to 2026-10-31 that way would download roughly 120 GB, so this script asks the Climate
Data Store, which cuts the same 11 x 11 cell block on the server, for about 60 MB. Both serve ERA5
(Hersbach et al., 2020), so the values agree to the netCDF packing of the Climate Data Store (the
script prints the largest difference on the four overlapping days). One request per month; the first
window starts on 2026-07-28 so that four days overlap the stored table, and `expver` is kept as the
service reports it (`0001` final, `0005` ERA5T, the preliminary release). Writes one parquet per
variable under the experiment's folder, in the stored table's layout (`time`, `latitude`,
`longitude`, `value`, `expver`). The service accepts a request for `precipitation_type` but returns
no such field in its netCDF files, so that variable is not written here; the Google store holds it.
No coordinate is printed or written to a filename.

    uv run --no-sync --with cdsapi --with netCDF4 python e00607_fetch_era5_box_cds.py
"""

import zipfile
from pathlib import Path
from typing import Final

import cdsapi  # ty: ignore[unresolved-import]
import fetch_era5_stage1_box_arco as stage1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xarray as xr
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work" / "era5_stage1_box_cds"
CDS_NAMES: Final[dict[str, str]] = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "sp": "surface_pressure",
    "msl": "mean_sea_level_pressure",
    "tcc": "total_cloud_cover",
    "ssrd": "surface_solar_radiation_downwards",
    "fdir": "total_sky_direct_solar_radiation_at_surface",
    "strd": "surface_thermal_radiation_downwards",
    "tp": "total_precipitation",
}
WINDOWS: Final[list[tuple[int, int, list[int]]]] = [
    (2026, 7, [28, 29, 30, 31]),
    (2026, 8, list(range(1, 32))),
    (2026, 9, list(range(1, 31))),
    (2026, 10, list(range(1, 32))),
]


def area() -> list[float]:
    """Return the request area (north, west, south, east) of the stored box, padded by 0.01."""
    lat_q, lon_q = stage1.cell_axes()
    step = stage1.GRID_STEP_DEG
    return [
        max(lat_q) * step + 0.01,
        min(lon_q) * step - 0.01,
        min(lat_q) * step - 0.01,
        max(lon_q) * step + 0.01,
    ]


def fetch_window(*, client: cdsapi.Client, year: int, month: int, days: list[int]) -> Path:
    """Download one window of all variables as netCDF, skipping it if the file exists."""
    target = WORK / f"cds_{year}-{month:02d}.nc"
    if target.exists():
        return target
    request = {
        "product_type": ["reanalysis"],
        "variable": list(CDS_NAMES.values()),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area(),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    partial = target.with_suffix(".partial")
    client.retrieve("reanalysis-era5-single-levels", request, str(partial))
    partial.rename(target)
    return target


def open_all(*, path: Path) -> list[xr.Dataset]:
    """Open the netCDF, or each netCDF inside a zip the service may return."""
    if zipfile.is_zipfile(path):
        folder = path.with_suffix("")
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(folder)
        return [xr.open_dataset(p) for p in sorted(folder.glob("*.nc"))]
    return [xr.open_dataset(path)]


def tidy(*, datasets: list[xr.Dataset], short: str) -> pl.DataFrame:
    """Return one variable as the stored table's layout, merging `expver` slots if present."""
    pieces = []
    for dataset in datasets:
        names = [
            v for v in dataset.data_vars if v in (short, stage1.VARIABLES_BY_NAME[short].short_name)
        ]
        if not names:
            continue
        array = dataset[names[0]]
        expver_dim = "expver" in array.dims
        time_name = "valid_time" if "valid_time" in array.dims else "time"
        if expver_dim:
            labels = [f"{int(e):04d}" for e in array["expver"].values]
            for label, sub in zip(labels, array.transpose("expver", time_name, ...), strict=True):
                pieces.append(_frame(array=sub, time_name=time_name, expver=label))
        else:
            expver = str(dataset["expver"].values.ravel()[0]) if "expver" in dataset else "0001"
            pieces.append(_frame(array=array, time_name=time_name, expver=expver.zfill(4)))
    return pl.concat(pieces).drop_nulls("value").sort("time", "latitude", "longitude")


def _frame(*, array: xr.DataArray, time_name: str, expver: str) -> pl.DataFrame:
    array = array.transpose(time_name, "latitude", "longitude")
    times = array[time_name].values
    lats = array["latitude"].values.astype(np.float64)
    lons = array["longitude"].values.astype(np.float64)
    values = array.values.astype(np.float32)
    n = len(times)
    return pl.DataFrame(
        {
            "time": pl.Series(
                np.repeat(times.astype("datetime64[us]"), len(lats) * len(lons))
            ).dt.replace_time_zone("UTC"),
            "latitude": np.tile(np.repeat(lats, len(lons)), n),
            "longitude": np.tile(lons, n * len(lats)),
            "value": values.reshape(-1),
            "expver": [expver] * values.size,
        }
    ).filter(pl.col("value").is_not_nan())


def main() -> None:
    """Fetch every window, write one parquet per variable, and compare the overlap."""
    WORK.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(quiet=True)
    files = [fetch_window(client=client, year=y, month=m, days=d) for y, m, d in WINDOWS]
    opened = [open_all(path=f) for f in files]
    stored_dir = stage1.OUTPUT_DIR
    for short in CDS_NAMES:
        frame = pl.concat([tidy(datasets=ds, short=short) for ds in opened]).unique(
            subset=["time", "latitude", "longitude"], keep="last"
        )
        stored = (
            pl.scan_parquet(stored_dir / f"{short}.parquet")
            .filter(pl.col("time") >= pl.datetime(2026, 7, 28, time_zone="UTC"))
            .collect()
        )
        both = frame.join(stored, on=["time", "latitude", "longitude"], suffix="_stored")
        variable = stage1.VARIABLES_BY_NAME[short]
        worst = (both["value"] - both["value_stored"]).abs().max()
        print(
            f"{short}: {frame.height:,} rows; overlap {both.height:,} rows with the stored table, "
            f"largest difference {worst:.3g} ({worst / variable.maximum:.1e} of the variable's maximum)",  # ty: ignore[unsupported-operator]
            flush=True,
        )
        new = frame.filter(pl.col("time") > stored["time"].max())
        new.write_parquet(WORK / f"{short}.parquet")
        print(
            f"  new rows {new.height:,}; expver {new['expver'].value_counts().sort('expver').rows()}"
        )


if __name__ == "__main__":
    main()
