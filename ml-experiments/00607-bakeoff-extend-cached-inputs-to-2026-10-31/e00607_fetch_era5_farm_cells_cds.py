# ruff: noqa: E501
"""Extend experiment 288's ERA5 farm-cell wind table from 2024-04 to 2026-10 via the Climate Data Store.

Experiment 288 read the 43 farm cells from Google's ARCO-ERA5 store. Another 31 months that way would
read about 90,000 global fields (roughly 360 GB), so this script asks the Climate Data Store for the
bounding box of the 43 cells (four wind variables, hourly, the same ERA5 fields), cuts the cells out,
and writes the new rows in the stored layout (`cell_id`, `time`, `u10`, `v10`, `u100`, `v100`) to
`era5_farm_cells_2024-04_2026-10.parquet` in the experiment's data folder. The overlap check compares
the first request's first 3 days of April... against none (the stored table ends 2024-03-31), so
the request starts on 2024-03-29 to give 3 days of overlap. No coordinate is printed.

    PYTHONPATH=ml-experiments/00288-pooled-head-on-public-wind-farms \
        uv run --no-sync --with cdsapi --with netCDF4 python e00607_fetch_era5_farm_cells_cds.py
"""

from pathlib import Path
from typing import Final

import cdsapi  # ty: ignore[unresolved-import]
import e00288_fetch_era5 as farm  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xarray as xr
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work" / "era5_farm_cells_cds"
CDS_NAMES: Final[dict[str, str]] = farm.VARIABLES
WINDOWS: Final[list[tuple[int, list[int], None]]] = [
    (2024, [3], None),
    (2024, [4, 5, 6], None),
    (2024, [7, 8, 9], None),
    (2024, [10, 11, 12], None),
    (2025, [1, 2, 3], None),
    (2025, [4, 5, 6], None),
    (2025, [7, 8, 9], None),
    (2025, [10, 11, 12], None),
    (2026, [1, 2, 3], None),
    (2026, [4, 5, 6], None),
    (2026, [7, 8, 9], None),
    (2026, [10], None),
]
"""Three-month requests: the service refused 6 to 10 months of four variables over this box as too large."""


def main() -> None:
    """Fetch the windows, cut the cells, compare the overlap, and write the new rows."""
    WORK.mkdir(parents=True, exist_ok=True)
    cells = farm.farm_cells().select("cell_id", "lat_q", "lon_q").unique().sort("cell_id")
    north = (cells["lat_q"].max() + 1) / 4
    south = (cells["lat_q"].min() - 1) / 4
    west = (cells["lon_q"].min() - 1) / 4
    east = (cells["lon_q"].max() + 1) / 4
    client = cdsapi.Client(quiet=True)
    frames = []
    for year, months, _ in WINDOWS:
        target = WORK / f"cds_{year}_{months[0]:02d}-{months[-1]:02d}.nc"
        if not target.exists():
            partial = target.with_suffix(".partial")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": list(CDS_NAMES.values()),
                    "year": [str(year)],
                    "month": [f"{m:02d}" for m in months],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [north, west, south, east],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(partial),
            )
            partial.rename(target)
        frames.append(cut(path=target, cells=cells))
    new = pl.concat(frames).unique(subset=["cell_id", "time"]).sort("cell_id", "time")
    stored = pl.read_parquet(farm.OUT_DIR / "era5_farm_cells.parquet").filter(
        pl.col("time") >= pl.datetime(2024, 3, 29, time_zone="UTC")
    )
    both = new.join(stored, on=["cell_id", "time"], suffix="_stored")
    for v in CDS_NAMES:
        print(
            f"{v}: overlap {both.height:,} rows, largest difference {(both[v] - both[f'{v}_stored']).abs().max():.3g} m/s"
        )
    out = new.filter(pl.col("time") > pl.datetime(2024, 3, 31, 23, time_zone="UTC"))
    out.write_parquet(WORK.parent / "era5_farm_cells_2024-04_2026-10.parquet")
    print(
        f"wrote {out.height:,} rows, {out['time'].min()} to {out['time'].max()}, {out['cell_id'].n_unique()} cells"
    )


def cut(*, path: Path, cells: pl.DataFrame) -> pl.DataFrame:
    """Return the farm cells' four variables from one downloaded window."""
    import zipfile

    files = [path]
    if zipfile.is_zipfile(path):
        folder = path.with_suffix("")
        folder.mkdir(exist_ok=True)
        zipfile.ZipFile(path).extractall(folder)
        files = sorted(folder.glob("*.nc"))
    pieces: dict[str, pl.DataFrame] = {}
    for file in files:
        dataset = xr.open_dataset(file)
        time_name = "valid_time" if "valid_time" in dataset.dims else "time"
        for short in CDS_NAMES:
            if short not in dataset:
                continue
            array = dataset[short]
            if "expver" in array.dims:
                array = array.bfill("expver").isel(expver=0)
            lat_idx = [
                int(np.argmin(np.abs(dataset["latitude"].values - q / 4))) for q in cells["lat_q"]
            ]
            lon_idx = [
                int(np.argmin(np.abs(dataset["longitude"].values - (q / 4))))
                for q in cells["lon_q"]
            ]
            values = array.transpose(time_name, "latitude", "longitude").values[:, lat_idx, lon_idx]
            hours = dataset[time_name].values.astype("datetime64[us]")
            pieces[short] = pl.DataFrame(
                {
                    "cell_id": np.tile(cells["cell_id"].to_numpy(), len(hours)).astype(np.int32),
                    "time": pl.Series(np.repeat(hours, cells.height)).dt.replace_time_zone("UTC"),
                    short: values.reshape(-1).astype(np.float32),
                }
            )
    out = pieces["u10"]
    for short in list(CDS_NAMES)[1:]:
        out = out.join(pieces[short], on=["cell_id", "time"], how="inner")
    return out.drop_nulls()


if __name__ == "__main__":
    main()
