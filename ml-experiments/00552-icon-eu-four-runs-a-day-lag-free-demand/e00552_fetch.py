# ruff: noqa: E402
"""Fetch every Dynamical.org ICON-EU run from 2026-02-10 to 2026-06-30 at the 16 Primary points.

One parquet per run goes to `OUTPUT_DIR / runs`, so a re-run skips finished runs. Coordinates stay
in this process; the output holds the Primary index (0 to 15), not a coordinate.
"""

import os
import sys
import time
from pathlib import Path
from typing import Final

import dynamical_catalog
import numpy as np
import polars as pl
import xarray as xr

sys.path[:0] = [str(p) for p in Path("ml-experiments").glob("00087*")]
import e00087_common as common  # ty: ignore[unresolved-import]

DATASET_ID: Final[str] = "dwd-icon-eu-forecast-5-day"
OUTPUT_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"]) / "studies" / "icon_eu_multirun" / "runs"
)
VARIABLES: Final[tuple[str, ...]] = (
    "downward_direct_short_wave_radiation_flux_surface",
    "downward_diffuse_short_wave_radiation_flux_surface",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
)
FIRST_RUN: Final[str] = "2026-02-10T00:00"
LAST_RUN: Final[str] = "2026-06-30T18:00"
N_LEADS: Final[int] = 79
"""Hourly leads 0 to 78 h."""


def main() -> None:
    """Fetch and checkpoint each run in turn."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store = dynamical_catalog.open(DATASET_ID, chunks=None)[list(VARIABLES)]
    primaries = common.series_table().filter(pl.col("substation_type") == "Primary")
    lat, lon = store["latitude"].to_numpy(), store["longitude"].to_numpy()
    indexers = {
        "latitude": xr.DataArray(
            [int(np.argmin(abs(lat - v))) for v in primaries["latitude"]], dims="primary"
        ),
        "longitude": xr.DataArray(
            [int(np.argmin(abs(lon - v))) for v in primaries["longitude"]], dims="primary"
        ),
    }
    inits = store["init_time"].to_numpy()
    inits = inits[(inits >= np.datetime64(FIRST_RUN)) & (inits <= np.datetime64(LAST_RUN))]
    print(f"{len(inits)} runs", flush=True)
    for i, init in enumerate(inits):
        path = OUTPUT_DIR / f"{str(init)[:16].replace(':', '').replace('-', '')}.parquet"
        if path.exists():
            continue
        start = time.time()
        run = store.sel(init_time=init).isel(lead_time=slice(0, N_LEADS)).isel(indexers).load()
        frame = pl.from_pandas(run.to_dataframe().reset_index()).with_columns(
            init_time=pl.lit(init).cast(pl.Datetime("ns"))
        )
        tmp = path.with_suffix(".tmp")
        frame.write_parquet(tmp)
        tmp.rename(path)
        print(f"{i + 1}/{len(inits)} {path.stem} {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
