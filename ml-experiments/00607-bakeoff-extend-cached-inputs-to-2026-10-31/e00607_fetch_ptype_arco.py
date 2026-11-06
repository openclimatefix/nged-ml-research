# ruff: noqa: D103, E501
"""Read ERA5 precipitation type from Google's ARCO-ERA5 store for 2026-07-28 to 2026-10-31.

The Climate Data Store returns no `precipitation_type`, so this reads the one variable from ARCO,
as the stage-1 box did. One process, 4 threads, one checkpoint per month. Writes the 11 x 11 cell
block in the stored layout to `era5_stage1_box_cds/ptype.parquet`, keeping only the rows after the
stored table's last hour, and prints the largest difference on the four overlapping days.

    uv run --no-sync --with gcsfs --with zarr python e00607_fetch_ptype_arco.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import fetch_era5_stage1_box_arco as stage1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work" / "era5_stage1_box_cds"
CACHE: Final[Path] = WORK / "_ptype_months"
MONTHS: Final[list[tuple[datetime, datetime]]] = [
    (datetime(2026, 7, 28, tzinfo=UTC), datetime(2026, 8, 1, tzinfo=UTC)),
    (datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)),
    (datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)),
    (datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)),
]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    store = stage1.ArcoStore(threads=4)
    variable = stage1.VARIABLES_BY_NAME["ptype"]
    pieces = []
    for start, stop in MONTHS:
        path = CACHE / f"{start:%Y-%m-%d}.parquet"
        if not path.exists():
            hours = list(
                np.arange(
                    np.datetime64(start.replace(tzinfo=None), "h"),
                    np.datetime64(stop.replace(tzinfo=None), "h"),
                    np.timedelta64(1, "h"),
                )
            )
            frame = store.read_month(variable=variable, hours=hours)
            stage1.check_month_values(frame=frame, variable=variable)
            frame.write_parquet(path.with_suffix(".partial"))
            path.with_suffix(".partial").rename(path)
            print(f"{start:%Y-%m-%d}: {frame.height:,} rows", flush=True)
        pieces.append(pl.read_parquet(path))
    new = pl.concat(pieces).sort("time", "latitude", "longitude")
    stored = pl.read_parquet(stage1.OUTPUT_DIR / "ptype.parquet").filter(
        pl.col("time") >= pl.datetime(2026, 7, 28, time_zone="UTC")
    )
    both = new.join(stored, on=["time", "latitude", "longitude"], suffix="_stored")
    print(
        f"overlap {both.height:,} rows, largest difference {(both['value'] - both['value_stored']).abs().max()}"
    )
    new.filter(pl.col("time") > stored["time"].max()).write_parquet(WORK / "ptype.parquet")
    print("wrote ptype.parquet")


if __name__ == "__main__":
    main()
