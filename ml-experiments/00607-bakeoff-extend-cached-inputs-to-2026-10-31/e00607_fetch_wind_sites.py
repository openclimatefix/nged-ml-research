"""Fetch Open-Meteo's ERA5 wind at the three wind sites, 2026-08-20 to 2026-10-31, and validate.

The script that built `wind_era5.parquet` is not in the repository, so this script re-creates the
request (hourly `wind_speed_100m`, `wind_direction_100m`, `wind_speed_10m`; `models=era5`; UTC; km/h)
and first checks that it reproduces the stored table over the window 2026-08-20 to 2026-09-10,
which `wind_era5.parquet` already holds. It writes only the new window to
`wind_era5_2026-09-11_2026-10-31.parquet` beside the stored table. Rows are keyed by the anonymous
label `W1` to `W3`; no coordinate is written or printed.

    uv run --no-sync python e00607_fetch_wind_sites.py
"""

import json
import urllib.request
from datetime import date
from typing import Final

import polars as pl
from studies.pv_dataset import wind_sites
from studies.sources import ERA5_SITE_POINTS_DIR

STORED: Final = ERA5_SITE_POINTS_DIR / "wind_era5.parquet"
OUTPUT: Final = ERA5_SITE_POINTS_DIR / "wind_era5_2026-09-11_2026-10-31.parquet"
FIRST: Final[date] = date(2026, 8, 20)
LAST: Final[date] = date(2026, 10, 31)
VARIABLES: Final[tuple[str, ...]] = ("wind_speed_100m", "wind_direction_100m", "wind_speed_10m")


def fetch_site(*, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the hourly table for one site."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}&start_date={FIRST}&end_date={LAST}"
        f"&hourly={','.join(VARIABLES)}&models=era5&timezone=UTC"
    )
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310
        payload = json.loads(response.read())
    hourly = payload["hourly"]
    return pl.DataFrame({"time": hourly["time"], **{v: hourly[v] for v in VARIABLES}}).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M").dt.replace_time_zone("UTC")
    )


def main() -> None:
    """Fetch, validate against the stored table, and write the new window."""
    frames = []
    for row in wind_sites().iter_rows(named=True):
        frames.append(
            fetch_site(latitude=row["latitude"], longitude=row["longitude"]).with_columns(
                site=pl.lit(row["site"])
            )
        )
    new = pl.concat(frames).select("site", "time", *VARIABLES).drop_nulls().sort("site", "time")
    stored = pl.read_parquet(STORED)
    overlap = new.join(stored, on=["site", "time"], suffix="_stored")
    print(f"overlap rows with the stored table: {overlap.height:,} (stored rows in window: "
          f"{stored.filter(pl.col('time') >= pl.datetime(2026, 8, 20, time_zone='UTC')).height:,})")
    for v in VARIABLES:
        gap = (overlap[v] - overlap[f"{v}_stored"]).abs().max()
        print(f"{v}: largest absolute difference from the stored table {gap}")
    out = new.filter(pl.col("time") > stored["time"].max())
    out.write_parquet(OUTPUT)
    print(f"wrote {out.height:,} rows, {out['time'].min()} to {out['time'].max()}")
    expected = 3 * 24 * ((LAST - date(2026, 9, 11)).days + 1)
    print(f"expected {expected:,} rows; sites {out['site'].n_unique()}")


if __name__ == "__main__":
    main()
