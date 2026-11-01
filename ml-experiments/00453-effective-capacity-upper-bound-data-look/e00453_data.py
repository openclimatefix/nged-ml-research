"""Experiment 453: load the nine metered series, their power, capacity, and weather proxies.

Writes `series.parquet` (columns `site`, `tech`, `time`, `power`, `cap`, `wind`, `clear`) in
`WORK_DIR`. `wind` is ERA5 100 m wind speed in m/s (wind farms only) and `clear` is the Haurwitz
clear-sky global horizontal irradiance in W/m2 (solar series only). Coordinates stay in this
process.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00451-*/env.sh
uv run --no-sync python ml-experiments/00453-*/e00453_data.py
```
"""

from pathlib import Path
from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.power import scan_power
from studies.pv_dataset import CAPACITY_DELTA_URI, pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00453-effective-capacity-upper-bound-data-look"
WORK_DIR: Final[Path] = DATA_DIR / "work"
SERIES_PATH: Final[Path] = WORK_DIR / "series.parquet"
SOLAR_CONSTANT_W_M2: Final[float] = 1361.0


def labels() -> pl.DataFrame:
    """Return `time_series_id`, `site` (Farm A.. / Solar A..), `tech`, coordinates, and capacity."""
    wind = wind_sites().with_columns(
        tech=pl.lit("wind"),
        site=pl.col("site").replace_strict({"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}),
    )
    solar = pv_sites().with_columns(tech=pl.lit("solar"), site="Solar " + pl.col("site"))
    sites = pl.concat([wind, solar]).select(
        "time_series_id", "site", "tech", "latitude", "longitude"
    )
    cap = (
        pl.scan_delta(CAPACITY_DELTA_URI)
        .sort("time")
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").last().alias("cap"))
        .collect()
    )
    return sites.join(cap, on="time_series_id", how="left", validate="1:1")


def haurwitz_ghi(*, time: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """Return the Haurwitz clear-sky GHI (W/m2) at UTC datetime64 `time` for a location.

    The stamps are period-ending half-hours, so the irradiance is evaluated 15 minutes earlier, at
    the middle of the period.
    """
    t = time - np.timedelta64(15, "m")
    day = (t.astype("datetime64[D]") - t.astype("datetime64[Y]").astype("datetime64[D]")).astype(
        float
    ) + 1.0
    hours = (t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")
    gamma = 2 * np.pi / 365.0 * (day - 1.0 + (hours - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
    )
    equation_of_time_min = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    solar_hours = hours + longitude / 15.0 + equation_of_time_min / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_hours - 12.0))
    lat = np.deg2rad(latitude)
    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        hour_angle
    )
    cos_zenith = np.clip(cos_zenith, 1e-6, None)
    ghi = 1098.0 * cos_zenith * np.exp(-0.059 / cos_zenith)
    return np.where(cos_zenith > 0.01, ghi, 0.0)


def build() -> None:
    """Write `work/series.parquet`."""
    sites = labels()
    ids = sites["time_series_id"].to_list()
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids))
        .select("time_series_id", "time", "power")
        .collect()
        .filter(pl.col("power").is_not_null() & pl.col("power").is_finite())
        .sort("time_series_id", "time")
    )
    wind_ids = sites.filter(pl.col("tech") == "wind")["time_series_id"].to_list()
    era5 = (
        pl.scan_parquet(c230.ANALYSIS_PATH)
        .filter(pl.col("time_series_id").is_in(wind_ids))
        .select("time_series_id", pl.col("valid_time").alias("time"), wind="wind_speed_100m")
        .collect()
    )
    frame = power.join(era5, on=["time_series_id", "time"], how="left").join(
        sites.drop("latitude", "longitude"), on="time_series_id"
    )
    parts = []
    for row in sites.filter(pl.col("tech") == "solar").iter_rows(named=True):
        part = frame.filter(pl.col("time_series_id") == row["time_series_id"])
        ghi = haurwitz_ghi(
            time=part["time"].dt.replace_time_zone(None).to_numpy(),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        parts.append(part.with_columns(clear=pl.Series(ghi, dtype=pl.Float64)))
    out = pl.concat(
        [
            frame.filter(pl.col("tech") == "wind").with_columns(
                clear=pl.lit(None, dtype=pl.Float64)
            ),
            *parts,
        ],
        how="diagonal_relaxed",
    )
    out = out.select("site", "tech", "time", "power", "cap", "wind", "clear").sort("site", "time")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(SERIES_PATH)
    print(out.group_by("site").agg(pl.len(), pl.col("wind").is_not_null().sum().alias("n_wind")))


if __name__ == "__main__":
    build()
