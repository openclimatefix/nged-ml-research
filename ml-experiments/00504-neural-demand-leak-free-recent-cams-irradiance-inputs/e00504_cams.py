"""CAMS input columns for experiment 504, built from the stored CAMS table.

Two kinds of column, all derived from `cams_demand_points.parquet` (hourly CAMS irradiance and
clear-sky irradiance per Primary, hour-ending stamps in UTC):

- Leak-free columns, which depend on the forecast origin only through the latest whole UTC day
  that a live service could hold. The table has no availability stamp, so the plan assumes a
  delay of 36 hours: day D is available at an origin T when D + 1 00:00 UTC is at or before
  T - 36 h. Origins are 09:00 UTC, so the latest available day is the third day before the
  origin's date. `cams_recent_ghi` is the half-hourly CAMS irradiance at the target half-hour's
  time of day on that day, `cams_recent_ghi_3day` is its mean over that day and the two before,
  and `cams_recent_cloudiness` is that day's summed irradiance over its summed clear-sky
  irradiance.
- Realised columns, which read the target half-hour itself (experiment 408 and 480's columns):
  `cams_unit_pv_shape` and `cams_clear_sky_index`. A serving system cannot supply them at most
  leads; they are the report-only upper bound.

`extra_columns` returns the five columns for arrays of (origin, valid) times. `stale_days` moves
the leak-free day earlier (the one-day-earlier check).
"""

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.dp_features import CAMS_PATH, FLEET
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw

COLUMNS: Final[tuple[str, ...]] = (
    "cams_recent_ghi",
    "cams_recent_ghi_3day",
    "cams_recent_cloudiness",
    "cams_unit_pv_shape",
    "cams_clear_sky_index",
)
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "A-cams": COLUMNS[:3],
    "A-cams-real": COLUMNS[3:],
    "A-cams-lag-only": COLUMNS[2:3],
}
DELAY_HOURS: Final[float] = 36.0
STEP_US: Final[int] = 1_800_000_000
HOUR_US: Final[int] = 3_600_000_000
DAY_US: Final[int] = 86_400_000_000
STEPS_PER_DAY: Final[int] = 48
MIN_CLEAR_SKY_W_M2: Final[float] = 1.0
MAX_CLEAR_SKY_INDEX: Final[float] = 1.5


@dataclass(frozen=True)
class SiteCams:
    """One Primary's CAMS series on the half-hourly grid and its daily cloudiness."""

    first_us: int
    """The first half-hour end stamp, microseconds since the epoch."""
    ghi: np.ndarray
    unit_pv: np.ndarray
    clear_sky_index: np.ndarray
    first_day: int
    """The UTC day number (days since the epoch) of the first entry of `cloudiness`."""
    cloudiness: np.ndarray


@cache
def site_cams(*, site: str, latitude: float, longitude: float) -> SiteCams:
    """Return a Primary's half-hourly CAMS arrays and daily cloudiness (cached)."""
    table = (
        pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2", "clear_sky_ghi_w_m2"])
        .filter(pl.col("site") == site)
        .sort("time")
    )
    assert table.drop_nulls().height == table.height, f"{site}: the CAMS table has nulls"
    hourly = table.select("time", pl.col("ghi_w_m2").cast(pl.Float64))
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=latitude, longitude=longitude)
    stamps = sky.half_hour_end_time.astype("datetime64[us]").astype(np.int64)
    assert (np.diff(stamps) == STEP_US).all(), f"{site}: the half-hourly grid has gaps"
    clear = haurwitz_w_m2(apparent_zenith_deg=sky.zenith_deg)
    index = np.divide(
        sky.ghi_w_m2, clear, out=np.zeros_like(clear), where=clear >= MIN_CLEAR_SKY_W_M2
    )
    # A stamp at hour h belongs to the hour before it, so shift back one hour to get its UTC day.
    hour_us = table["time"].dt.epoch("us").to_numpy()
    day = (hour_us - HOUR_US) // DAY_US
    days, inverse = np.unique(day, return_inverse=True)
    ghi_sum = np.bincount(inverse, weights=table["ghi_w_m2"].to_numpy().astype(np.float64))
    clear_sum = np.bincount(
        inverse, weights=table["clear_sky_ghi_w_m2"].to_numpy().astype(np.float64)
    )
    assert (np.diff(days) == 1).all()
    cloudiness = np.divide(ghi_sum, clear_sum, out=np.zeros_like(ghi_sum), where=clear_sum > 0)
    return SiteCams(
        first_us=int(stamps[0]),
        ghi=sky.ghi_w_m2.astype(np.float32),
        unit_pv=plant_power_mw(sky=sky, parameters=FLEET).astype(np.float32),
        clear_sky_index=np.clip(index, 0.0, MAX_CLEAR_SKY_INDEX).astype(np.float32),
        first_day=int(days[0]),
        cloudiness=cloudiness.astype(np.float32),
    )


def latest_day(*, origin_us: np.ndarray, stale_days: int = 0) -> np.ndarray:
    """Return the UTC day number of the latest whole day available at each origin.

    Day D is available when D + 1 00:00 UTC is at or before the origin minus the delay. Origins
    at 09:00 UTC give the third day before the origin's date (21:00 UTC two days earlier is the
    cutoff). `stale_days` moves the day earlier.
    """
    cutoff = origin_us - int(DELAY_HOURS * HOUR_US)
    return cutoff // DAY_US - 1 - stale_days


def extra_columns(
    *,
    cams: SiteCams,
    origin_us: np.ndarray,
    valid_us: np.ndarray,
    stale_days: int = 0,
) -> np.ndarray:
    """Return the five raw CAMS columns (`COLUMNS` order), float32, for rows of (origin, valid).

    Raises:
        ValueError: A needed CAMS value lies outside the table.
    """
    day = latest_day(origin_us=origin_us, stale_days=stale_days)
    # A half-hour ending exactly at 00:00 is the last half-hour of its day, so tod runs 1..48 steps.
    tod = (valid_us - 1) % DAY_US + 1
    out = np.empty((len(origin_us), len(COLUMNS)), dtype=np.float32)
    # Index of the stamp "time of day tod on day d": (d * DAY + tod - first) / STEP.
    idx = [((day - j) * DAY_US + tod - cams.first_us) // STEP_US for j in range(3)]
    real = (valid_us - cams.first_us) // STEP_US
    cloud = day - cams.first_day
    for name, index, size in (
        ("recent", np.concatenate(idx), len(cams.ghi)),
        ("real", real, len(cams.ghi)),
        ("cloud", cloud, len(cams.cloudiness)),
    ):
        if index.min() < 0 or index.max() >= size:
            msg = f"CAMS {name} index outside the table: {index.min()}..{index.max()} of {size}"
            raise ValueError(msg)
    ghi = np.stack([cams.ghi[i] for i in idx])
    out[:, 0] = ghi[0]
    out[:, 1] = ghi.mean(axis=0)
    out[:, 2] = cams.cloudiness[cloud]
    out[:, 3] = cams.unit_pv[real]
    out[:, 4] = cams.clear_sky_index[real]
    return out
