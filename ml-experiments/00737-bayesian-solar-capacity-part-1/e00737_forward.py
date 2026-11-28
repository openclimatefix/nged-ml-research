"""The forward model of experiment 737: the AC output of one megawatt of a fleet of panels.

The fit's solar term and the injected fleets come from the same function, `fleet_unit_output`. It
turns a `studies.pv_physics.SunAndSky` and the air temperature and wind speed into the output of one
megawatt of AC capacity: the plane-of-array irradiance of east-, south-, and west-facing panels, the
Faiman cell temperature, the temperature derate of the DC power, and then the clipping at the AC
capacity. The derate comes before the clipping, because an inverter clips the DC power that the
hot panels actually deliver. A half-hour with the sun at or below `MIN_ELEVATION_DEG` is exactly
zero. `studies.pv_physics` and `studies.pv_separation` hold no cell-temperature model, so the
derate is new code here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from e00737_common import (
    FAIMAN_U0,
    FAIMAN_U1,
    FLEET_MIX,
    MIN_ELEVATION_DEG,
    PRIOR_DC_AC,
    TEMPERATURE_COEFFICIENT_PER_K,
)
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.pv_separation import FLEET_ORIENTATIONS, FLEET_TILT_DEG

STANDARD_CELL_TEMPERATURE_C: Final[float] = 25.0
"""The cell temperature at which a panel's DC rating is quoted."""
FIXED_ORIENTATIONS: Final[tuple[float, float, float]] = (
    FLEET_ORIENTATIONS[0][1],
    FLEET_ORIENTATIONS[1][1],
    FLEET_ORIENTATIONS[2][1],
)
"""The azimuths of the east, south, and west panels, clockwise from north."""


@dataclass(frozen=True)
class FleetParams:
    """The shape of a fleet: its orientation mix, tilt, and DC:AC ratio.

    Attributes:
        mix: The east, south, and west shares, summing to 1.
        tilt_deg: The panels' tilt from the horizontal.
        dc_ac_ratio: The DC rating divided by the AC capacity.
    """

    mix: tuple[float, float, float]
    tilt_deg: float
    dc_ac_ratio: float


PRIOR_FLEET: Final[FleetParams] = FleetParams(
    mix=FLEET_MIX, tilt_deg=FLEET_TILT_DEG, dc_ac_ratio=PRIOR_DC_AC
)
"""The fleet that the fit assumes."""


def faiman_cell_temperature_c(
    *, poa_w_m2: np.ndarray, air_c: np.ndarray, wind_ms: np.ndarray
) -> np.ndarray:
    """Return the Faiman cell temperature, `T_air + G / (u0 + u1 x wind)`.

    Args:
        poa_w_m2: The plane-of-array irradiance `G`.
        air_c: The air temperature in degrees Celsius.
        wind_ms: The wind speed in metres per second.

    Returns:
        The cell temperature in degrees Celsius.
    """
    return air_c + poa_w_m2 / (FAIMAN_U0 + FAIMAN_U1 * wind_ms)


def derated_ac_unit_output(
    *, poa_w_m2: np.ndarray, air_c: np.ndarray, wind_ms: np.ndarray, dc_ac_ratio: float
) -> np.ndarray:
    """Return the AC output of one megawatt of AC capacity: derate first, then clip at 1 MW.

    The DC power is the DC rating (the AC capacity times the ratio) times `G / 1000` times
    `1 - 0.004 (T_cell - 25)`. The AC output is that power clipped to the interval from zero to one
    megawatt.

    Args:
        poa_w_m2: The plane-of-array irradiance.
        air_c: The air temperature in degrees Celsius.
        wind_ms: The wind speed in metres per second.
        dc_ac_ratio: The DC rating divided by the AC capacity.

    Returns:
        The output in megawatts per megawatt of AC capacity.
    """
    cell = faiman_cell_temperature_c(poa_w_m2=poa_w_m2, air_c=air_c, wind_ms=wind_ms)
    derate = 1.0 + TEMPERATURE_COEFFICIENT_PER_K * (cell - STANDARD_CELL_TEMPERATURE_C)
    dc = dc_ac_ratio * poa_w_m2 / 1000.0 * derate
    return np.clip(dc, 0.0, 1.0)


def fleet_unit_output(
    *, sky: SunAndSky, air_c: np.ndarray, wind_ms: np.ndarray, fleet: FleetParams
) -> np.ndarray:
    """Return the output of one megawatt of AC capacity of a fleet, at each half-hour.

    Each orientation is clipped on its own, and the fleet's output is the shares-weighted sum. A
    half-hour with the sun at or below `MIN_ELEVATION_DEG` is exactly zero.

    Args:
        sky: The sun and sky of the half-hours.
        air_c: The air temperature at each half-hour's end, in degrees Celsius.
        wind_ms: The wind speed at each half-hour's end, in metres per second.
        fleet: The fleet's mix, tilt, and DC:AC ratio.

    Returns:
        The output in megawatts per megawatt of AC capacity, never negative.
    """
    out = np.zeros(len(sky.ghi_w_m2))
    for share, azimuth in zip(fleet.mix, FIXED_ORIENTATIONS, strict=True):
        poa = plane_of_array_w_m2(
            sky=sky, tilt_deg=fleet.tilt_deg, azimuth_deg=azimuth, tracker=False
        )
        out += share * derated_ac_unit_output(
            poa_w_m2=poa, air_c=air_c, wind_ms=wind_ms, dc_ac_ratio=fleet.dc_ac_ratio
        )
    sun_up = (90.0 - sky.zenith_deg) > MIN_ELEVATION_DEG
    return np.where(sun_up, out, 0.0)


def interpolate_hourly(
    *, hourly_times: np.ndarray, values: np.ndarray, at_times: np.ndarray
) -> np.ndarray:
    """Interpolate an hourly series linearly to other instants, NaN outside its range.

    A value at an instant that falls on an hour equals that hour's value, and a value half-way
    between two hours is their mean.

    Args:
        hourly_times: The hourly instants, `datetime64`, ascending.
        values: The value at each hourly instant.
        at_times: The instants to evaluate at, `datetime64`.

    Returns:
        The interpolated values.
    """
    seconds = hourly_times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    target = at_times.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    return np.interp(target, seconds, values, left=np.nan, right=np.nan)


def trailing_mean(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return the mean of each value and the `window - 1` before it (fewer at the start).

    Args:
        values: A finite series on a regular grid.
        window: The number of grid steps in the mean.

    Returns:
        The trailing mean, the same length as `values`.
    """
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    n = np.arange(1, len(values) + 1)
    lo = np.maximum(n - window, 0)
    return (csum[n] - csum[lo]) / (n - lo)
