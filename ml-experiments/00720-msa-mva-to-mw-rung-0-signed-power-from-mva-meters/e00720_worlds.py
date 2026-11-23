# ruff: noqa: E501
"""Experiment 720: injected readings, and synthetic worlds with a known sign.

All arrays are `(n_days, 48)`: one row per UTC day and one column per half-hour. Power is in points
of capacity (1 point is 1% of a Primary's capacity). The reading of an MVA meter is the half-hour
mean of `sqrt(P'² + Q²)`, where `P'` is the signed power inside the half-hour. A crossing half-hour
therefore reads above `|mean P|`, as the design's section 2.5 requires.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

STEPS_PER_DAY: Final[int] = 48
N_SUBSTEPS: Final[int] = 6
"""Sub-steps per half-hour in the within-half-hour interpolation of the injected reading."""


def reading_from_signed(*, x: np.ndarray, q: float, n_sub: int = N_SUBSTEPS) -> np.ndarray:
    """Return the reading of each half-hour, averaging `sqrt(P'² + Q²)` over `n_sub` sub-steps.

    `x` holds the half-hour means of signed power on a flat time axis (any shape; the last axis is
    time and is interpolated, so a `(n_days, 48)` array is read as one run of `n_days × 48` steps).
    `P'` is the linear interpolation through the half-hour means placed at the half-hour centres, and
    the sub-steps are the midpoints of `n_sub` equal parts of the half-hour. Where `x` is NaN the
    reading is NaN.
    """
    flat = x.reshape(-1)
    n = flat.size
    centres = np.arange(n, dtype=np.float64)
    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5  # within one half-hour, in half-hours
    acc = np.zeros(n)
    good = np.isfinite(flat)
    fill = np.where(good, flat, 0.0)
    for off in offsets:
        p = np.interp(centres + off, centres, fill)
        acc += np.sqrt(p**2 + q**2)
    out = acc / n_sub
    out = np.where(good, out, np.nan)
    return out.reshape(x.shape)


def reading_from_mean(*, x: np.ndarray, q: float) -> np.ndarray:
    """Return `sqrt(x² + Q²)`, the instantaneous reading that experiments 412 to 420 used."""
    return np.sqrt(x**2 + q**2)


def shift_days(*, a: np.ndarray, days: int) -> np.ndarray:
    """Return `a` from `days` days earlier (the sky-shifted placebo); the first rows are NaN."""
    out = np.full_like(a, np.nan, dtype=np.float64)
    out[days:] = a[:-days]
    return out


def clear_sky_shape(*, doy: np.ndarray, n_steps: int = STEPS_PER_DAY) -> np.ndarray:
    """Return a `(n_days, 48)` clear-sky yield between 0 and 1: longer and higher in summer."""
    h = (np.arange(n_steps) + 0.5) / n_steps * 24.0
    season = np.cos(2.0 * np.pi * (doy - 172.0) / 365.0)  # +1 at midsummer
    half_day = 4.5 + 3.5 * (season + 1.0) / 2.0  # hours either side of 12 UTC
    amp = 0.45 + 0.55 * (season + 1.0) / 2.0
    x = (h[None, :] - 12.0) / half_day[:, None]
    shape = np.where(np.abs(x) < 1.0, np.cos(0.5 * np.pi * np.clip(x, -1.0, 1.0)) ** 1.3, 0.0)
    return amp[:, None] * shape


@dataclass(frozen=True)
class SyntheticWorld:
    """A synthetic Primary with a known signed power.

    Attributes:
        x: Signed power in points of capacity, `(n_days, 48)`.
        reading: The folded reading (within-half-hour averaged), same shape.
        u: The yield curve in 0 to 1 (clear sky times the day's cloud factor), same shape.
        p: The PV estimate in points (`c_est × u`), same shape.
        temperature: Daily temperature in degrees Celsius, `(n_days,)`.
        day_type: Daily type 0 (weekday), 1 (Saturday), 2 (Sunday), `(n_days,)`.
        doy: Day of year, `(n_days,)`.
        export_share: Share of days with at least one reverse-flow half-hour.
    """

    x: np.ndarray
    reading: np.ndarray
    u: np.ndarray
    p: np.ndarray
    temperature: np.ndarray
    day_type: np.ndarray
    doy: np.ndarray
    export_share: float


def synthetic_world(
    *,
    n_days: int,
    capacity: float,
    seed: int,
    q: float = 2.0,
    estimate_scale: float = 1.0,
    sunny_share_summer: float = 0.8,
    start_doy: int = 1,
    pv_fraction: float = 1.0,
) -> SyntheticWorld:
    """Return a synthetic Primary: `X = D − pv_fraction × capacity × u`, read through the folded meter.

    Gross demand `D` depends on half-hour, day type and temperature only. Each day has a cloud factor
    that is near 1 on a sunny day and near 0.15 on a dull day, and the probability of a sunny day
    follows the season (`sunny_share_summer` at midsummer, a tenth of that at midwinter). The
    estimate `p = estimate_scale × pv_fraction × capacity × u` stands in for F7's PV estimate.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(n_days)
    doy = (start_doy - 1 + day) % 365 + 1
    day_type = np.where(day % 7 == 5, 1, np.where(day % 7 == 6, 2, 0))
    season = np.cos(2.0 * np.pi * (doy - 172.0) / 365.0)
    temp = 10.0 - 8.0 * season + rng.normal(0.0, 2.5, n_days)
    h = (np.arange(STEPS_PER_DAY) + 0.5) / STEPS_PER_DAY * 24.0
    shape = (
        0.55
        + 0.25 * np.exp(-0.5 * ((h - 9.0) / 2.5) ** 2)
        + 0.35 * np.exp(-0.5 * ((h - 18.5) / 2.5) ** 2)
    )
    type_scale = np.array([1.0, 0.85, 0.75])[day_type]
    demand = (
        capacity
        * 0.35
        * shape[None, :]
        * type_scale[:, None]
        * (1.0 + 0.012 * (10.0 - temp))[:, None]
    )
    resid = np.zeros((n_days, STEPS_PER_DAY))
    state = np.zeros(n_days)
    for t in range(STEPS_PER_DAY):
        state = 0.9 * state + rng.normal(0.0, 0.01 * capacity, n_days)
        resid[:, t] = state
    demand = demand + resid
    p_sunny = sunny_share_summer * (0.1 + 0.9 * (season + 1.0) / 2.0)
    sunny = rng.random(n_days) < p_sunny
    k_day = np.where(sunny, rng.uniform(0.85, 1.0, n_days), rng.uniform(0.05, 0.25, n_days))
    wiggle = np.clip(1.0 + 0.04 * rng.normal(size=(n_days, STEPS_PER_DAY)), 0.8, 1.2)
    u = np.clip(clear_sky_shape(doy=doy) * k_day[:, None] * wiggle, 0.0, 1.0)
    pv_capacity = pv_fraction * capacity
    x = demand - pv_capacity * u
    reading = reading_from_signed(x=x, q=q)
    p = estimate_scale * pv_capacity * u
    export_share = float(np.mean((x < 0.0).any(axis=1)))
    return SyntheticWorld(x, reading, u, p, temp, day_type, doy, export_share)
