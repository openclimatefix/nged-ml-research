"""Experiment 709, arm L-stat: window statistics of raw power and their year-on-year changes.

L-stat reads 12 statistics of a Primary's own raw power (a share of capacity) over a trailing
window of 28 days that ends on day `d`. Every value stored at index `d` uses readings up to the end
of day `d` only, so the signal at the origin of day `D` is the entry of day `D - 1`.

**The 12 statistics** (`STAT_NAMES`) are five level statistics (the 5th, 50th, and 95th
percentile of all half-hours in the window, the night mean, and the midday mean), three
profile-shape statistics (the morning, midday, and evening band mean minus the window mean), and
four residual-shape statistics (the 5th percentile, the 95th percentile, and the interquartile
range of the residual around the window's own half-hour median profile, and the weekday mean minus
the weekend mean of the day-mean residual). A window needs at least 60% of its days valid (24
readings or more per day).

**The seasonal reference** is the same 28-day change one year earlier. The change is
`c(t) = x(t) - x(t - 28)` and the year-on-year change is `y(t) = c(t) - c(t - 364)`. The reference
skips a day that an earlier alarm episode has contaminated (`e00709_statdetect`), and the scale of
`y` is `robust_scale`.

**The daily analogue** of each statistic (`WindowStats.daily`) is the same statistic of one day's
readings. It is used only to place the onset of an alarm by a least-squares split.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Final

import e00709_features as features
import numpy as np

STAT_NAMES: Final[tuple[str, ...]] = (
    "p05", "p50", "p95", "night", "midday",
    "morning_shape", "midday_shape", "evening_shape",
    "residual_p05", "residual_p95", "residual_iqr", "weekday_minus_weekend",
)  # fmt: skip
N_STATS: Final[int] = len(STAT_NAMES)
P05: Final[int] = 0
P50: Final[int] = 1
NIGHT: Final[int] = 3
MIDDAY: Final[int] = 4
N_LEVEL_STATS: Final[int] = 5
"""The first five statistics are level statistics (the onset is split on one of them)."""
RESIDUAL_SHAPE: Final[slice] = slice(8, 11)
"""The three residual-shape statistics that the shape evidence reads."""
WINDOW_DAYS: Final[int] = 28
MIN_VALID_SHARE: Final[float] = 0.6
MIN_VALID_DAYS: Final[int] = math.ceil(MIN_VALID_SHARE * WINDOW_DAYS)
YEAR_DAYS: Final[int] = 364
SCALE_MIN_VALUES: Final[int] = 365
SCALE_FLOOR: Final[float] = 0.005
"""Smallest scale of a year-on-year change, 0.5% of capacity."""
SUNDAY_OFFSET: Final[int] = 6
"""2019-09-01, day 0 of the grid, is a Sunday (`date.weekday()` is 6)."""


@dataclass(frozen=True)
class WindowStats:
    """The statistics of one Primary on every day.

    Attributes:
        stat: Window statistics at the end of each day, shape (days, 12), NaN where the window has
            fewer than 17 valid days.
        daily: The daily analogue of each statistic, shape (days, 12).
        valid_day: True on days with at least 24 readings, shape (days,).
    """

    stat: np.ndarray
    daily: np.ndarray
    valid_day: np.ndarray


def residual_shape(*, block: np.ndarray) -> tuple[float, float, float]:
    """Return the 5th and 95th percentile and the interquartile range of the residual of a block.

    The residual is each reading minus the block's own half-hour median profile.

    Args:
        block: Readings shaped (days, 48), NaN where missing.

    Returns:
        The three statistics, NaN where the block holds no reading.
    """
    if not np.isfinite(block).any():
        return math.nan, math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(block, axis=0)
    residual = block - profile[None, :]
    values = residual[np.isfinite(residual)]
    low, q25, q75, high = np.percentile(values, [5, 25, 75, 95])
    return float(low), float(high), float(q75 - q25)


def _daily_grid(*, power_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the power as (days, 48) with invalid days set to NaN, and the valid-day flags."""
    n_days = power_site.size // features.HALF_HOURS_PER_DAY
    grid = power_site.reshape(n_days, features.HALF_HOURS_PER_DAY)
    valid = np.isfinite(grid).sum(axis=1) >= features.MIN_READINGS_PER_DAY
    return np.where(valid[:, None], grid, np.nan), valid


def window_statistics(*, power_site: np.ndarray) -> WindowStats:
    """Compute the window statistics and their daily analogues for one Primary.

    A window needs 17 valid days, but its last day may be invalid. The daily analogue needs the
    day itself valid.

    Args:
        power_site: One Primary's power over capacity, a whole number of days of half-hours.

    Returns:
        The statistics on every day.
    """
    grid, valid = _daily_grid(power_site=power_site)
    n_days = grid.shape[0]
    weekend = ((np.arange(n_days) + SUNDAY_OFFSET) % 7) >= 5
    night, morning, midday, evening = features.BAND_HALF_HOURS
    stat = np.full((n_days, N_STATS), np.nan)
    daily = np.full((n_days, N_STATS), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for day in range(WINDOW_DAYS - 1, n_days):
            first = day - WINDOW_DAYS + 1
            if valid[first : day + 1].sum() < MIN_VALID_DAYS:
                continue
            block = grid[first : day + 1]
            stat[day] = _window_row(block=block, weekend=weekend[first : day + 1])
            if not valid[day]:
                continue
            daily[day] = _daily_row(
                x=grid[day], profile=np.nanmedian(block, axis=0), weekend=bool(weekend[day]),
                bands=(night, morning, midday, evening),
            )  # fmt: skip
    return WindowStats(stat=stat, daily=daily, valid_day=valid)


def _window_row(*, block: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Return the 12 statistics of one 28-day block of readings."""
    night, morning, midday, evening = features.BAND_HALF_HOURS
    values = block[np.isfinite(block)]
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    mean = float(values.mean())
    bands = [float(np.nanmean(block[:, b])) for b in (night, morning, midday, evening)]
    low, high, iqr = residual_shape(block=block)
    profile = np.nanmedian(block, axis=0)
    day_residual = np.nanmean(block - profile[None, :], axis=1)
    finite = np.isfinite(day_residual)
    weekday_part, weekend_part = day_residual[finite & ~weekend], day_residual[finite & weekend]
    gap = (
        float(weekday_part.mean() - weekend_part.mean())
        if weekday_part.size and weekend_part.size
        else math.nan
    )
    return np.array(
        [
            p05, p50, p95, bands[0], bands[2],
            bands[1] - mean, bands[2] - mean, bands[3] - mean,
            low, high, iqr, gap,
        ]
    )  # fmt: skip


def _daily_row(
    *, x: np.ndarray, profile: np.ndarray, weekend: bool, bands: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Return the daily analogue of the 12 statistics for one day of 48 readings."""
    values = x[np.isfinite(x)]
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    mean = float(values.mean())
    means = [float(np.nanmean(x[b])) for b in bands]
    residual = x - profile
    finite = residual[np.isfinite(residual)]
    if finite.size:
        low, q25, q75, high = np.percentile(finite, [5, 25, 75, 95])
        iqr, day_mean = q75 - q25, float(finite.mean())
    else:
        low = high = iqr = day_mean = math.nan
    return np.array(
        [
            p05, p50, p95, means[0], means[2],
            means[1] - mean, means[2] - mean, means[3] - mean,
            low, high, iqr, math.nan if weekend else day_mean,
        ]
    )  # fmt: skip


def change_statistics(*, stat: np.ndarray) -> np.ndarray:
    """Return the 28-day change `c(t) = x(t) - x(t - 28)` of every statistic.

    Args:
        stat: Window statistics, shape (days, 12).

    Returns:
        The change, NaN where either window is missing.
    """
    out = np.full_like(stat, np.nan)
    out[WINDOW_DAYS:] = stat[WINDOW_DAYS:] - stat[:-WINDOW_DAYS]
    return out


def robust_scale(*, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the median and the robust scale of each statistic's year-on-year change.

    Args:
        values: Year-on-year changes outside alarm episodes, shape (days, 12), NaN where missing.

    Returns:
        The median, and 1.4826 times the median absolute deviation floored at `SCALE_FLOOR`. Both
        are NaN for a statistic with fewer than `SCALE_MIN_VALUES` finite values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median[None, :]), axis=0)
    enough = np.isfinite(values).sum(axis=0) >= SCALE_MIN_VALUES
    scale = np.maximum(1.4826 * mad, SCALE_FLOOR)
    return np.where(enough, median, np.nan), np.where(enough, scale, np.nan)


def yearly_difference(*, series: np.ndarray) -> np.ndarray:
    """Return `series(t) - series(t - 364)`, NaN where either is missing."""
    out = np.full_like(series, np.nan)
    out[YEAR_DAYS:] = series[YEAR_DAYS:] - series[:-YEAR_DAYS]
    return out


def fit_year_on_year_ramp(
    *, windows: WindowStats, first: int, last: int
) -> tuple[int, float] | None:
    """Fit the onset of a step in the year-on-year midday-minus-night level.

    A step on day `o` moves a 28-day window linearly over its next 28 days, so the series is fitted
    as `m + s * clip((t - o + 1) / 28, 0, 1)` by least squares for every candidate `o`.

    Args:
        windows: The Primary's window statistics.
        first: First day of the fitting range.
        last: Last day of the fitting range.

    Returns:
        The onset day and the size `s` of the step (negative for a fall), or None if fewer than 60
        days are finite.
    """
    level = windows.stat[:, MIDDAY] - windows.stat[:, NIGHT]
    y = yearly_difference(series=level)
    days = np.arange(first, last + 1)
    ok = np.isfinite(y[days])
    days, y = days[ok], y[days[ok]]
    if days.size < 60:
        return None
    best: tuple[float, int, float] | None = None
    for onset in range(int(days[0]) + 7, int(days[-1]) - 7):
        ramp = np.clip((days - onset + 1) / WINDOW_DAYS, 0.0, 1.0)
        design = np.stack([np.ones_like(ramp), ramp], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        cost = float(((y - design @ coefficients) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, onset, float(coefficients[1]))
    return None if best is None else (best[1], best[2])
