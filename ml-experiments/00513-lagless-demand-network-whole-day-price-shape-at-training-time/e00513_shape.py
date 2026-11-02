# ruff: noqa: E501
"""Whole-day shape features of a half-hourly series: price, placebo, and national control.

The ten shape features describe one local settlement day (Europe/London midnight to midnight, so a
day has 46, 48, or 50 half-hours) using only that day's own values. Six vary per half-hour and four
are constant within the day.

Per half-hour: the rank within the day as a fraction in [0, 1] (`rank`), the value minus the day's
mean (`minus_mean`), the signed hours from the day's minimum (`hours_from_min`, negative before it),
the signed hours to the day's maximum (`hours_to_max`, negative after it), and indicators of the 8
lowest and the 8 highest half-hours of the day (`cheapest_4h`, `dearest_4h`).

Per day: the minimum, the maximum, the spread (maximum minus minimum), and the mean of the
16:00 to 19:00 local half-hours minus the mean of the 00:00 to 05:00 local half-hours
(`evening_minus_overnight`).

A day is "missing" when any of its half-hours has no usable value. A missing day has all ten shape
columns 0 and its missing flag 1. The one shared draw of the shuffled-day placebo is `placebo_draw`.
"""

from datetime import UTC, datetime
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

LONDON: Final[ZoneInfo] = ZoneInfo("Europe/London")
STEP_US: Final[int] = 1_800_000_000
FEATURES: Final[tuple[str, ...]] = (
    "rank", "minus_mean", "hours_from_min", "hours_to_max", "cheapest_4h", "dearest_4h",
    "day_min", "day_max", "day_spread", "evening_minus_overnight",
)  # fmt: skip
PER_DAY: Final[frozenset[str]] = frozenset(
    {"day_min", "day_max", "day_spread", "evening_minus_overnight"}
)
LEVEL_FEATURES: Final[tuple[str, ...]] = (
    "minus_mean", "day_min", "day_max", "day_spread", "evening_minus_overnight",
)  # fmt: skip
"""Features in the units of the series (standardised); the other five are on fixed scales."""
N_COLUMNS: Final[int] = len(FEATURES) + 1
"""The ten features plus the missing flag."""
WINDOW_HALF_HOURS: Final[int] = 8
"""The 4-hour charging and discharging windows, in half-hours."""


def local_day_of(*, times_us: np.ndarray) -> np.ndarray:
    """Return each UTC microsecond time's local settlement date as days since 1970-01-01."""
    ts = pl.Series(times_us.astype(np.int64)).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    local = ts.dt.convert_time_zone("Europe/London")
    return local.dt.date().cast(pl.Int32).to_numpy()


def day_features(*, values: np.ndarray, local_minutes: np.ndarray) -> np.ndarray:
    """Return the (n, 10) shape features of one complete day's values, in `FEATURES` order.

    Args:
        values: The day's finite values, one per half-hour in time order (46, 48, or 50).
        local_minutes: Each half-hour's local minutes after local midnight.

    Returns:
        The ten features for each half-hour.
    """
    n = len(values)
    order = np.argsort(values, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n)
    idx = np.arange(n)
    cheap = np.zeros(n)
    dear = np.zeros(n)
    cheap[order[:WINDOW_HALF_HOURS]] = 1.0
    dear[order[-WINDOW_HALF_HOURS:]] = 1.0
    lo, hi = float(values.min()), float(values.max())
    evening = (local_minutes >= 16 * 60) & (local_minutes < 19 * 60)
    overnight = local_minutes < 5 * 60
    spread_evening = float(values[evening].mean() - values[overnight].mean())
    out = np.empty((n, len(FEATURES)))
    out[:, 0] = rank / (n - 1)
    out[:, 1] = values - values.mean()
    out[:, 2] = (idx - int(np.argmin(values))) / 2.0
    out[:, 3] = (int(np.argmax(values)) - idx) / 2.0
    out[:, 4] = cheap
    out[:, 5] = dear
    out[:, 6] = lo
    out[:, 7] = hi
    out[:, 8] = hi - lo
    out[:, 9] = spread_evening
    return out


def grid_features(
    *, grid_start_us: int, grid_length: int, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shape columns on the dense grid, the day index of each half-hour, and the day table.

    Args:
        grid_start_us: UTC microseconds of grid index 0.
        grid_length: The number of half-hours on the grid.
        values: One value per grid half-hour, NaN where there is no usable value.

    Returns:
        `(columns, day_id, missing_days)`: a float32 (grid_length, 11) array (ten features, then the
        missing flag), the integer id of each half-hour's local day, and a boolean per day id that is
        True when the day is missing.
    """
    times = grid_start_us + np.arange(grid_length, dtype=np.int64) * STEP_US
    day = local_day_of(times_us=times)
    _, day_id = np.unique(day, return_inverse=True)
    n_days = int(day_id.max()) + 1
    local = (
        pl.Series(times).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )  # fmt: skip
    minutes = local.dt.hour().to_numpy().astype(
        np.int64
    ) * 60 + local.dt.minute().to_numpy().astype(np.int64)
    columns = np.zeros((grid_length, N_COLUMNS), dtype=np.float32)
    missing = np.zeros(n_days, dtype=bool)
    starts = np.searchsorted(day_id, np.arange(n_days), side="left")
    stops = np.searchsorted(day_id, np.arange(n_days), side="right")
    for d in range(n_days):
        a, b = int(starts[d]), int(stops[d])
        v = values[a:b]
        if len(v) not in (46, 48, 50) or not np.isfinite(v).all():
            missing[d] = True
            columns[a:b, -1] = 1.0
            continue
        columns[a:b, : len(FEATURES)] = day_features(values=v, local_minutes=minutes[a:b])
    return columns, day_id, missing


def standardise_levels(*, columns: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Return a copy with the level features z-scored on the non-missing `fit_rows`.

    Rank is already in [0, 1], hours are divided by 12, and indicators stay 0 or 1; every level
    feature is divided by the standard deviation of its own values in `fit_rows` after subtracting
    the mean. Missing-day rows stay 0 in all ten features.
    """
    out = columns.copy()
    ok = fit_rows & (columns[:, -1] == 0)
    for name in LEVEL_FEATURES:
        j = FEATURES.index(name)
        mean = float(columns[ok, j].mean())
        sd = max(float(columns[ok, j].std()), 1e-6)
        live = columns[:, -1] == 0
        out[live, j] = (columns[live, j] - mean) / sd
    out[:, 2] /= 12.0
    out[:, 3] /= 12.0
    return out


def placebo_draw(
    *, day_id: np.ndarray, missing: np.ndarray, first_local_day: np.ndarray, seed: int = 513
) -> tuple[np.ndarray, int]:
    """Return, per day id, the day id whose shape it carries under the shuffled-day placebo, or -1.

    The donor shares the day's calendar month, day type (Monday to Friday, Saturday, Sunday), and
    length in half-hours. Days are shuffled once with a seeded generator inside each group and each
    day takes the next day in that order, so a donor is never the day itself and no donor is used
    twice. A group of one day, and every missing day, gets -1 (placebo missing).

    Args:
        day_id: The day id of every grid half-hour.
        missing: True per day id for a day with no usable value.
        first_local_day: For each day id, the local date as days since 1970-01-01.
        seed: The generator seed.

    Returns:
        The donor day id of each day id (-1 for none), and the number of non-missing days with -1.
    """
    n_days = len(missing)
    length = np.bincount(day_id, minlength=n_days)
    dates = first_local_day.astype("datetime64[D]")
    month = dates.astype("datetime64[M]").astype(int) % 12
    weekday = (first_local_day + 3) % 7  # 1970-01-01 is a Thursday; Monday = 0
    kind = np.where(weekday < 5, 0, weekday - 4)
    donor = np.full(n_days, -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    keys = {}
    for d in range(n_days):
        if not missing[d]:
            keys.setdefault((int(month[d]), int(kind[d]), int(length[d])), []).append(d)
    for key in sorted(keys):
        members = np.array(keys[key])
        if len(members) < 2:
            continue
        shuffled = rng.permutation(members)
        donor[shuffled] = np.roll(shuffled, -1)
    unmatched = int(((donor < 0) & ~missing).sum())
    return donor, unmatched


def placebo_columns(*, columns: np.ndarray, day_id: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Return the shape columns with each day replaced by its donor day's, half-hour for half-hour."""
    out = np.zeros_like(columns)
    out[:, -1] = 1.0
    starts = np.searchsorted(day_id, np.arange(len(donor)), side="left")
    stops = np.searchsorted(day_id, np.arange(len(donor)), side="right")
    for d in range(len(donor)):
        a, b = int(starts[d]), int(stops[d])
        k = int(donor[d])
        if k < 0:
            continue
        out[a:b] = columns[int(starts[k]) : int(stops[k])]
    return out


def utc_us(*, t: datetime) -> int:
    """Return a datetime as UTC microseconds."""
    return int(t.astimezone(UTC).timestamp() * 1_000_000)
