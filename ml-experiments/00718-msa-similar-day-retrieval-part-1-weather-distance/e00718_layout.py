# ruff: noqa: E501
"""Experiment 718: the arms' row counts, the age buckets, and the age classes (design sections 2 and 3).

A block is a 24-hour row of 48 half-hours that starts at 12:00 UTC, and its age is the number of
days from its start to the contract origin. Experiment 713 indexes three parameter tables by the
slot of a history row. This experiment indexes them by an age bucket: ages 2 to 15 keep one bucket
each (buckets 0 to 13), and ten further buckets (14 to 23) cover older ages. The placebos draw from
age classes, which are the single class of ages 2 to 15 and each older bucket.

This module is pure index arithmetic, with no data access and no torch.
"""

from typing import Final

import numpy as np

MIN_AGE: Final = 2
"""The age of the newest history row, as in experiment 713."""
NEWEST_FIXED_MAX_AGE: Final = 15
"""The oldest age of the 14 fixed rows of experiment 713 (ages 2 to 15)."""
N_FIXED_ROWS: Final = 14
SHORT_POOL_MAX_AGE: Final = 91
"""The oldest age of the 90-day pool of arms `R`, `R-A`, and `P90`."""
LONG_POOL_MAX_AGE: Final = 379
"""The oldest age of the 378-day pool of arms `U`, `U-A`, `U27`, and `U27-A`."""
N_BUCKETS: Final = 24
"""Age buckets: 14 single ages (2 to 15) and 10 older ranges."""
N_KEY_ROWS: Final = N_BUCKETS + 1
"""Rows of the across-day key bias: the target row, then one per bucket."""
OLD_BUCKET_LOWER_BOUNDS: Final = (16, 23, 30, 44, 64, 92, 183, 344, 387, 752)
"""The lowest age of each of the ten buckets from 14 to 23 (section 3.2)."""
N_OLD_BUCKETS: Final = len(OLD_BUCKET_LOWER_BOUNDS)
STEPS_PER_DAY: Final = 48
N_LEAD_DAYS: Final = 14

ROWS_PER_ARM: Final = {
    "Z": 14,
    "R": 14,
    "R-A": 14,
    "P90": 14,
    "U": 14,
    "U-A": 14,
    "UX": 14,
    "U27": 27,
    "U27-A": 27,
}
"""History rows per arm."""
MEASURE_WEATHER: Final = 0
MEASURE_RECENCY: Final = 1
MEASURE_WEEKDAY: Final = 2
MEASURE_FIXED: Final = 3
MEASURE_CALENDAR: Final = 4
MEASURE_RANDOM: Final = 5
MEASURE_NAMES: Final = ("weather", "recency", "weekday", "fixed", "calendar", "random")
"""What chose a row: the measure codes of the `measure` array of a row table."""
ABSENT: Final = -1
"""The age of a row that no eligible block fills."""


def age_bucket(*, ages: np.ndarray) -> np.ndarray:
    """Return the bucket (0 to 23) of every age; an absent row (age -1) takes bucket 0.

    Ages 2 to 15 map to buckets 0 to 13, and an age of 16 or more to 14 plus the index of the
    last lower bound at or below it.
    """
    a = np.asarray(ages, dtype=np.int64)
    old = np.searchsorted(np.array(OLD_BUCKET_LOWER_BOUNDS), a, side="right") - 1
    out = np.where(a < NEWEST_FIXED_MAX_AGE + 1, a - MIN_AGE, N_FIXED_ROWS + old)
    return np.where(a < 0, 0, out).astype(np.int64)


def age_class(*, ages: np.ndarray) -> np.ndarray:
    """Return the placebo age class of every age: 0 for ages 2 to 15, and 1 to 10 for the older buckets."""
    a = np.asarray(ages, dtype=np.int64)
    return np.where(a < NEWEST_FIXED_MAX_AGE + 1, 0, age_bucket(ages=a) - N_FIXED_ROWS + 1)


def class_bounds(*, max_age: int) -> list[tuple[int, int]]:
    """Return the (lowest age, highest age) of each age class up to `max_age`, youngest first."""
    lows = [MIN_AGE, *OLD_BUCKET_LOWER_BOUNDS]
    highs = [NEWEST_FIXED_MAX_AGE, *(b - 1 for b in OLD_BUCKET_LOWER_BOUNDS[1:]), 10**9]
    return [(lo, min(hi, max_age)) for lo, hi in zip(lows, highs, strict=True) if lo <= max_age]


def block_index(*, contract_idx: np.ndarray | int, ages: np.ndarray) -> np.ndarray:
    """Return the grid indices (..., 48) of the blocks of `ages` before the contract origins (`c - 48 a + j`)."""
    c = np.asarray(contract_idx, dtype=np.int64)
    a = np.asarray(ages, dtype=np.int64)
    return (c - STEPS_PER_DAY * a)[..., None] + np.arange(STEPS_PER_DAY)


def contract_idx_of_day(*, day: np.ndarray | int) -> np.ndarray:
    """Return the grid index of 12:00 UTC on day number `day` (the grid index of 00:00 divided by 48), `48 day + 24`."""
    return np.asarray(day, dtype=np.int64) * STEPS_PER_DAY + 24
