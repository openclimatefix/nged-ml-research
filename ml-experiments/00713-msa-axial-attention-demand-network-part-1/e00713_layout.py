# ruff: noqa: E501
"""Experiment 713: the days-by-half-hours matrix, as pure index arithmetic (no data access).

A sample is one Primary, one contract origin (12:00 UTC on day D), and one lead day `k` (1 to 14).
Its matrix has 15 rows of 48 columns. Row 0 is the target row. Rows 1 to 14 are the history rows of
ages 2 to 15, newest first. Every row is a 24-hour block that starts at 12:00 UTC, so column `j` of
every row has the same time of day: column 0 is the half-hour stamped 12:00 UTC.

A half-hour is stamped with its period end on the grid of experiment 422 (`GRID_START_US`,
`STEP_US`), and the grid index of a stamp is `(stamp - GRID_START) / STEP`. The native origin of the
Candidate is the contract origin minus 3 hours (6 grid steps), and a power reading is visible only if
its grid index is at most `native_origin_idx - 48`, which is 27 hours before the contract origin.
"""

from collections.abc import Collection
from datetime import UTC, date, datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np

STEPS_PER_DAY: Final = 48
"""Half-hours per row and per day."""
N_HISTORY_ROWS: Final = 14
MIN_AGE: Final = 2
"""The age of the newest history row, in days: the 24-hour block of age 1 holds no usable power."""
MAX_AGE: Final = MIN_AGE + N_HISTORY_ROWS - 1
"""The age of the oldest history row (15)."""
N_LEAD_DAYS: Final = 14
NATIVE_OFFSET_STEPS: Final = 6
"""Grid steps from the native origin (09:00 UTC) to the contract origin (12:00 UTC)."""
POWER_CAP_STEPS: Final = 48
"""A reading is visible only this many steps before the native origin or earlier."""
NEWEST_VISIBLE_BEFORE_CONTRACT: Final = NATIVE_OFFSET_STEPS + POWER_CAP_STEPS
"""54 steps: the newest visible reading is stamped 27 hours before the contract origin."""
PLACEBO_MIN_AGE: Final = 16
PLACEBO_MAX_AGE: Final = 379
PLACEBO_SEED: Final = 0
HOLIDAY_DAY_TYPE: Final = 7
"""The day type of a token whose local day is a bank holiday in England."""
LOCAL_ZONE: Final = ZoneInfo("Europe/London")
DAY_US: Final = 86_400 * 1_000_000
HALF_HOUR_US: Final = 1_800_000_000


def england_bank_holidays(*, years: Collection[int]) -> frozenset[date]:
    """Return the bank holidays of England in the given years, from the `holidays` package.

    The Candidate's date-time columns read the same source (`e00087_common.holiday_gaps`).
    """
    import holidays

    return frozenset(holidays.country_holidays("GB", subdiv="ENG", years=sorted(years)))


def contract_index(*, day_start_us: np.ndarray | int, grid_start_us: int) -> np.ndarray:
    """Return the grid index of 12:00 UTC on the day that starts at `day_start_us` (00:00 UTC)."""
    stamp = np.asarray(day_start_us, dtype=np.int64) + 12 * 3_600_000_000
    return ((stamp - grid_start_us) // HALF_HOUR_US).astype(np.int64)


def columns() -> np.ndarray:
    """Return the column numbers 0 to 47."""
    return np.arange(STEPS_PER_DAY, dtype=np.int64)


def target_grid_index(*, contract_idx: np.ndarray, lead_day: np.ndarray) -> np.ndarray:
    """Return the (N, 48) grid indices of the target row: `c + 48 (k - 1) + j` with `lead_day = k - 1`."""
    c = np.asarray(contract_idx, dtype=np.int64)[:, None]
    k = np.asarray(lead_day, dtype=np.int64)[:, None]
    return c + STEPS_PER_DAY * k + columns()[None, :]


def history_ages() -> np.ndarray:
    """Return the ages 2 to 15 of the 14 history rows of arm `M*`, newest first."""
    return np.arange(MIN_AGE, MAX_AGE + 1, dtype=np.int64)


def history_grid_index(*, contract_idx: np.ndarray, ages: np.ndarray | None = None) -> np.ndarray:
    """Return the (N, 14, 48) grid indices of the history rows: `c - 48 a + j`.

    `ages` is (14,) for every sample, or (N, 14) for the placebo.
    """
    c = np.asarray(contract_idx, dtype=np.int64)[:, None, None]
    a = history_ages() if ages is None else np.asarray(ages, dtype=np.int64)
    a = np.broadcast_to(a, (len(c), N_HISTORY_ROWS))[:, :, None]
    return c - STEPS_PER_DAY * a + columns()[None, None, :]


def newest_visible_index(*, contract_idx: np.ndarray) -> np.ndarray:
    """Return, per sample, the newest grid index at which power may be read (`c - 54`)."""
    return np.asarray(contract_idx, dtype=np.int64) - NEWEST_VISIBLE_BEFORE_CONTRACT


def visible_by_time(*, contract_idx: np.ndarray, grid_index: np.ndarray) -> np.ndarray:
    """Return which grid indices obey the 27-hour rule, shape of `grid_index` (N, ...)."""
    limit = newest_visible_index(contract_idx=contract_idx)
    limit = limit.reshape((-1,) + (1,) * (grid_index.ndim - 1))
    return grid_index <= limit


def _local_seconds(*, grid_start_us: int, n_grid: int) -> np.ndarray:
    """Return the local (Europe/London) clock time in seconds since 1970 of each stamp minus 30 minutes."""
    first_hour = grid_start_us - HALF_HOUR_US
    n_hours = n_grid // 2 + 2
    offsets = np.empty(n_hours, dtype=np.int64)
    for h in range(n_hours):
        t = datetime.fromtimestamp((first_hour + h * 3_600_000_000) / 1e6, UTC)
        offsets[h] = int(t.astimezone(LOCAL_ZONE).utcoffset().total_seconds())  # ty: ignore[unresolved-attribute]
    start = np.arange(n_grid, dtype=np.int64) * HALF_HOUR_US + first_hour
    hour_of = (start - first_hour) // 3_600_000_000
    return start // 1_000_000 + offsets[hour_of]


def local_half_hours_for_grid(*, grid_start_us: int, n_grid: int) -> np.ndarray:
    """Return the local half-hour of the day (0 to 47) of every grid index, at the stamp minus 30 minutes."""
    local_s = _local_seconds(grid_start_us=grid_start_us, n_grid=n_grid)
    return ((local_s % 86_400) // 1_800).astype(np.uint8)


def day_types_for_grid(
    *, grid_start_us: int, n_grid: int, holidays: Collection[date]
) -> np.ndarray:
    """Return the day type (uint8) of every grid index.

    The day type of a token is the local weekday (0 to 6, Europe/London) of the token's own
    half-hour, taken at the stamp minus 30 minutes, or 7 where that local day is a bank holiday in
    England. A table over the whole grid makes a lookup of any sample an index operation.
    """
    local_day = _local_seconds(grid_start_us=grid_start_us, n_grid=n_grid) // 86_400
    weekday = ((local_day + 3) % 7).astype(np.uint8)  # 1970-01-01 was a Thursday (weekday 3)
    holiday_days = np.array(sorted((d - date(1970, 1, 1)).days for d in holidays), dtype=np.int64)
    is_holiday = np.isin(local_day, holiday_days)
    return np.where(is_holiday, np.uint8(HOLIDAY_DAY_TYPE), weekday).astype(np.uint8)


def day_type_reference(*, grid_index: int, grid_start_us: int, holidays: Collection[date]) -> int:
    """Return the day type of one grid index with `datetime` and `zoneinfo` only (the KA7 reference)."""
    stamp = datetime.fromtimestamp((grid_start_us + grid_index * HALF_HOUR_US) / 1e6, UTC)
    local = (stamp - timedelta(minutes=30)).astimezone(LOCAL_ZONE)
    return HOLIDAY_DAY_TYPE if local.date() in holidays else local.weekday()


def masked_columns_of_newest_row() -> np.ndarray:
    """Return the columns of the age-2 row that are not visible (43 to 47)."""
    idx = history_grid_index(contract_idx=np.zeros(1, dtype=np.int64))[0, 0]
    return np.nonzero(
        ~visible_by_time(contract_idx=np.zeros(1, dtype=np.int64), grid_index=idx[None])[0]
    )[0]


def placebo_ages(
    *, primary: int, origin_day: int, available: np.ndarray | None = None
) -> np.ndarray:
    """Return the 14 placebo ages (ascending) of one sample, drawn from ages 16 to 379.

    The draw is a fixed function of (placebo seed, Primary, origin day number), so it is the same in
    training and at prediction and for every network seed. `available` is an optional boolean array
    over ages 16 to 379 (length 364) that marks blocks holding any reading; the draw is then from the
    available ages only. Fewer than 14 available ages return fewer than 14 ages; the caller marks the
    remaining rows as absent.
    """
    pool = np.arange(PLACEBO_MIN_AGE, PLACEBO_MAX_AGE + 1, dtype=np.int64)
    if available is not None:
        assert available.shape == pool.shape, available.shape
        pool = pool[available]
    rng = np.random.default_rng([PLACEBO_SEED, int(primary), int(origin_day)])
    n = min(N_HISTORY_ROWS, len(pool))
    return np.sort(rng.choice(pool, size=n, replace=False))
