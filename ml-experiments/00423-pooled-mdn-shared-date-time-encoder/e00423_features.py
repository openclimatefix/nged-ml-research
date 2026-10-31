"""Calendar indices and row sets of experiment 423.

The shared calendar encoder reads five integer indices per row, all computed from the valid time
alone (never from the observed power), so the training rows and the forecast rows use the same
function:

- `half_hour`: the local half-hour of the day, 0 to 47, from the valid time (the stamp at the end
  of the half-hour) in Europe/London, with no 30-minute shift. This is the convention of the
  hand-made columns of experiment 87, so only the encoding differs from R. The two days a year
  with a clock change share the indices of the other days. The row sets below take the local date
  of `valid_time - 30 minutes` instead, as `e00396_arms.add_calendar` does;
- `weekday`: Monday 0 to Sunday 6;
- `day_of_month`: 0 to 30;
- `day_of_year`: the ordinal day minus one, 0 to 365 (so 1 March has index 59 in a common year and
  60 in a leap year);
- `holiday_name`: one of the 12 categories of `HOLIDAY_NAMES`, from the `holidays` package's England
  calendar, the calendar experiment 87 uses for the hand-made holiday columns.

The row sets of the plan are boolean columns of the local date of the valid time:
`bank_holiday_day`, `bank_holiday_week` (Monday to Sunday), `christmas_set` (24 December to 1
January), and `easter_set` (Monday of Holy Week to Easter Monday).
"""

from datetime import date, timedelta
from functools import cache
from typing import Final

import holidays
import numpy as np
import polars as pl

HALF_HOURS_PER_DAY: Final[int] = 48
DAYS_PER_WEEK: Final[int] = 7
MAX_DAYS_IN_MONTH: Final[int] = 31
MAX_DAYS_IN_YEAR: Final[int] = 366
HOLIDAY_NAMES: Final[tuple[str, ...]] = (
    "none",
    "New Year's Day",
    "Good Friday",
    "Easter Monday",
    "Early May",
    "Spring",
    "Summer",
    "Christmas Day",
    "Boxing Day",
    "other bank holiday",
    "substitute day",
    "24 to 31 December, not a holiday",
)
"""The 12 holiday-name categories, in index order."""
N_HOLIDAY_NAMES: Final[int] = len(HOLIDAY_NAMES)
INDEX_COLUMNS: Final[tuple[str, ...]] = (
    "half_hour",
    "weekday",
    "day_of_month",
    "day_of_year",
    "holiday_name",
)
"""The columns of `calendar_indices`, in order."""
INDEX_SIZES: Final[dict[str, int]] = {
    "half_hour": HALF_HOURS_PER_DAY,
    "weekday": DAYS_PER_WEEK,
    "day_of_month": MAX_DAYS_IN_MONTH,
    "day_of_year": MAX_DAYS_IN_YEAR,
    "holiday_name": N_HOLIDAY_NAMES,
}
"""The number of categories of each index column."""
ROW_SET_COLUMNS: Final[tuple[str, ...]] = (
    "bank_holiday_day",
    "bank_holiday_week",
    "christmas_set",
    "easter_set",
)
FIRST_YEAR: Final[int] = 2018
LAST_YEAR: Final[int] = 2029
_NAME_TO_INDEX: Final[dict[str, int]] = {
    "New Year's Day": 1,
    "Good Friday": 2,
    "Easter Monday": 3,
    "May Day": 4,
    "Spring Bank Holiday": 5,
    "Late Summer Bank Holiday": 6,
    "Christmas Day": 7,
    "Boxing Day": 8,
}
OTHER_BANK_HOLIDAY: Final[int] = 9
SUBSTITUTE_DAY: Final[int] = 10
CHRISTMAS_WEEK_NO_HOLIDAY: Final[int] = 11
CHRISTMAS_SET_START: Final[tuple[int, int]] = (12, 24)
CHRISTMAS_SET_END: Final[tuple[int, int]] = (1, 1)
EASTER_SET_LENGTH_DAYS: Final[int] = 7
"""Days from the Monday of Holy Week to Easter Monday."""

# The hand-made calendar columns of experiment 397's feature matrix (18 in all: its `calendar` and
# `holidays` groups). The encoder arms drop all of them except `covid` and, for E1n and E1, the
# three holiday columns below.
HAND_MADE_COLUMNS: Final[tuple[str, ...]] = (
    "sin_hour",
    "cos_hour",
    "sin_hour_2",
    "cos_hour_2",
    *[f"dow_{d}" for d in range(1, 8)],
    "covid",
    "christmas_inside",
    "christmas_days",
    "christmas_tent",
    "bank_holiday",
    "nearest_days",
    "nearest_inside",
)
COVID_COLUMN: Final[str] = "covid"
HOLIDAY_INPUT_COLUMNS: Final[tuple[str, ...]] = ("bank_holiday", "nearest_days", "nearest_inside")
"""The holiday columns that stay real inputs in E1n and E1 (standardised as usual)."""


def kept_columns(*, names: list[str], keep_holiday_inputs: bool) -> list[int]:
    """Return the indices of the columns an encoder arm keeps as direct inputs to the trunk.

    Every column that is not one of the 18 hand-made calendar columns is kept. Of the 18, `covid`
    is kept, and so are `bank_holiday`, `nearest_days`, and `nearest_inside` if
    `keep_holiday_inputs`.

    Args:
        names: The input column names of experiment 397's feature matrix, in matrix order.
        keep_holiday_inputs: Whether the three holiday columns stay (E1n and E1) or go (E0 and E2).

    Returns:
        The kept column indices, ascending.

    Raises:
        ValueError: If one of the 18 hand-made columns is missing from `names`.
    """
    missing = sorted(set(HAND_MADE_COLUMNS) - set(names))
    if missing:
        msg = f"hand-made calendar columns missing from the feature matrix: {missing}"
        raise ValueError(msg)
    kept_hand_made = {COVID_COLUMN, *(HOLIDAY_INPUT_COLUMNS if keep_holiday_inputs else ())}
    return [i for i, n in enumerate(names) if n not in HAND_MADE_COLUMNS or n in kept_hand_made]


@cache
def _bank_holidays() -> dict[date, str]:
    """Return the England bank holidays (date to the `holidays` package's name), 2018 to 2029."""
    return dict(
        holidays.country_holidays("GB", subdiv="ENG", years=range(FIRST_YEAR, LAST_YEAR + 1))
    )


def _holiday_index(*, day: date, name: str | None) -> int:
    """Return the holiday-name category of a calendar day."""
    if name is None:
        in_christmas_week = (day.month, day.day) >= CHRISTMAS_SET_START and day.month == 12
        return CHRISTMAS_WEEK_NO_HOLIDAY if in_christmas_week else 0
    if "(observed)" in name:
        return SUBSTITUTE_DAY
    return _NAME_TO_INDEX.get(name, OTHER_BANK_HOLIDAY)


@cache
def day_table() -> pl.DataFrame:
    """Return one row per calendar day 2018 to 2029 with its holiday category and row-set flags.

    Columns: `local_date`, `holiday_name` (index into `HOLIDAY_NAMES`), and the four booleans of
    `ROW_SET_COLUMNS`.
    """
    bank = _bank_holidays()
    easter_mondays = [d for d, n in bank.items() if n == "Easter Monday"]
    easter_days = {
        m - timedelta(days=k) for m in easter_mondays for k in range(EASTER_SET_LENGTH_DAYS + 1)
    }
    bank_weeks = {d - timedelta(days=d.weekday()) for d in bank}
    days = [
        date(FIRST_YEAR, 1, 1) + timedelta(days=i)
        for i in range((date(LAST_YEAR, 12, 31) - date(FIRST_YEAR, 1, 1)).days + 1)
    ]
    return pl.DataFrame(
        {
            "local_date": days,
            "holiday_name": [_holiday_index(day=d, name=bank.get(d)) for d in days],
            "bank_holiday_day": [d in bank for d in days],
            "bank_holiday_week": [d - timedelta(days=d.weekday()) in bank_weeks for d in days],
            "christmas_set": [
                (d.month, d.day) >= CHRISTMAS_SET_START or (d.month, d.day) <= CHRISTMAS_SET_END
                for d in days
            ],
            "easter_set": [d in easter_days for d in days],
        },
        schema_overrides={"holiday_name": pl.Int64},
    )


def _local_time(*, valid_time: pl.Expr, shift_to_start: bool) -> pl.Expr:
    """Return the Europe/London time of `valid_time`, or of the start of the half-hour it ends.

    The model-input indices use the period-end stamp's own local time (`shift_to_start=False`), the
    convention of the hand-made columns of experiment 87 (`derived_columns`), so that only the
    encoding differs between R and the encoder arms. The row sets used for scoring take the local
    date of the half-hour's start (`shift_to_start=True`), as `e00396_arms.add_calendar` does.
    """
    shifted = valid_time.dt.offset_by("-30m") if shift_to_start else valid_time
    return shifted.dt.convert_time_zone("Europe/London")


def _with_local_date(*, frame: pl.DataFrame, shift_to_start: bool) -> pl.DataFrame:
    """Return `frame` with the day-table columns joined on the local date of `valid_time`."""
    dated = frame.with_columns(
        local_date=_local_time(
            valid_time=pl.col("valid_time"), shift_to_start=shift_to_start
        ).dt.date(),
    )
    joined = dated.join(
        day_table(), on="local_date", how="left", validate="m:1", maintain_order="left"
    )
    if joined["holiday_name"].null_count():
        msg = f"a valid time is outside {FIRST_YEAR} to {LAST_YEAR}"
        raise ValueError(msg)
    return joined


def add_row_sets(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with `local_date` and the four row-set booleans, from its `valid_time`.

    Args:
        frame: Any frame with a UTC `valid_time` column.

    Returns:
        The frame with `local_date` and the columns of `ROW_SET_COLUMNS`.
    """
    return _with_local_date(frame=frame, shift_to_start=True).drop("holiday_name")


def calendar_indices(*, valid_us: np.ndarray) -> np.ndarray:
    """Return the five calendar indices of rows, as an int64 array (rows, 5).

    Args:
        valid_us: Each row's valid time, in microseconds since the epoch (UTC).

    Returns:
        The columns of `INDEX_COLUMNS`: local half-hour, weekday, day of month, day of year, and
        holiday-name category.
    """
    frame = pl.DataFrame({"valid_time": valid_us}).with_columns(
        pl.col("valid_time").cast(pl.Datetime("us", "UTC"))
    )
    local = _local_time(valid_time=pl.col("valid_time"), shift_to_start=False)
    joined = _with_local_date(frame=frame, shift_to_start=False).with_columns(
        half_hour=local.dt.hour().cast(pl.Int64) * 2 + local.dt.minute().cast(pl.Int64) // 30,
        weekday=local.dt.weekday().cast(pl.Int64) - 1,
        day_of_month=local.dt.day().cast(pl.Int64) - 1,
        day_of_year=local.dt.ordinal_day().cast(pl.Int64) - 1,
    )
    out = joined.select(INDEX_COLUMNS).to_numpy().astype(np.int64)
    for j, name in enumerate(INDEX_COLUMNS):
        if out[:, j].min() < 0 or out[:, j].max() >= INDEX_SIZES[name]:
            msg = f"calendar index {name} is outside 0 to {INDEX_SIZES[name] - 1}"
            raise ValueError(msg)
    return out
