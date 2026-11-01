# ruff: noqa: E501
"""Christmas-period regime inputs, weights, lag substitution, and replay rows for experiment 485.

The Christmas window is 22 December to 2 January inclusive, by UTC date, everywhere. Four pieces:

- `class_maps`: for each Primary, the day type (0 weekday-like, 1 Saturday-like, 2 Sunday-like) of
  each calendar date in the window, voted from the Primary's own earlier Christmases;
- `extra_columns`: three added inputs, `sat_like` and `sun_like` (H1) and `lag_substituted` (H3);
- `substitute_lags`: replaces a day or weekly lag that falls inside the window by the value at the
  same half-hour of the latest earlier day of the target's day type outside the window;
- `window_weight`: the multiplier of the Christmas rows (H2).

Day types outside the window follow the weekday (Monday to Friday 0, Saturday 1, Sunday 2); bank
holidays outside the window are not remapped.
"""

from datetime import UTC, datetime
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np

US_PER_DAY: Final[int] = 86_400 * 1_000_000
STEP_PER_DAY: Final[int] = 48
N_EXTRA: Final[int] = 3
"""Added input columns: `sat_like`, `sun_like`, `lag_substituted`."""
WINDOW_WEIGHT: Final[float] = 5.0
REPLAY_FRACTION: Final[float] = 0.25
MIN_LEVEL_DAYS: Final[int] = 5
SEARCH_DAYS: Final[int] = 70
NAMES, _MASK, _LAYOUT = base.feature_names(scheme="summary")
COL: Final[dict[str, int]] = {n: i for i, n in enumerate(NAMES)}


def _date_parts(*, day: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (month, day of month, weekday 0=Monday) of integer days since the epoch."""
    d = day.astype("datetime64[D]")
    month = d.astype("datetime64[M]").astype(int) % 12 + 1
    dom = (d - d.astype("datetime64[M]")).astype(int) + 1
    weekday = (day + 3) % 7
    return month, dom, weekday


def in_window_day(*, day: np.ndarray) -> np.ndarray:
    """Return True where an epoch day is 22 December to 2 January."""
    month, dom, _ = _date_parts(day=day)
    return ((month == 12) & (dom >= 22)) | ((month == 1) & (dom <= 2))


def in_window(*, valid_us: np.ndarray) -> np.ndarray:
    """Return True where a valid time (microseconds) is inside the window."""
    return in_window_day(day=valid_us // US_PER_DAY)


def dow_class(*, day: np.ndarray) -> np.ndarray:
    """Return 0 for Monday to Friday, 1 for Saturday, 2 for Sunday."""
    weekday = (day + 3) % 7
    return np.where(weekday == 5, 1, np.where(weekday == 6, 2, 0))


def daily_means(*, power: np.ndarray) -> tuple[int, np.ndarray]:
    """Return the first epoch day of the grid and each day's mean power (NaN under 44 half-hours)."""
    idx = np.arange(len(power), dtype=np.int64)
    day = (prep422.GRID_START_US + idx * prep422.STEP_US) // US_PER_DAY
    day0 = int(day[0])
    d = day - day0
    ok = np.isfinite(power)
    total = np.bincount(d[ok], weights=power[ok].astype(np.float64), minlength=int(d.max()) + 1)
    count = np.bincount(d[ok], minlength=int(d.max()) + 1)
    return day0, np.where(count >= 44, total / np.maximum(count, 1), np.nan)


def class_map(
    *, power: np.ndarray, refit: datetime
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Return a Primary's day type for each calendar date in the window, voted over earlier years.

    A year counts only if its window ended at least 4 days before the refit. In each year, the three
    reference levels are the mean daily power of the weekdays, Saturdays, and Sundays between 15
    November and 14 December and between 10 January and 10 February (outside the window), and a
    window day takes the nearest level. The year votes are combined by mode; a tie goes to the
    higher class. The map keeps each year's vote as (window year, class); `row_class` uses only the
    votes from years before a row's own window year, so training rows never see their own year.
    """
    day0, mean = daily_means(power=power)
    refit_day = int(refit.timestamp() // 86_400)
    votes: dict[tuple[int, int], list[int]] = {}
    for year in range(2019, refit.year + 1):
        start = int(datetime(year, 12, 22, tzinfo=UTC).timestamp() // 86_400)
        end = start + 11
        if end + 40 > refit_day or start - 40 - day0 < 0 or end + 40 - day0 >= len(mean):
            continue
        ref_days = np.r_[start - 37 : start, end + 10 : end + 40]
        ref_days = ref_days[~in_window_day(day=ref_days)]
        values = mean[ref_days - day0]
        classes = dow_class(day=ref_days)
        levels = np.array(
            [
                np.nanmean(values[classes == c])
                if (np.isfinite(values[classes == c])).sum() >= MIN_LEVEL_DAYS
                else np.nan
                for c in range(3)
            ]
        )
        if not np.isfinite(levels).all():
            continue
        for day in range(start, end + 1):
            value = mean[day - day0]
            if not np.isfinite(value):
                continue
            month, dom, _ = _date_parts(day=np.array([day]))
            votes.setdefault((int(month[0]), int(dom[0])), []).append(
                (year, int(np.argmin(np.abs(levels - value))))
            )
    return votes


def class_maps(
    *, clean: np.ndarray, refit: datetime
) -> list[dict[tuple[int, int], list[tuple[int, int]]]]:
    """Return `class_map` for each row of `clean` (Primaries, grid)."""
    return [class_map(power=clean[p], refit=refit) for p in range(clean.shape[0])]


def row_class(*, day: np.ndarray, primary: np.ndarray, maps: list | None) -> np.ndarray:
    """Return the day type of each row's date: the voted class in the window, else the weekday."""
    out = dow_class(day=day).astype(np.int64)
    if maps is None:
        return out
    inside = in_window_day(day=day)
    month, dom, _ = _date_parts(day=day)
    year = (day.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970) - (month == 1)
    for i in np.nonzero(inside)[0]:
        past = [
            c for y, c in maps[int(primary[i])].get((int(month[i]), int(dom[i])), []) if y < year[i]
        ]
        if past:
            out[i] = max(range(3), key=lambda c: (past.count(c), c))
    return out


def substitute_lags(
    *,
    x: np.ndarray,
    valid_us: np.ndarray,
    lead: np.ndarray,
    primary: np.ndarray,
    clean: np.ndarray,
    maps: list,
    standardiser: base.Standardiser,
) -> np.ndarray:
    """Return a 0/1 flag per row and edit `x` in place: lags inside the window are replaced.

    For a row whose day or weekly lag half-hour falls inside the window, the lag becomes the
    cleaned power at the same half-hour of the latest earlier day that is outside the window, has
    the target date's day type, and is observed. The lag is never younger than the original (the
    search only moves back), so the newest-usable-observation rule holds. If no such day exists
    within `SEARCH_DAYS` days the lag is left as it was. The replacement is standardised like the
    column and its missing flag is cleared.
    """
    target_idx = ((valid_us - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
    gap = np.rint(lead.astype(np.float64) * 2.0).astype(np.int64) + common.NEWEST_AGE
    target_day = valid_us // US_PER_DAY
    target_class = row_class(day=target_day, primary=primary, maps=maps)
    flag = np.zeros(len(x), dtype=np.float32)
    grid_start_day = prep422.GRID_START_US // US_PER_DAY
    for tag, period in (("day", STEP_PER_DAY), ("week", 7 * STEP_PER_DAY)):
        lag_idx = target_idx - period * np.ceil(gap / period).astype(np.int64)
        lag_day = (prep422.GRID_START_US + lag_idx * prep422.STEP_US) // US_PER_DAY
        todo = np.nonzero(in_window_day(day=lag_day))[0]
        if not len(todo):
            continue
        found = np.full(len(x), np.nan)
        steps = np.zeros(len(x), dtype=np.int64)
        pending = np.zeros(len(x), dtype=bool)
        pending[todo] = True
        for m in range(SEARCH_DAYS):
            cand = lag_idx[todo] - STEP_PER_DAY * m
            cand_day = (prep422.GRID_START_US + cand * prep422.STEP_US) // US_PER_DAY
            ok = (
                (cand >= 0)
                & ~in_window_day(day=cand_day)
                & (dow_class(day=cand_day) == target_class[todo])
            )
            ok &= pending[todo]
            value = np.full(len(todo), np.nan)
            for p in np.unique(primary[todo][ok]):
                sel = ok & (primary[todo] == p)
                value[sel] = clean[p][cand[sel]]
            hit = ok & np.isfinite(value)
            found[todo[hit]] = value[hit]
            steps[todo[hit]] = m
            pending[todo[hit]] = False
            if not pending[todo].any():
                break
        done = np.isfinite(found)
        for name in (f"lag_{tag}",):
            col = COL[name]
            z = (found[done] - standardiser.mean[col]) / standardiser.std[col]
            x[done, col] = np.clip(z, -base.INPUT_CLIP_SD, base.INPUT_CLIP_SD)
        x[done, COL[f"lag_{tag}_missing"]] = 0.0
        x[done, COL[f"lag_{tag}_age"]] += steps[done] * 24.0 / base.LEAD_SCALE_HOURS
        flag[done] = 1.0
    del grid_start_day
    return flag


def extra_columns(
    *,
    x: np.ndarray,
    valid_us: np.ndarray,
    lead: np.ndarray,
    primary: np.ndarray,
    clean: np.ndarray,
    maps: list,
    standardiser: base.Standardiser,
    day_type: bool,
    substitute: bool,
) -> np.ndarray:
    """Return `x` with `N_EXTRA` columns appended: `sat_like`, `sun_like`, `lag_substituted`.

    `day_type` fills the first two (window days only) and `substitute` edits the lags and fills the
    third. With both False the columns are zero, so the network's width is the same in every arm.
    """
    out = np.zeros((len(x), x.shape[1] + N_EXTRA), dtype=np.float32)
    out[:, : x.shape[1]] = x
    n = x.shape[1]
    if day_type:
        day = valid_us // US_PER_DAY
        inside = in_window_day(day=day)
        cls = row_class(day=day, primary=primary, maps=maps)
        out[:, n] = (inside & (cls == 1)).astype(np.float32)
        out[:, n + 1] = (inside & (cls == 2)).astype(np.float32)
    if substitute:
        part = out[:, :n]
        out[:, n + 2] = substitute_lags(
            x=part,
            valid_us=valid_us,
            lead=lead,
            primary=primary,
            clean=clean,
            maps=maps,
            standardiser=standardiser,
        )
    return out


def window_weight(*, valid_us: np.ndarray) -> np.ndarray:
    """Return `WINDOW_WEIGHT` on window rows and 1 elsewhere (multiplies the recency weight)."""
    return np.where(in_window(valid_us=valid_us), WINDOW_WEIGHT, 1.0).astype(np.float32)
