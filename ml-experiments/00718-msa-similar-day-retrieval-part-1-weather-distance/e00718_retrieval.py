# ruff: noqa: E501, PLR0911
"""Experiment 718: retrieval of history rows, the placebo draws, and the row tables (design section 2).

A row table holds, for each (Primary, origin day, lead day), the ages of the history rows of one arm,
newest first, and what chose each row. Training and prediction read the same table. The builders read
only inputs that exist at the origin (design section 5): the target row's weather, the calendar, each
pool block's age, whether the block holds power that the 27 h rule makes visible, and the pool
blocks' own weather. No measure reads a power value, and eligibility reads whether a reading exists.

All arrays are NumPy. The distance is computed in float64, and two blocks tie when their distances
differ by less than 1e-9 relative, in which case the younger block wins.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import e00718_layout as lay
import numpy as np

N_WEATHER: Final = 7
"""Shortwave, longwave, 2 m temperature, 2 m dew point, surface pressure, log precipitation, 10 m wind speed."""
DAY_TYPE_WEIGHT: Final = 1.0
MIN_VISIBLE_HALF_HOURS: Final = 24
"""A block is eligible when at least this many of its 48 half-hours hold visible, finite power."""
TIE_RELATIVE: Final = 1e-9
PLACEBO_SEED: Final = 0
NEWEST_VISIBLE_BEFORE_CONTRACT: Final = 54
"""The newest visible reading is stamped 27 hours (54 half-hours) before the contract origin."""
GRID_START_DATE: Final = date(2019, 9, 1)
CALENDAR_OFFSET_DIVISOR: Final = 3.0
CHRISTMAS_WINDOW_START: Final = (12, 20)
CHRISTMAS_WINDOW_END: Final = (1, 5)
N_WEATHER_ROWS_U: Final = 5
N_RECENT_ROWS_U: Final = 5
N_WEEKDAY_ROWS_U: Final = 4
N_WEATHER_ROWS_U27: Final = 7
N_WEEKDAY_ROWS_U27: Final = 6
N_R_ROWS: Final = 14
ARM_TAG: Final = {"R-A": 1, "P90": 2, "U-A": 3, "U27-A": 4}
"""Extra entropy per placebo arm, so that two placebos never share one stream."""


@dataclass(frozen=True)
class PoolArrays:
    """The fold's arrays that retrieval reads, on the CPU.

    Attributes:
        finite_cumsum: (16, G + 1) int64, the running count of finite power readings per Primary, so
            that the readings of a block are a difference of two entries.
        weather: (16, G, 7) float64, the standardised ERA5 and CAMS weather at each grid index.
        day_type: (G,) int64, the day type of every grid index (experiment 713's per-token day type).
        day_class: (G,) int64 or None, the calendar day class of every grid index (arm `UX` only).
        christmas_offset: (G // 48,) float64 or None, the offset in days of the block that starts at
            12:00 UTC of each day number from the nearest earlier 25 December.
        in_window: (G // 48,) bool or None, whether the block starts from 20 December to 5 January.
    """

    finite_cumsum: np.ndarray
    weather: np.ndarray
    day_type: np.ndarray
    day_class: np.ndarray | None = None
    christmas_offset: np.ndarray | None = None
    in_window: np.ndarray | None = None

    @property
    def n_grid(self) -> int:
        """Return the number of half-hours of the grid."""
        return self.weather.shape[1]


@dataclass(frozen=True)
class TargetRow:
    """The target row of one (Primary, origin day, lead day) sample.

    Attributes:
        weather: (48, 7) float64 weather as the network reads it (ENS mean, or ERA5 in stage 1).
        filled: (48,) bool, whether each half-hour has weather.
        day_type: (48,) int64 day types of the target row.
        day_class: (48,) int64 calendar day classes, or None.
        christmas_offset: Offset in days of the target block's start from 25 December, or 0.
        in_window: Whether the target block starts from 20 December to 5 January.
    """

    weather: np.ndarray
    filled: np.ndarray
    day_type: np.ndarray
    day_class: np.ndarray | None = None
    christmas_offset: float = 0.0
    in_window: bool = False


@dataclass
class RowTable:
    """The history rows of one arm for a set of Primaries and origin days.

    Attributes:
        arm: The arm name.
        primaries: (P,) the Primary index of each first-axis entry.
        days: (D,) the day number (`c // 48`) of each origin day.
        ages: (P, D, 14, n_rows) int16 ages, ascending, `ABSENT` where no block fills the row.
        measure: (P, D, 14, n_rows) int8, the measure code of each row.
        no_weather: (P, D, 14) bool, a target row with no weather (the weather rows took the youngest).
        distance: (P, D, 14, n_rows) float32, the weather distance of each chosen block to the target row
            (NaN for an absent row, or where the target row has no weather). It is a read-out only: the
            network never reads it.
    """

    arm: str
    primaries: np.ndarray
    days: np.ndarray
    ages: np.ndarray
    measure: np.ndarray
    no_weather: np.ndarray
    distance: np.ndarray | None = None

    def lookup(self, *, primary: np.ndarray, day: np.ndarray, lead_day: np.ndarray) -> np.ndarray:
        """Return the (N, n_rows) ages of the samples, by Primary index, day number, and lead day."""
        pi = np.searchsorted(self.primaries, primary)
        di = np.searchsorted(self.days, day)
        assert bool((self.primaries[pi] == primary).all()), "a Primary is not in the table"
        assert bool((self.days[di] == day).all()), "an origin day is not in the table"
        return self.ages[pi, di, lead_day].astype(np.int64)

    def lookup_measure(
        self, *, primary: np.ndarray, day: np.ndarray, lead_day: np.ndarray
    ) -> np.ndarray:
        """Return the (N, n_rows) measure codes of the samples."""
        pi = np.searchsorted(self.primaries, primary)
        di = np.searchsorted(self.days, day)
        return self.measure[pi, di, lead_day].astype(np.int64)

    def digest(self) -> str:
        """Return the sha256 (16 hex digits) of the arm, the keys, and the contents."""
        h = hashlib.sha256()
        h.update(self.arm.encode())
        for arr in (
            self.primaries,
            self.days,
            self.ages,
            self.measure,
            self.no_weather,
            self.distance,
        ):
            if arr is not None:
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------- calendar


def day_class_for_grid(*, grid_start_us: int, n_grid: int, holidays: set[date]) -> np.ndarray:
    """Return the calendar day class of every grid index, from the local date of the stamp minus 30 minutes.

    The classes are those of the exploration: 0 working weekday, 1 Saturday, 2 Sunday, 3 bank
    holiday in England, 4 bridge day from 27 to 31 December, 5 Christmas Eve weekday, and 6 first
    working days from 2 to 5 January. The order of tests is the exploration's.
    """
    import e00713_layout as lay713  # ty: ignore[unresolved-import]

    local_s = lay713._local_seconds(grid_start_us=grid_start_us, n_grid=n_grid)
    local_day = local_s // 86_400
    by_day: dict[int, int] = {}
    out = np.empty(n_grid, dtype=np.int64)
    for i, o in enumerate(local_day):
        o_int = int(o)
        if o_int not in by_day:
            by_day[o_int] = _class_of_date(
                d=date(1970, 1, 1) + timedelta(days=o_int), holidays=holidays
            )
        out[i] = by_day[o_int]
    return out


def _class_of_date(*, d: date, holidays: set[date]) -> int:
    """Return the day class of one local date."""
    if d in holidays:
        return 3
    if d.weekday() == 5:
        return 1
    if d.weekday() == 6:
        return 2
    if d.month == 12 and d.day == 24:
        return 5
    if d.month == 12 and 27 <= d.day <= 31:
        return 4
    if d.month == 1 and 2 <= d.day <= 5:
        return 6
    return 0


def christmas_arrays(*, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, per day number, the block start's offset in days from the nearest earlier 25 December, whether it lies from 20 December to 5 January, and its winter (the year of that 25 December)."""
    offsets = np.empty(n_days, dtype=np.float64)
    window = np.empty(n_days, dtype=bool)
    winter = np.empty(n_days, dtype=np.int64)
    for k in range(n_days):
        d = GRID_START_DATE + timedelta(days=k)
        year = d.year if d.month >= 7 else d.year - 1
        offsets[k] = (d - date(year, 12, 25)).days
        winter[k] = year
        window[k] = (d.month == 12 and d.day >= CHRISTMAS_WINDOW_START[1]) or (
            d.month == 1 and d.day <= CHRISTMAS_WINDOW_END[1]
        )
    return offsets, window, winter


def day_number_of_date(*, d: date) -> int:
    """Return the day number (days from 2019-09-01) of a date."""
    return (d - GRID_START_DATE).days


def datetime_of_day_number(*, day: int) -> datetime:
    """Return 00:00 UTC of a day number."""
    return datetime(2019, 9, 1, tzinfo=UTC) + timedelta(days=int(day))


# ---------------------------------------------------------------------------------- eligibility


def eligible_ages(
    *, arrays: PoolArrays, primary: int, contract_idx: int, ages: np.ndarray
) -> np.ndarray:
    """Return which of `ages` hold at least 24 visible, finite half-hours (the exploration's rule).

    A reading is visible when its grid index is at most `contract_idx - 54`. A block that begins
    before the grid start is not eligible. Only whether a reading exists is read, never its value.
    """
    lo = contract_idx - lay.STEPS_PER_DAY * ages
    hi = np.minimum(lo + lay.STEPS_PER_DAY, contract_idx - NEWEST_VISIBLE_BEFORE_CONTRACT + 1)
    inside = lo >= 0
    cs = arrays.finite_cumsum[primary]
    count = cs[np.clip(hi, 0, None)] - cs[np.clip(lo, 0, None)]
    return inside & (count >= MIN_VISIBLE_HALF_HOURS)


# ---------------------------------------------------------------------------------- selection


def pick_nearest(
    *, distance: np.ndarray, ages: np.ndarray, allowed: np.ndarray, n: int
) -> np.ndarray:
    """Return the `n` allowed ages with the smallest distance, applying the tie rule.

    Among the allowed blocks, the nearest is found, and every block whose distance is within 1e-9
    relative of it is a tie, of which the youngest is taken. The step repeats until `n` ages are
    chosen or no allowed block is left. The result is in the order chosen.
    """
    idx = np.flatnonzero(allowed)
    if len(idx) == 0 or n <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(distance[idx], kind="stable")
    idx = idx[order]
    d = distance[idx]
    alive = np.ones(len(idx), dtype=bool)
    chosen: list[int] = []
    while len(chosen) < n and bool(alive.any()):
        pos = np.flatnonzero(alive)
        m = d[pos[0]]
        limit = m + TIE_RELATIVE * abs(m)
        tied = pos[d[pos] <= limit]
        pick = tied[np.argmin(ages[idx[tied]])]
        chosen.append(int(ages[idx[pick]]))
        alive[pick] = False
    return np.array(chosen, dtype=np.int64)


def weather_distance(
    *, target: TargetRow, pool_weather: np.ndarray, pool_day_type: np.ndarray
) -> np.ndarray:
    """Return the distance of the target row to each pool block (design section 2.2).

    `pool_weather` is (A, 48, 7) float64 and `pool_day_type` (A, 48). The distance is the square
    root of the summed squared differences over the filled half-hours, scaled by 48 over their
    number, plus 1.0 times the share of the 48 columns whose day type differs. A target row with no
    filled half-hour has no distance, and the caller takes the youngest blocks.
    """
    n_filled = int(target.filled.sum())
    assert n_filled > 0, "a target row without weather has no distance"
    diff = pool_weather[:, target.filled, :] - target.weather[None, target.filled, :]
    total = (diff * diff).sum(axis=(1, 2)) * (lay.STEPS_PER_DAY / n_filled)
    share = (pool_day_type != target.day_type[None, :]).mean(axis=1)
    return np.sqrt(total) + DAY_TYPE_WEIGHT * share


def calendar_distance(
    *, target: TargetRow, pool_day_class: np.ndarray, pool_offset: np.ndarray
) -> np.ndarray:
    """Return the calendar distance of arm `UX`: the share of half-hours whose day class differs, plus the offset difference over 3."""
    assert target.day_class is not None
    share = (pool_day_class != target.day_class[None, :]).mean(axis=1)
    return share + np.abs(pool_offset - target.christmas_offset) / CALENDAR_OFFSET_DIVISOR


def same_weekday(*, ages: np.ndarray, lead_day: int) -> np.ndarray:
    """Return which block ages start on the same weekday as the target block.

    The target block starts `lead_day` days after the origin day and a pool block `age` days before
    it, so the two start weekdays agree when `lead_day + age` is a multiple of 7. Bank holidays are
    ignored, as in the exploration.
    """
    return (ages + lead_day) % 7 == 0


def fill_union(
    *, weather_rows: np.ndarray, ages: np.ndarray, ok: np.ndarray, lead_day: int, n_weather: int,
    n_recent: int, n_weekday: int,
) -> tuple[list[int], list[int]]:  # fmt: skip
    """Return the ages and measure codes of the union: weather rows, then recency rows, then weekday rows.

    `weather_rows` holds the weather-nearest ages already ordered by the caller's measure, so the
    first `n_weather` are used. A later measure that meets an age already chosen takes its next best
    block, so no age repeats.
    """
    chosen = [int(a) for a in weather_rows[:n_weather]]
    measures = [lay.MEASURE_WEATHER] * len(chosen)
    taken = np.isin(ages, chosen)
    younger = pick_nearest(
        distance=ages.astype(np.float64), ages=ages, allowed=ok & ~taken, n=n_recent
    )
    chosen += [int(a) for a in younger]
    measures += [lay.MEASURE_RECENCY] * len(younger)
    taken = np.isin(ages, chosen)
    same = same_weekday(ages=ages, lead_day=lead_day)
    weekday = pick_nearest(
        distance=ages.astype(np.float64), ages=ages, allowed=ok & ~taken & same, n=n_weekday
    )
    chosen += [int(a) for a in weekday]
    measures += [lay.MEASURE_WEEKDAY] * len(weekday)
    return chosen, measures


def draw_matched(
    *, rng: np.random.Generator, treatment_ages: np.ndarray, ok: np.ndarray, ages: np.ndarray,
    exclude: np.ndarray, max_age: int,
) -> np.ndarray:  # fmt: skip
    """Return ages that replace `treatment_ages` one for one by drawing from the same age class.

    Each replaced row is drawn uniformly, without repeats, from the eligible blocks of its age class
    that are not in `exclude`. A class with too few blocks passes its shortfall to the next older
    class, and a shortfall at the oldest class leaves fewer rows than were replaced, which the caller
    pads with `ABSENT`. The result is sorted ascending.
    """
    bounds = lay.class_bounds(max_age=max_age)
    cls = lay.age_class(ages=treatment_ages)
    need = np.bincount(cls, minlength=len(bounds))[: len(bounds)].astype(np.int64)
    pool_ok = ok & ~np.isin(ages, exclude)
    drawn: list[int] = []
    carry = 0
    for c, (lo, hi) in enumerate(bounds):
        want = int(need[c]) + carry
        cand = ages[pool_ok & (ages >= lo) & (ages <= hi)]
        cand = np.setdiff1d(cand, np.array(drawn, dtype=np.int64))
        take = min(want, len(cand))
        if take:
            drawn += [int(a) for a in rng.choice(cand, size=take, replace=False)]
        carry = want - take
    return np.sort(np.array(drawn, dtype=np.int64))


# ---------------------------------------------------------------------------------- arms


def _pad(*, ages: list[int], measures: list[int], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Sort rows by ascending age and pad to `n_rows` with `ABSENT`."""
    order = np.argsort(np.array(ages, dtype=np.int64), kind="stable")
    a = np.full(n_rows, lay.ABSENT, dtype=np.int16)
    m = np.full(n_rows, lay.MEASURE_FIXED, dtype=np.int8)
    a[: len(ages)] = np.array(ages, dtype=np.int64)[order]
    m[: len(ages)] = np.array(measures, dtype=np.int64)[order]
    assert len(set(ages)) == len(ages), "an age repeats inside a sample"
    return a, m


def rows_for_sample(
    *, arm: str, target: TargetRow, ages: np.ndarray, ok: np.ndarray, distance: np.ndarray | None,
    cal_distance: np.ndarray | None, lead_day: int, primary: int, day: int,
) -> tuple[np.ndarray, np.ndarray, bool]:  # fmt: skip
    """Return the ages, the measure codes, and the no-weather flag of one sample of `arm`.

    Args:
        arm: `Z`, `R`, `R-A`, `P90`, `U`, `U-A`, `U27`, `U27-A`, or `UX`.
        target: The target row (its weather and day types).
        ages: (A,) the pool ages, ascending, of the arm's pool (2 to 91, or 2 to the long pool).
        ok: (A,) eligibility of each pool age.
        distance: (A,) the weather distance, or None when the target row has no weather.
        cal_distance: (A,) the calendar distance (arm `UX` only).
        lead_day: The lead day index 0 to 13.
        primary: The Primary index.
        day: The origin's day number.
    """
    n_rows = lay.ROWS_PER_ARM[arm]
    no_weather = distance is None
    near = distance if distance is not None else ages.astype(np.float64)
    if arm == "Z":
        a = list(range(lay.MIN_AGE, lay.NEWEST_FIXED_MAX_AGE + 1))
        return (*_pad(ages=a, measures=[lay.MEASURE_FIXED] * len(a), n_rows=n_rows), no_weather)
    if arm == "R":
        got = pick_nearest(distance=near, ages=ages, allowed=ok, n=N_R_ROWS)
        return (
            *_pad(
                ages=[int(a) for a in got], measures=[lay.MEASURE_WEATHER] * len(got), n_rows=n_rows
            ),
            no_weather,
        )
    if arm in {"R-A", "P90", "U-A", "U27-A"}:
        rng = np.random.default_rng(
            [PLACEBO_SEED, int(primary), int(day), int(lead_day), ARM_TAG[arm]]
        )
        if arm == "P90":
            cand = ages[ok]
            n = min(N_R_ROWS, len(cand))
            got = (
                np.sort(rng.choice(cand, size=n, replace=False)).astype(np.int64)
                if n
                else np.empty(0, np.int64)
            )
            return (
                *_pad(
                    ages=[int(a) for a in got],
                    measures=[lay.MEASURE_RANDOM] * len(got),
                    n_rows=n_rows,
                ),
                no_weather,
            )
        base_arm = {"R-A": "R", "U-A": "U", "U27-A": "U27"}[arm]
        t_ages, t_meas, _ = rows_for_sample(
            arm=base_arm, target=target, ages=ages, ok=ok, distance=distance, cal_distance=cal_distance,
            lead_day=lead_day, primary=primary, day=day,
        )  # fmt: skip
        t_ages, t_meas = t_ages[t_ages >= 0].astype(np.int64), t_meas[: int((t_ages >= 0).sum())]
        keep_mask = t_meas == (lay.MEASURE_RECENCY if arm == "U-A" else lay.MEASURE_FIXED)
        kept = t_ages[keep_mask]
        replaced = t_ages[~keep_mask]
        got = draw_matched(
            rng=rng, treatment_ages=replaced, ok=ok & ~np.isin(ages, kept), ages=ages,
            exclude=t_ages, max_age=int(ages.max()),
        )  # fmt: skip
        out = [*kept, *got]
        meas = [lay.MEASURE_FIXED if arm == "U27-A" else lay.MEASURE_RECENCY] * len(kept) + [
            lay.MEASURE_RANDOM
        ] * len(got)
        return (*_pad(ages=[int(a) for a in out], measures=meas, n_rows=n_rows), no_weather)
    if arm in {"U", "UX"}:
        use_calendar = arm == "UX" and target.in_window and cal_distance is not None
        ok_u = ok & (ages <= lay.LONG_POOL_MAX_AGE)
        if use_calendar:
            assert cal_distance is not None
            in_window = ok & _window_ages(ages=ages, day=day, lead_day=lead_day)
            cal = pick_nearest(
                distance=cal_distance, ages=ages, allowed=in_window, n=N_WEATHER_ROWS_U
            )
            rest = pick_nearest(
                distance=near, ages=ages, allowed=ok_u & ~np.isin(ages, cal),
                n=N_WEATHER_ROWS_U - len(cal),
            )  # fmt: skip
            w_ages = np.concatenate([cal, rest])
            chosen_cal = len(cal)
        else:
            w_ages = pick_nearest(distance=near, ages=ages, allowed=ok_u, n=N_WEATHER_ROWS_U)
            chosen_cal = 0
        a, m = fill_union(
            weather_rows=w_ages, ages=ages, ok=ok_u, lead_day=lead_day, n_weather=N_WEATHER_ROWS_U,
            n_recent=N_RECENT_ROWS_U, n_weekday=N_WEEKDAY_ROWS_U,
        )  # fmt: skip
        for i in range(chosen_cal):
            m[i] = lay.MEASURE_CALENDAR
        return (*_pad(ages=a, measures=m, n_rows=n_rows), no_weather)
    if arm == "U27":
        fixed = list(range(lay.MIN_AGE, lay.NEWEST_FIXED_MAX_AGE + 1))
        older = ages > lay.NEWEST_FIXED_MAX_AGE
        w = pick_nearest(distance=near, ages=ages, allowed=ok & older, n=N_WEATHER_ROWS_U27)
        a, m = fill_union(
            weather_rows=w, ages=ages, ok=ok & older, lead_day=lead_day, n_weather=N_WEATHER_ROWS_U27,
            n_recent=0, n_weekday=N_WEEKDAY_ROWS_U27,
        )  # fmt: skip
        return (
            *_pad(ages=fixed + a, measures=[lay.MEASURE_FIXED] * len(fixed) + m, n_rows=n_rows),
            no_weather,
        )
    msg = f"unknown arm {arm!r}"
    raise ValueError(msg)


def _window_ages(*, ages: np.ndarray, day: int, lead_day: int) -> np.ndarray:
    """Return which block ages start from 20 December to 5 January of a winter earlier than the target block's."""
    _, window, winter = _CHRISTMAS_CACHE
    return window[day - ages] & (winter[day - ages] != winter[day + lead_day])


_CHRISTMAS_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] = christmas_arrays(n_days=4000)
"""The offsets and the window flags of 4,000 day numbers (2019-09-01 to 2030), shared by all callers."""


def long_pool_ages(*, arm: str, day: int) -> np.ndarray:
    """Return the ages of the arm's pool for the origin day `day`, ascending."""
    if arm == "UX":
        return np.arange(lay.MIN_AGE, day + 1, dtype=np.int64)
    max_age = lay.SHORT_POOL_MAX_AGE if arm in {"R", "R-A", "P90"} else lay.LONG_POOL_MAX_AGE
    return np.arange(lay.MIN_AGE, max_age + 1, dtype=np.int64)


TargetProvider = Callable[[int, int], tuple[np.ndarray, np.ndarray]]
"""Maps (Primary, origin day number) to the 14 target rows' weather (14, 48, 7) and filled mask (14, 48)."""


YoungWeather = Callable[[int, int], "np.ndarray | None"]
"""Returns the stored-forecast weather of the blocks of ages 2 to 5 for a (Primary, origin day)."""
YOUNGEST_ANALYSIS_AGE: Final = 6
"""The live-fair rule reads the analysis from this age on."""


def build_table(
    *, arm: str, arrays: PoolArrays, primaries: np.ndarray, days: np.ndarray,
    target_provider: TargetProvider, with_calendar: bool | None = None,
    young_weather: YoungWeather | None = None,
) -> RowTable:  # fmt: skip
    """Build the row table of one arm (see `build_tables`)."""
    return build_tables(
        arms=[arm], arrays=arrays, primaries=primaries, days=days, target_provider=target_provider,
        with_calendar=with_calendar, young_weather=young_weather,
    )[arm]  # fmt: skip


def build_tables(  # noqa: PLR0915
    *, arms: Sequence[str], arrays: PoolArrays, primaries: np.ndarray, days: np.ndarray,
    target_provider: TargetProvider, with_calendar: bool | None = None,
    young_weather: YoungWeather | None = None,
) -> dict[str, RowTable]:  # fmt: skip
    """Build the row tables of several arms in one pass, computing each weather distance once.

    The distance of a (Primary, origin day, lead day) sample to every block of ages 2 to 379 is
    computed once, and every arm that reads weather takes the prefix of it that its pool covers
    (arm `R`'s pool is the first 90 ages), so that `R`, `R-A`, `U`, `U-A`, `U27`, `U27-A`, and `UX`
    cost one distance computation instead of seven (design section 11).

    Args:
        arms: The arm names.
        arrays: The fold's retrieval arrays.
        primaries: (P,) Primary indices, ascending.
        days: (D,) origin day numbers, ascending.
        target_provider: Returns the 14 target rows' weather for a (Primary, day).
        with_calendar: Whether to read the calendar arrays (default: arm `UX` only).
        young_weather: The selection swap's live-fair rule: for a (Primary, day) it returns the
            (4, 48, 7) float64 weather that the blocks of ages 2 to 5 read in the distance (the
            stored ensemble forecast), or None to keep the analysis. Ages 6 and above always read
            the analysis. The default None is the pool of every arm.
    """
    weather_arms = {"R", "R-A", "U", "U-A", "UX", "U27", "U27-A"}
    shape = (len(primaries), len(days), lay.N_LEAD_DAYS)
    outs: dict[str, dict[str, Any]] = {}
    for arm in arms:
        n_rows = lay.ROWS_PER_ARM[arm]
        ages_out = np.full((*shape, n_rows), lay.ABSENT, dtype=np.int16)
        outs[arm] = {
            "ages": ages_out,
            "measure": np.full(ages_out.shape, lay.MEASURE_FIXED, dtype=np.int8),
            "no_weather": np.zeros(shape, dtype=bool),
            "distance": np.full(ages_out.shape, np.nan, dtype=np.float32),
            "calendar": (arm == "UX") if with_calendar is None else with_calendar,
        }
    any_weather = any(a in weather_arms for a in arms)
    for pi, p in enumerate(primaries):
        for di, day in enumerate(days):
            c = int(lay.contract_idx_of_day(day=int(day)))
            long_ages = np.arange(lay.MIN_AGE, lay.LONG_POOL_MAX_AGE + 1, dtype=np.int64)
            long_ages = long_ages[c - lay.STEPS_PER_DAY * long_ages >= 0]
            weather_t, filled_t = target_provider(int(p), int(day))
            pool_w = pool_dt = None
            if any_weather and len(long_ages):
                grid = lay.block_index(contract_idx=c, ages=long_ages)
                pool_w = arrays.weather[int(p)][grid]
                young = None if young_weather is None else young_weather(int(p), int(day))
                if young is not None:
                    assert young.shape == (4, lay.STEPS_PER_DAY, N_WEATHER)
                    assert long_ages[0] == lay.MIN_AGE
                    n_young = int((long_ages <= YOUNGEST_ANALYSIS_AGE - 1).sum())
                    pool_w = pool_w.copy()
                    pool_w[:n_young] = young[:n_young]
                pool_dt = arrays.day_type[grid]
            per_arm = {}
            for arm in arms:
                pool_ages = long_pool_ages(arm=arm, day=int(day))
                pool_ages = pool_ages[c - lay.STEPS_PER_DAY * pool_ages >= 0]
                ok = eligible_ages(arrays=arrays, primary=int(p), contract_idx=c, ages=pool_ages)
                per_arm[arm] = (pool_ages, ok, int((pool_ages <= lay.LONG_POOL_MAX_AGE).sum()))
            for lead in range(lay.N_LEAD_DAYS):
                top = c + lay.STEPS_PER_DAY * lead + np.arange(lay.STEPS_PER_DAY)
                has_weather = bool(filled_t[lead].any())
                long_distance = None
                for arm in arms:
                    o = outs[arm]
                    pool_ages, ok, n_w = per_arm[arm]
                    cal_ok = o["calendar"] and arrays.day_class is not None
                    target = TargetRow(
                        weather=weather_t[lead], filled=filled_t[lead], day_type=arrays.day_type[top],
                        day_class=arrays.day_class[top] if cal_ok else None,
                        christmas_offset=float(_CHRISTMAS_CACHE[0][day + lead]),
                        in_window=bool(_CHRISTMAS_CACHE[1][day + lead]),
                    )  # fmt: skip
                    distance = None
                    if arm in weather_arms and len(pool_ages) and has_weather:
                        if long_distance is None:
                            assert pool_w is not None
                            long_distance = weather_distance(
                                target=target,
                                pool_weather=pool_w,
                                pool_day_type=pool_dt,  # ty: ignore[invalid-argument-type]
                            )
                        distance = np.full(len(pool_ages), np.inf)
                        distance[:n_w] = long_distance[:n_w]
                    cal = None
                    if cal_ok and len(pool_ages):
                        grid_c = lay.block_index(contract_idx=c, ages=pool_ages)
                        cal = calendar_distance(
                            target=target, pool_day_class=arrays.day_class[grid_c],  # ty: ignore[not-subscriptable]
                            pool_offset=_CHRISTMAS_CACHE[0][day - pool_ages],
                        )  # fmt: skip
                    a, m, nw = rows_for_sample(
                        arm=arm, target=target, ages=pool_ages, ok=ok, distance=distance, cal_distance=cal,
                        lead_day=lead, primary=int(p), day=int(day),
                    )  # fmt: skip
                    o["ages"][pi, di, lead], o["measure"][pi, di, lead] = a, m
                    if distance is not None:
                        got = a >= 0
                        pos = np.searchsorted(pool_ages, a[got])
                        o["distance"][pi, di, lead][got] = distance[pos]
                    o["no_weather"][pi, di, lead] = nw and arm in weather_arms
    return {
        arm: RowTable(
            arm=arm,
            primaries=np.asarray(primaries),
            days=np.asarray(days),
            ages=o["ages"],
            measure=o["measure"],
            no_weather=o["no_weather"],
            distance=o["distance"],
        )
        for arm, o in outs.items()
    }
