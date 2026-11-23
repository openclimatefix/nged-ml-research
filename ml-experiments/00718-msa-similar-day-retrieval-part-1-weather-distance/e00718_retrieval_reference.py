# ruff: noqa: E501
"""Experiment 718, WP2: an independent recomputation of the row tables, written from design section 2 alone (KR3).

This module imports nothing from the table builder (`e00718_retrieval`, `e00718_tables`, `e00718_layout`).
It takes the loaded arrays as plain NumPy arrays and picks the rows of one sample with explicit loops
over the pool blocks, in float64, and sums the 336 squared differences of a distance with `math.fsum`
(an exactly rounded sum), so that its summation order differs from the builder's `ndarray.sum`.

Inputs of every function: `finite` (P, G) bool, whether a power reading exists and is unflagged at each
half-hour of the grid; `weather` (P, G, 7) float64; `day_type` (G,) int; `day_class` (G,) int or None;
`grid_start` the date of grid index 0; and the target row (48, 7) with its (48,) filled mask. A block of
age `a` before the contract origin `c = 48 d + 24` covers the grid indices `c - 48 a` to `c - 48 a + 47`.
"""

import math
from datetime import date, timedelta

import numpy as np

STEPS = 48
MIN_VISIBLE = 24
LAST_VISIBLE_BEFORE = 54  # a reading is visible when its index is at most c - 54
TIE = 1e-9
POOL_MAX = {"R": 91, "U": 379, "U27": 379}
WEATHER, RECENCY, WEEKDAY, FIXED, CALENDAR = "weather", "recency", "weekday", "fixed", "calendar"


def contract(*, day: int) -> int:
    """Return the grid index of 12:00 UTC on day number `day`."""
    return STEPS * day + 24


def is_eligible(*, finite: np.ndarray, p: int, c: int, age: int) -> bool:
    """Return whether the block holds at least 24 visible, finite half-hours, and starts inside the grid."""
    lo = c - STEPS * age
    if lo < 0:
        return False
    count = 0
    for j in range(STEPS):
        idx = lo + j
        if idx <= c - LAST_VISIBLE_BEFORE and bool(finite[p, idx]):
            count += 1
    return count >= MIN_VISIBLE


def weather_distance(
    *, weather: np.ndarray, day_type: np.ndarray, p: int, c: int, age: int, lead: int,
    target_w: np.ndarray, target_filled: np.ndarray,
) -> float:  # fmt: skip
    """Return the distance of the target block to the pool block of `age` (design section 2.2)."""
    lo = c - STEPS * age
    top = c + STEPS * lead
    terms: list[float] = []
    for j in range(STEPS):
        if not bool(target_filled[j]):
            continue
        for v in range(weather.shape[2]):
            diff = float(weather[p, lo + j, v]) - float(target_w[j, v])
            terms.append(diff * diff)
    n_filled = int(np.sum(target_filled))
    total = math.fsum(terms) * (STEPS / n_filled)
    differ = sum(1 for j in range(STEPS) if int(day_type[lo + j]) != int(day_type[top + j]))
    return math.sqrt(total) + 1.0 * (differ / STEPS)


def calendar_distance(
    *, day_class: np.ndarray, p_unused: int, c: int, age: int, lead: int, offset_block: float, offset_target: float
) -> float:  # fmt: skip
    """Return the share of half-hours whose day class differs, plus the offset difference over 3."""
    del p_unused
    lo = c - STEPS * age
    top = c + STEPS * lead
    differ = sum(1 for j in range(STEPS) if int(day_class[lo + j]) != int(day_class[top + j]))
    return differ / STEPS + abs(offset_block - offset_target) / 3.0


def christmas(*, grid_start: date, day: int) -> tuple[float, bool, int]:
    """Return the offset in days of a block start from the nearest earlier 25 December, whether it starts from 20 December to 5 January, and its winter."""
    d = grid_start + timedelta(days=int(day))
    winter = d.year if d.month >= 7 else d.year - 1
    offset = float((d - date(winter, 12, 25)).days)
    in_window = (d.month == 12 and d.day >= 20) or (d.month == 1 and d.day <= 5)
    return offset, in_window, winter


def nearest(*, scores: dict[int, float], allowed: list[int], n: int) -> list[int]:
    """Return up to `n` ages of `allowed` in the order chosen: the smallest score, with a tie (relative 1e-9) to the younger age."""
    left = sorted(allowed)
    out: list[int] = []
    while left and len(out) < n:
        best = min(scores[a] for a in left)
        limit = best + TIE * abs(best)
        tied = [a for a in left if scores[a] <= limit]
        pick = min(tied)
        out.append(pick)
        left.remove(pick)
    return out


def reference_rows(
    *, arm: str, finite: np.ndarray, weather: np.ndarray, day_type: np.ndarray, day_class: np.ndarray | None,
    grid_start: date, p: int, day: int, lead: int, target_w: np.ndarray, target_filled: np.ndarray,
) -> tuple[list[tuple[int, str]], dict[int, float]]:  # fmt: skip
    """Return the (age, measure) rows of one sample of `arm` (`R`, `U`, `U27`, or `UX`), sorted by age, and the distances used.

    The second result maps each pool age to the distance that ordered it (the weather distance, or the age
    for a target row with no filled half-hour), so that a caller can tell a tie from a mismatch.
    """
    c = contract(day=day)
    base = "U" if arm == "UX" else arm
    ages = [
        a
        for a in range(2, (day if arm == "UX" else POOL_MAX[base]) + 1)
        if is_eligible(finite=finite, p=p, c=c, age=a)
    ]
    long_ages = [a for a in ages if a <= POOL_MAX["U"]] if arm == "UX" else ages
    has_weather = int(np.sum(target_filled)) > 0
    dist: dict[int, float] = {}
    for a in long_ages:
        dist[a] = (
            weather_distance(
                weather=weather,
                day_type=day_type,
                p=p,
                c=c,
                age=a,
                lead=lead,
                target_w=target_w,
                target_filled=target_filled,
            )
            if has_weather
            else float(a)
        )
    rows: list[tuple[int, str]] = []
    if arm == "R":
        rows = [(a, WEATHER) for a in nearest(scores=dist, allowed=long_ages, n=14)]
    elif arm in {"U", "UX"}:
        calendar: list[int] = []
        _, target_in_window, target_winter = christmas(grid_start=grid_start, day=day + lead)
        if arm == "UX" and target_in_window:
            assert day_class is not None
            t_off = christmas(grid_start=grid_start, day=day + lead)[0]
            cal_scores = {}
            window_ages = []
            for a in ages:
                off, win, winter = christmas(grid_start=grid_start, day=day - a)
                if win and winter != target_winter:
                    window_ages.append(a)
                    cal_scores[a] = calendar_distance(
                        day_class=day_class,
                        p_unused=p,
                        c=c,
                        age=a,
                        lead=lead,
                        offset_block=off,
                        offset_target=t_off,
                    )
            calendar = nearest(scores=cal_scores, allowed=window_ages, n=5)
        rest = nearest(
            scores=dist, allowed=[a for a in long_ages if a not in calendar], n=5 - len(calendar)
        )
        weather_rows = calendar + rest
        rows = [(a, CALENDAR if a in calendar else WEATHER) for a in weather_rows]
        younger = nearest(
            scores={a: float(a) for a in long_ages},
            allowed=[a for a in long_ages if a not in weather_rows],
            n=5,
        )
        rows += [(a, RECENCY) for a in younger]
        taken = {a for a, _ in rows}
        same = [a for a in long_ages if a not in taken and (a + lead) % 7 == 0]
        rows += [
            (a, WEEKDAY) for a in nearest(scores={a: float(a) for a in same}, allowed=same, n=4)
        ]
    elif arm == "U27":
        older = [a for a in long_ages if a > 15]
        w = nearest(scores=dist, allowed=older, n=7)
        rows = [(a, FIXED) for a in range(2, 16)] + [(a, WEATHER) for a in w]
        taken = {a for a, _ in rows}
        same = [a for a in older if a not in taken and (a + lead) % 7 == 0]
        rows += [
            (a, WEEKDAY) for a in nearest(scores={a: float(a) for a in same}, allowed=same, n=6)
        ]
    else:
        msg = f"no reference for arm {arm!r}"
        raise ValueError(msg)
    return sorted(rows), dist
