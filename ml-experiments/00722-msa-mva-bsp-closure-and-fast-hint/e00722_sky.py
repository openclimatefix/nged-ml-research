# ruff: noqa: E501
"""Experiment 722: cloud classes, the class audit, same-sky pairs, stamp lags, and stretches.

Everything here is pure: arrays on the half-hour grid of `e00722_core` in, arrays and tables out.
The sky functions use CAMS and sun geometry only, and the audit uses the recorded output of the
six farms of experiment 39 as an independent measurement. No Primary's reading enters this module
except in `touch_level`, `touches`, and `stretches`, which job 3 calls after the audit.
"""

import itertools
from dataclasses import dataclass
from typing import Final

import e00722_core as core
import numpy as np

CLASS_OTHER: Final[int] = 0
"""A day that is none of the three classes, or has too little CAMS data."""
CLASS_INTERMITTENT: Final[int] = 1
"""An intermittent day (section 4)."""
CLASS_CLEAR: Final[int] = 2
"""A clear day (section 4)."""
CLASS_OVERCAST: Final[int] = 3
"""An overcast day (section 4)."""


def day_of(*, n: int) -> np.ndarray:
    """Return the UTC day number of each grid row."""
    return np.arange(n) // core.HALF_HOURS_PER_DAY


def month_of_day(*, n_days: int) -> np.ndarray:
    """Return the calendar month (1 to 12) of each grid day."""
    first = np.datetime64(core.GRID_FIRST_END.date(), "D")
    days = first + np.arange(n_days)
    return days.astype("datetime64[M]").astype(int) % 12 + 1


def clear_sky_index(*, ghi: np.ndarray, ghi_clear: np.ndarray) -> np.ndarray:
    """Return `k = ghi / ghi_clear`, NaN where the clear-sky irradiance is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ghi_clear > 0, ghi / ghi_clear, np.nan)


def daylight(*, zenith_deg: np.ndarray, min_elevation_deg: float) -> np.ndarray:
    """Return True where the sun's elevation is at least `min_elevation_deg`."""
    return np.isfinite(zenith_deg) & ((90.0 - zenith_deg) >= min_elevation_deg)


def fast_change(*, k: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Return `|k_t - k_{t-1}|` where both half-hours are lit and have an index, else NaN."""
    out = np.full(k.shape, np.nan)
    both = lit[1:] & lit[:-1] & np.isfinite(k[1:]) & np.isfinite(k[:-1])
    out[1:][both] = np.abs(k[1:][both] - k[:-1][both])
    return out


def days_with_data(*, k: np.ndarray, lit: np.ndarray, n_days: int) -> np.ndarray:
    """Return True for each day with at least `MIN_DAYLIGHT_HALF_HOURS` lit half-hours of CAMS data."""
    day = day_of(n=len(k))
    use = lit & np.isfinite(k)
    return np.bincount(day[use], minlength=n_days) >= core.MIN_DAYLIGHT_HALF_HOURS


def classify_days(*, k: np.ndarray, lit: np.ndarray, n_days: int) -> np.ndarray:
    """Return the class code of each day from the clear-sky index of the lit half-hours.

    Args:
        k: The clear-sky index on the grid (NaN where CAMS has no row).
        lit: True at daylight half-hours at the chosen elevation limit.
        n_days: Number of grid days.

    Returns:
        An integer per day: `CLASS_INTERMITTENT`, `CLASS_CLEAR`, `CLASS_OVERCAST`, or
        `CLASS_OTHER`. A day with fewer than `MIN_DAYLIGHT_HALF_HOURS` lit half-hours is other.
    """
    day = day_of(n=len(k))
    use = lit & np.isfinite(k)
    n_lit = np.bincount(day[use], minlength=n_days)
    total = np.bincount(day[use], weights=k[use], minlength=n_days)
    kbar = np.divide(total, n_lit, out=np.full(n_days, np.nan), where=n_lit > 0)
    jump = fast_change(k=k, lit=lit)
    big = np.isfinite(jump) & (jump >= core.FAST_STEP_K)
    n_fast = np.bincount(day[big], minlength=n_days)
    out = np.full(n_days, CLASS_OTHER, dtype=int)
    enough = n_lit >= core.MIN_DAYLIGHT_HALF_HOURS
    lo, hi = core.INTERMITTENT_K_BAND
    out[enough & (n_fast >= core.INTERMITTENT_MIN_FAST) & (kbar >= lo) & (kbar <= hi)] = (
        CLASS_INTERMITTENT
    )
    out[enough & (kbar >= core.CLEAR_MIN_K) & (n_fast < core.CLEAR_MAX_FAST)] = CLASS_CLEAR
    out[enough & (kbar < core.OVERCAST_MAX_K)] = CLASS_OVERCAST
    return out


def fast_irradiance_change(
    *, u: np.ndarray, u_clear: np.ndarray, k: np.ndarray, lit: np.ndarray, n_days: int
) -> np.ndarray:
    """Return `du~_t = (u_t - u_{t-1}) - kbar_d * (u_clear_t - u_clear_{t-1})` on lit half-hours.

    The second term removes the arc of the sun. `kbar_d` is the day's mean clear-sky index over
    lit half-hours. Rows where either half-hour lacks a value or is dark are NaN.
    """
    day = day_of(n=len(u))
    use = lit & np.isfinite(k)
    n_lit = np.bincount(day[use], minlength=n_days)
    total = np.bincount(day[use], weights=k[use], minlength=n_days)
    kbar = np.divide(total, n_lit, out=np.full(n_days, np.nan), where=n_lit > 0)
    out = np.full(u.shape, np.nan)
    ok = lit[1:] & lit[:-1]
    d_u = u[1:] - u[:-1]
    d_c = u_clear[1:] - u_clear[:-1]
    out[1:] = np.where(ok, d_u - kbar[day[1:]] * d_c, np.nan)
    return out


@dataclass(frozen=True)
class AuditRow:
    """One line of the class audit."""

    condition: int
    scope: str
    value: float
    threshold: str
    passed: bool
    readable: bool
    n_days: int


def count_big_changes(
    *, output_mw: np.ndarray, capacity_mw: float, lit: np.ndarray, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per day the number of farm-output changes above 10% of capacity, and day coverage.

    Coverage is the share of the day's daylight half-hours that hold an output reading. A change
    counts only where both half-hours are lit and have a reading.
    """
    day = day_of(n=len(output_mw))
    have = lit & np.isfinite(output_mw)
    n_lit = np.bincount(day[lit], minlength=n_days)
    n_have = np.bincount(day[have], minlength=n_days)
    coverage = np.divide(n_have, n_lit, out=np.zeros(n_days), where=n_lit > 0)
    d = np.full(output_mw.shape, np.nan)
    both = have[1:] & have[:-1]
    d[1:][both] = np.abs(output_mw[1:][both] - output_mw[:-1][both])
    big = np.isfinite(d) & (d > core.AUDIT_BIG_CHANGE_SHARE_OF_CAPACITY * capacity_mw)
    return np.bincount(day[big], minlength=n_days), coverage


def audit_farm(
    *, classes: np.ndarray, n_big: np.ndarray, coverage: np.ndarray, scope: str
) -> list[AuditRow]:
    """Return audit conditions 1 and 2 for one farm.

    Condition 1: of the intermittent days, at least 70% show at least 4 changes of the farm's
    output above 10% of its capacity. Condition 2: of the clear days, at most 5% do. Only days
    with farm-output coverage of at least `MIN_FARM_DAY_COVERAGE` count, and a condition is not
    readable with fewer than `AUDIT_MIN_FARM_DAYS` marked days.
    """
    usable = coverage >= core.MIN_FARM_DAY_COVERAGE
    rows = []
    for condition, cls, text in (
        (1, CLASS_INTERMITTENT, f"at least {core.AUDIT_INTERMITTENT_MIN_SHARE}"),
        (2, CLASS_CLEAR, f"at most {core.AUDIT_CLEAR_MAX_SHARE}"),
    ):
        days = usable & (classes == cls)
        n = int(days.sum())
        share = float((n_big[days] >= core.AUDIT_MIN_BIG_CHANGES).mean()) if n else float("nan")
        readable = n >= core.AUDIT_MIN_FARM_DAYS
        if condition == 1:
            passed = readable and share >= core.AUDIT_INTERMITTENT_MIN_SHARE
        else:
            passed = readable and share <= core.AUDIT_CLEAR_MAX_SHARE
        rows.append(AuditRow(condition, scope, share, text, passed, readable, n))
    return rows


def audit_season_share(
    *, classes: np.ndarray, month: np.ndarray, has_data: np.ndarray, scope: str
) -> AuditRow:
    """Return audit condition 3 for one Primary: the intermittent share of April to September days.

    Only days with CAMS data count in the denominator.
    """
    days = np.isin(month, core.AUDIT_SEASON_MONTHS) & has_data
    n = int(days.sum())
    share = float((classes[days] == CLASS_INTERMITTENT).mean()) if n else float("nan")
    lo, hi = core.AUDIT_INTERMITTENT_SHARE_BAND
    ok = n > 0 and lo <= share <= hi
    return AuditRow(3, scope, share, f"{lo} to {hi}", ok, n > 0, n)


def audit_elevation_sensitivity(
    *, classes_10: np.ndarray, classes_15: np.ndarray, has_data: np.ndarray, scope: str
) -> AuditRow:
    """Return audit condition 4: the share of days whose class changes at an elevation limit of 15 degrees."""
    n = int(has_data.sum())
    share = float((classes_10[has_data] != classes_15[has_data]).mean()) if n else float("nan")
    passed = n > 0 and share <= core.AUDIT_MAX_CHANGED_SHARE
    return AuditRow(4, scope, share, f"at most {core.AUDIT_MAX_CHANGED_SHARE}", passed, n > 0, n)


def audit_passes(*, rows: list[AuditRow]) -> bool:
    """Return True when every readable condition passes and each condition has a readable row.

    An unreadable farm row (too few marked days) is ignored only when another farm is readable
    for the same condition. Conditions 3 and 4 must be readable.
    """
    for condition in (1, 2, 3, 4):
        cond = [r for r in rows if r.condition == condition]
        readable = [r for r in cond if r.readable]
        if not readable or not all(r.passed for r in readable):
            return False
    return True


def correlation_of_changes(*, a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Return the Pearson correlation of two change series over the rows where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < core.MIN_PAIR_ROWS or a[ok].std() == 0 or b[ok].std() == 0:
        return float("nan"), n
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), n


def fast_k_change(*, k: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Return the signed `k_t - k_{t-1}` where both half-hours are lit and have an index."""
    out = np.full(k.shape, np.nan)
    both = lit[1:] & lit[:-1] & np.isfinite(k[1:]) & np.isfinite(k[:-1])
    out[1:][both] = k[1:][both] - k[:-1][both]
    return out


def peak_lag(
    *, x: np.ndarray, y: np.ndarray, max_lag: int = core.STAMP_MAX_LAG
) -> tuple[int, float]:
    """Return the lag in half-hours at which `|corr(x[t], y[t - lag])|` peaks, and that correlation.

    A positive lag means that `x` is a delayed copy of `y`: a stamp on `x` that sits `lag`
    half-hours later than the same instant on `y`. Rows with a missing value are skipped in each
    correlation.
    """
    best_lag, best = 0, -1.0
    best_corr = float("nan")
    for lag in range(-max_lag, max_lag + 1):
        ys = core.shift_rows(x=y, rows=lag)
        ok = np.isfinite(x) & np.isfinite(ys)
        if ok.sum() < core.MIN_PAIR_ROWS or x[ok].std() == 0 or ys[ok].std() == 0:
            continue
        c = float(np.corrcoef(x[ok], ys[ok])[0, 1])
        if abs(c) > best:
            best_lag, best, best_corr = lag, abs(c), c
    return best_lag, best_corr


# ---------------------------------------------------------------------------------------------
# Touches and stretches (section 4)
# ---------------------------------------------------------------------------------------------
def touch_level(*, reading: np.ndarray, lit: np.ndarray, capacity_mw: float) -> float:
    """Return `B_i`: a third of the 99th percentile of the daylight half-hour changes, at least 10 points."""
    d = np.full(reading.shape, np.nan)
    both = lit[1:] & lit[:-1]
    d[1:][both] = np.abs(reading[1:][both] - reading[:-1][both])
    v = d[np.isfinite(d)]
    if v.size == 0:
        return float("nan")
    level = float(np.quantile(v, core.TOUCH_QUANTILE)) / core.TOUCH_DIVISOR
    return max(level, core.TOUCH_MIN_POINTS * capacity_mw / 100.0)


def touches(*, reading: np.ndarray, lit: np.ndarray, level: float) -> np.ndarray:
    """Return True at daylight half-hours whose raw reading is a local minimum at or below `level`.

    The raw reading is used because a 3-point running median removes the single-half-hour
    minimum of a fast crossing.
    """
    out = np.zeros(reading.shape, dtype=bool)
    mid = reading[1:-1]
    out[1:-1] = (
        lit[1:-1] & np.isfinite(mid) & (mid <= level) & (mid <= reading[:-2]) & (mid <= reading[2:])
    )
    return out


def running_median3(*, x: np.ndarray) -> np.ndarray:
    """Return the 3-point running median (the ends keep their value)."""
    out = x.copy()
    stack = np.stack([x[:-2], x[1:-1], x[2:]])
    out[1:-1] = np.median(stack, axis=0)
    return out


def day_stretches(
    *, touch: np.ndarray, lit: np.ndarray, usable: np.ndarray
) -> dict[int, list[tuple[int, int, bool]]]:
    """Return per day the stretches as `(first row, last row, is_post_bounce)`.

    A stretch is a maximal run of lit half-hours between two consecutive touches, or between a
    touch and the end of daylight, less one half-hour beside each touch. The first and the last
    stretch of a day are anchors; every other stretch is post-bounce. A day with a lit half-hour
    that has no reading (`usable` False) is skipped.
    """
    day = day_of(n=len(touch))
    out: dict[int, list[tuple[int, int, bool]]] = {}
    for d in np.unique(day[lit]):
        rows = np.flatnonzero((day == d) & lit)
        if not usable[rows].all():
            continue
        cuts = [r for r in rows if touch[r]]
        bounds = [rows[0] - 1, *cuts, rows[-1] + 1]
        spans = []
        for a, b in itertools.pairwise(bounds):
            lo = a + 1 + (1 if a in cuts else 0)
            hi = b - 1 - (1 if b in cuts else 0)
            if hi >= lo:
                spans.append((lo, hi))
        n = len(spans)
        out[int(d)] = [(int(lo), int(hi), 0 < i < n - 1) for i, (lo, hi) in enumerate(spans)]
    return out


def post_bounce_counts(
    *, stretches: dict[int, list[tuple[int, int, bool]]], n_days: int
) -> np.ndarray:
    """Return the number of post-bounce stretches per day, -1 for a day with no stretch table."""
    out = np.full(n_days, -1, dtype=int)
    for d, spans in stretches.items():
        out[d] = sum(1 for *_, post in spans if post)
    return out


def merged_by_median(
    *, reading: np.ndarray, lit: np.ndarray, level: float, usable: np.ndarray
) -> int:
    """Return how many stretches the 3-point median rule would merge, against the raw rule."""
    raw = day_stretches(
        touch=touches(reading=reading, lit=lit, level=level), lit=lit, usable=usable
    )
    med = day_stretches(
        touch=touches(reading=running_median3(x=reading), lit=lit, level=level),
        lit=lit,
        usable=usable,
    )
    n_raw = sum(len(v) for v in raw.values())
    n_med = sum(len(v) for v in med.values())
    return n_raw - n_med


def concordance(
    *, d_reading: np.ndarray, d_sky: np.ndarray, usable: np.ndarray
) -> tuple[float, int, bool]:
    """Return the signed concordance `C`, the number of usable pairs, and whether it is evidence.

    `C = (n_same - n_opposite) / (n_same + n_opposite)` over the usable pairs. A stretch gives no
    evidence with fewer than 8 usable pairs or `|C|` under 0.25. This pure function belongs to
    rung U1, and the first slice does not run it.
    """
    ok = usable & np.isfinite(d_reading) & np.isfinite(d_sky)
    same = int((np.sign(d_reading[ok]) == np.sign(d_sky[ok])).sum())
    n = int(ok.sum())
    opposite = n - same
    c = (same - opposite) / n if n else float("nan")
    evidence = n >= 8 and abs(c) >= 0.25
    return float(c), n, bool(evidence)
