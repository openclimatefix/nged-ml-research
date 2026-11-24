# ruff: noqa: E501
"""Experiment 722: the named bars, the bootstrap, the matched-time de-meaning, and the decision rules.

Every threshold of the experiment is a `Final` constant of this module, set before any real data
are read. The functions are pure (arrays in, arrays out) so that the tests can run them on
synthetic arrays. Nothing here reads a file.

**Conventions.** A grid index `i` stands for the half-hour that ends at `GRID_FIRST_END + i * 30
minutes`, and a day is `i // 48`, because the first half-hour starts at 00:00 UTC. A meter reads
positive for import. Every series is in megawatts, apart from the add-back count of job 1, which
uses points of the Primary's capacity.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import product
from typing import Final, Literal

import numpy as np

# ---------------------------------------------------------------------------------------------
# Grid and calendar
# ---------------------------------------------------------------------------------------------
HALF_HOURS_PER_DAY: Final[int] = 48
"""Half-hours in a UTC day."""

GRID_FIRST_END: Final[datetime] = datetime(2019, 9, 1, 0, 30, tzinfo=UTC)
"""The end stamp of the grid's first half-hour (it starts at 2019-09-01 00:00 UTC)."""

HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time on or after this instant is read."""

N_GRID: Final[int] = int((HOLDOUT - GRID_FIRST_END) / timedelta(minutes=30))
"""Half-hours whose end stamp is before the holdout."""

# ---------------------------------------------------------------------------------------------
# Bootstrap and detection limits (spec 8.1, 8.3, 8.4)
# ---------------------------------------------------------------------------------------------
N_BOOT: Final[int] = 1000
"""Bootstrap replicates, each resampling whole days."""

SEED: Final[int] = 20261124
"""The seed of every bootstrap."""

SE_MULT: Final[float] = 2.8
"""Detection limit, in bootstrap standard errors."""

INTERVAL_PERCENTILES: Final[tuple[float, float]] = (2.5, 97.5)
"""The 95% interval's percentiles."""

CORE_SECOND_CAP: Final[float] = 3600.0
"""A job whose projection exceeds 1 core-hour stops."""

# ---------------------------------------------------------------------------------------------
# Job 1 (F1, F5, F7, section 4)
# ---------------------------------------------------------------------------------------------
ADDBACK_NULL_BAND: Final[tuple[float, float]] = (-0.2, 0.2)
"""An interval inside this band means "added back, or not below this Primary" (F1)."""

EMBEDDED_BAND: Final[tuple[float, float]] = (-1.3, -0.7)
"""An interval inside this band means "embedded and not added back" (F1)."""

ENGINE_BELOW_POINTS: Final[float] = -2.0
"""Reading minus the engine record, in points of capacity, below which form `|X| + A` fails."""

MIN_ENGINE_COVERAGE: Final[float] = 0.9
"""Share of the engine record's half-hours that the BSP meter must cover, or the engine test is void."""

MIN_GENERATOR_COVERAGE: Final[float] = 0.8
"""A generator series with less coverage of the grid is not subtracted from a residual."""

DOMINANT_FARM_SHARE: Final[float] = 1.0 / 3.0
"""One farm "dominates" D12's PV when it holds at least this share of the capacity (F7)."""

MIN_SAME_SKY_PAIRS: Final[int] = 4
"""Fewer same-sky pairs than this skips jobs 4 and 5 (F5)."""

SAME_SKY_CORRELATION: Final[float] = 0.5
"""A host and a farm are a same-sky pair when their fast clear-sky-index changes correlate this much."""

MIN_PAIR_ROWS: Final[int] = 500
"""A pair's correlation needs at least this many common daylight half-hours."""

STAMP_MAX_LAG: Final[int] = 2
"""The stamp check scans lags of plus and minus this many half-hours."""

STAMP_MIN_CORR: Final[float] = 0.05
"""A series whose best lagged correlation with the sky change is under this has no signal for the stamp check."""

ENGINE_IDENTITY_CORR: Final[float] = 0.99
"""A generator series that correlates with the engine record at least this much is the engine's own series."""

LONG_GAP_HALF_HOURS: Final[int] = 48
"""A gap longer than one day."""

DAYLIGHT_ELEVATION_DEG: Final[float] = 10.0
"""Sun elevation at least this is a daylight half-hour (section 4)."""

ELEVATION_SENSITIVITY_DEG: Final[float] = 15.0
"""Class audit condition 4 repeats the class at this elevation."""

DARK_ELEVATION_DEG: Final[float] = 0.0
"""A dark half-hour has the sun at or below this elevation."""

FAST_STEP_K: Final[float] = 0.15
"""A daylight half-hour counts as a fast change when `|k_t - k_{t-1}|` is at least this."""

INTERMITTENT_MIN_FAST: Final[int] = 6
"""An intermittent day holds at least this many fast half-hours."""

INTERMITTENT_K_BAND: Final[tuple[float, float]] = (0.35, 0.85)
"""An intermittent day's mean clear-sky index lies in this band."""

CLEAR_MIN_K: Final[float] = 0.85
"""A clear day's mean clear-sky index is at least this."""

CLEAR_MAX_FAST: Final[int] = 2
"""A clear day holds fewer fast half-hours than this."""

OVERCAST_MAX_K: Final[float] = 0.35
"""An overcast day's mean clear-sky index is under this."""

MIN_DAYLIGHT_HALF_HOURS: Final[int] = 6
"""A day with fewer daylight half-hours of CAMS data is not classed."""

AUDIT_INTERMITTENT_MIN_SHARE: Final[float] = 0.70
"""Audit 1: of the intermittent days, at least this share show enough farm-output changes."""

AUDIT_CLEAR_MAX_SHARE: Final[float] = 0.05
"""Audit 2: of the clear days, at most this share show enough farm-output changes."""

AUDIT_MIN_BIG_CHANGES: Final[int] = 4
"""A day shows this many half-hour changes of farm output above the size threshold."""

AUDIT_BIG_CHANGE_SHARE_OF_CAPACITY: Final[float] = 0.10
"""The size threshold, as a share of the farm's capacity."""

AUDIT_SEASON_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""Audit 3 counts days from April to September."""

AUDIT_INTERMITTENT_SHARE_BAND: Final[tuple[float, float]] = (0.20, 0.60)
"""Audit 3: the intermittent share of days at every Primary lies in this band."""

AUDIT_MAX_CHANGED_SHARE: Final[float] = 0.10
"""Audit 4: at most this share of days change class when the elevation limit moves to 15 degrees."""

AUDIT_MIN_FARM_DAYS: Final[int] = 10
"""A farm with fewer marked days than this is not readable for audits 1 and 2."""

MIN_FARM_DAY_COVERAGE: Final[float] = 0.9
"""A farm's day counts only when this share of its daylight half-hours holds an output reading."""

# ---------------------------------------------------------------------------------------------
# Job 2 bars (spec 8.4, copied from section 5.6 with F2 and S3)
# ---------------------------------------------------------------------------------------------
ENGINE_DETECT_MAX: Final[float] = 0.5
"""Detection limit, engine: 2.8 bootstrap standard errors under 0.5; otherwise void."""

D15_DETECT_MAX: Final[float] = 0.3
"""Detection limit, D15 closure: 2.8 bootstrap standard errors under 0.3; otherwise void."""

ENGINE_SLOPE_BAND: Final[tuple[float, float]] = (-1.3, -0.7)
"""Engine record: the slope interval excludes 0 and lies within this band."""

ENGINE_PLACEBO_SHIFT_DAYS: Final[tuple[int, ...]] = (7, 14)
"""Circular shifts of the engine record inside the record; each interval includes 0."""

D15_CLOSURE_BAND: Final[tuple[float, float]] = (0.7, 1.3)
"""D15 closure: the interval of the slope lies within this band."""

D15_AUC_MIN: Final[float] = 0.85
"""D15 sign: the area under the curve is at least this."""

D15_SIZE_NOISE_MULT: Final[float] = 2.0
"""D15 sign scores hours whose `|dm|` is at least this many noise scales of the residual's change."""

D15_MIN_HOURS: Final[int] = 200
"""D15 sign needs at least this many scorable hours."""

PLACEBO_SHIFT_DAYS: Final[int] = 7
"""The D15 and D12 reading placebos shift the de-meaned reading by this many days."""

AUC_NULL: Final[float] = 0.5
"""The D15 placebo's interval of the area includes this value."""

ENGINE_LAG_HALF_HOURS: Final[int] = 4
"""The engine test uses 2-hour changes."""

HOURLY_LAG_HALF_HOURS: Final[int] = 2
"""The D15 test uses non-overlapping hourly changes."""

# ---------------------------------------------------------------------------------------------
# Job 3 bars (spec 8.4, copied from section 5.7 with F3 and F4)
# ---------------------------------------------------------------------------------------------
U2B_DETECT_MAX: Final[float] = 1.0
"""Detection limit: 2.8 bootstrap standard errors (whole-day blocks) under 1; otherwise void."""

CALIBRATION_LOWER_ABOVE: Final[float] = 0.5
"""D12, calibration fit: the lower end of the interval is above this."""

TEST_REVERSE_UPPER_BELOW: Final[float] = -0.5
"""D12, test fit: an upper end below this supports reverse flow."""

TEST_REFUTE_LOWER_ABOVE: Final[float] = 0.5
"""D12, test fit: a lower end above this refutes reverse flow."""

ZERO: Final[float] = 0.0
"""D3 (both fits) must lie above this, and a shifted-date interval must include it."""

QUIET_PAD_HALF_HOURS: Final[int] = 6
"""A quiet half-hour lies at least this far from a change in the engine state."""

ENGINE_STEP_DROP_HALF_HOURS: Final[int] = 2
"""Job 3 drops rows within this many half-hours of a change in the engine state (S4)."""

MIN_QUIET_PER_KEY: Final[int] = 5
"""A matched key with fewer quiet half-hours than this gives no mean."""

TOUCH_QUANTILE: Final[float] = 0.99
"""The touch level is one third of this quantile of the Primary's daylight half-hour changes."""

TOUCH_DIVISOR: Final[float] = 3.0
"""The quantile's divisor in the touch level."""

TOUCH_MIN_POINTS: Final[float] = 10.0
"""The touch level is at least this many points of capacity."""

ONE_STRETCH: Final[int] = 1
"""The bars bind on days with exactly this many post-bounce stretches."""

MANY_STRETCHES: Final[int] = 3
"""Days with at least this many post-bounce stretches are reported apart."""

TEST_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""The test fit uses April to September."""

# ---------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------
BSP_A_PRIMARIES: Final[tuple[str, ...]] = (
    "D3", "D8", "D12", "D22", "D5", "D14", "D15", "D16", "D18",
)  # fmt: skip
"""The Primaries that spec 8.3 check 1 asserts BSP A holds."""

BSP_B_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D4", "D19", "D21", "D9", "D10", "D11")
"""The Primaries that spec 8.3 check 1 asserts BSP B holds."""

BSP_A_UNSIGNED: Final[tuple[str, ...]] = ("D3", "D8", "D12", "D22")
"""The MVA-only Primaries of BSP A."""

BSP_A_SIGNED: Final[tuple[str, ...]] = ("D5", "D14", "D15", "D16", "D18")
"""The signed Primaries of BSP A."""

BSP_B_UNSIGNED: Final[tuple[str, ...]] = ("D2", "D4", "D19", "D21")
"""The MVA-only Primaries of BSP B."""

BSP_B_SIGNED: Final[tuple[str, ...]] = ("D9", "D10", "D11")
"""The signed Primaries of BSP B."""

TWO_TRANSFORMERS_2025: Final[tuple[str, ...]] = ("D15", "D18")
"""The two signed transformers whose history starts in 2025."""

Outcome = Literal["pass", "fail", "void"]
Verdict = Literal["supported", "not supported", "inconclusive", "invalid"]
AddbackLabel = Literal["added_back_or_not_below", "embedded_not_added_back", "unresolved"]
GeneratorClass = Literal["outside_or_added_back", "embedded_not_added_back", "unresolved"]
TestOutcome = Literal["supports_reverse_flow", "refutes_reverse_flow", "unresolved", "void"]


# ---------------------------------------------------------------------------------------------
# Changes, matched keys, de-meaning
# ---------------------------------------------------------------------------------------------
def change(*, x: np.ndarray, lag: int) -> np.ndarray:
    """Return `x[t] - x[t - lag]`, NaN where either value is missing and in the first `lag` rows."""
    out = np.full(x.shape, np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out


def shift_rows(*, x: np.ndarray, rows: int) -> np.ndarray:
    """Return `y[t] = x[t - rows]` with NaN where the source row does not exist."""
    out = np.full(x.shape, np.nan)
    if rows >= 0:
        out[rows:] = x[: len(x) - rows]
    else:
        out[:rows] = x[-rows:]
    return out


def circular_shift(*, x: np.ndarray, rows: int) -> np.ndarray:
    """Return `x` rolled by `rows` inside its own length, so that every row is kept."""
    return np.roll(x, rows)


def matched_key(
    *, local_half_hour: np.ndarray, month: np.ndarray, day_type: np.ndarray
) -> np.ndarray:
    """Return an integer key for (local half-hour, month, day type); -1 where any part is missing."""
    ok = (local_half_hour >= 0) & (month >= 1) & (day_type >= 0)
    key = (local_half_hour.astype(np.int64) * 13 + month) * 2 + day_type
    return np.where(ok, key, -1)


def quiet_mask(*, engine_change: np.ndarray, calendar_ok: np.ndarray) -> np.ndarray:
    """Return half-hours at least `QUIET_PAD_HALF_HOURS` from any change in the engine state.

    Args:
        engine_change: True at the first half-hour with a new engine state.
        calendar_ok: True where the engine frame has a row, so that the half-hour has a key.
    """
    near = within_distance(flag=engine_change, distance=QUIET_PAD_HALF_HOURS)
    return calendar_ok & ~near


def within_distance(*, flag: np.ndarray, distance: int) -> np.ndarray:
    """Return True where `flag` is True at some row within `distance` rows (both sides)."""
    cs = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    n = len(flag)
    idx = np.arange(n)
    lo = np.clip(idx - distance, 0, n)
    hi = np.clip(idx + distance + 1, 0, n)
    return (cs[hi] - cs[lo]) > 0


def demean_matched(*, x: np.ndarray, key: np.ndarray, quiet: np.ndarray) -> np.ndarray:
    """Subtract each key's mean over quiet half-hours from `x`.

    This is a hindsight statistic: the mean pools every quiet half-hour of the span, past and
    future alike, because the jobs of this experiment test a hindsight pathway and not a forecast.
    A key with fewer than `MIN_QUIET_PER_KEY` quiet half-hours with a value gives NaN.

    Args:
        x: The series (a change, in megawatts).
        key: The matched key of each half-hour; -1 means no key.
        quiet: True at quiet half-hours.
    """
    use = quiet & np.isfinite(x) & (key >= 0)
    n_keys = int(key.max()) + 2 if key.size else 1
    total = np.bincount(key[use], weights=x[use], minlength=n_keys)
    count = np.bincount(key[use], minlength=n_keys)
    mean = np.full(n_keys, np.nan)
    enough = count >= MIN_QUIET_PER_KEY
    mean[enough] = total[enough] / count[enough]
    out = np.full(x.shape, np.nan)
    valid = key >= 0
    out[valid] = x[valid] - mean[key[valid]]
    return out


# ---------------------------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------------------------
def leave_out_residual(
    *,
    bsp: np.ndarray,
    children: Mapping[str, np.ndarray],
    leave_out: str | None,
    generators: Sequence[np.ndarray],
) -> np.ndarray:
    """Return `R(-i) = B - sum of every other child + sum of the subtracted generators`.

    Every generator's output is positive. A generator that sits outside every Primary, or whose
    output a Primary's series adds back, enters with a plus sign, because both leave the same
    remainder (spec 8.7 item 3). The scored child's own series is never subtracted. With
    `leave_out=None` every child is subtracted.
    """
    out = bsp.astype(np.float64).copy()
    for label, series in children.items():
        if label != leave_out:
            out = out - series
    for series in generators:
        out = out + series
    return out


def all_children_residual(*, leave_out: np.ndarray, reading: np.ndarray) -> np.ndarray:
    """Return the residual that subtracts every child: `R(-i)` minus the scored child's reading.

    The two forms differ by exactly the Primary's own reading (S1, S6), so the all-children ratio
    of a change equals the leave-out ratio minus 1.
    """
    return leave_out - reading


# ---------------------------------------------------------------------------------------------
# Whole-day bootstrap of least squares through the origin
# ---------------------------------------------------------------------------------------------
def day_weights(*, n_days: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Return the number of times each calendar day is drawn in each replicate, `(n_boot, n_days)`.

    One draw serves every regression of a BSP, so the children share their resampled days.
    """
    return rng.multinomial(n_days, np.full(n_days, 1.0 / n_days), size=n_boot).astype(np.float64)


@dataclass(frozen=True)
class Fit:
    """A regression through the origin with a whole-day bootstrap.

    Attributes:
        names: The regressors' names.
        coef: The point estimates.
        se: The bootstrap standard errors.
        lo: The lower ends of the 95% intervals.
        hi: The upper ends of the 95% intervals.
        n_rows: Rows used.
        n_days: Days with at least one row.
    """

    names: tuple[str, ...]
    coef: np.ndarray
    se: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n_rows: int
    n_days: int

    def get(self, name: str) -> tuple[float, float, float, float]:
        """Return the point, standard error, lower end, and upper end of one regressor."""
        i = self.names.index(name)
        return float(self.coef[i]), float(self.se[i]), float(self.lo[i]), float(self.hi[i])


def _solve(*, xtx: np.ndarray, xty: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", np.linalg.pinv(xtx), xty)


def subset_weights(*, weights: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Restrict the master draw to the days present in a regression, rescaled to their number.

    The master draw `weights` resamples every calendar day of the grid. A regression that holds
    only some days keeps those days' draw counts and rescales each replicate so that the counts
    sum to the number of days present. Least squares through the origin is unchanged by a scale of
    the weights, so each replicate is a bootstrap of the present days, and the days are still
    shared with every other regression of the same master draw. A replicate that draws none of the
    present days gets weights of zero and a NaN estimate.
    """
    sub = weights[:, present]
    total = sub.sum(axis=1, keepdims=True)
    return np.divide(sub * len(present), total, out=np.zeros_like(sub), where=total > 0)


def bootstrap_ols(
    *,
    y: np.ndarray,
    x: np.ndarray,
    day: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
) -> Fit:
    """Fit `y` on the columns of `x` through the origin and bootstrap by whole days.

    Args:
        y: The response, `(n,)`.
        x: The regressors, `(n, p)` (a 1-D array is one regressor).
        day: The calendar day of each row, an integer in `[0, weights.shape[1])`.
        weights: The draw counts from `day_weights`.
        names: One name per regressor.

    Returns:
        The fit. All fields are NaN when fewer than `p + 1` rows or 2 days have values.
    """
    x2 = x.reshape(len(y), -1)
    p = x2.shape[1]
    ok = np.isfinite(y) & np.isfinite(x2).all(axis=1)
    yy, xx, dd = y[ok], x2[ok], day[ok]
    present, compact = np.unique(dd, return_inverse=True)
    n_days = int(present.size)
    if ok.sum() < p + 1 or n_days < 2:
        nan = np.full(p, np.nan)
        return Fit(tuple(names), nan, nan, nan, nan, int(ok.sum()), n_days)
    stats = np.zeros((n_days, p * p + p))
    col = 0
    for a in range(p):
        for b in range(p):
            stats[:, col] = np.bincount(compact, weights=xx[:, a] * xx[:, b], minlength=n_days)
            col += 1
    for a in range(p):
        stats[:, col] = np.bincount(compact, weights=xx[:, a] * yy, minlength=n_days)
        col += 1
    total = stats.sum(axis=0)
    point = _solve(xtx=total[: p * p].reshape(p, p), xty=total[p * p :])
    boot = subset_weights(weights=weights, present=present) @ stats
    est = _solve(xtx=boot[:, : p * p].reshape(-1, p, p), xty=boot[:, p * p :])
    lo, hi = np.percentile(est, INTERVAL_PERCENTILES, axis=0)
    return Fit(tuple(names), point, est.std(axis=0), lo, hi, int(ok.sum()), n_days)


def weighted_auc(*, score: np.ndarray, label: np.ndarray, w: np.ndarray) -> float:
    """Return the weighted area under the curve with ties counted half; `label` 1 is the positive."""
    uniq, group = np.unique(score, return_inverse=True)
    pos = np.bincount(group, weights=w * (label == 1), minlength=len(uniq))
    neg = np.bincount(group, weights=w * (label == 0), minlength=len(uniq))
    if pos.sum() <= 0 or neg.sum() <= 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def bootstrap_auc(
    *, score: np.ndarray, label: np.ndarray, day: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    """Return the area under the curve, its bootstrap standard error, and its 95% interval.

    The replicates resample whole days with the draw counts of `day_weights`, restricted to the
    days present as in `subset_weights`. The positive class is `label == 1`.
    """
    ok = np.isfinite(score) & np.isfinite(label)
    s, lab, dd = score[ok], label[ok].astype(int), day[ok]
    point = weighted_auc(score=s, label=lab, w=np.ones(len(s)))
    if not np.isfinite(point):
        return point, float("nan"), float("nan"), float("nan")
    present, compact = np.unique(dd, return_inverse=True)
    sub = subset_weights(weights=weights, present=present)
    boots = np.array(
        [weighted_auc(score=s, label=lab, w=sub[b][compact]) for b in range(sub.shape[0])]
    )
    lo, hi = np.nanpercentile(boots, INTERVAL_PERCENTILES)
    return point, float(np.nanstd(boots)), float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# Bars as functions
# ---------------------------------------------------------------------------------------------
def detection_limit(*, se: float) -> float:
    """Return 2.8 bootstrap standard errors."""
    return SE_MULT * se


def is_void(*, se: float, max_limit: float) -> bool:
    """Return True when the detection limit is not under `max_limit` (a NaN limit is void)."""
    limit = detection_limit(se=se)
    return not bool(np.isfinite(limit) and limit < max_limit)


def interval_within(*, lo: float, hi: float, band: tuple[float, float]) -> bool:
    """Return True when the whole interval lies inside the closed band."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo >= band[0] and hi <= band[1])


def interval_excludes(*, lo: float, hi: float, value: float) -> bool:
    """Return True when `value` lies outside the closed interval."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and (value < lo or value > hi))


def interval_includes(*, lo: float, hi: float, value: float) -> bool:
    """Return True when `value` lies inside the closed interval."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= value <= hi)


def engine_bar(
    *, lo: float, hi: float, se: float, shifted: Mapping[int, tuple[float, float]]
) -> Outcome:
    """Return the outcome of the engine record test (spec 8.4).

    The test is void over its detection limit. It passes when the slope interval excludes 0 and
    lies within -1.3 to -0.7, and each circular shift's interval includes 0.
    """
    if is_void(se=se, max_limit=ENGINE_DETECT_MAX):
        return "void"
    main = interval_excludes(lo=lo, hi=hi, value=ZERO) and interval_within(
        lo=lo, hi=hi, band=ENGINE_SLOPE_BAND
    )
    placebos = all(
        interval_includes(lo=a, hi=b, value=ZERO) for a, b in (shifted[k] for k in sorted(shifted))
    )
    return "pass" if main and placebos and bool(shifted) else "fail"


def d15_closure_bar(*, lo: float, hi: float, se: float) -> Outcome:
    """Return the outcome of the D15 closure test: the interval lies within 0.7 to 1.3."""
    if is_void(se=se, max_limit=D15_DETECT_MAX):
        return "void"
    return "pass" if interval_within(lo=lo, hi=hi, band=D15_CLOSURE_BAND) else "fail"


def d15_sign_bar(*, auc: float, n_hours: int) -> Outcome:
    """Return the outcome of the D15 sign test: at least 200 hours, and an area of at least 0.85."""
    if n_hours < D15_MIN_HOURS or not np.isfinite(auc):
        return "void"
    return "pass" if auc >= D15_AUC_MIN else "fail"


def d15_placebo_bar(*, sign_outcome: Outcome, lo: float, hi: float) -> Outcome:
    """Return the outcome of the D15 placebo: the interval of the area includes 0.5.

    The placebo is void whenever the sign test is void, so a void test is never read as a pass.
    """
    if sign_outcome == "void":
        return "void"
    return "pass" if interval_includes(lo=lo, hi=hi, value=AUC_NULL) else "fail"


def verdict_u2a(*, outcomes: Mapping[str, Outcome]) -> Verdict:
    """Return the U2a verdict (F2).

    `not supported` when any readable test fails; `supported` when at least one test is readable
    and every readable test and its placebo pass; `inconclusive` when no test is readable.
    """
    values = list(outcomes.values())
    if "fail" in values:
        return "not supported"
    if "pass" in values:
        return "supported"
    return "inconclusive"


def calibration_bar(*, lo: float, se: float) -> Outcome:
    """Return the outcome of D12's calibration fit: the lower end is above +0.5."""
    if is_void(se=se, max_limit=U2B_DETECT_MAX):
        return "void"
    return "pass" if np.isfinite(lo) and lo > CALIBRATION_LOWER_ABOVE else "fail"


def reverse_flow_test(*, lo: float, hi: float, se: float) -> TestOutcome:
    """Return what D12's test fit says: an upper end below -0.5 supports reverse flow.

    A lower end above +0.5 refutes reverse flow, and any other interval is unresolved. The test
    is void over its detection limit.
    """
    if is_void(se=se, max_limit=U2B_DETECT_MAX):
        return "void"
    if hi < TEST_REVERSE_UPPER_BELOW:
        return "supports_reverse_flow"
    if lo > TEST_REFUTE_LOWER_ABOVE:
        return "refutes_reverse_flow"
    return "unresolved"


def positive_control_bar(*, lo: float, se: float) -> Outcome:
    """Return the outcome of a control whose interval must lie above 0 (D3, and the BSP B four)."""
    if is_void(se=se, max_limit=U2B_DETECT_MAX):
        return "void"
    return "pass" if np.isfinite(lo) and lo > ZERO else "fail"


def shifted_date_bar(*, lo: float, hi: float, se: float) -> Outcome:
    """Return the outcome of the shifted-date control: the interval includes 0."""
    if is_void(se=se, max_limit=U2B_DETECT_MAX):
        return "void"
    return "pass" if interval_includes(lo=lo, hi=hi, value=ZERO) else "fail"


def verdict_u2b(
    *, calibration: Outcome, test: TestOutcome, controls: Mapping[str, Outcome]
) -> Verdict:
    """Return the U2b verdict.

    `not supported` when the calibration bar or a control fails or the test refutes reverse flow.
    Otherwise `inconclusive` when a fit is void or the test is unresolved, and `supported` when
    the calibration bar, the reverse-flow bar, and every control pass. An explicit failure takes
    precedence over a void fit, because a failed bar closes the rung whatever else is void.
    """
    if calibration == "fail" or test == "refutes_reverse_flow" or "fail" in controls.values():
        return "not supported"
    if (
        calibration == "pass"
        and test == "supports_reverse_flow"
        and all(v == "pass" for v in controls.values())
    ):
        return "supported"
    return "inconclusive"


def verdict_u1(*, audit_ok: bool, alignment_ok: bool, n_pairs: int, bars_pass: bool) -> Verdict:
    """Return the U1 verdict from section 8.5 (a pure rule; the scoring job is not in this slice).

    `invalid` when the class audit or the alignment check fails; `inconclusive` with fewer than
    four same-sky pairs; `supported` when every bar passes; `not supported` otherwise.
    """
    if not (audit_ok and alignment_ok):
        return "invalid"
    if n_pairs < MIN_SAME_SKY_PAIRS:
        return "inconclusive"
    return "supported" if bars_pass else "not supported"


def all_outcome_combinations(*, names: Sequence[str]) -> list[dict[str, Outcome]]:
    """Return every assignment of `pass`, `fail`, and `void` to the named tests."""
    values: tuple[Outcome, ...] = ("pass", "fail", "void")
    return [dict(zip(names, combo, strict=True)) for combo in product(values, repeat=len(names))]


# ---------------------------------------------------------------------------------------------
# Add-back (F1)
# ---------------------------------------------------------------------------------------------
def classify_addback(*, lo: float, hi: float) -> AddbackLabel:
    """Classify a Primary-on-generator coefficient interval by the bands of F1."""
    if interval_within(lo=lo, hi=hi, band=ADDBACK_NULL_BAND):
        return "added_back_or_not_below"
    if interval_within(lo=lo, hi=hi, band=EMBEDDED_BAND):
        return "embedded_not_added_back"
    return "unresolved"


def generator_class(*, labels: Sequence[AddbackLabel]) -> GeneratorClass:
    """Return a generator's class from its labels at every Primary and era under the same BSP.

    A generator at -1 on any Primary is embedded below that Primary and not added back. A
    generator at 0 on every Primary is outside every Primary or added back, and both leave the
    same remainder. Any other pattern is unresolved.
    """
    if "embedded_not_added_back" in labels:
        return "embedded_not_added_back"
    if labels and all(label == "added_back_or_not_below" for label in labels):
        return "outside_or_added_back"
    return "unresolved"


def subtract_generator(
    *, cls: GeneratorClass, units: str, coverage: float, is_engine: bool
) -> bool:
    """Return True when a generator's output enters a residual.

    A generator enters only when it is outside every Primary or added back, is metered in MW (an
    unsigned series cannot be subtracted as if signed), covers at least `MIN_GENERATOR_COVERAGE`
    of the grid, and is not the engine record's own series.
    """
    return (
        cls == "outside_or_added_back"
        and units == "MW"
        and coverage >= MIN_GENERATOR_COVERAGE
        and not is_engine
    )


@dataclass(frozen=True)
class BelowCount:
    """The count of half-hours where the reading minus the engine record is below -2 points."""

    n_below: int
    n_total: int
    share: float
    form_hint: str


def below_count(*, reading_mw: np.ndarray, engine_mw: np.ndarray, capacity_mw: float) -> BelowCount:
    """Count half-hours with `reading - engine` under -2 points of capacity (F1).

    Form `|X + A|` allows such half-hours and form `|X| + A` forbids them, so a positive count
    rules out `|X| + A`. A zero count does not choose between the forms.
    """
    ok = np.isfinite(reading_mw) & np.isfinite(engine_mw)
    points = 100.0 * (reading_mw[ok] - engine_mw[ok]) / capacity_mw
    n_below = int((points < ENGINE_BELOW_POINTS).sum())
    n_total = int(ok.sum())
    share = n_below / n_total if n_total else float("nan")
    hint = "rules_out_abs_x_plus_a" if n_below > 0 else "no_discrimination"
    return BelowCount(n_below=n_below, n_total=n_total, share=share, form_hint=hint)


def dominance_band(*, largest_farm_mw: float, total_pv_mw: float | None) -> str:
    """Return `below_a_third`, `a_third_or_more`, or `unknown` for F7's share, never a capacity."""
    if total_pv_mw is None or not np.isfinite(total_pv_mw) or total_pv_mw <= 0:
        return "unknown"
    return (
        "a_third_or_more"
        if largest_farm_mw / total_pv_mw >= DOMINANT_FARM_SHARE
        else "below_a_third"
    )


# ---------------------------------------------------------------------------------------------
# D15 hourly sign statistic
# ---------------------------------------------------------------------------------------------
def robust_scale(*, x: np.ndarray) -> float:
    """Return the median-absolute-deviation noise scale, `1.4826 * MAD`, of the finite values."""
    v = x[np.isfinite(x)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def hourly_sign_rows(
    *,
    d_residual: np.ndarray,
    d_folded: np.ndarray,
    flow: np.ndarray,
    hour_end: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the scorable-hour table of the D15 sign test.

    Args:
        d_residual: The hourly change of the leave-out residual (de-meaned), on the grid.
        d_folded: The hourly change of the folded reading `m = |X|` (de-meaned), on the grid.
        flow: The true signed flow `X` on the grid (the truth only).
        hour_end: True on half-hours that end an hour (the non-overlapping hours).

    Returns:
        Arrays over grid rows: `scorable` (size threshold met, crossing hours excluded), `size_ok`,
        `crossing`, `label` (1 import, 0 export, NaN crossing), and `ratio`.
    """
    scale = robust_scale(x=d_residual)
    flow_before = shift_rows(x=flow, rows=HOURLY_LAG_HALF_HOURS)
    flow_mid = shift_rows(x=flow, rows=1)
    finite = hour_end & np.isfinite(d_residual) & np.isfinite(d_folded) & np.isfinite(flow)
    finite &= np.isfinite(flow_before) & np.isfinite(flow_mid)
    size_ok = finite & (np.abs(d_folded) >= D15_SIZE_NOISE_MULT * scale)
    signs = np.stack([np.sign(flow_before), np.sign(flow_mid), np.sign(flow)])
    crossing = finite & ~((signs == signs[0]).all(axis=0) & (signs[0] != 0))
    label = np.where(flow > 0, 1.0, 0.0)
    label[~finite | crossing] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(d_folded) > 0, d_residual / d_folded, np.nan)
    return {
        "scorable": size_ok & ~crossing,
        "size_ok": size_ok,
        "crossing": size_ok & crossing,
        "finite": finite,
        "label": label,
        "ratio": ratio,
        "scale": np.full(len(flow), scale),
    }
