# ruff: noqa: E501
"""Experiment 723: the named constants, the bars, the permutation rule, and the verdict rules.

Every threshold of the experiment is a `Final` constant of this module, set before any real data
are read, and every value is the one the plan page (`result.md`, section "Decision rule") states.
The functions are pure (numbers and arrays in, outcomes out), so the tests run them on synthetic
inputs. Nothing here reads a file.

**Provisional definitions.** A few definitions that a job needs are not stated on the plan page.
Each is a constant or a function marked `PROVISIONAL` and listed in the report that accompanies the
code, so that the maintainer or the manager can rule on it before the first run.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import e00722_core as core722  # ty: ignore[unresolved-import]
import numpy as np

# ---------------------------------------------------------------------------------------------
# Constants copied from experiment 722 (imported, never redefined)
# ---------------------------------------------------------------------------------------------
HALF_HOURS_PER_DAY: Final[int] = core722.HALF_HOURS_PER_DAY
"""Half-hours in a UTC day."""

N_GRID: Final[int] = core722.N_GRID
"""Half-hours whose end stamp is before the holdout of 2026-07-01."""

N_BOOT: Final[int] = 1000
"""Bootstrap draws that resample whole calendar days."""

SEED: Final[int] = 20261124
"""The fixed seed of the bootstrap."""

TWO_TRANSFORMERS_2025: Final[tuple[str, ...]] = core722.TWO_TRANSFORMERS_2025
"""The two signed transformers whose history starts in 2025."""

DAYLIGHT_ELEVATION_DEG: Final[float] = core722.DAYLIGHT_ELEVATION_DEG
"""A daylight half-hour has a sun elevation of at least this many degrees."""

# ---------------------------------------------------------------------------------------------
# Named constants of the plan page ("Named constants", in the decision rule)
# ---------------------------------------------------------------------------------------------
AREA_BAR: Final[float] = 0.85
"""Pooled area under the receiver operating characteristic curve: the point estimate must reach this."""

AREA_BAR_LEVEL_1: Final[float] = 0.80
"""The point estimate at injection level 1.0 alone must reach this."""

AREA_LOWER_BAR: Final[float] = 0.75
"""The lower end of the 95% interval of the pooled area must exceed this."""

PLACEBO_GAP_BAR: Final[float] = 0.15
"""The real sensor's area minus the area of the sensor shifted by 28 days must reach this."""

MAX_POOLED_SE: Final[float] = 0.036
"""The bootstrap standard error of the pooled area must not exceed this."""

MAX_TRANSFER_SE: Final[float] = 0.05
"""The bootstrap standard error of the transfer share must not exceed this."""

SIZE_OF_SKY_INDEX_CHANGE: Final[float] = 0.10
"""A pair is usable when the farm sensor's index changes by at least this much."""

SIZE_OF_READING_CHANGE_POINTS: Final[float] = 0.2
"""A pair is usable when the reading's fast change is at least this many points (1% of capacity)."""

DEAD_BAND: Final[float] = 0.25
"""`C` above +0.25 is export evidence, `C` below -0.25 is import evidence."""

MIN_USABLE_PAIRS: Final[int] = 8
"""A stretch with fewer usable pairs is not scorable and gives no evidence."""

HALF: Final[float] = 0.5
"""The coherence floor, the transfer share, and the recall bar."""

FLOAT_SLACK: Final[float] = 1e-12
"""Guards a comparison at a bar against floating-point rounding (0.75 * 0.6 is 0.44999999999999996); it moves no bar."""

POWERED_PAIR_NEED: Final[float] = 0.45
"""A pair and level is powered when the coherence times the dilution is at least this."""

ARC_REMOVAL_MAX: Final[float] = 0.1
"""The correlation of a farm's `dZ~` with `d u_clear` has a size of at most this."""

ARC_VOID_SHARE: Final[float] = 0.01
"""The arc term is void if the share of hour-ending half-hours on which `u_clear` does not change exceeds this."""

D15_NO_EVIDENCE_SHARE: Final[float] = 0.90
"""At least this share of D15's stretches gives no evidence, where the bar binds."""

FALSE_EVIDENCE_SHARE: Final[float] = 0.02
"""The share of scorable stretches that may give false export evidence (the floor rule)."""

MIN_PAIRS: Final[int] = 4
"""The minimum number of qualifying pairs, and of powered pairs at level 2.0."""

MIN_HOSTS: Final[int] = 2
"""The powered pairs sit on at least this many hosts."""

MIN_STRETCHES: Final[int] = 30
"""The minimum number of scorable stretches for most bars."""

MIN_TRANSFER_STRETCHES: Final[int] = 100
"""The transfer check is void under this many stretches."""

N_LATER_DAYS: Final[int] = 413
"""The calendar days of the later span, which job W0 asserts."""

N_WORLDS: Final[int] = 32
"""The worlds of job W2 in the timing gate's projection."""

CAP_CORE_HOURS: Final[float] = 1.0
"""The cap of each job."""

CAP_CORE_SECONDS: Final[float] = CAP_CORE_HOURS * 3600.0
"""The cap of each job in core-seconds."""

CALIBRATION_BIN_MIN: Final[int] = 100
"""A reliability bin of at least this many stretches is checked."""

CALIBRATION_TOLERANCE: Final[float] = 0.1
"""No bin of at least 100 stretches may be off the diagonal by more than this."""

SE_FACTOR: Final[float] = 1.25
"""The predicted standard error is this factor times the naive figure (hour-to-hour changes only)."""

DETECTION_SE_MULT: Final[float] = 2.8
"""The detection limit in bootstrap standard errors."""

PLACEBO_SHIFT_DAYS: Final[int] = 28
"""The placebo shifts the sensor by this many days."""

TRAILING_CAPACITY_DAYS: Final[int] = 91
"""The capacity window of the farm sensor's index."""

TRANSFER_MARGIN_HALF_HOURS: Final[int] = 2
"""A change within 1 hour (2 half-hours) of a switching window is masked."""

LEVELS: Final[tuple[float, float]] = (1.0, 2.0)
"""The injection levels."""

REACTIVE_FLOOR_POINTS: Final[float] = 2.0
"""The reactive floor of the folded readings, in points of capacity."""

CAPACITY_P95: Final[float] = 95.0
"""The percentile of `Z / u_clear` that sets the farm sensor's capacity `c_Z`."""

TEST_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""April to September, the months of the coherence condition and of job W3's read-outs."""

POST_FIT_A_BAND: Final[tuple[float, float]] = (0.6, 1.5)
"""Fit A reads "same-sky generation outside D15 explains the excess" when the interval lies inside."""

POST_FIT_A_HIGH: Final[float] = 2.0
"""Fit A reads "add-back or generation the farm sensor does not follow" when the lower end exceeds this."""

POST_FIT_B_COEF_BAND: Final[tuple[float, float]] = (-1.0, 0.0)
"""Fit B supports an add-back when the farm's megawatt coefficient lies in this band."""

D10_ABSENT_AREA: Final[float] = 0.5
"""A blind meter-sensor area below this at folded D10 contradicts the corrected identity."""

REACTIVE_SUBSTEPS: Final[int] = 6
"""PROVISIONAL: sub-steps of the within-half-hour interpolation (design section 2.5 says "for example 6")."""

OVERCAST_LOOKBACK_DAYS: Final[tuple[int, int]] = (2, 91)
"""The trailing ages, in days, of the overcast days whose median change is removed from the reading."""

MIDDAY_UTC_HOURS: Final[tuple[float, float]] = (11.0, 14.0)
"""PROVISIONAL: the UTC half-open interval of "midday" for the injection level's reference rows."""

SUMMER_MONTHS: Final[tuple[int, ...]] = (6, 7, 8)
"""PROVISIONAL: "summer" for the injection level's reference rows."""

SUNNY_MIN_K: Final[float] = core722.CLEAR_MIN_K
"""PROVISIONAL: "sunny" is a clear-sky index of at least the clear-day threshold of experiment 722."""

Outcome = Literal["pass", "fail", "void"]
Verdict = Literal["supported", "partly supported", "not supported", "inconclusive", "invalid"]
Evidence = Literal["export", "import", "none"]
Truth = Literal["export", "import", "mixed"]


def floor_allowance(*, n: int) -> int:
    """Return `floor(0.02 * n)`, the number of false export stretches the floor rule allows."""
    return math.floor(FALSE_EVIDENCE_SHARE * n + 1e-9)


def evidence_of(*, c: float, n_usable: int) -> Evidence:
    """Classify a stretch by its concordance: export above +0.25, import below -0.25, else none."""
    if n_usable < MIN_USABLE_PAIRS or not np.isfinite(c):
        return "none"
    if c > DEAD_BAND:
        return "export"
    if c < -DEAD_BAND:
        return "import"
    return "none"


def truth_of(*, share_export_half_hours: float) -> Truth:
    """Return the truth label of a stretch from its share of half-hours with `X < 0`.

    Export when at least 80% of the stretch's half-hours have `X < 0`, import when at most 20% do,
    and mixed otherwise (version 2, section 6.3).
    """
    if share_export_half_hours >= 0.8:
        return "export"
    if share_export_half_hours <= 0.2:
        return "import"
    return "mixed"


def detection_limit(*, se: float) -> float:
    """Return 2.8 bootstrap standard errors."""
    return DETECTION_SE_MULT * se


def predicted_se(*, sd_response: float, sd_regressor: float, n_rows: int) -> float:
    """Return the predicted standard error: 1.25 times `sd(y) / (sd(x) * sqrt(n))`.

    The factor applies to hour-to-hour changes only.
    """
    if n_rows <= 0 or sd_regressor <= 0:
        return float("nan")
    return SE_FACTOR * sd_response / (sd_regressor * math.sqrt(n_rows))


def dilution(*, sd_injected: float, sd_host: float) -> float:
    """Return `d = s_F / sqrt(s_F^2 + s_H^2)`."""
    denom = math.hypot(sd_injected, sd_host)
    return sd_injected / denom if denom > 0 else float("nan")


def is_powered(*, coherence: float, dilution_value: float) -> bool:
    """Return True when the injected farm's coherence times the dilution is at least 0.45."""
    value = coherence * dilution_value
    return bool(np.isfinite(value) and value >= POWERED_PAIR_NEED - FLOAT_SLACK)


def expected_concordance(*, rho: float) -> float:
    """Return `(2 / pi) * arcsin(rho)`, the expected concordance of jointly normal changes."""
    return 2.0 / math.pi * math.asin(rho)


def arc_removal_ok(*, r: float) -> bool:
    """Return True when the size of a farm's correlation with `d u_clear` is at most 0.1."""
    return bool(np.isfinite(r) and abs(r) <= ARC_REMOVAL_MAX)


def coherence_ok(*, r: float) -> bool:
    """Return True when a farm's coherence with the other farms of a host's sensor is at least 0.5."""
    return bool(np.isfinite(r) and r >= HALF)


# ---------------------------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------------------------
def false_export_bar(*, n_export_evidence: int, n_scorable_import: int) -> Outcome:
    """Return the outcome of the false-export bar (and of the two controls of job W3).

    The bar passes when at most `floor(0.02 * n)` of the `n` scorable true import stretches give
    export evidence, and is void under 30 scorable stretches.
    """
    if n_scorable_import < MIN_STRETCHES:
        return "void"
    return "pass" if n_export_evidence <= floor_allowance(n=n_scorable_import) else "fail"


def transfer_bar(*, share_import_evidence: float, n_stretches: int, se: float) -> Outcome:
    """Return the outcome of the transfer check on real import stretches.

    At least 50% of true import stretches have `C` below -0.25. The bar is void under 100
    stretches or with a bootstrap standard error above 0.05.
    """
    if n_stretches < MIN_TRANSFER_STRETCHES or not np.isfinite(se) or se > MAX_TRANSFER_SE:
        return "void"
    return "pass" if share_import_evidence >= HALF else "fail"


def dead_band_bar(
    *, n_no_evidence: int, n_stretches: int, binds: bool
) -> Outcome | Literal["report"]:
    """Return the outcome of the D15 dead-band bar.

    The bar is void under 30 stretches (every stretch counts, scorable or not), `report` when it
    does not bind, and otherwise passes at a share of at least 90%.
    """
    if not binds:
        return "report"
    if n_stretches < MIN_STRETCHES:
        return "void"
    return "pass" if n_no_evidence / n_stretches >= D15_NO_EVIDENCE_SHARE else "fail"


def d15_dead_band_binds(*, farm_class: str | None) -> bool:
    """Return True when the farm inside D15's polygon is classed "outside every metered Primary".

    PROVISIONAL: no farm inside the polygon, or an embedded or unresolved class, makes the bar
    report-only.
    """
    return farm_class == "outside_or_added_back"


def recall_bar(*, n_export_evidence: int, n_scorable_export: int) -> Outcome:
    """Return the outcome of the recall bar on D10's real export stretches.

    The bar is void under 30 scorable stretches (the verdict then rests on the other bars), and
    otherwise needs a recall of at least 0.5.
    """
    if n_scorable_export < MIN_STRETCHES:
        return "void"
    return "pass" if n_export_evidence / n_scorable_export >= HALF else "fail"


def placebo_bar(*, real_area: float, shifted_area: float) -> Outcome:
    """Return the outcome of the shifted-sensor bar: the real area exceeds the shifted by at least 0.15."""
    gap = real_area - shifted_area
    if not np.isfinite(gap):
        return "void"
    return "pass" if gap >= PLACEBO_GAP_BAR else "fail"


def area_bar(*, point: float, lo: float, point_level_1: float) -> Outcome:
    """Return the outcome of the pooled area bar.

    The point estimate must reach 0.85, the lower end of the 95% interval must exceed 0.75, and
    the point estimate at level 1.0 alone must reach 0.80.
    """
    if not all(np.isfinite(v) for v in (point, lo, point_level_1)):
        return "void"
    ok = point >= AREA_BAR and lo > AREA_LOWER_BAR and point_level_1 >= AREA_BAR_LEVEL_1
    return "pass" if ok else "fail"


def calibration_bar(
    *, predicted: np.ndarray, truth_export: np.ndarray, n_bins: int = 10
) -> Outcome:
    """Return the outcome of the calibration bar on stretches with a predicted export probability.

    The predicted probabilities fall into `n_bins` equal bins on 0 to 1. No bin of at least 100
    stretches may sit off the diagonal (its mean prediction against its export share) by more than
    0.1. The bar is void when no bin holds 100 stretches. The page does not define the likelihood
    ratio or the probability built from it, so job W2 supplies none and reports the bar void
    (PROVISIONAL, see the report).
    """
    ok = np.isfinite(predicted) & np.isfinite(truth_export)
    p, y = predicted[ok], truth_export[ok]
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    checked = False
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.sum() >= CALIBRATION_BIN_MIN:
            checked = True
            if abs(float(p[in_bin].mean()) - float(y[in_bin].mean())) > CALIBRATION_TOLERANCE:
                return "fail"
    return "pass" if checked else "void"


# ---------------------------------------------------------------------------------------------
# Verdict of job W2
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class W2Inputs:
    """What the verdict rule of job W2 reads.

    Attributes:
        correctness_ok: False when the holdout guard, the day count of ruling 2, or a known-answer
            test on synthetic arrays failed.
        alignment_ok: True when the cross-correlation peaks at lag 0.
        timestamp_defect: True when a failing test shows a time-stamp defect.
        n_pairs_keeping_sensor: Qualifying pairs whose host keeps a sensor after the audit.
        n_powered_pairs_level_2: Pairs powered at level 2.0.
        n_powered_hosts: Hosts that hold a powered pair.
        pooled_se: Bootstrap standard error of the pooled area.
        n_export_stretches: Scorable export stretches in the pool.
        n_import_stretches: Scorable import stretches in the pool.
        level_1_readable: False when either class at level 1.0 holds under 30 scorable stretches.
        outcomes: The outcome of each bar by name: `area`, `placebo`, `false_export`, `transfer`,
            `d15_dead_band`, `recall`, and `calibration`.
    """

    correctness_ok: bool
    alignment_ok: bool
    timestamp_defect: bool
    n_pairs_keeping_sensor: int
    n_powered_pairs_level_2: int
    n_powered_hosts: int
    pooled_se: float
    n_export_stretches: int
    n_import_stretches: int
    level_1_readable: bool
    outcomes: Mapping[str, str]


W2_VERDICT_ROWS: Final[tuple[str, ...]] = (
    "row_1_invalid",
    "row_2_void_before_run",
    "row_3_not_readable",
    "row_4_supported",
    "row_5_partly_supported",
    "row_6_not_supported",
)
"""The rows of the verdict table, in the order in which the first match gives the verdict."""


def void_before_run(
    *, n_pairs_keeping_sensor: int, n_powered_pairs_level_2: int, n_powered_hosts: int
) -> bool:
    """Return True when job W2 is `inconclusive (void before run)` and builds no world."""
    return (
        n_pairs_keeping_sensor < MIN_PAIRS
        or n_powered_pairs_level_2 < MIN_PAIRS
        or n_powered_hosts < MIN_HOSTS
    )


def verdict_w2(*, x: W2Inputs) -> tuple[Verdict, str]:
    """Return the verdict of job W2 and the row that gave it; the first matching row wins.

    Row 1, `invalid`: a correctness check fails, or the alignment check fails together with a
    time-stamp defect. Row 2, `inconclusive` (void before run). Row 3, `inconclusive` (not
    readable): the pooled standard error exceeds 0.036, either class holds under 30 scorable
    stretches, level 1.0 is not readable, the false-export bar is void, or the alignment check
    fails and no time-stamp defect is found. Row 4, `supported`: every binding bar passes. Row 5,
    `partly supported`: every binding bar passes except the transfer check (failed or void), the
    D15 dead band where it binds, or both. Row 6, `not supported`: otherwise.
    """
    o = x.outcomes
    if not x.correctness_ok or (not x.alignment_ok and x.timestamp_defect):
        return "invalid", "row_1_invalid"
    if void_before_run(
        n_pairs_keeping_sensor=x.n_pairs_keeping_sensor,
        n_powered_pairs_level_2=x.n_powered_pairs_level_2,
        n_powered_hosts=x.n_powered_hosts,
    ):
        return "inconclusive", "row_2_void_before_run"
    unreadable = (
        not np.isfinite(x.pooled_se)
        or x.pooled_se > MAX_POOLED_SE
        or x.n_export_stretches < MIN_STRETCHES
        or x.n_import_stretches < MIN_STRETCHES
        or not x.level_1_readable
        or o.get("false_export") == "void"
        or o.get("area") == "void"
        or not x.alignment_ok
    )
    if unreadable:
        return "inconclusive", "row_3_not_readable"
    # Bars that always bind; a void recall or calibration bar is neutral.
    binding = [o.get("area"), o.get("placebo"), o.get("false_export")]
    binding += [v for v in (o.get("recall"), o.get("calibration")) if v != "void"]
    if any(v != "pass" for v in binding):
        return "not supported", "row_6_not_supported"
    transfer_ok = o.get("transfer") == "pass"
    dead_band_ok = o.get("d15_dead_band") in ("pass", "report", "void")
    if transfer_ok and dead_band_ok:
        return "supported", "row_4_supported"
    return "partly supported", "row_5_partly_supported"


def verdict_w3(*, control_outcomes: Mapping[str, str], masked: bool) -> tuple[str, str]:
    """Return job W3's verdict and the reason; the first matching row wins.

    (1) A control read on 30 or more scorable stretches fails: `a control fails, the hint is
    withdrawn at D12`, labelled "transfers not masked" when the switching logs gave no mask. (2) A
    control is void, or the switching logs gave no mask: `void`. (3) Otherwise: `controls pass,
    the read-outs stand`.
    """
    if any(v == "fail" for v in control_outcomes.values()):
        label = "a control fails, the hint is withdrawn at D12"
        return (label if masked else label + " (transfers not masked)"), "row_1"
    if any(v == "void" for v in control_outcomes.values()) or not masked:
        return "void", "row_2"
    return "controls pass, the read-outs stand", "row_3"


def project_w2_seconds(
    *, seconds_per_world: float, seconds_per_draw: float, n_worlds: int = N_WORLDS
) -> float:
    """Return `seconds per world * 32 worlds + seconds per draw * 1,000`, the timing gate's projection.

    `n_worlds` is the number of worlds the job will score, which is 32 when 12 pairs qualify; a
    test asserts that the count of worlds scored equals this number.
    """
    return seconds_per_world * n_worlds + seconds_per_draw * N_BOOT


def over_cap(*, projected_seconds: float, cap_seconds: float = CAP_CORE_SECONDS) -> bool:
    """Return True when a projection exceeds the cap."""
    return projected_seconds > cap_seconds


# ---------------------------------------------------------------------------------------------
# Bootstrap (whole calendar days)
# ---------------------------------------------------------------------------------------------
def day_weights(*, n_days: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """Return the draw counts of each calendar day in each replicate, `(n_boot, n_days)`.

    Every pair, level, and host takes its stretches from the same drawn days.
    """
    return core722.day_weights(n_days=n_days, n_boot=n_boot, rng=np.random.default_rng(seed))


def bootstrap_area(
    *, score: np.ndarray, is_export: np.ndarray, day: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float]:
    """Return the pooled area, its bootstrap standard error, and its 95% interval (export is positive)."""
    return core722.bootstrap_auc(
        score=score, label=is_export.astype(float), day=day, weights=weights
    )


def bootstrap_share(
    *, hit: np.ndarray, day: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Return a share of stretches and its whole-day bootstrap standard error."""
    if hit.size == 0:
        return float("nan"), float("nan")
    present, compact = np.unique(day, return_inverse=True)
    sub = core722.subset_weights(weights=weights, present=present)
    num = np.bincount(compact, weights=hit.astype(float), minlength=len(present))
    den = np.bincount(compact, minlength=len(present)).astype(float)
    boots_num, boots_den = sub @ num, sub @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = np.where(boots_den > 0, boots_num / boots_den, np.nan)
    return float(hit.mean()), float(np.nanstd(boots))


# ---------------------------------------------------------------------------------------------
# The permutation rule for future engine records (amendment section 4.2; no job runs it here)
# ---------------------------------------------------------------------------------------------
ENGINE_DETECT_MAX: Final[float] = 0.5
"""The detection-limit gate of the engine rule, retained."""

ENGINE_P_LEVEL: Final[float] = 0.05
"""The level of the one-sided permutation `p`."""


def permutation_p(*, main_slope: float, shifted_slopes: Sequence[float]) -> float:
    """Return `(1 + number of shifted slopes at or below the main) / (1 + number of shifts)`."""
    shifted = np.asarray(shifted_slopes, dtype=float)
    return float((1 + int((shifted <= main_slope).sum())) / (1 + shifted.size))


def weekly_not_interpretable(*, shifted_by_day: Mapping[int, float]) -> bool:
    """Return True when the mean slope of the weekly shifts differs from the others' by over 2 standard errors."""
    weekly = np.array([v for d, v in shifted_by_day.items() if d % 7 == 0])
    other = np.array([v for d, v in shifted_by_day.items() if d % 7 != 0])
    if weekly.size < 1 or other.size < 2:
        return False
    se = float(other.std(ddof=1)) * math.sqrt(1.0 / weekly.size + 1.0 / other.size)
    return bool(se > 0 and abs(float(weekly.mean()) - float(other.mean())) > 2.0 * se)


def engine_rule_outcome(
    *,
    detection_limit_main: float,
    main_slope: float,
    ci: tuple[float, float],
    shifted_by_day: Mapping[int, float],
) -> str:
    """Return `void`, `not interpretable`, `seen`, `seen, size differs`, or `not seen`.

    For a record of future days only. No outcome claims that the size is confirmed.
    """
    if detection_limit_main > ENGINE_DETECT_MAX:
        return "void"
    if weekly_not_interpretable(shifted_by_day=shifted_by_day):
        return "not interpretable"
    p = permutation_p(main_slope=main_slope, shifted_slopes=list(shifted_by_day.values()))
    if p > ENGINE_P_LEVEL:
        return "not seen"
    return "seen" if ci[0] <= -1.0 <= ci[1] else "seen, size differs"


# ---------------------------------------------------------------------------------------------
# Readings of the two D15 fits (read-outs with patterns stated in advance; they license nothing)
# ---------------------------------------------------------------------------------------------
def fit_a_reading(*, lo: float, hi: float) -> str:
    """Return the pre-stated reading of fit A, the slope on D15's change with the farm sensor added.

    An interval inside 0.6 to 1.5 means that same-sky generation outside D15 explains the excess.
    A lower end above 2.0 means an add-back inside D15's series, or generation that the farm
    sensor does not follow. Anything else is unresolved.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unresolved"
    if lo >= POST_FIT_A_BAND[0] and hi <= POST_FIT_A_BAND[1]:
        return "same-sky generation outside D15 explains the excess"
    if lo > POST_FIT_A_HIGH:
        return "add-back inside D15's series, or generation that the farm sensor does not follow"
    return "unresolved"


def fit_b_supports_addback(*, slope: float, farm_coefficient: float) -> bool:
    """Return True when fit B's slope on D15's change lies in 0.6 to 1.5 and the farm's megawatt coefficient in -1 to 0.

    PROVISIONAL: "falls inside" is read as the point estimates.
    """
    return bool(
        np.isfinite(slope)
        and np.isfinite(farm_coefficient)
        and POST_FIT_A_BAND[0] <= slope <= POST_FIT_A_BAND[1]
        and POST_FIT_B_COEF_BAND[0] <= farm_coefficient <= POST_FIT_B_COEF_BAND[1]
    )


def meter_blind_contradicts(*, area: float) -> bool:
    """Return True when the blind meter-sensor area at folded D10 is below 0.5, which contradicts the corrected identity."""
    return bool(np.isfinite(area) and area < D10_ABSENT_AREA)
