# ruff: noqa: E501
"""Experiment 729: every named constant and bar of the plan, the split rule, and the verdict rule.

Every number that the page's decision rule pre-registers is a `Final` here, so the page and the code
are compared by reading this one file. The functions are pure: they read numbers and return numbers,
and they touch no data file (except `write_margin_file`, which refuses to overwrite). No function
prints or logs a month, a date, an engine size, a series identifier, or a real value.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np

# --------------------------------------------------------------------------- identity and rows

ROW_SET_ID: Final = "e00729_synthetic_d8_v1"
"""The row set: the 48 half-hours of each UTC day of the test months, 8 draws each."""
HOLDOUT_START_ISO: Final = "2026-07-01"
"""No valid time on or after this date is read by any code of this experiment."""
HALF_HOURS_PER_DAY: Final = 48
WINDOW_STEPS: Final = 96
"""A window is 48 hours: 12:00 UTC the day before to 12:00 UTC the day after."""
WINDOW_STEPS_24: Final = 48
MIDDLE_START: Final = 24
"""Index in the 96-step window of the first half-hour of the scored day (00:00 UTC)."""
CAPACITY_MVA: Final = 10.379
"""D8's effective capacity in MVA; 1 point is 1% of it."""
POINTS: Final = 100.0
D8_SERIES_ID: Final = 26
BSP_SERIES_ID: Final = 1
"""The time series identifiers of D8 and of the bulk supply point A meter in the power store."""

# --------------------------------------------------------------------------- the split rule

MIN_DAYS_IN_MONTH: Final = 20
MIN_HALF_HOURS_IN_DAY: Final = 40
RULE_TEST: Final = (3, 2)
"""(period, remainder): position p in a group is a test month when p mod period equals remainder."""
RULE_VALIDATION_OTHER: Final = (6, 1)
LOCKDOWN_FIRST_MONTH: Final = "2020-03"
LOCKDOWN_LAST_MONTH: Final = "2021-07"
"""The fixed lockdown window of the data-processing page, as first and last month."""
DROP_MARGIN_DAYS: Final = 2
"""A window spans 48 hours, so training drops every day within 2 days of a test or validation month."""
SplitName = Literal["training", "validation", "test"]
DaySplit = Literal["training", "validation", "test", "dropped"]


def month_splits(
    *, months: Sequence[str], n_days: Sequence[int], bsp_days: Sequence[int]
) -> dict[str, SplitName]:
    """Apply the plan's four-step rule to the base calendar and return each base month's split.

    Args:
        months: Calendar months as `YYYY-MM`, in any order.
        n_days: The number of base days in each month (same order as `months`).
        bsp_days: The number of base days in each month on which the BSP A meter has data.

    Returns:
        A map from each month with at least `MIN_DAYS_IN_MONTH` base days to its split.
    """
    rows = sorted(
        (m, nd, bd)
        for m, nd, bd in zip(months, n_days, bsp_days, strict=True)
        if nd >= MIN_DAYS_IN_MONTH
    )
    result: dict[str, SplitName] = {}
    groups: dict[bool, list[str]] = {True: [], False: []}
    for month, _, bsp in rows:
        if LOCKDOWN_FIRST_MONTH <= month <= LOCKDOWN_LAST_MONTH:
            result[month] = "training"
        else:
            groups[bsp >= MIN_DAYS_IN_MONTH].append(month)
    period, remainder = RULE_TEST
    for has_bsp, members in groups.items():
        for position, month in enumerate(members, start=1):
            if position % period == remainder:
                result[month] = "test"
            elif (has_bsp and position == len(members)) or (
                not has_bsp and position % RULE_VALIDATION_OTHER[0] == RULE_VALIDATION_OTHER[1]
            ):
                result[month] = "validation"
            else:
                result[month] = "training"
    return result


def month_key(*, day: date) -> str:
    """Return `YYYY-MM` of a date."""
    return f"{day.year:04d}-{day.month:02d}"


def assign_day_splits(*, days: Sequence[date], splits: Mapping[str, SplitName]) -> list[DaySplit]:
    """Return each base day's split, with the guard days dropped.

    A training day within `DROP_MARGIN_DAYS` of a test or validation month is dropped, and so is a
    validation day within that margin of a test month. A day in a month that the rule left out (fewer
    than `MIN_DAYS_IN_MONTH` days) is dropped.

    Args:
        days: The base days.
        splits: The output of `month_splits`.

    Returns:
        The split of each day, or "dropped".
    """
    out: list[DaySplit] = []
    for day in days:
        own = splits.get(month_key(day=day))
        if own is None:
            out.append("dropped")
            continue
        near = {
            splits.get(month_key(day=day + timedelta(days=k)))
            for k in range(-DROP_MARGIN_DAYS, DROP_MARGIN_DAYS + 1)
        }
        if (own == "training" and near & {"test", "validation"}) or (
            own == "validation" and "test" in near
        ):
            out.append("dropped")
        else:
            out.append(own)
    return out


# --------------------------------------------------------------------------- the world mix

WORLD_TYPES: Final = ("none", "engine", "solar", "both")
TEST_MIX: Final = {"none": 0.15, "engine": 0.30, "solar": 0.20, "both": 0.35}
"""The fixed world mix of every test and validation bank."""
TEST_MIX_INJECTED: Final = 1.0 - TEST_MIX["none"]
"""The injected share of the test banks (0.85): the prior given to B1 and B2."""
SHIFTED_NONE_SHARE: Final = 0.50
"""The no-injection share of the shifted-mix bank."""
NONE_SHARE_RANGE: Final = (0.05, 0.60)
"""A training bank draws its own no-injection share uniformly from this range."""
OTHER_PROPORTIONS: Final = {"engine": 0.30, "solar": 0.20, "both": 0.35}
SEGMENT_DRAWS_TRAIN: Final = 32
SEGMENT_DRAWS_EVAL: Final = 8
TRAIN_SEED_BASE: Final = 7290
EVAL_SEED_BASE: Final = 729_000
VALIDATION_SEED_BASE: Final = 729_250
"""The validation bank has its own seed space, so that no validation and test segment share a draw."""
SHIFTED_SEED_BASE: Final = 729_500
BSP_SAMPLER_SHARE: Final = 0.5
"""Half of each epoch's windows come from training months that carry real BSP steps."""


def world_probabilities(*, none_share: float) -> dict[str, float]:
    """Return the probability of each world type for a bank with the given no-injection share."""
    rest = 1.0 - none_share
    total = sum(OTHER_PROPORTIONS.values())
    return {"none": none_share} | {k: rest * v / total for k, v in OTHER_PROPORTIONS.items()}


def quota_world_types(*, rng: np.random.Generator, n: int, none_share: float) -> list[str]:
    """Assign the world types of `n` segments by quota, so that the realised mix is the planned mix.

    Each world type gets `n` times its probability, rounded down, and the segments left over go to
    the types with the largest remainders. The assignment is then shuffled.

    Args:
        rng: The generator that shuffles the assignment.
        n: The number of segments.
        none_share: The planned no-injection share.

    Returns:
        The world type of each segment.
    """
    probs = world_probabilities(none_share=none_share)
    exact = {w: n * p for w, p in probs.items()}
    counts = {w: int(np.floor(v + 1e-9)) for w, v in exact.items()}
    left = n - sum(counts.values())
    for w in sorted(exact, key=lambda k: (counts[k] - exact[k], k))[:left]:
        counts[w] += 1
    flat = [w for w, c in counts.items() for _ in range(c)]
    return [flat[i] for i in rng.permutation(len(flat))]


def draw_none_share(*, rng: np.random.Generator) -> float:
    """Draw a training bank's no-injection share from `NONE_SHARE_RANGE`."""
    return float(rng.uniform(*NONE_SHARE_RANGE))


def draw_world_types(*, rng: np.random.Generator, n: int, none_share: float) -> list[str]:
    """Draw the world types of `n` segments."""
    probs = world_probabilities(none_share=none_share)
    return [str(w) for w in rng.choice(list(probs), size=n, p=list(probs.values()))]


# --------------------------------------------------------------------------- simulator ranges

RANGES: Final = {
    "n_units": (1, 4), "unit_points": (10.0, 50.0), "min_unit_points": (0.0, 50.0), "total_points_cap": (0.0, 120.0),
    "no_run_share": (0.2, 0.7), "evening_prob": (0.3, 0.9), "evening_start_h": (15.0, 17.5),
    "evening_hours": (2.0, 6.0), "morning_prob": (0.1, 0.6), "morning_start_h": (5.5, 8.5),
    "morning_hours": (1.0, 8.0), "midday_prob": (0.0, 0.3), "midday_start_h": (10.0, 14.0),
    "midday_hours": (1.0, 4.0), "weekend_factor": (0.1, 0.8), "spells_per_30_days": (0.0, 2.0),
    "spell_days": (2, 19), "unit_output_share": (0.6, 1.0), "ramp_half_hours": (0.0, 2.0),
    "trip_prob": (0.0, 0.02), "trip_half_hours": (1, 4), "solar_capacity_points": (5.0, 150.0),
    "tilt_deg": (10.0, 40.0), "azimuth_deg": (120.0, 240.0), "dc_ac_ratio": (1.0, 1.4),
    "noise_sd": (0.05, 0.05), "q_mva": (0.0, 0.3), "q_factor": (0.8, 1.0), "beta": (0.6, 1.0),
}  # fmt: skip
"""Every simulator parameter and its allowed range; the leakage check asserts each draw lies in it."""
SPELL_LEVEL_RANGE: Final = (0.3, 1.0)
"""A multi-day spell runs at this share of the total engine size ("part or full output")."""
NOISE_AUTOCORRELATION: Final = 0.9
"""Half-hour autocorrelation of the multiplicative noise on source S2."""
S1_SHARE: Final = 0.5
"""Half of the solar segments draw source S1 (the metered farms)."""
MIN_HH_FARM_DAY: Final = 40
SUBSTEPS_PER_HALF_HOUR: Final = 6
"""The engine simulator runs on a 5-minute grid, so a half-hour holding a switch has a partial mean."""
SUN_UP_ZENITH_DEG: Final = 90.0

# --------------------------------------------------------------------------- baselines and floors

FLOOR_POINTS: Final = 0.5
"""Every lobe's spread is at least this many points (the lobe-width floor)."""
QHAT_MVA: Final = 0.1
QHAT_POINTS: Final = QHAT_MVA / CAPACITY_MVA * POINTS
DEAD_BAND_POINTS: Final = 2.0
DAYLIGHT_HOURS_UTC: Final = (9, 13)
"""Half-hours that start from 09:00 up to, but not including, 13:00 UTC count as daylight."""
QUANTILE_LEVELS: Final = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
AUX_WEIGHT: Final = 0.5
"""Weight of an auxiliary head's negative log likelihood beside the main loss."""

# --------------------------------------------------------------------------- the bootstrap

BLOCK_DAYS: Final = 7
N_CONTRASTS: Final = 12
LEVEL: Final = 1.0 - 0.05 / N_CONTRASTS
"""Bonferroni level across the 12 contrasts (99.583%)."""
N_RESAMPLES: Final = 20_000
GENERATOR_SEED: Final = 729
MIN_KISH_FACTOR: Final = 20
MARGIN_FLOOR_POINTS: Final = 0.25
MARGIN_SD_MULTIPLE: Final = 2.0
ARCH_TIE_POINTS: Final = 0.125
"""Half of the margin's floor: an architecture this close to the best wins if it has fewer weights."""

# --------------------------------------------------------------------------- harm limits and bars

HARM_CRPS_POINTS: Final = 0.12
HARM_MAE_POINTS: Final = 0.25
H2_FALSE_SIGN_SHARE: Final = 0.005
RELIABILITY_BAR: Final = 0.01
RELIABILITY_BINS: Final = 10
BIN_GAP_BAR: Final = 0.1
BIN_MIN_COUNT: Final = 200
LOBE_COVERAGE_BAR: Final = 0.85
LOBE_LEVELS: Final = (0.05, 0.95)
MIN_REVERSE_ALL: Final = 2000
MIN_REVERSE_BSP: Final = 200
K1_MAE_RATIO: Final = 0.5
K1_RUNNING_POINTS: Final = 10.0
K1_MIN_RUNNING: Final = 200
K2_SIGN_SHARE: Final = 0.005
K2_ENGINE_SHARE: Final = 0.02
R2_TOLERANCE_POINTS: Final = 2.0
GATE_RATIO: Final = (0.5, 2.0)
GATE_MIN_VALUE_MVA: Final = 1.0
GATE_STEP_BINS: Final = 5
GATE_PLACEBO_SHIFTS: Final = (3, 20)
REAL_UP_TO: Final = "2026-06-30"
"""The last valid date that any real check reads."""

# --------------------------------------------------------------------------- training regime

ARCHITECTURES: Final = ("dae", "unet", "bilstm")
ARCH_SEEDS: Final = (0, 1, 2)
MARGIN_SEEDS: Final = (0, 1, 2, 3, 4)
EPOCH_WINDOWS: Final = 8000
BATCH_SIZE: Final = 256
LEARNING_RATE: Final = 1e-3
WEIGHT_DECAY: Final = 1e-4
CLIP_NORM: Final = 1.0
LR_PATIENCE: Final = 10
STOP_PATIENCE: Final = 25
MAX_EPOCHS: Final = 200
TIMING_EPOCHS: Final = 5
GPU_BUDGET_HOURS: Final = 10.0
GPU_MEMORY_GB: Final = 6.0
N_FITS: Final = 47
MASK_PROBABILITY: Final = 0.5
MASK_HOURS: Final = (1, 24)
LAG_DAYS: Final = (7, 14)

# --------------------------------------------------------------------------- arms and contrasts

ARMS: Final = ("A", "B", "C", "D", "H", "Dq", "D24", "E", "F", "G1", "G2")
LADDER_ARMS: Final = ("B", "C", "H", "Dq", "D24", "E", "F", "G1", "G2")
"""The 9 arms other than A and D, which are fitted on the chosen architecture."""
CONTRASTS: Final = {
    "C1": ("D", "Bstar"), "C2": ("A", "P0f"), "C3": ("B", "A"), "C4": ("C", "B"),
    "C5": ("D", "C"), "C6": ("Dq", "D"), "C7": ("E", "D"), "C8": ("F", "D"),
    "C9": ("G1", "D"), "C10": ("G2", "D"), "C11": ("D24", "D"), "C12": ("H", "A"),
}  # fmt: skip
"""Each contrast as (arm minus arm); a negative difference is a gain."""
BSP_CONTRASTS: Final = ("C4", "C12")
PINBALL_CONTRASTS: Final = ("C6",)
"""Contrasts read on the 11-quantile pinball loss, not on signed CRPS."""
Verdict = Literal[
    "stopped", "invalid", "failed", "not supported", "supported", "partly supported", "small gain",
    "inconclusive",
]  # fmt: skip


# --------------------------------------------------------------------------- the margin and the limits


def harm_limit_points(*, reverse_share: float) -> float:
    """Return the import harm limit, half of the margin floor spread over the import share."""
    return 0.5 * MARGIN_FLOOR_POINTS / (1.0 - reverse_share)


def margin_points(*, val_crps_by_seed: Sequence[float]) -> float:
    """Return the margin m: the larger of the floor and twice the sd over seeds 0 to 4.

    Args:
        val_crps_by_seed: Arm D's validation-month signed CRPS for each of seeds 0 to 4.

    Returns:
        The margin in points of capacity.
    """
    if len(val_crps_by_seed) != len(MARGIN_SEEDS):
        msg = f"the margin needs {len(MARGIN_SEEDS)} seeds, got {len(val_crps_by_seed)}"
        raise ValueError(msg)
    sd = float(np.std(np.asarray(val_crps_by_seed, dtype=np.float64), ddof=1))
    return max(MARGIN_FLOOR_POINTS, MARGIN_SD_MULTIPLE * sd)


def write_margin_file(*, path: Path, val_crps_by_seed: Sequence[float]) -> dict:
    """Write the margin file once and refuse to overwrite it.

    The file is created with a hard link from a temporary name, so a second writer fails with
    `FileExistsError` and the first file is never replaced, even by a concurrent process.

    Args:
        path: Where `margin.json` goes.
        val_crps_by_seed: Arm D's validation-month signed CRPS for seeds 0 to 4.

    Returns:
        The record written.
    """
    record = {
        "margin_points": margin_points(val_crps_by_seed=val_crps_by_seed),
        "floor_points": MARGIN_FLOOR_POINTS,
        "val_crps_by_seed": [float(v) for v in val_crps_by_seed],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return record


def read_margin_file(*, path: Path) -> float:
    """Return the stored margin in points, and fail when the file does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: m must be written before any test contrast is read"
        raise FileNotFoundError(msg)
    return float(json.loads(path.read_text())["margin_points"])


def choose_architecture(*, scores: Mapping[str, float], weights: Mapping[str, int]) -> str:
    """Return the architecture with the lowest validation CRPS, or a lighter one within the tie.

    Args:
        scores: Mean over seeds 0 to 2 of arm D's validation signed CRPS, by architecture.
        weights: The number of weights of each architecture.

    Returns:
        The chosen architecture's name.
    """
    best = min(scores, key=lambda a: scores[a])
    close = [a for a in scores if scores[a] <= scores[best] + ARCH_TIE_POINTS]
    return min(close, key=lambda a: (weights[a], scores[a]))


def choose_baseline(*, b1: float, b2: float) -> str:
    """Return "B1" or "B2", whichever has the lower validation signed CRPS (B1 on a tie)."""
    return "B2" if b2 < b1 else "B1"


# --------------------------------------------------------------------------- the widening


def kish_number(*, weights: Sequence[float] | np.ndarray) -> float:
    """Return the Kish effective number `(sum w)^2 / sum w^2` of non-negative weights."""
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0]
    return 0.0 if w.size == 0 else float(w.sum() ** 2 / np.square(w).sum())


def widening_ratio(*, kish: float, level: float = LEVEL) -> float:
    """Return the Student t quantile (kish - 1 degrees of freedom) over the normal quantile.

    Args:
        kish: The Kish number of the smaller factor.
        level: The two-sided level of the contrast.

    Returns:
        The ratio, or infinity when the Kish number is 1 or less.
    """
    from scipy import stats

    if kish <= 1.0:
        return float("inf")
    tail = 1.0 - (1.0 - level) / 2.0
    return float(stats.t.ppf(tail, kish - 1.0) / stats.norm.ppf(tail))


@dataclass(frozen=True)
class Reading:
    """A contrast's point estimate and interval, in points of capacity.

    Attributes:
        point: The point estimate (negative is a gain).
        lower: The lower end of the interval.
        upper: The upper end of the interval.
    """

    point: float
    lower: float
    upper: float

    @property
    def half_width(self) -> float:
        """Half of the interval's width."""
        return (self.upper - self.lower) / 2.0


# --------------------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match verdict table reads.

    Attributes:
        stopped: Row 0 applies (halted, over budget, gate failed, counts short, or a review gate open).
        invalid: Row 1 applies (a correctness check failed).
        failed: Row 2 applies (a needed fit raised and was not rerun).
        h1_pass: Harm check H1 passed.
        h2_pass: Harm check H2 passed.
        c1: The headline contrast C1 (arm D minus B*).
        margin: The margin m in points.
        calibration_pass: The sign calibration bar passed.
        lobe_pass: The lobe coverage bar passed.
    """

    stopped: bool
    invalid: bool
    failed: bool
    h1_pass: bool
    h2_pass: bool
    c1: Reading
    margin: float
    calibration_pass: bool
    lobe_pass: bool


VERDICT_WORDS: Final = {
    0: "stopped", 1: "invalid", 2: "failed", 3: "not supported", 4: "supported",
    5: "partly supported", 6: "small gain", 7: "inconclusive", 8: "not supported",
}  # fmt: skip


def verdict_row(*, v: VerdictInputs) -> int:  # noqa: PLR0911
    """Return the first row of the verdict table that matches (0 to 8).

    Args:
        v: The inputs of the table.

    Returns:
        The row number; `VERDICT_WORDS` maps it to the verdict word.
    """
    if v.stopped:
        return 0
    if v.invalid:
        return 1
    if v.failed:
        return 2
    if not (v.h1_pass and v.h2_pass):
        return 3
    gain = v.c1.point <= -v.margin and v.c1.upper < 0.0
    if gain and v.calibration_pass and v.lobe_pass:
        return 4
    if gain:
        return 5
    if v.c1.upper < 0.0:
        return 6
    if v.c1.lower <= 0.0 <= v.c1.upper and v.c1.half_width >= v.margin:
        return 7
    return 8


def verdict_word(*, v: VerdictInputs) -> Verdict:
    """Return the verdict word of `verdict_row`."""
    return VERDICT_WORDS[verdict_row(v=v)]  # ty: ignore[invalid-return-type]


def harm_passes(
    *, crps_excess: float, mae_excess: float, false_sign_share: float | None = None
) -> bool:
    """Return whether a harm check passes (H1 without a share, H2 with one).

    Args:
        crps_excess: Arm D's signed CRPS minus P0f's on the import half-hours, in points.
        mae_excess: Arm D's normalised median error minus P0's, in points.
        false_sign_share: H2 only: the share of real half-hours with P(X<0) above 0.5.

    Returns:
        True when every limit holds.
    """
    ok = crps_excess <= HARM_CRPS_POINTS and mae_excess <= HARM_MAE_POINTS
    if false_sign_share is not None:
        ok = ok and false_sign_share <= H2_FALSE_SIGN_SHARE
    return bool(ok)


def reachability_ok(*, n_reverse_all: int, n_reverse_bsp: int) -> bool:
    """Return whether the test banks hold enough reverse-flow half-hours outside the dead band."""
    return n_reverse_all >= MIN_REVERSE_ALL and n_reverse_bsp >= MIN_REVERSE_BSP


def gate_ratio_ok(*, synthetic: float, real: float) -> bool:
    """Return whether a synthetic share lies within 0.5 to 2.0 times the real share."""
    return GATE_RATIO[0] * real <= synthetic <= GATE_RATIO[1] * real


def is_finite_number(*, value: float) -> bool:
    """Return whether a number is finite."""
    return math.isfinite(value)
