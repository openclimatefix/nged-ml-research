# ruff: noqa: E501
"""Experiment 732: every named constant and bar of the plan, the limits files, and the verdict rule.

Every number that the page's decision rule pre-registers is a `Final` here, so the page and the code
are compared by reading this one file. The functions are pure, except the write-once file writers,
which refuse to overwrite. No function prints or logs a month, a date, an engine size, a series
identifier, or a real value.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00732_paths  # noqa: F401

# isort: split
import e00729_core as core729
import e00730_core as core730
import numpy as np

# --------------------------------------------------------------------------- identity and rows

ROW_SET_ID: Final = "e00730_synthetic_d8_v1"
"""The row set of the sign and the solar output: experiment 730's 94,080 scored half-hours."""
OVERLAY_ROW_SET_ID: Final = "e00732_overlay_d8_v1"
"""The row set of the transfer steps and the injection recovery."""
N_TEST_ROWS: Final = 94_080
"""The row count of experiment 730's test bank, which every stored array must match."""
BRIER_CANARY: Final = 0.00853
"""Experiment 730's printed Brier score of arm S, to five decimal places."""
BRIER_CANARY_DECIMALS: Final = 5
MARGIN_730: Final = 0.00122
"""Experiment 730's margin, to the figure that the page prints; `margin.json` must agree."""
MARGIN_730_FILE_TOLERANCE: Final = 5e-6
POOLED_LIMIT_730: Final = 0.0071
"""The pooled harm limit, which the recomputed limit must equal."""
TRAIN_REVERSE_SHARE_TOLERANCE: Final = 0.10
"""The training share of reverse flow may differ from experiment 730's by this share of it."""
MIN_OUTPUT_SHARE: Final = core730.MIN_OUTPUT_SHARE
"""Every simulated component must produce output on at least this share (0.99)."""
EXACT_SHARE: Final = 1.0
MAX_UNPAIRED_SHARE: Final = 0.05
"""A training month may have at most this share of windows without a sibling pair."""
MIN_TRANSFER_STEPS_TEST: Final = 400
MIN_WINDOWS_PER_AMOUNT: Final = 800
MIN_STEP_POINTS_REACHABLE: Final = 5.0

# --------------------------------------------------------------------------- the bootstrap

BLOCK_DAYS: Final = core729.BLOCK_DAYS
N_CONTRASTS: Final = 7
LEVEL: Final = 1.0 - 0.05 / N_CONTRASTS
"""Bonferroni level across the 7 counted contrasts, C1 to C7 (99.286%). The harm reading HB, the
bar reading R23, and every other report-only reading are read at the same level outside the count."""
N_RESAMPLES: Final = 20_000
GENERATOR_SEED: Final = 732
MASK_SEED: Final = 730
"""The fixed masks of experiment 730 stay the same here."""

# --------------------------------------------------------------------------- the synthetic world

TRANSFER_RATE_TRAIN: Final = (0.0, 6.0)
"""Transfers per 30 days in training: uniform per segment."""
TRANSFER_RATE_FIXED: Final = 6.0
"""Transfers per 30 days in the test and validation overlays."""
TRANSFER_HOURS: Final = (1.0, 96.0)
"""Duration of one transfer: 1 hour to 4 days, log-uniform."""
TRANSFER_POINTS: Final = (3.0, 40.0)
"""Size T0 of one transfer in points, log-uniform."""
READ_PARTNERS: Final = ("D3", "D5", "D14")
"""The neighbours that the neural network reads; a read partner is drawn with equal shares."""
UNREAD_PARTNER: Final = "unread"
"""Another Primary under BSP A that the neural network does not read (no mirror, BSP unchanged)."""
OTHER_BSP_PARTNER: Final = "other_bsp"
"""A Primary under another BSP (no mirror, the BSP meter moves by beta T)."""
PARTNER_MIX: Final = {"read": 0.6, UNREAD_PARTNER: 0.2, OTHER_BSP_PARTNER: 0.2}
"""The share of drawn transfers by partner case."""
SHAPES: Final = ("constant", "demand", "solar")
SHAPE_MIX: Final = {"constant": 0.6, "demand": 0.2, "solar": 0.2}
"""The share of drawn transfers by shape."""
SOLAR_SHAPE_KAPPA: Final = (0.3, 1.0)
"""The depth kappa of a solar-shaped transfer, uniform; it is halved until the floor holds."""
SHAPE_FLOOR: Final = 0.1
"""A transfer's load never falls below this share of T0 on any half-hour (so T never changes sign)."""
DONOR_CAP: Final = 0.8
"""The load that a donor gives is at most this share of the donor's demand on every half-hour."""
DONOR_REDRAWS: Final = 20
"""A size that breaks the donor rule is redrawn this many times, and then the transfer is dropped."""
MAX_DROPPED_SHARE: Final = 0.05
"""At most this share of the drawn transfers may be dropped by the donor rule."""
MIRROR_JITTER_SHARE: Final = 0.3
"""The share of mirrored transfers whose mirror is jittered; the rest are exact."""
MIRROR_JITTER_RANGE: Final = (0.7, 1.3)
"""The factor j of a jittered mirror: the partner's series changes by minus j T."""
MIRROR_TOLERANCE_POINTS: Final = 0.01
"""The mirror check: the mean partner change equals minus j times the mean T within this many points."""
MIN_PARTNER_COVERAGE: Final = 0.5
"""A read partner must hold a reading on at least this share of the transfer's half-hours."""
TICKS_PER_HALF_HOUR: Final = 6
"""A transfer starts on a 5-minute grid."""
MAX_REDRAWS: Final = 200
"""Placement redraws of a transfer whose edges or span do not fit."""
INJECTION_POINTS: Final = (2.0, 60.0)
"""The extra injection a in training: log-uniform per window and epoch."""
TEST_INJECTION_POINTS: Final = (5.0, 20.0, 50.0)
TEST_SIBLING_SOURCES: Final = ("S2",)
"""The test overlay lends its extra injection from a sibling of this source only (S1 is a real farm's metered output)."""
EPS_RANGE: Final = (0.05, 0.3)
"""The raised-sky factor in training: uniform per window."""
TEST_EPS: Final = 0.2
CSI_CLIP: Final = 1.5
RAISABLE_MAX_CSI: Final = 1.25
SUN_UP_COSZ: Final = 0.1
UNRELIABLE_MIN_COSZ: Final = 0.2
"""A zero of S1 solar output counts as an unreliable target only at a cosine of the zenith angle of at least this."""
MAX_UNRELIABLE_SHARE: Final = 0.05
"""A bank may have at most this share of unreliable half-hours among its S1 sun-up half-hours of solar worlds."""
NIGHT_COSZ: Final = 0.0
SEED_TRAIN_OVERLAY: Final = 732_100
SEED_VALIDATION_OVERLAY: Final = 732_050
SEED_TEST_OVERLAY: Final = 732_000
OVERLAY_KINDS: Final = ("train", "validation", "test")
BANK_CODE: Final = {"train": 0, "validation": 1, "test": 2, "shifted": 3}
DELTA_SHARE_OF_A: Final = 0.05
"""The injection readings keep half-hours where delta is at least this share of a."""
DELTA_ACTIVE_POINTS: Final = 0.01
"""A half-hour counts as carrying an injection when delta is above this many points."""
CLEAR_DAY_CSI: Final = 0.8
"""A clear day has a mean clear-sky index of at least this on its sun-up half-hours."""
MIN_SOLAR_TARGET_SHARE: Final = 0.99
"""The solar target is above zero on at least this share of sun-up half-hours of clear solar days."""
TENSOR_CHECK_TOLERANCE: Final = 1e-5
"""Channel values closer than this count as equal in the checks on the input tensors."""

# --------------------------------------------------------------------------- training regime

N_FITS: Final = 24
SEEDS: Final = (0, 1, 2)
MAX_EPOCHS: Final = core730.MAX_EPOCHS
EPOCH_WINDOWS: Final = core730.EPOCH_WINDOWS
BATCH_SIZE: Final = core730.BATCH_SIZE
STYLE: Final = "R"
"""Every arm hides one random span of the reading on half of its training windows."""
SOLAR_SCALE: Final = 10.0
"""The solar output is `SOLAR_SCALE * softplus(raw)` points."""
SIZE_SCALE: Final = 10.0
"""The step size is `SIZE_SCALE * raw` points."""
HUBER_SCALE: Final = 1.0
STEP_THRESHOLD: Final = 0.5
STEP_TOLERANCE_HALF_HOURS: Final = 1
BIG_STEP_POINTS: Final = 5.0

LOSS_TERMS: Final = ("sign", "solar", "step", "size", "injection", "night", "monotone")
BASE_WEIGHTS: Final = {
    "sign": 1.0, "solar": 0.02, "step": 1.0, "size": 0.02, "injection": 0.02, "night": 0.02,
    "monotone": 0.02,
}  # fmt: skip
"""The weights of the seven loss terms of U*. No weight is tuned on data."""


@dataclass(frozen=True)
class ArmSpec732:
    """The design of one arm of experiment 732.

    Attributes:
        name: The arm's name.
        arch: "unet" or "bilstm".
        heads: The arm has the solar, step, and size heads (False for S+).
        weights: The weight of each loss term (0 drops the term).
        stop_gradient: The solar, step, and size heads read the backbone through a stop-gradient.
        inject: The arm sees each window's injected member.
        raised: The arm sees each window's raised-sky member.
        neighbours: The arm reads the 12 neighbour channels (False for U-N, whose batches hold none).
        bsp_zeroed: The arm's BSP-step channel and its mask are zero in every window (U-noBSP).
        role: The arm's pre-registered role.
    """

    name: str
    arch: str
    heads: bool
    weights: Mapping[str, float]
    stop_gradient: bool = False
    inject: bool = False
    raised: bool = False
    neighbours: bool = True
    bsp_zeroed: bool = False
    role: str = ""

    @property
    def n_outputs(self) -> int:
        """Output channels: the sign logit, then the solar output, step logit, and step size."""
        return 4 if self.heads else 1


def _w(**changes: float) -> dict[str, float]:
    """Return `BASE_WEIGHTS` with some terms changed."""
    return {**BASE_WEIGHTS, **changes}


ARMS: Final = {
    "U*": ArmSpec732("U*", "unet", True, _w(), inject=True, raised=True, role="headline"),
    "U-NC": ArmSpec732(
        "U-NC", "unet", True, _w(injection=0.0, night=0.0, monotone=0.0), inject=True,
        role="control of C2 and C3",
    ),
    "U-NCw": ArmSpec732(
        "U-NCw", "unet", True, _w(solar=0.08, injection=0.0, night=0.0, monotone=0.0), inject=True,
        role="report only: weight-matched control of R10 and R11",
    ),
    "S+": ArmSpec732(
        "S+", "unet", False, _w(solar=0.0, step=0.0, size=0.0, injection=0.0, night=0.0, monotone=0.0),
        role="control of C6",
    ),
    "U-SG": ArmSpec732(
        "U-SG", "unet", True, _w(), stop_gradient=True, inject=True, raised=True,
        role="report only: stop-gradient",
    ),
    "U*-L": ArmSpec732("U*-L", "bilstm", True, _w(), inject=True, raised=True, role="report only: architecture"),
    "U-N": ArmSpec732(
        "U-N", "unet", True, _w(), inject=True, raised=True, neighbours=False,
        role="control of C7: no neighbour channels",
    ),
    "U-noBSP": ArmSpec732(
        "U-noBSP", "unet", True, _w(), inject=True, raised=True, bsp_zeroed=True,
        role="report only: control of R19, the BSP channel is zero",
    ),
}  # fmt: skip
FIT_ARMS: Final = tuple(ARMS)
COUNTED_ARMS: Final = ("S+", "U*", "U-N", "U-NC")
"""The four arms (12 fits) that the counted contrasts C1 to C7 read besides stored arms and rules."""
NEEDED_ARMS: Final = COUNTED_ARMS
"""A failed fit of one of these arms, not rerun under the rerun rule, makes verdict row 2."""
REPORT_ONLY_ARMS: Final = ("U-NCw", "U-SG", "U*-L", "U-noBSP")
FIT_FAMILIES: Final = (
    ("timing", ()),
    ("S+, U*", ("S+", "U*")),
    ("U-N", ("U-N",)),
    ("U-NC", ("U-NC",)),
    ("U-noBSP", ("U-noBSP",)),
    ("U-NCw", ("U-NCw",)),
    ("U-SG", ("U-SG",)),
    ("U*-L", ("U*-L",)),
)
"""The families of fits, in order; a "fits starting" line precedes each. The report-only arms come
last, so that a stop at the stop line loses report-only readings before any counted reading."""
DROP_ORDER: Final = ("U*-L", "U-SG", "U-NCw", "U-noBSP")
"""The report-only arms that the timing gate drops, in order, while the projection exceeds the stop line."""
STORED_730: Final = {
    "S": ("S", "unet"),
    "S-L": ("S", "bilstm"),
    "SD*": ("SD-R", "unet"),
}
"""The stored arms of experiment 730 read as controls: name -> (its fit name, architecture)."""
CONTRASTS: Final = {
    "C1": ("U*", "S"), "C2": ("U*", "U-NC"), "C3": ("U*", "U-NC"), "C4": ("U*", "B-PV2"),
    "C5": ("U*", "B-TN"), "C6": ("U*", "S+"), "C7": ("U*", "U-N"),
}  # fmt: skip
"""Each counted contrast as (arm minus arm); a negative difference is a gain. C3 reads the injection error."""
HARM_READING: Final = "HB"
"""U* minus S in the band of 1 to 3 MVA, a one-sided lower bound at `LEVEL`, outside the count of 7."""
REPORT_ONLY: Final = {
    "R1": "U-SG minus S+, Brier score of P(X<0)",
    "R2": "U* minus U-SG, Brier score of P(X<0)",
    "R3": "U*-L minus S-L (730), Brier score of P(X<0)",
    "R4": "S+ minus S (730), Brier score of P(X<0)",
    "R5": "U* minus U-NC, Brier score of P(X<0)",
    "R6": "U* minus B-PV1, solar mean absolute error",
    "R7": "U*-L minus U*, solar mean absolute error",
    "R8": "U* minus S+, Brier score of P(X<0) on the transfer overlay",
    "R9": "U* minus U-NC, share of monotonicity violations",
    "R10": "U* minus U-NCw, solar mean absolute error",
    "R11": "U* minus U-NCw, mean absolute injection error",
    "R12": "C4 on the solar-only rows and on the no-injection rows, each alone",
    "R13": "U* minus U-N, Brier score of P(X<0) on the plain test rows",
    "R14": "U* minus U-N, Brier score of P(X<0) on the transfer overlay, split by whether the transfer has a mirror at a read neighbour",
    "R15": "U-N minus B-T, Brier score of the step event",
    "R16": "C7 split by partner case",
    "R17": "C5 and C7 split by the transfer's shape",
    "R18": "recall of U*, U-N, B-TN, and B-T on solar-shaped transfers minus recall on constant transfers",
    "R19": "U* minus U-noBSP, Brier score of P(X<0) on the plain test rows of the two BSP test months",
    "R20": "R19 split by world type, and R19 on all plain test rows",
    "R21": "U* with the BSP channel and its mask zero at scoring time minus U*, Brier score of P(X<0) on the two BSP test months",
    "R22": "U* minus U-noBSP, Brier score of the step event on the transfer overlay, split by partner case",
    "R23": "U* minus B-PV0, solar mean absolute error on the sun-up rows of solar-world windows",
}  # fmt: skip
N_REPORT_ONLY: Final = len(REPORT_ONLY)
TABLES: Final = ("T1", "T2", "T3", "T4", "T5", "T6")
"""The report tables of the plan's section "Contrasts"."""

# --------------------------------------------------------------------------- limits and bars

HARM_FRACTION: Final = core730.HARM_FRACTION
DECIDABLE_READING_MVA: Final = 1.0
DECIDABLE_READING_POINTS: Final = core730.DECIDABLE_READING_POINTS
BAND_EDGE_MVA: Final = 3.0
BAND_EDGE_POINTS: Final = BAND_EDGE_MVA / core729.CAPACITY_MVA * core729.POINTS
BANDS: Final = ("pooled", "1-3MVA", "3MVA+")
FALSE_ALARM_THRESHOLD: Final = core730.FALSE_ALARM_THRESHOLD
LIMIT_RANGE: Final = core730.LIMIT_RANGE
RECOVERY_BAND: Final = (0.9, 1.1)
"""The recovery ratio at each injected amount must lie in this band."""
RECOVERY_MEDIAN_ERROR: Final = 0.1
"""The median of |dPV - delta| over a must be at most this."""
RECOVERY_REVERSE_BAND: Final = (0.8, 1.2)
"""The recovery ratio on half-hours where the plain member is in reverse flow, pooled over amounts."""
ABSOLUTE_BAR_SHARE: Final = 0.5
"""The solar mean absolute error on sun-up solar-world half-hours is at most this share of the mean target."""
NIGHT_MEAN_POINTS: Final = 0.05
NIGHT_EXCEED_POINTS: Final = 0.5
NIGHT_EXCEED_SHARE: Final = 0.005
MONOTONE_TOLERANCE_POINTS: Final = 0.1
MONOTONE_MAX_SHARE: Final = 0.05
RESPONSE_MIN_TARGET_POINTS: Final = 1.0
RESPONSE_MIN_SHARE: Final = 0.5
TRANSFER_F1_BAR: Final = 0.6
SOLAR_BARS: Final = (
    "recovery", "recovery_reverse", "absolute", "better_than_sky", "night", "monotone", "response",
    "better_than_demand",
)  # fmt: skip
"""The eight solar bars, each read on U* on the test bank."""
SOLAR_CORE_BARS: Final = ("recovery", "recovery_reverse", "absolute", "better_than_sky")
"""The bars that a "partly" solar word still needs: the two recovery bars, the absolute bar, and the sky bar."""
DETECTABLE_RECALL: Final = 0.5
DETECTABLE_PRECISION: Final = 0.5
"""A size band is detectable when recall and precision at the 0.5 threshold both reach these."""
SIZE_BANDS_NOISE_UNITS: Final = (2.0, 4.0, 8.0)
"""Edges of T6's bands of size in D8's noise units: under 2, 2 to 4, 4 to 8, and 8 and more."""
SIZE_BANDS_POINTS: Final = (3.0, 5.0, 10.0, 20.0, 40.0)
DURATION_BANDS_HOURS: Final = (1.0, 3.0, 12.0, 48.0, 96.0)
MIN_STEPS_PER_SIDE: Final = 100
"""The detectability statement needs this many scored steps under, and at, 4 noise units."""
MIN_BSP_REVERSE_HALF_HOURS: Final = 200
"""R19 and the BSP word are void below this many reverse-flow half-hours in the two BSP test months."""
Q_HAT_MVA: Final = 0.1
"""The reactive floor that B-PV2 and B-PV1 remove from the reading."""
Q_HAT_POINTS: Final = Q_HAT_MVA / core729.CAPACITY_MVA * core729.POINTS

# --------------------------------------------------------------------------- real checks

P2_MEAN_POINTS: Final = 1.0
P2_EXCEED_POINTS: Final = 5.0
P2_EXCEED_SHARE: Final = 0.01
P3_SPEARMAN: Final = 0.5
P3_CLEAR_CSI: Final = 0.8
P3_CLOUDY_CSI: Final = 0.3
P3_RATIO: Final = 2.0
P3_MIDDAY_UTC: Final = (10, 14)
P4_MAX_DIFFERENCE_POINTS: Final = 1.0
K3_MIN_EDGES: Final = 10
K3_WINDOW_HOURS: Final = 2.0
K3_HIT_THRESHOLD: Final = 0.5
K1_MIN_REVERSE_DECIDABLE: Final = core730.K1_MIN_REVERSE_DECIDABLE

# --------------------------------------------------------------------------- compute and stop line

STOP_LINE_HOURS: Final = 16.0
MEMORY_CAP_GB: Final = 14.0
GPU_MEMORY_GB: Final = 6.0
TIMING_EPOCHS: Final = 5
TIMING_ARMS: Final = ("U*", "U-N", "U-NC", "S+", "U-SG", "U*-L")
INFERENCE_HOURS: Final = 0.3
TIMING_GATE_HOURS: Final = 0.05
FITS_PER_ARM: Final = len(SEEDS)
RATE_OF: Final = {
    "U*": "U*",
    "U-N": "U-N",
    "U-NC": "U-NC",
    "U-NCw": "U-NC",
    "S+": "S+",
    "U-SG": "U-SG",
    "U*-L": "U*-L",
    "U-noBSP": "U*",
}
"""The timed arm whose seconds an epoch prices each fitted arm."""
MAX_SLOT_WORKERS: Final = 6

# --------------------------------------------------------------------------- the verdict

Verdict = Literal["stopped", "invalid", "failed", "not supported", "supported", "inconclusive"]
VERDICT_WORDS: Final = {
    0: "stopped", 1: "invalid", 2: "failed", 3: "not supported", 4: "supported",
    5: "inconclusive", 6: "not supported",
}  # fmt: skip
"""The verdict table of the plan: the first row that matches is the verdict."""
FRONT_MATTER_WORD: Final = {"stopped": "inconclusive"}
"""Row 0 is recorded as `inconclusive` in the front matter."""


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match verdict table reads.

    Attributes:
        stopped: Row 0 applies.
        invalid: A correctness check of row 1 failed.
        silent_test_missed: A known-answer test of a silenced component did not fail (row 1).
        failed: Row 2 applies.
        h1_pooled_pass: H1 pooled passed for U*.
        h1_high_pass: H1 at 3 MVA and above passed for U*.
        h2_pass: H2 passed.
        band_fail: The band of 1 to 3 MVA is above its limit and HB's lower bound is above zero.
        c1: The headline contrast C1 (U* minus S).
        margin: The margin m.
    """

    stopped: bool
    invalid: bool
    silent_test_missed: bool
    failed: bool
    h1_pooled_pass: bool
    h1_high_pass: bool
    h2_pass: bool
    band_fail: bool
    c1: core729.Reading
    margin: float


def band_fails(*, band_mean: float, band_limit_value: float, hb_lower: float) -> bool:
    """Return whether the band of 1 to 3 MVA fails H1: above its limit and HB's lower bound above zero.

    A band that is above its limit with HB's lower bound at or below zero is "at the calibrated
    boundary" and does not fail.
    """
    return bool(band_mean > band_limit_value and hb_lower > 0.0)


def harm_row_fails(*, v: VerdictInputs) -> str | None:
    """Return the name of the first failed harm check of row 3, or None."""
    if not v.h1_pooled_pass:
        return "H1 pooled"
    if not v.h1_high_pass:
        return "H1 at 3 MVA and above"
    if not v.h2_pass:
        return "H2"
    if v.band_fail:
        return "H1 in the band of 1 to 3 MVA (against HB)"
    return None


def non_inferior(*, c1: core729.Reading, margin: float) -> bool:
    """Return whether C1's upper end is at most the margin (NI)."""
    return bool(c1.upper <= margin)


def verdict_row(*, v: VerdictInputs) -> int:  # noqa: PLR0911
    """Return the first row of the verdict table that matches (0 to 6)."""
    if v.stopped:
        return 0
    if v.invalid or v.silent_test_missed:
        return 1
    if v.failed:
        return 2
    if harm_row_fails(v=v) is not None:
        return 3
    if non_inferior(c1=v.c1, margin=v.margin):
        return 4
    if v.c1.lower <= 0.0:
        return 5
    return 6


def verdict_word(*, v: VerdictInputs) -> Verdict:
    """Return the verdict word of `verdict_row`."""
    return VERDICT_WORDS[verdict_row(v=v)]  # ty: ignore[invalid-return-type]


def front_matter_word(*, word: str) -> str:
    """Return the word that the front matter carries (row 0 is `inconclusive`)."""
    return FRONT_MATTER_WORD.get(word, word)


def ni_note(*, c1: core729.Reading, margin: float) -> str | None:
    """Return the sentence that row 4 carries when C1 is detectably worse by less than the margin."""
    if non_inferior(c1=c1, margin=margin) and c1.lower > 0.0:
        return "the sign is detectably worse, by less than the spread between seeds of experiment 730's S"
    return None


def sign_change_carrier(*, c6: core729.Reading, r4: core729.Reading) -> str:
    """Return which of C6 and R4 carries a change in the sign (rows 3 to 6 print both).

    A change in R4 comes from this experiment's training windows and inputs, and a change in C6
    comes from the extra heads.
    """
    c6_moves = _excludes_zero(reading=c6)
    r4_moves = _excludes_zero(reading=r4)
    if c6_moves and r4_moves:
        return "both C6 and R4"
    if c6_moves:
        return "C6"
    if r4_moves:
        return "R4"
    return "neither"


def _excludes_zero(*, reading: core729.Reading) -> bool:
    """Return whether a finite interval excludes zero."""
    finite = all(math.isfinite(z) for z in (reading.point, reading.lower, reading.upper))
    return bool(finite and (reading.upper < 0.0 or reading.lower > 0.0))


def is_gain(*, reading: core729.Reading) -> bool:
    """Return whether a counted reading is a gain: an interval that lies below zero."""
    return bool(math.isfinite(reading.upper) and reading.upper < 0.0)


def is_loss(*, reading: core729.Reading) -> bool:
    """Return whether a counted reading is a loss: an interval that lies above zero."""
    return bool(math.isfinite(reading.lower) and reading.lower > 0.0)


def reading_word(*, reading: core729.Reading | None) -> str:
    """Return "gain", "loss", "no detectable difference", or "void" for a reading read at a threshold of zero."""
    if reading is None or not all(
        math.isfinite(z) for z in (reading.point, reading.lower, reading.upper)
    ):
        return "void"
    if is_gain(reading=reading):
        return "gain"
    if is_loss(reading=reading):
        return "loss"
    return "no detectable difference"


def solar_word(*, bars: Mapping[str, bool], reachable: bool) -> str:
    """Return the solar word: "reasonable", "partly", "not shown", or "void".

    "reasonable" needs all eight bars. "partly" needs the two recovery bars, the absolute bar, and
    the bar against the sky alone, with another bar failing. A missing bar fails.
    """
    if not reachable:
        return "void"
    passed = {name: bool(bars.get(name, False)) for name in SOLAR_BARS}
    if all(passed.values()):
        return "reasonable"
    if all(passed[name] for name in SOLAR_CORE_BARS):
        return "partly"
    return "not shown"


def transfer_word(*, c5: core729.Reading | None, f1: float, reachable: bool) -> str:
    """Return the transfer output's reading word, read from C5 and the F1 bar.

    Args:
        c5: C5's reading (U* minus B-TN in the step Brier score), or None.
        f1: U*'s F1 at 0.5 on transfer steps of 5 points and more.
        reachable: The reachability count of transfer steps holds.

    Returns:
        "works", "works only in Brier score", "no better than the rule", "worse than the rule", or "void".
    """
    word = reading_word(reading=c5)
    if not reachable or word == "void":
        return "void"
    if word == "gain":
        return "works" if f1 >= TRANSFER_F1_BAR else "works only in Brier score"
    if word == "loss":
        return "worse than the rule"
    return "no better than the rule"


def neighbour_word(*, c7: core729.Reading | None, reachable: bool) -> str:
    """Return the neighbour word, read from C7.

    The word is "the neighbours help", "no detectable difference", "the neighbours hurt", or "void".
    """
    word = reading_word(reading=c7)
    if not reachable or word == "void":
        return "void"
    return {
        "gain": "the neighbours help",
        "loss": "the neighbours hurt",
        "no detectable difference": "no detectable difference",
    }[word]


def bsp_word(*, r19: core729.Reading | None, n_reverse_half_hours: int) -> str:
    """Return the BSP word, read from R19: "helps", "no detectable difference", "hurts", or "void".

    R19 reads the two test months that the BSP meter covers and is void below
    `MIN_BSP_REVERSE_HALF_HOURS` reverse-flow half-hours there.
    """
    word = reading_word(reading=r19)
    if n_reverse_half_hours < MIN_BSP_REVERSE_HALF_HOURS or word == "void":
        return "void"
    return {
        "gain": "helps",
        "loss": "hurts",
        "no detectable difference": "no detectable difference",
    }[word]


def detectable_band(*, recall: float, precision: float) -> bool:
    """Return whether a size band of table T6 is detectable (recall and precision both at least 0.5)."""
    return bool(recall >= DETECTABLE_RECALL and precision >= DETECTABLE_PRECISION)


# --------------------------------------------------------------------------- limits


@dataclass(frozen=True)
class BandLimit:
    """The harm limit of one reading band.

    Attributes:
        share: The share of reverse-flow rows among the band's decidable rows.
        limit: The limit L, rounded down to two significant figures.
        n_rows: The rows of the band.
    """

    share: float
    limit: float
    n_rows: int


def band_limit(*, x: np.ndarray, r: np.ndarray, lo: float, hi: float) -> BandLimit:
    """Return the harm limit of the rows with a reading in `[lo, hi)` points.

    Args:
        x: The signed truth in points.
        r: The reading in points.
        lo: The lowest reading of the band.
        hi: The highest reading of the band (exclusive).

    Returns:
        L = 0.25 s / (1 - s), rounded down to two significant figures.
    """
    keep = (r >= lo) & (r < hi)
    if not keep.any():
        return BandLimit(float("nan"), 0.0, 0)
    s = float(np.mean(x[keep] < 0.0))
    limit = core730.round_down_2sf(value=HARM_FRACTION * s / (1.0 - s)) if s < 1.0 else 0.0
    return BandLimit(s, limit, int(keep.sum()))


def band_limits(*, x: np.ndarray, r: np.ndarray) -> dict[str, BandLimit]:
    """Return the pooled, 1 to 3 MVA, and 3 MVA and above limits from the test bank's truth alone."""
    lo, edge = DECIDABLE_READING_POINTS, BAND_EDGE_POINTS
    return {
        "pooled": band_limit(x=x, r=r, lo=lo, hi=np.inf),
        "1-3MVA": band_limit(x=x, r=r, lo=lo, hi=edge),
        "3MVA+": band_limit(x=x, r=r, lo=edge, hi=np.inf),
    }


NOISE_UNIT_NAMES: Final = ("D8", "D3", "D5", "D14")
"""The four series that have a noise unit, in points."""


def write_limits_file(
    *, path: Path, limits: Mapping[str, BandLimit], s: float, n_rows: int,
    noise_units: Mapping[str, float] | None = None,
) -> dict:  # fmt: skip
    """Write `limits.json` once (a second write raises `FileExistsError`) and return the record.

    Args:
        path: The file.
        limits: The harm limits of the three bands.
        s: The test bank's share of reverse-flow half-hours.
        n_rows: The test bank's scored half-hours.
        noise_units: The four noise units in points (D8, D3, D5, D14), from the training base days
            alone; the file holds them before any fit.

    Returns:
        The record that the file holds.
    """
    if noise_units is not None and set(noise_units) != set(NOISE_UNIT_NAMES):
        msg = f"the noise units must name exactly {NOISE_UNIT_NAMES}"
        raise ValueError(msg)
    record = {
        "s": s, "n_rows": n_rows,
        "bands": {k: {"share": v.share, "limit": v.limit, "n_rows": v.n_rows} for k, v in limits.items()},
        "noise_units": None if noise_units is None else dict(noise_units),
    }  # fmt: skip
    core730.write_once_json(path=path, payload=record)
    return record


def read_limits_file(*, path: Path) -> dict[str, BandLimit]:
    """Read `limits.json` and fail when it does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: the limits must be written before any fit"
        raise FileNotFoundError(msg)
    bands = json.loads(path.read_text())["bands"]
    return {
        k: BandLimit(share=v["share"], limit=v["limit"], n_rows=v["n_rows"])
        for k, v in bands.items()
    }


def read_noise_units(*, path: Path) -> dict[str, float]:
    """Read the four noise units from `limits.json` and fail when they are missing."""
    if not path.exists():
        msg = f"{path} does not exist: the limits must be written before any fit"
        raise FileNotFoundError(msg)
    units = json.loads(path.read_text()).get("noise_units")
    if not units or set(units) != set(NOISE_UNIT_NAMES):
        msg = f"{path} holds no complete set of noise units"
        raise KeyError(msg)
    return {k: float(v) for k, v in units.items()}


def limit_problems(*, limits: Mapping[str, BandLimit]) -> list[str]:
    """Return why the recomputed limits stop the line at verdict row 0 (an empty list when none).

    The pooled limit must equal experiment 730's 0.0071 and lie in the range, and every band must
    hold rows.
    """
    problems = []
    pooled = limits["pooled"].limit
    if not math.isclose(pooled, POOLED_LIMIT_730, rel_tol=0.0, abs_tol=1e-12):
        problems.append("the recomputed pooled limit differs from experiment 730's")
    if not LIMIT_RANGE[0] <= pooled <= LIMIT_RANGE[1]:
        problems.append("the pooled limit lies outside its range")
    problems += [f"the band {k} holds no rows" for k, v in limits.items() if v.n_rows == 0]
    return problems


# --------------------------------------------------------------------------- stop line and timing


def projected_hours(
    *, seconds_per_epoch: Mapping[str, float], epochs: int = MAX_EPOCHS,
    arms: Sequence[str] = FIT_ARMS,
) -> float:  # fmt: skip
    """Return the GPU-process-hours of the fits of `arms` at `epochs` epochs each, at each arm's own rate.

    Args:
        seconds_per_epoch: The timed seconds an epoch of each arm in `TIMING_ARMS`.
        epochs: The cap of every fit.
        arms: The arms fitted (all eight by default).

    Returns:
        The fits plus the inference allowance and the timing gate.
    """
    missing = sorted({RATE_OF[a] for a in arms} - set(seconds_per_epoch))
    if missing:
        msg = f"no timing for {missing}"
        raise KeyError(msg)
    fits = sum(FITS_PER_ARM * seconds_per_epoch[RATE_OF[a]] * epochs for a in arms) / 3600.0
    return fits + INFERENCE_HOURS + TIMING_GATE_HOURS


@dataclass(frozen=True)
class DropPlan:
    """The arms that the timing gate keeps and drops.

    Attributes:
        kept: The arms fitted, in `FIT_ARMS` order.
        dropped: The report-only arms dropped, in the order dropped (their readings are void).
        hours: The projection of the kept arms at the cap.
        stop: The 12 fits of the counted arms alone exceed the stop line (verdict row 0).
    """

    kept: tuple[str, ...]
    dropped: tuple[str, ...]
    hours: float
    stop: bool


def drop_report_only(
    *, seconds_per_epoch: Mapping[str, float], epochs: int = MAX_EPOCHS
) -> DropPlan:
    """Apply the stop-line rule: drop report-only arms in `DROP_ORDER` until the projection fits.

    If the counted arms alone still exceed the stop line, the plan says `stop`.
    """
    kept = list(FIT_ARMS)
    dropped: list[str] = []
    hours = projected_hours(seconds_per_epoch=seconds_per_epoch, epochs=epochs, arms=kept)
    for arm in DROP_ORDER:
        if hours <= STOP_LINE_HOURS:
            break
        kept.remove(arm)
        dropped.append(arm)
        hours = projected_hours(seconds_per_epoch=seconds_per_epoch, epochs=epochs, arms=kept)
    return DropPlan(tuple(kept), tuple(dropped), hours, stop=hours > STOP_LINE_HOURS)


def timing_gate_report(*, seconds_per_epoch: Mapping[str, float], peak_gpu_gb: float, peak_rss_gb: float,
                       pair_seconds_per_epoch: float) -> dict[str, object]:  # fmt: skip
    """Return the projection at the cap and whether it, the GPU memory, and the memory pass the gate.

    The projection is that of the arms kept after the drop rule; the dropped arms are listed.
    """
    plan = drop_report_only(seconds_per_epoch=seconds_per_epoch)
    hours = plan.hours
    return {
        "dropped_arms": list(plan.dropped), "counted_arms_over_line": plan.stop,
        "hours_at_cap": hours, "stop_line_hours": STOP_LINE_HOURS,
        "within_stop_line": hours <= STOP_LINE_HOURS, "peak_gpu_gb": peak_gpu_gb,
        "gpu_ok": peak_gpu_gb <= GPU_MEMORY_GB, "peak_rss_gb": peak_rss_gb,
        "memory_ok": peak_rss_gb <= MEMORY_CAP_GB, "pair_seconds_per_epoch": pair_seconds_per_epoch,
        "seconds_per_epoch": dict(seconds_per_epoch),
    }  # fmt: skip


def timing_gate_passes(*, report: Mapping[str, object], manager_approved: bool) -> bool:
    """Return whether fits may start: memory and GPU within bounds, and the kept arms within the line or approved."""
    return bool(
        report["memory_ok"]
        and report["gpu_ok"]
        and (report["within_stop_line"] or manager_approved)
    )


# --------------------------------------------------------------------------- the gallery's list

PANEL_REQUIREMENTS: Final = {
    "G10_transfers": "every transfer changes the truth by the drawn load on every half-hour inside it, each edge is marked, a partial half-hour at an edge shows a smaller step, and the BSP steps mark both edges only for a transfer to a Primary under another BSP",
    "G11_pairs": "every pair's re-folded reading equals the original where delta is zero, the pair differs only on sun-up half-hours, and the truth drops by delta",
    "G12_raised_sky": "every raised index is above the original on every raisable half-hour and equal elsewhere, with the reading unchanged",
    "G13_targets": "the solar target, the step events, and the step sizes are non-empty for the world that carries them and aligned with their cause, with the hidden span shaded",
    "G14_parameters": "every histogram of a drawn transfer parameter lies inside its registered range, and the settled T0 after the donor rule is split by direction, partner, and duration",
    "G15_everything": "the window with an engine run, S2 solar output, a load transfer, and an extra injection shows every component and the injected member differs from the plain one",
    "G16_pairing": "every month of every training seed has a sibling pair for at least 95% of its windows",
    "G17_s1_pairs": "the pairs built from source S1 are written to the private folder only",
    "G18_neighbours": "every mirror is equal and opposite at the drawn partner and absent at the other neighbours, the neighbour moves by j times the transfer in the jittered case, and no neighbour moves when the partner is not read",
    "G19_shapes": "a constant, a demand-shaped, and a solar-shaped transfer of the same T0 differ in shape, share the mean T0, never fall below 0.1 T0, and leave the solar target unchanged",
    "G20_neighbour_table": "every neighbour covers at least 90% of the base-day half-hours over the whole table, the drawn sizes in D8's noise units reach both sides of 4, and every mirror check point lies on the line of equality",
    "G21_input_tensor": "every neighbour channel is flat where the list says it must move or zero inside the hidden span, the injected member differs from the plain one only in the reading, BSP, and sign-aligned channels, and U-N and U-noBSP differ from U* only in the neighbour and BSP channels",
}
"""The written list of what each new gallery panel must show. The session that runs the experiment
checks each panel against it and records the result in `PRIVATE/gallery/panel_check.json`."""
GALLERY_PRIVATE_ONLY: Final = ("G17_s1_pairs",)
GALLERY_PUBLISHED: Final = tuple(k for k in PANEL_REQUIREMENTS if k not in GALLERY_PRIVATE_ONLY)

SERIES_WINDOWS: Final = {
    "W1": "an evening engine run, the window with the median depth of reverse flow",
    "W2": "a morning engine run, the window with the median depth of reverse flow",
    "W3": "solar output alone, the window with the median depth of reverse flow",
    "W4": "engine and solar together, the window with the median depth of reverse flow",
    "W5": "a window whose reading comes within 0.3 MVA of zero on at least 4 half-hours",
    "W6": "the import window with U*'s largest mean P(X<0) on decidable rows",
    "W7": "the window on which U* gains most against S (730) in Brier score",
    "W8": "the window on which U* loses most against S (730) in Brier score",
    "W9": "one injection pair at an amount of 5 points",
    "W10": "one injection pair at an amount of 20 points",
    "W11": "one injection pair at an amount of 50 points",
    "W12": "the engine-only window with U*'s largest solar output",
    "W13": "the clear S2 window with U*'s largest solar error",
    "W14": "a load-transfer window that adds load",
    "W15": "a second load-transfer window that adds load",
    "W16": "a load-transfer window that removes load",
    "W17": "a second load-transfer window that removes load",
    "W18": "a load-transfer window inside a solar spell",
    "W19": "a load-transfer window at a partner under another BSP",
    "W20": "a solar-shaped transfer with a mirror",
    "W21": "a solar-shaped transfer without a mirror",
    "W22": "the mirrored transfer with U*'s largest gain over U-N in the step Brier score",
    "W23": "the transfer of at least 4 noise units that U* misses by the widest margin",
    "W24": "a real pre-engine window around a logged closure at D8 (reading and step probability only)",
    "W25": "a second real pre-engine window around a logged closure at D8",
}
"""The 25 single-window time series figures, each picked by a fixed rule. W20 to W25 are the windows
that the transfers, the neighbours, and the real closures add; W24 and W25 draw only if K3 is not void."""
SERIES_PRIVATE_SOURCES: Final = ("S1",)
"""A window whose solar output comes from this source goes to the private folder."""

# --------------------------------------------------------------------------- the schedule


def fit_jobs() -> list[tuple[str, str, int]]:
    """Return the 24 fits as (arm, architecture, seed), in the order of the families."""
    return [
        (arm, ARMS[arm].arch, seed) for _, arms in FIT_FAMILIES for arm in arms for seed in SEEDS
    ]


def family_of(*, arm: str) -> str:
    """Return the name of the family of fits that holds `arm`."""
    for name, arms in FIT_FAMILIES:
        if arm in arms:
            return name
    msg = f"no family holds arm {arm!r}"
    raise KeyError(msg)


def job_name(*, arm: str, arch: str, seed: int) -> str:
    """Return the file stem of a fit (`U*` becomes `Ustar`, so no name holds an asterisk)."""
    return f"{arm.replace('*', 'star').replace('+', 'plus')}_{arch}_s{seed}"


def stage_names() -> Sequence[str]:
    """Return the stage names of the command line, in the order of the plan."""
    return (
        "banks",
        "neighbours",
        "limits",
        "overlays",
        "gallery",
        "panels",
        "tensor",
        "solar_diag",
        "timing",
        "fit",
        "real",
        "score",
        "report",
        "figures",
        "loss_curves",
        "series",
        "chain",
    )
