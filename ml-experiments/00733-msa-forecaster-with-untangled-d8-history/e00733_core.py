# ruff: noqa: E501
"""Experiment 733: every named constant, bar, arm, contrast, and verdict row of the plan.

Every number that the page pre-registers is a `Final` here, so the page and the code are compared by
reading this one file. This module imports no other experiment, so that a reader can check the
registered numbers without following an import. The functions are pure. No function prints a date,
a series identifier, or a real value.

The plan is `docs/ml-experiments/00733-msa-forecaster-with-untangled-d8-history/result.md`; each
section name in a docstring below is a section of that page.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

# --------------------------------------------------------------------------- identity

EXPERIMENT_ID: Final = "00733"
SLUG: Final = "msa-forecaster-with-untangled-d8-history"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
CODE_VERSION: Final = 1
"""Written into every fit record and chain status. Raise it with every change that can alter a fitted state."""
ROW_SET_ID_REAL: Final = "backtest_2025_11_to_2026_06_daily_12utc_before_holdout"
ROW_SET_ID_SYNTHETIC: Final = "e00733_synthetic_forecast_d8_v1"
SCORED_PRIMARY: Final = "D8"
FIRST_UNSCORED_VALID_TIME: Final = "2026-07-01"
"""No valid time on or after this date is read."""

# --------------------------------------------------------------------------- folds, seeds, and the history grid

ALL_FOLDS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
SCREEN_FOLDS: Final = ("2025-11", "2026-01", "2026-05", "2026-06")
SCREEN_SEEDS: Final = (0, 1)
F0R_FOLDS: Final = ("2025-11", "2026-05")
F0R_SEEDS: Final = (0, 1)
HEADLINE_SEEDS: Final = (0, 1, 2, 3, 4)
PLACEBO_SEEDS: Final = (0, 1, 2)
SYNTHETIC_SEEDS: Final = (0, 1, 2)
WORLD_WEATHER_FOLD: Final = "2025-11"
"""The fold whose standardiser the synthetic world's weather columns use."""
N_HISTORY_ROWS: Final = 14
MIN_AGE: Final = 2
MAX_AGE: Final = MIN_AGE + N_HISTORY_ROWS - 1
STEPS_PER_DAY: Final = 48
GRID_START_DATE: Final = "2019-09-01"
"""The grid of experiment 422, 713, and 716 starts at 00:00 UTC on this date; a stamp is a period end."""
NEWEST_VISIBLE_BEFORE_CONTRACT: Final = 54
"""Half-hour steps from the newest visible stamp to the contract origin (27 hours), as experiment 713 fixes."""
NATIVE_OFFSET_STEPS: Final = 6
"""Steps from the native origin (09:00 UTC) to the contract origin (12:00 UTC)."""
ENSEMBLE_AVAILABLE_AFTER_HOURS: Final = 9
"""An ENS 00Z run is available this many hours after its initialisation time."""
WINDOW_STEPS: Final = 96
"""The half-hours of one untangling window (48 hours); its scored middle is the 48 steps from step 24."""
WINDOW_MIDDLE_START: Final = 24
PLACEBO_SHIFT_DAYS: Final = 28
MIN_HISTORY_ROWS: Final = 7
"""A synthetic sample is kept when at least this many of its 14 history rows exist."""
SYNTHETIC_LEAD_DAYS: Final = tuple(range(2, 11))
"""The scored lead days of the synthetic forecasting world."""
HEADLINE_LEAD_HOURS: Final = (24, 240)
"""Scored leads: from 24 h up to but not including 240 h after the 12:00 UTC origin."""
BAND_LEAD_DAYS: Final = ((2, 3), (4, 6), (7, 10))
MIDDAY_HOURS_UTC: Final = (10, 14)
LATE_SPRING_SUMMER_FOLDS: Final = ("2026-04", "2026-05", "2026-06")

# --------------------------------------------------------------------------- the untangled channels

CHANNEL_NAMES: Final = ("p_neg", "solar", "step_p")
"""The three channels: P(X<0), the solar output in points, and the transfer-step probability."""
N_CHANNELS: Final = len(CHANNEL_NAMES)
P_NEG, SOLAR, STEP_P = 0, 1, 2
SOLAR_CLIP_MAX: Final = 10.0
"""The solar channel `v` is clipped to 0 to this, in the units of the power input."""
COVERAGE_MIN: Final = 0.90
"""The channel table must cover at least this share of D8's visible history half-hours per fold."""
HISTORY_HIDDEN_WIDTH: Final = 32
HISTORY_OUT_WIDTH: Final = 64
HISTORY_INPUT_WIDTH: Final = N_CHANNELS + 1
"""The path's input is the three channels, each times the flag, and the flag."""
MASK_DROPOUT_SHARE: Final = 0.10
"""The share of training samples that get each of the two masks (age 2 and part of age 3; ages 2 to 5)."""
CAMS_RULE_FIRST_MASKED_COLUMN: Final = 25
LEARNING_RATES: Final = {"gamma": 3e-3, "w1_w2_head_map": 2e-3, "readout_table": 1e-2}

# Which untangled signal is usable: the rule of "Which untangled signals enter".
P_NEG_USABLE_ROWS_732: Final = (4, 5)
"""Experiment 732's verdict rows 4 (`supported`) and 5 (`inconclusive`) leave U* usable for P(X<0)."""
P_NEG_REQUIRED_CHECKS: Final = ("h2",)
"""Experiment 732 never scored K2, so the rule reads H2 alone (ruling of 2026-11-26)."""
SOLAR_USABLE_WORDS: Final = ("reasonable", "partly")
SOLAR_REQUIRED_CHECKS: Final = ("p1", "p3")
TRANSFER_USABLE_WORDS: Final = ("works", "works only in Brier score")
SOURCE_732: Final = "U*"
SOURCE_730: Final = "S"
SOURCE_SEEDS: Final = (0, 1, 2)

# --------------------------------------------------------------------------- the synthetic forecasting world

SYNTHETIC_TRAIN_BANK_SEEDS: Final = (3, 4)
"""Experiment 730's training banks that no fit of experiment 732 read."""
SYNTHETIC_LR: Final = 5e-4
SYNTHETIC_WEIGHT_DECAY: Final = 1e-3
SYNTHETIC_BATCH: Final = 256
SYNTHETIC_CLIP: Final = 1.0
SYNTHETIC_MAX_EPOCHS: Final = 80
SYNTHETIC_PATIENCE: Final = 20
SYNTHETIC_LOSS_WEIGHTS: Final = {"mixture_nll": 1.0, "sign_bce": 1.0}
SIGN_HEAD_HIDDEN: Final = 32
SIGN_HEAD_EXTRA_INPUTS: Final = 16
"""The sign head reads 64 trunk features plus these: the logit of the weighted P(X<0), the cover, a 14-way lead-day one-hot."""
B_HIST_WEIGHT_SAME_DAY_TYPE: Final = 3.0
B_HIST_WEIGHT_TWO_WORKING_DAYS: Final = 1.5
B_HIST_WEIGHT_OTHER: Final = 0.0
MAX_MISSING_TEST_SEGMENT_SHARE: Final = 0.5
"""More than this share of test segments without ENS runs stops the line at verdict row 0."""
TOLERANCE_ZERO_PATH: Final = 1e-6
TOLERANCE_POISON: Final = 1e-6
TOLERANCE_F0R: Final = 1e-6
G14_MEAN_LIMIT: Final = 0.01
"""G14: the mean absolute stored-minus-re-run difference of P(X<0) over origins more than
`G14_FAR_HOURS` before the visibility limit must stay below this (ruling of the design lead,
2026-11-26)."""
G14_FAR_HOURS: Final = 24.0
G14_BIN_HOURS: Final = 6.0
G14_NEAR_FACTOR: Final = 3.0
"""G14 (c'): the mean difference in the nearest 6-hour bin (0 to 6 hours before the limit) must be at
least this many times the mean difference over all half-hours more than `G14_DECAY_HOURS` before the
limit (ruling of the design lead, 2026-11-27, which replaces a monotone-decay rule because the real
differences decay noisily)."""
G14_DECAY_HOURS: Final = 12.0
"""G14 (c'), (c''): the distance before the limit beyond which a bin counts as far."""

# --------------------------------------------------------------------------- contrasts and bars

N_COUNTED_CONTRASTS: Final = 4
LEVEL: Final = 1.0 - 0.05 / N_COUNTED_CONTRASTS
"""Bonferroni level across C1 to C4 (98.75%). Every report-only reading is read at the same level."""
N_RESAMPLES_REAL: Final = 100_000
N_RESAMPLES_SYNTHETIC: Final = 20_000
GENERATOR_SEED: Final = 733
BLOCK_DAYS: Final = 7
KISH_WIDEN_BELOW: Final = 20.0
REAL_FOLD_FACTOR_DF: Final = len(ALL_FOLDS) - 1
"""One Primary leaves one factor, the 8 fold months, so the widening uses a Student t with 7 degrees of freedom."""
SMALLEST_GAIN: Final = 0.02
"""The smallest gain worth porting, g: 2% of M*'s CRPS at D8 on the headline band."""
SCREEN_STOP_POINT: Final = 0.02
"""A screen point above +2% of M*'s CRPS stops the rest of FU and FU-S."""
TAIL_PINBALL_MARGIN_POINTS: Final = 0.05
"""FU is not worse than M* by more than this many points on the q01 or q99 pinball loss."""
SKILL_BAR: Final = 0.10
RELIABILITY_FACTOR: Final = 0.1
"""The reliability term is at most this times the test rows' share of reverse flow (experiment 730's bar)."""
MIN_REVERSE_HALF_HOURS: Final = 500
HARM_LIMIT_730: Final = 0.0071
READING_BANDS_MVA: Final = (1.0, 3.0)

# --------------------------------------------------------------------------- compute

SLOT: Final = "A2"
SLOT_CORES: Final = "16-21"
SLOT_MEMORY_GB: Final = 14
MIN_AVAILABLE_GB: Final = 25
MAX_LOAD: Final = 24
GPU_GB: Final = 6
HOURS_REAL_FIT: Final = 0.08
HOURS_SYNTHETIC_FIT: Final = 0.3
STOP_LINE_HOURS: Final = 12.0
CHAIN_FAILED_EXIT: Final = 5
"""A failed fit or check ends the chain with this exit code, which the launcher does not read as "slot busy"."""
WORK_HOURS: Final = {
    "smoke": 0.15, "timing_gate": 0.15, "reruns": 0.2, "F0r": 0.32, "FU": 3.2, "FU-S": 1.92,
    "Y0 and YU": 1.8, "FS": 1.92, "YT": 0.9,
}  # fmt: skip
COUNTED_WORK: Final = ("smoke", "timing_gate", "reruns", "F0r", "FU", "FU-S", "Y0 and YU")
REPORT_ONLY_WORK: Final = ("FS", "YT")
RSS_STOP_GB: Final = 13.0
"""A job whose peak resident memory exceeds this stops the chain (the slot holds 14 GB)."""
PLANNED_HOURS: Final = 10.56
"""The pre-registered cost, in GPU-process-hours."""

# --------------------------------------------------------------------------- arms

Where = Literal["real", "synthetic"]
Source = Literal["stored", "fit"]


@dataclass(frozen=True)
class ArmSpec733:
    """One arm of the plan.

    Attributes:
        name: The arm's name.
        where: "real" folds or the "synthetic" forecasting world.
        history: What the 14 history rows read.
        sign_head: Whether the arm has the sign head.
        seeds: The seeds the arm runs.
        folds: The folds the arm runs on (empty on the synthetic world).
        role: The arm's role.
        counted: Whether a counted contrast reads the arm.
        source: "stored" (read from experiment 713's stored forecasts) or "fit".
        channels: Whether the usable untangled channels enter the history rows.
        shifted_days: How many days earlier each channel is taken (the placebo), or 0.
        channel_subset: The channels the arm reads, by name (`None` means every usable one).
        true_channels: Whether the history reads the true signs and solar output (the oracle arm).
    """

    name: str
    where: Where
    history: str
    sign_head: bool
    seeds: tuple[int, ...]
    folds: tuple[str, ...]
    role: str
    counted: bool
    source: Source = "fit"
    channels: bool = False
    shifted_days: int = 0
    channel_subset: tuple[str, ...] | None = None
    true_channels: bool = False

    @property
    def n_fits(self) -> int:
        """Return the number of fits of the arm (zero for a stored arm)."""
        if self.source == "stored":
            return 0
        return len(self.seeds) * max(1, len(self.folds))

    @property
    def hours_per_fit(self) -> float:
        """Return the planned GPU-process-hours of one fit."""
        return HOURS_SYNTHETIC_FIT if self.where == "synthetic" else HOURS_REAL_FIT


ARMS: Final = {
    a.name: a
    for a in (
        ArmSpec733(
            "M*",
            "real",
            "The folded reading",
            False,
            HEADLINE_SEEDS,
            ALL_FOLDS,
            "The control of C1",
            True,
            source="stored",
        ),
        ArmSpec733(
            "F0r",
            "real",
            "The folded reading, every new channel absent",
            False,
            F0R_SEEDS,
            F0R_FOLDS,
            "Gate: reproduces M* exactly",
            False,
        ),
        ArmSpec733(
            "FU",
            "real",
            "The folded reading and every usable untangled channel",
            False,
            HEADLINE_SEEDS,
            ALL_FOLDS,
            "The headline arm",
            True,
            channels=True,
        ),
        ArmSpec733(
            "FU-S",
            "real",
            "As FU, with each channel taken 28 days earlier",
            False,
            PLACEBO_SEEDS,
            ALL_FOLDS,
            "The placebo of C2",
            True,
            channels=True,
            shifted_days=PLACEBO_SHIFT_DAYS,
        ),
        ArmSpec733(
            "FS",
            "real",
            "The folded reading and P(X<0)",
            False,
            PLACEBO_SEEDS,
            ALL_FOLDS,
            "Report only: the sign alone",
            False,
            channels=True,
            channel_subset=("p_neg",),
        ),
        ArmSpec733(
            "Y0",
            "synthetic",
            "The folded reading",
            True,
            SYNTHETIC_SEEDS,
            (),
            "The control of C4",
            True,
        ),
        ArmSpec733(
            "YU",
            "synthetic",
            "The folded reading and the untangled channels",
            True,
            SYNTHETIC_SEEDS,
            (),
            "The headline arm of the sign forecast",
            True,
            channels=True,
        ),
        ArmSpec733(
            "YT",
            "synthetic",
            "The folded reading and the true channels",
            True,
            SYNTHETIC_SEEDS,
            (),
            "Report only: the oracle history",
            False,
            true_channels=True,
        ),
    )
}
FIT_ORDER: Final = ("F0r", "FU-screen", "FU", "FU-S", "Y0", "YU", "FS", "YT")
"""The fits run in this order, so that a stop at the stop line loses report-only arms first."""
REPORT_ONLY_ARMS: Final = ("FS", "YT")
DROP_ORDER: Final = ("YT", "FS")
"""When the projection exceeds the stop line, report-only arms are dropped in this order."""
BASELINES_SYNTHETIC: Final = ("B-hist", "B-clim", "O")

# --------------------------------------------------------------------------- contrasts and readings


@dataclass(frozen=True)
class ContrastSpec:
    """A contrast: the first arm minus the second, in a metric, on rows. A negative difference is a gain."""

    name: str
    first: str
    second: str
    metric: str
    rows: str
    seeds: tuple[int, ...]
    question: str


CONTRASTS: Final = {
    c.name: c
    for c in (
        ContrastSpec(
            "C1",
            "FU",
            "M*",
            "CRPS",
            "D8, real folds, leads 24 h to 240 h",
            HEADLINE_SEEDS,
            "Do the untangled history signals improve the forecast of D8's reading?",
        ),
        ContrastSpec(
            "C2",
            "FU",
            "FU-S",
            "CRPS",
            "D8, real folds, leads 24 h to 240 h",
            PLACEBO_SEEDS,
            "Does the gain come from the untangled values of each history day?",
        ),
        ContrastSpec(
            "C3",
            "YU",
            "B-hist",
            "Brier score of P(X<0)",
            "Synthetic test target rows, lead days 2 to 10",
            SYNTHETIC_SEEDS,
            "Does the sign head forecast reverse flow better than persistence of the untangled history?",
        ),
        ContrastSpec(
            "C4",
            "YU",
            "Y0",
            "Brier score of P(X<0)",
            "Synthetic test target rows, lead days 2 to 10",
            SYNTHETIC_SEEDS,
            "Do the untangled history signals help the sign forecast?",
        ),
    )
}
READINGS: Final = {
    "R1": "FS minus M*, CRPS at D8: the sign alone",
    "R2": "FU minus FS, CRPS at D8: the solar and transfer channels beyond the sign",
    "R3": "FU minus M*, CRPS pooled over the other 15 Primaries: harm away from D8",
    "R4": "FU with every untangled channel masked at prediction, minus FU, CRPS at D8",
    "R5": "YT minus Y0, Brier score of P(X<0): the ceiling of the history path",
    "R6": "YU minus Y0, CRPS of the forecast reading on the synthetic world",
    "R7": "YU minus O, Brier score of P(X<0): the distance to the untangling network on the target day",
    "R8": "C1 by lead band, with each fold month left out in turn, in the midday half-hours of folds 2026-04 to 2026-06, and on cold winter nights",
    "R9": "The q01, q05, q95, and q99 pinball loss and coverage, and the avoidable-curtailment cost, for FU, FS, and M* at D8",
    "R10": "The four load-shift scenarios at D8 for FU, FS, and M*",
    "R11": "The sign head of YU by lead band and by reading band",
    "R12": "FU scored with the stored untangled outputs of days D - 2 and D - 1 in place of the re-run, minus FU",
}
N_READINGS: Final = len(READINGS)

# --------------------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class Reading:
    """A contrast's point estimate and interval, relative to the control's score where the page says so."""

    point: float
    lower: float
    upper: float

    @property
    def finite(self) -> bool:
        """Return whether the point and both ends are finite."""
        return all(math.isfinite(v) for v in (self.point, self.lower, self.upper))

    @property
    def gain(self) -> bool:
        """Return whether the interval's upper end is below zero (a gain of the first arm)."""
        return self.finite and self.upper < 0.0

    @property
    def loss(self) -> bool:
        """Return whether the interval's lower end is above zero (a loss of the first arm)."""
        return self.finite and self.lower > 0.0


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match verdict table reads.

    Attributes:
        stopped: Row 0 applies.
        invalid: A correctness check of row 1 failed.
        failed: A fit that C1 or C2 needs raised and was not rerun (row 2).
        c1: C1, relative to M*'s CRPS at D8 on the headline band (None when unread).
        c2: C2, on the same scale.
        guards_hold: C1 keeps its sign leaving each fold month out, the mean absolute error of the
            median moves with CRPS, and FU is not worse than M* by more than 0.05 points on q01 or q99.
    """

    stopped: bool
    invalid: bool
    failed: bool
    c1: Reading | None
    c2: Reading | None
    guards_hold: bool


VERDICT_TABLE: Final = (
    (0, "The line stops before C1 is read", "stopped"),
    (1, "A correctness check fails", "invalid"),
    (2, "A fit that C1 or C2 needs raised and was not rerun", "failed"),
    (3, "C1 and C2 are gains, C1's point is at or below -g, and the guards hold", "supported"),
    (4, "C1 and C2 are gains and the guards hold, and C1's point is above -g", "small gain"),
    (5, "C1 is a gain, and C2 is not a gain or a guard fails", "partly supported"),
    (6, "C1 is a loss", "refuted"),
    (7, "Every other case (C1's interval includes zero)", "not supported"),
)
"""The verdict table of the plan: the first row that matches is the verdict."""
FRONT_MATTER_WORD: Final = {"stopped": "inconclusive"}
STOP_REASONS: Final = (
    "the manager halts the line",
    "F0r differs from M*",
    "the channel table covers less than 90% of a fold's visible history half-hours",
    "a bank digest differs from experiment 730's",
    "more than half of the synthetic test segments lack ENS runs",
    "experiment 732 has not finished",
    "a gallery panel does not show what its written list requires",
    "the timing projection of the counted fits exceeds the stop line",
    "a review gate has not passed when a fit would start",
)
CORRECTNESS_CHECKS: Final = (
    "an origin-time weather assertion",
    "a leakage guard",
    "the poisoned-copy checks",
    "the bit-for-bit check of the zero-initialised path",
    "the starting-rule check of the sign head",
    "a known-answer unit test of the bootstrap or the Brier score",
)


def verdict_row(*, v: VerdictInputs) -> int:  # noqa: PLR0911
    """Return the first row of the verdict table that matches (0 to 7).

    A counted reading that is missing or not finite is row 1 (`invalid`), never row 7: C1 always,
    and C2 whenever C1 is a gain, because only then does C2 change the row.
    """
    if v.stopped:
        return 0
    if v.invalid:
        return 1
    if v.failed:
        return 2
    c1, c2 = v.c1, v.c2
    if c1 is None or not c1.finite:
        return 1
    if c1.gain and (c2 is None or not c2.finite):
        return 1
    c1_gain = c1 is not None and c1.gain
    c2_gain = c2 is not None and c2.gain
    if c1_gain and c2_gain and v.guards_hold:
        assert c1 is not None
        return 3 if c1.point <= -SMALLEST_GAIN else 4
    if c1_gain:
        return 5
    if c1 is not None and c1.loss:
        return 6
    return 7


def verdict_word(*, v: VerdictInputs) -> str:
    """Return the verdict word of `verdict_row`."""
    return VERDICT_TABLE[verdict_row(v=v)][2]


def front_matter_word(*, word: str) -> str:
    """Return the word that the front matter carries (row 0 is `inconclusive`)."""
    return FRONT_MATTER_WORD.get(word, word)


def sign_forecast_word(
    *,
    c3: Reading | None,
    skill: float,
    reliability: float,
    share_reverse: float,
    n_reverse: int,
    seeds_present: bool,
) -> str:
    """Return the sign-forecast word, which never changes the verdict.

    Args:
        c3: C3 (YU minus B-hist, Brier score).
        skill: YU's Brier skill against B-clim.
        reliability: YU's reliability term.
        share_reverse: The scored test rows' share of reverse flow.
        n_reverse: The number of reverse-flow half-hours among the scored target rows.
        seeds_present: Whether every seed of YU is present.

    Returns:
        "skilful", "beats persistence, little skill", "beats persistence, miscalibrated", "no better
        than persistence", "worse than persistence", or "void".
    """
    if c3 is None or not c3.finite or not seeds_present or n_reverse < MIN_REVERSE_HALF_HOURS:
        return "void"
    if c3.loss:
        return "worse than persistence"
    if not c3.gain:
        return "no better than persistence"
    if not skill >= SKILL_BAR:
        return "beats persistence, little skill"
    if not reliability <= RELIABILITY_FACTOR * share_reverse:
        return "beats persistence, miscalibrated"
    return "skilful"


def history_word(*, c4: Reading | None, r5: Reading | None) -> tuple[str, str]:
    """Return the history word and the R5 note printed beside it.

    Args:
        c4: C4 (YU minus Y0, Brier score).
        r5: R5 (YT minus Y0, Brier score).

    Returns:
        The word ("helps", "no detectable difference", "hurts", or "void") and the note. The note
        says that a null C4 tells nothing about the untangling network when R5 is not a gain.
    """
    if c4 is None or not c4.finite:
        return "void", "C4 is unread"
    word = "helps" if c4.gain else "hurts" if c4.loss else "no detectable difference"
    if r5 is None or not r5.finite:
        return word, "R5 is unread"
    if r5.gain:
        return word, "R5 is a gain: the true sign in the history helps the sign head"
    return word, (
        "R5 is not a gain: even the true sign in the history does not help the sign head, "
        "so a null C4 says nothing about the untangling network"
    )


def t_widening(*, df: float, level: float = LEVEL) -> float:
    """Return the ratio of the two-sided Student t quantile to the normal quantile at `level`.

    The real folds have one factor of 8 fold months (7 degrees of freedom), so the interval is
    widened by this ratio, as experiments 729 to 732 widen theirs.
    """
    from scipy import stats

    q = 0.5 + level / 2.0
    return float(stats.t.ppf(q, df) / stats.norm.ppf(q))


def planned_hours(*, names: tuple[str, ...]) -> float:
    """Return the planned GPU-process-hours of the named work items."""
    return round(sum(WORK_HOURS[n] for n in names), 2)


def projected_hours(
    *, seconds_per_real_fit: float, seconds_per_synthetic_fit: float, dropped: tuple[str, ...] = ()
) -> float:
    """Return the GPU-process-hours of the fits at a measured rate, with `dropped` arms left out.

    FU-S, FS, and F0r run at FU's rate; Y0 and YT at YU's, as the plan's timing gate projects.
    """
    real = sum(a.n_fits for n, a in ARMS.items() if a.where == "real" and n not in dropped)
    synthetic = sum(
        a.n_fits for n, a in ARMS.items() if a.where == "synthetic" and n not in dropped
    )
    return (real * seconds_per_real_fit + synthetic * seconds_per_synthetic_fit) / 3600.0


def stop_arms(*, hours: Mapping[str, float] | None = None, projected: float) -> tuple[str, ...]:
    """Return the report-only arms to drop so that the projection fits under the stop line.

    Args:
        hours: The projected GPU-process-hours of each report-only arm (defaults to the plan's table).
        projected: The projection of every arm, in GPU-process-hours.

    Returns:
        The arms to drop, in `DROP_ORDER`, until the projection fits (empty when it already does).
    """
    table = dict(hours or {"YT": WORK_HOURS["YT"], "FS": WORK_HOURS["FS"]})
    dropped: list[str] = []
    for arm in DROP_ORDER:
        if projected <= STOP_LINE_HOURS:
            break
        projected -= table[arm]
        dropped.append(arm)
    return tuple(dropped)


# --------------------------------------------------------------------------- the gallery

GALLERY_PANELS: Final = {
    "G10": (
        "the base days of the month on an axis of days, for three test segments",
        "the 14 history rows and the target rows of lead days 1 to 14 inside the segment marked",
        "the scored lead days 2 to 10 shaded and the segment's edges shaded",
        "every history and target row inside the segment is shown",
    ),
    "G11": (
        "for three S2 test segments: the folded reading, the signed truth, the injected engine, and the injected solar output over 10 target days",
        "reverse-flow half-hours shaded",
        "S1 segments as an aggregate of the share of reverse flow by hour of day",
        "reverse flow on target days in at least two of the three segments",
    ),
    "G12": (
        "the correlation by lead day between the ENS shortwave of the base date's 00Z run and the CAMS sky of the same target half-hours",
        "the correlation falls with lead day and stays above zero",
    ),
    "G13": (
        "for three S2 test segments: the history's untangled P(X<0) against the indicator of X < 0",
        "the untangled solar output against the injected solar output, over 14 history days",
        "a nonzero solar channel on sun-up half-hours of solar segments",
        "a P(X<0) above 0.5 on most reverse-flow half-hours of at least 1 MVA",
    ),
    "G14": (
        "the distribution of the difference between the stored and the re-run outputs on the synthetic validation bank, by hour after the visibility limit",
        "the poison test passes: NaN and 1e6 placed in every hidden input after the limit leave the re-run output unchanged",
        "the mean absolute difference over origins more than 24 hours before the limit is below 0.01",
        "the mean absolute difference in the 6 hours before the limit is at least 3 times the mean more than 12 hours before it, and every 6-hour bin more than 12 hours before the limit has a mean below 0.01",
    ),
    "G15": (
        "the share of reverse flow in the synthetic target rows by lead day and by hour of day, with B-clim beside it",
        "a share above zero on sun-up half-hours",
    ),
}
"""The written list of what each new gallery panel must show (G1 to G9 are experiment 730's)."""

# --------------------------------------------------------------------------- limits file

LIMITS_KEYS: Final = ("smallest_gain", "b_clim", "n_synthetic")
"""`limits.json` holds g, the climatology of reverse flow, and the counts of the synthetic world."""


def limits_payload(
    *, b_clim: Mapping[str, object], n_synthetic: Mapping[str, int]
) -> dict[str, object]:
    """Return the content of `limits.json`: g, B-clim, and the counts, fixed before any fit."""
    return {
        "smallest_gain": SMALLEST_GAIN,
        "b_clim": dict(b_clim),
        "n_synthetic": dict(n_synthetic),
    }
