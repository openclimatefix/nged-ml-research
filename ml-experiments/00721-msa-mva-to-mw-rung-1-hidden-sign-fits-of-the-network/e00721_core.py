# ruff: noqa: E501
"""Experiment 721, rung 1: the named constants, the bars, the flag, and the verdict rows.

The flag is the stopped-while-falling test.

Every number that the page's decision rule pre-registers is a `Final` here, so the page and the code
are compared by reading this one file. The functions are pure: they read numbers and return numbers,
and they touch no data file (except `write_mde_file`, which refuses to overwrite).

The verdict table is first-match and `verdict_row` is its only implementation. The bars of G1 to G3
are `bar_g1_g2` and `check_g3`, and the flag of the training budget is `stopped_while_falling`.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- identity and rows

EXPERIMENT_ID: Final = "00721"
SLUG: Final = "msa-mva-to-mw-rung-1-hidden-sign-fits-of-the-network"
FOLDER: Final = f"{EXPERIMENT_ID}-{SLUG}"
ROW_SET_ID: Final = "backtest_2025_11_to_2026_06_daily_12utc_before_holdout"
"""Experiment 713's row set, restricted to `FOLDS`."""
FOLDS: Final = ("2025-12", "2026-04", "2026-05", "2026-06")
"""The four folds. Fold 2025-12 is the dark fold, kept for the no-harm reading."""
DARK_FOLD: Final = "2025-12"
SUNNY_FOLDS: Final = ("2026-04", "2026-05", "2026-06")
MDE_FOLD: Final = "2026-05"
"""The fold on which the two seeds of `SG-0` set the second term of the minimum detectable
effect."""
SEEDS: Final = (0, 1)
"""Seeds 0 and 1 give the verdict. Seeds 2 to 4 run only after the futility test and give a
secondary read."""
EXTRA_SEEDS: Final = (2, 3, 4)
HOLDOUT_START_ISO: Final = "2026-07-01"
"""No valid time on or after this date is read."""
LEAD_MIN_HOURS: Final = 24.0
LEAD_MAX_HOURS_EXCLUSIVE: Final = 240.0
HIDDEN_PAIR: Final = ("D5", "D10")
"""The two signed-MW Primaries whose sign is hidden (rotation r1 of experiment 420)."""
VISIBLE_SIGNED: Final = ("D9", "D11", "D14", "D16")
"""The four signed-MW Primaries that stay visible and supervise the sign head."""
MVA_ONLY: Final = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The eight MVA-only Primaries (G4)."""
UNFOLDED_SIGNED: Final = ("D15", "D18")
"""The two short-history Primaries whose stored readings are signed (negative on some half-hours).
The page names no folding for them: they train and score on the readings as stored, with the plain
likelihood, no export gate, and no sign labels."""
SIGN_SUPERVISION_EXCLUDED: Final = (*HIDDEN_PAIR, "D15")
"""Primaries kept out of the sign head's supervision in every world and stage."""

# --------------------------------------------------------------------------- the bars

MDE_FLOOR_POINTS: Final = 0.404
"""Experiment 420's rotation r1 figure of the minimum detectable effect, in points of capacity."""
G3_LIMIT_POOLED_POINTS: Final = 0.10
"""G3's limit on the point estimate pooled over the two hidden Primaries (decision 8, Option A)."""
G3_LIMIT_PER_PRIMARY_POINTS: Final = 0.20
"""G3's limit on the point estimate at each hidden Primary (decision 8, Option A)."""
SIGN_ACCURACY_BAR: Final = 0.95
"""The synthetic world's sign accuracy outside the dead band at the two exporting Primaries."""
FALSE_CALL_RATE_MAX: Final = 0.01
"""The synthetic world's share of sunny midday half-hours at which a non-exporter is called reverse
flow."""
DEAD_BAND_POINTS: Final = 2.0
"""A half-hour with |signed power| under this many points is outside the scored sign classes."""
REACTIVE_FLOOR_POINTS: Final = 2.0
"""The reactive floor `Q` of the folded reading `sqrt(X² + Q²)`, in points of capacity."""
SHIFT_DAYS: Final = 28
"""The shifted-sky placebo reads the sky from this many days earlier."""
U_MIN: Final = 0.3
"""The export gate's `u_min` and the likelihood floor of the margin."""
SYNTH_MAX_PASSES: Final = 400
SYNTH_PATIENCE: Final = 30
INJECTION_LEVEL: Final = 2.0
"""The injected farm's size as a multiple of the host's median absolute sunny summer-midday
reading."""
BLOCK_DAYS: Final = 10
"""The bootstrap's block length, in consecutive valid dates within a fold."""
MIN_RESAMPLING_UNITS: Final = 20
LEVEL: Final = 1.0 - 0.05 / 3.0
"""The two-sided level of each counted reading: 98.33% (a Bonferroni split of 5% over three
readings)."""
N_RESAMPLES: Final = 20_000
GENERATOR_SEED: Final = 721
NOISE_WINDOW_PASSES: Final = 10
"""The flag's noise is the standard deviation of the validation loss over this many last passes."""
CONTINUATION_CAP_PASSES: Final = 150
CONTINUATION_PATIENCE: Final = 30
FLAG_NOISE_MULTIPLE: Final = 0.5
FLAG_DESCENT_SHARE: Final = 0.05
PLATEAU_FALSE_FLAG_CEILING: Final = 0.02
"""The false-flag rate per stage-1 stage that the Stage 0 plateau test allows."""
G3_POWER_TARGET: Final = 0.80
AREA_MARGIN: Final = 0.10
"""Rung 0b's own bar (`V1_AUC_GAP`): the area of `SG-A` minus the area of `SG-A-S`."""
MIN_CLASS_TOKENS: Final = 30
"""An area computed on fewer tokens of either class is `not evaluable`."""
HARM_STOP_POINTS: Final = 0.20
"""The quick look's harm stop, read on the rows whose signed truth is at or above zero."""
BRANCH_B_THRESHOLD_POINTS: Final = 0.20
"""Branch B's threshold on the two interim G1 point estimates (separate from every other
constant)."""
SE_REQUIRED: Final = 3.0
"""The reachability and noise checks require the margin to be at least this many standard errors."""
NOISE_CHECK_AREAS: Final = (0.95, 0.85)
NOISE_CHECK_CORRELATION: Final = 0.0
DEMAND_NOISE_DAY_OFFSET_POINTS: Final = 5.0
"""Standard deviation of the per-day offset added to the synthetic gross demand when the
reachability check fails."""
REAL_FOLD_CAP_FIRST: Final = 60
REAL_FOLD_CAP_RERUN: Final = 120
REAL_FOLD_PATIENCE: Final = 8
CAPACITY_SENSITIVITY: Final = (0.7, 1.3)
FOLD_HISTORY_SIGN_PROB: Final = 0.5
"""A supervised sample's history rows are folded with this probability."""
HEAD_BIAS_START: Final = -4.0
CONTINUATION_FLOOR_GPU_HOURS: Final = 0.6
GPU_HOURS_PER_FIT: Final = 0.2
GPU_HOURS_CAP: Final = 14.0
SIGN_CE_WEIGHT: Final = 1.0
"""Weight of the cross-entropy between `ρ` and the true sign, added to the stage 1 loss only (Branch
A). The design lead accepted 1.0 as a named constant, listed on the page."""
SIGN_HEAD_STAGE: Final = 1
"""The one stage in which the sign head trains (Branch A). In stages 2 and 3 the head is frozen: no
cross-entropy, no gradient, and no parameter in any optimiser group."""
SYNTH_PIPELINE_FOLD: Final = "2026-04"
"""The real fold whose pipeline supplies the synthetic world's grid, weather, and cutoff."""
SYNTH_EXPORTERS: Final = ("D9", "D11")
"""The synthetic world's two exporting Primaries (their real sky and temperature are used)."""
SYNTH_NON_EXPORTERS: Final = ("D2", "D3")
"""The synthetic world's two Primaries that never export."""
CONTINUATION_NEEDS: Final = "SG-A in W2, stage 1"
"""The one fit and stage in which the continuation and the flag run (Branch A)."""
N_INJECTION_SUBSTEPS: Final = 6
"""Sub-steps of the within-half-hour interpolation. Experiment 720's code owns the number; this copy
is asserted equal in a test."""
RANDOM_SEED_OF_WORLD: Final = 721

Reading = Literal["pass", "miss"]
G3Reading = Literal["pass", "fail", "unresolved"]
Branch = Literal["A", "B"]


# --------------------------------------------------------------------------- the widening


def kish_effective_number(*, counts: Sequence[float] | np.ndarray) -> float:
    """Return the Kish effective number of blocks, `(sum n)² / sum n²`, of the blocks' row counts."""
    n = np.asarray(counts, dtype=np.float64)
    n = n[n > 0]
    if n.size == 0:
        return 0.0
    return float(n.sum() ** 2 / np.square(n).sum())


def widening_factors(*, kish_b: float, level: float = LEVEL) -> tuple[float, float]:
    """Return the two factors that widen a bootstrap interval about its point estimate.

    The first is the ratio of the Student t quantile to the normal quantile at `level` with `B - 1`
    degrees of freedom, and the second is `sqrt(B / (B - 1))`, with `B` the Kish effective number of
    blocks. A `B` of 1 or less has no degrees of freedom: both factors are infinite.
    """
    if kish_b <= 1.0:
        return float("inf"), float("inf")
    tail = 1.0 - (1.0 - level) / 2.0
    t_ratio = float(stats.t.ppf(tail, kish_b - 1.0) / stats.norm.ppf(tail))
    return t_ratio, float(np.sqrt(kish_b / (kish_b - 1.0)))


def widened_interval(
    *, point: float, lower: float, upper: float, kish_b: float
) -> tuple[float, float]:
    """Return the interval after widening both ends about `point` by the two factors."""
    t_ratio, root = widening_factors(kish_b=kish_b)
    factor = t_ratio * root
    return point - factor * (point - lower), point + factor * (upper - point)


# --------------------------------------------------------------------------- the minimum detectable effect


def mde_points(*, seed_difference_points: float) -> float:
    """Return the larger of 0.404 points and the seed difference of `SG-0`, in points."""
    return max(MDE_FLOOR_POINTS, abs(float(seed_difference_points)))


def write_mde_file(
    *, path: Path, seed_difference_points: float, detail: dict | None = None
) -> dict:
    """Write the MDE file and refuse to overwrite it.

    The file is created with a hard link from a temporary name, so a second writer fails with
    `FileExistsError` and the first file is never replaced, even by a concurrent process.
    """
    record = {
        "mde_points": mde_points(seed_difference_points=seed_difference_points),
        "seed_difference_points": abs(float(seed_difference_points)),
        "floor_points": MDE_FLOOR_POINTS,
        "fold": MDE_FOLD,
        **(detail or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return record


def read_mde_file(*, path: Path) -> float:
    """Return the stored MDE in points, and fail when the file does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: the MDE must be written before any contrast is compared with a bar"
        raise FileNotFoundError(msg)
    return float(json.loads(path.read_text())["mde_points"])


# --------------------------------------------------------------------------- the bars


def bar_g1_g2(*, point: float, upper: float, mde: float) -> Reading:
    """Return the G1 or G2 reading, `pass` when the point is at most -MDE and the upper end is below 0."""
    return "pass" if (point <= -mde and upper < 0.0) else "miss"


@dataclass(frozen=True)
class G3Check:
    """One of the six limit checks of G3.

    Attributes:
        reading: `plain` (`W0`, seeds 0 and 1 together) or `open` (`SG-A-open`, seed 0).
        scope: `pooled`, `D5`, or `D10`.
        point: The point estimate of `SG-A` minus `P0`, in points.
        lower: The lower end of the widened interval.
        limit: The limit on the point estimate.
    """

    reading: Literal["plain", "open"]
    scope: Literal["pooled", "D5", "D10"]
    point: float
    lower: float
    limit: float

    @property
    def status(self) -> G3Reading:
        """Return `pass` within the limit, `fail` when harm is shown, else `unresolved`."""
        if self.point <= self.limit:
            return "pass"
        return "fail" if self.lower > 0.0 else "unresolved"


def g3_limit(*, scope: str) -> float:
    """Return the limit of a G3 scope: 0.10 pooled and 0.20 at each Primary."""
    return G3_LIMIT_POOLED_POINTS if scope == "pooled" else G3_LIMIT_PER_PRIMARY_POINTS


def check_g3(*, checks: Sequence[G3Check]) -> G3Reading:
    """Return G3: `fail` when any check shows harm.

    `unresolved` when any check is over its limit with no harm shown, else `pass`.

    The page says G3 fails only when harm is shown and is otherwise `unresolved`, and it does not
    say how
    the six checks combine. Any shown harm fails, because one Primary or one reading with harm is
    harm.
    """
    statuses = [c.status for c in checks]
    if "fail" in statuses:
        return "fail"
    if "unresolved" in statuses:
        return "unresolved"
    return "pass"


# --------------------------------------------------------------------------- the flag


def noise_of_curve(*, val_history: Sequence[float]) -> float:
    """Return the sample standard deviation of the last 10 passes of a curve (all, when fewer)."""
    tail = np.asarray(val_history[-NOISE_WINDOW_PASSES:], dtype=np.float64)
    return float(np.std(tail, ddof=1)) if tail.size > 1 else 0.0


def replay_stop(*, val_history: Sequence[float], patience: int) -> tuple[int, float]:
    """Return the pass count and the best loss at which `patience` ends a run on `val_history`.

    A pass is an improvement when it beats the best so far by more than 1e-5, as in experiment 713's
    loop.
    A curve that never exhausts `patience` returns its full length.
    """
    best, stale = float("inf"), 0
    for i, v in enumerate(val_history):
        if v < best - 1e-5:
            best, stale = float(v), 0
        else:
            stale += 1
            if stale >= patience:
                return i + 1, best
    return len(val_history), best


@dataclass(frozen=True)
class FlagResult:
    """What the stopped-while-falling flag reads from one stage.

    Attributes:
        flagged: Whether the stage stopped while its validation loss was still falling.
        gap: The patience-8 best minus the patience-30 best.
        noise: The sample standard deviation over the last 10 passes of the patience-8 run.
        descent: The stage's first-pass validation loss minus the patience-30 best.
        stop8: Passes of the patience-8 run.
        stop30: Passes of the continued run (patience 30, cap 150 in all).
    """

    flagged: bool
    gap: float
    noise: float
    descent: float
    stop8: int
    stop30: int


def stopped_while_falling(*, val_history: Sequence[float], p8_cap: int | None = None) -> FlagResult:
    """Return the flag of one stage from its whole validation curve.

    `val_history` is the curve of the continued run (the run that patience 8 stops and patience 30
    goes
    on from, with at most 150 passes in all). The patience-8 stop and the patience-30 stop are both
    read
    off the same curve, which is exact because the passes after the patience-8 stop do not change
    the
    passes before it. The stage is flagged when the gap exceeds 0.5 times the noise and is at least
    5%
    of the descent. `p8_cap` is the cap of the patience-8 run (120 on a real fold), so that a stage
    still improving at its cap has the patience-8 best of its first 120 passes.
    """
    curve = list(val_history)[:CONTINUATION_CAP_PASSES]
    stop8, best8 = replay_stop(
        val_history=curve if p8_cap is None else curve[:p8_cap], patience=REAL_FOLD_PATIENCE
    )
    stop30, best30 = replay_stop(val_history=curve, patience=CONTINUATION_PATIENCE)
    gap = best8 - best30
    noise = noise_of_curve(val_history=curve[:stop8])
    descent = curve[0] - best30
    flagged = bool(
        gap > FLAG_NOISE_MULTIPLE * noise and descent > 0.0 and gap >= FLAG_DESCENT_SHARE * descent
    )
    return FlagResult(flagged, gap, noise, descent, stop8, stop30)


# --------------------------------------------------------------------------- the verdict rows


@dataclass(frozen=True)
class Stops:
    """The stops of row 0 (no verdict). Each field is true when that stop applies.

    Attributes:
        manager_halt: The manager halts the line (the quick look's harm stop, or a declined family
            of fits).
        stage0_flag_test_failed: The Stage 0 test of the flag failed.
        power_check_failed: The power check of G3 failed.
        reachability_failed_after_noise: The reachability check failed after the demand noise was
            added.
        noise_check_failed: The noise check of the margin failed.
        branch_b_maintainer_stop: The maintainer stopped the line when Branch B returned decision 2.
        injection_not_built: The farm injection at the visible Primaries is not built (decision 4).
        floor_declined: The manager declined further `SG-A` fits at the continuation floor.
        area_not_evaluable: An area of the synthetic world is `not evaluable`.
        decision_unconfirmed: Decision 1, 2, 3, 7, or 8 is still unconfirmed when the first wave
            would start.
    """

    manager_halt: bool = False
    stage0_flag_test_failed: bool = False
    power_check_failed: bool = False
    reachability_failed_after_noise: bool = False
    noise_check_failed: bool = False
    branch_b_maintainer_stop: bool = False
    injection_not_built: bool = False
    floor_declined: bool = False
    area_not_evaluable: bool = False
    decision_unconfirmed: bool = False

    def any(self) -> bool:
        """Return whether any stop of row 0 applies."""
        return any(self.__dict__.values())


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match table reads.

    Attributes:
        stops: The stops of row 0.
        correctness_failed: A correctness check failed (holdout guard, fold point of the loss,
            origin-time
            weather assertion, no-two-worlds figure guard, or a known-answer test on synthetic
            arrays).
        synthetic_failed: The synthetic world fails (row 2).
        synthetic_cap_hit: Any fit of `SG-A` in the synthetic world hit the 400-pass cap.
        fit_failed: A fit raised and was not rerun under the rerun rule, or a fit that a counted
            reading needs is missing.
        g1: The reading of G1.
        g2: The reading of G2.
        g3: The reading of G3.
        sg_a_flagged: At least one counted stage of an `SG-A` fit in `W2` is flagged. A fit that ran
            without
            its continuation after the floor stop counts as flagged. Always false under Branch B.
    """

    stops: Stops = field(default_factory=Stops)
    correctness_failed: bool = False
    synthetic_failed: bool = False
    synthetic_cap_hit: bool = False
    fit_failed: bool = False
    g1: Reading = "miss"
    g2: Reading = "miss"
    g3: G3Reading = "unresolved"
    sg_a_flagged: bool = False


@dataclass(frozen=True)
class VerdictResult:
    """The row that matched, the verdict word.

    And whether a correctness failure goes in the first line.
    """

    row: int
    verdict: str
    correctness_in_first_line: bool


def verdict_row(*, v: VerdictInputs) -> VerdictResult:  # noqa: PLR0911
    """Return the first row of the verdict table that matches `v`, in the order of the page.

    Row 0: stopped, no verdict. Row 1: `invalid`. Row 2: the synthetic world fails (`inconclusive`
    when any
    `SG-A` fit hit the 400-pass cap, else `not supported`). Row 3: `failed`. Row 4: G3 fails, `not
    supported`.
    Row 5: G1, G2, and G3 pass, `supported`. Row 6: G3 `unresolved`, `inconclusive`. Row 7: G3
    passes, G1 or
    G2 misses, and an `SG-A` stage is flagged, `inconclusive`. Row 8: G3 passes, G1 misses, no flag,
    `not
    supported`. Row 9: G3 passes, G1 passes, G2 misses, no flag, `inconclusive`.
    """
    note = v.correctness_failed
    if v.stops.any():
        return VerdictResult(0, "stopped, no verdict", note)
    if v.correctness_failed:
        return VerdictResult(1, "invalid", note)
    if v.synthetic_failed:
        return VerdictResult(2, "inconclusive" if v.synthetic_cap_hit else "not supported", note)
    if v.fit_failed:
        return VerdictResult(3, "failed", note)
    if v.g3 == "fail":
        return VerdictResult(4, "not supported", note)
    if v.g3 == "pass" and v.g1 == "pass" and v.g2 == "pass":
        return VerdictResult(5, "supported", note)
    if v.g3 == "unresolved":
        return VerdictResult(6, "inconclusive", note)
    if v.sg_a_flagged:
        return VerdictResult(7, "inconclusive", note)
    if v.g1 == "miss":
        return VerdictResult(8, "not supported", note)
    return VerdictResult(9, "inconclusive", note)


def futility_fires(*, g1_point: float) -> bool:
    """Return whether the futility test fires: the pooled G1 point estimate is at or above zero."""
    return g1_point >= 0.0


def seeds_2_to_4_may_run(*, result: VerdictResult, g1_point: float) -> bool:
    """Return whether seeds 2 to 4 may run.

    Only after a row 6 or row 9 `inconclusive` whose futility test does not fire.
    """
    return result.row in {6, 9} and not futility_fires(g1_point=g1_point)


# --------------------------------------------------------------------------- the arms

Readings = Literal["folded", "signed"]
Loss = Literal["plain", "folded"]


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the page's arm table.

    Attributes:
        name: The page's arm name.
        net: `Z` is experiment 713's network `M*`; `HT` adds experiment 716's two PV token paths.
        readings: What the history and the target hold at the hidden pair and the eight MVA-only
            Primaries.
        loss: `plain` is the mixture negative log-likelihood of the reading; `folded` is the folded
            likelihood.
        sign_head: Whether `ρ` is learned (`SG-A` and its variants) or fixed at 0.
        shift_days: Days by which the sky is shifted (the placebo), 0 for the real sky.
        force_gate_open: Whether the per-Primary export gate is forced open (`SG-A-open`).
        worlds: The worlds the arm is fitted in.
    """

    name: str
    net: Literal["Z", "HT"]
    readings: Readings
    loss: Loss
    sign_head: bool
    shift_days: int
    force_gate_open: bool
    worlds: tuple[str, ...]


ARM_SPECS: Final[dict[str, ArmSpec]] = {
    "P0": ArmSpec("P0", "Z", "folded", "plain", False, 0, False, ("W2", "W0")),
    "S": ArmSpec("S", "HT", "signed", "plain", False, 0, False, ("W2", "W0")),
    "SG-0": ArmSpec("SG-0", "HT", "folded", "folded", False, 0, False, ("W2", "W0")),
    "SG-A": ArmSpec("SG-A", "HT", "folded", "folded", True, 0, False, ("W2", "W0", "synthetic")),
    "SG-A-S": ArmSpec(
        "SG-A-S", "HT", "folded", "folded", True, SHIFT_DAYS, False, ("W2", "synthetic")
    ),
    "SG-A-open": ArmSpec("SG-A-open", "HT", "folded", "folded", True, 0, True, ("W0",)),
}
