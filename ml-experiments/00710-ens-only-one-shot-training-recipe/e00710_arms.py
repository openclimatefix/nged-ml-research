"""The arms of experiment 710 as data: stage plans, weights, schedules, samplers, and phases.

This module holds data and two pure planning functions, so importing it reads no file. The design
is `data/ml-experiments/briefs/ens_one_shot/design.md` (sections d, e, and j), and the plan page is
`docs/ml-experiments/00710-ens-only-one-shot-training-recipe/result.md`.

**An arm is a list of stages on the pooled mixture density network of experiment 707.** A stage
names its rows (`era`, window `F`, the last 730 days, or the last 365 days), its weight rule, its
mechanism (a loss weight or a sampling probability), its learning rate, and its stopping rule. A
stage with a `state_key` stores its weights, so arms that share a chain share the state.

**Every arm has two injection states.** `n` trains with no synthetic switching events and `i`
trains with the (fold, seed) world of `e00710_inject` in every stage.
"""

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Final, Literal, get_args

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00710-ens-only-one-shot-training-recipe"
WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder; every output lives under `WORK_DIR / "work"`."""
EXPERIMENT_707_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00707-ens-backfill-pretraining-gain-against-ens-history-length"
)
"""Experiment 707's data folder: read only (stored `A_F`, `R_F`, and the `S_ens(F)` states)."""

TUNING_MONTHS: Final[tuple[str, ...]] = ("2024-10", "2025-01", "2025-03", "2025-06")
"""The four tuning folds. Arms are ranked on the last two; `PP`'s half-lives use the first two."""
RANKING_MONTHS: Final[tuple[str, ...]] = ("2025-03", "2025-06")
HALF_LIFE_MONTHS: Final[tuple[str, ...]] = ("2024-10", "2025-01")
SCORED_MONTHS: Final[tuple[str, ...]] = (
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
"""The 12 scored monthly folds, the same as experiment 707's `ALL_MONTHS`."""
TUNING_SEEDS: Final[tuple[int, ...]] = (0, 1)
SCORED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
EXTRA_SEEDS: Final[tuple[int, ...]] = (3, 4)
STORED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
"""The seeds that experiment 707 stored for `A_F`, `R_F`, and the `S_ens(F)` states."""
BASE_LR: Final[float] = 5e-4
FINE_TUNE_LR: Final[float] = 2e-4
MAX_EPOCHS: Final[int] = 40
ONE_SHOT_MAX_EPOCHS: Final[int] = 59
"""A one-stage arm on window F stops after 59 epochs, so that its step cap equals `M2`'s."""
PATIENCE: Final[int] = 6
RECENT_DAYS: Final[int] = 730
"""The last 730 days: the fine-tune window, and the rows of the selection set `V_sel`."""

StateType = Literal["i", "n"]
RowsType = Literal["era", "F", "W730", "W365"]
KindType = Literal["flat", "exp", "floor", "schedule", "per_primary", "placebo"]
MechanismType = Literal["loss", "sampler"]
LrKindType = Literal["flat", "by_epoch", "cosine"]
ValType = Literal["all_held", "recent_held"]
PhaseType = Literal["T", "S", "C", "SC"]

STORED_707_KEYS: Final[dict[str, str]] = {
    "era": "era_ftw730",
    "ensF": "ensF_ftw730",
    "era+ensF": "era+ensF_ftw730",
}
"""A stage's `state_key` and the folder of experiment 707's stored state that equals it."""


@dataclass(frozen=True)
class WeightRule:
    """How a stage weights its rows.

    Attributes:
        kind: `flat` (all ones), `exp`, `floor`, `schedule` (a half-life per epoch),
            `per_primary` (the learned half-lives of `PP`), or `placebo` (the exponential weight of
            permuted ages).
        half_life_years: The half-life for `exp`, `floor`, and `placebo`; `None` for `placebo` means
            the half-life `h_T` that the command line gives.
        floor: The weight floor for `floor`.
        schedule: A key of `e00710_weights.SCHEDULES` for `schedule`.
    """

    kind: KindType = "flat"
    half_life_years: float | None = None
    floor: float = 0.0
    schedule: str | None = None


@dataclass(frozen=True)
class LrRule:
    """The learning rate of a stage.

    Attributes:
        kind: `flat` (the stage's base rate), `by_epoch` (the `rates` list), or `cosine` (a cosine
            with warm restarts, evaluated per optimiser step).
        rates: For `by_epoch`, pairs of (first epoch index, rate), in order.
        cycle_epochs: For `cosine`, the epochs in one cycle.
        start_rates: For `cosine`, each cycle's start rate; the cosine ends at a tenth of it.
    """

    kind: LrKindType = "flat"
    rates: tuple[tuple[int, float], ...] = ()
    cycle_epochs: int = 0
    start_rates: tuple[float, ...] = ()


@dataclass(frozen=True)
class StagePlan:
    """One training stage.

    Attributes:
        name: `S_era`, `S_ens`, `FT`, `ONE`, `RFE`, or `CYC`.
        rows: The stage's rows.
        weight: The weight rule.
        mechanism: A loss weight, or a sampler that draws rows in proportion to the weight.
        lr: The base learning rate.
        lr_rule: The learning-rate schedule.
        init: `scratch` or `previous` (the weights after the stage before).
        state_key: The key under which the stage's weights are stored, or `None` if they are not.
        max_epochs: The most epochs.
        patience: Epochs without improvement that stop the stage (ignored if `fixed_epochs`).
        select_from: The first epoch (0-based) that counts for selection and for patience.
        select: `best` selects the epoch with the lowest selection loss, `last` the last epoch.
        fixed_epochs: Whether the stage always runs `max_epochs` epochs.
        ema_decay: The decay of an exponential moving average of the weights, or `None`.
        val: `all_held` selects on every held-out week of the stage's rows, as experiment 707 does
            for `S_era` and `S_ens`. `recent_held` selects on `V_sel`, the held-out weeks of the
            last 730 days.
        step_cap: `m2` caps the steps at those of `M2` (40 epochs on window F plus 40 on the last
            730 days); `None` is no cap.
        train_days: For a stage on the last 365 days or 730 days, the window length in days.
    """

    name: str
    rows: RowsType
    weight: WeightRule = field(default_factory=WeightRule)
    mechanism: MechanismType = "loss"
    lr: float = BASE_LR
    lr_rule: LrRule = field(default_factory=LrRule)
    init: Literal["scratch", "previous"] = "scratch"
    state_key: str | None = None
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    select_from: int = 0
    select: Literal["best", "last"] = "best"
    fixed_epochs: bool = False
    ema_decay: float | None = None
    val: ValType = "recent_held"
    step_cap: Literal["m2"] | None = None
    train_days: int | None = None


@dataclass(frozen=True)
class Arm:
    """One arm: its stages, its kind, and the phases and plans it belongs to.

    Attributes:
        name: The arm's name on the plan page, such as `W-h0.5`.
        stages: The stages in order.
        kind: `reference`, `maintainer`, `creative`, or `placebo`.
        tuned: Whether Phase T (the tuning folds) fits the arm.
        scored: `core` (always in Phase S), `finalist` (a creative arm that Phase S fits if it is a
            finalist), `placebo`, or `none`.
        i_only: Whether the arm exists in the `i` state only (the placebo and `M1` on tuning).
        free_from: For `M3`, the arm whose first stage state it forecasts from.
    """

    name: str
    stages: tuple[StagePlan, ...]
    kind: Literal["reference", "maintainer", "creative", "placebo"] = "creative"
    tuned: bool = True
    scored: Literal["core", "finalist", "placebo", "none"] = "none"
    i_only: bool = False
    free_from: str | None = None

    def folder(self, *, state: StateType) -> str:
        """Return the arm's folder name for a state, such as `W-h0.5-i`.

        Args:
            state: `i` or `n`.

        Returns:
            The folder name in `seed_preds/`, `models/`, `logs/`, and `bank/`.
        """
        return f"{self.name}-{state}"


FLAT: Final[WeightRule] = WeightRule()


def _exp(*, h: float) -> WeightRule:
    return WeightRule(kind="exp", half_life_years=h)


S_ERA: Final[StagePlan] = StagePlan(
    name="S_era", rows="era", state_key="era", val="all_held", init="scratch"
)
S_ENS_FROM_ERA: Final[StagePlan] = StagePlan(
    name="S_ens", rows="F", state_key="era+ensF", val="all_held", init="previous"
)
S_ENS: Final[StagePlan] = StagePlan(name="S_ens", rows="F", state_key="ensF", val="all_held")


def _ft(*, h: float, ema: float | None = None) -> StagePlan:
    return StagePlan(
        name="FT",
        rows="W730",
        weight=_exp(h=h),
        lr=FINE_TUNE_LR,
        init="previous",
        ema_decay=ema,
        train_days=RECENT_DAYS,
    )


def _one(
    *, weight: WeightRule, mechanism: MechanismType = "loss", rows: RowsType = "F", days: int | None
) -> StagePlan:
    return StagePlan(
        name="ONE",
        rows=rows,
        weight=weight,
        mechanism=mechanism,
        max_epochs=ONE_SHOT_MAX_EPOCHS if rows == "F" else MAX_EPOCHS,
        step_cap="m2" if rows == "F" else None,
        train_days=days,
    )


_S_SCHEDULE: Final[StagePlan] = StagePlan(
    name="ONE",
    rows="F",
    weight=WeightRule(kind="schedule", schedule="S"),
    lr_rule=LrRule(kind="by_epoch", rates=((0, BASE_LR), (5, FINE_TUNE_LR))),
    max_epochs=ONE_SHOT_MAX_EPOCHS,
    select_from=5,
    step_cap="m2",
)
_CYC: Final[StagePlan] = StagePlan(
    name="CYC",
    rows="F",
    weight=WeightRule(kind="schedule", schedule="CYC"),
    lr_rule=LrRule(kind="cosine", cycle_epochs=4, start_rates=(BASE_LR, BASE_LR, FINE_TUNE_LR)),
    max_epochs=12,
    fixed_epochs=True,
    select_from=8,
)
_RFE: Final[StagePlan] = StagePlan(
    name="RFE",
    rows="W365",
    lr=FINE_TUNE_LR,
    init="previous",
    max_epochs=1,
    fixed_epochs=True,
    select="last",
    train_days=365,
)

ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm(
            name="M1",
            kind="reference",
            stages=(S_ERA, S_ENS_FROM_ERA, _ft(h=0.5)),
            scored="core",
            i_only=True,
        ),
        Arm(name="M2", kind="maintainer", stages=(S_ENS, _ft(h=0.5)), scored="core"),
        Arm(
            name="M3",
            kind="maintainer",
            stages=(S_ENS,),
            scored="core",
            free_from="M2",
        ),
        Arm(
            name="W-h0.5",
            kind="maintainer",
            stages=(_one(weight=_exp(h=0.5), days=None),),
            scored="core",
        ),
        Arm(
            name="W-h1",
            kind="maintainer",
            stages=(_one(weight=_exp(h=1.0), days=None),),
            scored="core",
        ),
        Arm(
            name="W-h2",
            kind="maintainer",
            stages=(_one(weight=_exp(h=2.0), days=None),),
            scored="core",
        ),
        Arm(name="S", kind="maintainer", stages=(_S_SCHEDULE,), scored="core"),
        Arm(
            name="HW730",
            stages=(_one(weight=FLAT, rows="W730", days=RECENT_DAYS),),
            scored="core",
        ),
        Arm(name="HW365", stages=(_one(weight=FLAT, rows="W365", days=365),), scored="finalist"),
        Arm(
            name="FLR",
            stages=(
                _one(weight=WeightRule(kind="floor", half_life_years=0.5, floor=0.1), days=None),
            ),
            scored="finalist",
        ),
        Arm(
            name="SMP",
            stages=(_one(weight=_exp(h=0.5), mechanism="sampler", days=None),),
            scored="finalist",
        ),
        Arm(name="RFE", stages=(S_ENS, _RFE), scored="finalist"),
        Arm(
            name="EMA",
            stages=(S_ENS, _ft(h=0.5, ema=0.999)),
            scored="finalist",
        ),
        Arm(name="CYC", stages=(_CYC,), scored="finalist"),
        Arm(
            name="PP",
            stages=(_one(weight=WeightRule(kind="per_primary"), days=None),),
            scored="finalist",
        ),
        Arm(name="FTh90d", stages=(S_ENS, _ft(h=90.0 / 365.25)), scored="finalist"),
        Arm(name="FTh1", stages=(S_ENS, _ft(h=1.0)), scored="finalist"),
        Arm(
            name="P",
            kind="placebo",
            stages=(_one(weight=WeightRule(kind="placebo"), days=None),),
            tuned=False,
            scored="placebo",
            i_only=True,
        ),
    )
}
"""Every arm, keyed by name. The `FT` stage of `M1`, `M2`, `EMA`, `FTh90d`, and `FTh1` uses 2e-4."""
CORE_SCORED: Final[tuple[str, ...]] = (
    "M1",
    "M2",
    "M3",
    "W-h0.5",
    "W-h1",
    "W-h2",
    "S",
    "HW730",
    "P",
)
"""The arms that Phase S always scores (the two finalists are added)."""
CONTRAST_NAMES: Final[dict[str, tuple[str, str]]] = {
    "C1": ("B*", "M1"),
    "C2": ("M2", "M1"),
    "C3": ("O*", "M2"),
    "C4": ("W-h_T", "M3"),
    "C5": ("W-h_T", "P"),
    "C6": ("S", "W-h0.5"),
    "C7": ("HW730", "M3"),
}
"""Each tested contrast as (treatment, reference), all in the `i` state."""
FAMILY_LEVELS: Final[dict[str, tuple[float, int]]] = {
    "R": (100.0 - 5.0 / 6.0, 6),
    "W": (100.0 - 5.0 / 8.0, 8),
    "I": (100.0 - 5.0 / 12.0, 12),
}
"""Each contrast family's two-sided level in percent and its number of readings (Bonferroni)."""
FAMILY_OF: Final[dict[str, str]] = {
    "C1": "R",
    "C2": "R",
    "C3": "R",
    "C4": "W",
    "C5": "W",
    "C6": "W",
    "C7": "W",
    "I1": "I",
    "I2": "I",
    "I3": "I",
    "I4": "I",
    "I5": "I",
    "I6": "I",
}


@dataclass(frozen=True)
class Job:
    """One fit: an arm in a state, on a fold, with a network seed."""

    arm: str
    state: StateType
    month: str
    seed: int


def plan_jobs(
    *,
    phase: PhaseType,
    minimal: bool,
    h_t: float = 0.5,
    finalists: tuple[str, ...] = (),
    b_star: str = "M2",
    o_star: str | None = None,
    include_pp: bool = False,
) -> list[Job]:
    """Return the jobs of a phase, for the minimal or the full plan, in priority order.

    **Phase T** fits every tuned arm in both states on the four tuning folds with seeds 0 and 1,
    except that `M1` is fitted in `i` only; the minimal plan drops the `n` state except for `M2`,
    whose `n` state the second stop check needs. **Phase S** fits
    the scored folds with seeds 0 to 2: the `i` state of `M1`, `M2`, `M3`, `W-h_T`, `P`, and `B*` in
    the minimal plan, and the `i` state of every core arm and finalist (`M3-i` is forecast from
    `M2-i`'s state) plus the `n` state of the arms that have no stored forecasts in the full plan.
    **Phase C** adds seeds 3 and 4 of `M1-i`, `M2-i`, and `B*-i` (and `B*-n` in the full plan, if
    `B*` has no stored `n` forecasts). **Phase SC** is Phases S and C together.

    The jobs are sorted by priority tier (`job_tier`), then seed, then fold, so that every tier-1
    contrast holds all 12 folds at seed 0 before any tier-1 job of seed 1 starts.

    Args:
        phase: `T`, `S`, `C`, or `SC`.
        minimal: Whether to return the minimal plan.
        h_t: The half-life in years of the best `W` arm on the tuning folds.
        finalists: The creative arms that Phase S fits.
        b_star: The ENS-only recipe `B*` that the tuning folds name.
        o_star: The one-shot recipe `O*` that the tuning folds name, or `None` if not yet known.
        include_pp: Whether Phase T includes `PP`. The half-lives of `PP` come from the `W-h` and
            `M3` fits of Phase T itself, so `PP` is left out of the first tuning pass and added in a
            second pass once the half-life file exists.

    Returns:
        The jobs, in tier, seed, fold, then arm order.
    """
    w_t = f"W-h{h_t:g}"
    jobs: list[Job] = []
    if phase == "T":
        for month, seed, arm in product(TUNING_MONTHS, TUNING_SEEDS, ARMS.values()):
            if not arm.tuned or (arm.name == "PP" and not include_pp):
                continue
            only_i = (minimal and arm.name != "M2") or arm.i_only
            states: tuple[StateType, ...] = ("i",) if only_i else ("i", "n")
            jobs.extend(Job(arm.name, s, month, seed) for s in states)
    if phase in ("S", "SC"):
        i_arms = ["M1", "M2", "M3", w_t, "P", b_star] if minimal else [*CORE_SCORED, *finalists]
        if o_star is not None:
            i_arms.append(o_star)
        n_arms = [] if minimal else [a for a in (*CORE_SCORED, *finalists) if needs_new_n_fit(a)]
        for month, seed in product(SCORED_MONTHS, SCORED_SEEDS):
            jobs.extend(Job(a, "i", month, seed) for a in dict.fromkeys(i_arms))
            jobs.extend(Job(a, "n", month, seed) for a in n_arms)
    if phase in ("C", "SC"):
        arms_i = dict.fromkeys(["M1", "M2", b_star])
        for month, seed in product(SCORED_MONTHS, EXTRA_SEEDS):
            jobs.extend(Job(a, "i", month, seed) for a in arms_i)
            if not minimal and needs_new_n_fit(b_star):
                jobs.append(Job(b_star, "n", month, seed))
    unique = list(dict.fromkeys(jobs))
    order = {name: n for n, name in enumerate(ARMS)}
    return sorted(
        unique,
        key=lambda j: (
            job_tier(job=j, h_t=h_t, b_star=b_star, o_star=o_star),
            j.seed,
            j.month,
            j.state,
            order.get(j.arm, len(order)),
        ),
    )


CHOICES_FILE: Final = WORK_DIR / "work" / "tuning_choices.json"
"""The tuning folds' choices (`b_star`, `o_star`, `h_t`, `finalists`), written by the scorer."""


STOP_CHECKS_FILE: Final = WORK_DIR / "work" / "stop_checks.json"
"""The two tuning stop checks (`M2-i` against `M1-i` and `M2-n`) and whether each passed."""


def read_choices() -> dict:
    """Return the tuning choices that `e00710_score.py --tuning` wrote.

    Returns:
        A dict with the keys `b_star`, `o_star`, `h_t`, and `finalists`.

    Raises:
        SystemExit: If the scorer has not written the choices yet.
    """
    if not CHOICES_FILE.exists():
        msg = (
            f"{CHOICES_FILE} is missing: run e00710_score.py --tuning first, or give --b-star, "
            "--o-star, and --h-t explicitly"
        )
        raise SystemExit(msg)
    return json.loads(CHOICES_FILE.read_text())


def job_tier(*, job: Job, h_t: float, b_star: str, o_star: str | None) -> int:
    """Return a job's priority tier: 1 first, 3 last.

    Tier 1 holds the `i` state of `M1`, `M2`, `M3`, `B*`, `O*`, `W-h_T`, and `P`, which the
    headline contrasts C1 to C5 read, on every fold and seed (Phase C included). Tier 2 holds the
    `i` state of every other arm (`S`, `HW730`, the finalists, and the other `W` half-lives).
    Tier 3 holds every `n` state. Every job on a tuning fold is tier 1.

    Args:
        job: The job.
        h_t: The half-life in years of `W-h_T`.
        b_star: The ENS-only recipe `B*`.
        o_star: The one-shot recipe `O*`, or `None`.

    Returns:
        1, 2, or 3.
    """
    if job.month in TUNING_MONTHS:
        return 1
    if job.state == "n":
        return 3
    first = {"M1", "M2", "M3", "P", b_star, f"W-h{h_t:g}"}
    if o_star is not None:
        first.add(o_star)
    return 1 if job.arm in first else 2


def needs_new_n_fit(name: str) -> bool:
    """Return whether the `n` state of an arm needs a new fit (`M1`, `M2`, `M3`, `P` do not)."""
    return name not in ("M1", "M2", "M3", "P")


def stage_plan_text(*, arm: Arm) -> str:
    """Return an arm's stages as one line, such as `S_era>S_ens>FT`.

    Args:
        arm: The arm.

    Returns:
        The stage names joined with `>`.
    """
    return ">".join(s.name for s in arm.stages)


def print_table() -> None:
    """Print every arm with its stages, its phases, and its weight rules."""
    for arm in ARMS.values():
        rules = ", ".join(f"{s.weight.kind}/{s.mechanism}" for s in arm.stages)
        print(
            f"{arm.name:8s} {stage_plan_text(arm=arm):16s} {arm.kind:11s} tuned={arm.tuned!s:5s} "
            f"scored={arm.scored:9s} i_only={arm.i_only!s:5s} {rules}"
        )
    for minimal in (True, False):
        counts = {
            phase: len(plan_jobs(phase=phase, minimal=minimal)) for phase in get_args(PhaseType)
        }
        print(f"jobs by phase, minimal={minimal}: {counts}")


if __name__ == "__main__":
    print_table()
