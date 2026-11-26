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
N_CONTRASTS: Final = 6
LEVEL: Final = 1.0 - 0.05 / N_CONTRASTS
"""Bonferroni level across the 6 counted contrasts (99.167%)."""
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
"""Size of one transfer in points, log-uniform."""
TRANSFER_INSIDE_BSP_A: Final = 0.7
"""Probability that the partner Primary sits under BSP A (the BSP meter is unchanged)."""
TICKS_PER_HALF_HOUR: Final = 6
"""A transfer starts on a 5-minute grid."""
MAX_REDRAWS: Final = 200
INJECTION_POINTS: Final = (2.0, 60.0)
"""The extra injection a in training: log-uniform per window and epoch."""
TEST_INJECTION_POINTS: Final = (5.0, 20.0, 50.0)
EPS_RANGE: Final = (0.05, 0.3)
"""The raised-sky factor in training: uniform per window."""
TEST_EPS: Final = 0.2
CSI_CLIP: Final = 1.5
RAISABLE_MAX_CSI: Final = 1.25
SUN_UP_COSZ: Final = 0.1
NIGHT_COSZ: Final = 0.0
SEED_TRAIN_OVERLAY: Final = 732_100
SEED_VALIDATION_OVERLAY: Final = 732_050
SEED_TEST_OVERLAY: Final = 732_000
OVERLAY_KINDS: Final = ("train", "validation", "test")
BANK_CODE: Final = {"train": 0, "validation": 1, "test": 2, "shifted": 3}
DELTA_SHARE_OF_A: Final = 0.05
"""The injection readings keep half-hours where delta is at least this share of a."""

# --------------------------------------------------------------------------- training regime

N_FITS: Final = 18
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
        role: The arm's pre-registered role.
    """

    name: str
    arch: str
    heads: bool
    weights: Mapping[str, float]
    stop_gradient: bool = False
    inject: bool = False
    raised: bool = False
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
}  # fmt: skip
FIT_ARMS: Final = tuple(ARMS)
NEEDED_ARMS: Final = ("U*", "U-NC", "S+")
"""A failed fit of one of these arms makes verdict row 2 (C1, C2, C3, C6, R4, and the bars)."""
REPORT_ONLY_ARMS: Final = ("U-NCw", "U-SG", "U*-L")
FIT_FAMILIES: Final = (
    ("timing", ()),
    ("U*, U-NC", ("U*", "U-NC")),
    ("U-NCw, S+", ("U-NCw", "S+")),
    ("U-SG", ("U-SG",)),
    ("U*-L", ("U*-L",)),
)
"""The families of fits, in order; a "fits starting" line precedes each."""
STORED_730: Final = {
    "S": ("S", "unet"),
    "S-L": ("S", "bilstm"),
    "SD*": ("SD-R", "unet"),
}
"""The stored arms of experiment 730 read as controls: name -> (its fit name, architecture)."""
CONTRASTS: Final = {
    "C1": ("U*", "S"), "C2": ("U*", "U-NC"), "C3": ("U*", "U-NC"), "C4": ("U*", "B-PV2"),
    "C5": ("U*", "B-T"), "C6": ("U*", "S+"),
}  # fmt: skip
"""Each counted contrast as (arm minus arm); a negative difference is a gain. C3 reads the injection error."""
REPORT_ONLY: Final = {
    "R1": "U-SG minus S+, Brier of P(X<0)", "R2": "U* minus U-SG, Brier of P(X<0)",
    "R3": "U*-L minus S-L (730), Brier of P(X<0)", "R4": "S+ minus S (730), Brier of P(X<0)",
    "R5": "U* minus U-NC, Brier of P(X<0)", "R6": "U* minus B-PV1, solar mean absolute error",
    "R7": "U*-L minus U*, solar mean absolute error", "R8": "U* minus S+, Brier of P(X<0) on the transfer overlay",
    "R9": "U* minus U-NC, share of monotonicity violations", "R10": "U* minus U-NCw, solar mean absolute error",
    "R11": "U* minus U-NCw, mean absolute injection error", "R12": "C4 on the solar-only rows and on the no-injection rows",
    "B13": "U* minus S (730), mean P(X<0) on import half-hours of 1 to 3 MVA",
}  # fmt: skip
N_REPORT_ONLY: Final = len(REPORT_ONLY)

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
RECOVERY_MEDIAN_ERROR: Final = 0.1
NIGHT_MEAN_POINTS: Final = 0.05
NIGHT_EXCEED_POINTS: Final = 0.5
NIGHT_EXCEED_SHARE: Final = 0.005
MONOTONE_TOLERANCE_POINTS: Final = 0.1
MONOTONE_MAX_SHARE: Final = 0.05
RESPONSE_MIN_TARGET_POINTS: Final = 1.0
RESPONSE_MIN_SHARE: Final = 0.5
TRANSFER_F1_BAR: Final = 0.6
SOLAR_BARS: Final = ("recovery", "night", "monotone", "response", "better_than_demand")
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
TIMING_ARMS: Final = ("U*", "U-NC", "S+", "U-SG", "U*-L")
INFERENCE_HOURS: Final = 0.25
TIMING_GATE_HOURS: Final = 0.05
FITS_PER_ARM: Final = len(SEEDS)
RATE_OF: Final = {"U*": "U*", "U-NC": "U-NC", "U-NCw": "U-NC", "S+": "S+", "U-SG": "U-SG", "U*-L": "U*-L"}
"""The timed arm whose seconds an epoch prices each fitted arm."""
MAX_SLOT_WORKERS: Final = 6

# --------------------------------------------------------------------------- the verdict

Verdict = Literal["stopped", "invalid", "failed", "not supported", "supported", "partly supported", "inconclusive"]
VERDICT_WORDS: Final = {
    0: "stopped", 1: "invalid", 2: "failed", 3: "not supported", 4: "supported",
    5: "partly supported", 6: "not supported", 7: "inconclusive", 8: "not supported",
}  # fmt: skip
FRONT_MATTER_WORD: Final = {"stopped": "inconclusive"}
"""Row 0 is recorded as `inconclusive` in the front matter."""


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match verdict table reads.

    Attributes:
        stopped: Row 0 applies.
        invalid: Row 1 applies.
        failed: Row 2 applies.
        h1_pooled_pass: H1 pooled passed for U*.
        h1_high_pass: H1 at 3 MVA and above passed for U*.
        h2_pass: H2 passed.
        band_fail: The band of 1 to 3 MVA is above its limit and B13's lower end is above zero.
        c1: The headline contrast C1 (U* minus S).
        margin: The margin m.
        solar_bars: Whether each of the five solar bars passed.
        c2_gain: C2's interval excludes zero on the negative side.
    """

    stopped: bool
    invalid: bool
    failed: bool
    h1_pooled_pass: bool
    h1_high_pass: bool
    h2_pass: bool
    band_fail: bool
    c1: core729.Reading
    margin: float
    solar_bars: Mapping[str, bool]
    c2_gain: bool


def solar_bars_pass(*, bars: Mapping[str, bool]) -> bool:
    """Return whether every one of the five solar bars is present and passed (a missing bar fails)."""
    return all(bool(bars.get(name, False)) for name in SOLAR_BARS)


def harm_row_fails(*, v: VerdictInputs) -> str | None:
    """Return the name of the first failed harm check of row 3, or None."""
    if not v.h1_pooled_pass:
        return "H1 pooled"
    if not v.h1_high_pass:
        return "H1 at 3 MVA and above"
    if not v.h2_pass:
        return "H2"
    if v.band_fail:
        return "H1 in the band of 1 to 3 MVA (against B13)"
    return None


def verdict_row(*, v: VerdictInputs) -> int:  # noqa: PLR0911
    """Return the first row of the verdict table that matches (0 to 8)."""
    if v.stopped:
        return 0
    if v.invalid:
        return 1
    if v.failed:
        return 2
    if harm_row_fails(v=v) is not None:
        return 3
    non_inferior = v.c1.upper <= v.margin
    if non_inferior:
        if not solar_bars_pass(bars=v.solar_bars):
            return 6
        return 4 if v.c2_gain else 5
    if v.c1.lower <= 0.0:
        return 7
    return 8


def verdict_word(*, v: VerdictInputs) -> Verdict:
    """Return the verdict word of `verdict_row`."""
    return VERDICT_WORDS[verdict_row(v=v)]  # ty: ignore[invalid-return-type]


def front_matter_word(*, word: str) -> str:
    """Return the word that the front matter carries (row 0 is `inconclusive`)."""
    return FRONT_MATTER_WORD.get(word, word)


def transfer_word(*, c5: core729.Reading | None, f1: float, reachable: bool) -> str:
    """Return the transfer output's verdict word, read from C5 and the F1 bar.

    Args:
        c5: C5's reading (U* minus B-T in the step Brier score), or None.
        f1: U*'s F1 at 0.5 on transfer steps of 5 points and more.
        reachable: The reachability count of transfer steps holds.

    Returns:
        "works", "works only in Brier score", "no better than the rule", "worse than the rule", or "void".
    """
    if not reachable or c5 is None or not all(math.isfinite(z) for z in (c5.point, c5.lower, c5.upper)):
        return "void"
    if c5.upper < 0.0:
        return "works" if f1 >= TRANSFER_F1_BAR else "works only in Brier score"
    if c5.lower > 0.0:
        return "worse than the rule"
    return "no better than the rule"


def is_gain(*, reading: core729.Reading) -> bool:
    """Return whether a counted reading is a gain: an interval that lies below zero."""
    return bool(math.isfinite(reading.upper) and reading.upper < 0.0)


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


def write_limits_file(*, path: Path, limits: Mapping[str, BandLimit], s: float, n_rows: int) -> dict:
    """Write `limits.json` once (a second write raises `FileExistsError`) and return the record."""
    record = {
        "s": s, "n_rows": n_rows,
        "bands": {k: {"share": v.share, "limit": v.limit, "n_rows": v.n_rows} for k, v in limits.items()},
    }  # fmt: skip
    core730.write_once_json(path=path, payload=record)
    return record


def read_limits_file(*, path: Path) -> dict[str, BandLimit]:
    """Read `limits.json` and fail when it does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: the limits must be written before any fit"
        raise FileNotFoundError(msg)
    bands = json.loads(path.read_text())["bands"]
    return {k: BandLimit(share=v["share"], limit=v["limit"], n_rows=v["n_rows"]) for k, v in bands.items()}


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


def projected_hours(*, seconds_per_epoch: Mapping[str, float], epochs: int = MAX_EPOCHS) -> float:
    """Return the GPU-process-hours of the 18 fits at `epochs` epochs each, at each arm's own rate.

    Args:
        seconds_per_epoch: The timed seconds an epoch of each arm in `TIMING_ARMS`.
        epochs: The cap of every fit.

    Returns:
        The fits plus the inference allowance and the timing gate.
    """
    missing = [a for a in set(RATE_OF.values()) if a not in seconds_per_epoch]
    if missing:
        msg = f"no timing for {missing}"
        raise KeyError(msg)
    fits = sum(FITS_PER_ARM * seconds_per_epoch[RATE_OF[a]] * epochs for a in FIT_ARMS) / 3600.0
    return fits + INFERENCE_HOURS + TIMING_GATE_HOURS


def timing_gate_report(*, seconds_per_epoch: Mapping[str, float], peak_gpu_gb: float, peak_rss_gb: float,
                       pair_seconds_per_epoch: float) -> dict[str, float | bool]:  # fmt: skip
    """Return the projection at the cap and whether it, the GPU memory, and the memory pass the gate."""
    hours = projected_hours(seconds_per_epoch=seconds_per_epoch)
    return {
        "hours_at_cap": hours, "stop_line_hours": STOP_LINE_HOURS,
        "within_stop_line": hours <= STOP_LINE_HOURS, "peak_gpu_gb": peak_gpu_gb,
        "gpu_ok": peak_gpu_gb <= GPU_MEMORY_GB, "peak_rss_gb": peak_rss_gb,
        "memory_ok": peak_rss_gb <= MEMORY_CAP_GB, "pair_seconds_per_epoch": pair_seconds_per_epoch,
        "seconds_per_epoch": dict(seconds_per_epoch),
    }  # fmt: skip


def timing_gate_passes(*, report: Mapping[str, object], manager_approved: bool) -> bool:
    """Return whether fits may start: memory and GPU within bounds, and the projection within the line or approved."""
    return bool(report["memory_ok"] and report["gpu_ok"] and (report["within_stop_line"] or manager_approved))


# --------------------------------------------------------------------------- the gallery's list

PANEL_REQUIREMENTS: Final = {
    "G10_transfers": "every transfer changes the truth by the drawn size on every half-hour inside it, each edge is marked, and a partial half-hour at an edge shows a smaller step",
    "G11_pairs": "every pair's re-folded reading equals the original where delta is zero, the pair differs only on sun-up half-hours, and the truth drops by delta",
    "G12_raised_sky": "every raised index is above the original on every raisable half-hour and equal elsewhere, with the reading unchanged",
    "G13_targets": "the solar target, the step events, and the step sizes are non-empty for the world that carries them and aligned with their cause",
    "G14_parameters": "every histogram of a drawn transfer parameter lies inside its registered range",
    "G15_everything": "the window with an engine run, S2 solar output, a load transfer, and an extra injection shows every component and the injected member differs from the plain one",
    "G16_pairing": "every month of every training seed has a sibling pair for at least 95% of its windows",
    "G17_s1_pairs": "the pairs built from source S1 are written to the private folder only",
}
"""The written list of what each new gallery panel must show. The session that runs the experiment
checks each panel against it and records the result in `PRIVATE/gallery/panel_check.json`."""
GALLERY_PUBLISHED: Final = ("G10_transfers", "G11_pairs", "G12_raised_sky", "G13_targets", "G14_parameters", "G15_everything", "G16_pairing")
GALLERY_PRIVATE_ONLY: Final = ("G17_s1_pairs",)

# --------------------------------------------------------------------------- the schedule


def fit_jobs() -> list[tuple[str, str, int]]:
    """Return the 18 fits as (arm, architecture, seed), in the order of the families."""
    return [(arm, ARMS[arm].arch, seed) for _, arms in FIT_FAMILIES for arm in arms for seed in SEEDS]


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
    return ("banks", "limits", "overlays", "gallery", "panels", "timing", "fit", "real", "score", "report", "figures", "series", "chain")
