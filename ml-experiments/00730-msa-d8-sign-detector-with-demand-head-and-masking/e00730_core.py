# ruff: noqa: E501
"""Experiment 730: every named constant and bar of the plan, the limits files, and the verdict rule.

Every number that the page's decision rule pre-registers is a `Final` here, so the page and the code
are compared by reading this one file. The functions are pure, except the write-once file writers,
which refuse to overwrite. No function prints or logs a month, a date, an engine size, a series
identifier, or a real value.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00730_paths  # noqa: F401

# isort: split
import e00729_core as core729
import numpy as np

# --------------------------------------------------------------------------- identity and rows

ROW_SET_ID: Final = "e00730_synthetic_d8_v1"
"""The row set: the 48 half-hours of each UTC day of the 8 test months, 8 draws each."""
REQUIRED_RUN_ID: Final = "2"
"""The banks stage asserts that `e00729_core.RUN_ID` equals this value."""
MIN_OUTPUT_SHARE: Final = 0.99
"""Every simulated component must produce output on at least this share of the days or half-hours
where output must exist. `tests/smoke_real_files.py` quotes this same constant."""
EXACT_SHARE: Final = 1.0
"""The bar where the simulator leaves no room for an exception (the no-injection truth and the fold)."""
CLEAR_SUN_UP_CSI: Final = 0.8
"""A clear sun-up half-hour has a CAMS clear-sky index of at least this value at D8."""
MAX_NO_SKY_SHARE: Final = 0.05
"""A share of days with no satellite sky at any farm above this stops the line (verdict row 0)."""
COMPONENT_SEASONS: Final = ("winter", "spring", "summer", "autumn")
"""Meteorological seasons of the UTC month, for the by-season component counts."""

# --------------------------------------------------------------------------- the bootstrap

BLOCK_DAYS: Final = core729.BLOCK_DAYS
N_CONTRASTS: Final = 7
LEVEL: Final = 1.0 - 0.05 / N_CONTRASTS
"""Bonferroni level across the 7 counted contrasts (99.286%)."""
N_REPORT_ONLY_CONTRASTS: Final = 22
"""R1 to R22, read at `LEVEL` and counted in no verdict row."""
N_RESAMPLES: Final = 20_000
GENERATOR_SEED: Final = 730
MASK_SEED: Final = 730
"""The generator seed of the fixed masks of the demand read-out and the causal validation cuts."""

# --------------------------------------------------------------------------- limits and bars

MARGIN_FLOOR_FRACTION: Final = 0.02
"""The margin's floor f is this share of P0's Brier score s, rounded down to two significant figures."""
HARM_FRACTION: Final = 0.25
"""The harm limit L is this fraction of the decidable reverse-flow share per import row."""
LIMIT_RANGE: Final = (0.003, 0.02)
"""The line stops at verdict row 0 when L falls outside this range."""
DECIDABLE_READING_MVA: Final = 1.0
"""A decidable row has a reading of at least this value."""
DECIDABLE_READING_POINTS: Final = DECIDABLE_READING_MVA / core729.CAPACITY_MVA * core729.POINTS
DETECTION_BAR: Final = 0.75
"""SD*'s mean P(X<0) on reverse-flow rows with a reading of at least 1 MVA must be at least this."""
RELIABILITY_FRACTION: Final = 0.1
"""The sign calibration bar is this share of s."""
BIN_GAP_BAR: Final = core729.BIN_GAP_BAR
BIN_MIN_COUNT: Final = core729.BIN_MIN_COUNT
RELIABILITY_BINS: Final = core729.RELIABILITY_BINS
DEAD_BAND_POINTS: Final = core729.DEAD_BAND_POINTS
FALSE_ALARM_THRESHOLD: Final = 0.5
"""A false alarm is a half-hour with P(X<0) above this value."""
PROB_CLIP: Final = 1e-6
"""Probabilities are clipped to this value and one minus it in the log loss."""
MARGIN_SD_MULTIPLE: Final = 2.0
TIE_FRACTION: Final = 0.5
"""The tie windows of the architecture screen and of the choice of SD* are this share of f."""
MIN_REVERSE_ALL: Final = core729.MIN_REVERSE_ALL
MIN_REVERSE_BSP: Final = core729.MIN_REVERSE_BSP
K1_MIN_REVERSE_DECIDABLE: Final = 100
"""K1 is void when fewer half-hours than this are reverse flow with a reading of at least 1 MVA."""
DEMAND_GAIN_POINTS: Final = 1.0
"""The predicted gain of the W-trained demand head over the demand climatology (report only)."""

# --------------------------------------------------------------------------- training regime

LAMBDA: Final = 0.1
LAMBDA_REPORT: Final = 0.5
TRANSFORMER_VARIANTS: Final = ("transformer", "gru_transformer")
"""The two transformer variants; C6 reads the one with the lower validation Brier score of arm S."""
ARCHITECTURES: Final = ("bilstm", "unet", "transformer", "gru_transformer")
SCREEN_ARMS: Final = ("S-A", "S")
ARCH_SEEDS: Final = (0, 1, 2)
MARGIN_SEEDS: Final = (0, 1, 2, 3, 4)
ARM_729_KEY: Final = {"D-sign": "Dsign"}
"""Experiment 729's key (in `ARM_SPECS`) of each refitted arm whose name differs here."""
REFIT_729_ARMS: Final = ("A", "D", "D-sign")
"""Experiment 729's arms refitted on the rebuilt banks, with 729's protocol (cap 200)."""
REFIT_729_ARCH: Final = "bilstm"
MAX_EPOCHS: Final = 300
EPOCH_WINDOWS: Final = core729.EPOCH_WINDOWS
BATCH_SIZE: Final = core729.BATCH_SIZE
LEARNING_RATE: Final = core729.LEARNING_RATE
WEIGHT_DECAY: Final = core729.WEIGHT_DECAY
CLIP_NORM: Final = core729.CLIP_NORM
LR_PATIENCE: Final = core729.LR_PATIENCE
STOP_PATIENCE: Final = core729.STOP_PATIENCE
MASK_PROBABILITY: Final = core729.MASK_PROBABILITY
"""Half of the training windows of a masking style are masked."""
STYLES: Final = ("0", "K", "W", "R")
"""The masking styles: none, engine blocks, whole day, and random spans."""
MORNING_BLOCK_START_H: Final = (5.5, 8.5)
MORNING_BLOCK_HOURS: Final = (1.0, 8.0)
EVENING_BLOCK_START_H: Final = (15.0, 17.5)
EVENING_BLOCK_HOURS: Final = (2.0, 6.0)
"""The engine blocks of style K, in local time; the same ranges as the engine simulator's."""
CAUSAL_K: Final = (0, 2, 6, 12, 24, 48)
"""The numbers of half-hours after the scored half-hour that stay visible in the causal read-out."""
CAUSAL_VALIDATION_CUTS: Final = 4
"""Each validation window carries this many cuts, fixed by `MASK_SEED`."""
HOURLY_PAIR: Final = 2
"""Half-hours per hour of the weather sources."""
NEAREST_HOLIDAY_CLIP_DAYS: Final = 7
"""The nearest-holiday distance is clipped to plus or minus this many days."""
PRETRAIN_SEEDS: Final = (0, 1, 2)
PRETRAIN_REFIT_ISO: Final = "2026-07-01"
"""The refit that experiment 428's `build_fold` is given: no valid time on or after it is read."""
PRETRAIN_MARGIN_DAYS: Final = 2
"""Pre-training drops every half-hour within this many days of a held-out month or record day."""
ENCODER_WEATHER_COLUMNS: Final = 7
ENCODER_DATETIME_COLUMNS: Final = 21
ENCODER_INPUT_COLUMNS: Final = ENCODER_WEATHER_COLUMNS + ENCODER_DATETIME_COLUMNS
ENCODER_OUTPUT_CHANNELS: Final = 64
"""The weather encoder's 32 outputs and the datetime encoder's 32 outputs."""
ENCODER_PEAK_HOURS_UTC: Final = (11, 13)
"""The clear-sky peak of D8's encoder input table must lie in this window of UTC hours."""
TRANSFORMER_WIDTH: Final = 128
TRANSFORMER_LAYERS: Final = 3
TRANSFORMER_HEADS: Final = 4
TRANSFORMER_FF: Final = 512
TRANSFORMER_DROPOUT: Final = 0.1
GRU_TRANSFORMER_TOKEN: Final = 64

# --------------------------------------------------------------------------- compute and stop line

STOP_LINE_HOURS: Final = 8.0
"""The manager's default stop line, in GPU-process-hours at the epoch cap."""
MEMORY_CAP_GB: Final = 14.0
"""Slot A2's memory cap; the pre-training row build's peak resident memory must stay under it."""
GPU_MEMORY_GB: Final = 6.0
TIMING_EPOCHS: Final = 5
TIMING_PRETRAIN_EPOCHS: Final = 2
SLOW_SECONDS_PER_EPOCH: Final = {"transformer": 1.0, "gru_transformer": 1.2}
"""The plan's estimates, used only to price a fit before it is measured."""
N_FITS: Final = 59
N_PRETRAIN_RUNS: Final = 3

# --------------------------------------------------------------------------- arms and contrasts

ARMS_NEW: Final = (
    "S-A", "S", "SD-0", "SD-K", "SD-W", "SD-R", "SD*", "SDe-PT", "SDe-FR", "SD*-C", "SD*-lambda",
)  # fmt: skip
CONTRASTS: Final = {
    "C1": ("SD*", "S"), "C2": ("SD-K", "SD-0"), "C3": ("SD-W", "SD-0"), "C4": ("SD-R", "SD-0"),
    "C5": ("SDe-PT", "SDe-FR"), "C6": ("transformer", "bilstm"), "C7": ("SD*-C", "SD*"),
}  # fmt: skip
"""Each counted contrast as (arm minus arm); a negative difference is a gain."""
REPORT_ONLY_ARMS: Final = ("SD*-lambda", "A", "D", "D-sign")
"""A failed fit of these arms voids a report-only reading and does not change the verdict."""
NEEDED_ARMS: Final = ("S-A", "S", "SD-0", "SD-K", "SD-W", "SD-R")
"""A failed fit of one of these arms makes verdict row 2 (the screen, C1 to C4, and the margin)."""
VOID_BY_ARM: Final = {"SD*-C": "C7", "SDe-PT": "C5", "SDe-FR": "C5"}
"""A failed fit of one of these arms makes the contrast void; the verdict rows read the other arms."""
Verdict = Literal[
    "stopped", "invalid", "failed", "not supported", "supported", "partly supported", "small gain",
    "inconclusive",
]  # fmt: skip

# --------------------------------------------------------------------------- rounding and limits


def round_down_2sf(*, value: float) -> float:
    """Round a positive number down to two significant figures (0 for a number at or below 0)."""
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** (exponent - 1)
    return round(math.floor(round(value / scale, 9)) * scale, 1 - exponent + 9)


@dataclass(frozen=True)
class Limits:
    """The numbers computed once from the rebuilt test bank's truth, before any fit.

    Attributes:
        s: P0's Brier score: the share of reverse-flow half-hours of the test bank.
        s1: The share of reverse-flow half-hours among half-hours with a reading of at least 1 MVA.
        floor: The margin's floor f (2% of s, rounded down to two significant figures).
        harm_limit: The harm limit L (a quarter of s1 over one minus s1, rounded down).
        reliability_bar: The sign calibration bar, a tenth of s.
        n_rows: The number of scored half-hours.
        n_decidable: The number of scored half-hours with a reading of at least 1 MVA.
    """

    s: float
    s1: float
    floor: float
    harm_limit: float
    reliability_bar: float
    n_rows: int
    n_decidable: int


def compute_limits(*, x: np.ndarray, r: np.ndarray) -> Limits:
    """Return f, L, and the calibration bar from the test bank's truth and reading (no model output).

    Args:
        x: The signed truth in points of every scored half-hour.
        r: The reading in points of the same half-hours.

    Returns:
        The limits.
    """
    s = float(np.mean(x < 0.0))
    decidable = r >= DECIDABLE_READING_POINTS
    s1 = float(np.mean(x[decidable] < 0.0)) if decidable.any() else float("nan")
    return Limits(
        s=s,
        s1=s1,
        floor=round_down_2sf(value=MARGIN_FLOOR_FRACTION * s),
        harm_limit=round_down_2sf(value=HARM_FRACTION * s1 / (1.0 - s1)),
        reliability_bar=RELIABILITY_FRACTION * s,
        n_rows=len(x),
        n_decidable=int(decidable.sum()),
    )


def limit_in_range(*, limit: float) -> bool:
    """Return whether L lies in `LIMIT_RANGE`; outside it the line stops at verdict row 0."""
    return LIMIT_RANGE[0] <= limit <= LIMIT_RANGE[1]


def write_once_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file once and refuse to overwrite it.

    The file is created with a hard link from a temporary name, so a second writer fails with
    `FileExistsError` and the first file is never replaced, even by a concurrent process.

    Args:
        path: The file.
        payload: The record.

    Raises:
        FileExistsError: When the file exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_limits_file(*, path: Path, limits: Limits) -> None:
    """Write `limits.json` once; a second write raises `FileExistsError`."""
    write_once_json(path=path, payload=limits.__dict__)


def read_limits_file(*, path: Path) -> Limits:
    """Read `limits.json`, and fail when it does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: f and L must be written before any fit"
        raise FileNotFoundError(msg)
    return Limits(**json.loads(path.read_text()))


def margin(*, floor: float, val_brier_s_by_seed: Sequence[float]) -> float:
    """Return m: the larger of f and twice the sd over seeds 0 to 4 of arm S's validation Brier score.

    Args:
        floor: The margin's floor f.
        val_brier_s_by_seed: Arm S's validation Brier score for each of seeds 0 to 4.

    Returns:
        The margin.
    """
    if len(val_brier_s_by_seed) != len(MARGIN_SEEDS):
        msg = f"the margin needs {len(MARGIN_SEEDS)} seeds, got {len(val_brier_s_by_seed)}"
        raise ValueError(msg)
    sd = float(np.std(np.asarray(val_brier_s_by_seed, dtype=np.float64), ddof=1))
    return max(floor, MARGIN_SD_MULTIPLE * sd)


def write_margin_file(*, path: Path, floor: float, val_brier_s_by_seed: Sequence[float]) -> dict:
    """Write `margin.json` once; a second write raises `FileExistsError`.

    Args:
        path: Where `margin.json` goes.
        floor: The margin's floor f.
        val_brier_s_by_seed: Arm S's validation Brier score for seeds 0 to 4.

    Returns:
        The record written.
    """
    record = {
        "margin": margin(floor=floor, val_brier_s_by_seed=val_brier_s_by_seed),
        "floor": floor,
        "val_brier_s_by_seed": [float(v) for v in val_brier_s_by_seed],
    }
    write_once_json(path=path, payload=record)
    return record


def read_margin_file(*, path: Path) -> float:
    """Return the stored margin, and fail when the file does not exist yet."""
    if not path.exists():
        msg = f"{path} does not exist: m must be written before any test contrast is read"
        raise FileNotFoundError(msg)
    return float(json.loads(path.read_text())["margin"])


# --------------------------------------------------------------------------- choices on validation


def choose_architecture(
    *, scores: Mapping[str, float], weights: Mapping[str, int], floor: float
) -> str:
    """Return the architecture with the lowest validation Brier score, or a lighter one in the tie.

    Args:
        scores: The mean over the 6 fits of arms S-A and S (seeds 0 to 2) of the validation Brier
            score, by architecture.
        weights: The number of weights of each architecture.
        floor: The margin's floor f; an architecture within `TIE_FRACTION * f` of the best wins if it
            has fewer weights.

    Returns:
        The chosen architecture's name.
    """
    best = min(scores, key=lambda a: scores[a])
    close = [a for a in scores if scores[a] <= scores[best] + TIE_FRACTION * floor]
    return min(close, key=lambda a: (weights[a], scores[a]))


def choose_style(
    *, val_brier: Mapping[str, float], whole_day_demand_mae: Mapping[str, float], floor: float
) -> str:
    """Return the masking style of SD*: the lowest validation Brier score, or a close one with better demand.

    A style within `TIE_FRACTION * f` of the lowest wins if its demand head has the lower mean
    validation error on windows with the whole day hidden.

    Args:
        val_brier: The mean over seeds 0 to 2 of each style's validation Brier score.
        whole_day_demand_mae: Each style's mean validation demand error on whole-day-hidden windows.
        floor: The margin's floor f.

    Returns:
        One of `STYLES`.
    """
    best = min(val_brier, key=lambda s: val_brier[s])
    close = [s for s in val_brier if val_brier[s] <= val_brier[best] + TIE_FRACTION * floor]
    return min(close, key=lambda s: (whole_day_demand_mae[s], val_brier[s]))


# --------------------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class VerdictInputs:
    """Everything the first-match verdict table reads.

    Attributes:
        stopped: Row 0 applies.
        invalid: Row 1 applies (a correctness check failed).
        failed: Row 2 applies (a needed fit raised and was not rerun).
        h1_pass: Harm check H1 passed for SD*.
        h2_pass: Harm check H2 passed for SD*.
        c1: The headline contrast C1 (SD* minus S).
        margin: The margin m.
        calibration_pass: The sign calibration bar passed.
        detection_pass: The detection bar passed.
    """

    stopped: bool
    invalid: bool
    failed: bool
    h1_pass: bool
    h2_pass: bool
    c1: core729.Reading
    margin: float
    calibration_pass: bool
    detection_pass: bool


VERDICT_WORDS: Final = {
    0: "stopped", 1: "invalid", 2: "failed", 3: "not supported", 4: "supported",
    5: "partly supported", 6: "small gain", 7: "inconclusive", 8: "not supported",
}  # fmt: skip


def verdict_row(*, v: VerdictInputs) -> int:  # noqa: PLR0911
    """Return the first row of the verdict table that matches (0 to 8)."""
    if v.stopped:
        return 0
    if v.invalid:
        return 1
    if v.failed:
        return 2
    if not (v.h1_pass and v.h2_pass):
        return 3
    gain = v.c1.point <= -v.margin and v.c1.upper < 0.0
    if gain and v.calibration_pass and v.detection_pass:
        return 4
    if gain:
        return 5
    if v.c1.upper < 0.0:
        return 6
    includes_zero = v.c1.lower <= 0.0 <= v.c1.upper
    if includes_zero and (
        v.c1.half_width >= v.margin or (v.c1.point <= -v.margin and v.c1.half_width < v.margin)
    ):
        return 7
    return 8


def verdict_word(*, v: VerdictInputs) -> Verdict:
    """Return the verdict word of `verdict_row`."""
    return VERDICT_WORDS[verdict_row(v=v)]  # ty: ignore[invalid-return-type]


# --------------------------------------------------------------------------- stop line and timing


PRETRAIN_CAP_EPOCHS: Final = 60
"""Experiment 510's epoch cap, which the pre-training loop keeps."""
INFERENCE_HOURS: Final = 0.2
"""The plan's allowance for validation, test, the 48-pass causal protocol, and the masked read-out.

When the timing stage has measured one causal read-out, the allowance is this value less the
estimated share of the causal read-out, which the measurement replaces (see `inference_hours`)."""
CAUSAL_READOUTS: Final = 18
"""One causal read-out per value of k (6) and per seed (3) of SD*-C."""
CAUSAL_ESTIMATE_HOURS: Final = 0.1
"""The part of `INFERENCE_HOURS` that the plan allows for the 18 causal read-outs."""


def inference_hours(*, causal_seconds_per_call: float | None) -> float:
    """Return the inference allowance, with the causal read-out measured when a timing is given.

    Args:
        causal_seconds_per_call: The measured seconds of one `causal_predictions` call at k = 0 on
            the test bank, or None to use the plan's allowance.

    Returns:
        The hours of inference.
    """
    if causal_seconds_per_call is None:
        return INFERENCE_HOURS
    return (
        INFERENCE_HOURS - CAUSAL_ESTIMATE_HOURS + CAUSAL_READOUTS * causal_seconds_per_call / 3600.0
    )


PRETRAIN_ESTIMATE_HOURS: Final = 0.45
"""The plan's upper estimate of the 3 pre-training runs, used until a run has been timed."""


def projected_hours(
    *,
    seconds_per_epoch: Mapping[str, float],
    chosen: str | None,
    epochs: int = MAX_EPOCHS,
    pretrain_seconds_per_epoch: float | None = None,
    demand_seconds_per_epoch: float | None = None,
    causal_seconds_per_call: float | None = None,
) -> float:
    """Return the GPU-process-hours of the plan's 56 GPU fits at `epochs` epochs each.

    The 24 screen fits use each architecture's own time (6 per architecture), and the 2 fits of arm S
    for seeds 3 and 4 and every later new fit (12 style, 6 encoder, 3 causal, 3 lambda fits) use the
    chosen architecture's time, or the slowest architecture's before the screen has chosen. The 9
    refits of experiment 729's arms use the bilstm's time at 200 epochs.

    Args:
        seconds_per_epoch: The measured seconds per epoch of each architecture.
        chosen: The chosen architecture, or None before the screen.
        epochs: The cap of every new fit.
        pretrain_seconds_per_epoch: The measured seconds an epoch of pre-training, or None to use
            the plan's estimate for the 3 runs.
        demand_seconds_per_epoch: The measured seconds an epoch of SD-W on the bilstm. Its ratio to
            arm S on the bilstm (at least 1) scales the later fits, which all carry the demand head.
        causal_seconds_per_call: The measured seconds of one causal read-out at k = 0 on the test
            bank. It is scaled by the 18 read-outs (6 values of k, 3 seeds).

    Returns:
        The projection in hours, with the pre-training and the inference allowance.
    """
    screen = sum(6 * seconds_per_epoch[a] for a in ARCHITECTURES)
    later_rate = seconds_per_epoch[chosen] if chosen else max(seconds_per_epoch.values())
    if demand_seconds_per_epoch is not None:
        later_rate *= max(1.0, demand_seconds_per_epoch / seconds_per_epoch["bilstm"])
    later = (2 + 12 + 6 + 3 + 3) * later_rate
    refit = 9 * seconds_per_epoch[REFIT_729_ARCH] * core729.MAX_EPOCHS / epochs
    fits = (screen + later + refit) * epochs / 3600.0
    if pretrain_seconds_per_epoch is None:
        pretrain = PRETRAIN_ESTIMATE_HOURS
    else:
        pretrain = N_PRETRAIN_RUNS * pretrain_seconds_per_epoch * PRETRAIN_CAP_EPOCHS / 3600.0
    return fits + pretrain + inference_hours(causal_seconds_per_call=causal_seconds_per_call)


def timing_gate_report(
    *,
    seconds_per_epoch: Mapping[str, float],
    peak_memory_gb: float,
    chosen: str | None = None,
    pretrain_seconds_per_epoch: float | None = None,
    demand_seconds_per_epoch: float | None = None,
    causal_seconds_per_call: float | None = None,
) -> dict[str, float | bool]:
    """Return the projection at the cap, and whether it and the peak memory pass the gate.

    Args:
        seconds_per_epoch: The measured seconds per epoch of each architecture.
        peak_memory_gb: The peak resident memory of the pre-training row build, in GB.
        chosen: The chosen architecture, once the screen has chosen.
        pretrain_seconds_per_epoch: The measured seconds an epoch of pre-training, or None.
        demand_seconds_per_epoch: The measured seconds an epoch of SD-W on the bilstm, or None.
        causal_seconds_per_call: The measured seconds of one causal read-out, or None.

    Returns:
        The projection, the peak memory, and `within_stop_line` and `memory_ok`.
    """
    cap = projected_hours(
        seconds_per_epoch=seconds_per_epoch, chosen=chosen,
        pretrain_seconds_per_epoch=pretrain_seconds_per_epoch,
        demand_seconds_per_epoch=demand_seconds_per_epoch,
        causal_seconds_per_call=causal_seconds_per_call,
    )  # fmt: skip
    return {
        "hours_at_cap": cap, "stop_line_hours": STOP_LINE_HOURS,
        "within_stop_line": cap <= STOP_LINE_HOURS, "peak_memory_gb": peak_memory_gb,
        "memory_cap_gb": MEMORY_CAP_GB, "memory_ok": peak_memory_gb < MEMORY_CAP_GB,
    }  # fmt: skip


def timing_gate_passes(*, report: Mapping[str, float | bool], manager_approved: bool) -> bool:
    """Return whether fits may start.

    The peak memory must be under the cap, and the projection must be within the stop line or the
    manager must have approved it.
    """
    return bool(report["memory_ok"] and (report["within_stop_line"] or manager_approved))


# --------------------------------------------------------------------------- the gallery's list

PANEL_REQUIREMENTS: Final = {
    "G1_ramp": "the engine output rises over 2 half-hours to its running level",
    "G1_evening": "the engine output is above zero on an evening block and zero before it",
    "G1_morning": "the engine output is above zero on a morning block",
    "G1_weekend": "the engine output is drawn for a weekend or bank-holiday day",
    "G1_part_load": "the engine output stays below the sum of its unit sizes",
    "G1_units": "the one-unit day and the several-unit day differ in their number of levels",
    "G1_trip": "the engine output drops to zero inside a running block",
    "G1_spell": "the engine output stays above zero through a multi-day spell",
    "G2_clear": "every clear-day panel has solar output above zero from mid-morning to mid-afternoon",
    "G2_cloudy": "every cloudy-day panel has solar output below the clear-day output of its season",
    "G3_aggregate": "the S1 aggregate bands are ordered 10% at most 50% at most 90% in every season",
    "G3_private": "the single S1 days are written to the private folder only",
    "G4_bsp": "the synthetic BSP steps differ from the real steps where the injection is above zero",
    "G5_fold": "the folded reading never falls below the reactive floor and equals the truth far from zero",
    "G6_base": "the no-injection day's truth equals the base demand",
    "G7_masks": "the hidden span of each masking style is zero in the reading channel and shaded",
    "G8_realism": "the real and the synthetic aggregates are drawn on the same axes; no real farm series",
    "G9_components": "every share of the component checks is drawn against the bar MIN_OUTPUT_SHARE",
}
"""The written list of what each gallery panel must show. The design lead checks each panel against
it and records the result in `PRIVATE/gallery/panel_check.json`."""


# --------------------------------------------------------------------------- the schedule


def screen_jobs() -> list[tuple[str, str, int]]:
    """Return the architecture screen: arms S-A and S on 4 architectures and 3 seeds (24 fits)."""
    return [(a, arch, s) for a in SCREEN_ARMS for arch in ARCHITECTURES for s in ARCH_SEEDS]


def style_jobs(*, arch: str) -> list[tuple[str, str, int]]:
    """Return the masking-style fits SD-0, SD-K, SD-W, and SD-R (seeds 0 to 2: 12 fits)."""
    return [(f"SD-{s}", arch, seed) for s in STYLES for seed in ARCH_SEEDS]


def margin_fit_jobs(*, arch: str) -> list[tuple[str, str, int]]:
    """Return the 2 further fits of arm S (seeds 3 and 4) that set the margin, after the screen."""
    return [("S", arch, s) for s in (3, 4)]


def later_jobs(*, arch: str) -> list[tuple[str, str, int]]:
    """Return the fits after the style choice: SDe-PT, SDe-FR, SD*-C, and SD*-lambda (12 fits)."""
    return [(a, arch, s) for a in ("SDe-PT", "SDe-FR", "SD*-C", "SD*-lambda") for s in ARCH_SEEDS]


def refit_jobs() -> list[tuple[str, str, int]]:
    """Return the 9 refits of experiment 729's arms A, D, and D-sign on the bilstm."""
    return [(a, "bilstm", s) for a in REFIT_729_ARMS for s in ARCH_SEEDS]
