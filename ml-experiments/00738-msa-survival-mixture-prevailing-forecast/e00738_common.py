"""Constants, paths, and small file helpers shared by the scripts of experiment 738.

Nothing here reads private data. Every file the experiment writes goes under `OUT`, which is
`data/ml-experiments/00738-msa-survival-mixture-prevailing-forecast/` unless the environment
variable `E00738_OUT` names another folder (the smoke test and the unit tests use a scratch folder).
Experiment 713's folder is only ever read.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDER: Final[str] = "00738-msa-survival-mixture-prevailing-forecast"
EXPERIMENT_ID: Final[int] = 738


def _output_folder() -> Path:
    """Return the experiment's output folder, refusing an override that is not a scratch folder.

    Returns:
        `E00738_OUT` when set (and `E00738_SCRATCH=1` or a `scratch` path component allows it),
        otherwise the experiment's folder under the shared data folder.

    Raises:
        RuntimeError: When `E00738_OUT` is set to a path that is not marked as a scratch folder.
    """
    override = os.environ.get("E00738_OUT")
    if override is None:
        return ML_EXPERIMENTS_DATA_DIR / FOLDER
    path = Path(override)
    if os.environ.get("E00738_SCRATCH") != "1" and "scratch" not in path.parts:
        msg = f"E00738_OUT={override} is not a scratch folder (set E00738_SCRATCH=1 to allow it)"
        raise RuntimeError(msg)
    return path


OUT: Final[Path] = _output_folder()
"""The only folder this experiment writes to."""

# --- Read-only inputs: experiment 713 ----------------------------------------------------------
NAME_713: Final[str] = "00713-msa-axial-attention-demand-network-part-1"
ROOT_713: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_713
"""Experiment 713's data folder. This experiment reads its stored states and forecasts, never
writes."""
CODE_HASH_713: Final[str] = "e85b84e2ea04b3cb"
"""`e00713_arms.code_hash()` that the stored `M1` fits carry (frozen checkout `7c29114a8`)."""
ROW_SET_ID: Final[str] = "backtest_2025_11_to_2026_06_daily_12utc_before_holdout"

# --- Arms, folds, seeds ------------------------------------------------------------------------
ARM_M1: Final[str] = "M1"
ARM_W0: Final[str] = "W0"
HEAD_ARMS: Final[tuple[str, ...]] = ("LP", "LD", "LM", "LR")
"""Frozen `M1` plus a level head: full persistence, decay, mixture, and mixture with learned
hazards."""
SURVIVAL_ARMS: Final[tuple[str, ...]] = ("LD", "LM", "LR")
"""The candidates for S*, in the order that breaks a tie (the simpler arm first)."""
FIT_ARMS: Final[tuple[str, ...]] = (ARM_W0, *HEAD_ARMS)
ALL_ARMS: Final[tuple[str, ...]] = (ARM_M1, *FIT_ARMS)
FOLDS: Final[tuple[str, ...]] = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
SCREEN_FOLDS: Final[tuple[str, ...]] = ("2025-11", "2026-01", "2026-03", "2026-06")
OTHER_FOLDS: Final[tuple[str, ...]] = tuple(m for m in FOLDS if m not in SCREEN_FOLDS)
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SCREEN_SEED: Final[int] = 0
SIX_MW: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT_MVA: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
REPORT_ONLY_PRIMARIES: Final[tuple[str, ...]] = ("D15", "D18")
NEVER_DRAWN: Final[str] = "D8"
"""The Primary whose real series is never drawn in a published plot."""

# --- Thresholds and decision rule --------------------------------------------------------------
N1_THRESHOLD: Final[float] = -0.05
N3_THRESHOLD: Final[float] = -0.03
V1_FAIL: Final[float] = 0.02
V2_FLAG: Final[float] = 0.10
Q_VALUES: Final[tuple[float, ...]] = (0.025, 0.05, 0.14)
Q_MAIN: Final[float] = 0.05
TIE_MARGIN: Final[float] = 0.10
N_BONFERRONI: Final[int] = 3
LEVEL_BONFERRONI: Final[float] = 100.0 * (1.0 - 0.05 / N_BONFERRONI)
"""98.33 percent: the level of the interval of N1, N3, and C2."""
LEVEL_PLAIN: Final[float] = 95.0
N_RESAMPLES: Final[int] = 100_000
NO_SHIFT_BAND: Final[float] = 0.05
"""N1 is `not supported` when its t interval lies inside plus or minus this."""
DEPARTURE_POINTS: Final[float] = 0.5
"""Points of capacity by which S*'s median departs from `M1`'s on real event-free rows."""
LEAD_BAND_HOURS: Final[tuple[float, float]] = (24.0, 240.0)
"""Scored leads: low inclusive, high exclusive, in hours after the 12:00 UTC contract origin."""
AGE_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under_10", 0.0, 10.0),
    ("over_10", 10.0, 1e9),
    ("over_15", 15.0, 1e9),
)
"""Age bands of C1 to C3: name, lower bound (inclusive) and upper bound (exclusive) in days."""
A1_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-2", 0.0, 3.0),
    ("3-5", 3.0, 6.0),
    ("6-10", 6.0, 11.0),
    ("11-15", 11.0, 16.0),
    ("over 15", 16.0, 1e9),
)
A1_FALSE_ALARM_MAX: Final[float] = 0.05
CURVE_ERROR_MAX: Final[float] = 0.05
CURVE_ERROR_DAYS: Final[tuple[float, ...]] = (1.0, 3.0, 7.0, 14.0, 28.0, 56.0)
STRATUM_TOLERANCE: Final[float] = 0.03
STRATUM_TOLERANCE_END: Final[float] = 0.01
EXPECTED_SHARES: Final[dict[str, float]] = {
    "far": 0.38,
    "middle": 0.10,
    "near": 0.015,
    "ended_24h": 0.015,
    "ended_1_7": 0.088,
    "event_free": 0.225,
    "onset_ahead": 0.17,
}
"""The stratum shares that the plan simulated, over origins on days 0 to 29 of a month."""

# --- Fits --------------------------------------------------------------------------------------
HEAD_LR: Final[float] = 1e-3
PASS_CAP: Final[int] = 60
PATIENCE: Final[int] = 8
REMEDY_PASS_CAP: Final[int] = 120
REMEDY_PATIENCE: Final[int] = 15
LEVEL_DEPTH_LIMIT: Final[float] = 0.7
"""Largest depth of the level term, in shares of capacity (covers experiment 650's 0.65)."""
N_HAZARDS: Final[int] = 10
HAZARD_OFFSET: Final[float] = 3.0
SURVIVAL_FLOOR: Final[float] = 1e-3
"""Floor on S(age) in the denominator of the survival ratio."""
HIDDEN: Final[int] = 64
N_HISTORY: Final[int] = 14
MIN_AGE_DAYS: Final[int] = 2
"""The newest history row is 2 days old at the contract origin."""
A1_Z: Final[float] = 2.0
A1_MIN_REFERENCE_ROWS: Final[int] = 3
MAD_TO_SD: Final[float] = 1.4826

# --- Worlds ------------------------------------------------------------------------------------
GAP_MEAN_DAYS: Final[float] = 118.0
BURN_IN_DAYS: Final[int] = 112
TRAIN_SHARE_EXPECTED: Final[float] = 0.25
SCORING_TRAIN_PASSES: Final[int] = 12
SCORING_HELD_PASSES: Final[int] = 4
HELD_FIRST: Final[int] = 200
SEED_SPACE_TRAIN: Final[int] = 7381
SEED_SPACE_VALID: Final[int] = 7382
SEED_SPACE_RESAMPLED: Final[int] = 7383
SEED_SPACE_650: Final[int] = 650
"""The first element of the seed key of experiment 650's generator, which the scoring worlds use."""
REAL_TRANSFERS: Final[int] = 66
REAL_SUBDAY_SHARE: Final[float] = 31 / 66

# --- Budget and the run ------------------------------------------------------------------------
STOP_GPU_HOURS: Final[float] = 7.0
STOP_CPU_HOURS: Final[float] = 12.0
BUDGET_MARKER: Final[str] = "budget_extended.ok"
EXIT_FAILED: Final[int] = 3
"""Exit code of a failed task or stage (the launcher's refusal uses another code)."""
EXIT_CHAIN_FAILED: Final[int] = 5
"""Exit code of a failed chain, and of the gate's second failure."""
EXIT_GPU_BUSY: Final[int] = 75
GPU_BACKOFF_SECONDS: Final[int] = 300
MEMORY_MAX_GB: Final[int] = 16


def atomic_json(*, obj: object, path: Path) -> None:
    """Write a JSON file under a temporary name and rename it.

    Args:
        obj: The JSON-serialisable object.
        path: The final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.rename(path)


def sha256_of(*, path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_log(*, stage: str, task: str, start: float, device: str, out: Path | None = None) -> None:
    """Append one line (stage, task, device, start, end) to the per-phase time log.

    Args:
        stage: The phase, such as `fit`.
        task: The task's name.
        start: The task's start, from `time.time()`.
        device: `cpu` or `gpu`.
        out: The folder that holds the log (`OUT` by default).
    """
    stamp = "%FT%T"
    begin, end = (time.strftime(stamp, time.gmtime(v)) for v in (start, time.time()))
    folder = out if out is not None else OUT
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "time_log.csv").open("a") as f:
        f.write(f"{stage},{task},{device},{begin},{end}\n")


def fit_paths(*, arm: str, month: str, seed: int, out: Path | None = None) -> dict[str, Path]:
    """Return the state, log, and done-marker paths of one fit.

    Args:
        arm: The arm.
        month: The fold month.
        seed: The seed.
        out: The output folder (`OUT` by default).

    Returns:
        The paths under `networks/`, `fits/`, and `done/`.
    """
    root = out if out is not None else OUT
    stem = f"{month}_s{seed}"
    return {
        "state": root / "networks" / arm / f"{stem}.pt",
        "log": root / "fits" / arm / f"{stem}.json",
        "done": root / "done" / f"fit_{arm}_{stem}.ok",
    }


def month_index(*, month: str) -> int:
    """Return the zero-based index of a fold month in `FOLDS`.

    Args:
        month: A fold month such as `2025-11`.

    Returns:
        The index.
    """
    return FOLDS.index(month)


# --- The grid of experiment 713 (guarded by a test against `e00713_layout`) ---------------------
GRID_START_US: Final[int] = 1_567_296_000_000_000
STEP_US: Final[int] = 1_800_000_000
NATIVE_OFFSET_STEPS: Final[int] = 6
WORLDS: Final[tuple[str, ...]] = ("train_family", "held_out", "resampled", "real")
GROUP_OF_PRIMARY: Final[dict[str, tuple[str, ...]]] = {
    "all14": (*SIX_MW, *EIGHT_MVA),
    "six": SIX_MW,
    "eight": EIGHT_MVA,
    "eight_no_d8": tuple(p for p in EIGHT_MVA if p != NEVER_DRAWN),
    "short_history_two": REPORT_ONLY_PRIMARIES,
}
EXIT_REMEDY: Final[int] = 6
"""Exit code of the gate's first failure: the remedy flag is set and `gate-remedy` runs next."""
