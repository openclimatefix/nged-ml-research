"""Constants, paths, and small file helpers shared by the scripts of experiment 737.

Nothing here reads private data. Every file the experiment writes goes under `OUT`, which is
`data/ml-experiments/00737-bayesian-solar-capacity-part-1/` unless the environment variable
`E00737_OUT` names another folder (the smoke test uses a scratch folder).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLDER: Final[str] = "00737-bayesian-solar-capacity-part-1"


def _output_folder() -> Path:
    """Return the experiment's output folder, refusing an override that is not a scratch folder.

    Returns:
        `E00737_OUT` when set (and `E00737_SCRATCH=1` or a `scratch` path component allows it),
        otherwise the experiment's folder under the shared data folder.

    Raises:
        RuntimeError: When `E00737_OUT` is set to a path that is not marked as a scratch folder.
    """
    override = os.environ.get("E00737_OUT")
    if override is None:
        return ML_EXPERIMENTS_DATA_DIR / FOLDER
    path = Path(override)
    if os.environ.get("E00737_SCRATCH") != "1" and "scratch" not in path.parts:
        msg = f"E00737_OUT={override} is not a scratch folder (set E00737_SCRATCH=1 to allow it)"
        raise RuntimeError(msg)
    return path


OUT: Final[Path] = _output_folder()
"""The only folder this experiment writes to."""

# --- Input files, all read only -------------------------------------------------------------
E327: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
"""Experiment 327's store: `power/power_<id>.npy` and `forecast/C/forecast_<id>.npy`."""
FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep/inputs/features.parquet"
)
"""ERA5 `temperature_2m` and `wind_speed_10m`, and the `is_holiday` flag, per series."""
FARMS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00179-metered-solar-farm-exploration/cleaned_solar_farms.parquet"
)
"""Six metered solar farms, `time_series_id`, `time`, `power`, `clear_sky_ghi_w_m2`."""
CAPACITY_TABLE_649: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00649-pv-capacity-table-season-year-from-stored-probes/work/capacity_table.parquet"
)
"""Experiment 649's season-year capacity table (the reference for the real-record match)."""
FARM_IDS: Final[tuple[int, ...]] = (21, 22, 23, 29, 30, 31)
"""The six farms' series identifiers, in the order that numbers them farm 0 to farm 5."""

# --- Grid and time --------------------------------------------------------------------------
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
"""The start of half-hour index 0."""
GRID_LENGTH: Final[int] = 78816
"""Half-hours from 2022-01-01 to 2026-07-01 (exclusive), 18 quarters."""
CUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Nothing after this instant is read. It is the end of the last half-hour of the grid."""
N_QUARTERS: Final[int] = 18
HALF_HOURS_PER_DAY: Final[int] = 48
WEATHER_WARMUP_DAYS: Final[int] = 5
"""ERA5 is read this many days before the grid starts, so that the 72-hour trailing mean exists."""

# --- Primaries and priors -------------------------------------------------------------------
PRIMARIES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six signed-megawatt Primaries of part 1, in ascending D-number order."""
FLEET_MIX: Final[tuple[float, float, float]] = (0.2, 0.6, 0.2)
"""East, south, and west shares of the prior's fleet."""
PRIOR_DC_AC: Final[float] = 1.15
MIN_RELIABILITY: Final[float] = 0.9
DAYLIGHT_GHI_W_M2: Final[float] = 5.0
"""A half-hour with CAMS irradiance above this is a daylight half-hour of the noise model."""
HDD_BASE_C: Final[float] = 15.5
FAIMAN_U0: Final[float] = 25.0
FAIMAN_U1: Final[float] = 6.84
TEMPERATURE_COEFFICIENT_PER_K: Final[float] = -0.004
DRIFT_MW_PER_QUARTER: Final[float] = 0.10
FIRST_QUARTER_MEAN_MW: Final[float] = 2.0
FIRST_QUARTER_SD_MW: Final[float] = 2.0
GROWTH_SD_GRID: Final[tuple[float, ...]] = (0.15, 0.30, 0.60)
RIDGE_SCALE_GRID: Final[tuple[float, ...]] = (1.0, 0.1, 0.01)
"""Multipliers of the baseline's ridge precision (`RIDGE_PER_COLUMN`) that the tuning compares."""
RIDGE_STEP_GROWTH_SD: Final[float] = 0.30
"""Growth standard deviation held fixed while the tuning chooses the ridge scale."""


class Tuned(NamedTuple):
    """The two values that the tuning chooses, read from `tuned.json` by every scored fit."""

    growth_sd: float
    """Growth standard deviation of the random-walk prior, megawatts per quarter."""
    ridge_scale: float
    """Multiplier of the baseline's ridge precision."""


QT_MEAN_MW: Final[float] = 2.0
QT_SD_MW: Final[float] = 5.0
DOF: Final[float] = 4.0
EM_MAX_ITER: Final[int] = 30
EM_TOL_CAPACITY_MW: Final[float] = 2e-3
EM_TOL_SIGMA: Final[float] = 1e-3
WARM_ITER: Final[int] = 10
RATIO_GRID: Final[tuple[float, ...]] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
INTERVAL_Z: Final[float] = 1.2815515655446004
"""The 90th percentile of the standard normal, so that mean +/- z sd is a central 80% interval."""
MAX_AUTOCORR: Final[float] = 0.95
MIN_QUARTER_SHARE: Final[float] = 0.6
"""A quarter is scored when this share of its daylight half-hours is usable."""
MIN_Q_ROWS: Final[int] = 1000

# --- Worlds ---------------------------------------------------------------------------------
SIZES_MW: Final[tuple[float, ...]] = (1.0, 2.5, 5.0)
SHAPES: Final[tuple[str, ...]] = ("ramp", "step", "accelerating")
TUNING_SEEDS: Final[tuple[int, ...]] = tuple(range(737000, 737010))
TUNING_SIZES_MW: Final[tuple[float, ...]] = (2.0, 4.0)
MAIN_SEEDS: Final[tuple[int, ...]] = (737101, 737102, 737103)
MATCHED_SEEDS: Final[tuple[int, ...]] = (737101, 737102)
SPECIAL_SEED: Final[int] = 737101
"""The seed of the cloud, transfer, placebo, and control worlds."""
STEP_QUARTER: Final[int] = 8
"""The step shape is zero before this quarter and the final capacity from it."""
TILT_RANGE_DEG: Final[tuple[float, float]] = (15.0, 40.0)
RATIO_RANGE: Final[tuple[float, float]] = (1.0, 1.5)
DIRICHLET_CONCENTRATION: Final[float] = 20.0
"""The mixture of an injected fleet is a Dirichlet draw with this total around the prior's mix."""
CLOUD_LOAD_SHARE: Final[float] = 0.25
TRANSFER_MW: Final[float] = 1.5
TRANSFER_DAYS: Final[int] = 30
TRANSFER_COUNT: Final[int] = 3
FARM_MIN_FINITE_SHARE: Final[float] = 0.6
FARM_PHYSICS_SHARE: Final[float] = 0.5
MIN_ELEVATION_DEG: Final[float] = 5.0
EXPECTED_INJECTED_WORLDS: Final[int] = 210
EXPECTED_UNINJECTED_FITS: Final[int] = 12
EXPECTED_TUNING_WORLDS: Final[int] = 120

# --- Gates, scoring, and the run ------------------------------------------------------------
G1_DRAWS: Final[int] = 100
G1_SEED: Final[int] = 737900
RIDGE_DRAWS: Final[int] = 30
"""Synthetic draws on which the tuning chooses the ridge scale."""
RIDGE_SEED: Final[int] = 737200
"""Seed of the first ridge draw. The draws use 737200 to 737229, which differ from every scored seed
(737101 to 737103), the tuning seeds (737000 to 737009), and the G1 seeds (737900 upwards)."""
G1_TOLERANCE: Final[float] = 0.03
G1_COVERAGE: Final[tuple[float, float]] = (0.75, 0.85)
G1B_MIN_SHARE: Final[float] = 0.95
G2_RTOL: Final[float] = 1e-9
G2_MIN_NONZERO_SHARE: Final[float] = 0.99
G2_BRIGHT_W_M2: Final[float] = 50.0
G5_MIN_CONVERGED: Final[float] = 0.99
C1_THRESHOLD: Final[float] = 0.10
C2_THRESHOLD: Final[float] = 0.05
SIGN_TEST_ALPHA: Final[float] = 0.025
B1_BAND: Final[tuple[float, float]] = (0.85, 1.15)
B2_BAND: Final[tuple[float, float]] = (0.65, 0.92)
B3_MAX_RATIO: Final[float] = 0.25
B4_MAX_DROP: Final[float] = 0.10
B5_MAX_DROP: Final[float] = 0.10
B6_MAX: Final[float] = 0.35
B7_MAX_MW: Final[float] = 0.25
BOOTSTRAP_DRAWS: Final[int] = 10000
BOOTSTRAP_SEED: Final[int] = 737
PAST_ONLY_CUTS: Final[tuple[int, ...]] = (5, 9, 13, 16)
"""Zero-based quarters at whose ends the real record is cut for the past-only sample."""
EXIT_FAILED: Final[int] = 3
"""Exit code of a failed task or stage."""
ARMS: Final[tuple[str, ...]] = ("A", "F", "Qt", "Q", "Ad", "Af", "O", "R")
"""The eight arms of a world, in storage order."""


def quarter_index(*, end_time: np.ndarray) -> np.ndarray:
    """Return the zero-based quarter of each half-hour, by its end time in UTC.

    The last half-hour of the grid ends exactly at the cut, which would start a nineteenth quarter,
    so the index is clipped to the last quarter.

    Args:
        end_time: Half-hour end stamps, `datetime64`.

    Returns:
        Integers 0 to `N_QUARTERS - 1`.
    """
    months = end_time.astype("datetime64[M]")
    year = months.astype("datetime64[Y]").astype(int) + 1970
    month = months.astype(int) % 12
    index = (year - GRID_START.year) * 4 + month // 3
    return np.clip(index, 0, N_QUARTERS - 1).astype(np.int64)


def grid_end_times() -> np.ndarray:
    """Return the end stamp of every half-hour of the grid (UTC, `datetime64[us]`).

    Returns:
        The stamps; the first is 00:30 on the first day, the last is the cut.
    """
    start = np.datetime64(GRID_START.replace(tzinfo=None), "us")
    return start + (np.arange(1, GRID_LENGTH + 1) * 30 * 60 * 1_000_000).astype("timedelta64[us]")


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


def atomic_npz(*, arrays: dict[str, np.ndarray], path: Path) -> None:
    """Write a compressed `.npz` under a temporary name and rename it.

    Args:
        arrays: The arrays by name.
        path: The final path (ending in `.npz`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.rename(path)


def sha256_of(*, path: Path) -> str:
    """Return the hex SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def time_log(*, stage: str, task: str, start: float, out: Path | None = None) -> None:
    """Append one line (stage, task, device, start, end) to the per-phase time log.

    Args:
        stage: The phase, such as `fit`.
        task: The task's name.
        start: The task's start, from `time.time()`.
        out: The folder that holds the log (`OUT` by default).
    """
    stamp = "%FT%T"
    begin, end = (time.strftime(stamp, time.gmtime(v)) for v in (start, time.time()))
    folder = out if out is not None else OUT
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "time_log.csv").open("a") as f:
        f.write(f"{stage},{task},cpu,{begin},{end}\n")
