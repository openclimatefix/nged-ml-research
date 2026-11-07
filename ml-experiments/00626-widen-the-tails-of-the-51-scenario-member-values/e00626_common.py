# ruff: noqa: E501
"""Shared pieces of experiment 626: the stored runs of experiment 436 and the four arms' value rules.

No fit lives here. A run file (`work/runs/<date>.npz` of experiment 436) holds the 51 values of
25 nodes (16 Primaries, 3 wind farms, 6 solar farms) at 702 half-hours, ascending along the last
axis, at levels k/52, and the actuals. The arms change the 51 values and keep the Schaake ranks
of experiment 436's arm J3.
"""

import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_436: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00436-joint-scenarios-across-demand-wind-solar-and-leads" / "work"
)
N_STEPS: Final[int] = 702
N_PRIMARIES: Final[int] = 16
WIND_FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SOLAR_FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
STEP: Final[timedelta] = timedelta(minutes=30)
KIND: Final[np.ndarray] = np.array([0] * 16 + [1] * 3 + [2] * 6, dtype=np.int8)

# Experiment 436's scoring module imports its build module, which imports torch through the wind
# code. Scoring needs only the constants below, so a stand-in module supplies them.
_stub = types.ModuleType("e00436_build")
_stub.N_NODES = 25  # ty: ignore[unresolved-attribute]
_stub.N_STEPS = N_STEPS  # ty: ignore[unresolved-attribute]
_stub.N_MEMBERS = 51  # ty: ignore[unresolved-attribute]
_stub.N_PRIMARIES = N_PRIMARIES  # ty: ignore[unresolved-attribute]
_stub.WIND_FARMS = WIND_FARMS  # ty: ignore[unresolved-attribute]
_stub.SOLAR_FARMS = SOLAR_FARMS  # ty: ignore[unresolved-attribute]
_stub.STEP = STEP  # ty: ignore[unresolved-attribute]
_stub.WORK_DIR = WORK_436  # ty: ignore[unresolved-attribute]
_stub.KIND = KIND  # ty: ignore[unresolved-attribute]
sys.modules.setdefault("e00436_build", _stub)

SLUG: Final[str] = "00626-widen-the-tails-of-the-51-scenario-member-values"
WORK_626: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
FAMILIES: Final[tuple[str, ...]] = ("demand", "wind", "solar")
N: Final[int] = 51
MID: Final[int] = 25
"""Index of the median value (level 26/52 = 0.5)."""
BANDS: Final[dict[str, tuple[float, float]]] = {
    "h0_12": (0.0, 12.0),
    "h12_36": (12.0, 36.0),
    "h36_120": (36.0, 120.0),
    "h120_240": (120.0, 240.0),
}
"""Lead bands in hours, open below and closed above."""
LEVELS: Final[dict[str, float]] = {"q05": 0.05, "q10": 0.10, "q90": 0.90, "q95": 0.95}
TEST_MONTHS: Final[tuple[str, ...]] = (
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
FACTOR_GRID: Final[np.ndarray] = np.round(np.arange(0.8, 2.501, 0.01), 2)
STRETCH_GRID: Final[np.ndarray] = np.round(np.arange(0.9, 1.801, 0.01), 2)
INSTABILITY: Final[float] = 0.30
TARGET: Final[float] = 0.05


def band_of_step() -> np.ndarray:
    """Return the lead band index (0 to 3) of each step, or -1 beyond 240 hours."""
    lead = (np.arange(N_STEPS) + 1) * 0.5
    out = np.full(N_STEPS, -1, dtype=int)
    for i, (low, high) in enumerate(BANDS.values()):
        out[(lead > low) & (lead <= high)] = i
    return out


def valid_times(*, init: datetime) -> list[datetime]:
    """Return the valid time of each of the 702 steps of a run."""
    return [init + (i + 1) * STEP for i in range(N_STEPS)]


def load_run(*, path: Path) -> dict:
    """Return a run file's arrays and its init time."""
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["init"] = datetime.fromisoformat(str(out["init"]) + "+00:00").astimezone(UTC)
    out["values"] = out["values"].astype(np.float64)
    out["actual"] = out["actual"].astype(np.float64)
    return out


def weibull(*, values: np.ndarray, level: float) -> np.ndarray:
    """Return the weibull quantile (position level x 52) of sorted values (..., 51)."""
    pos = level * (N + 1) - 1
    low = int(np.floor(pos))
    frac = pos - low
    return values[..., low] * (1 - frac) + values[..., low + 1] * frac


def clip_physical(*, x: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """Clip wind and solar values (axis 0 = node) to [0, 100] points of capacity."""
    physical = (kind > 0).reshape(-1, *([1] * (x.ndim - 1)))
    return np.where(physical, np.clip(x, 0.0, 100.0), x)


def stretched(*, values: np.ndarray, stretch: np.ndarray) -> np.ndarray:
    """Return the 51 values re-drawn at levels spread by `stretch` about the median (..., 51).

    Member j takes the value of the quantile function at index 25 + stretch x (j - 25), where the
    quantile function is linear between the stored values and extended past the first and last
    value by the slope of the first and last pair.
    """
    x = MID + stretch[..., None] * (np.arange(N) - MID)
    i = np.clip(np.floor(x).astype(int), 0, N - 2)
    frac = x - i
    low = np.take_along_axis(values, i, axis=-1)
    high = np.take_along_axis(values, i + 1, axis=-1)
    return low * (1 - frac) + high * frac


def rescaled(*, x: np.ndarray, med: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Return x (..., 51) rescaled about `med` (..., 1) by `up` above it and `down` below it."""
    factor = np.where(x >= med, up[..., None], down[..., None])
    return med + factor * (x - med)
