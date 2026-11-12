"""Paths, constants, loaders, injections and metrics shared by the scripts of experiment 694.

The engine record is called "Gas A" everywhere that is printed or drawn. D8 is a substation and may
be named.
"""

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "gas-engine-state-tracker-part-1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00694-{SLUG}"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00694-{SLUG}"
)
PANEL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
"""Experiment 196's stored panel: per-site `power` arrays on the half-hour grid from GRID_START."""
RECORD_PATH_GLOB: Final[str] = "00194-*/g2_window.parquet"
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
N_GRID: Final[int] = 78816
SOLAR_START: Final[datetime] = datetime(2025, 3, 24, tzinfo=UTC)
POST_SOLAR_START: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
ENGINE_CONNECTED: Final[datetime] = datetime(2023, 6, 13, tzinfo=UTC)
NET_START: Final[datetime] = datetime(2023, 9, 1, tzinfo=UTC)
NET_END: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
FADE_START: Final[datetime] = datetime(2026, 3, 1, tzinfo=UTC)
D8_SERIES: Final[int] = 26
PRE_ROLL: Final[int] = 28 * 48
"""Half-hours of history before a segment's first tracked half-hour, for the causal spread."""
SEGMENT_LEN: Final[int] = 120 * 48
INJECT_OFFSET: Final[int] = 38 * 48
"""Half-hours from the start of the tracked part of a segment to the start of the injection."""
Q_FLOOR: Final[float] = 0.14
"""Reactive-power floor in MVA, experiment 196's value."""
HOSTS: Final[tuple[str, ...]] = ("D3", "D4", "D11", "D14", "D16", "D2", "D10")
"""Host series; D8 before the engine is a further host in the first period only."""
EXPECTED_UNIT_COUNTS: Final[tuple[int, ...]] = (1704, 137, 133, 80)
"""Experiment 194's half-hours at 0, 1, 2, and 3 units."""


def stamp(*, when: datetime) -> int:
    """Index of the half-hour starting at `when`."""
    return int((when - GRID_START).total_seconds() // 1800)


def load_panel() -> dict[int, dict]:
    """Return the stored panel."""
    return pickle.loads(PANEL_PATH.read_bytes())


def panel_power(*, panel: dict[int, dict], label: str) -> np.ndarray:
    """Return a site's power array (MVA or MW, NaN where dropped) on the grid."""
    entry = next(v for v in panel.values() if v["site"] == label)
    return np.asarray(entry["power"], dtype=float)


def load_record() -> pl.DataFrame:
    """Return Gas A's stored 44-day record: time, power (MW), price, units."""
    path = next(ML_EXPERIMENTS_DATA_DIR.glob(RECORD_PATH_GLOB))
    return pl.read_parquet(path)


def record_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return the record's output (MW, clipped at zero) and its whole units running."""
    frame = load_record().sort("time")
    return np.clip(frame["power"].to_numpy().astype(float), 0.0, None), frame["units"].to_numpy()


def runs_of(*, units: np.ndarray) -> list[tuple[int, int]]:
    """Run-length encode a unit-count series as (units, duration in half-hours)."""
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(units) + 1):
        if i == len(units) or units[i] != units[start]:
            runs.append((int(units[start]), i - start))
            start = i
    return runs


def synthetic_units(*, units: np.ndarray, seed: int, k_cap: int) -> np.ndarray:
    """Resample the record's runs with replacement into a schedule as long as the record."""
    rng = np.random.default_rng(seed)
    runs = runs_of(units=units)
    out: list[int] = []
    while len(out) < len(units):
        level, duration = runs[int(rng.integers(len(runs)))]
        out.extend([min(level, k_cap)] * duration)
    return np.asarray(out[: len(units)], dtype=int)


def inject_folded(*, y: np.ndarray, g: np.ndarray, q: float = Q_FLOOR) -> np.ndarray:
    """Subtract output `g` (MW) from the real power behind the reading and read it back as |S|."""
    real = np.sqrt(np.maximum(y**2 - q**2, 0.0))
    return np.sqrt((real - g) ** 2 + q**2)


def inject_signed(*, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Subtract output `g` (MW) from the reading directly, floored at 0.01."""
    return np.maximum(y - g, 0.01)


def causal_spread(*, y: np.ndarray) -> np.ndarray:
    """Trailing 28-day robust spread of the lag-1 difference by half-hour of day.

    The spread is the median absolute lag-1 difference at the same half-hour of the day on each of
    the previous 28 days, times 1.4826. Entries without 7 earlier days are NaN.
    """
    n = len(y)
    d = np.full(n, np.nan)
    d[1:] = np.abs(np.diff(y))
    stack = np.full((28, n), np.nan)
    for j in range(1, 29):
        stack[j - 1, 48 * j :] = d[: n - 48 * j]
    count = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(stack, axis=0)
    spread = 1.4826 * med
    spread[count < 7] = np.nan
    return np.maximum(spread, 0.03)


SPREAD_LEVELS: Final[np.ndarray] = np.geomspace(0.04, 1.5, 8)


def quantise_spread(*, spread: np.ndarray) -> np.ndarray:
    """Index of the nearest of the 8 log-spaced spread levels."""
    filled = np.where(np.isfinite(spread), spread, np.nanmedian(spread))
    return np.argmin(np.abs(np.log(filled)[:, None] - np.log(SPREAD_LEVELS)[None, :]), axis=1)


def f1_counts(*, truth: np.ndarray, flag: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for boolean arrays."""
    return (
        int((truth & flag).sum()),
        int((~truth & flag).sum()),
        int((truth & ~flag).sum()),
        int((~truth & ~flag).sum()),
    )


def f1_of(*, tp: float, fp: float, fn: float) -> float:
    """Half-hour F1 from counts; 0 when nothing is flagged and nothing is true."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def switch_errors(*, truth: np.ndarray, flag: np.ndarray, window: int = 6) -> list[int]:
    """Absolute half-hour distance from each true switch to the nearest flagged switch.

    Only distances up to `window` are returned; a true switch with no flagged switch within the
    window is skipped, and the caller counts the match rate from the number of true switches.
    """
    t_sw = np.flatnonzero(np.diff(truth.astype(int)) != 0)
    f_sw = np.flatnonzero(np.diff(flag.astype(int)) != 0)
    if len(f_sw) == 0:
        return []
    out = []
    for s in t_sw:
        dist = int(np.min(np.abs(f_sw - s)))
        if dist <= window:
            out.append(dist)
    return out


def n_true_switches(*, truth: np.ndarray) -> int:
    """Number of changes of the boolean truth."""
    return int((np.diff(truth.astype(int)) != 0).sum())
