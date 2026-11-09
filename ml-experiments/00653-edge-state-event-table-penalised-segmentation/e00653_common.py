"""Shared inputs of experiment 653: the residual panel, the neighbour graph, and the grid.

The residual is the observed cleaned power minus the forecast of the lag-free convex demand model
of experiment 130, as stored by experiment 327 (arm C). All arrays are on the half-hour grid
starting 2022-01-01 00:00 UTC and ending before 2026-07-01 (the final-test window is never read).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = 78816
BLOCK: Final[int] = 12
"""Half-hours per 6-hour block."""
N_BLOCKS: Final[int] = GRID_LENGTH // BLOCK
DAY: Final[int] = 48
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
E327: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
E580: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
)
NEIGHBOURS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00383-load-shift-xgboost-oracle-first-part-1/work/neighbours_nged.json"
)
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00653-edge-state-event-table-penalised-segmentation"
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 24, 25, 26)
"""The 15 series with a stored forecast (series 12, D15, is not on the grid)."""
MVA_SITES: Final[frozenset[str]] = frozenset()
"""Sites whose reading is MVA; filled by `load_panel` from the series table's group column."""
SHORT_HISTORY: Final[dict[str, int]] = {
    "D18": (datetime(2025, 3, 1, tzinfo=UTC) - GRID_START).days * DAY
}
"""Rows before this half-hour index are dropped (the noisy start of a short-history Primary)."""


def idx_of(*, when: datetime) -> int:
    """Half-hour index of a UTC datetime."""
    return int((when - GRID_START).total_seconds() // 1800)


SCORED_FROM: Final[int] = idx_of(when=datetime(2024, 1, 1, tzinfo=UTC))
"""Half-hour index of 2024-01-01: 2022 and 2023 are the tuning years, the rest is scored."""


class Panel:
    """Residual, power, capacity, and neighbours of the 15 Primaries."""

    def __init__(self) -> None:
        """Load the residual panel from `SERIES_PATH`."""
        table = pl.read_parquet(SERIES_PATH)
        info = {int(r["time_series_id"]): r for r in table.iter_rows(named=True)}
        self.sites = [info[s]["site"] for s in SERIES_IDS]
        self.capacity = {
            info[s]["site"]: float(info[s]["effective_capacity_mw"]) for s in SERIES_IDS
        }
        self.group = {info[s]["site"]: info[s]["group"] for s in SERIES_IDS}
        self.residual: dict[str, np.ndarray] = {}
        self.power: dict[str, np.ndarray] = {}
        for s in SERIES_IDS:
            site = info[s]["site"]
            power = np.load(E327 / f"power/power_{s}.npy")
            forecast = np.load(E327 / f"forecast/C/forecast_{s}.npy")
            cut = SHORT_HISTORY.get(site, 0)
            power = power.copy()
            power[:cut] = np.nan
            self.power[site] = power
            self.residual[site] = power - forecast
        raw = json.loads(NEIGHBOURS.read_text())
        known = set(self.sites)
        self.neighbours = {a: sorted(b for b in raw.get(a, []) if b in known) for a in self.sites}
        for a in self.sites:
            for b in self.neighbours[a]:
                if a not in self.neighbours[b]:
                    self.neighbours[b] = sorted([*self.neighbours[b], a])
        self.edges = sorted({tuple(sorted((a, b))) for a in self.sites for b in self.neighbours[a]})

    def mask_common_mode(self, *, residual: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Remove the median across Primaries of each Primary's residual in noise-free units."""
        stack = np.stack([residual[s] / self._scale(residual[s]) for s in self.sites])
        med = np.nanmedian(stack, axis=0)
        return {s: residual[s] - np.nan_to_num(med) * self._scale(residual[s]) for s in self.sites}

    @staticmethod
    def _scale(x: np.ndarray) -> float:
        d = np.nanmedian(np.abs(x - np.nanmedian(x)))
        return float(1.4826 * d) or 1.0


def block_means(*, x: np.ndarray) -> np.ndarray:
    """Mean of each 6-hour block, NaN when fewer than 6 half-hours have a reading."""
    b = x[: N_BLOCKS * BLOCK].reshape(N_BLOCKS, BLOCK)
    n = np.isfinite(b).sum(axis=1)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(np.isfinite(b), b, np.nan), axis=1)
    m[n < BLOCK // 2] = np.nan
    return m


def band_of(*, days: float | None) -> str:
    """Duration band of an event; None means the end lies beyond the record (permanent or open)."""
    if days is None:
        return "permanent"
    if days < 1:
        return "3h-1d"
    if days < 7:
        return "1-7d"
    if days < 21:
        return "7-21d"
    if days < 60:
        return "21-60d"
    return "60d+"
