# ruff: noqa: E501
"""Gas A's day-block injection and the host choice for experiment 695's host arms.

Gas A is the 44-day record of a gas engine's output (experiment 194's stored `g2_window.parquet`).
Its output, units, and dates are never printed or drawn. The record is cut into whole UTC days,
and a day is drawn at random, with replacement, within its weekday class (Monday to Friday, or
Saturday and Sunday) for every day of the host's history. Ramps and part load survive inside each
day. Experiment 694's folded rule then subtracts the output from the host's real power.
"""

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl
from e00694_common import inject_folded, load_record  # ty: ignore[unresolved-import]
from e00695_common import (
    ERA1_START,
    HOLDOUT,
    HOST_CANDIDATES,
    Q_FLOOR_MVA,
    grid_index,
)

DAY: Final[int] = 48
OUTPUT_BIN_MW: Final[float] = 0.25
"""The width of the output bins the truth climatology and the add-back use."""
DRAW_SEED: Final[int] = 0


def record_days() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return Gas A's whole days as (output in MW, units running), grouped by weekday class.

    A day is kept when the record holds all 48 half-hours of the UTC date. The gap in the record
    removes its days.
    """
    frame = (
        load_record()
        .sort("time")
        .with_columns(date=pl.col("time").dt.date(), weekday=pl.col("time").dt.weekday())
    )
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"weekday": [], "weekend": []}
    for part in frame.partition_by("date", maintain_order=True):
        if part.height != DAY:
            continue
        klass = "weekend" if int(part["weekday"][0]) >= 6 else "weekday"
        out[klass].append(
            (
                np.clip(part["power"].to_numpy().astype(float), 0.0, None),
                part["units"].to_numpy().astype(int),
            )
        )
    return out


def drive_series(*, n_grid: int, seed: int = DRAW_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return the injected output (MW) and units on the pipeline grid, resampled by whole days.

    Days from `ERA1_START` to the end of June 2026 are drawn within their weekday class. Half-hours
    outside that span carry no engine.
    """
    days = record_days()
    rng = np.random.default_rng(seed)
    g = np.zeros(n_grid)
    units = np.zeros(n_grid, dtype=int)
    day = ERA1_START
    while day < HOLDOUT:
        pool = days["weekend" if day.weekday() >= 5 else "weekday"]
        out, running = pool[int(rng.integers(len(pool)))]
        start = grid_index(when=day)
        stop = min(start + DAY, n_grid)
        g[start:stop], units[start:stop] = out[: stop - start], running[: stop - start]
        day += timedelta(days=1)
    return g, units


def inject_engine(*, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Return the host's reading with the engine subtracted from its real power (experiment 694's folded rule)."""
    return inject_folded(y=y, g=g, q=Q_FLOOR_MVA)


def output_bins(*, g: np.ndarray) -> np.ndarray:
    """Return the output bin index (0 means off) of each half-hour of the injected output."""
    return np.rint(np.asarray(g) / OUTPUT_BIN_MW).astype(int)


def choose_host(*, medians: dict[str, float], target: float) -> str:
    """Return the host candidate whose median reading is closest to D8's tracked-off median."""
    return min(HOST_CANDIDATES, key=lambda label: abs(medians[label] - target))


def d8_tracked_off_median(*, y_d8: np.ndarray, flag: np.ndarray, start: int, stop: int) -> float:
    """Return D8's median reading (MVA) over `start` to `stop` on half-hours with no unit running.

    Args:
        y_d8: D8's readings on the pipeline grid (MVA, NaN where dropped).
        flag: Experiment 694's hindsight flag aligned to the same grid (`True` where a unit runs).
        start: The first half-hour.
        stop: One past the last half-hour.

    Returns:
        The median over finite readings on unflagged half-hours.
    """
    window = y_d8[start:stop]
    keep = np.isfinite(window) & ~flag[start:stop]
    return float(np.median(window[keep]))
