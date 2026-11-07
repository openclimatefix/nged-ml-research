"""Experiment 642: the five upper-tail corrections as pure functions of quantile arrays.

Every function takes and returns arrays of shape (rows, 11) in the order of `TAU`, and never
reads the actual value of a row it corrects. The fitting functions take the fitting rows' actuals.
"""

from typing import Final

import numpy as np

TAU: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
MEDIAN: Final[int] = 5
Q90: Final[int] = 8
Q95: Final[int] = 9
UPPER: Final[slice] = slice(MEDIAN + 1, None)
GRID: Final[np.ndarray] = np.round(np.arange(0.5, 3.0001, 0.05), 2)
MIN_FIT_ROWS: Final[int] = 500
MIN_POOL_ROWS: Final[int] = 300
PIT_EDGE: Final[float] = 0.001


def scale_upper(*, q: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    """Multiply each quantile's distance above q50 by `factor` (per row or scalar)."""
    out = q.copy()
    f = np.asarray(factor, dtype=np.float64)
    f = f[:, None] if f.ndim else f
    out[:, UPPER] = q[:, [MEDIAN]] + f * (q[:, UPPER] - q[:, [MEDIAN]])
    return np.sort(out, axis=1)


def fit_scale(*, y: np.ndarray, q: np.ndarray) -> float:
    """The grid factor whose scaled q90 is exceeded by the share of rows closest to 10%."""
    if len(y) < MIN_FIT_ROWS:
        return 1.0
    up = q[:, Q90] - q[:, MEDIAN]
    shares = np.array([np.mean(y > q[:, MEDIAN] + c * up) for c in GRID])
    return float(GRID[np.argmin(np.abs(shares - 0.10))])


def pit_knots(*, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Levels and quantile values with both ends extended linearly to 0.001 and 0.999."""
    lo_slope = (q[:, 1] - q[:, 0]) / (TAU[1] - TAU[0])
    hi_slope = (q[:, -1] - q[:, -2]) / (TAU[-1] - TAU[-2])
    lo = q[:, 0] + (PIT_EDGE - TAU[0]) * lo_slope
    hi = q[:, -1] + (1 - PIT_EDGE - TAU[-1]) * hi_slope
    levels = np.concatenate([[PIT_EDGE], TAU, [1 - PIT_EDGE]])
    return levels, np.column_stack([lo, q, hi])


def pit(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Probability integral transform of `y` under the piecewise-linear quantile function.

    On a flat segment (equal neighbouring quantile values) the level is the segment's midpoint.
    """
    levels, k = pit_knots(q=q)
    k = np.maximum.accumulate(k, axis=1)
    idx = np.clip((y[:, None] >= k).sum(axis=1) - 1, 0, len(levels) - 2)
    rows = np.arange(len(y))
    lo, hi = k[rows, idx], k[rows, idx + 1]
    width = hi - lo
    frac = np.where(width > 0, (y - lo) / np.where(width > 0, width, 1.0), 0.5)
    frac = np.clip(frac, 0.0, 1.0)
    return levels[idx] + frac * (levels[idx + 1] - levels[idx])


def fit_pit_map(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The training PIT's quantile at each of the 11 levels (the identity map if too few rows)."""
    if len(y) < MIN_FIT_ROWS:
        return TAU.copy()
    return np.clip(np.quantile(pit(y=y, q=q), TAU), PIT_EDGE, 1 - PIT_EDGE)


def apply_pit_map(*, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Evaluate each row's quantile function at the mapped levels `p` (shape (11,))."""
    levels, k = pit_knots(q=q)
    k = np.maximum.accumulate(k, axis=1)
    idx = np.clip(np.searchsorted(levels, p, side="right") - 1, 0, len(levels) - 2)
    w = (p - levels[idx]) / (levels[idx + 1] - levels[idx])
    return np.sort(k[:, idx] * (1 - w) + k[:, idx + 1] * w, axis=1)


def add_upper_adjustment(*, q: np.ndarray, adjustment: np.ndarray) -> np.ndarray:
    """Add per-row adjustments (shape (rows, 5)) to the quantiles above q50 and sort."""
    out = q.copy()
    out[:, UPPER] += adjustment
    return np.sort(out, axis=1)
