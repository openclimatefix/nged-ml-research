"""Slow level of a Primary's residual: three arms (masked frozen seasonal, Huber, random walk)."""

from __future__ import annotations

from typing import Final

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from e00653_common import BLOCK, DAY, N_BLOCKS

KNOT_SPACING: Final[int] = 365 * 4
"""Blocks between spline knots (yearly)."""
HARMONICS: Final[int] = 3
MASK_UNITS: Final[float] = 1.5
MASK_WIDEN: Final[int] = 8
"""Blocks (2 days) added each side of a flagged window."""
RW_LAMBDA: Final[float] = (120 / (2 * np.pi)) ** 4
"""Penalty on the squared second difference; passes periods longer than about 30 days."""
HUBER_K: Final[float] = 1.345


def design(*, n: int, valid: int) -> np.ndarray:
    """Seasonal harmonics plus a yearly linear spline; short histories get one harmonic and knot."""
    t = np.arange(n)
    phase = 2 * np.pi * t / (365.25 * 4)
    short = valid < 3 * KNOT_SPACING
    k = 1 if short else HARMONICS
    cols = [f(j * phase) for j in range(1, k + 1) for f in (np.sin, np.cos)]
    knots = np.array([0, n - 1]) if short else np.arange(0, n + KNOT_SPACING, KNOT_SPACING)
    for i, kn in enumerate(knots):
        width = knots[1] - knots[0] if i == 0 else kn - knots[i - 1]
        hat = np.zeros(n)
        left = knots[i - 1] if i > 0 else kn - width
        right = knots[i + 1] if i + 1 < len(knots) else kn + width
        up = (t - left) / (kn - left)
        down = (right - t) / (right - kn)
        hat = np.clip(np.minimum(up, down), 0, None)
        cols.append(hat)
    return np.stack(cols, axis=1)


def _huber_fit(*, y: np.ndarray, x: np.ndarray, weight: np.ndarray, scale: float) -> np.ndarray:
    ok = np.isfinite(y) & (weight > 0)
    beta = np.zeros(x.shape[1])
    w = weight.copy()
    for _ in range(8):
        a = x[ok] * w[ok, None]
        b = y[ok] * w[ok]
        beta = np.linalg.lstsq(a.T @ a + 1e-6 * np.eye(x.shape[1]), a.T @ b, rcond=None)[0]
        r = (y - x @ beta) / scale
        w = weight * np.sqrt(
            np.where(np.abs(r) <= HUBER_K, 1.0, HUBER_K / np.maximum(np.abs(r), 1e-9))
        )
        w[~np.isfinite(y)] = 0
    return x @ beta


def rolling_mean(*, r: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean over `window` blocks, needing at least half of them."""
    ok = np.isfinite(r)
    c = np.cumsum(np.where(ok, r, 0.0))
    n = np.cumsum(ok.astype(float))
    half = window // 2
    lo = np.clip(np.arange(len(r)) - half - 1, -1, None)
    hi = np.clip(np.arange(len(r)) + half, None, len(r) - 1)
    s = c[hi] - np.where(lo >= 0, c[np.maximum(lo, 0)], 0)
    k = n[hi] - np.where(lo >= 0, n[np.maximum(lo, 0)], 0)
    out = np.full(len(r), np.nan)
    good = k >= window // 2
    out[good] = s[good] / k[good]
    return out


def noise_unit(*, y: np.ndarray) -> float:
    """Robust spread of one block mean, from successive differences (halved by root two)."""
    d = np.diff(y)
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or 1e-3


def fit_slow(*, y: np.ndarray, sigma: float, arm: str) -> np.ndarray:
    """Return the slow level on the block grid for one Primary.

    Args:
        y: Block means of the residual (NaN where missing).
        sigma: Noise unit in megawatts.
        arm: `masked` (default), `huber`, or `rw30`.
    """
    n = len(y)
    valid = int(np.isfinite(y).sum())
    if arm == "rw30":
        w = np.isfinite(y).astype(float)
        d2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        a = sp.diags(w) + RW_LAMBDA * (d2.T @ d2)
        return spla.spsolve(a.tocsc(), np.where(np.isfinite(y), y, 0.0) * w)
    x = design(n=n, valid=valid)
    mask_w = np.ones(n)
    slow = _huber_fit(y=y, x=x, weight=mask_w, scale=sigma * 2)
    if arm == "huber":
        return slow
    for _ in range(3):
        rm = rolling_mean(r=y - slow, window=12)
        flag = np.abs(np.nan_to_num(rm)) > MASK_UNITS * sigma
        grown = flag.copy()
        for k in range(1, MASK_WIDEN + 1):
            grown[k:] |= flag[:-k]
            grown[:-k] |= flag[k:]
        mask_w = np.where(grown, 0.0, 1.0)
        slow = _huber_fit(y=y, x=x, weight=mask_w, scale=sigma * 2)
    return slow


def to_half_hours(*, slow: np.ndarray) -> np.ndarray:
    """Linear interpolation of a block-grid level to the half-hour grid."""
    centres = (np.arange(N_BLOCKS) + 0.5) * BLOCK
    return np.interp(np.arange(N_BLOCKS * BLOCK), centres, slow)


assert DAY == 48
