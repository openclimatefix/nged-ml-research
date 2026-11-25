"""Experiment 729: the exact CRPS of the two-lobe mixture, mixture quantiles, and sign metrics.

All arrays are in points of D8's capacity. A two-lobe distribution is the pair of Gaussians
`(1 - p) N(mu_i, s_i)` and `p N(mu_r, s_r)`. The signed CRPS of a Gaussian mixture is exact
([Grimit et al. (2006)](https://doi.org/10.1256/qj.05.235)):
`CRPS = sum_i w_i A(y - mu_i, s_i) - 0.5 sum_ij w_i w_j A(mu_i - mu_j, sqrt(s_i^2 + s_j^2))`, with
`A(m, s) = m (2 Phi(m / s) - 1) + 2 s phi(m / s)`.
"""

from dataclasses import dataclass
from math import sqrt

import e00729_core as core
import numpy as np
from scipy import special

SQRT2: float = sqrt(2.0)
SQRT_2PI: float = sqrt(2.0 * np.pi)
POINT_SPREAD: float = 1e-6
"""The spread that stands in for a point distribution (P0)."""


def norm_cdf(*, z: np.ndarray) -> np.ndarray:
    """Return the standard normal CDF."""
    return 0.5 * special.erfc(-z / SQRT2)


def norm_pdf(*, z: np.ndarray) -> np.ndarray:
    """Return the standard normal density."""
    return np.exp(-0.5 * z * z) / SQRT_2PI


def abs_expectation(*, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Return `E|Z|` for `Z ~ N(m, s^2)`: `m (2 Phi(m / s) - 1) + 2 s phi(m / s)`."""
    z = m / s
    return m * (2.0 * norm_cdf(z=z) - 1.0) + 2.0 * s * norm_pdf(z=z)


@dataclass(frozen=True)
class Lobes:
    """A two-lobe distribution for each of `N` half-hours.

    Attributes:
        p: Probability of the reverse-flow lobe, `(N,)`.
        mu_i: Mean of the import lobe (positive), `(N,)`.
        s_i: Spread of the import lobe, `(N,)`.
        mu_r: Mean of the reverse-flow lobe (negative), `(N,)`.
        s_r: Spread of the reverse-flow lobe, `(N,)`.
    """

    p: np.ndarray
    mu_i: np.ndarray
    s_i: np.ndarray
    mu_r: np.ndarray
    s_r: np.ndarray

    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return weights, means, and spreads as `(N, 2)` arrays (import first)."""
        w = np.stack([1.0 - self.p, self.p], axis=1)
        mu = np.stack([self.mu_i, self.mu_r], axis=1)
        s = np.stack([self.s_i, self.s_r], axis=1)
        return w, mu, s

    def subset(self, *, rows: np.ndarray) -> "Lobes":
        """Return the distributions of the selected rows."""
        return Lobes(self.p[rows], self.mu_i[rows], self.s_i[rows], self.mu_r[rows], self.s_r[rows])


def mixture_crps(*, lobes: Lobes, y: np.ndarray) -> np.ndarray:
    """Return the exact CRPS of the two-lobe mixture at each observation, in points.

    Args:
        lobes: The distributions.
        y: The observed signed power, `(N,)`.

    Returns:
        The CRPS of each half-hour, `(N,)`.
    """
    w, mu, s = lobes.components()
    first = (w * abs_expectation(m=y[:, None] - mu, s=s)).sum(axis=1)
    second = np.zeros_like(first)
    for i in range(2):
        for j in range(2):
            second += (
                w[:, i]
                * w[:, j]
                * abs_expectation(m=mu[:, i] - mu[:, j], s=np.sqrt(s[:, i] ** 2 + s[:, j] ** 2))
            )
    return first - 0.5 * second


def mixture_cdf(*, lobes: Lobes, z: np.ndarray) -> np.ndarray:
    """Return the CDF of each half-hour's mixture at `z` (a scalar or an `(N,)` array)."""
    w, mu, s = lobes.components()
    zz = np.broadcast_to(np.asarray(z, dtype=np.float64), lobes.p.shape)[:, None]
    return (w * norm_cdf(z=(zz - mu) / s)).sum(axis=1)


def mixture_quantiles(
    *, lobes: Lobes, levels: tuple[float, ...] = core.QUANTILE_LEVELS
) -> np.ndarray:
    """Return the quantiles `(N, len(levels))` of each mixture, by bisection.

    Args:
        lobes: The distributions.
        levels: The probability levels.

    Returns:
        The quantiles in points.
    """
    w, mu, s = lobes.components()
    lo0 = (mu - 10.0 * s).min(axis=1)
    hi0 = (mu + 10.0 * s).max(axis=1)
    out = np.empty((len(lobes.p), len(levels)))
    for k, tau in enumerate(levels):
        lo, hi = lo0.copy(), hi0.copy()
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = mixture_cdf(lobes=lobes, z=mid) < tau
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[:, k] = 0.5 * (lo + hi)
    return out


def prob_negative(*, lobes: Lobes) -> np.ndarray:
    """Return P(X < 0): the mixture's mass below zero."""
    return mixture_cdf(lobes=lobes, z=np.zeros(len(lobes.p)))


def pinball_points(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points."""
    taus = np.asarray(core.QUANTILE_LEVELS)[None, :]
    err = y[:, None] - q
    return 2.0 * np.maximum(taus * err, (taus - 1.0) * err).mean(axis=1)


def quantile_crps(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the CRPS of the piecewise-linear distribution through the quantiles, per row.

    The CDF is 0 below the lowest quantile, jumps to 0.05 there, rises linearly through the
    quantiles, and jumps to 1 at the highest quantile.

    Args:
        q: Sorted quantiles `(N, 11)`.
        y: Observations `(N,)`.

    Returns:
        The CRPS, `(N,)`.
    """
    taus = np.asarray(core.QUANTILE_LEVELS)
    q = np.maximum.accumulate(q, axis=1)
    total = np.where(y < q[:, 0], q[:, 0] - y, 0.0) + np.where(y > q[:, -1], y - q[:, -1], 0.0)
    for k in range(q.shape[1] - 1):
        a, b = q[:, k], q[:, k + 1]
        fa, fb = taus[k], taus[k + 1]
        width = np.maximum(b - a, 1e-12)
        inside = (y > a) & (y < b)
        f_y = fa + (fb - fa) * (np.clip(y, a, b) - a) / width
        c_whole = (y <= a).astype(float)  # indicator over the whole segment when y is outside
        whole = (
            (b - a)
            * ((fa - c_whole) ** 2 + (fa - c_whole) * (fb - c_whole) + (fb - c_whole) ** 2)
            / 3.0
        )
        left = (y - a) * ((fa - 0.0) ** 2 + (fa - 0.0) * (f_y - 0.0) + (f_y - 0.0) ** 2) / 3.0
        right = (b - y) * ((f_y - 1.0) ** 2 + (f_y - 1.0) * (fb - 1.0) + (fb - 1.0) ** 2) / 3.0
        total += np.where(inside, left + right, whole)
    return total


def reliability_term(
    *, prob: np.ndarray, observed: np.ndarray, n_bins: int = core.RELIABILITY_BINS
) -> dict[str, float | list[float]]:
    """Return the reliability term of the Brier score of a probability, and its bins.

    The reliability term is the count-weighted mean squared gap between a bin's mean forecast and
    its observed frequency.

    Args:
        prob: Forecast probabilities.
        observed: Whether the event happened.
        n_bins: The number of equal-width probability bins.

    Returns:
        `reliability`, the largest gap of a bin with at least `BIN_MIN_COUNT` rows (`max_gap`), and
        per-bin `count`, `mean_forecast`, and `frequency` lists.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    count, mean_f, freq = [], [], []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        count.append(n)
        mean_f.append(float(prob[sel].mean()) if n else float("nan"))
        freq.append(float(observed[sel].mean()) if n else float("nan"))
    n_all = max(sum(count), 1)
    gaps = [(f - m) ** 2 for f, m, c in zip(freq, mean_f, count, strict=True) if c]
    weights = [c / n_all for c in count if c]
    big = [
        abs(f - m) for f, m, c in zip(freq, mean_f, count, strict=True) if c >= core.BIN_MIN_COUNT
    ]
    return {
        "reliability": float(np.dot(gaps, weights)) if gaps else float("nan"),
        "max_gap": max(big) if big else 0.0,
        "count": [float(c) for c in count],
        "mean_forecast": mean_f,
        "frequency": freq,
    }


def lobe_covers(*, lobes: Lobes, y: np.ndarray) -> np.ndarray:
    """Return whether the 5% to 95% range of the lobe on the true side covers the truth."""
    z = special.ndtri(core.LOBE_LEVELS[1])
    rev = y < 0
    mu = np.where(rev, lobes.mu_r, lobes.mu_i)
    s = np.where(rev, lobes.s_r, lobes.s_i)
    return np.abs(y - mu) <= z * s


def lobe_width(*, lobes: Lobes, y: np.ndarray) -> np.ndarray:
    """Return the width of the 5% to 95% range of the lobe on the true side."""
    z = special.ndtri(core.LOBE_LEVELS[1])
    return 2.0 * z * np.where(y < 0, lobes.s_r, lobes.s_i)
