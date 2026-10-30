"""Copy of experiment 400's `e00400_mixture.py`: mixture of two quantile forecasts: the CDF algebra, the exact CRPS, and the 11 quantiles.

Each forecast is a cumulative distribution function (CDF) that is linear between its 11 stored
quantiles (levels 0.01 to 0.99). The two end segments are extended to levels 0 and 1 at the slope of
the outer segment, so each forecast has 13 knots. The mixture CDF is `w * F1 + (1 - w) * F2`.
`mixture_crps` integrates `(F - H)^2` exactly (`H` is the step at the actual value), and
`mixture_quantiles` inverts the mixture CDF at the 11 levels. All values share one unit, such as
points of effective capacity.
"""

from typing import Final

import numpy as np

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KNOT_LEVELS: Final[np.ndarray] = np.concatenate([[0.0], LEVELS, [1.0]])


def knots(*, quantiles: np.ndarray) -> np.ndarray:
    """Return the 13 sorted knots of each row: the 11 quantiles and the two extended ends."""
    q = np.sort(quantiles.astype(np.float64), axis=1)
    low = q[:, 0] - (q[:, 1] - q[:, 0]) / (LEVELS[1] - LEVELS[0]) * LEVELS[0]
    high = q[:, -1] + (q[:, -1] - q[:, -2]) / (LEVELS[-1] - LEVELS[-2]) * (1.0 - LEVELS[-1])
    return np.concatenate([low[:, None], q, high[:, None]], axis=1)


def cdf_at(*, knot: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the CDF of each row's forecast at that row's points `x` (shape rows x points)."""
    count = (knot[:, None, :] <= x[:, :, None]).sum(axis=-1)
    k = np.clip(count - 1, 0, len(KNOT_LEVELS) - 2)
    x0 = np.take_along_axis(knot, k, axis=1)
    x1 = np.take_along_axis(knot, k + 1, axis=1)
    p0 = KNOT_LEVELS[k]
    p1 = KNOT_LEVELS[k + 1]
    dx = x1 - x0
    frac = np.where(dx > 0, (x - x0) / np.where(dx > 0, dx, 1.0), 1.0)
    out = p0 + np.clip(frac, 0.0, 1.0) * (p1 - p0)
    out = np.where(count == 0, 0.0, out)
    return np.where(x >= knot[:, -1:], 1.0, out)


class Pair:
    """Two forecasts' CDFs on the union of their knots (and optionally the actual value)."""

    def __init__(  # noqa: D107
        self, *, q1: np.ndarray, q2: np.ndarray, actual: np.ndarray | None = None
    ) -> None:
        k1, k2 = knots(quantiles=q1), knots(quantiles=q2)
        parts = [k1, k2] if actual is None else [k1, k2, actual.astype(np.float64)[:, None]]
        self.x = np.sort(np.concatenate(parts, axis=1), axis=1)
        self.f1 = cdf_at(knot=k1, x=self.x)
        self.f2 = cdf_at(knot=k2, x=self.x)
        self.actual = actual

    def crps(self, *, w: np.ndarray) -> np.ndarray:
        """Return the exact CRPS of the mixture with weight `w` on the first forecast."""
        if self.actual is None:
            msg = "the pair was built without actual values"
            raise ValueError(msg)
        f = w[:, None] * self.f1 + (1.0 - w[:, None]) * self.f2
        h = (self.x >= self.actual[:, None]).astype(np.float64)
        a = f[:, :-1] - h[:, :-1]
        b = f[:, 1:] - h[:, :-1]
        return ((self.x[:, 1:] - self.x[:, :-1]) * (a * a + a * b + b * b) / 3.0).sum(axis=1)

    def quantiles(self, *, w: np.ndarray) -> np.ndarray:
        """Return the mixture's 11 quantiles by inverting its CDF."""
        f = w[:, None] * self.f1 + (1.0 - w[:, None]) * self.f2
        out = np.empty((len(w), len(LEVELS)))
        last = f.shape[1] - 1
        for i, p in enumerate(LEVELS):
            j = np.clip((f < p).sum(axis=1), 1, last)
            f0 = np.take_along_axis(f, (j - 1)[:, None], axis=1)[:, 0]
            f1 = np.take_along_axis(f, j[:, None], axis=1)[:, 0]
            x0 = np.take_along_axis(self.x, (j - 1)[:, None], axis=1)[:, 0]
            x1 = np.take_along_axis(self.x, j[:, None], axis=1)[:, 0]
            df = f1 - f0
            out[:, i] = np.where(df > 0, x0 + (p - f0) / np.where(df > 0, df, 1.0) * (x1 - x0), x1)
        return out
