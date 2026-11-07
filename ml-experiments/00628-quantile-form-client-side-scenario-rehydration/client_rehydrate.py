# ruff: noqa: ANN001, ANN201, E501
"""The client's helper: draw coherent scenarios from stored quantiles (a sketch, not production code).

A client holds, for one forecast run, 11 stored quantiles per site and half-hour, and a small
dependence table: either a correlation factor and AR(1) coefficients (copula) or a set of past
observed paths (Schaake shuffle). `scenarios` turns them into K scenario paths.
"""

from typing import Final

import numpy as np
from scipy.special import ndtr, ndtri

LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
Z01: Final[float] = float(ndtri(0.01))
Z05: Final[float] = float(ndtri(0.05))


def invert(*, q: np.ndarray, u: np.ndarray, tail: str, bounds: tuple[float, float] | None):
    """Return the quantile function (q: (S, 11)) at levels u (K, S), with the named tail rule."""
    u = np.asarray(u, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    s = np.arange(q.shape[0])
    idx = np.clip(np.searchsorted(LEVELS, u, side="right") - 1, 0, len(LEVELS) - 2)
    frac = (u - LEVELS[idx]) / (LEVELS[idx + 1] - LEVELS[idx])
    out = q[s, idx] * (1 - frac) + q[s, idx + 1] * frac
    for side in (0, 1):
        a, b = (q[:, 0], q[:, 1]) if side == 0 else (q[:, 10], q[:, 9])
        if side == 0:
            edge, outside, lv = a, u < LEVELS[0], 0.01
            slope = (b - a) / (LEVELS[1] - LEVELS[0])
        else:
            edge, outside, lv = a, u > LEVELS[-1], 0.99
            slope = (a - b) / (LEVELS[-1] - LEVELS[-2])
        if tail == "clamp":
            tail_value = np.broadcast_to(edge, u.shape)
        elif tail == "linear":
            tail_value = edge + (u - lv) * slope
        elif tail == "gaussian":
            sigma = np.abs(slope) * (LEVELS[1] - LEVELS[0]) / (Z05 - Z01)
            z = ndtri(np.clip(u, 1e-9, 1 - 1e-9))
            tail_value = edge + sigma * (z - ndtri(lv))
        else:
            msg = f"unknown tail rule {tail}"
            raise ValueError(msg)
        out = np.where(outside, tail_value, out)
    if bounds is not None:
        out = np.clip(out, *bounds)
    return out


def copula_uniforms(*, chol: np.ndarray, phi: np.ndarray, k: int, steps: int, rng) -> np.ndarray:
    """Return uniform levels (k, nodes, steps) from the stored correlation factor and AR(1) terms."""
    n = chol.shape[0]
    eps = rng.standard_normal((k, n, steps)).astype(np.float32)
    innov = np.einsum("ij,mjs->mis", chol, eps) * np.sqrt(1 - phi**2)[None, :, None]
    c = np.empty_like(innov)
    c[:, :, 0] = np.einsum("ij,mj->mi", chol, eps[:, :, 0])
    for t in range(1, steps):
        c[:, :, t] = phi[None, :] * c[:, :, t - 1] + innov[:, :, t]
    return ndtr(c)


def schaake_uniforms(*, history: np.ndarray, k: int, rng) -> np.ndarray:
    """Return uniform levels (k, nodes, steps) from past paths (n, nodes, steps), jittered in rank."""
    n = history.shape[0]
    tie_break = rng.uniform(size=history.shape) * 1e-9  # ties (such as solar zeros) rank at random
    ranks = np.argsort(np.argsort(history + tie_break, axis=0), axis=0)
    reps = -(-k // n)
    jitter = rng.uniform(size=(reps, *history.shape))
    u = ((ranks[None] + jitter) / n).reshape(reps * n, *history.shape[1:])
    return u[:k]


def scenarios(*, q: np.ndarray, u: np.ndarray, tail: str, bounds: np.ndarray) -> np.ndarray:
    """Return scenarios (k, nodes, steps): q (nodes, steps, 11), u (k, nodes, steps)."""
    out = np.empty(u.shape, dtype=np.float32)
    for node in range(q.shape[0]):
        b = None if np.isnan(bounds[node, 0]) else (bounds[node, 0], bounds[node, 1])
        out[:, node] = invert(q=q[node], u=u[:, node], tail=tail, bounds=b)
    return out
