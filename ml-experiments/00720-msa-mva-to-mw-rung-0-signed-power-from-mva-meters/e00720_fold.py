# ruff: noqa: E501
"""Experiment 720, rung 0c: the folded likelihood of a Gaussian mixture, in plain NumPy.

The meter of an MVA Primary reads `m = sqrt(X² + Q²)`, where `X` is the signed real power and `Q` is
a constant reactive floor, all in points of capacity (or in any one unit, as long as `Q` shares it).
Away from zero `m = |X|`. The head of a forecasting network would give the density of `X` as a
mixture of Gaussians, and the likelihood of a reading is then the density of the folded variable
`|X|`. This module holds that density, its distribution function, its quantiles, the censored branch
for a reading at or below `Q`, and the conversion of a mean from standardised units to units that
fold about zero megawatts. No neural network is involved.
"""

from collections.abc import Callable
from typing import Final

import numpy as np
from scipy.special import logsumexp, ndtr

TAUS_11: Final[tuple[float, ...]] = (
    0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99,
)  # fmt: skip
"""The 11 quantile levels of the demand protocol's forecast."""
HISTORY_CLIP_Z: Final[float] = 5.0
"""The clip that experiment 713 applies to history power, in standardised units."""
LOG_SQRT_2PI: Final[float] = 0.5 * float(np.log(2.0 * np.pi))


def log_normal_pdf(*, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the log density of `N(mu, sigma²)` at `x`, element by element."""
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma) - LOG_SQRT_2PI


def reading_to_magnitude(*, m: np.ndarray, q: float) -> np.ndarray:
    """Return `y = sqrt(max(m² - Q², 0))`, the reading with the reactive floor removed."""
    return np.sqrt(np.maximum(m**2 - q**2, 0.0))


def magnitude_to_reading(*, y: np.ndarray, q: float) -> np.ndarray:
    """Return the reading `sqrt(y² + Q²)` of a magnitude `y` (a folded quantile goes through this)."""
    return np.sqrt(y**2 + q**2)


def folded_nll(
    *,
    m: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    q: float,
) -> np.ndarray:
    """Return the negative log-likelihood of each reading under the folded mixture.

    For a reading with `m > Q` the loss is `-logsumexp` over the components `k` and the two signs of
    `log pi_k + log N(y; ±mu_k, sigma_k²)`, with `y = sqrt(m² - Q²)`. The Jacobian `m / y` of the
    floor does not depend on the parameters and is dropped. For a reading with `m <= Q` the loss is
    `-log P(|X| <= Q)`, the censored branch.

    Args:
        m: Readings, shape `(N,)`.
        pi: Mixture weights, shape `(N, K)` or `(K,)`, summing to 1 over `K`.
        mu: Component means in the units of `m`, same shape rule as `pi`.
        sigma: Component standard deviations, same shape rule as `pi`.
        q: The reactive floor, in the units of `m`.

    Returns:
        The loss of each reading, shape `(N,)`.
    """
    pi, mu, sigma = (np.broadcast_to(a, (len(m), np.shape(a)[-1])) for a in (pi, mu, sigma))
    y = reading_to_magnitude(m=m, q=q)[:, None]
    log_pi = np.log(pi)
    both = np.concatenate(
        [
            log_pi + log_normal_pdf(x=y, mu=mu, sigma=sigma),
            log_pi + log_normal_pdf(x=y, mu=-mu, sigma=sigma),
        ],
        axis=1,
    )
    density = -logsumexp(both, axis=1)
    mass = censored_log_mass(q=q, pi=pi, mu=mu, sigma=sigma)
    return np.where(m <= q, -mass, density)


def censored_log_mass(*, q: float, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return `log P(|X| <= Q)` under the mixture, shape `(N,)` for `(N, K)` parameters."""
    upper = ndtr((q - mu) / sigma)
    lower = ndtr((-q - mu) / sigma)
    inside = np.maximum(upper - lower, 1e-300)
    return logsumexp(np.log(pi) + np.log(inside), axis=1)


def folded_cdf(*, y: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return `F_|X|(y) = sum_k pi_k [Phi((y - mu_k)/sigma_k) + Phi((y + mu_k)/sigma_k) - 1]`.

    `y` has shape `(N,)` and the mixture parameters have shape `(N, K)`. Values of `y` below zero
    give zero.
    """
    yy = y[:, None]
    value = np.sum(pi * (ndtr((yy - mu) / sigma) + ndtr((yy + mu) / sigma) - 1.0), axis=1)
    return np.where(y < 0.0, 0.0, value)


def unfolded_cdf(*, x: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the distribution function of the signed mixture at `x`, shape `(N,)`."""
    return np.sum(pi * ndtr((x[:, None] - mu) / sigma), axis=1)


def _bisect(
    *,
    cdf: Callable[[np.ndarray], np.ndarray],
    tau: float,
    lo: np.ndarray,
    hi: np.ndarray,
    n_iter: int = 80,
) -> np.ndarray:
    """Return the quantile at level `tau` of a monotone `cdf` by bisection on `[lo, hi]`."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < tau
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _span(*, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Return a bound `B` such that every quantile in 0.0005 to 0.9995 lies inside `[-B, B]`."""
    return float(np.max(np.abs(mu) + 8.0 * sigma)) + 1.0


def folded_quantiles(
    *, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, taus: tuple[float, ...] = TAUS_11
) -> np.ndarray:
    """Return the folded quantiles of the mixture, shape `(N, len(taus))`, by bisection.

    A folded quantile is not the absolute value of a signed quantile: the folded distribution
    function adds the mass at `-y` to the mass at `y`.
    """
    n = pi.shape[0]
    bound = _span(pi=pi, mu=mu, sigma=sigma)
    out = np.empty((n, len(taus)))
    for k, tau in enumerate(taus):
        out[:, k] = _bisect(
            cdf=lambda y: folded_cdf(y=y, pi=pi, mu=mu, sigma=sigma),
            tau=tau,
            lo=np.zeros(n),
            hi=np.full(n, bound),
        )
    return out


def signed_quantiles(
    *, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, taus: tuple[float, ...] = TAUS_11
) -> np.ndarray:
    """Return the quantiles of the unfolded (signed) mixture, shape `(N, len(taus))`."""
    n = pi.shape[0]
    bound = _span(pi=pi, mu=mu, sigma=sigma)
    out = np.empty((n, len(taus)))
    for k, tau in enumerate(taus):
        out[:, k] = _bisect(
            cdf=lambda x: unfolded_cdf(x=x, pi=pi, mu=mu, sigma=sigma),
            tau=tau,
            lo=np.full(n, -bound),
            hi=np.full(n, bound),
        )
    return out


def p_negative(*, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return `P(X < 0) = sum_k pi_k Phi(-mu_k / sigma_k)`, shape `(N,)`."""
    return np.sum(pi * ndtr(-mu / sigma), axis=1)


def fold_point_z(*, target_mean: float, target_std: float) -> float:
    """Return where zero megawatts sits in standardised units, `-mean/std`.

    Experiment 713 standardises power as `z = (m / capacity - mean) / std`, so a loss that folds about
    `z = 0` folds about the mean power and not about zero megawatts.
    """
    return -target_mean / target_std


def history_z_to_v(*, z: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Return history power in `v` units, which fold about zero megawatts.

    The history clip of ±5 is applied in `z` first, as experiment 713 does, and `mean/std` is then
    added, so `v = m / (capacity × std)` up to the clip and the fold point is `v = 0`.
    """
    clipped = np.clip(z, -HISTORY_CLIP_Z, HISTORY_CLIP_Z)
    return clipped + target_mean / target_std


def mean_to_v(*, mu_z: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Add `mean/std` to the head's means (in `z` units) so they fold about zero megawatts."""
    return mu_z + target_mean / target_std


def sample_mixture(
    *, rng: np.random.Generator, n: int, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Draw `n` values of the signed mixture with 1-D parameters of length `K`."""
    comp = rng.choice(len(pi), size=n, p=pi)
    return rng.normal(mu[comp], sigma[comp])
