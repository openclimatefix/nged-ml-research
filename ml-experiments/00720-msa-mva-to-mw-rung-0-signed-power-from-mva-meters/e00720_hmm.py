# ruff: noqa: E501
"""Experiment 720, rung 0e: the hindsight switching model, a hidden Markov model with a gridded residual.

A half-hour `t` of one Primary has signed power `X_t = D_t − c p_t − n_t g`, where `D_t = f_t + r_t`
is gross demand (a regression mean `f_t` plus a residual `r_t`), `p_t` is the estimate of unmetered
solar output, `c` its scale, `n_t` the number of generator units running (0 to `K`), and `g` the size
of one unit. All powers are in points of capacity. The meter reads the half-hour mean of
`sqrt(P'² + Q²)` plus Student-t noise (4 degrees of freedom), and `P'` runs linearly inside the
half-hour from the midpoint of the previous and current signed power to the current signed power, so
a crossing half-hour reads above its mean magnitude. The emission therefore depends on the pair of
states at `t − 1` and `t`, and the forward and backward passes are exact for the discretised model
at fixed parameters. The residual is a first-order autoregression held on a grid.

State index: `s = n × G + b` for unit count `n` and residual bin `b`. Nothing here reads real data.
The inferred unit count and the inferred generator output are never plotted or tabulated by any
script of this experiment, and they are written only under the private data folder.
"""

import time
from dataclasses import dataclass, field
from typing import Final

import e00720_fold as fold
import numpy as np
from scipy.special import gammaln

N_SUBSTEPS_HMM: Final[int] = 4
"""Sub-steps of the within-half-hour path in the pair emission (the injected readings use 6)."""
STUDENT_DF: Final[float] = 4.0


@dataclass
class HmmSpec:
    """The parameters of the model at one era of one Primary.

    Attributes:
        grid: Residual bins in points, shape `(G,)`, evenly spaced.
        phi: Autoregression coefficient of the residual (bounded at 0.98 or a finer grid).
        sigma_r: Innovation standard deviation of the residual, in points.
        unit_size: `g`, the size of one generator unit in points (zero when `n_max` is 0).
        n_max: `K`, the largest unit count.
        trans_n: Row-stochastic unit-count transition matrix, `(K + 1, K + 1)`.
        pv_scale: `c`, the scale on the PV estimate.
        q: The reactive floor in points.
        sigma_e: Scale of the Student-t noise, in points.
        df: Degrees of freedom of the noise (0 or less gives a Gaussian, a check of rung 0e).
    """

    grid: np.ndarray
    phi: float
    sigma_r: float
    unit_size: float
    n_max: int
    trans_n: np.ndarray
    pv_scale: float = 1.0
    q: float = 2.0
    sigma_e: float = 0.5
    df: float = STUDENT_DF
    transition_cache: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_bins(self) -> int:
        """Return the number of residual bins `G`."""
        return len(self.grid)

    @property
    def n_states(self) -> int:
        """Return the number of states `(K + 1) × G`."""
        return (self.n_max + 1) * self.n_bins

    @property
    def spacing(self) -> float:
        """Return the grid spacing `d`."""
        return float(self.grid[1] - self.grid[0])

    @property
    def sigma_total(self) -> float:
        """Return the emission scale including the quantisation term `d²/12`."""
        return float(np.sqrt(self.sigma_e**2 + self.spacing**2 / 12.0))


def residual_grid(*, phi: float, sigma_r: float, n_bins: int = 41, span: float = 4.0) -> np.ndarray:
    """Return an even grid spanning `±span` stationary standard deviations of the residual."""
    sd = sigma_r / np.sqrt(1.0 - phi**2)
    return np.linspace(-span * sd, span * sd, n_bins)


def residual_transition(*, spec: HmmSpec) -> np.ndarray:
    """Return the AR(1) transition between grid bins, rows summing to 1."""
    mean = spec.phi * spec.grid[:, None]
    k = np.exp(-0.5 * ((spec.grid[None, :] - mean) / spec.sigma_r) ** 2)
    return k / k.sum(axis=1, keepdims=True)


def full_transition(*, spec: HmmSpec) -> np.ndarray:
    """Return the `(S, S)` transition, the Kronecker product of unit counts and residual bins."""
    return np.kron(spec.trans_n, residual_transition(spec=spec))


def stationary(*, trans: np.ndarray) -> np.ndarray:
    """Return the stationary distribution of a row-stochastic matrix."""
    k = trans.shape[0]
    a = trans.T - np.eye(k)
    a[-1, :] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    s = np.maximum(np.linalg.solve(a, rhs), 0.0)
    return s / s.sum()


def state_power(*, spec: HmmSpec, f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return the signed power of every state at every time, shape `(T, S)`."""
    n = np.repeat(np.arange(spec.n_max + 1), spec.n_bins)
    r = np.tile(spec.grid, spec.n_max + 1)
    return f[:, None] + r[None, :] - spec.pv_scale * p[:, None] - spec.unit_size * n[None, :]


def path_mean_reading(*, a: np.ndarray, b: np.ndarray, q: float, n_sub: int) -> np.ndarray:
    """Return the mean of `sqrt(P'² + Q²)` on the path from `(a + b)/2` to `b`, broadcast over `a`, `b`."""
    start = 0.5 * (a + b)
    acc = np.zeros(np.broadcast(a, b).shape)
    for k in range(n_sub):
        frac = (k + 0.5) / n_sub
        acc += np.sqrt((start + (b - start) * frac) ** 2 + q**2)
    return acc / n_sub


def log_noise(*, resid: np.ndarray, scale: float, df: float) -> np.ndarray:
    """Return the log density of Student-t (or Gaussian when `df <= 0`) noise."""
    z = resid / scale
    if df <= 0.0:
        return -0.5 * z**2 - np.log(scale) - fold.LOG_SQRT_2PI
    return (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
        - 0.5 * (df + 1.0) * np.log1p(z**2 / df)
    )


def log_emission_pair(
    *, spec: HmmSpec, m_t: float, x_prev: np.ndarray, x_now: np.ndarray, n_sub: int
) -> np.ndarray:
    """Return `log P(m_t | state_{t-1} = i, state_t = j)` as an `(S, S)` array."""
    mean = path_mean_reading(a=x_prev[:, None], b=x_now[None, :], q=spec.q, n_sub=n_sub)
    return log_noise(resid=m_t - mean, scale=spec.sigma_total, df=spec.df)


def log_emission_first(*, spec: HmmSpec, m_0: float, x_0: np.ndarray) -> np.ndarray:
    """Return `log P(m_0 | state_0 = j)`, a flat path (no previous half-hour)."""
    mean = np.sqrt(x_0**2 + spec.q**2)
    return log_noise(resid=m_0 - mean, scale=spec.sigma_total, df=spec.df)


@dataclass
class Posterior:
    """The result of one forward and backward pass.

    Attributes:
        log_likelihood: Log-likelihood of the readings at the fixed parameters.
        filtered: `P(state_t | readings up to t)`, `(T, S)` (the as-of version).
        smoothed: `P(state_t | all readings)`, `(T, S)` (the hindsight version).
        xi_n: Expected unit-count transitions summed over `t`, `(K + 1, K + 1)`, for Baum-Welch.
        x_state: Signed power per state and time, `(T, S)`.
    """

    log_likelihood: float
    filtered: np.ndarray
    smoothed: np.ndarray
    xi_n: np.ndarray
    x_state: np.ndarray


def forward_backward(
    *,
    spec: HmmSpec,
    m: np.ndarray,
    f: np.ndarray,
    p: np.ndarray,
    n_sub: int = N_SUBSTEPS_HMM,
    want_xi: bool = True,
) -> Posterior:
    """Run the scaled forward and backward passes with pair emissions.

    Args:
        spec: The model parameters.
        m: Readings, `(T,)`; NaN readings are treated as missing (the emission is flat).
        f: Regression mean of gross demand, `(T,)`.
        p: PV estimate, `(T,)`.
        n_sub: Sub-steps of the within-half-hour path.
        want_xi: Whether to accumulate the expected unit-count transitions.

    Returns:
        The `Posterior`.
    """
    n_t = len(m)
    s = spec.n_states
    trans = full_transition(spec=spec)
    x = state_power(spec=spec, f=f, p=p)
    pi0 = (
        stationary(trans=spec.trans_n)[:, None]
        * stationary(trans=residual_transition(spec=spec))[None, :]
    )
    alpha = np.zeros((n_t, s))
    scale = np.zeros(n_t)
    loglik = 0.0
    a0 = np.log(pi0.reshape(-1) + 1e-300)
    if np.isfinite(m[0]):
        a0 = a0 + log_emission_first(spec=spec, m_0=m[0], x_0=x[0])
    shift = a0.max()
    a = np.exp(a0 - shift)
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    loglik += np.log(scale[0]) + shift
    for t in range(1, n_t):
        if np.isfinite(m[t]):
            le = log_emission_pair(spec=spec, m_t=m[t], x_prev=x[t - 1], x_now=x[t], n_sub=n_sub)
            sh = le.max()
            e = np.exp(le - sh)
        else:
            sh, e = 0.0, np.ones((s, s))
        a = alpha[t - 1] @ (trans * e)
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
        loglik += np.log(scale[t]) + sh
    beta = np.ones(s)
    smoothed = np.zeros((n_t, s))
    smoothed[-1] = alpha[-1]
    k = spec.n_max + 1
    xi_n = np.zeros((k, k))
    for t in range(n_t - 1, 0, -1):
        if np.isfinite(m[t]):
            le = log_emission_pair(spec=spec, m_t=m[t], x_prev=x[t - 1], x_now=x[t], n_sub=n_sub)
            e = np.exp(le - le.max())
        else:
            e = np.ones((s, s))
        te = trans * e
        if want_xi:
            xi = alpha[t - 1][:, None] * te * beta[None, :]
            xi /= xi.sum()
            xi_n += xi.reshape(k, spec.n_bins, k, spec.n_bins).sum(axis=(1, 3))
        beta_new = te @ beta
        beta = beta_new / beta_new.sum()
        g = alpha[t - 1] * beta
        smoothed[t - 1] = g / g.sum()
    return Posterior(loglik, alpha, smoothed, xi_n, x)


def update_trans_n(*, xi_n: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Return the Baum-Welch update of the unit-count transition matrix."""
    t = xi_n + floor
    return t / t.sum(axis=1, keepdims=True)


def summarise(
    *, post: Posterior, spec: HmmSpec, version: str = "smoothed"
) -> dict[str, np.ndarray]:
    """Return per-half-hour summaries: `p_negative`, `p_units` `(T, K+1)`, and the signed mean."""
    gamma = post.smoothed if version == "smoothed" else post.filtered
    neg = (post.x_state < 0.0).astype(np.float64)
    p_neg = (gamma * neg).sum(axis=1)
    p_units = gamma.reshape(len(gamma), spec.n_max + 1, spec.n_bins).sum(axis=2)
    return {
        "p_negative": p_neg,
        "p_units": p_units,
        "signed_mean": (gamma * post.x_state).sum(axis=1),
    }


def mixture_p_negative(*, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return `P(X < 0)` of a two-component Gaussian mixture, `(T,)`."""
    from scipy.stats import norm

    return (pi * norm.cdf(-mu / sigma)).sum(axis=1)


def two_lobe_mixture(
    *, post: Posterior, spec: HmmSpec, version: str = "smoothed", extra_var: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the signed predictive distribution as a two-component mixture, `(pi, mu, sigma)`.

    Component 0 is the negative lobe and component 1 the positive lobe. Each lobe is moment-matched
    to the posterior mass on its side of zero, with variance `d²/12 + σ_e² + extra_var` added.
    Arrays have shape `(T, 2)`. An empty lobe gets weight `1e-12` and the other lobe's mean.

    The Gaussian lobes leak mass across zero (by up to 0.15 in a lobe close to zero), so the mixture's own
    `P(X < 0)` differs from the weight of the negative lobe. `signed_quantiles_of` therefore truncates each
    lobe at zero, which keeps `P(X < 0)` equal to the posterior's.
    """
    gamma = post.smoothed if version == "smoothed" else post.filtered
    x = post.x_state
    kernel = spec.spacing**2 / 12.0 + spec.sigma_e**2 + extra_var
    out_pi = np.zeros((len(gamma), 2))
    out_mu = np.zeros((len(gamma), 2))
    out_var = np.zeros((len(gamma), 2))
    for lobe, mask in enumerate((x < 0.0, x >= 0.0)):
        w = gamma * mask
        mass = w.sum(axis=1)
        safe = np.where(mass > 1e-12, mass, 1.0)
        mean = (w * x).sum(axis=1) / safe
        var = (w * (x - mean[:, None]) ** 2).sum(axis=1) / safe
        out_pi[:, lobe] = np.maximum(mass, 1e-12)
        out_mu[:, lobe] = mean
        out_var[:, lobe] = var + kernel
    out_pi /= out_pi.sum(axis=1, keepdims=True)
    empty = out_pi < 1e-9
    other = out_mu[:, ::-1]
    out_mu = np.where(empty, other, out_mu)
    sigma = np.sqrt(out_var)
    return out_pi, out_mu, sigma


def truncated_signed_quantiles(*, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the 11 signed quantiles of a two-lobe mixture whose lobes are truncated at zero, `(T, 11)`.

    Lobe 0 is the Gaussian `N(mu0, sigma0)` restricted to `x < 0`, and lobe 1 the Gaussian restricted to
    `x >= 0`, each renormalised, so `P(X < 0)` is exactly `pi[:, 0]`.
    """
    from scipy.stats import norm

    a0 = np.maximum(norm.cdf(-mu[:, 0] / sigma[:, 0]), 1e-12)
    b1 = norm.cdf(-mu[:, 1] / sigma[:, 1])
    c1 = np.maximum(1.0 - b1, 1e-12)

    def cdf(x: np.ndarray) -> np.ndarray:
        neg = np.where(x < 0.0, norm.cdf((x - mu[:, 0]) / sigma[:, 0]) / a0, 1.0)
        pos = np.where(x < 0.0, 0.0, (norm.cdf((x - mu[:, 1]) / sigma[:, 1]) - b1) / c1)
        return pi[:, 0] * np.minimum(neg, 1.0) + pi[:, 1] * np.clip(pos, 0.0, 1.0)

    n = pi.shape[0]
    bound = float(np.max(np.abs(mu) + 8.0 * sigma)) + 1.0
    out = np.empty((n, len(fold.TAUS_11)))
    for k, tau in enumerate(fold.TAUS_11):
        out[:, k] = fold._bisect(cdf=cdf, tau=tau, lo=np.full(n, -bound), hi=np.full(n, bound))
    return out


def signed_quantiles_of(*, post: Posterior, spec: HmmSpec, version: str = "smoothed") -> np.ndarray:
    """Return the 11 signed quantiles of the two-lobe distribution with lobes truncated at zero, `(T, 11)`."""
    pi, mu, sigma = two_lobe_mixture(post=post, spec=spec, version=version)
    return truncated_signed_quantiles(pi=pi, mu=mu, sigma=sigma)


def signed_crps(*, quantiles: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the CRPS of each row, twice the mean pinball loss over the 11 quantile levels."""
    taus = np.asarray(fold.TAUS_11)[None, :]
    diff = truth[:, None] - quantiles
    pin = np.maximum(taus * diff, (taus - 1.0) * diff)
    return 2.0 * pin.mean(axis=1)


def brier_reliability(
    *, prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Return the Brier score, its reliability term, and the largest bin gap among bins of 200 rows.

    The reliability term is `Σ_b n_b (mean prob_b − mean outcome_b)² / N`.
    """
    idx = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    n = len(prob)
    rel = 0.0
    worst = 0.0
    for b in range(n_bins):
        sel = idx == b
        nb = int(sel.sum())
        if nb == 0:
            continue
        gap = float(prob[sel].mean() - outcome[sel].mean())
        rel += nb * gap**2 / n
        if nb >= 200:
            worst = max(worst, abs(gap))
    return {
        "brier": float(np.mean((prob - outcome) ** 2)),
        "reliability": rel,
        "max_gap_bin_200": worst,
    }


@dataclass
class CostGate:
    """Stops a run when the projected core-hours exceed the cap.

    The first pass is timed with `measure`, after which `check` raises if the projected cost of
    `n_passes_total` such passes exceeds `cap_hours`. One job at a time on one BLAS thread means that
    wall-clock time equals core-hours.
    """

    cap_hours: float = 15.0
    seconds_per_pass: float = float("nan")

    def measure(self, *, run_one_pass) -> None:  # noqa: ANN001
        """Time one pass of the callable and record the seconds."""
        t0 = time.perf_counter()
        run_one_pass()
        self.seconds_per_pass = time.perf_counter() - t0

    def projected_hours(self, *, n_passes_total: int) -> float:
        """Return the projected core-hours of `n_passes_total` passes."""
        return self.seconds_per_pass * n_passes_total / 3600.0

    def check(self, *, n_passes_total: int) -> float:
        """Return the projected hours, or raise `RuntimeError` when they exceed the cap."""
        hours = self.projected_hours(n_passes_total=n_passes_total)
        if not hours <= self.cap_hours:
            msg = f"projected {hours:.2f} core-hours exceeds the cap of {self.cap_hours:.1f}; stop and report"
            raise RuntimeError(msg)
        return hours
