"""The four trackers of experiment 694: M0 pairing, M1 count HMM, M2 grid filter, M2-dwell.

Every tracker reads one segment of a reading `y` (MVA, NaN where dropped) and the quantised causal
spread index `lvl` for each half-hour. None reads a demand forecast, a price, or the truth.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from e00694_common import Q_FLOOR, SPREAD_LEVELS

GRID_STEP: Final[float] = 0.1
GRID_MAX: Final[float] = 22.0
LEVEL_GRID: Final[np.ndarray] = np.arange(0.0, GRID_MAX + GRID_STEP / 2, GRID_STEP)
N_GRID: Final[int] = len(LEVEL_GRID)
OBS_SCALE: Final[float] = 0.08
"""Scale of the Student-t noise of the reading around its mean, in MVA."""
U_GRID: Final[np.ndarray] = np.arange(0.5, 3.51, 0.25)
STAY_PRIOR: Final[float] = 0.995
PRIOR_WEIGHT: Final[float] = 5.0
EM_ROUNDS: Final[int] = 5
K_MAX: Final[int] = 3
OCCUPANCY_COST: Final[float] = 0.02
"""Log-prior cost per unit per half-hour of running. It anchors "off" as the state with no units,
because differences alone cannot tell k = 0 from k = 1 held for the whole segment."""


def t_pdf(z: np.ndarray) -> np.ndarray:
    """Density of a Student-t variable with 4 degrees of freedom."""
    return 0.375 * (1.0 + z * z / 4.0) ** -2.5


def _kernels() -> np.ndarray:
    """One row-normalised random-walk kernel on the level grid per spread level."""
    diff = LEVEL_GRID[None, :] - LEVEL_GRID[:, None]
    out = np.empty((len(SPREAD_LEVELS), N_GRID, N_GRID))
    for i, scale in enumerate(SPREAD_LEVELS):
        kernel = t_pdf(diff / scale)
        out[i] = kernel / kernel.sum(axis=1, keepdims=True)
    return out


KERNELS: Final[np.ndarray] = _kernels()


def prior_matrix(*, k_max: int) -> np.ndarray:
    """Prior transition matrix over 0 to k_max units: stay 0.995, move elsewhere equally."""
    n = k_max + 1
    if n == 1:
        return np.ones((1, 1))
    a = np.full((n, n), (1 - STAY_PRIOR) / (n - 1))
    np.fill_diagonal(a, STAY_PRIOR)
    return a


@dataclass
class TrackResult:
    """The output of one tracker on one segment."""

    k_max: int
    unit: float
    p_any_smooth: np.ndarray
    p_any_filt: np.ndarray
    k_smooth: np.ndarray
    log_evidence: float
    transition: np.ndarray


def _bic(*, log_evidence: float, n_obs: int, k_max: int) -> float:
    n_par = k_max * (k_max + 1) + (1 if k_max > 0 else 0)
    return -2.0 * log_evidence + n_par * np.log(max(n_obs, 2))


# ---------------------------------------------------------------- M2: grid filter


def _state_transition(*, a_k: np.ndarray, dwell: bool) -> np.ndarray:
    """Full state transition matrix; with dwell the state is (age, k), age 0 meaning just moved."""
    if not dwell:
        return a_k
    n = a_k.shape[0]
    a = np.zeros((2 * n, 2 * n))
    for k in range(n):
        a[k, n + k] = 1.0  # just moved (age 0) -> settled (age 1), same k
        for k2 in range(n):
            if k2 == k:
                a[n + k, n + k] = a_k[k, k]
            else:
                a[n + k, k2] = a_k[k, k2]
    return a


def _means(*, k_max: int, unit: float, fold: bool, dwell: bool) -> np.ndarray:
    """Mean reading per state and level-grid point, shape (S, G)."""
    ks = np.arange(k_max + 1)
    if dwell:
        ks = np.concatenate([ks, ks])
    real = LEVEL_GRID[None, :] - ks[:, None] * unit
    return np.sqrt(real**2 + Q_FLOOR**2) if fold else real


def occupancy(*, k_max: int, dwell: bool) -> np.ndarray:
    """Prior factor per state for the half-hour spent running k units, shape (S, 1)."""
    ks = np.arange(k_max + 1)
    if dwell:
        ks = np.concatenate([ks, ks])
    return np.exp(-OCCUPANCY_COST * ks)[:, None]


def _emission(*, y_t: float, means: np.ndarray, occ: np.ndarray) -> np.ndarray:
    if not np.isfinite(y_t):
        return np.ones_like(means) * occ
    return t_pdf((y_t - means) / OBS_SCALE) / OBS_SCALE * occ


def _grid_forward(
    *, y: np.ndarray, lvl: np.ndarray, means: np.ndarray, a: np.ndarray, keep: bool, occ: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Scaled forward pass; returns the log evidence, the filtered states, and the scales."""
    n_t, n_s = len(y), means.shape[0]
    alpha = np.empty((n_t if keep else 0, n_s, N_GRID))
    scales = np.empty(n_t)
    start = np.full((n_s, N_GRID), 1.0 / N_GRID)
    cur = start * e_prior(n_s=n_s)[:, None] * _emission(y_t=y[0], means=means, occ=occ)
    total = cur.sum()
    cur /= total
    scales[0] = total
    log_ev = np.log(total)
    if keep:
        alpha[0] = cur
    at = a.T
    for t in range(1, n_t):
        pred = (at @ cur) @ KERNELS[lvl[t]]
        cur = pred * _emission(y_t=y[t], means=means, occ=occ)
        total = cur.sum()
        cur /= total
        scales[t] = total
        log_ev += np.log(total)
        if keep:
            alpha[t] = cur
    return float(log_ev), alpha, scales


def e_prior(*, n_s: int) -> np.ndarray:
    """Prior over states at the first half-hour: mostly zero units, a little on every other."""
    p = np.full(n_s, 0.2 / max(n_s - 1, 1))
    p[0] = 0.8 if n_s > 1 else 1.0
    return p / p.sum()


def _grid_backward(
    *,
    y: np.ndarray,
    lvl: np.ndarray,
    means: np.ndarray,
    a: np.ndarray,
    alpha: np.ndarray,
    scales: np.ndarray,
    occ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed state marginals (T, S) and expected transition counts (S, S)."""
    n_t, n_s, _ = alpha.shape
    gamma = np.empty((n_t, n_s))
    xi_sum = np.zeros((n_s, n_s))
    beta = np.ones((n_s, N_GRID))
    gamma[-1] = (alpha[-1] * beta).sum(axis=1)
    for t in range(n_t - 1, 0, -1):
        kernel = KERNELS[lvl[t]]
        v = _emission(y_t=y[t], means=means, occ=occ) * beta
        xi_sum += a * (((alpha[t - 1] @ kernel) @ v.T) / scales[t])
        beta = (a @ (v @ kernel.T)) / scales[t]
        gamma[t - 1] = (alpha[t - 1] * beta).sum(axis=1)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi_sum


def _m_step(*, xi: np.ndarray, k_max: int, dwell: bool) -> np.ndarray:
    n = k_max + 1
    counts = np.zeros((n, n))
    if dwell:
        for k in range(n):
            for k2 in range(n):
                counts[k, k2] = xi[n + k, n + k] if k2 == k else xi[n + k, k2]
    else:
        counts = xi.copy()
    counts += PRIOR_WEIGHT * prior_matrix(k_max=k_max)
    return counts / counts.sum(axis=1, keepdims=True)


def _fold_states(*, gamma: np.ndarray, k_max: int) -> np.ndarray:
    """Marginal over k from the state marginals (sums the two ages when there are any)."""
    n = k_max + 1
    return gamma[:, :n] + gamma[:, n:] if gamma.shape[1] > n else gamma


def grid_evidence(
    *,
    y: np.ndarray,
    lvl: np.ndarray,
    k_max: int,
    unit: float,
    fold: bool,
    a_k: np.ndarray,
    use_occupancy: bool = True,
) -> float:
    """Log evidence of M2 for given K, u, and transition matrix.

    With `use_occupancy=False` the occupancy prior is left out, so the value is a predictive
    density.
    """
    means = _means(k_max=k_max, unit=unit, fold=fold, dwell=False)
    return _grid_forward(
        y=y,
        lvl=lvl,
        means=means,
        a=_state_transition(a_k=a_k, dwell=False),
        keep=False,
        occ=occupancy(k_max=k_max, dwell=False) if use_occupancy else np.ones((k_max + 1, 1)),
    )[0]


def _grid_em(
    *, y: np.ndarray, lvl: np.ndarray, k_max: int, unit: float, fold: bool, dwell: bool, rounds: int
) -> tuple[np.ndarray, float]:
    """Run `rounds` of expectation-maximisation; return the transition matrix and the evidence."""
    a_k = prior_matrix(k_max=k_max)
    means = _means(k_max=k_max, unit=unit, fold=fold, dwell=dwell)
    occ = occupancy(k_max=k_max, dwell=dwell)
    for _ in range(rounds):
        a = _state_transition(a_k=a_k, dwell=dwell)
        _, alpha, scales = _grid_forward(y=y, lvl=lvl, means=means, a=a, keep=True, occ=occ)
        _, xi = _grid_backward(y=y, lvl=lvl, means=means, a=a, alpha=alpha, scales=scales, occ=occ)
        a_k = _m_step(xi=xi, k_max=k_max, dwell=dwell)
    a = _state_transition(a_k=a_k, dwell=dwell)
    ev, _, _ = _grid_forward(y=y, lvl=lvl, means=means, a=a, keep=False, occ=occ)
    return a_k, ev


def select_grid(*, y: np.ndarray, lvl: np.ndarray, fold: bool) -> tuple[int, float]:
    """Choose K and u by BIC without truth.

    The unit size is profiled at K = 3 with the prior transition matrix. Then K = 0 to 3 are each
    refitted by 3 rounds of expectation-maximisation at that unit size and compared by BIC, so a
    model with many switches is not penalised for the prior's transition probabilities.
    """
    n_obs = int(np.isfinite(y).sum())
    a3 = prior_matrix(k_max=K_MAX)
    evs = [
        grid_evidence(y=y, lvl=lvl, k_max=K_MAX, unit=float(u), fold=fold, a_k=a3) for u in U_GRID
    ]
    best_u = float(U_GRID[int(np.argmax(evs))])
    scores = {}
    zero = _means(k_max=0, unit=best_u, fold=fold, dwell=False)
    ev0, _, _ = _grid_forward(
        y=y, lvl=lvl, means=zero, a=np.ones((1, 1)), keep=False, occ=occupancy(k_max=0, dwell=False)
    )
    scores[0] = _bic(log_evidence=ev0, n_obs=n_obs, k_max=0)
    for k_max in (1, 2, 3):
        _, ev = _grid_em(y=y, lvl=lvl, k_max=k_max, unit=best_u, fold=fold, dwell=False, rounds=3)
        scores[k_max] = _bic(log_evidence=ev, n_obs=n_obs, k_max=k_max)
    return min(scores, key=lambda k: scores[k]), best_u


def grid_track(
    *,
    y: np.ndarray,
    lvl: np.ndarray,
    fold: bool = True,
    dwell: bool = False,
    fixed: tuple[int, float] | None = None,
) -> TrackResult:
    """Fit K and u without truth (BIC) and track the number of units running with M2.

    With `fixed=(K, u)` the selection is skipped, so the dwell variant reuses its sibling's choice.
    """
    k_sel, u_sel = fixed if fixed is not None else select_grid(y=y, lvl=lvl, fold=fold)
    n_t = len(y)
    if k_sel == 0:
        zero = np.zeros(n_t)
        return TrackResult(0, u_sel, zero, zero, np.zeros(n_t, dtype=int), 0.0, np.ones((1, 1)))
    a_k, _ = _grid_em(
        y=y, lvl=lvl, k_max=k_sel, unit=u_sel, fold=fold, dwell=dwell, rounds=EM_ROUNDS
    )
    means = _means(k_max=k_sel, unit=u_sel, fold=fold, dwell=dwell)
    occ = occupancy(k_max=k_sel, dwell=dwell)
    a = _state_transition(a_k=a_k, dwell=dwell)
    ev, alpha, scales = _grid_forward(y=y, lvl=lvl, means=means, a=a, keep=True, occ=occ)
    gamma, _ = _grid_backward(y=y, lvl=lvl, means=means, a=a, alpha=alpha, scales=scales, occ=occ)
    smooth = _fold_states(gamma=gamma, k_max=k_sel)
    filt = _fold_states(gamma=alpha.sum(axis=2), k_max=k_sel)
    return TrackResult(
        k_max=k_sel,
        unit=u_sel,
        p_any_smooth=1.0 - smooth[:, 0],
        p_any_filt=1.0 - filt[:, 0],
        k_smooth=np.argmax(smooth, axis=1),
        log_evidence=ev,
        transition=a_k,
    )


# ---------------------------------------------------------------- M1: count HMM on differences


def _diff_emissions(*, d: np.ndarray, spread: np.ndarray, k_max: int, unit: float) -> np.ndarray:
    """Emission of each lag-1 difference for each (previous k, new k); shape (T, K+1, K+1)."""
    ks = np.arange(k_max + 1)
    shift = -(ks[None, :] - ks[:, None]) * unit  # mean difference when k moves from row to col
    z = (d[:, None, None] - shift[None]) / spread[:, None, None]
    e = t_pdf(z) / spread[:, None, None]
    e *= np.exp(-OCCUPANCY_COST * ks)[None, None, :]
    e[~np.isfinite(d)] = np.exp(-OCCUPANCY_COST * ks)[None, :]
    return e


def _count_forward(*, e: np.ndarray, a_k: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n_t, n, _ = e.shape
    alpha = np.empty((n_t, n))
    scales = np.ones(n_t)
    cur = e_prior(n_s=n)
    alpha[0] = cur
    log_ev = 0.0
    for t in range(1, n_t):
        pred = (cur[:, None] * a_k * e[t]).sum(axis=0)
        total = pred.sum()
        cur = pred / total
        alpha[t] = cur
        scales[t] = total
        log_ev += np.log(total)
    return float(log_ev), alpha, scales


def _count_em(
    *, e: np.ndarray, k_max: int, rounds: int, want_gamma: bool = False
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Expectation-maximisation for M1: matrix, evidence, smoothed states, filtered states."""
    n_t = e.shape[0]
    a_k = prior_matrix(k_max=k_max)
    gamma = alpha = np.zeros((n_t, k_max + 1))
    ev = 0.0
    for rnd in range(rounds + 1):
        ev, alpha, scales = _count_forward(e=e, a_k=a_k)
        if rnd == rounds and not want_gamma:
            break
        beta = np.ones(k_max + 1)
        gamma = np.empty_like(alpha)
        gamma[-1] = alpha[-1]
        xi_sum = np.zeros((k_max + 1, k_max + 1))
        for t in range(n_t - 1, 0, -1):
            w = a_k * e[t] * beta[None, :] / scales[t]
            xi_sum += alpha[t - 1][:, None] * w
            beta = w.sum(axis=1)
            g = alpha[t - 1] * beta
            gamma[t - 1] = g / g.sum()
        if rnd < rounds:
            counts = xi_sum + PRIOR_WEIGHT * prior_matrix(k_max=k_max)
            a_k = counts / counts.sum(axis=1, keepdims=True)
    return a_k, ev, gamma, alpha


def count_track(
    *, y: np.ndarray, lvl: np.ndarray, fixed: tuple[int, float] | None = None
) -> TrackResult:
    """Fit K and u (BIC, as for M2) and track the number of units with the count HMM M1."""
    spread = SPREAD_LEVELS[lvl]
    d = np.full(len(y), np.nan)
    d[1:] = np.diff(y)
    n_obs = int(np.isfinite(d).sum())
    n_t = len(y)
    if fixed is None:
        a3 = prior_matrix(k_max=K_MAX)
        evs = [
            _count_forward(
                e=_diff_emissions(d=d, spread=spread, k_max=K_MAX, unit=float(u)), a_k=a3
            )[0]
            for u in U_GRID
        ]
        u_sel = float(U_GRID[int(np.argmax(evs))])
        scores = {}
        e0 = _diff_emissions(d=d, spread=spread, k_max=0, unit=u_sel)
        scores[0] = _bic(
            log_evidence=_count_forward(e=e0, a_k=np.ones((1, 1)))[0], n_obs=n_obs, k_max=0
        )
        for k_max in (1, 2, 3):
            e = _diff_emissions(d=d, spread=spread, k_max=k_max, unit=u_sel)
            ev = _count_em(e=e, k_max=k_max, rounds=3)[1]
            scores[k_max] = _bic(log_evidence=ev, n_obs=n_obs, k_max=k_max)
        k_sel = min(scores, key=lambda k: scores[k])
    else:
        k_sel, u_sel = fixed
    if k_sel == 0:
        zero = np.zeros(n_t)
        return TrackResult(0, u_sel, zero, zero, np.zeros(n_t, dtype=int), 0.0, np.ones((1, 1)))
    e = _diff_emissions(d=d, spread=spread, k_max=k_sel, unit=u_sel)
    a_k, ev, gamma, alpha = _count_em(e=e, k_max=k_sel, rounds=EM_ROUNDS, want_gamma=True)
    return TrackResult(
        k_max=k_sel,
        unit=u_sel,
        p_any_smooth=1.0 - gamma[:, 0],
        p_any_filt=1.0 - alpha[:, 0],
        k_smooth=np.argmax(gamma, axis=1),
        log_evidence=ev,
        transition=a_k,
    )


# ---------------------------------------------------------------- M0: difference pairing


def pairing_track(*, y: np.ndarray, lvl: np.ndarray) -> np.ndarray:
    """M0: flag half-hours between a merged fall of 1.5 MVA and the next merged rise of half of it.

    Consecutive same-sign differences each above 4 spreads are merged into one step. Returns a
    boolean array of the half-hours flagged as running.
    """
    spread = SPREAD_LEVELS[lvl]
    d = np.zeros(len(y))
    d[1:] = np.nan_to_num(np.diff(y))
    big = np.abs(d) > 4.0 * spread
    flag = np.zeros(len(y), dtype=bool)
    open_at: int | None = None
    fall = 0.0
    t = 1
    while t < len(y):
        if not big[t]:
            t += 1
            continue
        sign = np.sign(d[t])
        end = t
        while end + 1 < len(y) and big[end + 1] and np.sign(d[end + 1]) == sign:
            end += 1
        size = float(d[t : end + 1].sum())
        if open_at is None and size <= -1.5:
            open_at, fall = end, -size
        elif open_at is not None and size >= 0.5 * fall:
            flag[open_at:t] = True
            open_at = None
        t = end + 1
    if open_at is not None:
        flag[open_at:] = True
    return flag
