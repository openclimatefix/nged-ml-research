# ruff: noqa: E501
"""Causal and hindsight states of experiment 694's folded grid filter (M2), with level posteriors.

Experiment 694's `grid_track` returns the share of half-hours with a unit running. This module
reuses its model (the grid on the demand level L, the number of units k, the folded emission
y = sqrt((L - k u)^2 + q^2), the spread-scaled Student-t random walk of L) and adds what experiment
695 needs: for every half-hour, the marginal P(k = j) and the posterior mean of L.

**Three smoothers, one model.**

- `smooth_fixed_lag` gives the state at half-hour t from the readings up to t + 20 hours, so a state
  is published by the time the reading at t + 20 hours has arrived. Passing `y_obs` with the readings
  after a cut set to NaN gives the training-time version, which never reads past the cut.
- `smooth_full` runs the full backward pass: the hindsight state of arm AH, a bound and never a
  forecast.
- The forward pass (the filtered state) is experiment 694's `_grid_forward`.

Parameters (K, u, transition matrix) are fitted without truth by experiment 694's `select_grid` and
`_grid_em`.
"""

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from e00694_common import Q_FLOOR, SPREAD_LEVELS, causal_spread  # ty: ignore[unresolved-import]
from e00694_trackers import (  # ty: ignore[unresolved-import]
    EM_ROUNDS,
    KERNELS,
    LEVEL_GRID,
    N_GRID,
    OBS_SCALE,
    _grid_em,
    _grid_forward,
    _means,
    _state_transition,
    occupancy,
    select_grid,
    t_pdf,
)

LAG_HALF_HOURS: Final[int] = 40
"""The fixed lag: 20 hours of later readings. A reading arrives 3.5 hours late, so a state at t
rests on readings published by t + 23.5 hours; the network reads no power younger than 24 hours."""
CHUNK: Final[int] = 1500
"""Half-hours smoothed together in one batch of the fixed-lag backward recursion."""
N_STATES: Final[int] = 4
"""Columns of the state marginal: 0 to 3 units running."""
SPREAD_FILL: Final[float] = 0.15
"""Spread (MVA) of a half-hour with fewer than 7 earlier days, a constant so no cut changes it."""
KERNELS_T: Final[np.ndarray] = np.ascontiguousarray(np.transpose(KERNELS, (0, 2, 1)))
"""The transposed random-walk kernels, so the backward step is a plain matrix product."""


@dataclass(frozen=True)
class EraFit:
    """The unsupervised fit of one era: units K, unit size u (MW), and the transition matrix."""

    k_max: int
    unit: float
    transition: np.ndarray


@dataclass
class States:
    """Per half-hour state marginal and the level posterior mean (MVA), on a segment."""

    pk: np.ndarray
    lmean: np.ndarray


def spread_levels(*, y: np.ndarray) -> np.ndarray:
    """Return the index of the nearest of 8 spread levels from the trailing 28-day spread.

    The spread at a half-hour reads differences at the same half-hour of earlier days only, so it
    never reads a reading younger than 24 hours. A half-hour without it takes `SPREAD_FILL`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # an all-NaN day-lag stack
        spread = causal_spread(y=y)
    filled = np.where(np.isfinite(spread), spread, SPREAD_FILL)
    return np.argmin(np.abs(np.log(filled)[:, None] - np.log(SPREAD_LEVELS)[None, :]), axis=1)


def fit_era(*, y: np.ndarray, lvl: np.ndarray) -> EraFit:
    """Fit K, u and the transition matrix on a segment of readings, without truth."""
    k_max, unit = select_grid(y=y, lvl=lvl, fold=True)
    if k_max == 0:
        return EraFit(k_max=0, unit=unit, transition=np.ones((1, 1)))
    transition, _ = _grid_em(
        y=y, lvl=lvl, k_max=k_max, unit=unit, fold=True, dwell=False, rounds=EM_ROUNDS
    )
    return EraFit(k_max=k_max, unit=unit, transition=transition)


def _model(*, fit: EraFit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the state means (S, G), the occupancy prior (S, 1), and the state transition."""
    means = _means(k_max=fit.k_max, unit=fit.unit, fold=True, dwell=False)
    occ = occupancy(k_max=fit.k_max, dwell=False)
    return means, occ, _state_transition(a_k=fit.transition, dwell=False)


def forward(*, y: np.ndarray, lvl: np.ndarray, fit: EraFit) -> tuple[np.ndarray, np.ndarray]:
    """Return the filtered state (T, S, G) and the scales of the forward pass."""
    means, occ, a = _model(fit=fit)
    _, alpha, scales = _grid_forward(y=y, lvl=lvl, means=means, a=a, keep=True, occ=occ)
    return alpha, scales


def _emission_batch(*, y_u: np.ndarray, means: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Return the emission of each reading in `y_u` for every state and grid point: (m, S, G)."""
    z = (y_u[:, None, None] - means[None]) / OBS_SCALE
    e = t_pdf(z) / OBS_SCALE * occ[None]
    missing = ~np.isfinite(y_u)
    if missing.any():
        e[missing] = np.broadcast_to(occ[None], e[missing].shape)
    return e


def _marginals(*, gamma: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn joint posteriors (n, S, G) into the state marginal (n, 4) and the level mean (n,)."""
    gamma = gamma / gamma.sum(axis=(1, 2), keepdims=True)
    per_state = gamma.sum(axis=2)
    pk = np.zeros((len(gamma), N_STATES))
    pk[:, : k_max + 1] = per_state
    return pk, gamma.sum(axis=1) @ LEVEL_GRID


def smooth_fixed_lag(
    *,
    y_obs: np.ndarray,
    lvl: np.ndarray,
    fit: EraFit,
    alpha: np.ndarray,
    first: int,
    stop: int,
    lag: int = LAG_HALF_HOURS,
) -> States:
    """Return the fixed-lag states of half-hours `first` to `stop`, reading `y_obs` up to t + lag.

    The backward message of half-hour t is built from the emissions at t + 1 to t + lag only, so
    nothing after t + lag enters the state. `alpha` must come from `forward`, whose filtered state is
    causal, so a `y_obs` with readings set to NaN after a cut changes no state before the cut. Past
    the end of the data the recursion stops (the message stays uniform).

    Args:
        y_obs: The readings the backward messages may read (MVA, NaN where unobserved).
        lvl: The spread level index of each half-hour.
        fit: The era's fit.
        alpha: The filtered state from `forward`, on the same segment.
        first: The first half-hour of the segment to return.
        stop: One past the last half-hour to return.
        lag: The number of later half-hours read.

    Returns:
        The state marginals and level means of half-hours `first` to `stop`.
    """
    means, occ, a = _model(fit=fit)
    n_total = len(y_obs)
    pk_out = np.zeros((stop - first, N_STATES))
    lm_out = np.zeros(stop - first)
    for c0 in range(first, stop, CHUNK):
        c1 = min(c0 + CHUNK, stop)
        n = c1 - c0
        x = np.ones((n, fit.k_max + 1, N_GRID))
        for step in range(lag):
            u = c0 + lag - step + np.arange(n)
            live = np.nonzero(u < n_total)[0]
            if not len(live):
                continue
            uu = u[live]
            v = _emission_batch(y_u=y_obs[uu], means=means, occ=occ) * x[live]
            w = np.empty_like(v)
            levels = lvl[uu]
            for level in np.unique(levels):
                pick = levels == level
                w[pick] = v[pick] @ KERNELS_T[level]
            moved = np.einsum("ij,mjg->mig", a, w)
            moved /= moved.max(axis=(1, 2), keepdims=True)
            x[live] = moved
        pk, lm = _marginals(gamma=alpha[c0:c1] * x, k_max=fit.k_max)
        pk_out[c0 - first : c1 - first], lm_out[c0 - first : c1 - first] = pk, lm
    return States(pk=pk_out, lmean=lm_out)


def smooth_full(*, y: np.ndarray, lvl: np.ndarray, fit: EraFit) -> States:
    """Return the hindsight states: the full backward pass over the whole segment."""
    means, occ, a = _model(fit=fit)
    alpha, scales = forward(y=y, lvl=lvl, fit=fit)
    n_t = len(y)
    pk_out = np.zeros((n_t, N_STATES))
    lm_out = np.zeros(n_t)
    beta = np.ones((fit.k_max + 1, N_GRID))
    for t in range(n_t - 1, -1, -1):
        pk, lm = _marginals(gamma=(alpha[t] * beta)[None], k_max=fit.k_max)
        pk_out[t], lm_out[t] = pk[0], lm[0]
        if t == 0:
            break
        emission = _emission_batch(y_u=y[t : t + 1], means=means, occ=occ)[0]
        beta = (a @ ((emission * beta) @ KERNELS_T[lvl[t]])) / scales[t]
    return States(pk=pk_out, lmean=lm_out)


def identity_states(*, y: np.ndarray) -> States:
    """Return the states of a series with no engine: k = 0 everywhere, the level is the reading."""
    pk = np.zeros((len(y), N_STATES))
    pk[:, 0] = 1.0
    return States(pk=pk, lmean=y.astype(np.float64).copy())


def _states_of_segment(
    *,
    y: np.ndarray,
    lvl: np.ndarray,
    fit: EraFit,
    cut: int | None,
    cut_visible: int,
    lag: int | None,
) -> States:
    """Track one segment: fixed-lag states with a training cut, or hindsight states.

    Args:
        y: The segment's readings (MVA, NaN where dropped).
        lvl: The segment's spread levels.
        fit: The segment's fit.
        cut: Half-hours before this index (the fold's refit) are tracked with `y` hidden from
            `cut_visible` on; later half-hours see all of `y`. `None` means no cut.
        cut_visible: The first half-hour hidden from the training-time states (the refit minus the
            3.5 hours a reading takes to arrive).
        lag: The fixed lag in half-hours; `None` is the hindsight smoother.

    Returns:
        The states of every half-hour of the segment.
    """
    if fit.k_max == 0:
        return identity_states(y=y)
    if lag is None:
        return smooth_full(y=y, lvl=lvl, fit=fit)
    n_t = len(y)
    if cut is None or cut >= n_t:
        alpha, _ = forward(y=y, lvl=lvl, fit=fit)
        return smooth_fixed_lag(y_obs=y, lvl=lvl, fit=fit, alpha=alpha, first=0, stop=n_t, lag=lag)
    hidden = y.copy()
    hidden[cut_visible:] = np.nan
    alpha_hidden, _ = forward(y=hidden, lvl=lvl, fit=fit)
    before = smooth_fixed_lag(
        y_obs=hidden, lvl=lvl, fit=fit, alpha=alpha_hidden, first=0, stop=max(cut, 0), lag=lag
    )
    del alpha_hidden
    alpha, _ = forward(y=y, lvl=lvl, fit=fit)
    after = smooth_fixed_lag(y_obs=y, lvl=lvl, fit=fit, alpha=alpha, first=cut, stop=n_t, lag=lag)
    return States(
        pk=np.concatenate([before.pk, after.pk]), lmean=np.concatenate([before.lmean, after.lmean])
    )


def track_segment(
    *,
    y: np.ndarray,
    fit: EraFit,
    refit: int | None,
    reading_delay: int = 7,
    lag: int | None = LAG_HALF_HOURS,
) -> States:
    """Track a segment of readings with a fitted era model.

    Args:
        y: The segment's readings (MVA, NaN where dropped or unobserved).
        fit: The fit of this segment's era.
        refit: Index within the segment of the fold's refit; training-time states of half-hours
            before it never read a reading from `refit - reading_delay` on. `None` for no cut.
        reading_delay: Half-hours a reading takes to arrive (3.5 hours is 7).
        lag: The fixed lag in half-hours, or `None` for the hindsight smoother.

    Returns:
        The states of every half-hour of the segment.
    """
    lvl = spread_levels(y=y)
    cut_visible = None if refit is None else refit - reading_delay
    return _states_of_segment(
        y=y,
        lvl=lvl,
        fit=fit,
        cut=refit,
        cut_visible=0 if cut_visible is None else cut_visible,
        lag=lag,
    )


def cleaned_series(
    *, y: np.ndarray, states: States, fit: EraFit, mode: str = "folded"
) -> np.ndarray:
    """Return the cleaned reading (MVA) of a segment from its states.

    Where the most probable unit count is 0 the cleaned value is the reading itself. Elsewhere it is
    the posterior mean of the demand level (`mode="folded"`), or the reading plus the units' output
    with no fold (`mode="signed"`, arm A1n). A half-hour with no reading stays NaN.
    """
    khat = np.argmax(states.pk, axis=1)
    running = (khat >= 1) & np.isfinite(y)
    engine_free = states.lmean if mode == "folded" else y + khat * fit.unit
    return np.where(running, engine_free, y)


def fold_branch_share(*, y: np.ndarray, states: States, fit: EraFit) -> tuple[int, int]:
    """Return (rows with k >= 1 whose two cleanings differ by more than 1 MVA, rows with k >= 1).

    The two cleanings are the posterior-mean cleaning and the no-fold cleaning y + k u. Their
    difference measures how often the fold's two branches matter.
    """
    khat = np.argmax(states.pk, axis=1)
    running = (khat >= 1) & np.isfinite(y)
    if not running.any():
        return 0, 0
    folded = cleaned_series(y=y, states=states, fit=fit, mode="folded")
    signed = cleaned_series(y=y, states=states, fit=fit, mode="signed")
    return int((np.abs(folded - signed)[running] > 1.0).sum()), int(running.sum())


def q_floor() -> float:
    """Return the reactive-power floor (MVA) the tracker's fold uses."""
    return Q_FLOOR
