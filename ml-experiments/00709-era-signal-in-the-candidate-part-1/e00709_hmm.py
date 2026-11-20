"""Experiment 709: a forward-filtered sticky hidden-Markov model of a Primary's daily features.

Arm L-hmm follows the stickiness idea of [Fox et al.
(2011)](https://doi.org/10.1214/10-AOAS395): three Gaussian states per Primary over the six scaled
features, with a self-transition probability fixed at 1 - 1/180 per day. The means, the diagonal
variances, and the initial state probabilities are fitted by expectation-maximisation on the days
before a training cut, and the transition matrix stays fixed. The signal at the origin of day `D` is
the *forward-filtered* state distribution after day `D - 1`; there is no smoothing, so no later day
can change it.

**Class rule.** Duration and recurrence are the only evidence of class. The *base state* is the
state that the filter's most probable state occupies most often in the training days. A day whose
most probable state differs from the current base state is part of an excursion. An excursion
reports `unknown` for its first 28 days of dwell and `era` from the 28th day on, and the excursion's
state then becomes the base state (a new era). An excursion that the filter leaves within 28 days
was a switching event, reported as `switch` on the day the filter returns.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import logsumexp

N_STATES: Final[int] = 3
STAY_DAYS: Final[float] = 180.0
SELF_TRANSITION: Final[float] = 1.0 - 1.0 / STAY_DAYS
ERA_DWELL_DAYS: Final[int] = 28
EM_ITERATIONS: Final[int] = 25
MIN_VARIANCE: Final[float] = 0.05
STATE_NONE: Final[int] = 0
STATE_UNKNOWN: Final[int] = 1
STATE_ERA: Final[int] = 2
STATE_SWITCH: Final[int] = 3


@dataclass(frozen=True)
class HmmParameters:
    """Parameters of one Primary's sticky hidden-Markov model.

    Attributes:
        means: State means, shape (3, features), in scaled units.
        variances: Diagonal variances, shape (3, features).
        initial: Initial state probabilities.
    """

    means: np.ndarray
    variances: np.ndarray
    initial: np.ndarray


@dataclass(frozen=True)
class HmmSignal:
    """The forward-filtered signal of one Primary, by day.

    Attributes:
        probabilities: Filtered state probabilities after each day, shape (days, 3).
        state: Class code of each day: 0 none, 1 unknown, 2 era, 3 switch.
        dwell_days: Days that the most probable state has been the same, per day.
        era_index: Number of eras adopted by each day.
        change_vector: The current state's mean minus the base state's mean in scaled units,
            shape (days, features); the same for an era after it is adopted.
        active_excursion: Index of the excursion that is under way on each day, or -1. On the
            day the filter returns to the base state the ended excursion is still reported.
        persistent_excursion: Index of the excursion that became the current era, or -1.
        excursions: First day, last day (-1 while open), and adoption day (-1 if never adopted)
            of every excursion, in order.
    """

    probabilities: np.ndarray
    state: np.ndarray
    dwell_days: np.ndarray
    era_index: np.ndarray
    change_vector: np.ndarray
    active_excursion: np.ndarray
    persistent_excursion: np.ndarray
    excursions: list[tuple[int, int, int]]


def _transition() -> np.ndarray:
    off = (1.0 - SELF_TRANSITION) / (N_STATES - 1)
    return np.full((N_STATES, N_STATES), off) + np.eye(N_STATES) * (SELF_TRANSITION - off)


def _log_emission(*, x: np.ndarray, params: HmmParameters) -> np.ndarray:
    """Return the log density of each day under each state, shape (days, states)."""
    diff = x[:, None, :] - params.means[None, :, :]
    log_density = -0.5 * (
        np.log(2 * np.pi * params.variances)[None, :, :] + diff**2 / params.variances
    )
    return log_density.sum(axis=2)


def fit_hmm(*, scaled: np.ndarray, train_end_day: int, start_day: int) -> HmmParameters:
    """Fit the three-state model by expectation-maximisation on days before a cut.

    The days are initialised by splitting the level (the mean of the four band features) into
    terciles. The transition matrix is not fitted.

    Args:
        scaled: Scaled daily features, shape (days, features).
        train_end_day: First day that the fit must not read.
        start_day: First day with a valid scale.

    Returns:
        The fitted parameters.
    """
    block = scaled[start_day:train_end_day]
    x = block[np.isfinite(block).all(axis=1)]
    level = x[:, :4].mean(axis=1)
    cuts = np.quantile(level, [1 / 3, 2 / 3])
    labels = np.digitize(level, cuts)
    means = np.stack([x[labels == k].mean(axis=0) for k in range(N_STATES)])
    variances = np.stack(
        [np.maximum(x[labels == k].var(axis=0), MIN_VARIANCE) for k in range(N_STATES)]
    )
    params = HmmParameters(means=means, variances=variances, initial=np.full(N_STATES, 1 / 3))
    log_trans = np.log(_transition())
    for _ in range(EM_ITERATIONS):
        log_b = _log_emission(x=x, params=params)
        n = x.shape[0]
        log_alpha = np.empty((n, N_STATES))
        log_alpha[0] = np.log(params.initial) + log_b[0]
        for t in range(1, n):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]
        log_beta = np.zeros((n, N_STATES))
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )
        log_gamma = log_alpha + log_beta
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        weight = gamma.sum(axis=0) + 1e-9
        means = (gamma.T @ x) / weight[:, None]
        variances = np.maximum(
            np.stack(
                [
                    (gamma[:, k, None] * (x - means[k]) ** 2).sum(axis=0) / weight[k]
                    for k in range(N_STATES)
                ]
            ),
            MIN_VARIANCE,
        )
        params = HmmParameters(means=means, variances=variances, initial=gamma[0] + 1e-9)
    return params


def forward_filter(*, scaled: np.ndarray, params: HmmParameters, start_day: int) -> np.ndarray:
    """Return the forward-filtered state probabilities after each day.

    A day with a missing feature only applies the transition. Days before `start_day` hold the
    initial distribution.

    Args:
        scaled: Scaled daily features, shape (days, features).
        params: The fitted parameters.
        start_day: First day to read.

    Returns:
        An array (days, 3).
    """
    n_days = scaled.shape[0]
    transition = _transition()
    out = np.tile(params.initial / params.initial.sum(), (n_days, 1))
    belief = out[0].copy()
    for day in range(max(start_day, 0), n_days):
        belief = belief @ transition
        if np.isfinite(scaled[day]).all():
            log_b = _log_emission(x=scaled[day][None, :], params=params)[0]
            belief = belief * np.exp(log_b - log_b.max())
        belief = belief / belief.sum()
        out[day] = belief
    return out


def classify_states(
    *, probabilities: np.ndarray, params: HmmParameters, train_end_day: int, start_day: int
) -> HmmSignal:
    """Turn filtered probabilities into the class rule of arm L-hmm.

    Args:
        probabilities: The output of `forward_filter`.
        params: The parameters the filter used, for the change vector.
        train_end_day: First day after the training cut; the base state is the modal state before.
        start_day: First day with a valid scale.

    Returns:
        The class, dwell, era index, and change vector of each day, the open and the persistent
        excursion of each day, and the first day, last day, and adoption day of every excursion.
        The state is `none` before `train_end_day`, because the fit read those days.
    """
    n_days = probabilities.shape[0]
    mode = probabilities.argmax(axis=1)
    counts = np.bincount(mode[start_day:train_end_day], minlength=N_STATES)
    base = int(counts.argmax())
    state = np.zeros(n_days, dtype=np.int8)
    dwell = np.zeros(n_days, dtype=np.int64)
    era_index = np.zeros(n_days, dtype=np.int64)
    change = np.zeros((n_days, params.means.shape[1]))
    active = np.full(n_days, -1, dtype=np.int64)
    persistent = np.full(n_days, -1, dtype=np.int64)
    excursions: list[list[int]] = []
    eras = 0
    era_change = np.zeros(params.means.shape[1])
    excursion_state, current_era = -1, -1
    # The signal is `none` before the fit's training cut: the base state is only known from then.
    for day in range(max(start_day, train_end_day, 1), n_days):
        current = int(mode[day])
        dwell[day] = dwell[day - 1] + 1 if current == mode[day - 1] else 1
        if current != base:
            if current != excursion_state:
                if excursions and excursions[-1][1] == -1:
                    excursions[-1][1] = day - 1
                excursion_state = current
                excursions.append([day, -1, -1])
            k = len(excursions) - 1
            active[day] = k
            change[day] = params.means[current] - params.means[base]
            if day - excursions[k][0] + 1 >= ERA_DWELL_DAYS:
                era_change = change[day].copy()
                base, eras, excursion_state, current_era = current, eras + 1, -1, k
                excursions[k][2] = day
                excursions[k][1] = day
                state[day] = STATE_ERA
            else:
                state[day] = STATE_UNKNOWN
        elif excursion_state != -1:
            k = len(excursions) - 1
            excursions[k][1] = day
            active[day] = k
            state[day] = STATE_SWITCH
            excursion_state = -1
        era_index[day] = eras
        persistent[day] = current_era
        if state[day] == STATE_NONE and eras > 0:
            state[day] = STATE_ERA
            change[day] = era_change
    return HmmSignal(
        probabilities=probabilities,
        state=state,
        dwell_days=dwell,
        era_index=era_index,
        change_vector=change,
        active_excursion=active,
        persistent_excursion=persistent,
        excursions=[(a, b, c) for a, b, c in excursions],
    )
