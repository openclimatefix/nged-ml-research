"""The Bayesian capacity fit of experiment 737 and its seven rival and reference arms.

The model of arm A is linear in its coefficients. The power of half-hour `i` is
`baseline_i + level_q(i) - capacity_q(i) x s_i + e_i`, where `s_i` is the unit solar term and
`e_i` follows a Student-t density with 4 degrees of freedom, written as a scale mixture. The
expectation-maximisation loop sets each row's weight to `(4 + 1) / (4 + (r / sigma)^2)`, solves the
weighted normal equations with the Gaussian priors added, and sets each class's scale to the
weighted root mean square of its residuals. The posterior mean is the converged coefficients. The
posterior covariance is the sandwich covariance `A^-1 B A^-1` of the penalised Student-t
log-likelihood (`sandwich_capacity_covariance`), because the covariance `(X' W X)^-1` of the
converged weights understates the variance when the noise is lighter-tailed than the Student-t
density.

The priors: every baseline coefficient has a zero-mean Gaussian prior whose precision is the ridge
penalty of `studies.pv_separation` times a tuned ridge scale. The capacities follow a Gaussian
random walk with a positive drift (arm A), one flat capacity (arm F), or independent flat priors
(arm Qt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00737_common import (
    DOF,
    DRIFT_MW_PER_QUARTER,
    EM_MAX_ITER,
    EM_TOL_CAPACITY_MW,
    EM_TOL_SIGMA,
    FIRST_QUARTER_MEAN_MW,
    FIRST_QUARTER_SD_MW,
    MAX_AUTOCORR,
    MIN_Q_ROWS,
    N_QUARTERS,
    PRIOR_DC_AC,
    QT_MEAN_MW,
    QT_SD_MW,
    RATIO_GRID,
    WARM_ITER,
)
from e00737_forward import PRIOR_FLEET, FleetParams
from e00737_inputs import (
    SEASONAL_COLUMNS,
    FitData,
    PrimaryInputs,
    build_fit_data,
    ridge_precision,
    unit_term,
)
from scipy.linalg import cho_factor, cho_solve
from studies.pv_separation import RIDGE_PER_COLUMN, separate, solar_basis

MIN_AUTOCORR_PAIRS: Final[int] = 6
"""Fewest consecutive daylight pairs of a day for the day's lag-1 autocorrelation to count."""
LAGS_FOR_SIGMA_START: Final[int] = 48
"""The lag of the differences that give the noise scale its starting value."""


@dataclass(frozen=True)
class CapacityPrior:
    """A Gaussian prior on the capacity columns of the regression.

    Attributes:
        mean: The prior mean of each column, megawatts.
        precision: The prior precision matrix of the columns.
        group: The column that each quarter's capacity is read from.
    """

    mean: np.ndarray
    precision: np.ndarray
    group: np.ndarray


def random_walk_prior(*, growth_sd: float) -> CapacityPrior:
    """Return the random-walk prior of arm A.

    The first quarter's capacity has mean 2 MW and standard deviation 2 MW. Each later quarter's
    capacity is the previous one plus a Gaussian step of mean 0.10 MW and standard deviation
    `growth_sd`.

    Args:
        growth_sd: The standard deviation of the growth step, megawatts per quarter.

    Returns:
        The prior over the 18 quarterly capacities.
    """
    difference = np.eye(N_QUARTERS) - np.eye(N_QUARTERS, k=-1)
    step_precision = np.full(N_QUARTERS, 1.0 / growth_sd**2)
    step_precision[0] = 1.0 / FIRST_QUARTER_SD_MW**2
    mean = FIRST_QUARTER_MEAN_MW + DRIFT_MW_PER_QUARTER * np.arange(N_QUARTERS)
    return CapacityPrior(
        mean=mean,
        precision=difference.T @ np.diag(step_precision) @ difference,
        group=np.arange(N_QUARTERS),
    )


def flat_prior() -> CapacityPrior:
    """Return the prior of arm F: one capacity for every quarter, with no growth step.

    Returns:
        A one-column prior with mean 2 MW and standard deviation 2 MW.
    """
    return CapacityPrior(
        mean=np.array([FIRST_QUARTER_MEAN_MW]),
        precision=np.array([[1.0 / FIRST_QUARTER_SD_MW**2]]),
        group=np.zeros(N_QUARTERS, dtype=np.int64),
    )


def independent_prior() -> CapacityPrior:
    """Return the prior of arm Qt: an independent flat Gaussian on each quarter.

    Returns:
        A prior with mean 2 MW and standard deviation 5 MW for each of the 18 quarters.
    """
    return CapacityPrior(
        mean=np.full(N_QUARTERS, QT_MEAN_MW),
        precision=np.eye(N_QUARTERS) / QT_SD_MW**2,
        group=np.arange(N_QUARTERS),
    )


@dataclass(frozen=True)
class EmState:
    """The weights and scales that start (or end) an expectation-maximisation loop."""

    weights: np.ndarray
    sigma_day: float
    sigma_night: float


@dataclass(frozen=True)
class FitResult:
    """One fit's outcome.

    Attributes:
        theta: The coefficients: the baseline columns, then the capacity columns.
        capacity: The posterior mean of each quarter's capacity, megawatts (not cut at zero).
        sd: The standard deviation of each quarter's capacity from the sandwich covariance, with
            the autocorrelation inflation applied.
        n_iter: The expectation-maximisation iterations used.
        converged: Whether the loop stopped on its tolerance.
        state: The final weights and scales.
        infl: The multiplier `(1 + r) / (1 - r)` on the capacity variance.
        sandwich: Whether the sandwich covariance was used (False when the observed information
            was not positive definite and the covariance `(X' W X)^-1` stood in).
        evidence: The Gaussian log evidence at the final weights (NaN unless asked for).
        seconds: The wall time of the fit.
    """

    theta: np.ndarray
    capacity: np.ndarray
    sd: np.ndarray
    n_iter: int
    converged: bool
    state: EmState
    infl: float
    sandwich: bool
    evidence: float
    seconds: float


def lag1_inflation(
    *, z: np.ndarray, rows: np.ndarray, day: np.ndarray, daylight: np.ndarray
) -> float:
    """Return `(1 + r) / (1 - r)` for the median lag-1 autocorrelation `r` of daylight residuals.

    The autocorrelation is computed for each day over the pairs of consecutive daylight half-hours,
    and `r` is the median over the days with at least `MIN_AUTOCORR_PAIRS` pairs, clipped to the
    interval from 0 to `MAX_AUTOCORR`.

    Args:
        z: The whitened residual of each row.
        rows: The grid index of each row.
        day: Each row's day number.
        daylight: Whether each row is a daylight row.

    Returns:
        The variance multiplier, at least 1.
    """
    pair = (rows[1:] == rows[:-1] + 1) & daylight[1:] & daylight[:-1] & (day[1:] == day[:-1])
    a, b, d = z[:-1][pair], z[1:][pair], day[:-1][pair]
    if len(d) == 0:
        return 1.0
    _, inverse = np.unique(d, return_inverse=True)
    k = len(np.unique(d))
    n = np.bincount(inverse, minlength=k).astype(float)
    sa, sb = (
        np.bincount(inverse, weights=a, minlength=k),
        np.bincount(inverse, weights=b, minlength=k),
    )
    saa = np.bincount(inverse, weights=a * a, minlength=k)
    sbb = np.bincount(inverse, weights=b * b, minlength=k)
    sab = np.bincount(inverse, weights=a * b, minlength=k)
    keep = n >= MIN_AUTOCORR_PAIRS
    if not keep.any():
        return 1.0
    cov = sab[keep] / n[keep] - sa[keep] * sb[keep] / n[keep] ** 2
    va = saa[keep] / n[keep] - (sa[keep] / n[keep]) ** 2
    vb = sbb[keep] / n[keep] - (sb[keep] / n[keep]) ** 2
    corr = cov / np.sqrt(np.maximum(va * vb, 1e-300))
    r = float(np.clip(np.median(corr), 0.0, MAX_AUTOCORR))
    return (1.0 + r) / (1.0 - r)


def starting_sigma(*, y: np.ndarray, rows: np.ndarray, daylight: np.ndarray) -> tuple[float, float]:
    """Return starting noise scales from the robust spread of one-day differences.

    Args:
        y: The observations of the rows.
        rows: The grid index of each row.
        daylight: Whether each row is a daylight row.

    Returns:
        The daylight and the night scale, megawatts (1.0 when too few pairs exist).
    """
    position = np.searchsorted(rows, rows - LAGS_FOR_SIGMA_START)
    position = np.minimum(position, len(rows) - 1)
    has_partner = rows[position] == rows - LAGS_FOR_SIGMA_START
    later = np.flatnonzero(has_partner)
    if len(later) < 100:
        return 1.0, 1.0
    earlier = position[later]
    diff = y[later] - y[earlier]
    out = []
    for flag in (True, False):
        pick = daylight[later] == flag
        if pick.sum() < 50:
            out.append(float(np.median(out)) if out else 1.0)
            continue
        d = diff[pick]
        out.append(max(float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)), 1e-3))
    return out[0], out[1]


@dataclass(frozen=True)
class _EmRun:
    """The outcome of the expectation-maximisation loop (a private helper of `fit_student_t`)."""

    theta: np.ndarray
    factor: tuple
    lam: np.ndarray
    used_sigma_day: float
    used_sigma_night: float
    state: EmState
    n_iter: int
    converged: bool


def _run_em(
    *,
    x: np.ndarray,
    y: np.ndarray,
    daylight: np.ndarray,
    prior: CapacityPrior,
    ridge_abs: float,
    n_base: int,
    start: EmState,
    max_iter: int,
) -> _EmRun:
    """Run the weighted solve, weight update, and scale update until the tolerance or the limit.

    Args:
        x: The regression matrix (baseline columns, then the capacity columns).
        y: The observation of each row.
        daylight: Whether each row is a daylight row.
        prior: The capacity prior.
        ridge_abs: The absolute ridge precision of the baseline's coefficients.
        n_base: The number of baseline columns.
        start: The starting weights and scales.
        max_iter: The most iterations.

    Returns:
        The last solve (its coefficients, Cholesky factor, and row precisions) and the final state.
    """
    xw = np.empty_like(x)
    weights, sigma_day, sigma_night = start.weights, start.sigma_day, start.sigma_night
    prior_rhs = prior.precision @ prior.mean
    diag = np.arange(n_base)
    night = ~daylight
    previous: np.ndarray | None = None
    converged = False
    run: _EmRun | None = None
    for it in range(1, max_iter + 1):
        sigma = np.where(daylight, sigma_day, sigma_night)
        lam = weights / sigma**2
        np.multiply(x, np.sqrt(lam)[:, None], out=xw)
        gram = xw.T @ xw
        rhs = x.T @ (lam * y)
        gram[diag, diag] += ridge_abs / sigma_day**2
        gram[n_base:, n_base:] += prior.precision
        rhs[n_base:] += prior_rhs
        factor = cho_factor(gram, lower=True, check_finite=False)
        theta = cho_solve(factor, rhs, check_finite=False)
        resid = y - x @ theta
        weights = (DOF + 1.0) / (DOF + (resid / sigma) ** 2)
        wr2 = weights * resid**2
        new_day = float(np.sqrt(wr2[daylight].sum() / weights[daylight].sum()))
        new_night = (
            float(np.sqrt(wr2[night].sum() / weights[night].sum())) if night.any() else sigma_night
        )
        if previous is not None:
            moved = float(np.max(np.abs(theta[n_base:] - previous)))
            rel = max(abs(new_day / sigma_day - 1.0), abs(new_night / sigma_night - 1.0))
            converged = moved < EM_TOL_CAPACITY_MW and rel < EM_TOL_SIGMA
        previous = theta[n_base:].copy()
        run = _EmRun(
            theta=theta,
            factor=factor,
            lam=lam,
            used_sigma_day=sigma_day,
            used_sigma_night=sigma_night,
            state=EmState(weights=weights, sigma_day=new_day, sigma_night=new_night),
            n_iter=it,
            converged=converged,
        )
        sigma_day, sigma_night = new_day, new_night
        if converged:
            break
    assert run is not None
    return run


def _log_evidence(
    *,
    run: _EmRun,
    resid: np.ndarray,
    prior: CapacityPrior,
    ridge_abs: float,
    n_base: int,
) -> float:
    """Return the Gaussian log evidence of the data at the weights of the last solve.

    Args:
        run: The expectation-maximisation outcome.
        resid: The residual of each row at the posterior mean.
        prior: The capacity prior.
        ridge_abs: The absolute ridge precision of the baseline's coefficients.
        n_base: The number of baseline columns.

    Returns:
        The log evidence.
    """
    ridge_term = ridge_abs / run.used_sigma_day**2
    cap_gap = run.theta[n_base:] - prior.mean
    quad = (
        float((run.lam * resid**2).sum())
        + ridge_term * float((run.theta[:n_base] ** 2).sum())
        + float(cap_gap @ prior.precision @ cap_gap)
    )
    log_det_a = 2.0 * float(np.log(np.diag(run.factor[0])).sum())
    log_det_p = n_base * np.log(ridge_term) + float(np.linalg.slogdet(prior.precision)[1])
    return float(
        -0.5 * quad
        - 0.5 * log_det_a
        + 0.5 * log_det_p
        + 0.5 * np.log(run.lam).sum()
        - 0.5 * len(resid) * np.log(2.0 * np.pi)
    )


def sandwich_capacity_covariance(
    *,
    x: np.ndarray,
    resid: np.ndarray,
    sigma: np.ndarray,
    precision: np.ndarray,
    n_base: int,
) -> np.ndarray | None:
    """Return the capacity block of the sandwich covariance of the penalised Student-t fit.

    The fit maximises the Student-t log-likelihood plus the Gaussian log priors. Its score for row
    `i` is `psi_i x_i` with `psi_i = (DOF + 1) r_i / (DOF sigma_i^2 + r_i^2)`. The covariance of
    the estimate is `A^-1 B A^-1`, where `A` is the observed information
    `sum_i psi'_i x_i x_i' + precision` and `B` is `sum_i psi_i^2 x_i x_i'`. The matrix `B` has no
    prior term, because the prior is not random. When the noise really is Student-t, `A` and `B`
    agree and the covariance is the usual posterior covariance. When the noise is Gaussian, they
    differ, and the covariance is the variance that the Student-t estimator really has.

    Args:
        x: The regression matrix (baseline columns, then the capacity columns).
        resid: The residual of each row at the estimate.
        sigma: The noise scale that the last solve used for each row.
        precision: The prior precision of every column (ridge on the baseline, the capacity prior
            on the capacities), including the scale `1 / sigma_day^2` of the ridge.
        n_base: The number of baseline columns.

    Returns:
        The covariance of the capacity columns, or `None` when `A` is not positive definite.
    """
    scale2 = DOF * sigma**2
    denom = scale2 + resid**2
    psi = (DOF + 1.0) * resid / denom
    dpsi = (DOF + 1.0) * (scale2 - resid**2) / denom**2
    info = (x * dpsi[:, None]).T @ x + precision
    try:
        factor = cho_factor(info, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None
    selector = np.zeros((x.shape[1], x.shape[1] - n_base))
    selector[n_base:, :] = np.eye(x.shape[1] - n_base)
    left = cho_solve(factor, selector, check_finite=False)
    score_outer = (x * (psi**2)[:, None]).T @ x
    return left.T @ score_outer @ left


def fit_student_t(
    *,
    y: np.ndarray,
    base: np.ndarray,
    solar: np.ndarray,
    quarter: np.ndarray,
    daylight: np.ndarray,
    rows: np.ndarray,
    day: np.ndarray,
    prior: CapacityPrior,
    ridge_abs: float,
    init: EmState | None = None,
    max_iter: int = EM_MAX_ITER,
    evidence: bool = False,
    autocorrelation: bool = True,
) -> FitResult:
    """Fit baseline and capacities by Student-t expectation-maximisation, then give the posterior.

    The regression's capacity columns hold minus the unit solar term, so a positive capacity
    lowers the modelled power, as generation does.

    Args:
        y: The observation of each row.
        base: The baseline columns of each row.
        solar: The unit solar term `s_i` of each row.
        quarter: Each row's quarter, zero-based.
        daylight: Whether each row is a daylight row (the noise class).
        rows: The grid index of each row (ascending).
        day: Each row's day number.
        prior: The capacity prior.
        ridge_abs: The absolute ridge precision of the baseline's coefficients.
        init: Starting weights and scales (a warm start); `None` starts cold.
        max_iter: The most iterations.
        evidence: Whether to compute the Gaussian log evidence at the final weights.
        autocorrelation: Whether to inflate the variance by the lag-1 autocorrelation of the
            daylight residuals.

    Returns:
        The fit.
    """
    start = time.perf_counter()
    n, n_base = base.shape
    k = len(prior.mean)
    x = np.zeros((n, n_base + k))
    x[:, :n_base] = base
    x[np.arange(n), n_base + prior.group[quarter]] = -solar
    if init is None:
        sigma_day, sigma_night = starting_sigma(y=y, rows=rows, daylight=daylight)
        init = EmState(weights=np.ones(n), sigma_day=sigma_day, sigma_night=sigma_night)
    run = _run_em(
        x=x,
        y=y,
        daylight=daylight,
        prior=prior,
        ridge_abs=ridge_abs,
        n_base=n_base,
        start=init,
        max_iter=max_iter,
    )
    resid = y - x @ run.theta
    sigma_used = np.where(daylight, run.used_sigma_day, run.used_sigma_night)
    precision = np.zeros((n_base + k, n_base + k))
    precision[np.arange(n_base), np.arange(n_base)] = ridge_abs / run.used_sigma_day**2
    precision[n_base:, n_base:] = prior.precision
    cov = sandwich_capacity_covariance(
        x=x, resid=resid, sigma=sigma_used, precision=precision, n_base=n_base
    )
    sandwich = cov is not None
    if cov is None:
        selector = np.zeros((n_base + k, k))
        selector[n_base:, :] = np.eye(k)
        cov = cho_solve(run.factor, selector, check_finite=False)[n_base:, :]
    z = np.sqrt(run.state.weights) * resid / sigma_used
    infl = lag1_inflation(z=z, rows=rows, day=day, daylight=daylight) if autocorrelation else 1.0
    group = prior.group
    return FitResult(
        theta=run.theta,
        capacity=run.theta[n_base:][group],
        sd=np.sqrt(np.diag(cov)[group] * infl),
        n_iter=run.n_iter,
        converged=run.converged,
        state=run.state,
        infl=infl,
        sandwich=sandwich,
        evidence=(
            _log_evidence(run=run, resid=resid, prior=prior, ridge_abs=ridge_abs, n_base=n_base)
            if evidence
            else float("nan")
        ),
        seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class PrimaryModel:
    """A Primary's regression inputs, built once and reused by every world and arm.

    Attributes:
        inputs: The Primary's inputs.
        data: The baseline design of the rows the fit reads.
        terms: The unit solar term of each fixed-ratio fleet at those rows, by DC:AC ratio.
        q_basis: The four orientation curves of `studies.pv_separation.solar_basis` at those rows.
    """

    inputs: PrimaryInputs
    data: FitData
    terms: dict[float, np.ndarray]
    q_basis: np.ndarray


def prior_fleet_with_ratio(*, ratio: float) -> FleetParams:
    """Return the prior's fleet with another DC:AC ratio.

    Args:
        ratio: The DC:AC ratio.

    Returns:
        The fleet.
    """
    return FleetParams(mix=PRIOR_FLEET.mix, tilt_deg=PRIOR_FLEET.tilt_deg, dc_ac_ratio=ratio)


def build_model(*, inputs: PrimaryInputs) -> PrimaryModel:
    """Build a Primary's design, its solar terms at every ratio of arm Af, and arm Q's basis.

    Args:
        inputs: The Primary's inputs.

    Returns:
        The model inputs.
    """
    data = build_fit_data(inputs=inputs)
    ratios = sorted({PRIOR_DC_AC, *RATIO_GRID})
    terms = {
        r: unit_term(inputs=inputs, fleet=prior_fleet_with_ratio(ratio=r))[data.rows]
        for r in ratios
    }
    basis = solar_basis(sky=inputs.sky, dc_ac_ratio=PRIOR_DC_AC)[data.rows]
    return PrimaryModel(inputs=inputs, data=data, terms=terms, q_basis=basis)


def fit_arm(
    *,
    model: PrimaryModel,
    power: np.ndarray,
    prior: CapacityPrior,
    ridge_scale: float,
    solar: np.ndarray | None = None,
    init: EmState | None = None,
    max_iter: int = EM_MAX_ITER,
    evidence: bool = False,
) -> FitResult:
    """Fit one arm of the Student-t family to a world's power on the Primary's rows.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid (NaN where there is no reading).
        prior: The capacity prior.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
        solar: The unit solar term at the rows (the prior fleet at its ratio by default).
        init: A warm start.
        max_iter: The most iterations.
        evidence: Whether to compute the log evidence.

    Returns:
        The fit.
    """
    data = model.data
    return fit_student_t(
        y=power[data.rows],
        base=data.base,
        solar=model.terms[PRIOR_DC_AC] if solar is None else solar,
        quarter=data.quarter,
        daylight=data.daylight,
        rows=data.rows,
        day=data.day,
        prior=prior,
        ridge_abs=ridge_scale * data.ridge_abs,
        init=init,
        max_iter=max_iter,
        evidence=evidence,
    )


def fit_daily(
    *, model: PrimaryModel, power: np.ndarray, prior: CapacityPrior, ridge_scale: float
) -> FitResult:
    """Fit arm Ad: the same model on the sums of each day's daylight half-hours.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid.
        prior: The capacity prior.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.

    Returns:
        The fit; each row of it is one day, and the variance has no autocorrelation inflation.
    """
    data = model.data
    keep = data.daylight
    day = data.day[keep]
    starts = np.flatnonzero(np.concatenate([[True], day[1:] != day[:-1]]))
    y = np.add.reduceat(power[data.rows][keep], starts)
    base = np.add.reduceat(data.base[keep], starts, axis=0)
    solar = np.add.reduceat(model.terms[PRIOR_DC_AC][keep], starts)
    quarter = data.quarter[keep][starts]
    ridge_abs = ridge_scale * ridge_precision(seasonal=base[:, :SEASONAL_COLUMNS])
    return fit_student_t(
        y=y,
        base=base,
        solar=solar,
        quarter=quarter,
        daylight=np.ones(len(y), dtype=bool),
        rows=day[starts],
        day=day[starts],
        prior=prior,
        ridge_abs=ridge_abs,
        autocorrelation=False,
    )


def fit_free_ratio(
    *,
    model: PrimaryModel,
    power: np.ndarray,
    prior: CapacityPrior,
    ridge_scale: float,
    start: EmState,
) -> tuple[FitResult, float]:
    """Fit arm Af: choose the DC:AC ratio from the grid by the highest Gaussian evidence.

    Each ratio's fit starts from arm A's weights and runs at most `WARM_ITER` iterations.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid.
        prior: The capacity prior.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.
        start: Arm A's final weights and scales.

    Returns:
        The best fit and its ratio.
    """
    best: tuple[FitResult, float] | None = None
    for ratio in RATIO_GRID:
        fit = fit_arm(
            model=model,
            power=power,
            prior=prior,
            ridge_scale=ridge_scale,
            solar=model.terms[ratio],
            init=start,
            max_iter=WARM_ITER,
            evidence=True,
        )
        if best is None or fit.evidence > best[0].evidence:
            best = (fit, ratio)
    assert best is not None
    return best


def fit_residual_arm(
    *, model: PrimaryModel, power: np.ndarray, prior: CapacityPrior, ridge_scale: float
) -> FitResult:
    """Fit arm R: the residual of experiment 734's stored forecast, with a quarterly level.

    The baseline holds only the 18 quarterly levels, because the stored forecast has already
    removed the calendar and the weather.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid.
        prior: The capacity prior.
        ridge_scale: The tuned multiplier of the baseline's ridge precision.

    Returns:
        The fit.
    """
    data = model.data
    forecast = model.inputs.forecast[data.rows]
    ok = np.isfinite(forecast)
    levels = np.zeros((int(ok.sum()), N_QUARTERS))
    levels[np.arange(levels.shape[0]), data.quarter[ok]] = 1.0
    return fit_student_t(
        y=(power[data.rows] - forecast)[ok],
        base=levels,
        solar=model.terms[PRIOR_DC_AC][ok],
        quarter=data.quarter[ok],
        daylight=data.daylight[ok],
        rows=data.rows[ok],
        day=data.day[ok],
        prior=prior,
        ridge_abs=ridge_scale * RIDGE_PER_COLUMN * float((levels**2).sum(axis=0).mean()),
    )


def fit_quarters(*, model: PrimaryModel, power: np.ndarray) -> np.ndarray:
    """Fit arm Q: `studies.pv_separation.separate` on each quarter alone.

    The output handed to `separate` is minus the signed power, because generation is a positive
    output there. The baseline of each fit is the 336 calendar columns on that quarter's rows.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid.

    Returns:
        The total AC capacity of each quarter, megawatts (NaN with under 1,000 usable half-hours).
    """
    data = model.data
    out = np.full(N_QUARTERS, np.nan)
    for q in range(N_QUARTERS):
        pick = data.quarter == q
        if pick.sum() < MIN_Q_ROWS:
            continue
        fit = separate(
            output_mw=-power[data.rows][pick],
            basis=model.q_basis[pick],
            design=data.base[pick, :SEASONAL_COLUMNS],
        )
        if fit is not None:
            out[q] = fit.total_capacity_mw
    return out


@dataclass(frozen=True)
class WorldFits:
    """The arms' capacity paths for one world (or one real record)."""

    capacity: dict[str, np.ndarray]
    sd: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    """Per arm: iterations, converged, daylight scale, night scale, inflation, ratio, evidence,
    seconds, and whether the sandwich covariance was used (1.0) or the fit fell back to
    `(X' W X)^-1` (0.0)."""


def _info(*, fit: FitResult, ratio: float = PRIOR_DC_AC) -> np.ndarray:
    return np.array(
        [
            fit.n_iter,
            float(fit.converged),
            fit.state.sigma_day,
            fit.state.sigma_night,
            fit.infl,
            ratio,
            fit.evidence,
            fit.seconds,
            float(fit.sandwich),
        ]
    )


def fit_all_arms(
    *,
    model: PrimaryModel,
    power: np.ndarray,
    growth_sd: float,
    ridge_scale: float,
    oracle_term: np.ndarray | None,
) -> WorldFits:
    """Fit the eight arms of one world.

    Args:
        model: The Primary's model inputs.
        power: The world's power on the grid.
        growth_sd: The tuned growth standard deviation of arm A, megawatts per quarter.
        ridge_scale: The tuned multiplier of the baseline's ridge precision, used by every arm.
        oracle_term: The unit solar term of the injected fleet at the fit rows for arm O, or None
            to reuse arm A's fit (a world with no injected fleet, or whose fleet is the prior's).

    Returns:
        Each arm's capacity, standard deviation, and fit information.
    """
    cap: dict[str, np.ndarray] = {}
    sd: dict[str, np.ndarray] = {}
    info: dict[str, np.ndarray] = {}

    def put(*, arm: str, fit: FitResult, ratio: float = PRIOR_DC_AC) -> None:
        cap[arm], sd[arm], info[arm] = fit.capacity, fit.sd, _info(fit=fit, ratio=ratio)

    rw = random_walk_prior(growth_sd=growth_sd)
    fit_a = fit_arm(model=model, power=power, prior=rw, ridge_scale=ridge_scale)
    put(arm="A", fit=fit_a)
    put(arm="F", fit=fit_arm(model=model, power=power, prior=flat_prior(), ridge_scale=ridge_scale))
    put(
        arm="Qt",
        fit=fit_arm(model=model, power=power, prior=independent_prior(), ridge_scale=ridge_scale),
    )
    start = time.perf_counter()
    cap["Q"] = fit_quarters(model=model, power=power)
    sd["Q"] = np.full(N_QUARTERS, np.nan)
    info["Q"] = np.array(
        [0, 1.0, np.nan, np.nan, 1.0, PRIOR_DC_AC, np.nan, time.perf_counter() - start, np.nan]
    )
    put(arm="Ad", fit=fit_daily(model=model, power=power, prior=rw, ridge_scale=ridge_scale))
    fit_af, ratio = fit_free_ratio(
        model=model, power=power, prior=rw, ridge_scale=ridge_scale, start=fit_a.state
    )
    put(arm="Af", fit=fit_af, ratio=ratio)
    if oracle_term is None:
        put(arm="O", fit=fit_a)
    else:
        put(
            arm="O",
            fit=fit_arm(
                model=model, power=power, prior=rw, solar=oracle_term, ridge_scale=ridge_scale
            ),
        )
    put(arm="R", fit=fit_residual_arm(model=model, power=power, prior=rw, ridge_scale=ridge_scale))
    return WorldFits(capacity=cap, sd=sd, info=info)


def fits_to_arrays(*, fits: WorldFits) -> dict[str, np.ndarray]:
    """Flatten a `WorldFits` into named arrays for a `.npz` file.

    Args:
        fits: The fits.

    Returns:
        Arrays named `<arm>_cap`, `<arm>_sd`, and `<arm>_info`.
    """
    out: dict[str, np.ndarray] = {}
    for arm in fits.capacity:
        out[f"{arm}_cap"] = fits.capacity[arm]
        out[f"{arm}_sd"] = fits.sd[arm]
        out[f"{arm}_info"] = fits.info[arm]
    return out


def summarise_info(*, tables: list[dict[str, np.ndarray]]) -> pl.DataFrame:
    """Collect the iteration counts and convergence flags of fit files into one table.

    Args:
        tables: The loaded arrays of each file.

    Returns:
        Columns `arm`, `n_iter`, `converged`, and `sandwich`, one row per fit.
    """
    rows = []
    for arrays in tables:
        for key, value in arrays.items():
            if key.endswith("_info") and key[:-5] in ("A", "F", "Qt", "Ad", "Af", "O", "R"):
                rows.append(
                    {
                        "arm": key[:-5],
                        "n_iter": int(value[0]),
                        "converged": bool(value[1]),
                        "sandwich": bool(value[8]) if len(value) > 8 else None,
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "arm": pl.String,
            "n_iter": pl.Int64,
            "converged": pl.Boolean,
            "sandwich": pl.Boolean,
        },
    )
