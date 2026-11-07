"""Experiment 617: the turbine-step hidden Markov model and its daily input series.

The hidden state is `k`, the number of turbines out of service. Capacity is `C0 - s * k`, where
`C0` is the full-farm ceiling and `s = C0 / N` the rated power of one turbine. The observation is
the daily 90th percentile of power over the half-hours with a wind speed in the saturation range of
the power curve (`daily_series`). Days with no informative half-hour carry no observation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy.special import logsumexp
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00617-wind-turbine-step-capacity-known-answer-and-fit"
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00453-effective-capacity-upper-bound-data-look"
    / "work"
    / "series.parquet"
)
PLANT_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00347-wind-physics-hybrid-plant-then-mdn" / "plant_stage3.json"
)
SITES: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
DEV_END: Final = pl.datetime(2026, 7, 1, time_zone="UTC")
SPEED_UPPER_LIMIT: Final[float] = 22.0
CUT_OUT_MARGIN: Final[float] = 3.0
MIN_ROWS_PER_DAY: Final[int] = 6
N_GRID: Final[np.ndarray] = np.geomspace(3.0, 40.0, 24)
MAX_STATES: Final[int] = 11
REPAIR_PER_DAY: Final[float] = 0.1
SIGMA_FRACTION: Final[float] = 0.04
CONTAMINATION: Final[float] = 0.2
FREE_FINE_UNITS: Final[int] = 60
FREE_MAX_STATES: Final[int] = 31


@dataclass(frozen=True)
class Daily:
    """Daily upper-envelope series of one wind farm."""

    site: str
    day: np.ndarray  # integer days since the first reading
    value: np.ndarray  # daily 90th percentile of power over the informative half-hours, MW
    n_days: int  # span of the record in days
    ceiling: float  # C0, MW

    @property
    def years(self) -> float:
        return self.n_days / 365.25


def daily_series(site: str) -> Daily:
    """Return the daily series of `site` ("W1", "W2" or "W3") up to the end of development data."""
    plant = json.loads(PLANT_PATH.read_text())[site]
    lo = plant["rated"] + plant["blur"]
    hi = min(plant["cut_out"] - CUT_OUT_MARGIN, SPEED_UPPER_LIMIT)
    wind = pl.read_parquet(SERIES_PATH).filter(
        (pl.col("site") == SITES[site]) & (pl.col("time") < DEV_END)
    )
    first = wind["time"].min().replace(hour=0, minute=0, second=0, microsecond=0)  # ty: ignore[unresolved-attribute]
    rows = (
        wind.filter(pl.col("wind").is_between(lo, hi) & pl.col("power").is_not_null())
        .with_columns(day=(pl.col("time") - first).dt.total_days())
        .group_by("day")
        .agg(n=pl.len(), value=pl.col("power").quantile(0.9))
        .filter(pl.col("n") >= MIN_ROWS_PER_DAY)
        .sort("day")
    )
    n_days = int((wind["time"].max() - first).days) + 1  # ty: ignore[unresolved-attribute]
    value = rows["value"].to_numpy().astype(float)
    return Daily(
        site=site,
        day=rows["day"].to_numpy().astype(int),
        value=value,
        n_days=n_days,
        ceiling=float(np.quantile(value, 0.99)),
    )


def remove_fault_days(daily: Daily, *, half_window: int = 60, fraction: float = 0.9) -> Daily:
    """Drop days whose value is under `fraction` of the rolling 90th percentile of values."""
    keep = np.ones(len(daily.day), dtype=bool)
    for i, d in enumerate(daily.day):
        near = np.abs(daily.day - d) <= half_window
        keep[i] = daily.value[i] >= fraction * np.quantile(daily.value[near], 0.9)
    return Daily(daily.site, daily.day[keep], daily.value[keep], daily.n_days, daily.ceiling)


def transition_matrix(
    n_states: int, *, step_up: float, repair: float = REPAIR_PER_DAY
) -> np.ndarray:
    """Return the daily transition matrix over `k = 0 .. n_states - 1` turbines out.

    One more turbine fails with probability `step_up`, one is repaired with probability `repair`,
    and two move together with a tenth of those probabilities. Mass that would leave the grid stays.
    """
    jumps = {1: step_up, -1: repair, 2: step_up / 10, -2: repair / 10}
    matrix = np.zeros((n_states, n_states))
    for k in range(n_states):
        for jump, p in jumps.items():
            target = k + jump
            if 0 <= target < n_states and (jump > 0 or k > 0):
                matrix[k, target] += p
        matrix[k, k] = 1.0 - matrix[k].sum()
    return matrix


def free_size_transition(n_states: int, *, step_up: float, repair: float = REPAIR_PER_DAY) -> np.ndarray:
    """Return a transition matrix whose step sizes are free: geometric with mean 4 fine units."""
    weights = 0.75 ** np.arange(1, 25)
    weights /= weights.sum()
    matrix = np.zeros((n_states, n_states))
    for k in range(n_states):
        for j, w in enumerate(weights, start=1):
            if k + j < n_states:
                matrix[k, k + j] += step_up * w
            if k - j >= 0:
                matrix[k, k - j] += repair * w
        matrix[k, k] = 1.0 - matrix[k].sum()
    return matrix


def log_emission(daily: Daily, level: np.ndarray, *, sigma_fraction: float = SIGMA_FRACTION,
                 contamination: float = CONTAMINATION) -> np.ndarray:
    """Return the `(n_days, n_levels)` log-likelihood of each day's value at each capacity level.

    A day without a value has log-likelihood 0. The model is a Gaussian around the capacity level
    plus a uniform contamination from 0 to C0 for curtailment, trips, and wind-speed error.
    """
    out = np.zeros((daily.n_days, len(level)))
    sigma = sigma_fraction * daily.ceiling
    z = (daily.value[:, None] - level[None, :]) / sigma
    gauss = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))
    mix = (1 - contamination) * gauss + contamination / daily.ceiling
    out[daily.day] = np.log(mix + 1e-300)
    return out


@dataclass(frozen=True)
class Fit:
    """Forward-backward result for one number of turbines."""

    log_evidence: float
    smoothed: np.ndarray  # (n_days, n_states) posterior of k given all days
    filtered: np.ndarray  # (n_days, n_states) posterior of k given days up to each day
    level_step: float  # MW of one unit of k


def forward_backward(log_em: np.ndarray, matrix: np.ndarray, prior: np.ndarray) -> tuple[
    float, np.ndarray, np.ndarray
]:
    """Return log evidence, smoothed and filtered posteriors of a hidden Markov model."""
    n, m = log_em.shape
    em = np.exp(log_em - log_em.max(axis=1, keepdims=True))
    shift = log_em.max(axis=1)
    alpha = np.zeros((n, m))
    norm = np.zeros(n)
    pred = prior
    for t in range(n):
        a = pred * em[t]
        norm[t] = a.sum()
        alpha[t] = a / norm[t]
        pred = alpha[t] @ matrix
    beta = np.ones(m)
    smoothed = np.zeros((n, m))
    for t in range(n - 1, -1, -1):
        g = alpha[t] * beta
        smoothed[t] = g / g.sum()
        b = matrix @ (em[t] * beta)
        beta = b / b.sum()
    return float(np.log(norm).sum() + shift.sum()), smoothed, alpha


def fit_grid(
    daily: Daily, *, step_up: float, n_grid: np.ndarray = N_GRID,
    sigma_fraction: float = SIGMA_FRACTION, contamination: float = CONTAMINATION,
) -> list[Fit]:
    """Fit the hidden Markov model for every number of turbines in `n_grid`."""
    fits = []
    for n_turbines in n_grid:
        step = daily.ceiling / n_turbines
        n_states = min(int(round(n_turbines)), MAX_STATES)
        level = daily.ceiling - step * np.arange(n_states)
        prior = np.zeros(n_states)
        prior[0] = 0.9
        prior[1:] = 0.1 / max(n_states - 1, 1)
        ev, smooth, filt = forward_backward(
            log_emission(daily, level, sigma_fraction=sigma_fraction, contamination=contamination),
            transition_matrix(n_states, step_up=step_up),
            prior,
        )
        fits.append(Fit(ev, smooth, filt, step))
    return fits


def fit_free_size(daily: Daily, *, step_up: float) -> float:
    """Return the log evidence of the free-size model on a 60-unit lattice (the comparison)."""
    step = daily.ceiling / FREE_FINE_UNITS
    level = daily.ceiling - step * np.arange(FREE_MAX_STATES)
    prior = np.zeros(FREE_MAX_STATES)
    prior[0] = 0.9
    prior[1:] = 0.1 / (FREE_MAX_STATES - 1)
    ev, _, _ = forward_backward(
        log_emission(daily, level), free_size_transition(FREE_MAX_STATES, step_up=step_up), prior
    )
    return ev


def log_n_weights(fits: list[Fit]) -> np.ndarray:
    """Return the posterior over the number of turbines (uniform prior on the grid)."""
    ev = np.array([f.log_evidence for f in fits])
    return np.exp(ev - logsumexp(ev))


def deficit_summary(fits: list[Fit], weights: np.ndarray, *, filtered: bool = False) -> dict[str, np.ndarray]:
    """Return per-day posterior mean and 5% / 95% quantiles of the capacity deficit `C0 - C`, MW."""
    n_days = (fits[0].filtered if filtered else fits[0].smoothed).shape[0]
    values, probs = [], []
    for fit, w in zip(fits, weights, strict=True):
        post = fit.filtered if filtered else fit.smoothed
        values.append(fit.level_step * np.arange(post.shape[1]))
        probs.append(post * w)
    mean = sum((p * v).sum(axis=1) for p, v in zip(probs, values, strict=True))
    all_values = np.concatenate(values)
    order = np.argsort(all_values)
    cum = np.cumsum(np.concatenate(probs, axis=1)[:, order], axis=1)
    sorted_values = all_values[order]
    q05 = sorted_values[np.minimum((cum < 0.05).sum(axis=1), len(order) - 1)]
    q95 = sorted_values[np.minimum((cum < 0.95).sum(axis=1), len(order) - 1)]
    return {"mean": np.asarray(mean), "q05": q05, "q95": q95, "n_days": np.asarray(n_days)}
