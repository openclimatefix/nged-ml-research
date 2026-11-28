"""Survival curves for shift durations, the survival ratio, and the age rule A1.

The survival ratio is one formula, in days: `p(a, h) = S(a + h) / S(a)`, where `a` is the age of the
shift at the contract origin, `h` is the lead after the contract origin, and `S` is the survival
function of shift durations. `ratio` computes it for a fitted curve and `hazard_ratio` for the 10
daily hazards of the learned arm LR. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

import e00738_common as common
import numpy as np
from scipy import optimize, stats

FAMILIES: Final[dict[str, stats.rv_continuous]] = {
    "expon": stats.expon,
    "weibull": stats.weibull_min,
    "lognorm": stats.lognorm,
    "loglogistic": stats.fisk,
}
"""The four candidate families, as in experiment 652."""
SHORT_DAYS: Final[tuple[int, int]] = (3, 14)
LONG_DAYS: Final[tuple[int, int]] = (28, 112)
SHORT_SHARE: Final[float] = 0.5


class SurvivalFunction(Protocol):
    """Anything with a survival function of the duration in days."""

    def sf(self, t: np.ndarray) -> np.ndarray:
        """Return P(duration > t) for `t` in days."""
        ...


@dataclass(frozen=True)
class Curve:
    """A fitted duration distribution.

    Attributes:
        family: A key of `FAMILIES`.
        params: The shape parameters (none, or one), then the location 0, then the scale.
        nll: The negative log-likelihood of the durations it was fitted to.
    """

    family: str
    params: tuple[float, ...]
    nll: float

    def sf(self, t: np.ndarray) -> np.ndarray:
        """Return P(duration > t), with t in days."""
        return FAMILIES[self.family].sf(np.asarray(t, dtype=float), *self.params)


@dataclass(frozen=True)
class GeneratorCurve:
    """The true survival function of the training generator's drawn durations."""

    def sf(self, t: np.ndarray) -> np.ndarray:
        """Return P(duration > t): a mixture of two uniform ranges of whole days."""
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        for share, (lo, hi) in ((SHORT_SHARE, SHORT_DAYS), (1.0 - SHORT_SHARE, LONG_DAYS)):
            n = hi - lo + 1
            # the number of integer days d in [lo, hi] with d > t
            above = np.clip(hi - np.maximum(np.floor(t), lo - 1), 0, n)
            out += share * above / n
        return out


def fit_curve(*, durations_days: np.ndarray) -> Curve:
    """Return the best of the four families by likelihood, fitted to drawn durations.

    The durations are the generator's drawn (unclipped) durations of every shift, so the fit is not
    biased towards short shifts by clipping at the window edge.

    Args:
        durations_days: Drawn durations in days, all positive.

    Returns:
        The curve with the lowest negative log-likelihood.
    """
    d = np.asarray(durations_days, dtype=float)
    assert d.size > 0
    assert bool((d > 0).all()), "a duration is not positive"
    best: Curve | None = None
    for name, dist in FAMILIES.items():
        n_par = 0 if name == "expon" else 1

        def nll(theta: np.ndarray, dist: stats.rv_continuous = dist, n_par: int = n_par) -> float:
            shape = np.exp(theta[:n_par]) if n_par else ()
            return -float(dist.logpdf(d, *shape, 0.0, np.exp(theta[n_par])).sum())

        r = optimize.minimize(
            nll, x0=np.r_[np.zeros(n_par), np.log(d.mean())], method="Nelder-Mead"
        )
        par = (*(float(v) for v in np.exp(r.x[:n_par])), 0.0, float(np.exp(r.x[n_par])))
        if best is None or r.fun < best.nll:
            best = Curve(family=name, params=par, nll=float(r.fun))
    assert best is not None
    return best


def ratio(*, curve: SurvivalFunction, age_days: np.ndarray, lead_days: np.ndarray) -> np.ndarray:
    """Return p(a, h) = S(a + h) / S(a), floored in the denominator and clipped to 0 to 1.

    Args:
        curve: The survival function.
        age_days: The age of the shift at the contract origin, in days.
        lead_days: The lead after the contract origin, in days (broadcast against the age).

    Returns:
        The chance that the shift is still on at the lead.
    """
    a = np.asarray(age_days, dtype=float)
    h = np.asarray(lead_days, dtype=float)
    p = curve.sf(a + h) / np.maximum(curve.sf(a), common.SURVIVAL_FLOOR)
    return np.clip(p, 0.0, 1.0)


def curve_error(*, curve: SurvivalFunction) -> dict[float, float]:
    """Return the absolute error of the fitted S against the generator's true S at the checkpoints.

    Args:
        curve: The fitted curve.

    Returns:
        The error at each of `CURVE_ERROR_DAYS`.
    """
    t = np.array(common.CURVE_ERROR_DAYS)
    err = np.abs(curve.sf(t) - GeneratorCurve().sf(t))
    return {float(a): float(b) for a, b in zip(t, err, strict=True)}


def curve_error_ok(*, curve: SurvivalFunction) -> bool:
    """Return whether the largest curve error is under `CURVE_ERROR_MAX`."""
    return max(curve_error(curve=curve).values()) < common.CURVE_ERROR_MAX


def daily_hazards(*, curve: SurvivalFunction, n_days: int = common.N_HAZARDS) -> np.ndarray:
    """Return the `n_days` daily hazards that reproduce `curve` at whole days.

    Hazard `d` is `-log(S(d + 1) / S(d))`, so the product of daily survival terms equals `S` at
    every
    whole day up to `n_days`.

    Args:
        curve: The survival function.
        n_days: How many daily hazards.

    Returns:
        A non-negative array of length `n_days`.
    """
    t = np.arange(n_days + 1, dtype=float)
    s = np.maximum(curve.sf(t), 1e-12)
    return np.maximum(-np.log(s[1:] / s[:-1]), 0.0)


def cumulative_hazard(*, hazards: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    """Return H(t), the sum of each day's hazard times the share of that day before `t`.

    Days after the last hazard continue at the last hazard (the floor of the learned curve).

    Args:
        hazards: The daily hazards, length `N_HAZARDS`.
        t_days: Times in days after the shift began.

    Returns:
        H(t), with the shape of `t_days`.
    """
    t = np.asarray(t_days, dtype=float)
    days = np.arange(len(hazards), dtype=float)
    share = np.clip(t[..., None] - days, 0.0, 1.0)
    tail = np.maximum(t - len(hazards), 0.0) * hazards[-1]
    return (share * hazards).sum(axis=-1) + tail


def hazard_ratio(*, hazards: np.ndarray, age_days: np.ndarray, lead_days: np.ndarray) -> np.ndarray:
    """Return p(a, h) for a curve given by daily hazards, with the same floor as `ratio`.

    Args:
        hazards: The daily hazards, length `N_HAZARDS`.
        age_days: The age of the shift at the contract origin.
        lead_days: The lead after the contract origin, in days.

    Returns:
        The chance that the shift is still on at the lead.
    """
    a = np.asarray(age_days, dtype=float)
    s_a = np.exp(-cumulative_hazard(hazards=hazards, t_days=a))
    s_ah = np.exp(
        -cumulative_hazard(hazards=hazards, t_days=a + np.asarray(lead_days, dtype=float))
    )
    return np.clip(s_ah / np.maximum(s_a, common.SURVIVAL_FLOOR), 0.0, 1.0)


def noise_unit(*, row_means: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation of the daily differences of row means.

    Args:
        row_means: Daily mean standardised power over the fold's training window.

    Returns:
        The noise unit of A1.
    """
    d = np.diff(row_means[np.isfinite(row_means)])
    assert d.size > 0
    return float(common.MAD_TO_SD * np.median(np.abs(d - np.median(d))))


def row_summaries(*, hist_power: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean of the visible half-hours of each history row, and whether the row has any.

    Args:
        hist_power: Standardised history power, shape (..., 14, 48); 0 where not visible.
        visible: Whether each half-hour is visible, same shape.

    Returns:
        The row means (0 where no half-hour is visible) and the presence flags, each (..., 14).
    """
    n = visible.sum(axis=-1)
    total = np.where(visible, hist_power, 0.0).sum(axis=-1)
    present = n > 0
    return np.where(present, total / np.maximum(n, 1), 0.0), present


def age_rule_a1(*, hist_power: np.ndarray, visible: np.ndarray, noise: float) -> np.ndarray:
    """Return the age in days of the shift that A1 sees, 0 where it sees none.

    The reference level is the mean of the oldest 7 history rows. A1 counts the newest consecutive
    rows (ages 2, 3, ...) whose mean differs from the reference by more than 2 noise units. The age
    is the count plus 2. A row with no visible half-hour ends the count, and fewer than 3 reference
    rows with data means that no shift is seen.

    Args:
        hist_power: Standardised history power, shape (B, 14, 48), newest row first.
        visible: Visibility of each half-hour, same shape.
        noise: The noise unit from `noise_unit`.

    Returns:
        A float array (B,) of ages, 0.0 where no shift is seen.
    """
    means, present = row_summaries(hist_power=hist_power, visible=visible)
    ref_rows = slice(common.N_HISTORY // 2, common.N_HISTORY)
    n_ref = present[:, ref_rows].sum(axis=1)
    reference = np.where(present[:, ref_rows], means[:, ref_rows], 0.0).sum(axis=1) / np.maximum(
        n_ref, 1
    )
    differs = present & (np.abs(means - reference[:, None]) > common.A1_Z * noise)
    # the count of the newest consecutive rows that differ: the cumulative product stops at a gap
    run = np.cumprod(differs, axis=1).sum(axis=1)
    run = np.where(n_ref >= common.A1_MIN_REFERENCE_ROWS, run, 0)
    return np.where(run > 0, run + common.MIN_AGE_DAYS, 0).astype(np.float64)


def noise_unit_from_panel(*, z: np.ndarray, cutoff_idx: int) -> float:
    """Return A1's noise unit from the standardised panel of a fold's training window.

    Row means are taken over the 24-hour blocks that start at 12:00 UTC (the blocks of the history
    rows), per Primary. The day-to-day differences of the row means of every Primary are pooled.

    Args:
        z: Standardised power (16, grid), NaN where dropped.
        cutoff_idx: The fold's cutoff; nothing after it is read.

    Returns:
        The noise unit.
    """
    n_days = (cutoff_idx - 24) // common_per_day() - 1
    diffs = []
    for p in range(z.shape[0]):
        blocks = z[p, 24 : 24 + 48 * n_days].reshape(n_days, 48)
        with np.errstate(all="ignore"):
            means = np.nanmean(blocks, axis=1)
        diffs.append(np.diff(means[np.isfinite(means)]))
    return noise_unit(row_means=np.concatenate([np.array([0.0]), np.cumsum(np.concatenate(diffs))]))


def common_per_day() -> int:
    """Return the half-hours in a day."""
    return 48
