# ruff: noqa: E501
"""Experiment 720, rung 0b: a no-network posterior for the sign of each half-hour of an MVA reading.

Notation, in points of capacity. `v` is the reading with the reactive floor removed, `p` the
estimate of unmetered solar output, `u` the yield curve (0 to about 1). A half-hour that imported
has gross demand `g+ = p + v`, and one that exported has `g- = p - v`. With one Gaussian for the
gross demand of the column, `G` and spread `s_G`, the log-odds of export are

    b + log N(g-; G, s_G² + τ²) - log N(g+; G, s_G² + τ²),

which for equal variances reduce to `b + 2 v (p - G) / (s_G² + τ²)`: a half-hour exports when the PV
estimate exceeds the gross demand of its column. The read-out therefore turns on how `G` is built.

- `anchored_posterior`: `G` and `s_G` come from reference rows with `u < u_min` (dull or dark), at the
  same half-hour and day type, weighted by sky-blind weights (temperature similarity and age). Under
  restriction R1 an anchor row imported, so `G` does not depend on the posterior.
- `unanchored_posterior`: the control. `G` is a weighted mean of the neighbouring days' gross demand
  at the same half-hour, each starting at `ρ = 0`, with leave-one-out and a number of passes.

Everything is vectorised over days. No fit and no neural network is involved.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

U_MIN: Final[float] = 0.3
"""The likelihood gate of experiment 412: a half-hour can export only where `u` is at least this."""
PRIOR_LOG_ODDS: Final[float] = -4.0
DEAD_BAND_POINTS: Final[float] = 2.0
MIN_ANCHORS: Final[float] = 3.0
"""The effective number of anchor rows below which a column falls back to the trailing dull rows."""
WeightKind = Literal["day", "sky"]


@dataclass(frozen=True)
class SignParams:
    """The tunable constants of the read-outs.

    Attributes:
        prior_log_odds: `b`, the prior log-odds of export of a gated half-hour.
        tau: The spread added to the gross-demand spread, in points.
        temperature_scale: The temperature difference (degrees Celsius) at which a reference row's
            weight falls to `exp(-1/2)`.
        age_half_life_days: The half-life of the age weight of a reference row.
        floor_spread: The smallest `s_G`, in points.
        min_g_minus: A half-hour with `g- < -min_g_minus` cannot export (gross demand is not
            negative).
    """

    prior_log_odds: float = PRIOR_LOG_ODDS
    tau: float = 2.0
    temperature_scale: float = 3.0
    age_half_life_days: float = 365.0
    floor_spread: float = 1.0
    min_g_minus: float = 2.0


def log_odds(
    *,
    v: np.ndarray,
    p: np.ndarray,
    g_mean: np.ndarray,
    g_spread: np.ndarray,
    params: SignParams,
) -> np.ndarray:
    """Return the log-odds of export of each half-hour given the column's `G` and `s_G`."""
    var = g_spread**2 + params.tau**2
    g_plus = p + v
    g_minus = p - v
    lp = -0.5 * (g_plus - g_mean) ** 2 / var
    lm = -0.5 * (g_minus - g_mean) ** 2 / var
    out = params.prior_log_odds + lm - lp
    return np.where(g_minus < -params.min_g_minus, -np.inf, out)


def anchored_posterior(
    *,
    v: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    day_type: np.ndarray,
    temperature: np.ndarray,
    reference_days: np.ndarray,
    scored_days: np.ndarray,
    params: SignParams = SignParams(),  # noqa: B008
    u_min: float = U_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `ρ` for the scored days, built from anchor rows of the reference days only.

    Args:
        v: Reading with the floor removed, `(n_days, 48)`.
        p: PV estimate, same shape (the placebo passes the sky-shifted estimate).
        u: Real yield curve, same shape (it sets the gate and which rows are anchors).
        day_type: `(n_days,)` day type.
        temperature: `(n_days,)` daily temperature.
        reference_days: Day indices that anchor rows may come from (the training window).
        scored_days: Day indices to label. They must not overlap `reference_days`.
        params: The constants.
        u_min: The gate.

    Returns:
        `rho` with shape `(len(scored_days), 48)`, and `fallback` of the same shape: 1 where the
        column had under `MIN_ANCHORS` effective anchor rows and `G` came from all reference rows
        with `u < 2 × u_min` instead (the trailing dull rows).
    """
    assert not np.intersect1d(reference_days, scored_days).size, "scored days leak into the anchors"
    n_scored = len(scored_days)
    rho = np.zeros((n_scored, v.shape[1]))
    fallback = np.zeros((n_scored, v.shape[1]))
    ref_age = (scored_days.max() - reference_days).astype(np.float64)
    ref_age = np.abs(ref_age)
    age_w = 0.5 ** (ref_age / params.age_half_life_days)
    dtemp = (
        temperature[scored_days][:, None] - temperature[reference_days][None, :]
    ) / params.temperature_scale
    base_w = np.exp(-0.5 * dtemp**2) * age_w[None, :]
    same_type = day_type[scored_days][:, None] == day_type[reference_days][None, :]
    for h in range(v.shape[1]):
        g_ref = p[reference_days, h] + v[reference_days, h]
        usable = np.isfinite(g_ref)
        done = np.zeros(n_scored, dtype=bool)
        for cut, flag in ((u_min, 0.0), (2.0 * u_min, 1.0)):
            anchor = usable & (u[reference_days, h] < cut)
            w = base_w * same_type * anchor[None, :]
            n_eff = w.sum(axis=1)
            todo = (~done) & (n_eff >= MIN_ANCHORS)
            if not todo.any():
                continue
            safe = np.where(n_eff > 0, n_eff, 1.0)
            g0 = np.where(anchor, g_ref, 0.0)
            mean = (w * g0[None, :]).sum(axis=1) / safe
            var = (w * (g0[None, :] - mean[:, None]) ** 2).sum(axis=1) / safe
            spread = np.sqrt(np.maximum(var, params.floor_spread**2))
            lo = log_odds(
                v=v[scored_days, h],
                p=p[scored_days, h],
                g_mean=mean,
                g_spread=spread,
                params=params,
            )
            r = np.where(u[scored_days, h] >= u_min, expit(lo), 0.0)
            rho[todo, h] = r[todo]
            fallback[todo, h] = flag
            done |= todo
    return rho, fallback


def unanchored_posterior(
    *,
    v: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    day_type: np.ndarray,
    params: SignParams = SignParams(),  # noqa: B008
    weights: WeightKind = "day",
    half_window: int = 7,
    n_passes: int = 3,
    u_min: float = U_MIN,
) -> list[np.ndarray]:
    """Return `ρ` after each pass of the unanchored control, for every day with a full window.

    Every row of the window of `2 × half_window` neighbouring days starts at `ρ = 0`. In each pass
    `G` of a row is the weighted mean, over the other rows of its window (leave-one-out), of the
    gross demand `(1 − ρ) g+ + ρ g-`, and `s_G` the weighted spread. `weights = "day"` gives a
    neighbour weight 1 on the row's own day type and 0.2 otherwise, and `"sky"` multiplies that by a
    Gaussian in the difference of `u` at the same half-hour (peaked on rows of the same sky).

    Returns:
        A list of `n_passes` arrays `(n_days, 48)`; days within `half_window` of either end are NaN.
    """
    n_days, n_steps = v.shape
    offsets = [k for k in range(-half_window, half_window + 1) if k != 0]
    g_plus, g_minus = p + v, p - v
    rho = np.zeros((n_days, n_steps))
    out = []
    lo_d, hi_d = half_window, n_days - half_window
    for _ in range(n_passes):
        g_now = (1.0 - rho) * g_plus + rho * g_minus
        num = np.zeros((hi_d - lo_d, n_steps))
        den = np.zeros_like(num)
        sq = np.zeros_like(num)
        for k in offsets:
            sl = slice(lo_d + k, hi_d + k)
            w = np.where(day_type[lo_d:hi_d] == day_type[sl], 1.0, 0.2)[:, None] * np.ones(n_steps)
            if weights == "sky":
                w = w * np.exp(-0.5 * ((u[lo_d:hi_d] - u[sl]) / 0.15) ** 2)
            w = w * np.isfinite(g_now[sl])
            gn = np.nan_to_num(g_now[sl])
            num += w * gn
            den += w
            sq += w * gn**2
        mean = num / np.where(den > 0, den, 1.0)
        var = np.maximum(sq / np.where(den > 0, den, 1.0) - mean**2, params.floor_spread**2)
        lo = log_odds(
            v=v[lo_d:hi_d], p=p[lo_d:hi_d], g_mean=mean, g_spread=np.sqrt(var), params=params
        )
        r = np.where(u[lo_d:hi_d] >= u_min, expit(lo), 0.0)
        rho = np.zeros((n_days, n_steps))
        rho[lo_d:hi_d] = r
        full = np.full((n_days, n_steps), np.nan)
        full[lo_d:hi_d] = r
        out.append(full)
    return out


def auc(*, score: np.ndarray, positive: np.ndarray) -> float:
    """Return the area under the receiver operating characteristic curve (ties averaged)."""
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_false_rate(*, score: np.ndarray, positive: np.ndarray, rate: float = 0.05) -> float:
    """Return the recall at the threshold where `rate` of the negatives score at or above it."""
    neg = score[~positive]
    if neg.size == 0 or not positive.any():
        return float("nan")
    thr = np.quantile(neg, 1.0 - rate)
    return float(np.mean(score[positive] > thr))


def evaluate_sign(
    *,
    rho: np.ndarray,
    x_true: np.ndarray,
    u: np.ndarray,
    dead_band: float = DEAD_BAND_POINTS,
    u_min: float = U_MIN,
) -> dict[str, float]:
    """Score `ρ` against the signed truth on gated half-hours outside the dead band.

    A half-hour counts when `u >= u_min`, `ρ` is finite, and `|x_true| >= dead_band`. The positive
    class is `x_true < 0`. Also returns the counts, so every rate names its denominator.
    """
    ok = np.isfinite(rho) & np.isfinite(x_true) & (u >= u_min) & (np.abs(x_true) >= dead_band)
    pos = x_true[ok] < 0.0
    s = rho[ok]
    return {
        "auc": auc(score=s, positive=pos),
        "recall_at_5pct_false": recall_at_false_rate(score=s, positive=pos, rate=0.05),
        "n_rows": float(ok.sum()),
        "n_reverse": float(pos.sum()),
        "n_import": float((~pos).sum()),
    }


def false_reverse_rate(
    *, rho: np.ndarray, mask: np.ndarray, threshold: float = 0.5
) -> tuple[float, int]:
    """Return the share of `mask` rows with `ρ > threshold`, and the number of rows in `mask`."""
    sel = mask & np.isfinite(rho)
    n = int(sel.sum())
    return (float(np.mean(rho[sel] > threshold)) if n else float("nan")), n
