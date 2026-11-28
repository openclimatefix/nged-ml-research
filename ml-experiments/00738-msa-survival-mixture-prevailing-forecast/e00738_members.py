# ruff: noqa: E501
"""The 51-path check on the stored 11-quantile forecast.

CRPS at one valid time depends only on the marginal distribution at that valid time, so a mixture
whose shifted branch has probability p at a lead scores the same as 51 paths whose end times are
drawn from the same survival curve. A path is one draw of the base level (a quantile of `M1`'s forecast) plus the
shift times the survival indicator (1 until the path's end time, 0 after it). The base level and the
end time are drawn independently: the 51 end times are the conditional inverse of the survival
curve at evenly spaced levels, and the 51 base levels are the same evenly spaced levels in a fixed
order (a stride permutation) that is uncorrelated with the end-time order. The tolerances come
from `known_answer_gaps`, which scores the construction against a Gaussian mixture whose quantiles
are known exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import e00738_score as score_mod
import e00738_survival as surv
import numpy as np
import polars as pl
from scipy import stats

N_PATHS: Final[int] = 51
BASE_STRIDE: Final[int] = 19
"""The stride of the permutation of the base levels. It is coprime with 51 (3 times 17)."""
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
SHARE_TOLERANCE: Final[float] = 0.02
QUANTILE_TOLERANCE_POINTS: Final[float] = 0.5
CRPS_TOLERANCE_POINTS: Final[float] = 0.10
HORIZON_DAYS: Final[float] = 10.0
T_MAX_DAYS: Final[float] = 1.0e4
MAX_CHECK_ROWS: Final[int] = 2000
STEPS_PER_ROW: Final[int] = 48


def isf(*, curve: surv.SurvivalFunction, s: np.ndarray) -> np.ndarray:
    """Return the time t (days) with S(t) = s, by bisection (t = 0 where s >= 1, and the largest time where s <= 0)."""
    s = np.asarray(s, dtype=float)
    lo = np.zeros_like(s)
    hi = np.full_like(s, T_MAX_DAYS)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = curve.sf(mid) > s
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return np.where(s >= 1.0, 0.0, 0.5 * (lo + hi))


def end_times(*, curve: surv.SurvivalFunction, age_days: float, n: int = N_PATHS) -> np.ndarray:
    """Return `n` end times (days after the shift began) conditioned on the shift being on at `age_days`.

    End time `k` is the survival curve's inverse at `S(age) * (k - 0.5) / n` for k = 1 to n, so the
    times run from the longest (k = 1 is the smallest level) to the shortest.

    Args:
        curve: The survival function.
        age_days: The age of the shift at the origin.
        n: The number of paths.

    Returns:
        An array (n,) of end times, all at least `age_days`.
    """
    levels = (np.arange(1, n + 1) - 0.5) / n
    s_a = float(curve.sf(np.array([age_days]))[0])
    return np.maximum(isf(curve=curve, s=s_a * levels), age_days)


def base_levels(*, n: int = N_PATHS) -> np.ndarray:
    """Return the `n` base quantile levels in the order of the paths.

    The levels are the evenly spaced `(j + 0.5) / n`, in the order `(BASE_STRIDE * k) mod n`, so the
    level of path `k` is not tied to the path's end time (the end times run from the longest to the
    shortest with `k`).

    Args:
        n: The number of paths (coprime with `BASE_STRIDE`).

    Returns:
        An array (n,) of levels in 0 to 1.
    """
    assert math.gcd(BASE_STRIDE, n) == 1, "the stride must be coprime with the number of paths"
    return ((BASE_STRIDE * np.arange(n)) % n + 0.5) / n


def interp_quantile(*, q: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Return the base forecast's value at levels `tau`, by linear interpolation of its 11 quantiles.

    Args:
        q: The 11 quantiles (..., 11) at `common`-grid levels (0.01 to 0.99).
        tau: Levels in 0 to 1, shape (n,).

    Returns:
        An array (..., n); levels outside 0.01 to 0.99 take the end values.
    """
    flat = q.reshape(-1, 11)
    out = np.stack([np.interp(tau, LEVELS, row) for row in flat])
    return out.reshape(*q.shape[:-1], len(tau))


def paths_at_leads(
    *,
    base_q: np.ndarray,
    depth: float,
    age_days: float,
    lead_days: np.ndarray,
    curve: surv.SurvivalFunction,
    n: int = N_PATHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 51 paths at each lead, and the share of paths whose shift has ended by each lead.

    Args:
        base_q: The base forecast's 11 quantiles at each lead, (n_lead, 11).
        depth: The signed depth over capacity.
        age_days: The age of the shift at the origin.
        lead_days: The leads in days after the origin, (n_lead,).
        curve: The survival function.
        n: The number of paths.

    Returns:
        The paths (n_lead, n) and the ended share (n_lead,).
    """
    t_end = end_times(curve=curve, age_days=age_days, n=n)
    base = interp_quantile(q=base_q, tau=base_levels(n=n))
    on = (age_days + np.asarray(lead_days))[:, None] < t_end[None, :]
    return base + depth * on, 1.0 - on.mean(axis=1)


def fair_crps(*, paths: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the ensemble CRPS with the finite-sample correction, per row.

    Args:
        paths: Members, (rows, n).
        y: Outcomes, (rows,).

    Returns:
        mean |x_i - y| - sum_ij |x_i - x_j| / (2 n (n - 1)).
    """
    n = paths.shape[1]
    spread = np.abs(paths[:, :, None] - paths[:, None, :]).sum(axis=(1, 2)) / (2.0 * n * (n - 1))
    return np.abs(paths - y[:, None]).mean(axis=1) - spread


def energy_score(*, paths: np.ndarray, y: np.ndarray) -> float:
    """Return the energy score of paths (n, d) against an outcome vector (d,)."""
    n = paths.shape[0]
    a = np.linalg.norm(paths - y[None, :], axis=1).mean()
    b = np.linalg.norm(paths[:, None, :] - paths[None, :, :], axis=2).sum() / (2.0 * n * (n - 1))
    return float(a - b)


def variogram_score(*, paths: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    """Return the variogram score of order `p` of paths (n, d) against an outcome vector (d,)."""
    obs = np.abs(y[:, None] - y[None, :]) ** p
    sim = (np.abs(paths[:, :, None] - paths[:, None, :]) ** p).mean(axis=0)
    return float(((obs - sim) ** 2).sum())


def check_origin(
    *,
    base_q: np.ndarray,
    arm_q: np.ndarray,
    depth: float,
    age_days: float,
    lead_days: np.ndarray,
    curve: surv.SurvivalFunction,
    p_arm: np.ndarray,
) -> dict[str, float]:
    """Return the three quantities that the member-path check asserts, for one origin.

    Args:
        base_q: `M1`'s 11 quantiles at each lead, (n_lead, 11), over capacity.
        arm_q: The mixture arm's stored 11 quantiles at the same leads.
        depth: The arm's signed depth over capacity.
        age_days: A1's age.
        lead_days: The leads in days after the origin.
        curve: The fitted survival curve.
        p_arm: The arm's survival ratio at each lead.

    Returns:
        `ended_gap`: the gap between the share of ended paths and 1 - p at the last lead below 10
        days; `quantile_gap`: the mean absolute gap in points between the paths' 11 quantiles and the
        stored ones, over leads.
    """
    paths, ended = paths_at_leads(
        base_q=base_q, depth=depth, age_days=age_days, lead_days=lead_days, curve=curve
    )
    emp = np.quantile(paths, LEVELS, axis=1).T
    inside = np.nonzero(lead_days <= HORIZON_DAYS)[0]
    last = inside[-1]
    return {
        "ended_gap": float(abs(ended[last] - (1.0 - p_arm[last]))),
        "quantile_gap": float(np.abs(emp - arm_q).mean() * 100.0),
    }


def passes(*, ended_gap: float, quantile_gap: float, crps_gap: float) -> bool:
    """Return whether the three assertions hold."""
    return (
        ended_gap <= SHARE_TOLERANCE
        and quantile_gap <= QUANTILE_TOLERANCE_POINTS
        and crps_gap <= CRPS_TOLERANCE_POINTS
    )


def gaussian_mixture_quantiles(*, p: np.ndarray, depth: float, sd: float) -> np.ndarray:
    """Return the 11 quantiles of `p N(depth, sd) + (1 - p) N(0, sd)` for each `p`, by bisection.

    Args:
        p: The weight of the shifted branch, (n,).
        depth: The shift (the mean of the shifted branch).
        sd: The standard deviation of both branches.

    Returns:
        An array (n, 11).
    """
    p = np.asarray(p, dtype=float)[:, None]
    lo = np.full((p.shape[0], 11), min(0.0, depth) - 10.0 * sd)
    hi = np.full((p.shape[0], 11), max(0.0, depth) + 10.0 * sd)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = p * stats.norm.cdf(mid, loc=depth, scale=sd) + (1.0 - p) * stats.norm.cdf(
            mid, scale=sd
        )
        below = cdf < LEVELS[None, :]
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def known_answer_gaps(
    *,
    sd: float,
    depth: float,
    age_days: float,
    curve: surv.SurvivalFunction,
    lead_day: int,
    seed: int = 0,
) -> dict[str, float]:
    """Return the three gaps of the check on a case whose answer is known exactly.

    The base forecast is `N(0, sd)`, the shifted branch is `N(depth, sd)`, and the stored forecast
    is the exact mixture quantiles at the survival ratio of `curve`. A construction that is right
    returns gaps that come only from drawing 51 paths and from interpolating 11 quantiles, so these
    gaps set the tolerances. Values are shares of capacity (multiply by 100 for points).

    Args:
        sd: The standard deviation of the base forecast.
        depth: The signed depth.
        age_days: A1's age.
        curve: The survival curve that both the paths and the stored forecast follow.
        lead_day: The zero-based lead day (the 48 half-hours of that day are checked).
        seed: The seed of the outcomes drawn from the mixture.

    Returns:
        `ended_gap`, `quantile_gap` and `crps_gap` as `check_origin` and `check_frames` define them.
    """
    rng = np.random.default_rng(seed)
    leads = lead_day + np.arange(STEPS_PER_ROW) / STEPS_PER_ROW
    p = surv.ratio(curve=curve, age_days=np.full(STEPS_PER_ROW, age_days), lead_days=leads)
    base_q = np.tile(stats.norm.ppf(LEVELS, scale=sd), (STEPS_PER_ROW, 1))
    arm_q = gaussian_mixture_quantiles(p=p, depth=depth, sd=sd)
    got = check_origin(
        base_q=base_q,
        arm_q=arm_q,
        depth=depth,
        age_days=age_days,
        lead_days=leads,
        curve=curve,
        p_arm=p,
    )
    paths, _ = paths_at_leads(
        base_q=base_q, depth=depth, age_days=age_days, lead_days=leads, curve=curve
    )
    shifted = rng.random(STEPS_PER_ROW) < p
    y = rng.normal(0.0, sd, STEPS_PER_ROW) + depth * shifted
    emp = np.quantile(paths, LEVELS, axis=1).T
    crps_gap = float(
        np.abs(score_mod.crps_points(y=y, q=emp) - score_mod.crps_points(y=y, q=arm_q)).mean()
    )
    return {**got, "crps_gap": crps_gap}


@dataclass(frozen=True)
class WeibullSurvival:
    """A Weibull survival function of the duration in days (the known-answer cases use it)."""

    shape: float
    scale_days: float

    def sf(self, t: np.ndarray) -> np.ndarray:
        """Return P(duration > t) for `t` in days."""
        return np.exp(-((np.asarray(t, dtype=float) / self.scale_days) ** self.shape))


def tolerance_study(*, n_cases: int = 400, seed: int = 1) -> dict[str, dict[str, float]]:
    """Return the distribution of the three gaps over random known-answer cases.

    The cases draw the base standard deviation (1 to 5 points of capacity), the depth (3 to 20
    points, either sign), A1's age (2 to 14 days), the Weibull shape (0.7 to 1.5) and scale (5 to 40
    days), and the lead day (1 to 9). The tolerances sit above the mean of each gap over the cases,
    because the check reports a mean over thousands of rows, and below the gap of a wrong
    construction.

    Args:
        n_cases: The number of cases.
        seed: The seed of the draw.

    Returns:
        For each gap, the `mean`, `p95`, and `max` over the cases (points of capacity for the
        quantile and CRPS gaps, a share for the ended gap).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        got = known_answer_gaps(
            sd=rng.uniform(0.01, 0.05),
            depth=rng.choice([-1.0, 1.0]) * rng.uniform(0.03, 0.20),
            age_days=rng.uniform(2.0, 14.0),
            curve=WeibullSurvival(shape=rng.uniform(0.7, 1.5), scale_days=rng.uniform(5.0, 40.0)),
            lead_day=int(rng.integers(1, 10)),
            seed=i,
        )
        rows.append([got["ended_gap"], got["quantile_gap"], got["crps_gap"]])
    arr = np.array(rows)
    return {
        name: {
            "mean": float(arr[:, j].mean()),
            "p95": float(np.percentile(arr[:, j], 95)),
            "max": float(arr[:, j].max()),
        }
        for j, name in enumerate(("ended_gap", "quantile_gap", "crps_gap"))
    }


def check_frames(
    *,
    arm: pl.DataFrame,
    base: pl.DataFrame,
    diag: pl.DataFrame,
    curve: surv.SurvivalFunction,
    max_rows: int = MAX_CHECK_ROWS,
) -> dict[str, float]:
    """Run the member-path check on the rows of a mixture arm where A1 sees a shift.

    One row is one (Primary, origin, lead day): 48 half-hourly valid times. The arm's stored
    quantiles, `M1`'s stored quantiles, and the actuals come from the panel files, and A1's age and
    the signed depth from the diagnostics file of the arm.

    Args:
        arm: The arm's forecast frame of a panel, sorted by the keys.
        base: `M1`'s forecast frame of the same panel, with the same keys.
        diag: The arm's diagnostics frame (`time_series_id`, `origin_us`, `lead_day`, `age`,
            `depth_share`).
        curve: The survival curve that the arm was fitted with.
        max_rows: At most this many rows are checked, evenly spaced.

    Returns:
        The means over the checked rows: `ended_gap`, `quantile_gap`, `crps_gap` (points of
        capacity), the report-only `energy_score` and `variogram_score` of the 51 paths against the
        actual half-hours of the day, the report-only `ensemble_crps` (the fair CRPS of the 51
        paths) and `tail_weighted_crps` (the 11-level formula of the scorer on the paths' own 11
        quantiles), the largest `ended_gap`, and the number of rows checked. The `crps_gap` compares
        the 11-level formula on the paths' quantiles with the same formula on the stored quantiles,
        so it measures the paths and not the difference between two formulas.
    """
    qcols = [f"q{n}" for n in ("01", "05", "10", "25", "40", "50", "60", "75", "90", "95", "99")]
    assert arm.height == base.height
    assert arm.height % STEPS_PER_ROW == 0
    n = arm.height // STEPS_PER_ROW

    def blocks(frame: pl.DataFrame, col: str) -> np.ndarray:
        return frame[col].to_numpy().reshape(n, STEPS_PER_ROW)

    tsid = blocks(arm, "time_series_id")[:, 0]
    origin_us = (
        arm["power_fcst_init_time"].dt.epoch("us").to_numpy().reshape(n, STEPS_PER_ROW)[:, 0]
    )
    lead = blocks(arm, "lead_day")[:, 0]
    y = blocks(arm, "actual").astype(float)
    q_arm = arm.select(qcols).to_numpy().reshape(n, STEPS_PER_ROW, 11).astype(float)
    q_base = base.select(qcols).to_numpy().reshape(n, STEPS_PER_ROW, 11).astype(float)
    found = {
        (int(t), int(o), int(k)): (float(a), float(d))
        for t, o, k, a, d in zip(
            diag["time_series_id"].to_list(),
            diag["origin_us"].to_list(),
            diag["lead_day"].to_list(),
            diag["age"].to_list(),
            diag["depth_share"].to_list(),
            strict=True,
        )
    }
    rows = [
        i
        for i in range(n)
        if found.get((int(tsid[i]), int(origin_us[i]), int(lead[i])), (0.0, 0.0))[0] > 0
        and np.isfinite(y[i]).all()
    ]
    if len(rows) > max_rows:
        rows = [rows[j] for j in np.linspace(0, len(rows) - 1, max_rows).astype(int)]
    assert rows, "no row has a shift that A1 sees: nothing to check"
    acc: dict[str, list[float]] = {
        k: []
        for k in (
            "ended_gap",
            "quantile_gap",
            "crps_gap",
            "ensemble_crps",
            "tail_weighted_crps",
            "energy_score",
            "variogram_score",
        )
    }
    for i in rows:
        age, depth = found[(int(tsid[i]), int(origin_us[i]), int(lead[i]))]
        leads = (STEPS_PER_ROW * int(lead[i]) + np.arange(STEPS_PER_ROW)) / STEPS_PER_ROW
        p = surv.ratio(curve=curve, age_days=np.full(STEPS_PER_ROW, age), lead_days=leads)
        got = check_origin(
            base_q=q_base[i],
            arm_q=q_arm[i],
            depth=depth,
            age_days=age,
            lead_days=leads,
            curve=curve,
            p_arm=p,
        )
        paths, _ = paths_at_leads(
            base_q=q_base[i], depth=depth, age_days=age, lead_days=leads, curve=curve
        )
        acc["ended_gap"].append(got["ended_gap"])
        acc["quantile_gap"].append(got["quantile_gap"])
        emp = np.quantile(paths, LEVELS, axis=1).T
        tail_weighted = score_mod.crps_points(y=y[i], q=emp)
        stored = score_mod.crps_points(y=y[i], q=q_arm[i])
        acc["crps_gap"].append(float(np.abs(tail_weighted - stored).mean()))
        acc["ensemble_crps"].append(float(fair_crps(paths=paths, y=y[i]).mean()) * 100.0)
        acc["tail_weighted_crps"].append(float(tail_weighted.mean()))
        acc["energy_score"].append(energy_score(paths=paths.T, y=y[i]))
        acc["variogram_score"].append(variogram_score(paths=paths.T, y=y[i]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["ended_gap_max"] = float(np.max(acc["ended_gap"]))
    out["rows"] = float(len(rows))
    return out
