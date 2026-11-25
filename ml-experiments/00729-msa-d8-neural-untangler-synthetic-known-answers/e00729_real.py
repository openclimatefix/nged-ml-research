# ruff: noqa: E501
"""Experiment 729: the real-data checks R1, K1, K2, R2, and R3, and the harm check H2.

Every function takes arrays and returns aggregates. None returns or prints a date, an engine size, a
series identifier, or a per-day value. K1 and K2 read the metered engine record, which only this
module's callers load; the simulators never see it.
"""

from collections.abc import Sequence
from typing import Final

import e00729_baselines as base
import e00729_core as core
import e00729_dist as dist
import e00729_score as score
import e00729_sim as sim
import numpy as np

SIGN_CHANGE_NEIGHBOUR: Final = 1
"""A sign change is "next to" a near-zero half-hour when one lies within this many half-hours."""
NEAR_ZERO_POINTS: Final = core.GATE_MIN_VALUE_MVA / core.CAPACITY_MVA * core.POINTS


def real_segments(
    *, cal: sim.Calendar, use: np.ndarray, reading: np.ndarray, truth: np.ndarray
) -> list[sim.Segment]:
    """Build one pseudo-segment per calendar month from real days.

    Args:
        cal: The calendar.
        use: The days to include.
        reading: The reading in points to feed the networks, on the calendar axis.
        truth: The answer in points (NaN where there is none), on the calendar axis.

    Returns:
        The segments, in month order.
    """
    months = sorted({str(d)[:7] for d in cal.dates[use]})
    out = []
    for m in months:
        in_month = np.array([str(d)[:7] == m for d in cal.dates]) & use
        out.append(sim.real_segment(cal=cal, month=m, use=in_month, reading=reading, truth=truth))
    return out


def real_rows(
    *, cal: sim.Calendar, use: np.ndarray, reading: np.ndarray, truth: np.ndarray
) -> score.Rows:
    """Build scored rows from real days (see `real_segments`)."""
    segments = real_segments(cal=cal, use=use, reading=reading, truth=truth)
    return score.rows_from_segments(segments=segments, holidays=cal.holidays)


def r1_false_alarms(
    *, d: score.RowMetrics, p0f_metrics: score.RowMetrics, p0_metrics: score.RowMetrics
) -> dict[str, float | bool]:
    """R1 and H2: arm D against P0f and P0 on the real held-out base days (all import)."""
    keep = np.ones(len(d.crps), dtype=bool)
    return score.harm_check(
        d=d, p0f_metrics=p0f_metrics, p0_metrics=p0_metrics, keep=keep, real=True
    )


def folded_record_days(*, reading: np.ndarray, record: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the K1 reading and truth: the record subtracted from D8's reading, then folded.

    Args:
        reading: D8's reading in points on the record's days (gross demand with no engine in it).
        record: The metered engine record in points on the same half-hours.

    Returns:
        The folded reading `sqrt(X^2 + Qhat^2)` and the signed truth `X = reading - record`.
    """
    x = reading - record
    return np.sqrt(x**2 + core.QHAT_POINTS**2), x


def gaussian_mixture_median(*, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the median of an equal mixture of Gaussians over the first axis (seeds), by bisection.

    Args:
        mu: Means, `(S, N)`.
        sigma: Spreads, `(S, N)`.

    Returns:
        The median of each of the `N` mixtures.
    """
    lo = (mu - 10.0 * sigma).min(axis=0)
    hi = (mu + 10.0 * sigma).max(axis=0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = dist.norm_cdf(z=(mid[None, :] - mu) / sigma).mean(axis=0)
        below = cdf < 0.5
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def f1_score(*, truth: np.ndarray, flag: np.ndarray) -> float:
    """Return the half-hour F1 of a boolean flag against a boolean truth (0 when both empty)."""
    tp = int((truth & flag).sum())
    fp = int((~truth & flag).sum())
    fn = int((truth & ~flag).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def k1_engine_head(
    *, head_median: np.ndarray, record: np.ndarray, running: np.ndarray, m2_flag: np.ndarray
) -> dict[str, float | bool | str]:
    """K1: the engine head against the metered record, with M2's F1 as the second bar.

    Half-hours where the record has no row (NaN) are left out of every figure. The result holds the
    error ratio and the F1 scores only: the raw mean of the record and the count of running
    half-hours would let a reader estimate the engine's output, so neither is returned.

    Args:
        head_median: Arm G2's engine-head median (equal mixture of seeds 0 to 2) in points.
        record: The record in points (NaN where the record has no row).
        running: The record's own running label (`units >= 1`).
        m2_flag: M2's "running" flag on the same injected half-hours.

    Returns:
        The error ratio, the F1 scores, each bar, and `status` ("pass", "fail", or "void").
    """
    keep = np.isfinite(record)
    head_median, record, running, m2_flag = (
        a[keep] for a in (head_median, record, running, m2_flag)
    )
    n_running = int(running.sum())
    mae_head = float(np.mean(np.abs(head_median - record)))
    mae_zero = float(np.mean(np.abs(record)))
    ratio = mae_head / mae_zero if mae_zero > 0 else float("inf")
    flag = head_median > core.K1_RUNNING_POINTS
    f1_head = f1_score(truth=running, flag=flag)
    f1_m2 = f1_score(truth=running, flag=m2_flag)
    bar_mae = ratio <= core.K1_MAE_RATIO
    bar_f1 = f1_head >= f1_m2
    status = (
        "void" if n_running < core.K1_MIN_RUNNING else ("pass" if bar_mae and bar_f1 else "fail")
    )
    return {
        "mae_ratio": ratio, "f1_head": f1_head, "f1_m2": f1_m2, "bar_mae": bar_mae,
        "bar_f1": bar_f1, "status": status,
    }  # fmt: skip


def k2_placebo(*, p_neg: np.ndarray, head_median: np.ndarray) -> dict[str, float | bool | str]:
    """K2: D8's unchanged reading on the record's days; no reverse flow and no engine expected."""
    sign_share = float(np.mean(p_neg > 0.5))
    engine_share = float(np.mean(head_median > core.K1_RUNNING_POINTS))
    ok = sign_share <= core.K2_SIGN_SHARE and engine_share <= core.K2_ENGINE_SHARE
    return {
        "sign_share": sign_share, "engine_share": engine_share, "status": "pass" if ok else "fail",
    }  # fmt: skip


def refold_share(*, median: np.ndarray, reading: np.ndarray) -> float:
    """R2: the share of half-hours on which the folded median lies within 2 points of the reading."""
    folded = np.sqrt(median**2 + core.QHAT_POINTS**2)
    return float(np.mean(np.abs(folded - reading) <= core.R2_TOLERANCE_POINTS))


def r2_by_group(*, median: np.ndarray, reading: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """R2 for each group label (an era, or the K2 days)."""
    return {
        str(g): refold_share(median=median[groups == g], reading=reading[groups == g])
        for g in np.unique(groups)
    }


def r3_sign_by_group(*, p_neg: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """R3: the share of half-hours with P(X < 0) above 0.5 in each group (era, era and hour)."""
    return {str(g): float(np.mean(p_neg[groups == g] > 0.5)) for g in np.unique(groups)}


def near_zero_neighbour_share(*, p_neg: np.ndarray, reading: np.ndarray) -> float:
    """R3: the share of predicted sign changes that sit next to a half-hour below 1.0 MVA.

    Args:
        p_neg: P(X < 0) of consecutive half-hours of one series (NaN breaks the series).
        reading: The reading of the same half-hours in points.

    Returns:
        The share, or NaN when there is no predicted sign change.
    """
    sign = p_neg > 0.5
    change = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    if change.size == 0:
        return float("nan")
    near = reading < NEAR_ZERO_POINTS
    k = SIGN_CHANGE_NEIGHBOUR
    hits = [bool(near[max(c - k - 1, 0) : c + k + 1].any()) for c in change]
    return float(np.mean(hits))


def bsp_residual_level(*, bsp: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """R3: the BSP meter's level on flagged half-hours against the others, minus the day's median.

    The residual is the half-hour's BSP value minus the median of the same day's BSP values. The
    level test is too noisy to show a few megawatts, and the report says so.

    Args:
        bsp: BSP values in points, `(days, 48)`, NaN where missing.
        flagged: Half-hours with P(X < 0) above 0.5, `(days, 48)`.

    Returns:
        The mean residual on flagged half-hours, on the others, and the count flagged.
    """
    resid = bsp - np.nanmedian(bsp, axis=1, keepdims=True)
    ok = np.isfinite(resid)
    on = flagged & ok
    off = ~flagged & ok
    return {
        "flagged_mean": float(resid[on].mean()) if on.any() else float("nan"),
        "other_mean": float(resid[off].mean()) if off.any() else float("nan"),
        "n_flagged": float(on.sum()),
    }


def era_groups(*, dates: np.ndarray, day_dates: np.ndarray, day_eras: Sequence[str]) -> np.ndarray:
    """Return the era label of each row's date (rows hold dates as days since 1970-01-01)."""
    lookup = dict(
        zip(day_dates.astype("datetime64[D]").astype(int).tolist(), day_eras, strict=True)
    )
    return np.array([lookup[int(d)] for d in dates])


def b_star_on_rows(
    *, rows: score.Rows, model: base.Climatology | base.Regression, pi: float
) -> dist.Lobes:
    """Return the sign-prior baseline B1 or B2 (whichever `model` is) on real rows.

    Args:
        rows: The real rows.
        model: The fitted climatology (B1) or regression (B2), whichever the validation rule chose.
        pi: The prior share of injected half-hours; the stand-in is the test mix's injected share.

    Returns:
        The two-lobe baseline.
    """
    pb = model.prob_below(r=rows.features.r, f=rows.features)
    return base.sign_prior_lobes(f=rows.features, prob_below=pb, pi=pi)
