# ruff: noqa: E501
"""Experiment 720, rung 0e: the bar tables of the addendum, as functions that return pass or fail.

Every function takes numbers (never a generator series) and returns a dict with the measured value,
the bar, and `passed`. The verdict reads these, so the thresholds live here and nowhere else.
"""

from typing import Any, Final

import e00720_hmm as hmm
import numpy as np

V1_AUC_GAP: Final[float] = 0.1
V1_FALSE_REVERSE: Final[float] = 0.01
V2_MIN_GAIN_F1: Final[float] = 0.05
K1_F1: Final[float] = 0.769
V3_MAX_FLAGGED_SHARE: Final[float] = 0.03
V3_MAX_ACTIVE_DAYS_PER_MONTH: Final[float] = 1.0
BRIER_RELIABILITY_MAX: Final[float] = 0.01
BIN_GAP_MAX: Final[float] = 0.1
CRPS_SEED_NOISE_DEFAULT: Final[float] = 0.0


def v1_bar(
    *, auc_real: float, auc_shifted: float, false_reverse_forced_open: float
) -> dict[str, Any]:
    """V1: area with the real sky at least 0.1 above the shifted sky, and under 1% false reverse flow at D5."""
    gap = auc_real - auc_shifted
    return {
        "gap": gap,
        "bar_gap": V1_AUC_GAP,
        "false_reverse": false_reverse_forced_open,
        "bar_false": V1_FALSE_REVERSE,
        "passed": bool(gap >= V1_AUC_GAP and false_reverse_forced_open < V1_FALSE_REVERSE),
    }


def paired_bar(*, lower: float, point: float, min_point: float) -> dict[str, Any]:
    """A paired contrast passes when its interval's lower bound is above zero and its point reaches `min_point`."""
    return {
        "lower": lower,
        "point": point,
        "bar_point": min_point,
        "passed": bool(lower > 0.0 and point >= min_point),
    }


def v2_bar(
    *, lower: float, point: float, flagged_share: float, active_days_per_month: float
) -> dict[str, Any]:
    """V2: paired F1 gain over experiment 694's M2 with lower bound above 0 and a gain of 0.05, plus the false-alarm bars."""
    gain = paired_bar(lower=lower, point=point, min_point=V2_MIN_GAIN_F1)
    fa = (
        flagged_share <= V3_MAX_FLAGGED_SHARE
        and active_days_per_month <= V3_MAX_ACTIVE_DAYS_PER_MONTH
    )
    return {
        **gain,
        "flagged_share": flagged_share,
        "active_days_per_month": active_days_per_month,
        "false_alarm_passed": bool(fa),
        "passed": bool(gain["passed"] and fa),
    }


def k1_bar(*, f1: float) -> dict[str, Any]:
    """K1: F1 of at least 0.769 on its own."""
    return {"f1": f1, "bar": K1_F1, "passed": bool(f1 >= K1_F1)}


def v3_bar(
    *, flagged_share: float, active_days_per_month: float, forced_k3_flagged_share: float
) -> dict[str, Any]:
    """V3: each control period flags at most 3% and one active day a month, also with `K` forced to 3."""
    ok = (
        flagged_share <= V3_MAX_FLAGGED_SHARE
        and active_days_per_month <= V3_MAX_ACTIVE_DAYS_PER_MONTH
        and forced_k3_flagged_share <= V3_MAX_FLAGGED_SHARE
    )
    return {
        "flagged_share": flagged_share,
        "active_days_per_month": active_days_per_month,
        "forced_k3_flagged_share": forced_k3_flagged_share,
        "passed": bool(ok),
    }


def v3_all_periods(*, periods: dict[str, dict[str, float]]) -> dict[str, Any]:
    """V3 per period: every control period passes `v3_bar` with its own shares, for the selected `K` and the forced `K = 3`.

    Args:
        periods: Period label to `flagged_share`, `active_days_per_month`, and `forced_k3_flagged_share`.
    """
    per = {name: v3_bar(**vals) for name, vals in periods.items()}
    return {"periods": per, "passed": bool(per and all(p["passed"] for p in per.values()))}


def sign_probability_bar(*, prob: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    """Reliability term of the Brier score at most 0.01, and no bin of at least 200 rows off the diagonal by over 0.1."""
    r = hmm.brier_reliability(prob=prob, outcome=outcome)
    return {
        **r,
        "passed": bool(
            r["reliability"] <= BRIER_RELIABILITY_MAX and r["max_gap_bin_200"] <= BIN_GAP_MAX
        ),
    }


def signed_crps_bar(
    *,
    soft: np.ndarray,
    hard: np.ndarray,
    anchored: np.ndarray | None = None,
    seed_noise: float = CRPS_SEED_NOISE_DEFAULT,
    rng: np.random.Generator | None = None,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Signed CRPS below the hard-sign version's with a paired interval above zero, and no worse than the anchored read-out by more than `seed_noise`.

    Args:
        soft: Per-row CRPS of the two-lobe distribution.
        hard: Per-row absolute error of the hard-sign version (the same rows).
        anchored: Per-row CRPS of the anchored read-out in two-lobe form, same rows, or `None`.
        seed_noise: The difference a second seed of the treatment moves the score (a measured number).
        rng: Random generator for the bootstrap over rows.
        n_boot: Number of resamples.
    """
    rng = rng or np.random.default_rng(720)
    d = hard - soft
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = float(np.percentile(boots, 2.5))
    out: dict[str, Any] = {
        "gain_over_hard": float(d.mean()),
        "lower": lo,
        "soft": float(soft.mean()),
        "hard": float(hard.mean()),
    }
    ok = lo > 0.0
    if anchored is not None:
        diff = float(soft.mean() - anchored.mean())
        out["minus_anchored"] = diff
        ok = ok and diff <= seed_noise
    out["passed"] = bool(ok)
    return out


def both_versions(
    *, post: hmm.Posterior, spec: hmm.HmmSpec, x_true: np.ndarray, dead_band: float = 2.0
) -> dict[str, dict[str, Any]]:
    """Return the sign-probability and signed-CRPS bars for the smoothed and the filtered posterior, and their gap."""
    ok = np.isfinite(x_true) & (np.abs(x_true) >= dead_band)
    out: dict[str, dict[str, Any]] = {}
    for version in ("smoothed", "filtered"):
        pneg = hmm.summarise(post=post, spec=spec, version=version)["p_negative"]
        q = hmm.signed_quantiles_of(post=post, spec=spec, version=version)
        soft = hmm.signed_crps(quantiles=q, truth=np.nan_to_num(x_true))[ok]
        sgn = np.where(pneg > 0.5, -1.0, 1.0)
        mag = np.abs(np.nan_to_num(x_true))  # hindsight magnitude is nearly observed
        hard = np.abs(sgn * mag - np.nan_to_num(x_true))[ok]
        out[version] = {
            "sign": sign_probability_bar(prob=pneg[ok], outcome=(x_true[ok] < 0).astype(float)),
            "crps": signed_crps_bar(soft=soft, hard=hard),
        }
    out["gap"] = {
        "reliability": out["filtered"]["sign"]["reliability"]
        - out["smoothed"]["sign"]["reliability"],
        "crps": out["filtered"]["crps"]["soft"] - out["smoothed"]["crps"]["soft"],
    }
    return out
