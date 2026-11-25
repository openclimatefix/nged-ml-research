# ruff: noqa: E501
"""Experiment 720, rung 0e: the bar tables of the addendum, as functions that return pass or fail.

Every function takes numbers (never a generator series) and returns a dict with the measured value,
the bar, and `passed`. The verdict reads these, so the thresholds live here and nowhere else.
"""

from typing import Any, Final

import e00720_fold as fold
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

    Returns:
        A dictionary with `periods` (each period's `v3_bar` result) and `passed` (true when there is
        at least one period and every period passed).
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

    Returns:
        A dictionary with `gain_over_hard`, `lower` (the 2.5th percentile of the bootstrap), `soft`,
        `hard`, `minus_anchored` (only when `anchored` is given), and `passed`.
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


def anchored_two_lobe_crps(
    *, rho: np.ndarray, magnitude: np.ndarray, x_true: np.ndarray, width: float
) -> np.ndarray:
    """Return the per-row signed CRPS of the anchored read-out converted to the two-lobe form.

    The lobes sit at minus and plus the observed magnitude, with weights `rho` and `1 - rho` and a common
    standard deviation `width` (the hindsight model's own kernel width), so the two read-outs differ only in
    how they produce the sign probability.
    """
    rho = np.clip(np.nan_to_num(rho, nan=0.5), 1e-9, 1.0 - 1e-9)
    pi = np.stack([rho, 1.0 - rho], axis=1)
    mu = np.stack([-magnitude, magnitude], axis=1)
    sigma = np.full(mu.shape, width)
    q = fold.signed_quantiles(pi=pi, mu=mu, sigma=sigma)
    return hmm.signed_crps(quantiles=q, truth=np.nan_to_num(x_true))


def both_versions(
    *,
    post: hmm.Posterior,
    spec: hmm.HmmSpec,
    x_true: np.ndarray,
    magnitude: np.ndarray,
    anchored_rho: np.ndarray | None = None,
    seed_noise: float | None = None,
    dead_band: float = 2.0,
) -> dict[str, dict[str, Any]]:
    """Return the sign-probability and signed-CRPS bars for the smoothed and the filtered posterior, and their gap.

    Args:
        post: The hindsight model's posterior over the rows.
        spec: Its spec.
        x_true: The signed truth (NaN where not scored).
        magnitude: The observed magnitude `sqrt(m² - Q²)` of each row. The hard-sign reference uses it, so
            the soft distribution is not asked to beat a reference that knows the true magnitude.
        anchored_rho: The anchored read-out's `P(X < 0)` on the same rows, for the second half of the CRPS bar.
        seed_noise: The measured noise of a second seed. While it is `None` the second half is pending, the
            bar's `passed` reads the first half alone, and `second_half_pending` is true.
        dead_band: Rows with `|x_true|` under this are not scored.

    Returns:
        For each version, the sign bar (on the posterior's `P(X < 0)`) and the CRPS bar, plus the sign bar
        computed on the mixture's own `P(X < 0)` (`sign_mixture`), and the filtered-minus-smoothed gap.
    """
    ok = np.isfinite(x_true) & (np.abs(x_true) >= dead_band)
    out: dict[str, dict[str, Any]] = {}
    for version in ("smoothed", "filtered"):
        pneg = hmm.summarise(post=post, spec=spec, version=version)["p_negative"]
        pi, mu, sigma = hmm.two_lobe_mixture(post=post, spec=spec, version=version)
        q = hmm.truncated_signed_quantiles(pi=pi, mu=mu, sigma=sigma)
        soft = hmm.signed_crps(quantiles=q, truth=np.nan_to_num(x_true))[ok]
        sgn = np.where(pneg > 0.5, -1.0, 1.0)
        hard = np.abs(sgn * magnitude - np.nan_to_num(x_true))[ok]
        anchored = None
        if anchored_rho is not None and seed_noise is not None:
            anchored = anchored_two_lobe_crps(
                rho=anchored_rho, magnitude=magnitude, x_true=x_true, width=float(np.median(sigma))
            )[ok]
        outcome = (x_true[ok] < 0).astype(float)
        out[version] = {
            "sign": sign_probability_bar(prob=pneg[ok], outcome=outcome),
            "sign_mixture": sign_probability_bar(
                prob=pi[:, 0][ok], outcome=outcome
            ),  # truncated lobes: P(X < 0) is pi_0
            "crps": signed_crps_bar(
                soft=soft, hard=hard, anchored=anchored, seed_noise=seed_noise or 0.0
            ),
            "second_half_pending": seed_noise is None,
        }
    out["gap"] = {
        "reliability": out["filtered"]["sign"]["reliability"]
        - out["smoothed"]["sign"]["reliability"],
        "crps": out["filtered"]["crps"]["soft"] - out["smoothed"]["crps"]["soft"],
    }
    return out
