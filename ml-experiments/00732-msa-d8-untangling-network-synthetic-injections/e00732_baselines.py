# ruff: noqa: E501
"""Experiment 732: the baselines B-PV1, B-PV2 (solar output without a solar head) and B-T (step rule).

This module needs no torch. B-PV1 and B-PV2 read stored probabilities and demand outputs of experiment
730, and B-T is a logistic regression fitted on the validation overlay with Newton steps.
"""

from collections.abc import Sequence

import e00732_paths  # noqa: F401

# isort: split
import e00732_core as core
import numpy as np

BT_FEATURES: tuple[str, ...] = ("abs_dr", "abs_dr_over_median", "abs_dbsp", "abs_diff", "cosz")


def signed_reading_estimate(*, p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return X-hat: -M where P(X<0) is above 0.5 and +M otherwise, with M = sqrt(max(R^2 - Q-hat^2, 0)).

    Args:
        p: The seed-mean P(X<0) of each row.
        r: The reading R of each row, in points.

    Returns:
        The signed reading estimate, in points.
    """
    m = np.sqrt(np.maximum(r**2 - core.Q_HAT_POINTS**2, 0.0))
    return np.where(p > 0.5, -m, m)


def solar_from_demand(
    *, demand: np.ndarray, p: np.ndarray, r: np.ndarray, cosz: np.ndarray
) -> np.ndarray:
    """Return max(0, demand - X-hat) on sun-up half-hours and 0 at night, for one seed (or a seed mean).

    Args:
        demand: A demand estimate, in points (experiment 730's demand output, or D8's climatology median).
        p: P(X<0) of the same rows.
        r: The reading R.
        cosz: The cosine of the solar zenith angle.

    Returns:
        The solar estimate in points.
    """
    est = np.maximum(demand - signed_reading_estimate(p=p, r=r), 0.0)
    return np.where(cosz > core.SUN_UP_COSZ, est, 0.0)


def b_pv2(
    *,
    demand_by_seed: Sequence[np.ndarray],
    p_by_seed: Sequence[np.ndarray],
    r: np.ndarray,
    cosz: np.ndarray,
) -> np.ndarray:
    """Return B-PV2: the rule per seed of SD*, averaged over seeds 0 to 2 (it fails on a seed count that is not 3)."""
    if len(demand_by_seed) != len(core.SEEDS) or len(p_by_seed) != len(core.SEEDS):
        msg = "B-PV2 needs the three seeds of SD* (a missing seed makes the baseline void)"
        raise ValueError(msg)
    return np.mean(
        [
            solar_from_demand(demand=d, p=p, r=r, cosz=cosz)
            for d, p in zip(demand_by_seed, p_by_seed, strict=True)
        ],
        axis=0,
    )


def b_pv1(
    *, demand_median: np.ndarray, p_by_seed: Sequence[np.ndarray], r: np.ndarray, cosz: np.ndarray
) -> np.ndarray:
    """Return B-PV1: the same rule with the climatology median in place of the demand output, on S's P(X<0)."""
    if len(p_by_seed) != len(core.SEEDS):
        msg = "B-PV1 needs the three seeds of S"
        raise ValueError(msg)
    return np.mean(
        [solar_from_demand(demand=demand_median, p=p, r=r, cosz=cosz) for p in p_by_seed], axis=0
    )


def bt_features(
    *, r: np.ndarray, bsp: np.ndarray, cosz: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Return B-T's five inputs per half-hour, `(P, 48, 5)`, from the reading and the BSP step of each window.

    The inputs are the absolute half-hour change of the reading, that change over the window's median
    absolute change, the absolute half-hour change of the BSP meter, the absolute difference between
    the two changes, and the cosine of the solar zenith angle. The change at a half-hour is the reading
    minus the reading of the half-hour before, and is zero at the first half-hour and where either
    half-hour is invalid.

    Args:
        r: The reading in points, `(P, 48)`.
        bsp: The BSP meter's half-hour change in points, `(P, 48)` (zero where it has no data).
        cosz: The cosine of the solar zenith angle, `(P, 48)`.
        valid: Valid half-hours, `(P, 48)` bool.

    Returns:
        The feature array.
    """
    dr = np.zeros_like(r, dtype=np.float64)
    dr[:, 1:] = np.where(valid[:, 1:] & valid[:, :-1], r[:, 1:] - r[:, :-1], 0.0)
    a = np.abs(dr)
    med = np.array(
        [np.median(row[v]) if v.any() else 0.0 for row, v in zip(a, valid, strict=True)]
    )[:, None]
    scaled = a / np.maximum(med, 1e-3)
    db = np.abs(np.nan_to_num(bsp, nan=0.0))
    return np.stack([a, scaled, db, np.abs(a - db), cosz], axis=-1)


def fit_logistic(
    *, features: np.ndarray, y: np.ndarray, ridge: float = 1e-3, steps: int = 50
) -> np.ndarray:
    """Return the weights (bias first) of a ridge-penalised logistic regression fitted with Newton steps.

    Args:
        features: `(N, F)`.
        y: 0 or 1 labels `(N,)`.
        ridge: The penalty on the weights (not the bias).
        steps: The Newton steps.

    Returns:
        The `F + 1` weights.
    """
    if features.shape[0] == 0 or not (0.0 < float(np.mean(y)) < 1.0):
        msg = "B-T needs rows with and without a step"
        raise ValueError(msg)
    mu, sd = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-9)
    z = np.column_stack([np.ones(len(y)), (features - mu) / sd])
    w = np.zeros(z.shape[1])
    pen = np.full(z.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
        grad = z.T @ (p - y) + pen * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if float(np.abs(step).max()) < 1e-8:
            break
    return np.concatenate([w[:1] - (w[1:] * mu / sd).sum(), w[1:] / sd])


def predict_logistic(*, features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the logistic probability of each row of `features`."""
    return 1.0 / (1.0 + np.exp(-np.clip(weights[0] + features @ weights[1:], -30, 30)))
