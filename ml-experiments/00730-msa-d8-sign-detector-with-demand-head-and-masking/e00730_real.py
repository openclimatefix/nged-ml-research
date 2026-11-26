"""Experiment 730: the real-data readings (H2, K1, K2, R2, R3), as pure functions on arrays.

The real days are fed to the arms as pseudo-segments by experiment 729's `real` module. The
functions here read the stored probabilities. No function prints a date, and every output is an
aggregate over half-hours.
"""

import e00730_paths  # noqa: F401

# isort: split
import e00729_baselines as base
import e00730_core as core
import e00730_score as score
import e00730_signed as signed
import numpy as np


def h2(*, p: np.ndarray, limit: float) -> dict[str, float | bool]:
    """Return H2: SD*'s mean P(X < 0) and its share above 0.5 on the real held-out pre-engine days.

    On the test months without injection every half-hour is import and every reading is above
    1 MVA, so the check reads every half-hour.
    """
    return score.real_harm_check(p=p, limit=limit)


def k_reading(*, p: np.ndarray, x: np.ndarray, p0: float) -> dict[str, float]:
    """Return the Brier score and the log loss of K1 or K2 against P0's, which always says import.

    Args:
        p: The arm's P(X < 0) on the metered engine record's days.
        x: The signed truth of those half-hours (the folded reading less the record, for K1).
        p0: P0's probability (0).

    Returns:
        The two scores of the arm and of P0, and the number of half-hours.
    """
    q = np.full_like(p, p0)
    return {
        "brier": float(score.row_brier(p=p, x=x).mean()),
        "brier_p0": float(score.row_brier(p=q, x=x).mean()),
        "log_loss": float(score.row_log_loss(p=p, x=x).mean()),
        "n": float(p.size),
    }


def false_alarm_shares(*, p: np.ndarray) -> dict[str, float]:
    """Return the shares of real half-hours with P(X < 0) above 0.5 and above 0.9."""
    return {
        "above_half": float(np.mean(p > core.FALSE_ALARM_THRESHOLD)),
        "above_0.9": float(np.mean(p > 0.9)),
    }


def share_by(*, values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Return the share of True `values` in each group, as plain floats (aggregates only)."""
    return {str(g): float(np.mean(values[groups == g])) for g in np.unique(groups)}


def k1_against_baselines(
    *,
    rows: object,
    reading: np.ndarray,
    p: np.ndarray,
    mu: np.ndarray,
    climatology: base.Climatology,
    pi: float,
) -> dict[str, float]:
    """Return K1's scores beside B1's and the demand output's error beside the climatology's.

    Args:
        rows: The K1 rows (`x`, `r`, and `features`).
        reading: D8's real reading on the same half-hours (the rows of set `k2`, unfolded of the
            record), which the demand output is compared with.
        p: SD*'s P(X<0) on them.
        mu: SD*'s demand output on them, in points.
        climatology: Experiment 729's fitted demand climatology.
        pi: The injected share of the test bank, B1's prior.

    Returns:
        B1's Brier score and log loss, the demand output's mean absolute error against the
        reading, and the demand climatology's.
    """
    f = rows.features  # ty: ignore[unresolved-attribute]
    x, r = rows.x, reading  # ty: ignore[unresolved-attribute]
    b1 = np.clip(pi * climatology.prob_below(r=f.r, f=f), 0.0, 1.0)
    clim = signed.demand_climatology(climatology=climatology, features=f)
    return {
        "brier_b1": float(score.row_brier(p=b1, x=x).mean()),
        "log_loss_b1": float(score.row_log_loss(p=b1, x=x).mean()),
        "log_loss_p0": float(score.row_log_loss(p=np.zeros_like(p), x=x).mean()),
        "demand_mae": float(np.abs(mu - r).mean()),
        "demand_mae_climatology": float(np.abs(clim - r).mean()),
    }
