"""Experiment 730: the real-data readings (H2, K1, K2, R2, R3), as pure functions on arrays.

The real days are fed to the arms as pseudo-segments by experiment 729's `real` module. The
functions here read the stored probabilities. No function prints a date, and every output is an
aggregate over half-hours.
"""

import e00730_paths  # noqa: F401

# isort: split
import e00730_core as core
import e00730_score as score
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
