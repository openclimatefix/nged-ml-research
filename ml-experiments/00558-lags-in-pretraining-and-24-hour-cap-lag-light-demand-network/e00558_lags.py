# ruff: noqa: ANN001, ANN201
"""Experiment 558: lags capped at 24 hours, and block dropout of whole origin days.

Pure numpy functions, shared with later experiments. Grid indices are half-hours; the cap is 48
steps. At a forecast origin, the capped day lag of a target is the latest value of the target's
half-hour of day observed at least 24 h before the origin (24 to 48 h old). The capped week lag is
the latest same-weekday, same-half-hour value observed at least 24 h before the origin. The newest
lag does not exist in the capped recipe.
"""

from typing import Final

import numpy as np

STEPS_PER_DAY: Final[int] = 48
CAP_STEPS: Final[int] = 48
LEAD_SCALE_HOURS: Final[float] = 336.0  # experiment 397's age scale
P_DROP_DAY: Final[float] = 0.30
P_DAY_ONLY: Final[float] = 0.15


def capped_lag_indices(*, target_idx: np.ndarray, origin_idx: np.ndarray):
    """Return (day, week) capped-lag grid indices, each asserted >= 24 h before the origin."""
    target_idx, origin_idx = np.asarray(target_idx), np.asarray(origin_idx)
    assert (target_idx > origin_idx).all(), "every target must be after its origin"
    newest = origin_idx - CAP_STEPS
    gap = target_idx - newest
    day = target_idx - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(np.int64)
    week = target_idx - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(np.int64)
    assert (day <= newest).all(), "a capped day lag is younger than 24 h"
    assert (week <= newest).all(), "a capped week lag is younger than 24 h"
    assert (day > newest - STEPS_PER_DAY).all(), "capped day lag older than 48 h"
    return day, week


def capped_lag_ages_hours(*, lead_hours: np.ndarray):
    """Return (day, week) age in hours from the lag time to the target time."""
    gap = np.asarray(lead_hours, dtype=np.float64) + 24.0
    return 24.0 * np.ceil(gap / 24.0), 168.0 * np.ceil(gap / 168.0)


def capped_lag_values(*, clean: np.ndarray, primary, target_idx, origin_idx):
    """Return (day, week) raw lag values (NaN if unobserved) from the cleaned power grid `clean`."""
    day, week = capped_lag_indices(target_idx=target_idx, origin_idx=origin_idx)
    ok_d, ok_w = day >= 0, week >= 0
    vd = np.where(ok_d, clean[primary, np.clip(day, 0, None)], np.nan).astype(np.float32)
    vw = np.where(ok_w, clean[primary, np.clip(week, 0, None)], np.nan).astype(np.float32)
    return vd, vw


def block_dropout_masks(*, origin_idx: np.ndarray, seed: int):
    """Return (mask_all, mask_day_only) boolean arrays, drawn once per origin day.

    In `P_DROP_DAY` of days every lag is masked. Within those days, `P_DAY_ONLY` mask only the day
    lag (so `mask_day_only` is True only where `mask_all` is False for that row). The draw for a day
    depends only on (seed, day number), not on which rows are present.
    """
    day = np.asarray(origin_idx) // STEPS_PER_DAY
    uniq, inv = np.unique(day, return_inverse=True)
    draws = np.stack([np.random.default_rng([seed, int(d)]).random(2) for d in uniq])
    dropped = draws[:, 0] < P_DROP_DAY
    day_only = dropped & (draws[:, 1] < P_DAY_ONLY)
    mask_all = dropped & ~day_only
    return mask_all[inv], day_only[inv]
