# ruff: noqa: E501
"""Synthetic price-driven generators for experiment 570's known-answer test.

Each rule maps the raw market index price (GBP per MWh, NaN where missing) on the half-hourly grid to
generator output in MW for capacity `capacity_mw`. Positive output reduces net demand.
"""

import numpy as np

STEPS_PER_DAY = 48
THRESHOLD = 90.0
MIN_RUN, MIN_OFF = 8, 4  # half-hours: 4 h minimum run, 2 h minimum off
RAMP_UP, RAMP_DOWN = 2.0 * 0.5, 3.0 * 0.5  # MW per half-hour


def threshold_rule(*, price: np.ndarray, capacity_mw: float) -> np.ndarray:
    """R1: full output above 90 GBP per MWh.

    The minimum run is 4 h, the minimum off time 2 h, and the ramps 2 MW per hour up and 3 down.
    """
    out = np.zeros(len(price))
    level, on, held = 0.0, False, 0
    for t, p in enumerate(price):
        want = bool(np.isfinite(p) and p > THRESHOLD)
        if on:
            held += 1
            if not want and held >= MIN_RUN:
                on, held = False, 0
        else:
            held += 1
            if want and held >= MIN_OFF:
                on, held = True, 0
        target = capacity_mw if on else 0.0
        level = min(level + RAMP_UP, target) if target > level else max(level - RAMP_DOWN, target)
        out[t] = level
    return out


def rank_rule(*, price: np.ndarray, capacity_mw: float) -> np.ndarray:
    """R2: full output in the 6 dearest half-hours of each UTC day."""
    n = len(price) // STEPS_PER_DAY * STEPS_PER_DAY
    days = np.where(np.isfinite(price[:n]), price[:n], -np.inf).reshape(-1, STEPS_PER_DAY)
    rank = np.argsort(np.argsort(-days, axis=1), axis=1)
    out = np.zeros(len(price))
    out[:n] = ((rank < 6) & np.isfinite(days)).astype(float).ravel() * capacity_mw
    return out


def battery_rule(*, price: np.ndarray, capacity_mw: float) -> np.ndarray:
    """R3: charge (negative output) in the 8 cheapest half-hours of each UTC day, discharge in the 8 dearest."""
    n = len(price) // STEPS_PER_DAY * STEPS_PER_DAY
    days = np.where(np.isfinite(price[:n]), price[:n], np.nan).reshape(-1, STEPS_PER_DAY)
    up = np.where(np.isnan(days), -np.inf, days)
    down = np.where(np.isnan(days), np.inf, days)
    dear = np.argsort(np.argsort(-up, axis=1), axis=1) < 8
    cheap = np.argsort(np.argsort(down, axis=1), axis=1) < 8
    out = np.zeros(len(price))
    out[:n] = ((dear.astype(float) - cheap.astype(float)) * np.isfinite(days)).ravel() * capacity_mw
    return out


RULES = {"R1": threshold_rule, "R2": rank_rule, "R3": battery_rule}
