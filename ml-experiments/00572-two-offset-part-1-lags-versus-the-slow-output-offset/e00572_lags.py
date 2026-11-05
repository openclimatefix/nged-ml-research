# ruff: noqa: PT018, D205
# ruff: noqa: E501, ANN001
"""Experiment 572: experiment 560's day and week lags with a configurable cap (24 h or 3.5 h).

Experiment 560 (`e00560_data.lag_block`) takes the latest same-half-hour value, and the latest
same-weekday same-half-hour value, observed at least 24 h before the forecast origin. This module
repeats that construction with the cap as an argument, so that arm v3-3.5 can read lags observed at
least 3.5 h (7 half-hours) before the origin. With `cap_steps=48` it reproduces experiment 560's
columns exactly (asserted against `e00560_data.lag_block` in `check_matches_560`).
"""

from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00560_data as d560  # ty: ignore[unresolved-import]
import numpy as np

STEPS: Final[int] = lags.STEPS_PER_DAY
WEEK_STEPS: Final[int] = 7 * STEPS
CAP_24H: Final[int] = 48
CAP_3_5H: Final[int] = 7
"""Half-hours between the newest lag and the forecast origin: 48 is 24 h, 7 is 3.5 h."""
MAX_LEAD_HOURS: Final[float] = 351.0


def age_max(*, cap_steps: int) -> tuple[float, float]:
    """Return the missing-lag age code (day, week): the largest age at the longest lead, over 336."""
    gap_h = MAX_LEAD_HOURS + cap_steps * 0.5
    return (
        24.0 * float(np.ceil(gap_h / 24.0)) / lags.LEAD_SCALE_HOURS,
        168.0 * float(np.ceil(gap_h / 168.0)) / lags.LEAD_SCALE_HOURS,
    )


def lag_indices(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, cap_steps: int, extra_days: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (day, week) lag grid indices observed at least `cap_steps + 48 * extra_days` steps before the origin."""
    target_idx, origin_idx = np.asarray(target_idx), np.asarray(origin_idx)
    assert (target_idx > origin_idx).all(), "every target must be after its origin"
    newest = origin_idx - cap_steps - STEPS * extra_days
    gap = target_idx - newest
    day = target_idx - STEPS * np.ceil(gap / STEPS).astype(np.int64)
    week = target_idx - WEEK_STEPS * np.ceil(gap / WEEK_STEPS).astype(np.int64)
    assert (day <= newest).all() and (week <= newest).all(), "a lag is younger than its cap"
    return day, week


def lag_stats(*, rows, clean: np.ndarray, cap_steps: int) -> np.ndarray:
    """Return (mean, std) of the day and week lag values over the fold's training rows, shape (2, 2)."""
    train = ~rows.is_val
    day, week = lag_indices(
        target_idx=rows.target_idx[train], origin_idx=rows.origin_idx[train], cap_steps=cap_steps
    )
    out = np.zeros((2, 2))
    for j, idx in enumerate((day, week)):
        v = np.where(idx >= 0, clean[rows.primary[train], np.clip(idx, 0, None)], np.nan)
        out[j] = (np.nanmean(v), max(float(np.nanstd(v)), 1e-6))
    return out


def lag_block(
    *, clean, primary, target_idx, origin_idx, stats, cap_steps: int, extra_days: int = 0
) -> np.ndarray:  # fmt: skip
    """Return the 8 appended columns (37-44): value, missing flag, age for the day and week lag, then their grid indices."""
    day, week = lag_indices(
        target_idx=target_idx, origin_idx=origin_idx, cap_steps=cap_steps, extra_days=extra_days
    )
    amax = age_max(cap_steps=cap_steps + STEPS * extra_days)
    out = np.zeros((len(primary), 8), dtype=np.float32)
    for j, idx in enumerate((day, week)):
        raw = np.where(idx >= 0, clean[primary, np.clip(idx, 0, None)], np.nan)
        miss = ~np.isfinite(raw)
        z = np.clip((raw - stats[j, 0]) / stats[j, 1], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        age = (target_idx - idx) * 0.5 / lags.LEAD_SCALE_HOURS
        out[:, 3 * j] = np.where(miss, 0.0, z)
        out[:, 3 * j + 1] = miss
        out[:, 3 * j + 2] = np.where(miss, amax[j], age)
        out[:, 6 + j] = idx
    return out


def absent(*, x: np.ndarray, cap_steps: int, which: str = "all") -> None:
    """Set the lag columns of `x` to the absent encoding, in place (a masked lag has age code `age_max`)."""
    amax = age_max(cap_steps=cap_steps)
    for use, c0, age in (
        (which in {"all", "day"}, 37, amax[0]),
        (which == "all", 40, amax[1]),
    ):
        if use:
            x[..., c0], x[..., c0 + 1], x[..., c0 + 2] = 0.0, 1.0, age


def check_lag_ages(
    *, x: np.ndarray, origin_us: np.ndarray, primary: np.ndarray, origin_tab: np.ndarray,
    cap_hours: float, label: str, n: int = 200_000,
) -> str:  # fmt: skip
    """Assert row by row that every lag time is at least `cap_hours` before the origin, and that each
    lag's expected-weather run was issued at or before the lag time; return an example line.
    """
    grid_start, step = prep422.GRID_START_US, prep422.STEP_US
    sel = np.arange(len(primary))
    if len(sel) > n:
        sel = np.random.default_rng(0).choice(len(primary), n, replace=False)
    msg = []
    for j, name in enumerate(("day", "week")):
        idx = x[sel, 43 + j].astype(np.int64)
        ok = idx >= 0
        run = origin_tab[primary[sel][ok], idx[ok]]
        t_lag = grid_start + idx[ok] * step
        have = run > 0
        assert (run[have] <= t_lag[have]).all(), (
            f"{label}: a {name} lag's weather run is after the lag time"
        )
        assert (t_lag <= origin_us[sel][ok] - int(cap_hours * 3_600_000_000)).all(), (
            f"{label}: a {name} lag is within {cap_hours} h of the origin"
        )
        i = int(np.argmax(have))
        msg.append(
            f"{name}: missing-run share {1 - have.mean():.4f}; e.g. lag time {np.datetime64(int(t_lag[i]), 'us')}, "
            f"weather run origin {np.datetime64(int(run[i]), 'us')}, forecast origin {np.datetime64(int(origin_us[sel][ok][i]), 'us')}"
        )
    return f"{label}: " + " | ".join(msg)


def check_matches_560(*, clean, primary, target_idx, origin_idx, lead, stats) -> None:
    """Assert that the 24 h cap reproduces experiment 560's columns exactly (the first 100,000 rows)."""
    s = slice(0, 100_000)
    mine = lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        stats=stats, cap_steps=CAP_24H,
    )  # fmt: skip
    theirs = d560.lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        lead=lead[s], stats=stats,
    )  # fmt: skip
    assert np.array_equal(mine, theirs), "the 24 h cap differs from experiment 560's lag block"
