# ruff: noqa: E501, ANN001, ANN202
"""Experiment 581: the robust multi-week lag block (arms X3-med, the placebo, and X3-35).

Same module surface as `e00572_lags`, so experiment 572's fitting script can use it unchanged
(`e00581_arm.py` swaps it in). The block appended after the 37 lag-free columns has 15 columns:

- 0-2: the newest day lag (value, missing flag, age), exactly experiment 572's;
- 3-5: the week summary in the place of the week lag: the median of the same-weekday half-hours at
  weeks 1 to 5 (value), a missing flag (fewer than `MIN_WEEKS` of the 5 observed), and the mean age
  of the observed weeks;
- 6, 7: the grid indices of the newest day lag and of the first week used (they address the
  expected-weather table, as in experiment 560);
- 8: median absolute deviation of the weekly lags; 9, 10: median and median absolute deviation of
  the day lags at days 1 to 3; 11: robust z-score of the day lag against the weekly median; 12:
  robust z-score of the first weekly lag against the weekly median; 13: robust z-score of the day lag
  against the day median; 14: share of the 5 weeks observed.

All values are in units of the training standard deviation of the experiment 572 lag columns. With
`WEEK_OFFSET = 5` (the placebo) the five weeks are weeks 6 to 10.
"""

import warnings
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00572_lags as lg572  # ty: ignore[unresolved-import]
import numpy as np

STEPS: Final[int] = lg572.STEPS
WEEK_STEPS: Final[int] = lg572.WEEK_STEPS
CAP_24H: Final[int] = lg572.CAP_24H
CAP_3_5H: Final[int] = lg572.CAP_3_5H
N_WEEKS: Final[int] = 5
N_DAYS: Final[int] = 3
MIN_WEEKS: Final[int] = 3
MAD_FLOOR: Final[float] = 0.1
"""Smallest spread (in training standard deviations) a robust z-score divides by."""
N_BLOCK: Final[int] = 15
EXTRA_FIRST: Final[int] = 45
"""Column of the first of the seven robust columns in the model input."""
WEEK_OFFSET: int = 0
"""Weeks skipped before the five weekly lags: 0 for X3-med, 5 for the placebo (weeks 6 to 10)."""

age_max = lg572.age_max
lag_stats = lg572.lag_stats
check_lag_ages = lg572.check_lag_ages
lag_indices = lg572.lag_indices


def _grid_values(*, clean: np.ndarray, primary: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return the cleaned power at grid indices `idx` (NaN before the grid start or where flagged)."""
    return np.where(idx >= 0, clean[primary, np.clip(idx, 0, None)], np.nan)


def _mad(*, v: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Return the median absolute deviation along axis 1 around `med` (NaN rows give NaN)."""
    return np.nanmedian(np.abs(v - med[:, None]), axis=1)


def lag_block(
    *, clean, primary, target_idx, origin_idx, stats, cap_steps: int, extra_days: int = 0
) -> np.ndarray:  # fmt: skip
    """Return the 15 appended columns (37-51); see the module docstring for the layout."""
    day, week = lg572.lag_indices(
        target_idx=target_idx, origin_idx=origin_idx, cap_steps=cap_steps, extra_days=extra_days
    )
    week = week - WEEK_STEPS * WEEK_OFFSET
    amax = lg572.age_max(cap_steps=cap_steps)
    scale = lags.LEAD_SCALE_HOURS
    out = np.zeros((len(primary), N_BLOCK), dtype=np.float32)
    clip = prep.INPUT_CLIP_SD

    def z_of(*, raw, j):
        return np.clip((raw - stats[j, 0]) / stats[j, 1], -clip, clip)

    wk_idx = week[:, None] - WEEK_STEPS * np.arange(N_WEEKS)[None, :]
    wk = np.stack(
        [_grid_values(clean=clean, primary=primary, idx=wk_idx[:, k]) for k in range(N_WEEKS)],
        axis=1,
    )
    dy_idx = day[:, None] - STEPS * np.arange(N_DAYS)[None, :]
    dy = np.stack(
        [_grid_values(clean=clean, primary=primary, idx=dy_idx[:, k]) for k in range(N_DAYS)],
        axis=1,
    )
    n_wk = np.isfinite(wk).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wk_med = np.nanmedian(wk, axis=1)
        wk_mad = _mad(v=wk, med=wk_med)
        dy_med = np.nanmedian(dy, axis=1)
        dy_mad = _mad(v=dy, med=dy_med)
    day_raw = dy[:, 0]
    miss_d = ~np.isfinite(day_raw)
    out[:, 0] = np.where(miss_d, 0.0, z_of(raw=day_raw, j=0))
    out[:, 1] = miss_d
    out[:, 2] = np.where(miss_d, amax[0], (target_idx - day) * 0.5 / scale)
    miss_w = n_wk < MIN_WEEKS
    out[:, 3] = np.where(miss_w, 0.0, z_of(raw=wk_med, j=1))
    out[:, 4] = miss_w
    ages = np.where(np.isfinite(wk), (target_idx[:, None] - wk_idx) * 0.5 / scale, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_age = np.nanmean(ages, axis=1)
    out[:, 5] = np.where(miss_w, amax[1], mean_age)
    out[:, 6] = day
    out[:, 7] = week
    sd_w, sd_d = stats[1, 1], stats[0, 1]
    floor_w = np.maximum(wk_mad, MAD_FLOOR * sd_w)
    floor_d = np.maximum(dy_mad, MAD_FLOOR * sd_d)

    def rz(*, a, med, floor, bad):
        return np.where(bad, 0.0, np.clip((a - med) / floor, -clip, clip))

    out[:, 8] = np.where(miss_w, 0.0, np.clip(wk_mad / sd_w, 0.0, clip))
    out[:, 9] = np.where(miss_d, 0.0, z_of(raw=dy_med, j=0))
    out[:, 10] = np.where(miss_d, 0.0, np.clip(dy_mad / sd_d, 0.0, clip))
    out[:, 11] = rz(a=day_raw, med=wk_med, floor=floor_w, bad=miss_d | miss_w)
    out[:, 12] = rz(a=wk[:, 0], med=wk_med, floor=floor_w, bad=~np.isfinite(wk[:, 0]) | miss_w)
    out[:, 13] = rz(a=day_raw, med=dy_med, floor=floor_d, bad=miss_d)
    out[:, 14] = n_wk / N_WEEKS
    return out


def absent(*, x: np.ndarray, cap_steps: int, which: str = "all") -> None:
    """Set the lag columns of `x` to the absent encoding, in place, including the seven robust columns."""
    amax = age_max(cap_steps=cap_steps)
    if which in {"all", "day"}:
        x[..., 37], x[..., 38], x[..., 39] = 0.0, 1.0, amax[0]
        for c in (EXTRA_FIRST + 1, EXTRA_FIRST + 2, EXTRA_FIRST + 3, EXTRA_FIRST + 5):
            x[..., c] = 0.0
    if which == "all":
        x[..., 40], x[..., 41], x[..., 42] = 0.0, 1.0, amax[1]
        for c in (EXTRA_FIRST, EXTRA_FIRST + 4, EXTRA_FIRST + 6):
            x[..., c] = 0.0


def check_matches_560(*, clean, primary, target_idx, origin_idx, lead, stats) -> None:
    """Assert that the newest day lag (columns 0-2) and its index equal experiment 560's (the first 100,000 rows)."""
    import e00560_data as d560  # ty: ignore[unresolved-import]

    s = slice(0, 100_000)
    mine = lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        stats=stats, cap_steps=CAP_24H,
    )  # fmt: skip
    theirs = d560.lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        lead=lead[s], stats=stats,
    )  # fmt: skip
    assert np.array_equal(mine[:, :3], theirs[:, :3]), "the day lag differs from experiment 560's"
    assert np.array_equal(mine[:, 6], theirs[:, 6]), (
        "the day lag index differs from experiment 560's"
    )
    if WEEK_OFFSET == 0:
        assert np.array_equal(mine[:, 7], theirs[:, 7]), (
            "the first week index differs from experiment 560's"
        )
