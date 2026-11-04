# ruff: noqa: E501, PT018, D205
"""Inputs of experiment 560: capped-lag columns, block dropout, and the expected-input table.

The lag columns extend the 37 lag-free inputs (see `e00560_model` for the layout). The lag
definition and block dropout are experiment 558's (`e00558_lags`); `lag_indices` here adds the
"aged by extra days" variant used only at inference and is asserted equal to 558's at zero extra days.
"""

from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import numpy as np

STEPS: Final[int] = lags.STEPS_PER_DAY
WEEK_STEPS: Final[int] = 7 * STEPS
AGE_MAX_DAY: Final[float] = 24.0 * np.ceil((351.0 + 24.0) / 24.0) / lags.LEAD_SCALE_HOURS
AGE_MAX_WEEK: Final[float] = 168.0 * np.ceil((351.0 + 24.0) / 168.0) / lags.LEAD_SCALE_HOURS
"""Largest lag age (hours to the target over 336) at the longest lead (351 h) in the data."""
WEATHER_37: Final[list[int]] = list(range(7))
DATETIME_37: Final[list[int]] = list(range(14, 35))
EXPECTED_37: Final[list[int]] = WEATHER_37 + DATETIME_37


def lag_indices(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, extra_days: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (day, week) lag grid indices observed at least `24 h * (1 + extra_days)` before the origin."""
    target_idx, origin_idx = np.asarray(target_idx), np.asarray(origin_idx)
    newest = origin_idx - STEPS * (1 + extra_days)
    gap = target_idx - newest
    day = target_idx - STEPS * np.ceil(gap / STEPS).astype(np.int64)
    week = target_idx - WEEK_STEPS * np.ceil(gap / WEEK_STEPS).astype(np.int64)
    assert (day <= newest).all() and (week <= newest).all()
    if extra_days == 0:
        d558, w558 = lags.capped_lag_indices(target_idx=target_idx, origin_idx=origin_idx)
        assert (day == d558).all() and (week == w558).all(), "differs from experiment 558's builder"
    return day, week


def lag_stats(*, rows: data.Rows, clean: np.ndarray) -> np.ndarray:
    """Return (mean, std) of the day and week lag values over the fold's training rows, shape (2, 2)."""
    train = ~rows.is_val
    day, week = lag_indices(target_idx=rows.target_idx[train], origin_idx=rows.origin_idx[train])
    out = np.zeros((2, 2))
    for j, idx in enumerate((day, week)):
        v = np.where(idx >= 0, clean[rows.primary[train], np.clip(idx, 0, None)], np.nan)
        out[j] = (np.nanmean(v), max(float(np.nanstd(v)), 1e-6))
    return out


def lag_block(
    *, clean: np.ndarray, primary: np.ndarray, target_idx: np.ndarray, origin_idx: np.ndarray,
    lead: np.ndarray, stats: np.ndarray, extra_days: int = 0,
) -> np.ndarray:  # fmt: skip
    """Return the 8 appended columns (37-44) for rows; value standardised, missing filled with 0."""
    day, week = lag_indices(target_idx=target_idx, origin_idx=origin_idx, extra_days=extra_days)
    out = np.zeros((len(primary), 8), dtype=np.float32)
    for j, (idx, age_max) in enumerate(((day, AGE_MAX_DAY), (week, AGE_MAX_WEEK))):
        raw = np.where(idx >= 0, clean[primary, np.clip(idx, 0, None)], np.nan)
        miss = ~np.isfinite(raw)
        z = np.clip((raw - stats[j, 0]) / stats[j, 1], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        age = (target_idx - idx) * 0.5 / lags.LEAD_SCALE_HOURS
        out[:, 3 * j] = np.where(miss, 0.0, z)
        out[:, 3 * j + 1] = miss
        out[:, 3 * j + 2] = np.where(miss, age_max, age)
        out[:, 6 + j] = idx
    seen = out[:, 1] < 0.5
    day_h, week_h = lags.capped_lag_ages_hours(lead_hours=lead)
    assert np.allclose(out[seen, 2] * lags.LEAD_SCALE_HOURS, day_h[seen] + 24.0 * extra_days), (
        "day lag age differs from experiment 558's"
    )
    if extra_days == 0:
        seen = out[:, 4] < 0.5
        assert np.allclose(out[seen, 5] * lags.LEAD_SCALE_HOURS, week_h[seen]), (
            "week lag age differs from experiment 558's"
        )
    return out


def absent(*, x: np.ndarray, which: str = "all") -> None:
    """Set the lag columns of `x` (37-wide base + appended block) to the absent encoding, in place."""
    day = which in {"all", "day"}
    week = which == "all"
    for use, c0, age in ((day, 37, AGE_MAX_DAY), (week, 40, AGE_MAX_WEEK)):
        if use:
            x[..., c0], x[..., c0 + 1], x[..., c0 + 2] = 0.0, 1.0, age


# --- expected inputs ----------------------------------------------------------------------------


def raw_table(*, source: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (columns, run origin): per Primary and grid time, the unstandardised weather and datetime
    columns of the row with the smallest lead for that valid time, and that row's origin (microseconds).

    `source` is `ft` (ENS summaries, fine-tuning and scoring) or `pre` (pseudo-runs on ERA5/CAMS).
    NaN columns and origin 0 mark grid times with no row. The result is cached as `.npy` files.
    """
    f_col, f_org = cache / f"expected_{source}_cols.npy", cache / f"expected_{source}_origin.npy"
    if f_col.exists() and f_org.exists():
        return np.load(f_col, mmap_mode="r"), np.load(f_org, mmap_mode="r")
    keep37, _, _ = data.non_lag_layout()
    grid = data.grid_length()
    folder = prep.RAW_DIR / "summary" if source == "ft" else prep418.RAW_DIR
    table = prep.primaries()
    cols = np.full((table.height, grid, len(EXPECTED_37)), np.nan, dtype=np.float32)
    origin = np.zeros((table.height, grid), dtype=np.int64)
    for row in table.iter_rows(named=True):
        z = np.load(folder / f"{row['site']}.npz")
        gi = ((z["valid"] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        order = np.lexsort((z["lead"], gi))
        first = np.ones(len(order), dtype=bool)
        first[1:] = gi[order][1:] != gi[order][:-1]
        sel = order[first]
        sel = sel[(gi[sel] >= 0) & (gi[sel] < grid)]
        x = z["x"][sel][:, keep37][:, EXPECTED_37]
        cols[row["primary"], gi[sel]] = x
        origin[row["primary"], gi[sel]] = z["origin"][sel]
    cache.mkdir(parents=True, exist_ok=True)
    np.save(f_col, cols)
    np.save(f_org, origin)
    return cols, origin


def standardised_table(*, cols: np.ndarray, standardiser: prep.Standardiser) -> np.ndarray:
    """Return the table standardised like the model's own inputs, as float16, missing times as 0."""
    mean = standardiser.mean[EXPECTED_37]
    std = standardiser.std[EXPECTED_37]
    out = np.empty(cols.shape, dtype=np.float16)
    for p in range(cols.shape[0]):
        z = np.clip((np.asarray(cols[p]) - mean) / std, -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD)
        out[p] = np.nan_to_num(z, nan=0.0).astype(np.float16)
    return out


def check_expected(
    *, origin_tab: np.ndarray, primary: np.ndarray, x: np.ndarray, origin_us: np.ndarray, label: str,
    x_lag_rows: int = 200_000,
) -> str:  # fmt: skip
    """Assert row by row that each lag's weather run was issued at or before the lag time, itself at
    least 24 h before the forecast origin; return a one-line example and the missing share.
    """
    sel = np.arange(len(primary))
    if len(sel) > x_lag_rows:
        sel = np.random.default_rng(0).choice(len(primary), x_lag_rows, replace=False)
    msg = []
    for j, name in enumerate(("day", "week")):
        idx = x[sel, 43 + j].astype(np.int64)
        ok = idx >= 0
        run = origin_tab[primary[sel][ok], idx[ok]]
        t_lag = prep422.GRID_START_US + idx[ok] * prep422.STEP_US
        have = run > 0
        assert (run[have] <= t_lag[have]).all(), (
            f"{label}: a {name} lag's weather run is after the lag time"
        )
        assert (t_lag <= origin_us[sel][ok] - 24 * 3_600_000_000).all(), (
            f"{label}: {name} lag within 24 h of origin"
        )
        i = int(np.argmax(have))
        msg.append(
            f"{name}: missing-run share {1 - have.mean():.4f}; e.g. lag time {np.datetime64(int(t_lag[i]), 'us')}, weather run origin {np.datetime64(int(run[i]), 'us')}, forecast origin {np.datetime64(int(origin_us[sel][ok][i]), 'us')}"
        )
    return f"{label}: " + " | ".join(msg)
