# ruff: noqa: E501, ANN001, D205
"""Experiment 585: experiment 572's day and week lags plus up to five extra lags at other periods.

Same module surface as `e00572_lags`, so experiment 572's fitting script can use it unchanged
(`e00585_arm.py` swaps it in, as experiment 581 did). The block appended after the 37 lag-free
columns has `8 + 4 * K` columns for `K` extra lags:

- 0-7: experiment 572's block (day lag value, missing flag, age; week lag value, missing flag, age;
  the grid indices of the day and week lag times);
- `8 + 3 * k` to `10 + 3 * k`: extra lag `k` (value, missing flag, age);
- `8 + 3 * K + k`: the grid index of extra lag `k` (-1 where the lag is before the grid start).

A negative period `-D` means the week lag moved `D` days further back (arm N3 uses -7, a second
weekly lag; a fixed 14-day period would coincide with the week lag at many leads). An extra lag
with a positive period `P` days is the cleaned power at `target - P * ceil(gap / P)` days, where
`gap` is the distance from the target to the newest admissible reading (at least `cap_steps`
half-hours before the origin). For the periods used here (357, 364, 365, and 371 days) the
multiplier is 1 for every lead up to 15 days (asserted). `DROP_DAY` replaces the day lag's value, flag, and age by the absent code
(arm N1; the network also zeroes the day lag's expected-weather embeddings). `EXTRA_CLEAN`, when set, is the power grid that extra lags read (the day and week lags
still read the cleaned grid), which is how the year-ago lag is masked inside verified events.
"""

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00558_lags as lags  # ty: ignore[unresolved-import]
import e00572_lags as lg572  # ty: ignore[unresolved-import]
import numpy as np

STEPS = lg572.STEPS
WEEK_STEPS = lg572.WEEK_STEPS
CAP_24H = lg572.CAP_24H
CAP_3_5H = lg572.CAP_3_5H
LAST_US = int(np.datetime64("2026-06-30T23:30", "us").astype("int64"))

EXTRA_PERIODS_DAYS: tuple[int, ...] = ()
"""Periods (in days) of the extra lags; set by `e00585_arm.py`."""
DROP_DAY: bool = False
EXTRA_CLEAN: np.ndarray | None = None

age_max = lg572.age_max


def n_block() -> int:
    """Return the number of appended columns."""
    return 8 + 4 * len(EXTRA_PERIODS_DAYS)


def tau_col() -> int:
    """Return the column of the D5 knot position (after the lag block)."""
    return 37 + n_block()


def extra_value_col(k: int) -> int:
    """Return the model-input column of extra lag `k`'s value (missing flag and age follow)."""
    return 45 + 3 * k


def extra_idx_col(k: int) -> int:
    """Return the model-input column of extra lag `k`'s grid index."""
    return 45 + 3 * len(EXTRA_PERIODS_DAYS) + k


def extra_indices(
    *, target_idx: np.ndarray, origin_idx: np.ndarray, cap_steps: int, extra_days: int = 0
) -> list[np.ndarray]:
    """Return the grid index of each extra lag, at least `cap_steps + 48 * extra_days` steps before the origin."""
    target_idx, origin_idx = np.asarray(target_idx), np.asarray(origin_idx)
    newest = origin_idx - cap_steps - STEPS * extra_days
    gap = target_idx - newest
    _, week = lg572.lag_indices(
        target_idx=target_idx, origin_idx=origin_idx, cap_steps=cap_steps, extra_days=extra_days
    )
    out = []
    for days in EXTRA_PERIODS_DAYS:
        if days < 0:  # the week lag moved |days| days further back (always older than the cap)
            idx = week + STEPS * days
        else:
            period = days * STEPS
            idx = target_idx - period * np.ceil(gap / period).astype(np.int64)
        assert (idx <= newest).all(), f"the {days}-day lag is younger than its cap"
        out.append(idx)
    return out


def lag_stats(*, rows, clean: np.ndarray, cap_steps: int) -> np.ndarray:
    """Return (mean, std) of the day, week, and extra lag values over the training rows, shape (2 + K, 2)."""
    base = lg572.lag_stats(rows=rows, clean=clean, cap_steps=cap_steps)
    train = ~rows.is_val
    out = [base]
    for idx in extra_indices(
        target_idx=rows.target_idx[train], origin_idx=rows.origin_idx[train], cap_steps=cap_steps
    ):
        v = np.where(idx >= 0, clean[rows.primary[train], np.clip(idx, 0, None)], np.nan)
        out.append(np.array([[np.nanmean(v), max(float(np.nanstd(v)), 1e-6)]]))
    return np.concatenate(out)


def lag_block(
    *, clean, primary, target_idx, origin_idx, stats, cap_steps: int, extra_days: int = 0
) -> np.ndarray:  # fmt: skip
    """Return the appended columns (37 onward); see the module docstring for the layout."""
    k_extra = len(EXTRA_PERIODS_DAYS)
    out = np.zeros((len(primary), n_block()), dtype=np.float32)
    out[:, :8] = lg572.lag_block(
        clean=clean, primary=primary, target_idx=target_idx, origin_idx=origin_idx,
        stats=stats[:2], cap_steps=cap_steps, extra_days=extra_days,
    )  # fmt: skip
    amax = lg572.age_max(cap_steps=cap_steps)
    if DROP_DAY:
        out[:, 0], out[:, 1], out[:, 2] = 0.0, 1.0, amax[0]
    src = clean if EXTRA_CLEAN is None else EXTRA_CLEAN
    for k, idx in enumerate(
        extra_indices(
            target_idx=target_idx, origin_idx=origin_idx, cap_steps=cap_steps, extra_days=extra_days
        )
    ):
        raw = np.where(idx >= 0, src[primary, np.clip(idx, 0, None)], np.nan)
        miss = ~np.isfinite(raw)
        z = np.clip(
            (raw - stats[2 + k, 0]) / stats[2 + k, 1], -prep.INPUT_CLIP_SD, prep.INPUT_CLIP_SD
        )
        age = (target_idx - idx) * 0.5 / lags.LEAD_SCALE_HOURS
        out[:, 8 + 3 * k] = np.where(miss, 0.0, z)
        out[:, 9 + 3 * k] = miss
        out[:, 10 + 3 * k] = np.where(miss, EXTRA_AMAX[k], age)
        out[:, 8 + 3 * k_extra + k] = idx
    return out


EXTRA_AMAX: list[float] = []
"""Missing-lag age code of each extra lag: the age of its period at the longest lead."""


def set_periods(*, periods_days: tuple[int, ...], drop_day: bool = False) -> None:
    """Configure the extra lags and fill in their missing-age codes."""
    global EXTRA_PERIODS_DAYS, DROP_DAY
    EXTRA_PERIODS_DAYS, DROP_DAY = tuple(periods_days), drop_day
    EXTRA_AMAX.clear()
    gap_h = lg572.MAX_LEAD_HOURS + CAP_24H * 0.5
    for days in EXTRA_PERIODS_DAYS:
        d = abs(days)
        EXTRA_AMAX.append(
            24.0
            * (d if days > 0 else 7 + d)
            * float(np.ceil(gap_h / (24.0 * (d if days > 0 else 7))))
            / lags.LEAD_SCALE_HOURS
            if days > 0
            else (168.0 * float(np.ceil(gap_h / 168.0)) + 24.0 * d) / lags.LEAD_SCALE_HOURS
        )


def absent(*, x: np.ndarray, cap_steps: int, which: str = "all") -> None:
    """Set the lag columns of `x` to the absent encoding, in place; extra lags are masked with the week lag."""
    lg572.absent(x=x, cap_steps=cap_steps, which=which)
    if which == "all":
        for k in range(len(EXTRA_PERIODS_DAYS)):
            c = extra_value_col(k)
            x[..., c], x[..., c + 1], x[..., c + 2] = 0.0, 1.0, EXTRA_AMAX[k]


def check_lag_ages(
    *, x: np.ndarray, origin_us: np.ndarray, primary: np.ndarray, origin_tab: np.ndarray,
    cap_hours: float, label: str, n: int = 200_000, valid_us: np.ndarray | None = None,
) -> str:  # fmt: skip
    """Assert for every lag (day, week, and each extra) that its time is at least `cap_hours` before the
    origin and that its expected-weather run was issued at or before the lag time; return an example line.
    """
    msg = [
        lg572.check_lag_ages(
            x=x, origin_us=origin_us, primary=primary, origin_tab=origin_tab,
            cap_hours=cap_hours, label=label, n=n,
        )
    ]  # fmt: skip
    grid_start, step = prep422.GRID_START_US, prep422.STEP_US
    sel = np.arange(len(primary))
    if len(sel) > n:
        sel = np.random.default_rng(0).choice(len(primary), n, replace=False)
    for k, days in enumerate(EXTRA_PERIODS_DAYS):
        idx = x[sel, extra_idx_col(k)].astype(np.int64)
        tgt = ((valid_us[sel] - grid_start) // step) if valid_us is not None else None
        if tgt is not None and days > 0:
            assert (tgt - idx == days * STEPS).all(), (
                f"{label}: the {days}-day lag is not exactly {days} days back"
            )
        assert (grid_start + idx * step <= LAST_US).all(), f"{label}: a lag is after 2026-06-30"
        ok = idx >= 0
        t_lag = grid_start + idx[ok] * step
        assert (t_lag <= origin_us[sel][ok] - int(cap_hours * 3_600_000_000)).all(), (
            f"{label}: the {days}-day lag is within {cap_hours} h of the origin"
        )
        run = origin_tab[primary[sel][ok], idx[ok]]
        have = run > 0
        if (
            days >= 300
        ):  # read from the ERA5/CAMS table; the lag is 340 days or more before the origin
            assert (t_lag <= origin_us[sel][ok] - 340 * 86_400_000_000).all(), label
            have = np.zeros_like(have)
        assert (run[have] <= t_lag[have]).all(), (
            f"{label}: the {days}-day lag's weather run is after the lag time"
        )
        msg.append(
            f"{days}-day lag: lag-time share before the grid {1 - ok.mean():.4f}, missing weather-run share {1 - have.mean():.4f}"
        )
    return " | ".join(msg)


def check_matches_560(*, clean, primary, target_idx, origin_idx, lead, stats) -> None:
    """Assert that the day and week columns equal experiment 572's (and so 560's) for the 24 h cap."""
    s = slice(0, 100_000)
    if DROP_DAY:
        return
    mine = lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        stats=stats, cap_steps=CAP_24H,
    )  # fmt: skip
    theirs = lg572.lag_block(
        clean=clean, primary=primary[s], target_idx=target_idx[s], origin_idx=origin_idx[s],
        stats=stats[:2], cap_steps=CAP_24H,
    )  # fmt: skip
    assert np.array_equal(mine[:, :8], theirs), (
        "the day and week columns differ from experiment 572's"
    )
