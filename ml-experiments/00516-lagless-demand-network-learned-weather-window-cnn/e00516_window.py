# ruff: noqa: E501
"""Half-hourly weather windows on the ensemble's step grid, for training (ERA5) and serving (ensemble).

**One window rule at every lead.** Every window is read from a half-hourly series. At serving, that
series is the ensemble mean, which the stored summaries already hold as a linear interpolation of the
ensemble's steps: 3-hourly from the run's 00:00 UTC initialisation to 144 h, then 6-hourly to 360 h.
At training, the window is read from ERA5 that has first been subsampled to the same step grid and
then linearly interpolated, so the training and serving series are equally smooth. Beyond 144 h the
half-hourly series carries no information finer than 6 h, and nothing is finer than 3 h on day 1.

**Causality (the same rule as experiment 437).** A row belongs to the run initialised at `anchor`
(00:00 UTC) with forecast origin `anchor + 18` half-hours (09:00 UTC). Its window reads two sources:

- positions at or before the origin come from the *history*: each earlier run's first day, which
  is the half-hours in `(origin_k, origin_k + 48]` of run `k` (leads of about 9.5 to 33.5 h). The
  history holds only runs whose origin is strictly before the row's origin, because a position at
  or before the row's origin lies in the first day of an earlier run;
- positions after the origin come from the row's own run.

A position with no value (a run missing from the history, or a window position beyond the run's
360 h horizon) is NaN in `gather_windows`; `fill_nearest` then holds the nearest finite value along
the window, and `missed_runs` counts the history positions that had to be filled, in whole missed
runs (the unit the degradation rules use), so a missed previous run cannot silently shorten a window.

All positions are indices on a half-hourly grid; the caller owns the grid's origin.
"""

from typing import Final

import numpy as np

HALF_HOURS_PER_DAY: Final[int] = 48
ORIGIN_OFFSET: Final[int] = 18
"""The forecast origin (09:00 UTC) is 18 half-hours after the 00:00 UTC initialisation."""
EARLY_STEP: Final[int] = 6
"""The ensemble's step to 144 h, in half-hours (3 h)."""
LATE_STEP: Final[int] = 12
"""The ensemble's step from 144 h to 360 h, in half-hours (6 h)."""
EARLY_HORIZON: Final[int] = 288
"""144 h in half-hours."""
HORIZON: Final[int] = 720
"""360 h in half-hours."""
STEP_OFFSETS: Final[np.ndarray] = np.concatenate(
    [np.arange(0, EARLY_HORIZON, EARLY_STEP), np.arange(EARLY_HORIZON, HORIZON + 1, LATE_STEP)]
)
"""Half-hour offsets from the initialisation of the ensemble's 85 steps."""
CENTRED_OFFSETS: Final[np.ndarray] = np.arange(-24, 24)
"""A 24 h window centred on the valid time: 12 h before up to 11.5 h after."""
TRAILING_OFFSETS: Final[np.ndarray] = np.arange(-47, 1)
"""A 24 h window ending at the valid time."""


def _interpolation_plan(*, query_offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, per query offset, the knot offsets on either side and the weight on the upper knot."""
    upper = np.clip(
        np.searchsorted(STEP_OFFSETS, query_offsets, side="left"), 1, len(STEP_OFFSETS) - 1
    )
    lo, hi = STEP_OFFSETS[upper - 1], STEP_OFFSETS[upper]
    weight = (query_offsets - lo) / (hi - lo)
    return lo, hi, weight


def step_interpolate(
    *, series: np.ndarray, anchors: np.ndarray, query_offsets: np.ndarray
) -> np.ndarray:
    """Return a half-hourly series subsampled to the ensemble's step grid, then interpolated.

    For each run, the value at a query offset is the linear interpolation of the series at the two
    ensemble steps around the offset, so the result is exactly as smooth as the ensemble mean the
    summaries store.

    Args:
        series: A half-hourly series (ERA5 for training). Every ensemble step falls on a half-hour.
        anchors: The grid index of each run's 00:00 UTC initialisation.
        query_offsets: Half-hour offsets from the initialisation to evaluate, each in `[0, HORIZON]`.

    Returns:
        An array (runs, query offsets); NaN where either knot lies outside `series` or is NaN.
    """
    if query_offsets.min() < 0 or query_offsets.max() > HORIZON:
        msg = f"query offsets must lie in [0, {HORIZON}]"
        raise ValueError(msg)
    lo, hi, weight = _interpolation_plan(query_offsets=query_offsets)

    def knot(offset: np.ndarray) -> np.ndarray:
        index = anchors[:, None] + offset[None, :]
        inside = (index >= 0) & (index < len(series))
        return np.where(inside, series[np.clip(index, 0, len(series) - 1)], np.nan)

    return (1.0 - weight) * knot(lo) + weight * knot(hi)


def first_day_offsets() -> np.ndarray:
    """Return the offsets from the initialisation of a run's first day, `(origin, origin + 24 h]`."""
    return np.arange(ORIGIN_OFFSET + 1, ORIGIN_OFFSET + HALF_HOURS_PER_DAY + 1)


def history_from_runs(*, first_days: np.ndarray, anchors: np.ndarray, size: int) -> np.ndarray:
    """Return the history series: each run's first day written at its positions on the grid.

    Args:
        first_days: Per-run first-day values (runs, 48) for offsets from `first_day_offsets`.
        anchors: The grid index of each run's initialisation, one per run.
        size: The length of the grid.

    Returns:
        A dense series of length `size`, NaN where no run's first day covers the position.

    Raises:
        ValueError: If two runs' first days overlap on the grid.
    """
    history = np.full(size, np.nan)
    positions = anchors[:, None] + first_day_offsets()[None, :]
    inside = (positions >= 0) & (positions < size)
    flat_pos = positions[inside]
    if np.unique(flat_pos).size != flat_pos.size:
        msg = "two runs' first days overlap on the grid"
        raise ValueError(msg)
    history[flat_pos] = first_days[inside]
    return history


def era5_history_and_runs(
    *, era5: np.ndarray, anchors: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ERA5 as the training stand-in for the ensemble: the history, and the per-run series.

    Args:
        era5: Hourly ERA5 values on the half-hourly grid (the `:00` rows; the `:30` rows are not
            read, because ensemble steps fall on whole hours and a `:30` value includes the next
            hour).
        anchors: The grid index of each run's 00:00 UTC initialisation.
        size: The length of the grid.

    Returns:
        The history series (`size`) and the per-run series (runs, `HORIZON + 1`), both read from
        ERA5 through the ensemble's step grid.
    """
    first_days = step_interpolate(series=era5, anchors=anchors, query_offsets=first_day_offsets())
    runs = step_interpolate(series=era5, anchors=anchors, query_offsets=np.arange(HORIZON + 1))
    return history_from_runs(first_days=first_days, anchors=anchors, size=size), runs


def gather_windows(
    *,
    history: np.ndarray,
    runs: np.ndarray,
    anchors: np.ndarray,
    run_of_row: np.ndarray,
    valid_index: np.ndarray,
    window_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's window of one variable, and how many history positions were missing.

    Args:
        history: The history series on the grid (`history_from_runs`).
        runs: Each run's series by offset from its initialisation (runs, `HORIZON + 1`), NaN where
            the run has no value.
        anchors: The grid index of each run's initialisation.
        run_of_row: Each row's run, as an index into `runs` and `anchors`.
        valid_index: Each row's valid time on the grid.
        window_offsets: Half-hour offsets from the valid time to read (`CENTRED_OFFSETS` or
            `TRAILING_OFFSETS`).

    Returns:
        The windows (rows, offsets), NaN where the source holds no value, and the count per row of
        window positions at or before the origin whose history value is NaN.
    """
    anchor = anchors[run_of_row][:, None]
    position = valid_index[:, None] + window_offsets[None, :]
    after_origin = position > anchor + ORIGIN_OFFSET
    offset_in_run = position - anchor
    in_run = after_origin & (offset_in_run <= HORIZON)
    from_run = runs[run_of_row[:, None], np.clip(offset_in_run, 0, HORIZON)]
    from_history = np.where(
        (position >= 0) & (position < len(history)),
        history[np.clip(position, 0, len(history) - 1)],
        np.nan,
    )
    windows = np.where(after_origin, np.where(in_run, from_run, np.nan), from_history)
    missing_history = (~after_origin & ~np.isfinite(from_history)).sum(axis=1)
    return windows, missing_history


def fill_nearest(*, windows: np.ndarray) -> np.ndarray:
    """Return the windows with each NaN replaced by the nearest finite value along the window.

    A window with no finite value stays NaN. Ties between an earlier and a later value take the
    earlier one, which is the persistence fill (hold the last known value).
    """
    n_rows, width = windows.shape
    position = np.arange(width)
    finite = np.isfinite(windows)
    last = np.maximum.accumulate(np.where(finite, position[None, :], -1), axis=1)
    next_ = np.minimum.accumulate(np.where(finite, position[None, :], width)[:, ::-1], axis=1)[
        :, ::-1
    ]
    use_last = (last >= 0) & (
        (next_ >= width) | (position[None, :] - last <= next_ - position[None, :])
    )
    source = np.where(use_last, last, next_)
    valid_source = (source >= 0) & (source < width)
    filled = np.take_along_axis(windows, np.clip(source, 0, width - 1), axis=1)
    return np.where(valid_source, filled, np.nan).reshape(n_rows, width)


def missed_runs(*, missing_history: np.ndarray) -> np.ndarray:
    """Return the missing history positions as whole missed runs (48 half-hours per run, rounded up)."""
    return -(-missing_history // HALF_HOURS_PER_DAY)
