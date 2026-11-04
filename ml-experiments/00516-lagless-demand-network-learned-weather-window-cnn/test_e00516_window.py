# ruff: noqa: D103, E402
"""Check experiment 516's window builder: step-grid smoothness, causality, and missed runs."""

import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).parent)]

import e00516_window as w  # ty: ignore[unresolved-import]
import numpy as np

DAY = 48
N_DAYS = 40
SIZE = N_DAYS * DAY + w.HORIZON + 10


def _anchors(*, n: int = 20, first: int = 20 * DAY) -> np.ndarray:
    return first + DAY * np.arange(n)


def test_step_offsets_are_the_ensemble_grid() -> None:
    assert len(w.STEP_OFFSETS) == 85
    assert w.STEP_OFFSETS[0] == 0
    assert w.STEP_OFFSETS[-1] == w.HORIZON
    steps = np.diff(w.STEP_OFFSETS)
    assert set(steps[:48]) == {6}
    assert set(steps[48:]) == {12}


def test_step_interpolate_reproduces_a_linear_series() -> None:
    series = 0.01 * np.arange(SIZE, dtype=float)
    anchors = _anchors(n=3)
    offsets = np.arange(w.HORIZON + 1)
    got = w.step_interpolate(series=series, anchors=anchors, query_offsets=offsets)
    np.testing.assert_allclose(got, series[anchors[:, None] + offsets[None, :]])


def test_step_interpolate_smooths_a_fast_wiggle_to_the_step_grid() -> None:
    """A series that changes within a 3 h step is replaced by the straight line between steps."""
    series = np.zeros(SIZE)
    anchors = _anchors(n=1)
    series[anchors[0] + 3] = 10.0  # between the steps at offsets 0 and 6
    got = w.step_interpolate(series=series, anchors=anchors, query_offsets=np.arange(13))[0]
    np.testing.assert_allclose(got, 0.0)  # the wiggle is invisible, as it is in the ensemble mean
    series[anchors[0] + 6] = 6.0
    got = w.step_interpolate(series=series, anchors=anchors, query_offsets=np.arange(13))[0]
    np.testing.assert_allclose(got[:7], np.linspace(0.0, 6.0, 7))


def test_step_interpolate_beyond_day_six_is_linear_between_six_hourly_steps() -> None:
    series = np.zeros(SIZE)
    anchor = _anchors(n=1)
    series[anchor[0] + w.EARLY_HORIZON + 12] = 12.0
    query = np.arange(w.EARLY_HORIZON, w.EARLY_HORIZON + 25)
    got = w.step_interpolate(series=series, anchors=anchor, query_offsets=query)[0]
    np.testing.assert_allclose(got[:13], np.linspace(0.0, 12.0, 13))
    np.testing.assert_allclose(got[12:], np.linspace(12.0, 0.0, 13))


def test_step_interpolate_is_nan_off_the_series() -> None:
    series = np.ones(100)
    got = w.step_interpolate(series=series, anchors=np.array([90]), query_offsets=np.array([0, 12]))
    assert np.isfinite(got[0, 0])
    assert np.isnan(got[0, 1])


def test_step_interpolate_rejects_offsets_beyond_the_horizon() -> None:
    with pytest.raises(ValueError, match="query offsets"):
        w.step_interpolate(
            series=np.zeros(SIZE), anchors=_anchors(n=1), query_offsets=np.array([w.HORIZON + 1])
        )


def test_history_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        w.history_from_runs(first_days=np.zeros((2, DAY)), anchors=np.array([100, 110]), size=SIZE)


def _ensemble_like(*, series: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return w.era5_history_and_runs(era5=series, anchors=anchors, size=SIZE)


def test_trailing_window_before_the_origin_reads_the_previous_runs_first_day() -> None:
    series = np.arange(SIZE, dtype=float)
    anchors = _anchors()
    history, runs = _ensemble_like(series=series, anchors=anchors)
    row_run = 5
    valid = anchors[row_run] + w.ORIGIN_OFFSET + 10  # 5 h after the origin
    windows, missing = w.gather_windows(
        history=history,
        runs=runs,
        anchors=anchors,
        run_of_row=np.array([row_run]),
        valid_index=np.array([valid]),
        window_offsets=w.TRAILING_OFFSETS,
    )
    assert missing[0] == 0
    np.testing.assert_allclose(windows[0], series[valid + w.TRAILING_OFFSETS])


def test_a_later_run_never_enters_a_window() -> None:
    """Corrupting every run after the row's own run and its history leaves the window unchanged."""
    rng = np.random.default_rng(0)
    series = rng.normal(size=SIZE)
    anchors = _anchors()
    history, runs = _ensemble_like(series=series, anchors=anchors)
    row_run = 8
    valid = anchors[row_run] + w.ORIGIN_OFFSET + 40
    kwargs = {
        "anchors": anchors,
        "run_of_row": np.array([row_run]),
        "valid_index": np.array([valid]),
        "window_offsets": w.CENTRED_OFFSETS,
    }
    base, _ = w.gather_windows(history=history, runs=runs, **kwargs)
    runs_cut = runs.copy()
    runs_cut[row_run + 1 :] = np.nan
    history_cut = history.copy()
    history_cut[anchors[row_run] + w.ORIGIN_OFFSET + 1 :] = np.nan  # later first days
    cut, _ = w.gather_windows(history=history_cut, runs=runs_cut, **kwargs)
    np.testing.assert_array_equal(base, cut)


def test_a_missing_previous_run_is_counted_in_missed_runs() -> None:
    series = np.arange(SIZE, dtype=float)
    anchors = _anchors()
    history, runs = _ensemble_like(series=series, anchors=anchors)
    row_run = 6
    first = anchors[row_run - 1] + w.ORIGIN_OFFSET + 1
    history[first : first + DAY] = np.nan  # drop one whole earlier first day
    valid = anchors[row_run] + w.ORIGIN_OFFSET + 5  # a trailing window reaches 24 h back
    windows, missing = w.gather_windows(
        history=history,
        runs=runs,
        anchors=anchors,
        run_of_row=np.array([row_run]),
        valid_index=np.array([valid]),
        window_offsets=w.TRAILING_OFFSETS,
    )
    assert missing[0] == 48 - 5  # positions at or before the origin, not after
    assert w.missed_runs(missing_history=missing)[0] == 1
    assert int(np.isnan(windows).sum()) == missing[0]


def test_window_beyond_the_horizon_is_nan_then_filled_by_persistence() -> None:
    series = np.arange(SIZE, dtype=float)
    anchors = _anchors()
    history, runs = _ensemble_like(series=series, anchors=anchors)
    valid = anchors[3] + w.HORIZON - 4
    windows, _ = w.gather_windows(
        history=history,
        runs=runs,
        anchors=anchors,
        run_of_row=np.array([3]),
        valid_index=np.array([valid]),
        window_offsets=w.CENTRED_OFFSETS,
    )
    assert np.isnan(windows[0, -1])
    filled = w.fill_nearest(windows=windows)
    assert np.isfinite(filled).all()
    assert filled[0, -1] == series[anchors[3] + w.HORIZON]


def test_fill_nearest_picks_the_earlier_value_on_ties_and_keeps_all_nan() -> None:
    windows = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan], [np.nan, 2.0, np.nan]])
    filled = w.fill_nearest(windows=windows)
    np.testing.assert_array_equal(filled[0], [1.0, 1.0, 3.0])
    assert np.isnan(filled[1]).all()
    np.testing.assert_array_equal(filled[2], [2.0, 2.0, 2.0])


def test_era5_and_ensemble_like_windows_are_equally_smooth() -> None:
    """After the step-grid subsample, no 3 h change is finer than the ensemble's steps allow."""
    rng = np.random.default_rng(1)
    series = rng.normal(size=SIZE)
    anchors = _anchors(n=1)
    runs = w.step_interpolate(
        series=series, anchors=anchors, query_offsets=np.arange(w.HORIZON + 1)
    )[0]
    second_difference = np.diff(runs, n=2)
    knots = set(w.STEP_OFFSETS[1:-1] - 1)  # the series bends only at the steps
    bends = {int(i) for i in np.flatnonzero(np.abs(second_difference) > 1e-9)}
    assert bends <= knots


def test_two_grid_form_equals_step_interpolate_for_a_midnight_run() -> None:
    """Grid A before the 144th hour and grid B after it reproduce `step_interpolate`."""
    rng = np.random.default_rng(2)
    hour = np.arange(SIZE) // 2 % 24
    hour = np.where(np.arange(SIZE) % 2 == 0, hour, -1)
    series = rng.normal(size=SIZE)
    series[hour < 0] = np.nan
    anchor = 20 * DAY
    assert hour[anchor] == 0
    grid_a, grid_b = w.era5_step_grids(hourly=series, hour_of_day=hour)
    offsets = np.arange(w.HORIZON + 1)
    expected = w.step_interpolate(
        series=np.where(np.isfinite(series), series, 0.0),
        anchors=np.array([anchor]),
        query_offsets=offsets,
    )[0]
    got = np.where(offsets <= w.EARLY_HORIZON, grid_a[anchor + offsets], grid_b[anchor + offsets])
    np.testing.assert_allclose(got, expected, atol=1e-9)
