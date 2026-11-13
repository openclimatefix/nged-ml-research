# ruff: noqa: E501, PT018
"""Experiment 696's state-column builders on small synthetic inputs."""

from datetime import timedelta

import numpy as np
from e00695_common import PIPELINE_GRID_START  # ty: ignore[unresolved-import]
from e00696_state import (
    DAY,
    hide_half_of_origin_days,
    permute_days,
    rows_state,
    tracked_state,
    true_state,
    unit_vector,
)


def _day_ids(*, n_days: int) -> np.ndarray:
    """Return a grid whose first column is the day number and the others the half-hour of day."""
    day = np.repeat(np.arange(n_days), DAY)
    half_hour = np.tile(np.arange(DAY), n_days)
    return np.stack([day, half_hour, np.zeros(len(day))], axis=1).astype(np.float32)


def _group_of(*, day: int) -> tuple[int, bool]:
    """Return the calendar month and the weekend flag of the day `day` after the grid's start."""
    date = PIPELINE_GRID_START + timedelta(days=day)
    return date.month, date.weekday() >= 5


def test_tracked_state_values() -> None:
    """The expected share is the sum of j times the unit times P(k = j), over capacity."""
    pk = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.25, 0.75, 0.0], [1.0, 0.0, 0.0, 0.0]])
    unit = np.array([2.0, 2.0, 4.0])
    got = tracked_state(pk=pk, unit=unit, capacity=10.0)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got[:, 0], [0.5 * 2.0 / 10.0, (0.25 + 1.5) * 2.0 / 10.0, 0.0])
    np.testing.assert_allclose(got[:, 1], [0.5, 1.0, 0.0])
    assert (got[:, 2] == 0.0).all()


def test_unit_vector_switches_at_era_two() -> None:
    """Era 1's unit size applies before the first half-hour of era 2."""
    unit = unit_vector(n_grid=6, era2_first=4, unit1=1.5, unit2=2.5)
    assert unit.tolist() == [1.5, 1.5, 1.5, 1.5, 2.5, 2.5]


def test_true_state_values() -> None:
    """The true state is the output over capacity and whether any unit runs."""
    got = true_state(output=np.array([0.0, 3.0]), units=np.array([0, 2]), capacity=6.0)
    np.testing.assert_allclose(got, [[0.0, 0.0, 0.0], [0.5, 1.0, 0.0]])


def test_rows_state_reads_the_series_at_the_target_half_hour_and_zeros_elsewhere() -> None:
    """A row of the series reads the grid at its own target; a row of another Primary reads zeros."""
    grid = np.arange(30, dtype=np.float32).reshape(10, 3)
    got = rows_state(
        grid_state=grid, row=4, primary=np.array([4, 2, 4]), target_idx=np.array([7, 7, 0])
    )
    np.testing.assert_array_equal(got[0], grid[7])
    np.testing.assert_array_equal(got[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(got[2], grid[0])


def test_permute_days_keeps_groups_and_blocks_and_never_reads_past_the_cut() -> None:
    """The permutation shuffles whole days inside calendar month and weekday class, and only before `stop`."""
    n_days = 500
    state = _day_ids(n_days=n_days)
    first, partial = 48 * 20, 17
    stop = first + 48 * 400 + partial
    out = permute_days(state=state, first=first, stop=stop, seed=0)
    np.testing.assert_array_equal(out[:first], state[:first])
    np.testing.assert_array_equal(out[stop:], state[stop:])
    blocks = out[first : first + 48 * 400].reshape(400, DAY, 3)
    sources = blocks[:, 0, 0].astype(int)
    assert (blocks[:, :, 0] == sources[:, None]).all(), "a day was split"
    assert (blocks[:, :, 1] == np.arange(DAY)[None, :]).all(), "a day was reordered inside"
    destination_days = np.arange(20, 420)
    assert (sources != destination_days).mean() > 0.5, "the permutation left most days alone"
    for destination, source in zip(destination_days, sources, strict=True):
        assert _group_of(day=int(destination)) == _group_of(day=int(source))
    assert sorted(sources.tolist()) == destination_days.tolist(), "days were duplicated or lost"
    tail = out[first + 48 * 400 : stop]
    assert (tail[:, 0] == tail[0, 0]).all() and (tail[:, 1] == np.arange(partial)).all()
    assert _group_of(day=int(tail[0, 0])) == _group_of(day=420)
    changed = state.copy()
    changed[stop:] = -1.0
    np.testing.assert_array_equal(
        permute_days(state=changed, first=first, stop=stop, seed=0)[:stop], out[:stop]
    )


def test_permute_days_stays_within_its_era() -> None:
    """With an era boundary, a day only takes the content of a day of its own era."""
    state = _day_ids(n_days=500)
    first, boundary_day = 48 * 20, 260
    stop = first + 48 * 400 + 17
    out = permute_days(state=state, first=first, stop=stop, seed=0, era_starts=(48 * boundary_day,))
    blocks = out[first : first + 48 * 400].reshape(400, DAY, 3)
    sources = blocks[:, 0, 0].astype(int)
    destination_days = np.arange(20, 420)
    assert ((sources >= boundary_day) == (destination_days >= boundary_day)).all()
    assert (sources != destination_days).mean() > 0.5, "the permutation left most days alone"
    tail = out[first + 48 * 400 : stop]
    assert tail[0, 0] >= boundary_day, "the partial day crossed into the earlier era"


def test_hide_half_of_origin_days_hides_whole_days() -> None:
    """Every row of an origin day shares one coin, and a hidden row reads zeros and the flag."""
    origin = np.repeat(np.arange(400) * DAY, 5) + np.tile(np.arange(5), 400)
    state = np.ones((len(origin), 3), dtype=np.float32)
    out, hidden = hide_half_of_origin_days(
        state=state, origin_idx=origin, seed=3, rows=np.ones(len(origin), dtype=bool)
    )
    by_day = hidden.reshape(400, 5)
    assert (by_day == by_day[:, :1]).all()
    assert 0.4 < by_day[:, 0].mean() < 0.6
    assert (out[hidden] == [0.0, 0.0, 1.0]).all()
    assert (out[~hidden] == 1.0).all()


def test_hide_half_of_origin_days_leaves_other_primaries_alone() -> None:
    """A row of another Primary is never hidden and never carries the flag."""
    origin = np.repeat(np.arange(400) * DAY, 2)
    mine = np.tile([True, False], 400)
    state = np.stack([np.where(mine, 1.0, 0.0)] * 3, axis=1).astype(np.float32)
    out, hidden = hide_half_of_origin_days(state=state, origin_idx=origin, seed=3, rows=mine)
    assert not hidden[~mine].any()
    np.testing.assert_array_equal(out[~mine], state[~mine])
    assert hidden[mine].any()
