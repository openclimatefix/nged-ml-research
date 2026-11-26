"""Tests for the neighbour table functions, on invented arrays."""

from pathlib import Path

import e00732_neighbours as nb
import numpy as np
import pytest


def _world(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2021-01-01", "D") + np.arange(n)
    base = 50 + 10 * np.sin(np.arange(48) / 48 * 2 * np.pi)
    reading = base + rng.normal(0, 1, (n, 48))
    # A mirrored neighbour: its change is minus D8's change, plus noise.
    mirror = 80 - np.cumsum(np.diff(reading, axis=1, prepend=reading[:, :1]), axis=1)
    return dates, reading, mirror + rng.normal(0, 0.1, (n, 48))


def test_place_on_calendar_uses_the_grid_offset():
    grid = np.arange(48 * 5, dtype=float)
    dates = np.array(["2021-01-02", "2021-01-04"], dtype="datetime64[D]")
    out = nb.place_on_calendar(grid=grid, dates=dates, first_day=np.datetime64("2021-01-01"))
    assert out[0, 0] == 48
    assert out[1, 3] == 3 * 48 + 3
    far = nb.place_on_calendar(
        grid=grid,
        dates=np.array(["2021-02-01"], dtype="datetime64[D]"),
        first_day=np.datetime64("2021-01-01"),
    )
    assert np.isnan(far).all()


def test_coverage_gate_fails_when_a_neighbour_is_mostly_missing():
    _dates, reading, mirror = _world()
    table = {"D3": mirror, "D5": mirror.copy(), "D14": mirror.copy()}
    table["D14"][:, ::2] = np.nan
    shares = nb.coverage(table=table, reading=reading)
    assert shares["D3"] == 1.0
    problems = nb.coverage_problems(shares=shares)
    assert len(problems) == 1
    assert "D14" in problems[0]


def test_coverage_fails_when_d8_is_empty():
    with pytest.raises(nb.NeighbourError):
        nb.coverage(
            table={n: np.ones((2, 48)) for n in nb.NEIGHBOURS}, reading=np.full((2, 48), np.nan)
        )


def test_alignment_passes_at_lag_zero_and_fails_when_shifted():
    _dates, reading, mirror = _world()
    good = nb.lag_correlations(reading=reading, neighbour=mirror)
    assert nb.alignment_problems(correlations={"D3": good}) == []
    shifted = np.roll(mirror, 1, axis=1)
    bad = nb.lag_correlations(reading=reading, neighbour=shifted)
    assert nb.alignment_problems(correlations={"D3": bad}) != []


def test_lag_correlation_fails_loudly_on_a_constant_series():
    _dates, reading, _ = _world()
    with pytest.raises(nb.NeighbourError):
        nb.lag_correlations(reading=reading, neighbour=np.ones_like(reading))


def test_fold_matches_the_formula_and_keeps_nan():
    out = nb.fold(signed=np.array([-3.0, 0.0, np.nan]), floor=4.0)
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(4.0)
    assert np.isnan(out[2])


def test_step_statistic_finds_a_level_step_and_ignores_the_shape():
    n = 6
    dates = np.datetime64("2021-01-04", "D") + np.arange(n)
    weekend = np.zeros(n, dtype=bool)
    shape = np.zeros((2, 48))
    values = np.zeros((n, 48))
    values[2, 20:] = 5.0
    stat = nb.step_statistic(values=values, dates=dates, weekend=weekend, shape=shape)
    assert stat[2, 20] == pytest.approx(5.0)
    assert stat[2, 22] == pytest.approx(2.5)
    shaped = nb.step_statistic(
        values=values + 100.0, dates=dates, weekend=weekend, shape=np.full((2, 48), 100.0)
    )
    assert np.allclose(np.nan_to_num(shaped), np.nan_to_num(stat))


def test_step_statistic_drops_windows_across_a_day_gap():
    dates = np.array(["2021-01-04", "2021-01-09"], dtype="datetime64[D]")
    stat = nb.step_statistic(
        values=np.arange(96.0).reshape(2, 48),
        dates=dates,
        weekend=np.zeros(2, bool),
        shape=np.zeros((2, 48)),
    )
    assert np.isnan(stat[0, 46:]).all()
    assert np.isnan(stat[1, :4]).all()
    assert np.isfinite(stat[0, 10])


def test_noise_unit_is_a_robust_scale_and_fails_on_no_data():
    stat = np.tile(np.array([-2.0, -1.0, 1.0, 2.0, 100.0]), (3, 1))
    unit = nb.noise_unit(statistic=stat, train_rows=np.array([True, True, False]))
    assert unit == pytest.approx(1.4826 * 2.0)
    with pytest.raises(nb.NeighbourError):
        nb.noise_unit(statistic=np.full((2, 5), np.nan), train_rows=np.array([True, True]))
    with pytest.raises(nb.NeighbourError):
        nb.noise_unit(statistic=np.zeros((2, 5)), train_rows=np.array([True, True]))


def test_channels_carry_value_change_alignment_and_mask_and_hide_with_d8():
    folded = np.array([[10.0, 12.0, 11.0, np.nan, 9.0]])
    d8 = np.array([[0.0, 1.0, -1.0, 0.0, 0.0]])
    out = nb.channels(folded=folded, median=10.0, d8_change=d8)
    assert out.shape == (4, 1, 5)
    assert out[0, 0, 1] == pytest.approx(1.2)
    assert out[1, 0, 1] == pytest.approx(0.2)
    assert out[1, 0, 2] == pytest.approx(-0.1)
    assert out[2, 0, 1] == pytest.approx(0.2)
    assert out[2, 0, 2] == pytest.approx(0.1)
    assert list(out[3, 0]) == [1, 1, 1, 0, 1]
    assert out[1, 0, 4] == 0.0  # the change after a missing half-hour is not invented
    hidden = np.array([[False, False, True, True, False]])
    out_h = nb.channels(folded=folded, median=10.0, d8_change=d8, hidden=hidden)
    assert (out_h[:, 0, 2:4] == 0).all()
    assert list(out_h[3, 0]) == [1, 1, 0, 0, 1]


def test_table_roundtrip_is_written_once_and_checks_days(tmp_path: Path):
    dates, _reading, mirror = _world(3)
    arrays = dict.fromkeys(nb.NEIGHBOURS, mirror)
    arrays["D5"] = mirror.copy()
    arrays["D5"][0, 0] = np.nan
    nb.write_table(root=tmp_path, dates=dates, arrays=arrays)
    with pytest.raises(FileExistsError):
        nb.write_table(root=tmp_path, dates=dates, arrays=arrays)
    back = nb.read_table(root=tmp_path, dates=dates)
    assert np.isnan(back["D5"][0, 0])
    assert np.allclose(back["D3"], mirror)
    with pytest.raises(nb.NeighbourError):
        nb.read_table(root=tmp_path, dates=dates + np.timedelta64(1, "D"))


def test_gate_problems_collects_coverage_and_alignment():
    _dates, reading, mirror = _world()
    arrays = {"D3": mirror, "D5": np.roll(mirror, 1, axis=1), "D14": mirror}
    problems = nb.gate_problems(arrays=arrays, reading=reading)
    assert len(problems) == 1
    assert "D5" in problems[0]
