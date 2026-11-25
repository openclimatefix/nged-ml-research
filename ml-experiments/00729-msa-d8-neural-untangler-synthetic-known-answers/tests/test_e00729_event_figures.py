# ruff: noqa: E501, PT018, PLR1714
"""Tests of the event-generator figures on invented engine records and segments."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import e00729_event_figures as ev
import numpy as np
import polars as pl

START = np.datetime64("2030-01-07", "D")
ONE_DAY = np.timedelta64(1, "D")


def evening_engine(*, n_days: int, level: float = 30.0) -> np.ndarray:
    """Return days with a 4-hour run from 17:00 on every day."""
    out = np.zeros((n_days, 48))
    out[:, 34:42] = level
    return out


def test_weekday_of_monday_is_zero() -> None:
    assert ev.weekday_of(days=START + np.arange(7) * ONE_DAY).tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_event_stats_on_a_known_engine() -> None:
    days = evening_engine(n_days=14)
    stats = ev.event_stats(days=days, dates=START + np.arange(14) * ONE_DAY)
    assert stats is not None
    assert stats.n_days == {"weekday": 10, "weekend": 4}
    assert np.allclose(stats.run_lengths_h, 4.0)
    assert len(stats.run_lengths_h) == 14
    assert stats.weekly_starts.tolist() == [7.0, 7.0]
    ref = ev.reference_level(series=days.ravel())
    assert ref is not None
    prob = stats.run_probability(kind="weekday")
    assert prob[17] == 1.0 and prob[16] == 0.0 and prob[21] == 0.0
    assert np.allclose(stats.daily_energy, 8 * 0.5 * (30.0 / ref) / 24.0)
    assert np.allclose(stats.steps, 1.0)


def test_the_statistics_do_not_depend_on_the_engine_size() -> None:
    dates = START + np.arange(14) * ONE_DAY
    a = ev.event_stats(days=evening_engine(n_days=14, level=30.0), dates=dates)
    b = ev.event_stats(days=evening_engine(n_days=14, level=7.5), dates=dates)
    assert a is not None and b is not None
    assert np.allclose(a.steps, b.steps) and np.allclose(a.daily_energy, b.daily_energy)


def test_an_engine_that_never_runs_has_no_statistics() -> None:
    assert ev.event_stats(days=np.zeros((7, 48)), dates=START + np.arange(7) * ONE_DAY) is None


def test_a_day_with_a_missing_half_hour_splits_the_stretch() -> None:
    days = evening_engine(n_days=15)
    days[7, 3] = np.nan
    assert ev.complete_stretches(days=days) == [(0, 7), (8, 15)]
    stats = ev.event_stats(days=days, dates=START + np.arange(15) * ONE_DAY)
    assert stats is not None
    assert stats.weekly_starts.tolist() == [7.0, 7.0]


def test_a_run_cut_by_a_stretch_edge_is_not_a_run_length() -> None:
    days = np.zeros((7, 48))
    days[0, :10] = 30.0
    days[3, 30:40] = 30.0
    stats = ev.event_stats(days=days, dates=START + np.arange(7) * ONE_DAY)
    assert stats is not None
    assert stats.run_lengths_h.tolist() == [5.0]


def test_merge_sparse_bins_keeps_at_least_the_minimum_in_every_group() -> None:
    counts = np.array([1.0, 2.0, 10.0, 0.0, 3.0])
    groups = ev.merge_sparse_bins(real_counts=counts, min_count=5)
    assert (
        groups == [(0, 3), (3, 5)]
        or groups == [(0, 5)]
        or all(counts[a:b].sum() >= 5 for a, b in groups)
    )
    assert all(counts[a:b].sum() >= 5 for a, b in groups)
    assert groups[0][0] == 0 and groups[-1][1] == len(counts)
    assert all(g[1] == h[0] for g, h in itertools.pairwise(groups))


def test_merge_sparse_bins_with_too_few_events_gives_one_group() -> None:
    assert ev.merge_sparse_bins(real_counts=np.array([1.0, 0.0, 2.0]), min_count=5) == [(0, 3)]


def test_grouped_shares_sum_to_one_and_use_the_real_groups() -> None:
    real = np.array([0.5] * 6 + [3.0] * 6 + [30.0])
    simv = np.array([0.5, 1.5, 3.0, 5.0, 10.0, 30.0] * 10)
    groups, rs, ss = ev.grouped_shares(real=real, sim_values=simv, lower=ev.RUN_LENGTH_LOWER_H)
    assert np.isclose(rs.sum(), 1.0) and np.isclose(ss.sum(), 1.0)
    assert len(groups) == len(rs) == len(ss)
    labels = ev.group_labels(lower=ev.RUN_LENGTH_LOWER_H, groups=groups, unit=" h")
    assert len(labels) == len(groups)


def test_no_real_bin_shows_fewer_than_the_minimum_events() -> None:
    rng = np.random.default_rng(0)
    real = rng.exponential(3.0, size=40)
    rc = ev.bin_counts(values=real, lower=ev.RUN_LENGTH_LOWER_H)
    for a, b in ev.merge_sparse_bins(real_counts=rc):
        assert rc[a:b].sum() >= ev.MIN_REAL_EVENTS


def test_simulated_draws_are_reproducible_and_inside_the_ranges() -> None:
    a = ev.draw_engines(n=6)
    b = ev.draw_engines(n=6)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b, strict=True))
    for engine, params in a:
        assert engine.shape == (ev.N_DAYS_PER_DRAW, 48)
        assert engine.min() >= 0
        assert engine.max() <= params["total_points_cap"] + 1e-9


def test_picked_cases_are_distinct_windows_and_cover_several_kinds() -> None:
    draws = ev.draw_engines(n=60)
    cases = ev.pick_cases(draws=draws)
    assert len(cases) >= 8
    assert len({c.label for c in cases}) == len(cases)
    again = ev.pick_cases(draws=draws)
    assert [(c.label, c.draw, c.first_day) for c in cases] == [
        (c.label, c.draw, c.first_day) for c in again
    ]


def solar_segment(*, source: str, month: int, csi: float, n: int = 3) -> SimpleNamespace:
    hh = np.arange(48)
    day = np.where((hh >= 14) & (hh < 34), 20.0 * csi, 0.0)
    dates = np.datetime64(f"2030-{month:02d}-01", "D") + np.arange(n + 2) * ONE_DAY
    cal_index = np.r_[-1, np.arange(n), -1]
    return SimpleNamespace(
        source=source, dates=dates, cal_index=cal_index, solar=np.tile(day, (n + 2, 1)),
        csi=np.full((n + 2, 48), csi), s1_fallback=np.zeros(n + 2, dtype=bool),
    )  # fmt: skip


def test_pick_solar_days_takes_the_clearest_and_cloudiest_of_each_season() -> None:
    segs = [
        solar_segment(source="S2", month=m, csi=c)
        for m, c in ((1, 0.9), (1, 0.2), (7, 0.8), (7, 0.4), (4, 0.5))
    ]
    segs.append(solar_segment(source="S1", month=1, csi=0.99))
    days = ev.pick_solar_days(segments=segs, source="S2", dated=False)
    assert [d.label for d in days] == [
        "winter, clearest", "winter, cloudiest", "spring, clearest", "spring, cloudiest", "summer, clearest", "summer, cloudiest",
    ]  # fmt: skip
    assert days[0].output.max() == 20.0 * 0.9
    assert all(d.title_extra == "" for d in days)


def test_pick_solar_days_dates_only_when_asked() -> None:
    segs = [solar_segment(source="S1", month=7, csi=0.8)]
    assert ev.pick_solar_days(segments=segs, source="S1", dated=True)[0].title_extra.startswith(
        "2030-07"
    )
    assert ev.pick_solar_days(segments=segs, source="S2", dated=False) == []


def test_write_event_figures_keeps_real_series_out_of_the_published_files(tmp_path: Path) -> None:
    n_days = 40
    start = np.datetime64("2030-01-07T00:00", "m")
    times = start + (np.arange(n_days * 48) + 1) * np.timedelta64(30, "m")
    power = (
        evening_engine(n_days=n_days) * 0.10379 * (np.arange(n_days)[:, None] % 3 != 0) + 0.0
    ).ravel()
    record = pl.DataFrame(
        {"time": times.astype("datetime64[us]"), "power": power, "units": np.ones(n_days * 48)}
    )
    segs = [solar_segment(source="S2", month=m, csi=c) for m, c in ((1, 0.9), (1, 0.2), (7, 0.8))]
    segs.append(solar_segment(source="S1", month=7, csi=0.7))
    messages: list[str] = []
    out_dir, private = tmp_path / "figures", tmp_path / "PRIVATE" / "event_generator"
    written = ev.write_event_figures(
        segments=segs, record=record, out_dir=out_dir, private_dir=private, log=messages.append
    )
    names = {p.name for p in written}
    assert {
        "events_01_engine_days.svg",
        "events_02_solar_s2_days.svg",
        "events_03_hour_of_day.svg",
    } <= names
    assert all(p.exists() for p in written)
    assert (private / "real_engine_windows.svg").exists() and (
        private / "solar_s1_days.svg"
    ).exists()
    assert not any(p.parent == out_dir for p in private.iterdir())
    published_text = " ".join(p.read_text() for p in written) + " ".join(messages)
    assert "2030-01-" not in published_text
    assert "S1" not in published_text and "metered farms" not in published_text
