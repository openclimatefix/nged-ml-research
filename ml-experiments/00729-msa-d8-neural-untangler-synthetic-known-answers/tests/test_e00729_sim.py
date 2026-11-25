"""Tests of the engine and solar simulators, the fold, the synthetic BSP, and the segments."""

import e00729_core as core
import e00729_sim as sim
import numpy as np
import pytest
from synthetic_world import make_calendar


def engine(*, seed: int, n_days: int = 31) -> tuple[np.ndarray, dict[str, float]]:
    rec: dict[str, float] = {}
    out = sim.draw_engine(
        rng=np.random.default_rng(seed), utc_offset=np.zeros(n_days), quiet_day=np.arange(n_days) % 7 >= 5, record=rec
    )
    return out, rec


def test_engine_shape_cap_and_parameters_in_range() -> None:
    for seed in range(40):
        out, rec = engine(seed=seed)
        assert out.shape == (31, 48)
        assert out.min() >= 0.0
        assert out.max() <= core.RANGES["total_points_cap"][1] + 1e-9
        sim.assert_in_ranges(params=rec)


def test_engine_partial_mean_on_a_half_hour_that_holds_a_switch() -> None:
    levels = set()
    for seed in range(60):
        out, _ = engine(seed=seed)
        row = out[out.max(axis=1) > 0]
        if len(row):
            levels.update(np.round(np.unique(row), 3).tolist())
    # a pure on/off engine would only show a few levels; partial means add many more
    assert len(levels) > 100


def test_engine_runs_more_on_weekdays_than_weekends() -> None:
    week, weekend = [], []
    for seed in range(80):
        out, _ = engine(seed=seed, n_days=28)
        quiet = np.arange(28) % 7 >= 5
        week.append((out[~quiet] > 0).mean())
        weekend.append((out[quiet] > 0).mean())
    assert np.mean(week) > np.mean(weekend)


def test_engine_is_deterministic_in_its_seed() -> None:
    a, _ = engine(seed=5)
    b, _ = engine(seed=5)
    c, _ = engine(seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fold_is_the_square_root_of_the_squares_and_never_below_the_floor() -> None:
    x = np.array([[-30.0, -1.0, 0.0, 2.0, 50.0]])
    rec: dict[str, float] = {}
    r = sim.fold_reading(x=x, rng=np.random.default_rng(1), record=rec)
    assert np.all(r >= np.abs(x))
    q = np.sqrt(r**2 - x**2)
    max_q = core.RANGES["q_mva"][1] / core.CAPACITY_MVA * core.POINTS
    assert q.max() <= max_q + 1e-9
    assert r[0, 2] == pytest.approx(q[0, 2])
    assert 0.8 <= rec["q_factor"] <= 1.0


def test_synthetic_bsp_subtracts_beta_times_the_injection_and_keeps_missing() -> None:
    real = np.array([[1.0, np.nan, 3.0]])
    out = sim.synthetic_bsp(bsp_real=real, engine=np.array([[10.0, 10.0, 0.0]]), solar=np.array([[0.0, 5.0, 4.0]]), beta=0.5)
    assert out[0, 0] == pytest.approx(-4.0)
    assert np.isnan(out[0, 1])
    assert out[0, 2] == pytest.approx(1.0)


def test_hourly_style_noise_has_the_requested_autocorrelation_and_sd() -> None:
    noise = sim.ar1_noise(rng=np.random.default_rng(0), size=200_000)
    assert noise.std() == pytest.approx(0.05, rel=0.05)
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.9, abs=0.02)


def segment(*, world: str, seed: int = 0, cal: sim.Calendar | None = None, use: np.ndarray | None = None) -> sim.Segment:
    cal = cal or make_calendar()
    return sim.simulate_segment(
        cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool) if use is None else use, world=world, rng=np.random.default_rng(seed)
    )


def test_none_world_truth_is_the_base_day_and_the_reading_folds_it() -> None:
    seg = segment(world="none")
    on = seg.cal_index >= 0
    assert on.sum() == 31
    assert np.array_equal(seg.engine, np.zeros_like(seg.engine))
    assert np.array_equal(seg.solar, np.zeros_like(seg.solar))
    assert np.allclose(seg.x[on], seg.base[on])
    assert np.all(seg.r[on] >= np.abs(seg.x[on]) - 1e-4)
    assert np.isnan(seg.r[~on]).all()
    assert seg.source == "none"


def test_segment_truth_is_base_minus_engine_minus_solar() -> None:
    for world in ("engine", "solar", "both"):
        seg = segment(world=world, seed=3)
        on = seg.cal_index >= 0
        assert np.allclose(seg.x[on], (seg.base - seg.engine - seg.solar)[on], atol=1e-3)
        assert (seg.engine.sum() > 0) == (world in ("engine", "both"))
        assert (seg.solar.sum() > 0) == (world in ("solar", "both"))
        assert seg.solar.min() >= 0.0
        sim.assert_in_ranges(params=seg.params)


def test_solar_is_zero_at_night_and_capped_by_the_capacity_range() -> None:
    cal = make_calendar()
    for seed in range(12):
        seg = segment(world="solar", seed=seed, cal=cal)
        on = seg.cal_index >= 0
        assert seg.solar[on][:, 0:6].max() == 0.0
        assert seg.solar.max() <= 1.2 * 150.0 * 1.0001 + 1e-6 or seg.source == "S2"


def test_both_solar_sources_occur() -> None:
    sources = {segment(world="solar", seed=s).source for s in range(30)}
    assert sources == {"S1", "S2"}


def test_s1_falls_back_to_s2_on_a_day_with_no_farm_data() -> None:
    cal = make_calendar()
    cal.farms.power_ok[:, 5] = False
    found = False
    for seed in range(30):
        seg = segment(world="solar", seed=seed, cal=cal)
        if seg.source == "S1":
            day = np.flatnonzero(seg.cal_index == 5)[0]
            assert seg.s1_fallback[day]
            assert seg.solar[day].max() > 0
            assert seg.s1_fallback.sum() == 1
            found = True
    assert found


def test_segment_uses_only_the_allowed_days() -> None:
    cal = make_calendar()
    use = np.ones(len(cal.dates), dtype=bool)
    use[:3] = False
    seg = segment(world="none", cal=cal, use=use)
    assert (seg.cal_index >= 0).sum() == 28
    assert np.isnan(seg.r[1:4]).all()


def test_synthetic_bsp_keeps_the_real_noise_where_nothing_is_injected() -> None:
    cal = make_calendar()
    seg = segment(world="none", cal=cal)
    on = seg.cal_index >= 0
    assert np.allclose(seg.bsp[on], cal.bsp[seg.cal_index[on]], atol=1e-5)
    assert seg.has_bsp
    no_bsp = segment(world="none", cal=make_calendar(bsp=False))
    assert not no_bsp.has_bsp


def test_real_segment_carries_the_given_reading_and_truth() -> None:
    cal = make_calendar()
    seg = sim.real_segment(cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), reading=cal.reading, truth=cal.reading * 0.5)
    on = seg.cal_index >= 0
    assert np.allclose(seg.r[on], cal.reading[seg.cal_index[on]], atol=1e-4)
    assert np.allclose(seg.x[on], 0.5 * seg.r[on], atol=1e-4)
    assert seg.engine.sum() == 0.0
