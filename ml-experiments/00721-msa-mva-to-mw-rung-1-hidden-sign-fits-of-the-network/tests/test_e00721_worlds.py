# ruff: noqa: E501
"""Known-answer tests of the synthetic world, the injection, and the margin's checks, on synthetic
arrays."""

import e00721_core as core
import e00721_worlds as worlds
import numpy as np
import pytest

pytest.importorskip("e00720_worlds")

G = 48 * 400


def _calendar() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    day = np.arange(G) // 48
    half_hour = np.arange(G) % 48
    month = (np.arange("2025-01-01", "2025-01-01", dtype="datetime64[D]").size) + 1
    del month
    months = ((np.datetime64("2025-01-01") + day).astype("datetime64[M]").astype(int) % 12) + 1
    return day, half_hour, months


def _u(*, half_hour: np.ndarray, day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bell = np.clip(np.sin(np.pi * (half_hour - 12) / 24.0), 0.0, None) ** 2
    cloud = np.clip(rng.uniform(0.2, 1.0, day.max() + 1)[day], 0.0, 1.0)
    return bell * cloud


def test_calibration_hits_the_target_export_share() -> None:
    rng = np.random.default_rng(0)
    day, half_hour, months = _calendar()
    u = _u(half_hour=half_hour, day=day, rng=rng)
    demand = np.full(G, 0.5)
    cap = worlds.calibrate_capacity_share(demand=demand, u=u, month=months, target=0.8)
    share = worlds.export_share_of(x=(demand - cap * u)[None, :], month=months)[0]
    assert share == pytest.approx(0.8, abs=0.05)


def test_build_synthetic_exporters_export_and_the_others_never_do() -> None:
    rng = np.random.default_rng(1)
    day, half_hour, months = _calendar()
    u = np.stack([_u(half_hour=half_hour, day=day, rng=rng) for _ in range(4)])
    demand_mean = np.full((4, G), 0.5)
    world = worlds.build_synthetic(
        demand_mean=demand_mean, u=u, month=months, exporter=np.array([True, True, False, False]), seed=721,
        day_offset_points=5.0,
    )  # fmt: skip
    lo, hi = worlds.EXPORT_SHARE_RANGE
    assert all(lo - 0.1 <= s <= hi + 0.1 for s in world.export_share[:2])
    assert (world.export_share[2:] == 0.0).all()
    assert np.isfinite(world.reading).all()
    assert (world.reading >= core.REACTIVE_FLOOR_POINTS / 100.0 - 1e-9).all()
    again = worlds.build_synthetic(
        demand_mean=demand_mean, u=u, month=months, exporter=np.array([True, True, False, False]), seed=721,
        day_offset_points=5.0,
    )  # fmt: skip
    np.testing.assert_array_equal(world.x, again.x)


def test_shifted_sky_moves_by_28_days_and_leaves_nan_at_the_start() -> None:
    a = np.arange(48 * 40, dtype=np.float64)
    out = worlds.shifted_sky(a=a)
    assert np.isnan(out[: 28 * 48]).all()
    np.testing.assert_array_equal(out[28 * 48 :], a[: -28 * 48])


def test_sky_blind_rule_is_blind_without_a_unique_hypothesis() -> None:
    demand = np.array([0.5, 0.5, 0.1])
    reading = np.array([0.2, 0.9, 0.3])
    got = worlds.sky_blind_rule_scores(demand_mean=demand, reading_v=reading, pv_capacity=0.8)
    assert got.shape == (3,)
    assert set(np.unique(got)) <= {0.0, 0.5, 1.0}


def test_reachability_passes_when_the_sky_blind_rule_is_far_from_perfect() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    day = rng.integers(0, 120, n)
    demand = rng.uniform(0.3, 0.7, n)
    pv = rng.uniform(0.0, 1.2, n)
    x = demand - pv
    reading = np.sqrt(x**2 + 0.02**2)
    out = worlds.reachability_check(
        demand_mean=demand, reading=reading, x_true=x, u=np.ones(n), day=day, pv_capacity=1.2
    )  # fmt: skip
    assert out["evaluable"]
    assert out["area"] < 0.95
    assert isinstance(out["pass"], (bool, np.bool_))


def test_injection_size_is_level_times_the_median_sunny_midday_reading() -> None:
    n = 48 * 300
    stamp = (np.datetime64("2025-01-01", "us") + np.arange(n) * np.timedelta64(30, "m")).astype(
        np.int64
    )
    k = np.ones(n)
    y = np.full(n, -3.0)
    inc = worlds.injection_increment(y_pct=y, k=k, stamp_us=stamp, level=2.0)
    assert inc == pytest.approx(6.0)
    with pytest.raises(ValueError, match="sunny summer-midday"):
        worlds.injection_increment(y_pct=y, k=np.zeros(n), stamp_us=stamp)


def test_inject_and_fold_and_the_two_worlds_share_their_dropped_rows() -> None:
    n = 48 * 300
    stamp = (np.datetime64("2025-01-01", "us") + np.arange(n) * np.timedelta64(30, "m")).astype(
        np.int64
    )
    rng = np.random.default_rng(3)
    power = rng.normal(5.0, 1.0, (16, n))
    k = np.ones((16, n))
    farm = np.clip(np.sin(np.pi * (np.arange(n) % 48) / 48.0), 0.0, 1.0)
    gap_farm = farm.copy()
    gap_farm[100:110] = np.nan
    sites = [f"S{i}" for i in range(16)]
    sites[5], sites[10] = core.HIDDEN_PAIR
    for j, name in enumerate(core.VISIBLE_SIGNED):
        sites[j] = name
    farms = {s: (gap_farm if i == 0 else farm) for i, s in enumerate(sites)}
    w2 = worlds.build_world_arrays(
        world="W2", sites=sites, power_pct=power, k=k, stamp_us=stamp, farm_for_site=farms
    )
    w0 = worlds.build_world_arrays(
        world="W0", sites=sites, power_pct=power, k=k, stamp_us=stamp, farm_for_site=farms
    )
    np.testing.assert_array_equal(w2.drop, w0.drop)
    assert w2.drop[100:110].all()
    i5 = sites.index(core.HIDDEN_PAIR[0])
    assert w2.increment_pct[i5] > 0.0
    assert w0.increment_pct[i5] == 0.0
    iv = sites.index(core.VISIBLE_SIGNED[0])
    np.testing.assert_array_equal(w2.signed[iv], w0.signed[iv])
    assert np.isnan(w2.signed_view[i5, 100])


def test_the_figure_guard_refuses_two_worlds_on_shared_dates() -> None:
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([np.nan, np.nan, 3.0])
    worlds.assert_no_two_worlds_on_shared_dates(series={"host": a, "injected": b})
    with pytest.raises(ValueError, match="difference"):
        worlds.assert_no_two_worlds_on_shared_dates(series={"host": a, "truth": a})


def test_holdout_guard() -> None:
    ok = np.array([np.datetime64("2026-06-30", "us").astype(np.int64)])
    bad = np.array([np.datetime64("2026-07-01", "us").astype(np.int64)])
    worlds.holdout_guard(valid_us=ok)
    with pytest.raises(ValueError, match="2026-07-01"):
        worlds.holdout_guard(valid_us=bad)


def test_primary_gate_opens_on_an_exporter_and_stays_shut_on_a_non_exporter() -> None:
    n = 48 * 200
    stamp = (np.datetime64("2025-04-01", "us") + np.arange(n) * np.timedelta64(30, "m")).astype(
        np.int64
    )
    u = np.clip(np.sin(np.pi * (np.arange(n) % 48) / 48.0), 0.0, 1.0)
    k = np.ones(n)
    exporter = 5.0 - 40.0 * u
    non_exporter = np.full(n, 20.0) + np.random.default_rng(0).normal(0, 1.0, n)
    assert worlds.primary_gate_open(reading_points=exporter, u=u, k=k, valid_us=stamp)["open"]
    assert not worlds.primary_gate_open(reading_points=non_exporter, u=u, k=k, valid_us=stamp)[
        "open"
    ]
    few = worlds.primary_gate_open(
        reading_points=exporter[:50], u=u[:50], k=k[:50], valid_us=stamp[:50]
    )
    assert not few["open"]
