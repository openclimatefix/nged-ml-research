# ruff: noqa: E501
"""Tests of the baselines P0, P0f, B1, B2, and M2's sign prior."""

import json
import warnings
from pathlib import Path
from types import ModuleType

import e00729_baselines as base
import e00729_core as core
import e00729_paths as paths
import numpy as np
import pytest


def feats(n: int = 96, *, r: float = 20.0) -> base.Features:
    hh = np.arange(n) % 48
    return base.Features(
        r=np.full(n, r), local_hh=hh, local_month=np.full(n, 6), weekend=np.zeros(n, dtype=bool),
        weekday=(np.arange(n) // 48) % 7, holiday=np.zeros(n, dtype=bool), t2m=np.full(n, 15.0),
    )  # fmt: skip


def test_p0_reads_the_reading_as_import_after_removing_the_reactive_floor() -> None:
    f = feats(r=20.0)
    x = base.reading_to_import(r=f.r)
    assert np.allclose(x, np.sqrt(400.0 - core.QHAT_POINTS**2))
    assert base.reading_to_import(r=np.array([0.0]))[0] == 0.0
    lobes = base.p0(f=f)
    assert np.all(lobes.p == 0.0)
    assert np.allclose(lobes.mu_i, x)
    assert np.allclose(lobes.mu_r, -x)


def test_p0f_differs_from_p0_only_in_the_spread() -> None:
    f = feats()
    a, b = base.p0(f=f), base.p0f(f=f)
    assert np.allclose(a.mu_i, b.mu_i)
    assert np.all(b.s_i == core.FLOOR_POINTS)
    assert np.all(a.s_i < b.s_i)


def test_climatology_probability_is_the_empirical_cdf_in_each_cell() -> None:
    f = feats(48 * 40)
    demand = np.tile(np.arange(40, dtype=float), 48)
    demand = np.repeat(np.arange(40, dtype=float), 48)
    clim = base.Climatology.fit(demand=demand, f=f)
    low = base.Climatology.prob_below(clim, r=np.full(f.r.shape, -5.0), f=f)
    high = clim.prob_below(r=np.full(f.r.shape, 500.0), f=f)
    mid = clim.prob_below(r=np.full(f.r.shape, 19.5), f=f)
    assert np.all(low < 0.05)
    assert np.all(high > 0.95)
    assert np.allclose(mid, 0.5, atol=0.03)
    unseen = base.Features(**{**f.__dict__, "local_month": np.full(len(f.r), 1)})
    assert np.all(clim.prob_below(r=f.r, f=unseen) == 0.5)


def test_regression_recovers_a_known_demand_curve_and_its_cdf() -> None:
    f = feats(48 * 60)
    demand = 10.0 + 0.2 * f.local_hh + 0.3 * (f.t2m - 15.0)
    reg = base.Regression.fit(demand=demand, f=f)
    p_at = reg.prob_below(r=demand + 5 * reg.sd, f=f)
    assert np.all(p_at > 0.999)
    assert np.all(reg.prob_below(r=demand - 5 * reg.sd, f=f) < 0.001)
    assert np.allclose(reg.prob_below(r=demand, f=f), 0.5, atol=1e-6)
    demand[:10] = np.nan
    assert np.isfinite(base.Regression.fit(demand=demand, f=f).sd)


def test_sign_prior_scales_the_probability_by_pi_and_clips() -> None:
    f = feats(4)
    lobes = base.sign_prior_lobes(f=f, prob_below=np.array([0.0, 0.5, 1.0, 2.0]), pi=0.3)
    assert np.allclose(lobes.p, [0.0, 0.15, 0.3, 0.6])
    assert np.all(lobes.s_i == core.FLOOR_POINTS)
    assert np.all(base.sign_prior_lobes(f=f, prob_below=np.full(4, 9.0), pi=0.3).p <= 1.0)
    assert base.injected_share(mix=core.TEST_MIX) == pytest.approx(0.85)
    m2 = base.m2_lobes(f=f, prob_below=np.full(4, 0.5), p_any=np.array([0.0, 0.2, 1.0, 1.0]))
    assert np.allclose(m2.p, [0.0, 0.1, 0.5, 0.5])


@pytest.fixture
def tracker_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ModuleType, ModuleType]:
    """Return experiment 694's `e00694_common` and `e00694_trackers`, with invented private constants.

    Importing experiment 694 reads a private constants file. When none is present, the fixture writes
    a file of invented values (a unit size and a connection date that mean nothing) and points the
    reader at it. The test skips when experiment 694's folder is absent.
    """
    if not paths.sibling_available(prefix="00694-"):
        pytest.skip("experiment 694 is not available")
    constants = tmp_path / "constants.json"
    constants.write_text(
        json.dumps({"d8_engine_unit_mw": 10.0, "d8_engine_connected_date": "2022-01-01"})
    )
    monkeypatch.setenv("ENGINE_CONSTANTS_PATH", str(constants))
    import e00694_common as c694  # ty: ignore[unresolved-import]
    import e00694_trackers as t694  # ty: ignore[unresolved-import]

    return c694, t694


def folded_series(*, days: int, seed: int = 0) -> np.ndarray:
    """Return a half-hourly series of folded readings in points: demand plus a running engine."""
    rng = np.random.default_rng(seed)
    n = 48 * days
    demand = 40.0 + 10.0 * np.sin(np.arange(n) * 2 * np.pi / 48.0) + rng.normal(0, 0.5, n)
    engine = np.where((np.arange(n) // 24) % 3 == 0, 30.0, 0.0)
    return np.abs(demand - engine)


def test_the_local_causal_spread_equals_experiment_694s_on_a_series_of_28_days_or_more(
    tracker_modules: tuple[ModuleType, ModuleType],
) -> None:
    c694, _ = tracker_modules
    for days in (28, 60):
        y = folded_series(days=days) / 100.0 * 10.379
        ours = base.causal_spread_safe(y=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            theirs = c694.causal_spread(y=y)
        assert np.array_equal(np.isnan(ours), np.isnan(theirs))
        assert np.allclose(ours[np.isfinite(ours)], theirs[np.isfinite(theirs)])
        assert np.array_equal(
            base.fill_short_series_spread(y=y, spread=ours), ours, equal_nan=True
        )  # a long series is not filled


def test_m2_on_a_long_series_equals_experiment_694s_own_pipeline(
    tracker_modules: tuple[ModuleType, ModuleType],
) -> None:
    c694, t694 = tracker_modules
    reading = folded_series(days=60, seed=2)
    y = reading / core.POINTS * core.CAPACITY_MVA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lvl = c694.quantise_spread(spread=c694.causal_spread(y=y))
    expected = np.asarray(t694.grid_track(y=y, lvl=lvl).p_any_smooth)
    assert np.array_equal(base.m2_p_any(reading_points=reading), expected)


@pytest.mark.parametrize("days", [1, 3, 6, 10, 27])
def test_m2_on_a_short_series_no_longer_raises_and_returns_probabilities(
    days: int, tracker_modules: tuple[ModuleType, ModuleType]
) -> None:
    del tracker_modules  # the fixture makes experiment 694 importable
    p = base.m2_p_any(reading_points=folded_series(days=days, seed=days))
    assert p.shape == (48 * days,)
    assert np.isfinite(p).all()
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_the_short_series_fill_is_the_series_own_median_spread_with_a_floor() -> None:
    y = folded_series(days=10) / 100.0 * 10.379
    spread = base.causal_spread_safe(y=y)
    assert np.isnan(spread[: 48 * 7]).all()
    filled = base.fill_short_series_spread(y=y, spread=spread)
    own = max(1.4826 * float(np.median(np.abs(np.diff(y)))), base.SPREAD_FLOOR)
    assert np.allclose(filled[np.isnan(spread)], own)
    assert np.array_equal(filled[np.isfinite(spread)], spread[np.isfinite(spread)])
    flat = base.fill_short_series_spread(y=np.ones(100), spread=np.full(100, np.nan))
    assert np.allclose(flat, base.SPREAD_FLOOR)
