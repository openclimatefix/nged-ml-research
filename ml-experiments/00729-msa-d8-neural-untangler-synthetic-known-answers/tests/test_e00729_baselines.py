"""Tests of the baselines P0, P0f, B1, B2, and M2's sign prior."""

import e00729_baselines as base
import e00729_core as core
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
