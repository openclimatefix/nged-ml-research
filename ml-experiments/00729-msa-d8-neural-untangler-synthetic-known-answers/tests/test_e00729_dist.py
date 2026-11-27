"""Known-answer tests of the exact mixture CRPS, the mixture quantiles, and the sign metrics."""

import e00729_core as core
import e00729_dist as dist
import numpy as np
import pytest
from scipy import integrate, stats


def lobes(*, p: float, mu_i: float, s_i: float, mu_r: float, s_r: float) -> dist.Lobes:
    one = np.ones(1)
    return dist.Lobes(p * one, mu_i * one, s_i * one, mu_r * one, s_r * one)


def numeric_crps(*, p: float, mu_i: float, s_i: float, mu_r: float, s_r: float, y: float) -> float:
    def cdf(z: float) -> float:
        return (1 - p) * stats.norm.cdf(z, mu_i, s_i) + p * stats.norm.cdf(z, mu_r, s_r)

    def f(z: float) -> float:
        return (cdf(z) - (1.0 if z >= y else 0.0)) ** 2

    lo, hi = -80.0, 80.0
    return sum(
        integrate.quad(f, a, b, points=[y] if a < y < b else None, limit=200)[0]
        for a, b in [(lo, y), (y, hi)]
    )


@pytest.mark.parametrize("y", [-12.0, -3.0, 0.2, 6.0, 30.0])
def test_mixture_crps_matches_numerical_integration(y: float) -> None:
    args = {"p": 0.3, "mu_i": 8.0, "s_i": 1.5, "mu_r": -6.0, "s_r": 2.0}
    got = dist.mixture_crps(lobes=lobes(**args), y=np.array([y]))[0]
    assert got == pytest.approx(numeric_crps(**args, y=y), abs=1e-5)


def test_gaussian_floor_crps_is_0_117_at_zero_error() -> None:
    one = lobes(p=0.0, mu_i=5.0, s_i=0.5, mu_r=-5.0, s_r=0.5)
    assert dist.mixture_crps(lobes=one, y=np.array([5.0]))[0] == pytest.approx(0.117, abs=1e-3)


def test_point_distribution_scores_the_absolute_error() -> None:
    point = lobes(p=0.0, mu_i=4.0, s_i=dist.POINT_SPREAD, mu_r=-4.0, s_r=dist.POINT_SPREAD)
    assert dist.mixture_crps(lobes=point, y=np.array([4.3]))[0] == pytest.approx(0.3, abs=1e-4)
    assert dist.mixture_crps(lobes=point, y=np.array([-4.0]))[0] == pytest.approx(8.0, abs=1e-4)


def test_sign_probability_lowers_crps_on_the_true_side() -> None:
    y = np.array([-5.0])
    sure = lobes(p=0.9, mu_i=5.0, s_i=0.5, mu_r=-5.0, s_r=0.5)
    wrong = lobes(p=0.1, mu_i=5.0, s_i=0.5, mu_r=-5.0, s_r=0.5)
    assert dist.mixture_crps(lobes=sure, y=y)[0] < dist.mixture_crps(lobes=wrong, y=y)[0]


def test_mixture_quantiles_invert_the_cdf() -> None:
    d = lobes(p=0.4, mu_i=7.0, s_i=1.0, mu_r=-4.0, s_r=2.0)
    q = dist.mixture_quantiles(lobes=d)[0]
    assert np.all(np.diff(q) > 0)
    for tau, value in zip(core.QUANTILE_LEVELS, q, strict=True):
        assert dist.mixture_cdf(lobes=d, z=np.array([value]))[0] == pytest.approx(tau, abs=1e-6)


def test_prob_negative_is_the_mass_below_zero() -> None:
    d = lobes(p=0.25, mu_i=3.0, s_i=1.0, mu_r=-3.0, s_r=1.0)
    expected = 0.75 * stats.norm.cdf(-3.0) + 0.25 * stats.norm.cdf(3.0)
    assert dist.prob_negative(lobes=d)[0] == pytest.approx(expected)


def test_quantile_crps_matches_numerical_integration() -> None:
    q = np.array([[-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0]])
    taus = np.asarray(core.QUANTILE_LEVELS)
    for y in (-3.0, -1.2, 0.1, 1.9, 4.0):

        def cdf(z: float) -> float:
            return float(np.interp(z, q[0], taus, left=0.0, right=1.0))

        def f(z: float, y: float = y) -> float:
            return (cdf(z) - (1.0 if z >= y else 0.0)) ** 2

        pts = [*q[0].tolist(), y]
        expected = integrate.quad(f, -10, 10, points=sorted(pts), limit=400)[0]
        assert dist.quantile_crps(q=q, y=np.array([y]))[0] == pytest.approx(expected, abs=1e-5)


def test_pinball_points_is_twice_the_mean_loss() -> None:
    q = np.tile(np.arange(11.0), (1, 1))
    got = dist.pinball_points(q=q, y=np.array([5.0]))[0]
    taus = np.asarray(core.QUANTILE_LEVELS)
    err = 5.0 - np.arange(11.0)
    assert got == pytest.approx(2 * np.mean(np.maximum(taus * err, (taus - 1) * err)))


def test_reliability_term_zero_for_calibrated_and_positive_for_overconfident() -> None:
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, 20_000)
    calibrated = rng.uniform(size=prob.size) < prob
    rel = dist.reliability_term(prob=prob, observed=calibrated)
    assert rel["reliability"] < 0.002  # ty: ignore[unsupported-operator]
    assert rel["max_gap"] < 0.1  # ty: ignore[unsupported-operator]
    bad = dist.reliability_term(prob=np.full(1000, 0.9), observed=np.arange(1000) % 2 == 0)
    assert bad["reliability"] == pytest.approx(0.16, abs=1e-6)
    assert bad["max_gap"] == pytest.approx(0.4, abs=1e-6)
    assert bad["reliability"] > core.RELIABILITY_BAR  # ty: ignore[unsupported-operator]


def test_lobe_coverage_picks_the_lobe_on_the_true_side() -> None:
    d = dist.Lobes(
        np.zeros(2),
        np.array([5.0, 5.0]),
        np.array([0.5, 0.5]),
        np.array([-5.0, -5.0]),
        np.array([0.5, 0.5]),
    )
    y = np.array([5.5, -5.0])
    assert dist.lobe_covers(lobes=d, y=y).tolist() == [True, True]
    assert dist.lobe_covers(lobes=d, y=np.array([7.0, -3.0])).tolist() == [False, False]
    assert dist.lobe_width(lobes=d, y=y)[0] == pytest.approx(2 * 1.6449 * 0.5, abs=1e-3)
