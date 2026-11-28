"""Known-answer tests of the forward model, the capacity priors, and the Student-t fit."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("39", "40", "653", "660"):
    for _d in HERE.parent.glob(f"{_n.zfill(5)}-*"):
        sys.path.append(str(_d))

import e00737_fit as fit
import e00737_forward as fwd
import e00737_inputs as inputs
from e00737_common import GRID_LENGTH, N_QUARTERS, RIDGE_SCALE_GRID, grid_end_times, quarter_index


def test_faiman_cell_temperature_known_answer() -> None:
    cell = fwd.faiman_cell_temperature_c(
        poa_w_m2=np.array([800.0]), air_c=np.array([20.0]), wind_ms=np.array([2.0])
    )
    assert cell[0] == pytest.approx(20.0 + 800.0 / (25.0 + 6.84 * 2.0))
    assert cell[0] == pytest.approx(40.68, abs=0.01)


def test_derate_is_applied_before_the_clip() -> None:
    # 1000 W/m2, 5 C air, no wind: the cell is at 45 C, so the derate is 1 - 0.004 * 20 = 0.92.
    kwargs = {
        "poa_w_m2": np.array([1000.0]),
        "air_c": np.array([5.0]),
        "wind_ms": np.array([0.0]),
    }
    clipped_after = fwd.derated_ac_unit_output(dc_ac_ratio=1.0, **kwargs)[0]
    clipped_before = fwd.derated_ac_unit_output(dc_ac_ratio=1.5, **kwargs)[0]
    assert clipped_after == pytest.approx(0.92)
    assert clipped_before == pytest.approx(1.0)  # 1.5 x 0.92 = 1.38, then clipped to 1


def test_unit_output_is_never_negative_and_zero_in_the_dark() -> None:
    out = fwd.derated_ac_unit_output(
        poa_w_m2=np.array([0.0, 50.0]),
        air_c=np.array([10.0, 10.0]),
        wind_ms=np.array([1.0, 1.0]),
        dc_ac_ratio=1.15,
    )
    assert out[0] == 0.0
    assert 0.0 < out[1] < 1.0


def test_interpolation_midpoint_and_range() -> None:
    hours = np.array(["2022-01-01T00:00", "2022-01-01T01:00"], dtype="datetime64[us]")
    at = np.array(
        ["2022-01-01T00:30", "2022-01-01T01:00", "2022-01-01T02:00"], dtype="datetime64[us]"
    )
    out = fwd.interpolate_hourly(hourly_times=hours, values=np.array([4.0, 8.0]), at_times=at)
    assert out[0] == pytest.approx(6.0)
    assert out[1] == pytest.approx(8.0)
    assert np.isnan(out[2])


def test_trailing_mean_matches_a_direct_loop() -> None:
    values = np.arange(10.0) ** 2
    out = fwd.trailing_mean(values=values, window=3)
    assert out[5] == pytest.approx(values[3:6].mean())


def test_quarter_index_edges() -> None:
    times = np.array(
        ["2022-01-01T00:30", "2022-03-31T23:30", "2022-04-01T00:30", "2026-07-01T00:00"],
        dtype="datetime64[us]",
    )
    assert quarter_index(end_time=times).tolist() == [0, 0, 1, N_QUARTERS - 1]
    index = quarter_index(end_time=grid_end_times())
    assert len(index) == GRID_LENGTH
    assert index.min() == 0
    assert index.max() == N_QUARTERS - 1
    assert (np.diff(index) >= 0).all()


def test_cut_is_refused_at_and_after_the_cut() -> None:
    late = np.array(["2026-07-01T00:30"], dtype="datetime64[us]")
    with pytest.raises(ValueError, match="probe"):
        inputs.assert_cut(times=late, what="probe")
    inputs.assert_cut(times=np.array(["2026-07-01T00:00"], dtype="datetime64[us]"), what="probe")


def test_quarter_usable_rule() -> None:
    quarter = np.repeat(np.array([0, 1, 2]), 10)
    sun_up = np.ones(30, dtype=bool)
    sun_up[20:] = False  # no daylight in quarter 2
    usable = np.zeros(30, dtype=bool)
    usable[:6] = True  # 6 of 10 in quarter 0
    usable[10:15] = True  # 5 of 10 in quarter 1
    share = inputs.quarter_usable_share(quarter=quarter, sun_up=sun_up, usable_daylight=usable)
    assert share[0] == pytest.approx(0.6)
    assert share[1] == pytest.approx(0.5)
    assert np.isnan(share[2])
    scored = inputs.scored_quarters(share=share)
    assert scored[:3].tolist() == [True, False, False]


def test_random_walk_prior_is_the_dense_chain() -> None:
    sd = 0.3
    prior = fit.random_walk_prior(growth_sd=sd)
    p = prior.precision
    assert p.shape == (N_QUARTERS, N_QUARTERS)
    assert p[0, 0] == pytest.approx(1 / 4 + 1 / sd**2)
    assert p[5, 5] == pytest.approx(2 / sd**2)
    assert p[-1, -1] == pytest.approx(1 / sd**2)
    assert p[3, 4] == pytest.approx(-1 / sd**2)
    assert p[0, 5] == 0.0
    cov = np.linalg.inv(p)
    for k in (0, 1, 7, 17):
        assert cov[k, k] == pytest.approx(4.0 + k * sd**2)
    assert prior.mean[0] == pytest.approx(2.0)
    assert np.diff(prior.mean) == pytest.approx(0.10)


def test_flat_and_independent_priors() -> None:
    flat = fit.flat_prior()
    assert flat.precision.shape == (1, 1)
    assert set(flat.group.tolist()) == {0}
    assert len(flat.group) == N_QUARTERS
    qt = fit.independent_prior()
    assert qt.precision[0, 0] == pytest.approx(1 / 25)
    assert qt.group.tolist() == list(range(N_QUARTERS))


def _synthetic(
    *,
    truth_per_quarter: np.ndarray,
    outliers: bool,
    seed: int = 0,
    days_per_quarter: int = 30,
    noise_kind: str = "gaussian",
) -> dict:
    """Build a regression problem whose answer is the quarterly capacity `truth_per_quarter`.

    `noise_kind` is `"gaussian"` (standard deviation 0.05) or `"student_t"` (4 degrees of freedom,
    scale 0.05).
    """
    rng = np.random.default_rng(seed)
    n = N_QUARTERS * days_per_quarter * 48
    rows = np.arange(n)
    day = rows // 48
    hour = (rows % 48) / 2.0
    cloud = rng.uniform(0.3, 1.0, size=n // 48)[day]
    solar = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None) * cloud
    quarter = day // days_per_quarter
    base = np.column_stack([np.ones(n), np.cos(2 * np.pi * hour / 24.0)])
    beta = np.array([0.5, 0.2])
    noise = (
        0.05 * rng.standard_t(4.0, size=n)
        if noise_kind == "student_t"
        else rng.normal(0.0, 0.05, size=n)
    )
    if outliers:
        spikes = rng.random(n) < 0.01
        noise[spikes] += rng.choice([-1.0, 1.0], size=int(spikes.sum())) * 2.0
    y = base @ beta - truth_per_quarter[quarter] * solar + noise
    return {
        "y": y,
        "base": base,
        "solar": solar,
        "quarter": quarter,
        "daylight": solar > 0.0,
        "rows": rows,
        "day": day,
    }


def test_student_t_em_recovers_a_flat_capacity_despite_outliers() -> None:
    data = _synthetic(truth_per_quarter=np.full(N_QUARTERS, 3.0), outliers=True)
    result = fit.fit_student_t(prior=fit.flat_prior(), ridge_abs=1e-6, **data)
    assert result.converged
    assert result.capacity[0] == pytest.approx(3.0, abs=0.05)
    assert np.all(result.sd > 0.0)
    assert result.infl >= 1.0 - 1e-9
    ols_x = np.column_stack([data["base"], -data["solar"]])
    ols = np.linalg.lstsq(ols_x, data["y"], rcond=None)[0][-1]
    assert abs(result.capacity[0] - 3.0) <= abs(ols - 3.0) + 1e-3


def test_random_walk_fit_recovers_a_ramp() -> None:
    truth = np.linspace(1.0, 5.0, N_QUARTERS)
    data = _synthetic(truth_per_quarter=truth, outliers=False)
    result = fit.fit_student_t(prior=fit.random_walk_prior(growth_sd=0.3), ridge_abs=1e-6, **data)
    assert result.capacity == pytest.approx(truth, abs=0.1)
    # The posterior standard deviation is small against the ramp's range, and no quarter is NaN.
    assert np.all(np.isfinite(result.sd))
    assert result.sd.max() < 0.5


def test_flat_prior_cannot_follow_a_ramp_but_the_independent_prior_can() -> None:
    truth = np.linspace(1.0, 5.0, N_QUARTERS)
    data = _synthetic(truth_per_quarter=truth, outliers=False)
    flat = fit.fit_student_t(prior=fit.flat_prior(), ridge_abs=1e-6, **data)
    free = fit.fit_student_t(prior=fit.independent_prior(), ridge_abs=1e-6, **data)
    assert np.abs(flat.capacity - truth).mean() > 0.5
    assert np.abs(free.capacity - truth).mean() < 0.1


def test_posterior_matches_a_dense_solve_of_the_weighted_normal_equations() -> None:
    truth = np.linspace(2.0, 3.0, N_QUARTERS)
    data = _synthetic(truth_per_quarter=truth, outliers=False, seed=3)
    prior = fit.random_walk_prior(growth_sd=0.3)
    result = fit.fit_student_t(
        prior=prior, ridge_abs=1e-6, autocorrelation=False, max_iter=1, **data
    )
    n_base = data["base"].shape[1]
    x = np.zeros((len(data["y"]), n_base + N_QUARTERS))
    x[:, :n_base] = data["base"]
    x[np.arange(len(x)), n_base + data["quarter"]] = -data["solar"]
    start_day, start_night = fit.starting_sigma(
        y=data["y"], rows=data["rows"], daylight=data["daylight"]
    )
    sigma = np.where(data["daylight"], start_day, start_night)
    lam = 1.0 / sigma**2
    gram = x.T @ (x * lam[:, None])
    gram[:n_base, :n_base] += np.eye(n_base) * 1e-6 / start_day**2
    gram[n_base:, n_base:] += prior.precision
    rhs = x.T @ (lam * data["y"])
    rhs[n_base:] += prior.precision @ prior.mean
    solution = np.linalg.solve(gram, rhs)
    assert result.capacity == pytest.approx(solution[n_base:], abs=1e-6)
    # The standard deviation is the sandwich A^-1 B A^-1 of the Student-t score, computed densely.
    resid = data["y"] - x @ solution
    scale2 = 4.0 * sigma**2
    denom = scale2 + resid**2
    psi = 5.0 * resid / denom
    dpsi = 5.0 * (scale2 - resid**2) / denom**2
    info = x.T @ (x * dpsi[:, None])
    info[:n_base, :n_base] += np.eye(n_base) * 1e-6 / start_day**2
    info[n_base:, n_base:] += prior.precision
    inverse = np.linalg.inv(info)
    cov = inverse @ (x.T @ (x * (psi**2)[:, None])) @ inverse
    assert result.sandwich
    assert result.sd == pytest.approx(np.sqrt(np.diag(cov)[n_base:]), rel=1e-6)


def _sd_ratio(*, noise_kind: str, draws: int = 300) -> float:
    """Return the mean reported sd over the empirical sd of the capacity estimates of the draws."""
    estimates, reported = [], []
    for seed in range(draws):
        data = _synthetic(
            truth_per_quarter=np.full(N_QUARTERS, 3.0),
            outliers=False,
            seed=seed,
            days_per_quarter=6,
            noise_kind=noise_kind,
        )
        result = fit.fit_student_t(
            prior=fit.flat_prior(), ridge_abs=1e-6, autocorrelation=False, **data
        )
        estimates.append(result.capacity[0])
        reported.append(result.sd[0])
    return float(np.mean(reported) / np.std(estimates, ddof=1))


def test_reported_sd_matches_the_empirical_sd_on_gaussian_noise() -> None:
    # The Student-t scale is smaller than the Gaussian standard deviation, so the covariance
    # (X' W X)^-1 would understate the sd here by about 20%. The sandwich does not.
    assert _sd_ratio(noise_kind="gaussian") == pytest.approx(1.0, abs=0.10)


def test_reported_sd_is_not_inflated_on_student_t_noise() -> None:
    assert _sd_ratio(noise_kind="student_t") <= 1.15
    assert _sd_ratio(noise_kind="student_t") >= 0.85


def test_lag1_inflation_is_one_for_white_noise_and_large_for_an_ar1_series() -> None:
    rng = np.random.default_rng(1)
    n = 48 * 60
    rows = np.arange(n)
    day = rows // 48
    daylight = (rows % 48 >= 12) & (rows % 48 < 36)
    white = rng.normal(size=n)
    ar1 = np.zeros(n)
    shocks = rng.normal(size=n)
    for i in range(1, n):
        ar1[i] = 0.8 * ar1[i - 1] + shocks[i]
    low = fit.lag1_inflation(z=white, rows=rows, day=day, daylight=daylight)
    high = fit.lag1_inflation(z=ar1, rows=rows, day=day, daylight=daylight)
    assert low < 1.3
    assert high > 3.0


def _collinear_world(*, truth: float) -> dict:
    """Build a noise-free world whose baseline holds a column that is almost the solar column."""
    rng = np.random.default_rng(5)
    data = _synthetic(truth_per_quarter=np.full(N_QUARTERS, truth), outliers=False)
    solar = data["solar"]
    extra = solar + 0.05 * rng.uniform(size=len(solar)) * (solar > 0.0)
    data["base"] = np.column_stack([data["base"], extra])
    data["y"] = data["base"] @ np.array([0.5, 0.2, 0.3]) - truth * solar
    return data


def _capacity_at_ridge_scale(*, data: dict, scale: float) -> float:
    ridge_abs = scale * inputs.RIDGE_PER_COLUMN * float((data["base"] ** 2).sum(axis=0).mean())
    return float(fit.fit_student_t(prior=fit.flat_prior(), ridge_abs=ridge_abs, **data).capacity[0])


def test_smallest_ridge_scale_recovers_a_noise_free_capacity_and_the_default_shrinks_it() -> None:
    truth = 3.0
    data = _collinear_world(truth=truth)
    smallest = min(RIDGE_SCALE_GRID)
    assert _capacity_at_ridge_scale(data=data, scale=smallest) == pytest.approx(truth, rel=0.01)
    share_at_one = _capacity_at_ridge_scale(data=data, scale=1.0) / truth
    share_at_smallest = _capacity_at_ridge_scale(data=data, scale=smallest) / truth
    assert share_at_smallest > share_at_one
    assert share_at_one < 0.95


@dataclass
class _StubData:
    base: np.ndarray
    ridge_abs: float
    rows: np.ndarray
    quarter: np.ndarray
    daylight: np.ndarray
    day: np.ndarray


def test_ridge_scale_reaches_the_fit_of_every_arm_that_carries_the_baseline_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float] = []

    def stub(**kwargs: object) -> str:
        seen.append(float(kwargs["ridge_abs"]))  # ty: ignore[invalid-argument-type]
        return "fit"

    monkeypatch.setattr(fit, "fit_student_t", stub)
    n = 96
    rows = np.arange(n)
    data = _StubData(
        base=np.ones((n, fit.SEASONAL_COLUMNS)),
        ridge_abs=2.0,
        rows=rows,
        quarter=np.zeros(n, dtype=int),
        daylight=np.ones(n, dtype=bool),
        day=rows // 48,
    )
    forecast = np.zeros(n)
    model = SimpleNamespace(
        data=data,
        terms={fit.PRIOR_DC_AC: np.ones(n)},
        inputs=SimpleNamespace(forecast=forecast),
    )
    power = np.ones(n)
    prior = fit.flat_prior()
    assert fit.fit_arm(model=model, power=power, prior=prior, ridge_scale=0.1) == "fit"  # ty: ignore[invalid-argument-type]
    assert seen[-1] == pytest.approx(0.2)
    fit.fit_daily(model=model, power=power, prior=prior, ridge_scale=0.1)  # ty: ignore[invalid-argument-type]
    unscaled = inputs.ridge_precision(seasonal=np.add.reduceat(data.base, [0, 48], axis=0))
    assert seen[-1] == pytest.approx(0.1 * unscaled)
    fit.fit_residual_arm(model=model, power=power, prior=prior, ridge_scale=0.1)  # ty: ignore[invalid-argument-type]
    # Every row is in quarter 0, so the mean squared column norm over 18 columns is n / 18.
    assert seen[-1] == pytest.approx(0.1 * inputs.RIDGE_PER_COLUMN * n / N_QUARTERS)
