"""The job-level functions of jobs 2 and 3 on synthetic worlds (no real data, no `torch`)."""

import e00722_core as core
import e00722_d8core as d8core
import e00722_guard as guard
import e00722_u2a as u2a
import e00722_u2b as u2b
import numpy as np
import pytest
from e00722_testkit import block_flow, calendar, make_weights


def test_the_engine_test_recovers_minus_one_and_its_circular_placebos_include_zero():
    n_days = 100
    cal = calendar(n_days=n_days)
    n = n_days * 48
    rng = np.random.default_rng(14)
    record = np.full(n, np.nan)
    a, b = 1000, 1000 + 44 * 48
    record[a:b] = np.cumsum(rng.normal(0, 0.8, b - a)) + 10.0
    residual = -np.nan_to_num(record) + rng.normal(0, 0.2, n)
    out = u2a.engine_test(
        residual=residual,
        d8_reading=np.zeros(n),
        record=record,
        key=cal["key"],
        quiet=cal["quiet"],
        day=cal["day"],
        weights=make_weights(n_days=n_days),
    )
    m = out["main"]
    assert abs(m.coef[0] + 1.0) < 0.15
    assert m.n_days <= 46
    shifted = {d: (f.lo[0], f.hi[0]) for d, f in out["placebos"].items()}
    assert set(shifted) == {7, 14}
    assert all(abs(f.coef[0]) < 0.5 for f in out["placebos"].values())
    assert np.isclose(out["all_children"].coef[0], m.coef[0] - 0.0, atol=2.0)


def test_the_folded_child_test_reads_the_hidden_sign_and_the_placebo_does_not():
    n_days = 150
    cal = calendar(n_days=n_days)
    n = n_days * 48
    rng = np.random.default_rng(15)
    flow = block_flow(n=n, rng=rng)
    residual = flow + 3.0 + rng.normal(0, 0.02, n)
    out = u2a.folded_child_test(
        flow=flow,
        leave_out_residual=residual,
        key=cal["key"],
        quiet=cal["quiet"],
        day=cal["day"],
        weights=make_weights(n_days=n_days, n_boot=100),
        gate=guard.TimingGate(cap_core_seconds=1e9),
        total_fits=10,
    )
    assert abs(out["closure"].coef[0] - 1.0) < 0.1
    assert out["auc"][0] > 0.95
    assert out["n_scorable"] > core.D15_MIN_HOURS
    assert out["n_crossing_excluded"] > 0
    assert out["auc_placebo"][0] < 0.8
    assert 0.0 < out["reach"] <= 1.0
    all_slope = out["closure_all_children"].coef[0]
    assert all_slope < out["closure"].coef[0]


def test_hour_end_rows_are_the_odd_grid_rows():
    rows = u2a.hour_end_rows(n=6)
    assert rows.tolist() == [False, True, False, True, False, True]


def test_the_fast_sky_change_is_zero_in_the_dark_and_nan_without_cams():
    du = np.array([np.nan, np.nan, 0.5])
    zen = np.array([100.0, np.nan, 40.0])
    got = u2b.fill_sky_change(du=du, zenith=zen)
    assert got[0] == 0.0
    assert np.isnan(got[1])
    assert got[2] == 0.5


def test_calibration_rows_are_dark_half_hours_and_overcast_days():
    zen = np.array([100.0, 60.0, 60.0, np.nan])
    cls = np.array([0, 3, 1, 3])
    got = u2b.calibration_rows(zenith=zen, class_of_row=cls)
    assert got.tolist() == [True, True, False, False]


def test_day_shares_by_post_bounce_count():
    counts = np.array([0, 1, 1, 2, 3, 5, -1])
    shares = u2b.day_count_shares(counts=counts, eligible=np.ones(7, dtype=bool))
    assert shares["0"] == pytest.approx(1 / 6)
    assert shares["1"] == pytest.approx(2 / 6)
    assert shares["3_or_more"] == pytest.approx(2 / 6)


def test_the_exploratory_ratio_is_one_where_the_residual_follows_the_reading():
    rng = np.random.default_rng(16)
    reading = rng.normal(size=200)
    est, lo, hi, n = u2b.ratio_interval(num=reading, den=reading, rng=rng, n_boot=200)
    assert est == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
    assert n == 200
    third = u2b.lower_reading_third(reading=np.arange(60.0), events=np.arange(8, 52, 4))
    assert set(third.tolist()) == {0, 1, 2}


def test_excess_steps_subtract_the_matched_quiet_mean():
    n = 500
    x = np.zeros(n)
    x[100:104] = 4.0
    step = d8core.step_vec(x=x)
    key = np.zeros(n, dtype=np.int64)
    cand = {0: np.arange(200, 300)}
    got = u2b.excess_steps(step=step, events=np.array([100]), key=key, cand=cand)
    assert got[0] == pytest.approx(4.0)


def test_a_step_vector_needs_all_eight_values():
    x = np.arange(20.0)
    x[10] = np.nan
    step = d8core.step_vec(x=x)
    assert np.isnan(step[7:14]).all()
    assert step[5] == pytest.approx(4.0)
