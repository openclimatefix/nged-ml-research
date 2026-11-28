"""Tests of the survival curves and the age rule A1."""
# ruff: noqa: D103, E402

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import e00738_common as common
import e00738_survival as surv


def test_generator_curve_is_a_survival_function() -> None:
    g = surv.GeneratorCurve()
    t = np.array([0.0, 2.5, 3.0, 14.0, 14.5, 27.0, 28.0, 112.0, 200.0])
    s = g.sf(t)
    assert s[0] == pytest.approx(1.0)
    assert np.all(np.diff(s) <= 1e-12)
    assert s[-1] == pytest.approx(0.0)
    # half of the shifts last 3 to 14 days, so about half survive to day 14.5
    assert g.sf(np.array([14.5]))[0] == pytest.approx(0.5)
    assert g.sf(np.array([27.0]))[0] == pytest.approx(0.5)


def test_ratio_is_one_at_zero_lead_and_falls_with_lead() -> None:
    g = surv.GeneratorCurve()
    lead = np.array([0.0, 1.0, 5.0, 20.0, 60.0])
    p = surv.ratio(curve=g, age_days=np.full(5, 5.0), lead_days=lead)
    assert p[0] == pytest.approx(1.0)
    assert np.all(np.diff(p) <= 1e-12)
    assert np.all((p >= 0) & (p <= 1))


def test_ratio_floors_the_denominator() -> None:
    g = surv.GeneratorCurve()
    # beyond the longest possible shift S(a) is 0; the floor keeps the ratio finite and in 0 to 1
    p = surv.ratio(curve=g, age_days=np.array([300.0]), lead_days=np.array([1.0]))
    assert np.isfinite(p).all()
    assert 0.0 <= p[0] <= 1.0


def test_fit_curve_recovers_a_known_exponential() -> None:
    d = np.random.default_rng(0).exponential(30.0, size=4000) + 1e-3
    c = surv.fit_curve(durations_days=d)
    assert c.sf(np.array([30.0]))[0] == pytest.approx(np.exp(-1.0), abs=0.03)


def test_fit_curve_rejects_a_non_positive_duration() -> None:
    with pytest.raises(AssertionError):
        surv.fit_curve(durations_days=np.array([1.0, 0.0, 3.0]))


def test_fit_to_generator_durations_meets_the_error_bar() -> None:
    rng = np.random.default_rng(1)
    short = rng.integers(3, 15, size=2000)
    long = rng.integers(28, 113, size=2000)
    c = surv.fit_curve(durations_days=np.r_[short, long].astype(float))
    err = surv.curve_error(curve=c)
    assert set(err) == {float(t) for t in common.CURVE_ERROR_DAYS}
    assert surv.curve_error_ok(curve=surv.GeneratorCurve())


def test_daily_hazards_reproduce_the_curve_at_whole_days() -> None:
    g = surv.GeneratorCurve()
    hz = surv.daily_hazards(curve=g)
    assert hz.shape == (common.N_HAZARDS,)
    assert np.all(hz >= 0)
    days = np.arange(common.N_HAZARDS + 1, dtype=float)
    want = g.sf(days)
    got = np.exp(-surv.cumulative_hazard(hazards=hz, t_days=days))
    np.testing.assert_allclose(got[:11], np.maximum(want[:11], 1e-12), rtol=1e-6)


def test_cumulative_hazard_is_linear_inside_a_day_and_continues_after_the_last() -> None:
    hz = np.linspace(0.1, 0.5, common.N_HAZARDS)
    h = surv.cumulative_hazard(hazards=hz, t_days=np.array([0.0, 0.5, 1.0, 10.0, 12.0]))
    assert h[0] == 0.0
    assert h[1] == pytest.approx(0.5 * hz[0])
    assert h[2] == pytest.approx(hz[0])
    assert h[3] == pytest.approx(hz.sum())
    assert h[4] == pytest.approx(hz.sum() + 2 * hz[-1])


def test_hazard_ratio_matches_the_curve_ratio_at_whole_days() -> None:
    g = surv.GeneratorCurve()
    hz = surv.daily_hazards(curve=g)
    age = np.array([2.0, 3.0, 5.0])
    lead = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        surv.hazard_ratio(hazards=hz, age_days=age, lead_days=lead),
        surv.ratio(curve=g, age_days=age, lead_days=lead),
        atol=1e-6,
    )


def test_noise_unit_is_the_scaled_mad_of_daily_differences() -> None:
    means = np.cumsum(np.random.default_rng(2).normal(0, 0.1, 5000))
    u = surv.noise_unit(row_means=means)
    assert u == pytest.approx(0.1, rel=0.1)
    assert surv.noise_unit(row_means=np.r_[np.nan, means]) == pytest.approx(u)


def _history(*, levels: list[float], n_visible: int = 48) -> tuple[np.ndarray, np.ndarray]:
    power = np.tile(np.array(levels, dtype=float)[:, None], (1, 48))[None]
    visible = np.zeros_like(power, dtype=bool)
    visible[..., :n_visible] = True
    return power, visible


def test_a1_sees_no_shift_in_a_flat_history() -> None:
    power, visible = _history(levels=[0.0] * 14)
    assert surv.age_rule_a1(hist_power=power, visible=visible, noise=0.1)[0] == 0.0


def test_a1_age_is_the_count_of_newest_rows_plus_two() -> None:
    # rows 0 to 2 (the newest) sit 1.0 above the reference of the oldest 7 rows
    power, visible = _history(levels=[1.0, 1.0, 1.0] + [0.0] * 11)
    age = surv.age_rule_a1(hist_power=power, visible=visible, noise=0.1)
    assert age[0] == 3 + common.MIN_AGE_DAYS


def test_a1_count_stops_at_a_row_without_data() -> None:
    power, visible = _history(levels=[1.0, 1.0, 1.0, 1.0] + [0.0] * 10)
    visible[0, 2, :] = False
    age = surv.age_rule_a1(hist_power=power, visible=visible, noise=0.1)
    assert age[0] == 2 + common.MIN_AGE_DAYS


def test_a1_needs_three_reference_rows() -> None:
    power, visible = _history(levels=[1.0] * 7 + [0.0] * 7)
    visible[0, 7:, :] = False
    visible[0, 7:9, :] = True  # only 2 reference rows with data
    assert surv.age_rule_a1(hist_power=power, visible=visible, noise=0.1)[0] == 0.0


def test_row_summaries_ignore_hidden_values() -> None:
    power, visible = _history(levels=[5.0] * 14, n_visible=10)
    power[..., 10:] = 99.0
    means, present = surv.row_summaries(hist_power=power, visible=visible)
    assert np.all(means == 5.0)
    assert present.all()
    means, present = surv.row_summaries(hist_power=power, visible=np.zeros_like(visible))
    assert np.all(means == 0.0)
    assert not present.any()
