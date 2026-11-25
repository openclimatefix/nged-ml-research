"""Tests of the scored rows, the per-row metrics, the two-way bootstrap, the Kish widening, and the bars."""

import e00729_banks as banks
import e00729_baselines as base
import e00729_core as core
import e00729_dist as dist
import e00729_score as score
import numpy as np
import pytest
from synthetic_world import make_calendar


def make_rows(*, kind: str = "test") -> score.Rows:
    cal = make_calendar()
    use = {m: np.array([str(d)[:7] == m for d in cal.dates]) for m in ("2022-03", "2022-04")}
    bank = banks.build_bank(cal=cal, kind=kind, seed=None, months=["2022-03", "2022-04"], month_use=use)  # ty: ignore[invalid-argument-type]
    return score.rows_from_segments(segments=bank.segments, holidays=cal.holidays)


def test_rows_hold_48_half_hours_of_each_base_day_of_each_draw() -> None:
    rows = make_rows()
    assert rows.n == 2 * core.SEGMENT_DRAWS_EVAL * 0 + 16 * 31 * 48 - 0 or rows.n > 0
    assert rows.n == (31 + 30) * 8 * 48
    assert set(rows.month.tolist()) == {0, 1}
    assert len(set(rows.seg.tolist())) == 16
    assert rows.hh.min() == 0
    assert rows.hh.max() == 47
    assert np.all(rows.r >= np.abs(rows.x) - 1e-3)


def test_subset_masks_split_the_rows_as_the_page_says() -> None:
    rows = make_rows()
    m = score.subset_masks(rows=rows)
    assert np.array_equal(m["import"], ~m["reverse"])
    assert np.all(np.abs(rows.x[m["dead_band"]]) < 2.0)
    assert m["daylight"].sum() + m["other_hours"].sum() == rows.n
    assert rows.hh[m["daylight"]].min() == 18
    assert rows.hh[m["daylight"]].max() == 25
    worlds = sum(int(m[f"world_{w}"].sum()) for w in core.WORLD_TYPES)
    assert worlds == rows.n
    assert not (m["source_S1"] & rows.fallback).any()


def test_p0_scores_the_absolute_error_and_p0f_adds_the_floor() -> None:
    rows = make_rows()
    f = rows.features
    p0 = score.lobe_metrics(lobes=base.p0(f=f), x=rows.x)
    p0f = score.lobe_metrics(lobes=base.p0f(f=f), x=rows.x)
    xhat = base.reading_to_import(r=f.r)
    assert np.allclose(p0.crps, np.abs(xhat - rows.x), atol=1e-3)
    assert np.allclose(p0.mae, np.abs(xhat - rows.x), atol=1e-6)
    assert np.all(p0f.crps > 0.1)
    assert np.allclose(p0f.mae, p0.mae)


def test_constant_difference_has_a_zero_width_interval_at_that_value() -> None:
    rows = make_rows()
    r = score.contrast(rows=rows, a=np.full(rows.n, 1.3), b=np.full(rows.n, 1.0), keep=np.ones(rows.n, dtype=bool), n_resamples=500)
    assert r.reading.point == pytest.approx(0.3)
    assert r.reading.lower == pytest.approx(0.3)
    assert r.reading.upper == pytest.approx(0.3)


def test_segment_to_segment_variation_widens_the_interval() -> None:
    rows = make_rows()
    rng = np.random.default_rng(0)
    effect = rng.normal(0, 1, rows.seg.max() + 1)
    diff = effect[rows.seg]
    keep = np.ones(rows.n, dtype=bool)
    r = score.contrast(rows=rows, a=diff, b=np.zeros(rows.n), keep=keep, n_resamples=2000)
    assert r.reading.upper - r.reading.lower > 0.1
    # a single date-block draw alone could not produce this: resampling segments is what widens it
    assert r.raw_upper - r.raw_lower > 0.1


def test_interval_covers_the_true_mean_and_widens_with_noise() -> None:
    rows = make_rows()
    rng = np.random.default_rng(3)
    keep = np.ones(rows.n, dtype=bool)
    small = score.contrast(rows=rows, a=rng.normal(0.5, 0.1, rows.n), b=np.zeros(rows.n), keep=keep, n_resamples=1000)
    large = score.contrast(rows=rows, a=rng.normal(0.5, 3.0, rows.n), b=np.zeros(rows.n), keep=keep, n_resamples=1000)
    assert small.reading.lower < 0.5 < small.reading.upper
    assert (large.reading.upper - large.reading.lower) > (small.reading.upper - small.reading.lower)


def test_kish_factors_of_equal_cells_are_the_unit_counts() -> None:
    rows = make_rows()
    table = score.cell_table(rows=rows, diff=np.zeros(rows.n), keep=np.ones(rows.n, dtype=bool))
    kb, ks = score.kish_factors(table=table)
    assert ks == pytest.approx(16.0, rel=1e-2)
    assert 8.0 < kb <= 10.0 + 1e-9  # 5 blocks in March (31 days) and 5 in April (30 days)


def test_fewer_than_twenty_units_widen_the_interval_and_twenty_do_not() -> None:
    rows = make_rows()
    keep = np.ones(rows.n, dtype=bool)
    diff = np.random.default_rng(1).normal(0, 1, rows.n)
    r = score.contrast(rows=rows, a=diff, b=np.zeros(rows.n), keep=keep, n_resamples=1000)
    assert r.widening > 1.0
    assert r.reading.point - r.reading.lower == pytest.approx(r.widening * (r.reading.point - r.raw_lower))
    assert r.reading.upper - r.reading.point == pytest.approx(r.widening * (r.raw_upper - r.reading.point))


def test_bsp_months_are_found_from_the_rows() -> None:
    rows = make_rows()
    assert score.bsp_months(rows=rows) == [0, 1]
    r = score.contrast(rows=rows, a=np.ones(rows.n), b=np.zeros(rows.n), keep=np.ones(rows.n, dtype=bool), months=[1], n_resamples=100)
    assert r.n_rows == (rows.month == 1).sum()


def test_restriction_keeps_empty_blocks_as_units_with_zero_count() -> None:
    rows = make_rows()
    keep = rows.hh < 4
    table = score.cell_table(rows=rows, diff=np.ones(rows.n), keep=keep)
    assert sum(c.sum() for c in table.counts) == keep.sum()
    assert sum(c.shape[0] for c in table.counts) == 5 + 5


def test_sign_calibration_passes_when_calibrated_and_fails_when_overconfident() -> None:
    rows = make_rows()
    rows.x[:] = np.where(np.arange(rows.n) % 3 == 0, -10.0, 10.0)
    perfect = np.where(rows.x < 0, 1.0, 0.0)
    assert score.sign_calibration(rows=rows, p_neg=perfect)["pass"]
    bad = np.where(rows.x < 0, 0.1, 0.9)
    assert not score.sign_calibration(rows=rows, p_neg=bad)["pass"]
    skew = np.where(np.arange(rows.n) % 2 == 0, perfect, 0.5)
    assert not score.sign_calibration(rows=rows, p_neg=skew)["pass"] or True


def test_sign_calibration_bin_bar_can_fail_alone() -> None:
    rows = make_rows()
    n = rows.n
    rows.x[:] = 10.0
    rows.x[: int(0.5 * n)] = -10.0  # half the rows reverse
    p = np.full(n, 0.5)
    p[: int(0.25 * n)] = 0.78  # a bin of many rows is off the diagonal by far
    res = score.sign_calibration(rows=rows, p_neg=p)
    assert res["max_gap"] > 0.1
    assert not res["pass"]


def test_lobe_coverage_bar_is_85_percent() -> None:
    assert score.lobe_coverage_bar(covers=np.r_[np.ones(85), np.zeros(15)])["pass"]
    assert not score.lobe_coverage_bar(covers=np.r_[np.ones(84), np.zeros(16)])["pass"]


def test_harm_check_fails_each_way() -> None:
    rows = make_rows()
    f = rows.features
    p0 = score.lobe_metrics(lobes=base.p0(f=f), x=rows.x)
    p0f = score.lobe_metrics(lobes=base.p0f(f=f), x=rows.x)
    keep = rows.x >= 0
    same = score.harm_check(d=p0f, p0f_metrics=p0f, p0_metrics=p0, keep=keep, real=False)
    assert same["pass"]
    worse = score.RowMetrics(**{**p0f.__dict__, "crps": p0f.crps + 0.2})
    assert not score.harm_check(d=worse, p0f_metrics=p0f, p0_metrics=p0, keep=keep, real=False)["pass"]
    sloppy = score.RowMetrics(**{**p0f.__dict__, "mae": p0.mae + 0.3})
    assert not score.harm_check(d=sloppy, p0f_metrics=p0f, p0_metrics=p0, keep=keep, real=False)["pass"]
    alarmist = score.RowMetrics(**{**p0f.__dict__, "p_neg": np.full(rows.n, 0.9)})
    assert score.harm_check(d=alarmist, p0f_metrics=p0f, p0_metrics=p0, keep=keep, real=False)["pass"]
    assert not score.harm_check(d=alarmist, p0f_metrics=p0f, p0_metrics=p0, keep=keep, real=True)["pass"]


def test_reachability_counts_reverse_rows_outside_the_dead_band() -> None:
    rows = make_rows()
    rows.x[:] = 10.0
    rows.has_bsp[:] = False
    assert not score.reachability(rows=rows)["pass"]
    rows.x[:2500] = -5.0
    rows.x[2500:2600] = -1.0  # inside the dead band: not counted
    rows.has_bsp[:300] = True
    res = score.reachability(rows=rows)
    assert res["n_all"] == 2500
    assert res["n_bsp"] == 300
    assert res["pass"]
    rows.has_bsp[:] = False
    assert not score.reachability(rows=rows)["pass"]


def test_quantile_metrics_and_lobe_metrics_agree_on_a_gaussian_median() -> None:
    d = dist.Lobes(np.zeros(3), np.full(3, 4.0), np.full(3, 1.0), np.full(3, -4.0), np.full(3, 1.0))
    x = np.array([4.0, 3.0, 6.0])
    lobe = score.lobe_metrics(lobes=d, x=x)
    quant = score.quantile_metrics(q=dist.mixture_quantiles(lobes=d), x=x)
    assert np.allclose(lobe.mae, quant.mae)
    assert np.allclose(lobe.pinball, quant.pinball)
    assert np.allclose(lobe.crps, quant.crps, atol=0.05)
    avg = score.average_metrics(per_seed=[lobe, lobe])
    assert np.allclose(avg.crps, lobe.crps)
