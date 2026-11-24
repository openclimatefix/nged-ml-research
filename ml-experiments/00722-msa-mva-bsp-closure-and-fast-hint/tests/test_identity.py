"""The identity of section 5.2, reading noise, the common mode, and mixed stretches (synthetic)."""

import e00722_core as core
import e00722_sky as sky
import e00722_u2b as u2b
import numpy as np
from e00722_testkit import calendar, make_weights


def _group(*, q_points: float, n: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    flow = rng.normal(0.0, 6.0, n)
    reading = np.sqrt(flow**2 + q_points**2)
    other = rng.normal(20.0, 1.0, n)
    bsp = 4.0 + flow + other
    return {"flow": flow, "reading": reading, "other": other, "bsp": bsp}


def test_leave_out_ratio_is_at_least_plus_one_under_import_and_at_most_minus_one_under_export():
    for q in (0.0, 3.0):
        g = _group(q_points=q)
        resid = core.leave_out_residual(
            bsp=g["bsp"],
            children={"i": g["reading"], "j": g["other"]},
            leave_out="i",
            generators=[],
        )
        d_r, d_m = core.change(x=resid, lag=1), core.change(x=g["reading"], lag=1)
        flow_prev = core.shift_rows(x=g["flow"], rows=1)
        ok = np.abs(d_m) > 1e-6
        imp = ok & (g["flow"] > 0) & (flow_prev > 0)
        exp = ok & (g["flow"] < 0) & (flow_prev < 0)
        assert imp.sum() > 100
        assert exp.sum() > 100
        assert (d_r[imp] / d_m[imp]).min() >= 1.0 - 1e-9
        assert (d_r[exp] / d_m[exp]).max() <= -1.0 + 1e-9


def test_all_children_ratio_is_exactly_the_leave_out_ratio_minus_one():
    g = _group(q_points=3.0)
    resid = core.leave_out_residual(
        bsp=g["bsp"], children={"i": g["reading"], "j": g["other"]}, leave_out="i", generators=[]
    )
    every = core.leave_out_residual(
        bsp=g["bsp"], children={"i": g["reading"], "j": g["other"]}, leave_out=None, generators=[]
    )
    assert np.allclose(every, core.all_children_residual(leave_out=resid, reading=g["reading"]))
    d_m = core.change(x=g["reading"], lag=1)
    ok = np.abs(d_m) > 1e-6
    ratio_leave = core.change(x=resid, lag=1)[ok] / d_m[ok]
    ratio_all = core.change(x=every, lag=1)[ok] / d_m[ok]
    assert np.allclose(ratio_all, ratio_leave - 1.0)


def test_the_scored_childs_own_series_is_never_subtracted_and_generators_are_added():
    bsp = np.array([10.0, 12.0])
    child = {"i": np.array([3.0, 4.0]), "j": np.array([1.0, 1.0])}
    resid = core.leave_out_residual(
        bsp=bsp, children=child, leave_out="i", generators=[np.array([2.0, 2.0])]
    )
    assert resid.tolist() == [10.0 - 1.0 + 2.0, 12.0 - 1.0 + 2.0]


def test_reading_noise_pulls_the_leave_out_slope_to_zero_and_the_all_children_slope_to_minus_one():
    n_days = 60
    cal = calendar(n_days=n_days)
    n = n_days * core.HALF_HOURS_PER_DAY
    rng = np.random.default_rng(3)
    flow = 20.0 + rng.normal(0, 1.0, n)
    reading = flow
    noisy = reading + rng.normal(0, 6.0, n)
    resid = flow + 5.0
    w = make_weights(n_days=n_days)
    d_r = core.change(x=resid, lag=1)
    d_noisy = core.change(x=noisy, lag=1)
    d_all = core.change(x=core.all_children_residual(leave_out=resid, reading=noisy), lag=1)
    clean = core.bootstrap_ols(
        y=d_r, x=core.change(x=reading, lag=1), day=cal["day"], weights=w, names=["s"]
    )
    leave = core.bootstrap_ols(y=d_r, x=d_noisy, day=cal["day"], weights=w, names=["s"])
    every = core.bootstrap_ols(y=d_all, x=d_noisy, day=cal["day"], weights=w, names=["s"])
    assert abs(clean.coef[0] - 1.0) < 1e-9
    assert 0.0 < leave.coef[0] < 0.5
    assert every.coef[0] < -0.5
    assert np.isclose(every.coef[0], leave.coef[0] - 1.0)


def test_a_comoving_remainder_inflates_the_raw_coefficient_and_matched_demeaning_removes_it():
    n_days = 200
    cal = calendar(n_days=n_days)
    n = n_days * core.HALF_HOURS_PER_DAY
    rng = np.random.default_rng(4)
    ramp = 2.0 * np.sin(2 * np.pi * (np.arange(n) % 48) / 48.0)
    idio = rng.normal(0, 1.0, n)
    x = ramp + idio
    y = x + 0.8 * ramp + rng.normal(0, 0.1, n)
    w = make_weights(n_days=n_days)
    raw = core.bootstrap_ols(y=y, x=x, day=cal["day"], weights=w, names=["s"])
    dm = core.bootstrap_ols(
        y=core.demean_matched(x=y, key=cal["key"], quiet=cal["quiet"]),
        x=core.demean_matched(x=x, key=cal["key"], quiet=cal["quiet"]),
        day=cal["day"],
        weights=w,
        names=["s"],
    )
    assert raw.coef[0] > 1.4
    assert abs(dm.coef[0] - 1.0) < 0.1
    assert (raw.coef[0] - 1.0) > 4 * abs(dm.coef[0] - 1.0)


def _lit_day(*, touches_at: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 48 * 2
    lit = np.zeros(n, dtype=bool)
    lit[12:37] = True
    reading = np.full(n, 8.0)
    for t in touches_at:
        reading[t] = 0.0
    return reading, lit, np.ones(n, dtype=bool)


def test_stretch_rows_keep_only_days_with_exactly_one_post_bounce_stretch():
    classes = np.array([sky.CLASS_INTERMITTENT, sky.CLASS_INTERMITTENT])
    month = np.array([6, 6])
    for touches_at, n_post in (
        ([], 0),
        ([20], 0),
        ([20, 28], 1),
        ([16, 22, 28], 2),
        ([14, 18, 22, 26, 30], 4),
    ):
        reading, lit, _ = _lit_day(touches_at=touches_at)
        one, counts = u2b.stretch_rows(
            reading=reading, lit=lit, classes=classes, month=month, capacity_mw=10.0, want="one"
        )
        many, _ = u2b.stretch_rows(
            reading=reading, lit=lit, classes=classes, month=month, capacity_mw=10.0, want="many"
        )
        assert counts[0] == n_post
        assert one.any() == (n_post == 1)
        assert many.any() == (n_post >= core.MANY_STRETCHES)
    reading, lit, _ = _lit_day(touches_at=[20, 28])
    rows, _ = u2b.stretch_rows(
        reading=reading, lit=lit, classes=classes, month=month, capacity_mw=10.0, want="one"
    )
    assert np.flatnonzero(rows).tolist() == [22, 23, 24, 25, 26]


def test_pooled_stretches_land_in_the_gap_and_the_one_stretch_filter_recovers_the_sign():
    rng = np.random.default_rng(5)
    rows_x, rows_y, rows_day, n_stretches = [], [], [], []
    for d in range(120):
        parts = [(-1.5, 10)] if d < 60 else [(-1.5, 10), (+1.5, 20), (-1.5, 10)]
        for coef, length in parts:
            x = rng.normal(0, 1.0, length)
            rows_x.append(x)
            rows_y.append(coef * x + rng.normal(0, 0.05, length))
            rows_day.append(np.full(length, d))
            n_stretches.append(np.full(length, len(parts)))
    x, y = np.concatenate(rows_x), np.concatenate(rows_y)
    day, count = np.concatenate(rows_day), np.concatenate(n_stretches)
    w = make_weights(n_days=120)
    pooled = core.bootstrap_ols(y=y, x=x, day=day, weights=w, names=["s"])
    one = core.bootstrap_ols(
        y=np.where(count == 1, y, np.nan),
        x=np.where(count == 1, x, np.nan),
        day=day,
        weights=w,
        names=["s"],
    )
    se_pooled = float(pooled.se[0])
    assert -0.5 < pooled.coef[0] < 0.5
    assert (
        core.reverse_flow_test(lo=float(pooled.lo[0]), hi=float(pooled.hi[0]), se=se_pooled)
        == "unresolved"
    )
    assert core.reverse_flow_test(
        lo=float(one.lo[0]), hi=float(one.hi[0]), se=float(one.se[0])
    ) == ("supports_reverse_flow")
