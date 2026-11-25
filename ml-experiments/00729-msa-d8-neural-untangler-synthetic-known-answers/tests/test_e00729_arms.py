"""Tests of the arms' inputs, windows, masking, lags, and the BSP-balanced sampler."""

import e00729_arms as arms
import e00729_core as core
import e00729_sim as sim
import numpy as np
import pytest
from synthetic_world import make_calendar

STATS = arms.InputStats(10.0, 3.0)


def seg(*, world: str = "both", seed: int = 1, bsp: bool = True) -> tuple[sim.Segment, sim.Calendar]:
    cal = make_calendar(bsp=bsp)
    s = sim.simulate_segment(cal=cal, month="2022-03", use=np.ones(len(cal.dates), dtype=bool), world=world, rng=np.random.default_rng(seed))
    return s, cal


def test_channel_counts_follow_the_ladder() -> None:
    n = {k: arms.n_channels(spec=v) for k, v in arms.ARM_SPECS.items()}
    assert n["A"] == 2
    assert n["B"] == n["A"] + 4
    assert n["C"] == n["B"] + 2
    assert n["D"] == n["C"] + 6
    assert n["H"] == n["A"] + 2
    assert n["F"] == n["D"] + 4
    assert n["E"] == n["G1"] == n["G2"] == n["Dq"] == n["D24"] == n["D"]
    assert arms.ARM_SPECS["D24"].steps == 48
    assert arms.ARM_SPECS["Dq"].head == "quantile"
    assert arms.ARM_SPECS["G1"].aux == "solar"
    assert arms.ARM_SPECS["G2"].aux == "engine"
    assert arms.ARM_SPECS["E"].mask_training


def test_every_arm_differs_from_its_parent_in_one_way() -> None:
    d = arms.ARM_SPECS["D"]
    for name in ("Dq", "D24", "E", "F", "G1", "G2"):
        spec = arms.ARM_SPECS[name]
        diffs = [f for f in ("cams", "bsp", "calendar", "lags", "steps", "head", "aux", "mask_training") if getattr(spec, f) != getattr(d, f)]
        assert len(diffs) == 1, name


def test_window_has_96_steps_and_the_target_is_the_scored_day() -> None:
    s, cal = seg()
    spec = arms.ARM_SPECS["D"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    targets = arms.target_stack(seg=s)
    day = 10
    w, t = arms.build_window(stack=stack, targets=targets, dense_day=day, spec=spec)
    assert w.shape == (arms.n_channels(spec=spec), 96)
    assert t.shape == (4, 48)
    assert np.allclose(t[0], np.nan_to_num(s.x[day]), atol=1e-6)
    # the middle 24 hours of the window hold the scored day's reading
    assert np.allclose(w[0, 24:72], np.nan_to_num(s.r[day]) / arms.R_SCALE, atol=1e-5)
    # the first 12 hours hold the previous day's second half
    assert np.allclose(w[0, :24], np.nan_to_num(s.r[day - 1][24:]) / arms.R_SCALE, atol=1e-5)
    assert np.allclose(w[0, 72:], np.nan_to_num(s.r[day + 1][:24]) / arms.R_SCALE, atol=1e-5)


def test_the_24_hour_window_holds_only_the_scored_day() -> None:
    s, cal = seg()
    spec = arms.ARM_SPECS["D24"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    w, _ = arms.build_window(stack=stack, targets=arms.target_stack(seg=s), dense_day=10, spec=spec)
    assert w.shape[1] == 48
    assert np.allclose(w[0], np.nan_to_num(s.r[10]) / arms.R_SCALE, atol=1e-5)
    assert arms.middle_slice(spec=spec) == slice(0, 48)
    assert arms.middle_slice(spec=arms.ARM_SPECS["D"]) == slice(24, 72)


def test_missing_reading_is_zero_with_a_zero_mask() -> None:
    s, cal = seg()
    s.valid[10, 5:9] = False
    stack = arms.segment_stack(seg=s, spec=arms.ARM_SPECS["A"], holidays=cal.holidays, stats=STATS)
    cols = slice(10 * 48 + 5, 10 * 48 + 9)
    assert np.all(stack[0, cols] == 0.0)
    assert np.all(stack[1, cols] == 0.0)
    assert stack[1, 10 * 48 + 12] == 1.0
    padding = stack[1, :48]
    assert np.all(padding == 0.0)


def test_lag_channels_hold_the_earlier_days_own_folded_reading_not_the_scored_days() -> None:
    s, cal = seg(world="both")
    spec = arms.ARM_SPECS["F"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    day = 20
    w, _ = arms.build_window(stack=stack, targets=arms.target_stack(seg=s), dense_day=day, spec=spec)
    base = arms.n_channels(spec=arms.ARM_SPECS["D"])
    start = arms.window_start(dense_day=day, spec=spec)
    for k, lag in enumerate(core.LAG_DAYS):
        got = w[base + 2 * k]
        assert np.allclose(got, stack[0, start - lag * 48 : start - lag * 48 + 96])
        assert not np.allclose(got, w[0])
        assert w[base + 2 * k + 1].sum() > 0
    # the lagged day's output is its own: engine and solar differ between the days
    assert not np.allclose(s.engine[day], s.engine[day - 7]) or not np.allclose(s.solar[day], s.solar[day - 7])


def test_lag_outside_the_month_is_masked_with_zeros() -> None:
    s, cal = seg()
    spec = arms.ARM_SPECS["F"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    w, _ = arms.build_window(stack=stack, targets=arms.target_stack(seg=s), dense_day=3, spec=spec)
    base = arms.n_channels(spec=arms.ARM_SPECS["D"])
    assert np.all(w[base : base + 4] == 0.0)


def test_masking_hides_a_span_of_one_to_24_hours_in_both_reading_rows() -> None:
    rng = np.random.default_rng(0)
    lengths = set()
    for _ in range(200):
        w = np.ones((4, 96), dtype=np.float32)
        arms.mask_span(window=w, rng=rng, steps=96)
        hidden = np.flatnonzero(w[0] == 0)
        assert np.array_equal(hidden, np.flatnonzero(w[1] == 0))
        assert np.all(w[2:] == 1)
        assert np.array_equal(hidden, np.arange(hidden[0], hidden[0] + len(hidden)))
        lengths.add(len(hidden))
    assert min(lengths) == 2
    assert max(lengths) == 48
    assert all(n % 2 == 0 for n in lengths)


def test_only_arm_e_masks_training_windows() -> None:
    s, cal = seg()
    rng = np.random.default_rng(0)
    for name, expect in (("E", True), ("D", False)):
        spec = arms.ARM_SPECS[name]
        prep = arms.prepare_segments(segments=[s], spec=spec, holidays=cal.holidays, stats=STATS)
        picks = [(0, 10)] * 60
        x, _ = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, training=True)
        masked = (x[:, 1, 24:72] == 0).any(axis=1)
        assert masked.any() == expect
        x_eval, _ = arms.make_batch(prepared=prep, picks=picks, spec=spec, rng=rng, training=False)
        assert not (x_eval[:, 1, 24:72] == 0).any()
    assert 0.2 < masked.mean() < 0.8 or not expect


def test_calendar_rows_hold_no_month_or_day_of_year() -> None:
    for spec in arms.ARM_SPECS.values():
        names = arms.stack_names(spec=spec)
        assert not any("month" in n or "doy" in n or "year" in n for n in names)
    s, cal = seg()
    stack = arms.segment_stack(seg=s, spec=arms.ARM_SPECS["D"], holidays=cal.holidays, stats=STATS)
    names = arms.stack_names(spec=arms.ARM_SPECS["D"])
    holiday = stack[names.index("holiday")].reshape(-1, 48)[:, 0]
    dates = s.dates
    assert holiday[np.isin(dates, cal.holidays)].sum() == np.isin(dates, cal.holidays).sum()
    assert holiday.sum() == np.isin(dates, cal.holidays).sum()


def test_bsp_rows_are_steps_with_a_mask() -> None:
    s, cal = seg()
    spec = arms.ARM_SPECS["C"]
    stack = arms.segment_stack(seg=s, spec=spec, holidays=cal.holidays, stats=STATS)
    names = arms.stack_names(spec=spec)
    step, mask = stack[names.index("d_bsp")], stack[names.index("bsp_mask")]
    flat = s.bsp.reshape(-1)
    t = 10 * 48 + 7
    assert step[t] == pytest.approx((flat[t] - flat[t - 1]) / arms.BSP_SCALE, abs=1e-5)
    assert mask[t] == 1.0
    assert mask[0] == 0.0
    _, cal_no = seg(bsp=False)
    s2, _ = seg(bsp=False)
    stack2 = arms.segment_stack(seg=s2, spec=spec, holidays=cal_no.holidays, stats=STATS)
    assert np.all(stack2[names.index("bsp_mask")] == 0.0)


def test_sampler_takes_half_of_the_windows_from_segments_with_real_bsp() -> None:
    with_bsp, cal = seg(bsp=True)
    without, _ = seg(bsp=False, seed=2)
    spec = arms.ARM_SPECS["A"]
    prep = arms.prepare_segments(segments=[with_bsp, without, without], spec=spec, holidays=cal.holidays, stats=STATS)
    picks = arms.sample_picks(prepared=prep, n=1000, rng=np.random.default_rng(0))
    assert len(picks) == 1000
    share = np.mean([prep[i].has_bsp for i, _ in picks])
    assert share == pytest.approx(0.5, abs=1e-9)
    only = arms.sample_picks(prepared=prep[1:], n=50, rng=np.random.default_rng(0))
    assert len(only) == 50
    assert all(d in prep[i].days for i, d in picks)


def test_all_picks_cover_every_base_day_once_in_order() -> None:
    s, cal = seg()
    prep = arms.prepare_segments(segments=[s, s], spec=arms.ARM_SPECS["A"], holidays=cal.holidays, stats=STATS)
    picks = arms.all_picks(prepared=prep)
    assert len(picks) == 2 * 31
    assert picks == sorted(picks)
