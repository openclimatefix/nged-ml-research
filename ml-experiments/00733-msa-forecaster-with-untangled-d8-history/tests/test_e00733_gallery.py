# ruff: noqa: PT018
"""Tests of `e00733_gallery`: each panel's automatic check passes on a world with every component
and fails on a silent one."""

import json
from pathlib import Path
from typing import Any

import e00733_core as core
import e00733_gallery as gal
import e00733_synth as sy
import numpy as np
from synth_fixture import make_segment

SW = 3


def _world(seed: int = 0, n_days: int = 40) -> sy.SegmentArrays:
    seg = make_segment(n_days=n_days, seed=seed)
    sun = np.tile((np.arange(48) >= 12) & (np.arange(48) < 36), n_days)
    seg.hist_weather[:, SW] = np.where(sun, 300.0, 0.0)
    seg.solar_points[:] = np.where(sun, 15.0, 0.0)
    seg.u_raw[:, core.CHANNEL_NAMES.index("solar")] = np.where(sun, 12.0, 0.0)
    seg.x_neg[:] = np.where(seg.x_neg == 1.0, 1.0, 0.0)
    seg.u_raw[:, core.P_NEG] = np.where(seg.x_neg > 0.5, 0.9, 0.1)
    return seg


def test_g10_lists_every_row_inside_the_segment() -> None:
    """The layout lists the history rows that exist, none outside the segment, and the scored lead
    days 2 to 10."""
    seg = make_segment(n_days=40, missing_days=(10,))
    origin = gal.pick_origin(seg=seg)
    r = gal.g10_layout(seg=seg, origin_day=origin)
    assert r.passed, r.items
    assert r.data["history_exists"].sum() >= core.MIN_HISTORY_ROWS


def test_g11_fails_on_a_silent_solar_component_or_no_reverse_flow() -> None:
    """Zero injected solar output in every segment, or reverse flow in fewer than two segments,
    fails the panel."""
    segs = [_world(i) for i in range(3)]
    assert gal.g11_target_days(segments=segs, origin_days=[20, 20, 20]).passed
    for s in segs:
        s.solar_points[:] = 0.0
    assert not gal.g11_target_days(segments=segs, origin_days=[20, 20, 20]).passed
    segs = [_world(i) for i in range(3)]
    for s in segs[:2]:
        s.x_neg[:] = np.where(np.isnan(s.x_neg), np.nan, 0.0)
    assert not gal.g11_target_days(segments=segs, origin_days=[20, 20, 20]).passed


def test_g12_needs_a_positive_correlation_that_falls_with_lead_day() -> None:
    """A target-row weather that follows the observed sky less at longer leads passes; random
    weather fails."""
    seg = _world(1, n_days=60)
    rng = np.random.default_rng(0)
    for k in range(14):
        for d in range(seg.n_days):
            t = sy.target_slots(origin_day=np.array([d]), lead_index=np.array([k]))[0]
            if t.min() >= 0 and t.max() < seg.n_days * 48:
                obs = seg.hist_weather[t, SW]
                seg.top_weather[d, k, :, 0] = obs * (1 - 0.06 * k) + rng.normal(0, 30 + 25 * k, 48)
    assert gal.g12_weather_correlation(segments=[seg], ens_col=0, cams_col=SW).passed
    seg.top_weather[..., 0] = rng.normal(size=seg.top_weather[..., 0].shape)
    assert not gal.g12_weather_correlation(segments=[seg], ens_col=0, cams_col=SW).passed


def test_g13_fails_on_a_silent_solar_channel() -> None:
    """A solar channel of zero on sun-up half-hours fails; P(X<0) below one half on reverse flow
    fails."""
    segs = [_world(i) for i in range(3)]
    reading = [np.full(s.n_days * 48, 5.0) for s in segs]
    assert gal.g13_channels_against_truth(
        segments=segs, shortwave_col=SW, sun_up=10.0, min_reverse_points=1.0, reading_points=reading
    ).passed
    for s in segs:
        s.u_raw[:, core.CHANNEL_NAMES.index("solar")] = 0.0
    assert not gal.g13_channels_against_truth(
        segments=segs, shortwave_col=SW, sun_up=10.0, min_reverse_points=1.0, reading_points=reading
    ).passed
    segs = [_world(i) for i in range(3)]
    for s in segs:
        s.u_raw[:, core.P_NEG] = 0.1
    assert not gal.g13_channels_against_truth(
        segments=segs, shortwave_col=SW, sun_up=10.0, min_reverse_points=1.0, reading_points=reading
    ).passed


def _decaying_rerun(seg: Any, scale: float = 1.0) -> None:
    """Give a segment a re-run that differs from the stored output by a decaying amount."""
    n = seg.n_days
    seg.u_rerun = np.zeros((n, 2, 48, 3), np.float32)
    seg.u_rerun_ok = np.ones((n, 2, 48), bool)
    seg.u_rerun_ok[:2] = False
    for d in range(2, n):
        for r in (0, 1):
            day = d - 2 + r
            stamps = 48 * (r - 2) + np.arange(48) + 1 + core.NEWEST_VISIBLE_BEFORE_CONTRACT - 24
            hours_after = stamps / 2.0
            decay = scale * 0.002 * np.exp(np.minimum(hours_after, 0.0) / 12.0) + 1e-6
            seg.u_rerun[d, r] = seg.u_raw[day * 48 : (day + 1) * 48]
            seg.u_rerun[d, r, :, core.P_NEG] += decay.astype(np.float32)


def _binned(means: list[float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return a stack, its hours, and the 6-hour table whose bin k has mean `means[k]`.

    Bin 0 is the nearest the limit. A NaN entry leaves its bin empty.
    """
    hours, vals = [], []
    for k, m in enumerate(means):
        if np.isnan(m):
            continue
        h = -(k * 6.0 + 0.25 + 0.5 * np.arange(10))
        hours.append(h)
        vals.append(np.full(10, m))
    stack, hrs = np.concatenate(vals)[None], np.concatenate(hours)[None]
    return stack, hrs, gal.binned_difference(stack=stack, hours=hrs)


def _conditions(means: list[float]) -> tuple[bool, bool]:
    """Return G14's conditions (c') and (c'') for a profile with these 6-hour bin means."""
    stack, hours, table = _binned(means)
    return gal.g14_decay_conditions(stack=stack, hours=hours, table=table)


def test_g14_decay_conditions_follow_the_ruling_of_2026_11_27() -> None:
    """The validation bank's real bin means pass; a flat profile, a profile growing with distance,
    and empty bins fail."""
    assert _conditions([0.0123, 0.0028, 0.0019, 0.0036, 0.0011, 0.0005]) == (True, True)
    assert _conditions([0.003] * 6) == (False, True)
    assert _conditions([0.001, 0.002, 0.003, 0.004, 0.005, 0.006]) == (False, True)
    assert _conditions([0.05, 0.02, 0.012, 0.001, 0.001, 0.001]) == (True, False)
    assert _conditions([np.nan, 0.001, 0.001, 0.001]) == (False, False)
    assert _conditions([0.01, 0.01]) == (False, False)
    nothing = np.full((1, 3), np.nan)
    hours = np.array([[-1.0, -2.0, -3.0]])
    table = gal.binned_difference(stack=nothing, hours=hours)
    assert gal.g14_decay_conditions(stack=nothing, hours=hours, table=table) == (False, False)


def test_g14_follows_the_ruling_of_2026_11_27() -> None:
    """A decaying difference with a passed poison test passes; each ruling item can fail alone."""
    seg = _world(2)
    _decaying_rerun(seg)
    ok = gal.g14_rerun_difference(segments=[seg], poison_passed=True)
    assert ok.passed, ok.items
    assert ok.data["far_mean"] < core.G14_MEAN_LIMIT
    assert len(ok.data["binned_mean_abs"]) > 4
    assert not gal.g14_rerun_difference(segments=[seg], poison_passed=False).passed
    big = _world(2)
    _decaying_rerun(big, scale=100.0)
    bad = gal.g14_rerun_difference(segments=[big], poison_passed=True)
    assert not bad.items["mean_difference_below_limit_more_than_24_hours_before_the_limit"]
    flat = _world(2)
    _decaying_rerun(flat)
    assert flat.u_rerun is not None
    for d in range(2, flat.n_days):
        for r in (0, 1):
            day = d - 2 + r
            stored = flat.u_raw[day * 48 : (day + 1) * 48, core.P_NEG]
            flat.u_rerun[d, r, :, core.P_NEG] = stored + 0.003
    flagged = gal.g14_rerun_difference(segments=[flat], poison_passed=True)
    assert not flagged.items["nearest_bin_mean_at_least_3_times_the_mean_more_than_12_hours_before"]
    assert not gal.g14_rerun_difference(segments=[make_segment()], poison_passed=True).passed


def test_g15_needs_reverse_flow_on_sun_up_half_hours() -> None:
    """A world with reverse flow by day passes; a world with none fails."""
    seg = _world(3)
    r = gal.g15_base_rate(segments=[seg], clim=np.full(48, 0.2), shortwave_col=SW, sun_up=10.0)
    assert r.passed and r.data["share"].shape == (14, 48)
    seg.x_neg[:] = np.where(np.isnan(seg.x_neg), np.nan, 0.0)
    flat = gal.g15_base_rate(segments=[seg], clim=np.full(48, 0.2), shortwave_col=SW, sun_up=10.0)
    assert not flat.passed


def test_panel_check_marks_a_missing_panel_failed_and_is_read_by_the_guard(tmp_path: Path) -> None:
    """A panel with no result is recorded as failed; the file holds the banks' digest and the
    written lists."""
    ok = gal.g10_layout(seg=make_segment(), origin_day=20)
    payload = gal.panel_check(results=[ok], banks_digest="d")
    assert payload["panels"]["G10"] and not any(
        v for k, v in payload["panels"].items() if k != "G10"
    )
    path = gal.write_panel_check(root=tmp_path, payload=payload)
    assert json.loads(path.read_text())["banks_digest"] == "d"


def _standardised_world() -> sy.SegmentArrays:
    """Return a world whose weather column is standardised: night -0.6, day 1.2 (all below 10)."""
    seg = _world(4)
    sun = np.tile((np.arange(48) >= 12) & (np.arange(48) < 36), seg.n_days)
    seg.hist_weather[:, SW] = np.where(sun, 1.2, -0.6)
    return seg


def test_g13_and_g15_use_the_standardised_sun_up_threshold() -> None:
    """Standardised weather never exceeds the old 10 W/m2 constant, so the old comparison saw no
    sun; with the standardised threshold both panels pass, and with 10.0 both fail."""
    seg = _standardised_world()
    reading = [np.full(seg.n_days * 48, 5.0)]
    kw = {"segments": [seg], "shortwave_col": SW}
    assert gal.g13_channels_against_truth(
        **kw, sun_up=0.0, min_reverse_points=1.0, reading_points=reading
    ).passed
    assert not gal.g13_channels_against_truth(
        **kw, sun_up=10.0, min_reverse_points=1.0, reading_points=reading
    ).passed
    assert gal.g15_base_rate(**kw, clim=np.full(48, 0.2), sun_up=0.0).passed
    assert not gal.g15_base_rate(**kw, clim=np.full(48, 0.2), sun_up=10.0).passed
