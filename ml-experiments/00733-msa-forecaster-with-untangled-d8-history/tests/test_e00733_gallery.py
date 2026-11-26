# ruff: noqa: PT018
"""Tests of `e00733_gallery`: each panel's automatic check passes on a world with every component
and fails on a silent one."""

import json
from pathlib import Path

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
        segments=segs, shortwave_col=SW, min_reverse_points=1.0, reading_points=reading
    ).passed
    for s in segs:
        s.u_raw[:, core.CHANNEL_NAMES.index("solar")] = 0.0
    assert not gal.g13_channels_against_truth(
        segments=segs, shortwave_col=SW, min_reverse_points=1.0, reading_points=reading
    ).passed
    segs = [_world(i) for i in range(3)]
    for s in segs:
        s.u_raw[:, core.P_NEG] = 0.1
    assert not gal.g13_channels_against_truth(
        segments=segs, shortwave_col=SW, min_reverse_points=1.0, reading_points=reading
    ).passed


def test_g14_needs_zero_difference_before_the_limit() -> None:
    """A re-run equal to the stored output before the limit passes; a difference before the limit
    fails."""
    seg = _world(2)
    n = seg.n_days
    seg.u_rerun = np.zeros((n, 2, 48, 3), np.float32)
    seg.u_rerun_ok = np.ones((n, 2, 48), bool)
    seg.u_rerun_ok[:2] = False
    for d in range(2, n):
        for r in (0, 1):
            day = d - 2 + r
            seg.u_rerun[d, r] = seg.u_raw[day * 48 : (day + 1) * 48]
    after = gal.g14_rerun_difference(segments=[seg])
    assert after.passed, after.items
    seg.u_rerun[10, 0, 0, core.P_NEG] += 0.5
    assert not gal.g14_rerun_difference(segments=[seg]).passed
    assert not gal.g14_rerun_difference(segments=[make_segment()]).passed


def test_g15_needs_reverse_flow_on_sun_up_half_hours() -> None:
    """A world with reverse flow by day passes; a world with none fails."""
    seg = _world(3)
    r = gal.g15_base_rate(segments=[seg], clim=np.full(48, 0.2), shortwave_col=SW)
    assert r.passed and r.data["share"].shape == (14, 48)
    seg.x_neg[:] = np.where(np.isnan(seg.x_neg), np.nan, 0.0)
    assert not gal.g15_base_rate(segments=[seg], clim=np.full(48, 0.2), shortwave_col=SW).passed


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
