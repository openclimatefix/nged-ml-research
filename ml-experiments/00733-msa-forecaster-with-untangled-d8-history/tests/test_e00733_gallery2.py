"""Tests of `e00733_gallery2`: the selection rule, the automatic checks, and the G14 counts."""

from types import SimpleNamespace

import e00733_core as core
import e00733_gallery2 as g2
import e00733_synth as sy
import numpy as np
import pytest
from synth_fixture import make_segment


def _candidates(n: int = 40) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "origin_day": np.arange(n) * 5 + 2000,
        "p_max": rng.random(n) * 0.4,
        "step_max": rng.random(n) * 0.3,
        "csi_mean": rng.random(n),
        "clear_mean": rng.random(n) * 300,
    }


def test_the_rule_picks_every_kind_once_with_distinct_separated_origins() -> None:
    """Seven examples, three of them reverse flow, no candidate twice, origins 4 days apart."""
    c = _candidates()
    c["p_max"][[3, 11, 20, 25]] = [0.99, 0.9, 0.8, 0.7]
    c["step_max"][7] = 0.99
    got = g2.choose_windows(**c)
    assert [k for k, _ in got] == list(g2.KINDS_REAL)
    idx = [i for _, i in got]
    assert idx[:3] == [3, 11, 20]
    assert idx[3] == 7
    assert len(set(idx)) == len(idx)
    days = sorted(int(c["origin_day"][i]) for i in idx)
    assert min(np.diff(days)) >= g2.SEPARATION_DAYS


def test_the_rule_skips_a_window_next_to_a_chosen_origin() -> None:
    """A second high-probability window one day after the first is not taken."""
    c = _candidates()
    c["origin_day"][4] = c["origin_day"][3] + 1
    c["p_max"][3], c["p_max"][4] = 0.99, 0.98
    got = g2.choose_windows(**c)
    assert 4 not in [i for _, i in got]


def test_sunny_has_the_highest_and_cloudy_the_lowest_index_among_quiet_bright_windows() -> None:
    """Sunny and cloudy come from the bright, quiet windows."""
    c = _candidates()
    c["clear_mean"][:] = 100.0
    c["csi_mean"][:] = 0.5
    c["clear_mean"][30:40] = 500.0
    c["csi_mean"][[30, 31]] = [0.95, 0.05]
    c["p_max"][:] = 0.1
    c["csi_mean"][[0, 1]] = [0.99, 0.01]
    got = dict(g2.choose_windows(**c))
    assert got["sunny"] == 30
    assert got["cloudy"] == 31
    assert got["sunny"] != 0


def test_the_rule_raises_without_a_quiet_window() -> None:
    """No window below 0.5 leaves no sunny day, and the rule stops."""
    c = _candidates()
    c["p_max"][:] = 0.9
    with pytest.raises(g2.Gallery2Error):
        g2.choose_windows(**c)


def test_history_grid_index_matches_the_visibility_layout() -> None:
    """History row `a` holds the stamps `c - 48 a + j`; the shown rows are ages 5 and 4."""
    hist = g2.history_grid_index(origin_day=np.array([100]))
    assert hist.shape == (1, 14, 48)
    c = 100 * 48 + 24
    assert hist[0, 0, 0] == c - 48 * 2
    assert [hist[0, r, 0] for r in g2.shown_rows()] == [c - 48 * 5, c - 48 * 4]
    newest_shown = hist[0, g2.shown_rows()[-1], -1]
    assert newest_shown <= c - core.NEWEST_VISIBLE_BEFORE_CONTRACT


def test_dense_on_grid_places_a_day_at_the_stamp_convention() -> None:
    """The first half-hour of the grid's first day sits at grid index 1."""
    import e00733_channels as ch

    day = np.datetime64(core.GRID_START_DATE, "D") + np.timedelta64(2, "D")
    values = np.arange(48, dtype=float)[None, :]
    out = g2.dense_on_grid(dates=np.array([day]), values=values, n_grid=48 * 5)
    assert out[2 * 48 + 1] == 0.0
    assert out[2 * 48 + 48] == 47.0
    assert int(np.isfinite(out).sum()) == 48
    assert ch.GRID_START_EPOCH_DAY > 0


def test_candidate_origins_end_before_the_first_fold_month() -> None:
    """The held-out origins are a minority of the days and stop before the limit day."""
    last = 2200
    days = g2.candidate_origin_days(last_day=last)
    assert len(days) > 0
    assert days.max() < last
    assert days.min() >= g2.CANDIDATE_FIRST_DAY
    assert len(days) < (last - g2.CANDIDATE_FIRST_DAY) / 2


def _example(*, kind: str, solar_in_fu: bool = False, same: bool = False) -> g2.HistoryExample:
    rng = np.random.default_rng(3)
    fu = rng.random((96, 3)) * 0.3
    fu[:, core.SOLAR] = rng.random(96) if solar_in_fu else 0.0
    if kind == "reverse flow":
        fu[:5, core.P_NEG] = 0.9
    fus = fu.copy() if same else rng.random((96, 3)) * 0.3
    fus[:, core.SOLAR] = 0.0
    daylight = np.arange(96) % 48 > 14
    daylight &= np.arange(96) % 48 < 34
    return g2.HistoryExample(
        name="H1", kind=kind, target_mw=np.full(96, 5.0), fu=fu, fu_ok=np.ones(96, dtype=bool),
        fus=fus, fus_ok=np.ones(96, dtype=bool), stored_solar_mw=np.where(daylight, 2.0, 0.0),
        daylight=daylight, usable=core.CHANNEL_NAMES if solar_in_fu else ("p_neg", "step_p"),
        solar_mw_per_unit=1.0,
    )  # fmt: skip


def test_history_check_passes_on_a_good_example_and_fails_on_each_defect() -> None:
    """A reverse-flow example passes; a placebo equal to FU, no high probability, or an empty
    series fails."""
    good = _example(kind="reverse flow")
    assert g2.check_history_example(example=good).passed
    assert not g2.check_history_example(example=_example(kind="reverse flow", same=True)).passed
    quiet = _example(kind="reverse flow")
    quiet.fu[:, core.P_NEG] = 0.1
    assert not g2.check_history_example(example=quiet).passed
    empty = _example(kind="reverse flow")
    empty.target_mw[:] = np.nan
    assert not g2.check_history_example(example=empty).passed


def test_sunny_check_needs_stored_solar_and_a_solar_channel_that_follows_the_rule() -> None:
    """A dropped channel must be zero in FU, and a kept channel must be nonzero in daylight."""
    assert g2.check_history_example(example=_example(kind="sunny")).passed
    assert g2.check_history_example(example=_example(kind="sunny", solar_in_fu=True)).passed
    bad = _example(kind="sunny")
    bad.fu[:, core.SOLAR] = 0.5
    assert not g2.check_history_example(example=bad).passed
    no_sun = _example(kind="sunny")
    no_sun.stored_solar_mw[:] = 0.0
    assert not g2.check_history_example(example=no_sun).passed


def test_charts_draw_every_example_with_and_without_the_placebo() -> None:
    """The stacked charts build and name no date, capacity, or series identifier."""
    ex = _example(kind="reverse flow")
    for with_placebo in (False, True):
        spec = g2.history_chart(example=ex, with_placebo=with_placebo).to_dict()
        text = str(spec["title"]).lower()
        assert "capacity" not in text
        assert len(spec["vconcat"]) == (3 if with_placebo else 4)


def _arrays(*, solar: bool = True) -> tuple[sy.SegmentArrays, SimpleNamespace]:
    n_days = 6
    seg = make_segment(n_days=n_days, seed=2)
    s = n_days * sy.STEPS
    sun = np.tile((np.arange(48) >= 12) & (np.arange(48) < 36), n_days)
    seg.x_neg[:] = (np.arange(s) % 48 < 10).astype(np.float32)
    seg.u_raw[:, core.P_NEG] = np.where(seg.x_neg > 0.5, 0.9, 0.1)
    seg.u_raw[:, core.SOLAR] = np.where(sun, 6.0, 0.0)
    seg.solar_points[:] = np.where(sun & solar, 5.0, 0.0)
    bank = SimpleNamespace(
        base=np.full(s, 40.0), engine=np.full(s, 30.0), solar=seg.solar_points.copy(),
        r=np.where(np.isfinite(seg.reading_z), 10.0, np.nan), source="S2",
    )  # fmt: skip
    return seg, bank


def test_the_synthetic_window_is_never_a_source_s1_segment() -> None:
    """Only a segment of source S2 can be chosen, and a segment with no solar is passed over."""
    seg, _bank = _arrays()
    assert g2.choose_synthetic_window(arrays=[seg, seg], sources=["S1", "S2"])[0] == 1
    with pytest.raises(g2.Gallery2Error):
        g2.choose_synthetic_window(arrays=[seg], sources=["S1"])
    quiet, _ = _arrays(solar=False)
    with pytest.raises(g2.Gallery2Error):
        g2.choose_synthetic_window(arrays=[quiet], sources=["S2"])


def test_the_bank_segment_is_matched_by_its_reading() -> None:
    """The world segment finds the bank segment whose standardised reading it holds."""
    seg, bank = _arrays()
    consts = {"mean": 0.1, "std": 0.2}
    seg.reading_z[:] = np.where(np.isfinite(seg.reading_z), (10.0 / 100 - 0.1) / 0.2, np.nan)
    other = SimpleNamespace(r=np.full_like(bank.r, 50.0))
    assert g2.match_bank_segment(arrays=seg, bank_segments=[other, bank], constants=consts) == 1
    assert g2.match_bank_segment(arrays=seg, bank_segments=[other], constants=consts) is None


def test_synthetic_arms_receive_what_their_name_says() -> None:
    """Y0 receives no channel, YU the untangler's, and YT the truth; each check passes."""
    seg, bank = _arrays()
    window = g2.synthetic_window(arrays=seg, segment=bank, first_day=1, mw_per_point=0.1)
    assert g2.arm_signals(arm="Y0", window=window) == {"p": None, "solar": None}
    yu = g2.arm_signals(arm="YU", window=window)
    assert yu["p"] is not None
    assert np.allclose(yu["p"][window.x_neg > 0.5], 0.9)
    yt = g2.arm_signals(arm="YT", window=window)
    assert yt["p"] is not None
    assert np.array_equal(yt["p"], window.x_neg)
    for arm in g2.SYNTHETIC_ARMS:
        assert g2.check_synthetic_arm(arm=arm, window=window).passed, arm
        assert g2.synthetic_chart(arm=arm, window=window).to_dict()["vconcat"]
    with pytest.raises(g2.Gallery2Error):
        g2.arm_signals(arm="Z", window=window)


def test_a_silent_untangler_fails_the_yu_check() -> None:
    """P(reverse flow) below 0.5 on the reverse-flow half-hours fails YU."""
    seg, bank = _arrays()
    seg.u_raw[:, core.P_NEG] = 0.05
    window = g2.synthetic_window(arrays=seg, segment=bank, first_day=1, mw_per_point=0.1)
    assert not g2.check_synthetic_arm(arm="YU", window=window).passed


def _rerun_segment() -> sy.SegmentArrays:
    seg = make_segment(n_days=20, seed=5)
    n = seg.n_days
    seg.u_raw[:, core.P_NEG] = 0.9
    seg.x_neg[:] = 1.0
    seg.u_rerun = np.zeros((n, 2, 48, 3), dtype=np.float32)
    seg.u_rerun[..., core.P_NEG] = 0.9
    seg.u_rerun_ok = np.ones((n, 2, 48), dtype=bool)
    return seg


def test_g14_counts_rows_in_the_nearest_bin_and_the_large_differences() -> None:
    """Re-run values of 0 on the last visible half-hour count as large differences."""
    seg = _rerun_segment()
    base = g2.g14_counts(segments=[seg])
    assert base["n_rows"] > 0
    assert base["n_big"] == 0
    assert seg.u_rerun is not None
    seg.u_rerun[:, 1, 17, core.P_NEG] = 0.0
    got = g2.g14_counts(segments=[seg])
    assert got["n_big"] == got["n_big_at_limit"] > 0
    assert got["n_big_true_reverse"] == got["n_big"]
    assert got["hours_to_first_target"] == 27.0
    assert got["hours_to_first_scored"] == 51.0
    text = g2.g14_explanation(counts=got)
    assert text.count("\n\n") == 2
    assert str(int(got["n_rows"])) in text
    assert "27 hours" in text


def test_g14_counts_stop_when_the_bank_has_no_rerun() -> None:
    """A bank with no re-run raises rather than reporting zero rows."""
    with pytest.raises(g2.Gallery2Error):
        g2.g14_counts(segments=[make_segment(n_days=10)])


def test_the_stage_is_registered_as_a_cpu_stage() -> None:
    """The cli lists the stage and the guard treats it as CPU only."""
    import e00733_cli as cli
    import e00733_stages as st

    assert "gallery2" in cli.STAGES
    assert "gallery2" in st.CPU_ONLY_STAGES


def test_the_channels_are_gathered_as_the_fit_gathers_them() -> None:
    """FU reads the table at the token's grid index and FU-S reads it 28 days earlier."""
    pytest.importorskip("torch")
    import e00733_channels as ch

    n_grid = 48 * 120
    gi = np.arange(n_grid)
    values = np.stack([gi / n_grid, np.zeros(n_grid), gi[::-1] / n_grid], axis=1).astype(np.float32)
    table = ch.ChannelTable(grid_index=gi, values=values)
    choice = ch.ChannelChoice(
        p_neg_source="U*", usable=("p_neg", "step_p"), dropped=("solar",), reasons={},
        fu_equals_fs=False,
    )  # fmt: skip
    origin = np.array([100])
    ok = np.ones(n_grid, dtype=bool)
    kw = {
        "choice": choice,
        "table": table,
        "origin_day": origin,
        "reading_ok": ok,
        "n_grid": n_grid,
    }
    fu, fu_ok = g2.receive_channels(arm="FU", target_std=0.2, **kw)
    fus, fus_ok = g2.receive_channels(arm="FU-S", target_std=0.2, **kw)
    hist = g2.history_grid_index(origin_day=origin)[0, g2.shown_rows()].reshape(-1)
    assert fu_ok.all()
    assert np.allclose(fu[0].reshape(-1, 3)[:, core.P_NEG], hist / n_grid)
    assert not fu[..., core.SOLAR].any()
    assert np.allclose(fus[0].reshape(-1, 3)[:, core.P_NEG], (hist - 28 * 48) / n_grid)
    assert fus_ok.all()
