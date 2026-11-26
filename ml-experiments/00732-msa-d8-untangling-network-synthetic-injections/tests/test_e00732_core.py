"""Tests of the constants, the limits, the verdict table, and the schedule of experiment 732."""

import json
import math
from pathlib import Path
from typing import Any

import e00729_core as core729
import e00732_core as core
import numpy as np
import pytest


def reading(point: float, lower: float, upper: float) -> core729.Reading:
    return core729.Reading(point, lower, upper)


def inputs(**kw: Any) -> core.VerdictInputs:
    base: dict[str, Any] = {
        "stopped": False, "invalid": False, "silent_test_missed": False, "failed": False,
        "h1_pooled_pass": True, "h1_high_pass": True, "h2_pass": True, "band_fail": False,
        "c1": reading(0.0, -0.001, 0.001), "margin": core.MARGIN_730,
    }  # fmt: skip
    return core.VerdictInputs(**{**base, **kw})


def test_level_and_counts() -> None:
    assert core.N_CONTRASTS == len(core.CONTRASTS) == 7
    assert math.isclose(core.LEVEL, 1 - 0.05 / 7)
    assert round(100 * core.LEVEL, 3) == 99.286
    assert core.N_FITS == len(core.fit_jobs()) == 24 == 3 * len(core.ARMS) == 3 * 8
    assert core.N_REPORT_ONLY == 23
    assert list(core.REPORT_ONLY) == [f"R{i}" for i in range(1, 24)]
    assert core.HARM_READING not in core.CONTRASTS  # HB sits outside the count


def test_every_verdict_row_is_reachable() -> None:
    cases = {
        0: inputs(stopped=True), 1: inputs(invalid=True), 2: inputs(failed=True),
        3: inputs(h1_pooled_pass=False), 4: inputs(),
        5: inputs(c1=reading(0.002, -0.0001, 0.003)), 6: inputs(c1=reading(0.002, 0.0005, 0.003)),
    }  # fmt: skip
    for row, v in cases.items():
        assert core.verdict_row(v=v) == row
    assert set(core.VERDICT_WORDS) == set(cases)
    assert core.verdict_word(v=cases[5]) == "inconclusive"
    assert core.verdict_word(v=cases[6]) == "not supported"
    assert core.verdict_word(v=cases[4]) == "supported"


def test_a_silent_test_that_does_not_fail_is_row_one() -> None:
    assert core.verdict_row(v=inputs(silent_test_missed=True)) == 1
    assert core.verdict_row(v=inputs(silent_test_missed=True, failed=True)) == 1
    assert core.verdict_row(v=inputs(stopped=True, silent_test_missed=True)) == 0


def test_row_order_first_match_wins() -> None:
    assert core.verdict_row(v=inputs(stopped=True, invalid=True, failed=True)) == 0
    assert core.verdict_row(v=inputs(invalid=True, failed=True)) == 1
    assert core.verdict_row(v=inputs(h1_high_pass=False, c1=reading(0.01, 0.005, 0.02))) == 3
    assert core.verdict_row(v=inputs(h2_pass=False)) == 3
    assert core.verdict_row(v=inputs(band_fail=True)) == 3
    assert core.harm_row_fails(v=inputs(band_fail=True)) is not None
    assert core.harm_row_fails(v=inputs()) is None


def test_the_band_fails_only_with_a_positive_harm_lower_bound() -> None:
    assert core.band_fails(band_mean=0.09, band_limit_value=0.083, hb_lower=0.001)
    assert not core.band_fails(band_mean=0.09, band_limit_value=0.083, hb_lower=0.0)  # boundary
    assert not core.band_fails(band_mean=0.08, band_limit_value=0.083, hb_lower=0.01)


def test_non_inferiority_edge() -> None:
    m = core.MARGIN_730
    assert core.verdict_row(v=inputs(c1=reading(0.0, -0.001, m))) == 4
    assert core.verdict_row(v=inputs(c1=reading(0.0, -0.001, m + 1e-9))) == 5
    assert core.ni_note(c1=reading(0.0005, 0.0001, m), margin=m) is not None
    assert core.ni_note(c1=reading(0.0, -0.001, m), margin=m) is None


def test_which_of_c6_and_r4_carries_a_change() -> None:
    zero, down = reading(0.0, -0.01, 0.01), reading(-0.01, -0.02, -0.005)
    assert core.sign_change_carrier(c6=zero, r4=zero) == "neither"
    assert core.sign_change_carrier(c6=down, r4=zero) == "C6"
    assert core.sign_change_carrier(c6=zero, r4=down) == "R4"
    assert core.sign_change_carrier(c6=down, r4=down) == "both C6 and R4"
    nan = reading(float("nan"), float("nan"), float("nan"))
    assert core.sign_change_carrier(c6=nan, r4=zero) == "neither"


def test_front_matter_word_of_row_zero() -> None:
    assert core.front_matter_word(word="stopped") == "inconclusive"
    assert core.front_matter_word(word="invalid") == "invalid"


def test_the_solar_word_follows_the_eight_bars() -> None:
    allp = dict.fromkeys(core.SOLAR_BARS, True)
    assert len(core.SOLAR_BARS) == 8
    assert core.solar_word(bars=allp, reachable=True) == "reasonable"
    assert core.solar_word(bars={**allp, "night": False}, reachable=True) == "partly"
    assert core.solar_word(bars={**allp, "response": False}, reachable=True) == "partly"
    for name in core.SOLAR_CORE_BARS:
        assert core.solar_word(bars={**allp, name: False}, reachable=True) == "not shown"
    assert core.solar_word(bars={}, reachable=True) == "not shown"  # a missing bar fails
    assert core.solar_word(bars=allp, reachable=False) == "void"


def test_the_neighbour_and_bsp_words() -> None:
    gain, none, loss = (
        reading(-0.01, -0.02, -0.005),
        reading(0.0, -0.01, 0.01),
        reading(0.01, 0.005, 0.02),
    )
    assert core.neighbour_word(c7=gain, reachable=True) == "the neighbours help"
    assert core.neighbour_word(c7=none, reachable=True) == "no detectable difference"
    assert core.neighbour_word(c7=loss, reachable=True) == "the neighbours hurt"
    assert core.neighbour_word(c7=gain, reachable=False) == "void"
    assert core.bsp_word(r19=gain, n_reverse_half_hours=500) == "helps"
    assert core.bsp_word(r19=none, n_reverse_half_hours=500) == "no detectable difference"
    assert core.bsp_word(r19=loss, n_reverse_half_hours=500) == "hurts"
    assert core.bsp_word(r19=gain, n_reverse_half_hours=199) == "void"
    assert core.bsp_word(r19=None, n_reverse_half_hours=500) == "void"
    assert core.reading_word(reading=none) == "no detectable difference"


def test_detectable_band_needs_recall_and_precision() -> None:
    assert core.detectable_band(recall=0.5, precision=0.5)
    assert not core.detectable_band(recall=0.49, precision=0.9)
    assert not core.detectable_band(recall=0.9, precision=0.49)


def test_band_limits_follow_the_formula_and_round_down() -> None:
    r = np.array([5.0, 5.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0] * 10)
    x = r.copy()
    x[:2] = -5.0  # the first two rows of each ten: a quarter of the 1 to 3 MVA band reads reverse
    x[0] = x[1] = -5.0
    limits = core.band_limits(x=x, r=r)
    band = limits["1-3MVA"]
    s = float(np.mean(x[(r >= core.DECIDABLE_READING_POINTS) & (r < core.BAND_EDGE_POINTS)] < 0))
    assert math.isclose(band.share, s)
    assert band.limit <= 0.25 * s / (1 - s)
    assert band.limit == pytest.approx(0.25 * s / (1 - s), rel=0.1)
    assert limits["3MVA+"].limit == 0.0


def test_limits_file_is_write_once_and_read_back(tmp_path: Path) -> None:
    r = np.linspace(0.0, 60.0, 5000)
    x = np.where(r < 3.0, -r, r)
    limits = core.band_limits(x=x, r=r)
    path = tmp_path / "limits.json"
    core.write_limits_file(path=path, limits=limits, s=float(np.mean(x < 0)), n_rows=len(x))
    with pytest.raises(FileExistsError):
        core.write_limits_file(path=path, limits=limits, s=0.0, n_rows=0)
    assert core.read_limits_file(path=path)["pooled"].limit == limits["pooled"].limit
    assert json.loads(path.read_text())["n_rows"] == 5000
    with pytest.raises(FileNotFoundError):
        core.read_limits_file(path=tmp_path / "nope.json")


def test_limit_problems_catch_a_different_pooled_limit_and_empty_bands() -> None:
    ok = {"pooled": core.BandLimit(0.0277, core.POOLED_LIMIT_730, 10)}
    assert core.limit_problems(limits=ok) == []
    off = {"pooled": core.BandLimit(0.03, 0.0072, 10)}
    assert core.limit_problems(limits=off)
    empty = {**ok, "3MVA+": core.BandLimit(float("nan"), 0.0, 0)}
    assert any("3MVA+" in p for p in core.limit_problems(limits=empty))


def test_transfer_word_covers_every_case() -> None:
    gain, none, loss = (
        reading(-0.01, -0.02, -0.005),
        reading(0.0, -0.01, 0.01),
        reading(0.01, 0.005, 0.02),
    )
    assert core.transfer_word(c5=gain, f1=0.7, reachable=True) == "works"
    assert core.transfer_word(c5=gain, f1=0.5, reachable=True) == "works only in Brier score"
    assert core.transfer_word(c5=none, f1=0.9, reachable=True) == "no better than the rule"
    assert core.transfer_word(c5=loss, f1=0.9, reachable=True) == "worse than the rule"
    assert core.transfer_word(c5=gain, f1=0.9, reachable=False) == "void"
    assert (
        core.transfer_word(
            c5=reading(float("nan"), float("nan"), float("nan")), f1=0.9, reachable=True
        )
        == "void"
    )


def test_projection_prices_each_arm_at_its_own_rate() -> None:
    secs = {"U*": 1.6, "U-N": 1.6, "U-NC": 1.1, "S+": 0.45, "U-SG": 1.6, "U*-L": 2.1}
    hours = core.projected_hours(seconds_per_epoch=secs)
    # U-noBSP is priced at U*'s rate and U-NCw at U-NC's
    per_arm = 1.6 + 1.6 + 1.1 + 1.1 + 0.45 + 1.6 + 2.1 + 1.6
    assert hours == pytest.approx(3 * 300 * per_arm / 3600 + 0.3 + 0.05)
    with pytest.raises(KeyError):
        core.projected_hours(seconds_per_epoch={"U*": 1.0})


def test_the_drop_rule_removes_report_only_arms_from_the_end_of_the_fit_order() -> None:
    fast = {"U*": 1.6, "U-N": 1.6, "U-NC": 1.1, "S+": 0.45, "U-SG": 1.6, "U*-L": 2.1}
    assert core.drop_report_only(seconds_per_epoch=fast).dropped == ()
    # a rate that puts all 24 fits over the line while the 12 counted fits stay under it
    slow = {k: v * 7.0 for k, v in fast.items()}
    plan = core.drop_report_only(seconds_per_epoch=slow)
    assert plan.dropped[0] == "U*-L"
    assert plan.dropped == core.DROP_ORDER[: len(plan.dropped)]
    assert plan.hours <= core.STOP_LINE_HOURS
    assert not plan.stop
    assert plan.dropped
    assert set(plan.kept) >= set(core.COUNTED_ARMS)
    # every report-only arm is dropped last in the fit order
    order = [a for _, arms in core.FIT_FAMILIES for a in arms]
    assert order[: len(core.COUNTED_ARMS)] == ["S+", "U*", "U-N", "U-NC"]
    assert set(order[len(core.COUNTED_ARMS) :]) == set(core.REPORT_ONLY_ARMS)
    assert [a for a in order if a in core.DROP_ORDER] == list(core.DROP_ORDER)[::-1]
    # when the counted arms alone exceed the line, the line stops
    crawl = {k: v * 40 for k, v in fast.items()}
    stopped = core.drop_report_only(seconds_per_epoch=crawl)
    assert stopped.stop
    assert set(stopped.dropped) == set(core.REPORT_ONLY_ARMS)


def test_timing_gate_uses_the_kept_arms() -> None:
    secs = {"U*": 1.6, "U-N": 1.6, "U-NC": 1.1, "S+": 0.45, "U-SG": 1.6, "U*-L": 2.1}
    report = core.timing_gate_report(
        seconds_per_epoch=secs, peak_gpu_gb=3.0, peak_rss_gb=5.0, pair_seconds_per_epoch=0.2
    )
    assert core.timing_gate_passes(report=report, manager_approved=False)
    slow = core.timing_gate_report(
        seconds_per_epoch=dict.fromkeys(secs, 40.0),
        peak_gpu_gb=3.0,
        peak_rss_gb=5.0,
        pair_seconds_per_epoch=0.2,
    )
    assert slow["counted_arms_over_line"]
    assert not core.timing_gate_passes(report=slow, manager_approved=False)
    assert core.timing_gate_passes(report=slow, manager_approved=True)
    big = core.timing_gate_report(
        seconds_per_epoch=secs, peak_gpu_gb=7.0, peak_rss_gb=5.0, pair_seconds_per_epoch=0.2
    )
    assert not core.timing_gate_passes(report=big, manager_approved=True)


def test_arm_weights_and_inputs_follow_the_plan() -> None:
    a = core.ARMS
    assert len(a) == 8
    assert (
        a["U*"].weights["injection"]
        == a["U*"].weights["night"]
        == a["U*"].weights["monotone"]
        == 0.02
    )
    assert (
        a["U-NC"].weights["injection"]
        == a["U-NC"].weights["night"]
        == a["U-NC"].weights["monotone"]
        == 0.0
    )
    assert a["U-NCw"].weights["solar"] == 0.08
    assert a["U-NC"].weights["solar"] == 0.02
    assert all(v == 0.0 for k, v in a["S+"].weights.items() if k != "sign")
    assert not a["S+"].heads
    assert a["U-SG"].stop_gradient
    assert not a["U*"].stop_gradient
    assert a["U*-L"].arch == "bilstm"
    assert a["U*"].raised
    assert a["U-SG"].raised
    assert a["U*-L"].raised
    assert not a["U-NC"].raised
    assert a["U-NC"].inject
    assert a["U-NCw"].inject
    assert not a["S+"].inject
    # U-N and U-noBSP train like U* and remove one input each
    for name in ("U-N", "U-noBSP"):
        assert a[name].weights == a["U*"].weights
        assert a[name].inject
        assert a[name].raised
    assert not a["U-N"].neighbours
    assert a["U-N"].bsp_zeroed is False
    assert a["U-noBSP"].bsp_zeroed
    assert a["U-noBSP"].neighbours
    assert all(a[n].neighbours and not a[n].bsp_zeroed for n in a if n not in ("U-N", "U-noBSP"))


def test_the_transfer_constants_match_the_plan() -> None:
    assert sum(core.PARTNER_MIX.values()) == pytest.approx(1.0)
    assert core.PARTNER_MIX == {"read": 0.6, "unread": 0.2, "other_bsp": 0.2}
    assert sum(core.SHAPE_MIX.values()) == pytest.approx(1.0)
    assert core.SHAPE_MIX == {"constant": 0.6, "demand": 0.2, "solar": 0.2}
    assert core.DONOR_CAP == 0.8
    assert core.DONOR_REDRAWS == 20
    assert core.SHAPE_FLOOR == 0.1
    assert core.MIRROR_JITTER_SHARE == 0.3
    assert core.MIRROR_JITTER_RANGE == (0.7, 1.3)
    assert core.READ_PARTNERS == ("D3", "D5", "D14")
    assert core.SOLAR_SHAPE_KAPPA == (0.3, 1.0)


def test_limits_file_holds_the_noise_units_when_given(tmp_path: Path) -> None:
    r = np.linspace(0.0, 60.0, 5000)
    x = np.where(r < 3.0, -r, r)
    limits = core.band_limits(x=x, r=r)
    units = {"D8": 1.0, "D3": 2.0, "D5": 3.0, "D14": 4.0}
    path = tmp_path / "limits.json"
    core.write_limits_file(path=path, limits=limits, s=0.1, n_rows=len(x), noise_units=units)
    assert core.read_noise_units(path=path) == units
    with pytest.raises(ValueError, match="noise units"):
        core.write_limits_file(
            path=tmp_path / "bad.json", limits=limits, s=0.1, n_rows=1, noise_units={"D8": 1.0}
        )
    bare = tmp_path / "bare.json"
    core.write_limits_file(path=bare, limits=limits, s=0.1, n_rows=1)
    with pytest.raises(KeyError):
        core.read_noise_units(path=bare)


def test_job_names_have_no_asterisk_or_plus() -> None:
    names = {core.job_name(arm=a, arch=ar, seed=s) for a, ar, s in core.fit_jobs()}
    assert len(names) == 24
    assert not any("*" in n or "+" in n for n in names)


def test_every_arm_has_a_family_and_the_families_cover_the_fits() -> None:
    assert {core.family_of(arm=a) for a in core.FIT_ARMS} <= {n for n, _ in core.FIT_FAMILIES}
    assert sorted(a for _, arms in core.FIT_FAMILIES for a in arms) == sorted(core.FIT_ARMS)


def test_panels_g10_to_g21_and_windows_w1_to_w25_are_listed() -> None:
    assert sorted(core.PANEL_REQUIREMENTS) == sorted(
        [*core.GALLERY_PUBLISHED, *core.GALLERY_PRIVATE_ONLY]
    )
    assert len(core.PANEL_REQUIREMENTS) == 12
    assert sorted(k.split("_")[0] for k in core.PANEL_REQUIREMENTS) == sorted(
        f"G{i}" for i in range(10, 22)
    )
    assert core.GALLERY_PRIVATE_ONLY == ("G17_s1_pairs",)
    assert list(core.SERIES_WINDOWS) == [f"W{i}" for i in range(1, 26)]


def test_stage_names_put_the_neighbours_before_the_limits() -> None:
    names = list(core.stage_names())
    assert names.index("neighbours") < names.index("limits") < names.index("overlays")
    assert len(set(names)) == len(names)
    assert "loss_curves" in names
