"""Tests of the constants, the limits, the verdict table, and the schedule of experiment 732."""

import json
import math

import e00729_core as core729
import e00732_core as core
import numpy as np
import pytest


def reading(point: float, lower: float, upper: float) -> core729.Reading:
    return core729.Reading(point, lower, upper)


def inputs(**kw) -> core.VerdictInputs:
    base = {
        "stopped": False, "invalid": False, "failed": False, "h1_pooled_pass": True, "h1_high_pass": True,
        "h2_pass": True, "band_fail": False, "c1": reading(0.0, -0.001, 0.001), "margin": core.MARGIN_730,
        "solar_bars": dict.fromkeys(core.SOLAR_BARS, True), "c2_gain": True,
    }  # fmt: skip
    return core.VerdictInputs(**{**base, **kw})


def test_level_and_counts() -> None:
    assert core.N_CONTRASTS == len(core.CONTRASTS) == 6
    assert math.isclose(core.LEVEL, 1 - 0.05 / 6)
    assert core.N_FITS == len(core.fit_jobs()) == 18
    assert core.N_REPORT_ONLY == 13


def test_every_verdict_row_is_reachable() -> None:
    cases = {
        0: inputs(stopped=True), 1: inputs(invalid=True), 2: inputs(failed=True),
        3: inputs(h1_pooled_pass=False), 4: inputs(), 5: inputs(c2_gain=False),
        6: inputs(solar_bars={**dict.fromkeys(core.SOLAR_BARS, True), "night": False}),
        7: inputs(c1=reading(0.002, -0.0001, 0.003)), 8: inputs(c1=reading(0.002, 0.0005, 0.003)),
    }  # fmt: skip
    for row, v in cases.items():
        assert core.verdict_row(v=v) == row


def test_row_order_first_match_wins() -> None:
    assert core.verdict_row(v=inputs(stopped=True, invalid=True, failed=True)) == 0
    assert core.verdict_row(v=inputs(invalid=True, failed=True)) == 1
    assert core.verdict_row(v=inputs(h1_high_pass=False, c1=reading(0.01, 0.005, 0.02))) == 3
    assert core.verdict_row(v=inputs(h2_pass=False)) == 3
    assert core.verdict_row(v=inputs(band_fail=True)) == 3


def test_a_missing_solar_bar_fails() -> None:
    assert core.verdict_row(v=inputs(solar_bars={})) == 6
    assert core.verdict_row(v=inputs(solar_bars={"recovery": True})) == 6


def test_non_inferiority_edge() -> None:
    m = core.MARGIN_730
    assert core.verdict_row(v=inputs(c1=reading(0.0, -0.001, m))) == 4
    assert core.verdict_row(v=inputs(c1=reading(0.0, -0.001, m + 1e-9))) == 7


def test_front_matter_word_of_row_zero() -> None:
    assert core.front_matter_word(word="stopped") == "inconclusive"
    assert core.front_matter_word(word="invalid") == "invalid"


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


def test_limits_file_is_write_once_and_read_back(tmp_path) -> None:
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
    gain, none, loss = reading(-0.01, -0.02, -0.005), reading(0.0, -0.01, 0.01), reading(0.01, 0.005, 0.02)
    assert core.transfer_word(c5=gain, f1=0.7, reachable=True) == "works"
    assert core.transfer_word(c5=gain, f1=0.5, reachable=True) == "works only in Brier score"
    assert core.transfer_word(c5=none, f1=0.9, reachable=True) == "no better than the rule"
    assert core.transfer_word(c5=loss, f1=0.9, reachable=True) == "worse than the rule"
    assert core.transfer_word(c5=gain, f1=0.9, reachable=False) == "void"
    assert core.transfer_word(c5=reading(float("nan"), float("nan"), float("nan")), f1=0.9, reachable=True) == "void"


def test_projection_prices_each_arm_at_its_own_rate_and_stops_over_the_line() -> None:
    secs = {"U*": 1.6, "U-NC": 1.1, "S+": 0.45, "U-SG": 1.6, "U*-L": 2.1}
    hours = core.projected_hours(seconds_per_epoch=secs)
    assert hours == pytest.approx((3 * 300 * (1.6 + 1.1 + 1.1 + 0.45 + 1.6 + 2.1)) / 3600 + 0.3)
    report = core.timing_gate_report(seconds_per_epoch=secs, peak_gpu_gb=3.0, peak_rss_gb=5.0, pair_seconds_per_epoch=0.2)
    assert core.timing_gate_passes(report=report, manager_approved=False)
    slow = core.timing_gate_report(seconds_per_epoch={k: 40.0 for k in secs}, peak_gpu_gb=3.0, peak_rss_gb=5.0, pair_seconds_per_epoch=0.2)
    assert not core.timing_gate_passes(report=slow, manager_approved=False)
    assert core.timing_gate_passes(report=slow, manager_approved=True)
    big = core.timing_gate_report(seconds_per_epoch=secs, peak_gpu_gb=7.0, peak_rss_gb=5.0, pair_seconds_per_epoch=0.2)
    assert not core.timing_gate_passes(report=big, manager_approved=True)
    with pytest.raises(KeyError):
        core.projected_hours(seconds_per_epoch={"U*": 1.0})


def test_arm_weights_follow_the_plan() -> None:
    a = core.ARMS
    assert a["U*"].weights["injection"] == a["U*"].weights["night"] == a["U*"].weights["monotone"] == 0.02
    assert a["U-NC"].weights["injection"] == a["U-NC"].weights["night"] == a["U-NC"].weights["monotone"] == 0.0
    assert a["U-NCw"].weights["solar"] == 0.08 and a["U-NC"].weights["solar"] == 0.02
    assert all(v == 0.0 for k, v in a["S+"].weights.items() if k != "sign") and not a["S+"].heads
    assert a["U-SG"].stop_gradient and not a["U*"].stop_gradient
    assert a["U*-L"].arch == "bilstm"
    assert a["U*"].raised and a["U-SG"].raised and a["U*-L"].raised and not a["U-NC"].raised
    assert a["U-NC"].inject and a["U-NCw"].inject and not a["S+"].inject


def test_job_names_have_no_asterisk_or_plus() -> None:
    names = {core.job_name(arm=a, arch=ar, seed=s) for a, ar, s in core.fit_jobs()}
    assert len(names) == 18 and not any("*" in n or "+" in n for n in names)


def test_every_arm_has_a_family_and_the_families_cover_the_fits() -> None:
    assert {core.family_of(arm=a) for a in core.FIT_ARMS} <= {n for n, _ in core.FIT_FAMILIES}
    assert sorted(a for _, arms in core.FIT_FAMILIES for a in arms) == sorted(core.FIT_ARMS)


def test_panels_g10_to_g17_are_listed() -> None:
    assert sorted(core.PANEL_REQUIREMENTS) == sorted([*core.GALLERY_PUBLISHED, *core.GALLERY_PRIVATE_ONLY])
    assert len(core.PANEL_REQUIREMENTS) == 8
