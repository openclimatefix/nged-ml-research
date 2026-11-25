# ruff: noqa: E501
"""Tests that came out of the first code review: each fails on a mutation that the older tests missed."""

import dataclasses

import e00729_banks as banks
import e00729_core as core
import e00729_data as data
import e00729_dist as dist
import e00729_leakage as leak
import e00729_real as real
import e00729_score as score
import e00729_stages as stages
import numpy as np
import polars as pl
import pytest
from synthetic_world import make_calendar

SPLIT_FILE = """# Experiment 729 split months (private, never committed)

Rule text that mentions a | pipe and digits 1 | 2 | 3.

| Number | Position | Month | Split | Carries BSP steps |
|---|---|---|---|---|
| 1 | None | 2020-04 | training | False |
| 2 | 1 | 2022-01 | training | False |
| 3 | 2 | 2022-02 | test | False |
| 4 | None | 2020-05 | training | True |
| 5 | 3 | 2022-03 | validation | True |
"""


def test_the_private_splits_parse_keeps_lockdown_rows_with_a_none_position(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "splits.md"
    path.write_text(SPLIT_FILE)
    assert stages.parse_private_splits(path=path) == {
        "2020-04": "training", "2022-01": "training", "2022-02": "test", "2020-05": "training",
        "2022-03": "validation",
    }  # fmt: skip


def test_the_parse_matches_the_rule_for_a_file_written_in_the_plan_scripts_format(tmp_path) -> None:  # noqa: ANN001
    months = [f"2022-{m:02d}" for m in range(1, 10)]
    rule = core.month_splits(months=months, n_days=[28] * 9, bsp_days=[25] * 9)
    lines = ["| Number | Position | Month | Split | Carries BSP steps |", "|---|---|---|---|---|"]
    lines += [f"| {i + 1} | {i + 1} | {m} | {rule[m]} | True |" for i, m in enumerate(months)]
    path = tmp_path / "splits.md"
    path.write_text("\n".join(lines))
    assert stages.parse_private_splits(path=path) == rule
    assert set(rule.values()) >= {"training", "test"}


def test_a_parse_that_differs_from_the_rule_is_detected_by_comparison(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "splits.md"
    path.write_text(SPLIT_FILE)
    parsed = stages.parse_private_splits(path=path)
    assert parsed["2022-02"] == "test"
    assert parsed["2022-01"] == "training"
    assert set(parsed.values()) == {"training", "test", "validation"}


def test_real_step_is_nan_across_a_gap_between_days_and_at_missing_values() -> None:
    days = np.array(["2022-01-01", "2022-01-02", "2022-01-05"], dtype="datetime64[D]")
    values = np.arange(3 * 48, dtype=float).reshape(3, 48)
    values[0, 10] = np.nan
    steps = stages.real_step(values=values, days=days).reshape(3, 48)
    assert np.isnan(steps[0, 0])  # the first day has no previous half-hour
    assert steps[1, 0] == 1.0  # day 2 follows day 1
    assert np.isnan(steps[2, 0])  # day 5 follows a gap of two days
    assert steps[2, 1] == 1.0
    assert np.isnan(steps[0, 10])
    assert np.isnan(steps[0, 11])


def test_closure_days_cover_the_window_and_a_day_each_side_for_the_named_substation() -> None:
    from datetime import date, datetime

    sheet = pl.DataFrame(
        {
            "Substation number": [7, 7, 9],
            "Start": [datetime(2022, 3, 10, 8), datetime(2022, 6, 1), datetime(2022, 4, 1)],
            "End": [datetime(2022, 3, 11, 20), datetime(2022, 6, 1, 5), datetime(2022, 4, 2)],
        }
    )
    days = data.closure_days_from_sheets(sheets={"s": sheet}, substation_number=7)
    assert days == {
        date(2022, 3, 9), date(2022, 3, 10), date(2022, 3, 11), date(2022, 3, 12),
        date(2022, 5, 31), date(2022, 6, 1), date(2022, 6, 2),
    }  # fmt: skip
    assert (
        data.closure_days_from_sheets(sheets={"s": pl.DataFrame({"x": [1]})}, substation_number=7)
        == set()
    )


def test_verdict_row_7_needs_a_half_width_of_at_least_the_margin() -> None:
    at = core.VerdictInputs(
        stopped=False, invalid=False, failed=False, h1_pass=True, h2_pass=True,
        c1=core.Reading(-0.05, -0.30, 0.20), margin=0.25, calibration_pass=True, lobe_pass=True,
    )  # fmt: skip
    assert at.c1.half_width == pytest.approx(0.25)
    assert core.verdict_row(v=at) == 7
    below = dataclasses.replace(at, c1=core.Reading(-0.05, -0.29, 0.20))
    assert core.verdict_row(v=below) == 8


def make_metrics(n: int, **kw: np.ndarray) -> score.RowMetrics:
    zeros = np.zeros(n)
    fields = dict.fromkeys(score.RowMetrics.__dataclass_fields__, zeros)
    return score.RowMetrics(**{**fields, **kw})


def test_harm_check_reads_only_the_rows_it_is_given() -> None:
    n = 100
    keep = np.arange(n) < 50
    bad_outside = np.where(keep, 0.0, 5.0)
    d = make_metrics(n, crps=bad_outside, mae=bad_outside, p_neg=np.zeros(n))
    ok = score.harm_check(
        d=d, p0f_metrics=make_metrics(n), p0_metrics=make_metrics(n), keep=keep, real=True
    )
    assert ok["pass"]
    bad_inside = np.where(keep, 5.0, 0.0)
    d2 = make_metrics(n, crps=bad_inside, mae=np.zeros(n), p_neg=np.zeros(n))
    assert not score.harm_check(
        d=d2, p0f_metrics=make_metrics(n), p0_metrics=make_metrics(n), keep=keep, real=False
    )["pass"]
    alarm = np.where(keep, 0.0, 1.0)
    d3 = make_metrics(n, p_neg=alarm)
    assert score.harm_check(
        d=d3, p0f_metrics=make_metrics(n), p0_metrics=make_metrics(n), keep=keep, real=True
    )["pass"]


def test_the_reliability_term_weights_each_bin_by_its_count() -> None:
    rng = np.random.default_rng(0)
    big = np.full(900, 0.15)
    small = np.full(100, 0.85)
    prob = np.r_[big, small]
    observed = np.r_[
        rng.permutation(np.r_[np.ones(135), np.zeros(765)]).astype(bool),
        rng.permutation(np.r_[np.ones(65), np.zeros(35)]).astype(bool),
    ]
    rel = dist.reliability_term(prob=prob, observed=observed)
    assert rel["reliability"] == pytest.approx(
        0.1 * 0.2**2, rel=1e-6
    )  # equal bin weights would give 0.02


def test_the_bootstrap_widens_when_only_segments_within_each_month_differ() -> None:
    cal = make_calendar()
    use = {m: np.array([str(d)[:7] == m for d in cal.dates]) for m in ("2022-03", "2022-04")}
    bank = banks.build_bank(
        cal=cal, kind="test", seed=None, months=["2022-03", "2022-04"], month_use=use
    )  # ty: ignore[invalid-argument-type]
    rows = score.rows_from_segments(segments=bank.segments, holidays=cal.holidays)
    effect = np.where(rows.seg % 2 == 0, -1.0, 1.0)  # every month's segments average to zero
    r = score.contrast(
        rows=rows, a=effect, b=np.zeros(rows.n), keep=np.ones(rows.n, dtype=bool), n_resamples=2000
    )
    assert r.reading.point == pytest.approx(0.0, abs=1e-9)
    assert r.reading.upper - r.reading.lower > 0.2


def test_k1_returns_the_error_ratio_only_and_ignores_missing_record_half_hours() -> None:
    n = core.K1_MIN_RUNNING + 100
    record = np.r_[np.full(n, 30.0), np.zeros(500), np.full(300, np.nan)]
    head = np.r_[
        np.full(n, 30.0), np.zeros(500), np.full(300, 99.0)
    ]  # wrong only where the record is missing
    out = real.k1_engine_head(
        head_median=head,
        record=record,
        running=np.nan_to_num(record) >= 10,
        m2_flag=np.nan_to_num(record) >= 10,
    )
    assert out["mae_ratio"] == 0.0
    assert out["status"] == "pass"
    assert not {"mae_head", "mae_zero", "n_running"} & set(out)


def test_assert_banks_independent_fails_when_two_banks_share_a_drawn_parameter() -> None:
    cal = make_calendar()
    use = {"2022-03": np.array([str(d)[:7] == "2022-03" for d in cal.dates])}
    kw = {"cal": cal, "seed": None, "months": ["2022-03"], "month_use": use}
    val = banks.build_bank(kind="validation", **kw)  # ty: ignore[invalid-argument-type]
    test = banks.build_bank(kind="test", **kw)  # ty: ignore[invalid-argument-type]
    leak.assert_banks_independent(evaluation=[val, test])
    twin = banks.build_bank(kind="test", **kw)  # ty: ignore[invalid-argument-type]
    twin.kind = "validation"
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_banks_independent(evaluation=[twin, test])
    twin.root = test.root
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_seed_spaces(train_seeds=[0], evaluation=[twin, test])


def test_the_cams_check_passes_a_different_farm_and_fails_a_farm_in_the_same_cell() -> None:
    import dataclasses

    cal = make_calendar()
    assert not np.allclose(cal.farms.ghi[0], cal.d8_ghi)
    leak.assert_cams_differs(cal=cal)
    ghi = cal.farms.ghi.copy()
    ghi[1] = np.where(ghi[1] > 0, ghi[1], 1e-3)
    same = dataclasses.replace(
        cal, d8_ghi=ghi[1].copy(), farms=dataclasses.replace(cal.farms, ghi=ghi)
    )
    with pytest.raises(leak.CorrectnessFailureError):
        leak.assert_cams_differs(cal=same)
    # a farm that equals D8 at well under half of the daytime half-hours is accepted
    partly = ghi[1].copy()
    partly[:, np.arange(48) % 5 < 3] *= 2.0
    leak.assert_cams_differs(
        cal=dataclasses.replace(cal, d8_ghi=partly, farms=dataclasses.replace(cal.farms, ghi=ghi))
    )


def test_a_gate_with_one_finite_fit_reads_void_and_the_record_shows_the_finite_count() -> None:
    real_fit = banks.StepFit(-0.76, 0.015, 4.0, 100)
    nan = banks.StepFit(float("nan"), float("nan"), float("nan"), 0)
    one = banks.bsp_gate(
        real=real_fit,
        real_ci=(-0.9, -0.6),
        synthetic=[banks.StepFit(-0.8, 0.015, 4.0, 100), nan, nan],
    )
    assert one["void"]
    assert one["n_finite_fits"] == 1
    assert not one["pass"]
    few = [banks.StepFit(-0.8, 0.015, 4.0, 100)] * (core.GATE_MIN_SYNTHETIC_FITS - 1)
    assert banks.bsp_gate(real=real_fit, real_ci=(-0.9, -0.6), synthetic=few)["void"]
    enough = [
        banks.StepFit(-0.8, 0.01 + 0.001 * k, 3.0 + 0.2 * k, 100)
        for k in range(core.GATE_MIN_SYNTHETIC_FITS)
    ]
    ok = banks.bsp_gate(real=real_fit, real_ci=(-0.9, -0.6), synthetic=enough)
    assert not ok["void"]
    assert ok["n_finite_fits"] == core.GATE_MIN_SYNTHETIC_FITS


def test_the_gate_compares_the_standard_error_at_equal_pair_counts() -> None:
    synthetic = [banks.StepFit(-0.8, 0.02 + 0.001 * k, 4.0, 100) for k in range(10)]
    # the same se * sqrt(n) as the synthetic fits, from a real stretch with four times the pairs
    matched = banks.StepFit(-0.8, 0.025 / 2.0, 4.0, 400)
    assert banks.bsp_gate(real=matched, real_ci=(-0.9, -0.6), synthetic=synthetic)["se_inside"]
    raw = banks.StepFit(-0.8, 0.025, 4.0, 1)
    assert not banks.bsp_gate(real=raw, real_ci=(-0.9, -0.6), synthetic=synthetic)["se_inside"]
    assert not banks.bsp_gate(
        real=banks.StepFit(-0.8, 0.025, 4.0, 400), real_ci=(-0.9, -0.6), synthetic=synthetic
    )["se_inside"]


def test_the_engine_only_segments_are_spread_over_the_months() -> None:
    worlds = core.quota_world_types(rng=np.random.default_rng(0), n=64, none_share=0.15)
    dealt = banks.deal_across_months(worlds=worlds, n_months=8)
    assert sorted(dealt) == sorted(worlds)
    for month in range(8):
        chunk = dealt[month * 8 : (month + 1) * 8]
        assert len(chunk) == 8
        assert chunk.count("engine") in {2, 3}
        assert chunk.count("none") in {1, 2}


def make_root(tmp_path, *, margin_late: bool = False):  # noqa: ANN001
    import os

    (tmp_path / "scores").mkdir(parents=True, exist_ok=True)
    for rel, t in (("selection.json", 100), ("scores/bstar.json", 110), ("margin.json", 120)):
        (tmp_path / rel).write_text("{}")
        os.utime(tmp_path / rel, ns=(t * 10**9, t * 10**9))
    marker = tmp_path / "scores" / stages.TEST_MARKER
    marker.write_text("")
    os.utime(marker, ns=(150 * 10**9, 150 * 10**9))
    if margin_late:
        os.utime(tmp_path / "margin.json", ns=(200 * 10**9, 200 * 10**9))
    return tmp_path


def test_a_choice_rewritten_after_the_first_test_score_is_detected(tmp_path) -> None:  # noqa: ANN001
    root = make_root(tmp_path)
    events = stages.choice_events(root=root)
    assert events[-1] == "test_score"
    leak.assert_choices_before_test(events=events)
    late = stages.choice_events(root=make_root(tmp_path, margin_late=True))
    assert late.index("margin") > late.index("test_score")
    with pytest.raises(leak.CorrectnessFailureError, match="margin"):
        leak.assert_choices_before_test(events=late)


def test_the_first_test_score_marker_is_created_once(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "scores").mkdir()
    for rel in ("selection.json", "scores/bstar.json", "margin.json"):
        (tmp_path / rel).write_text("{}")
    stages.choice_events(root=tmp_path)
    marker = tmp_path / "scores" / stages.TEST_MARKER
    first = marker.stat().st_mtime_ns
    stages.choice_events(root=tmp_path)
    assert marker.stat().st_mtime_ns == first


def test_a_second_run_cannot_change_bstar(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "bstar.json"
    stages.write_bstar_once(path=path, bstar="B1")
    stages.write_bstar_once(path=path, bstar="B1")
    with pytest.raises(leak.CorrectnessFailureError):
        stages.write_bstar_once(path=path, bstar="B2")


def test_the_closure_log_catches_only_the_expected_failures(monkeypatch) -> None:  # noqa: ANN001
    import sys
    import types

    def fake_module(read):  # noqa: ANN001, ANN202
        module = types.SimpleNamespace(SWITCHING_LOG_PATH="x", METADATA_PATH="y")
        monkeypatch.setitem(sys.modules, "e00722_data", module)
        monkeypatch.setattr(pl, "read_excel", read)

    def missing(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise FileNotFoundError

    fake_module(missing)
    assert data.load_closure_days() == (frozenset(), "missing")

    def unreadable(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ValueError

    fake_module(unreadable)
    assert data.load_closure_days() == (frozenset(), "unreadable")

    def unexpected(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ZeroDivisionError

    fake_module(unexpected)
    with pytest.raises(ZeroDivisionError):
        data.load_closure_days()
