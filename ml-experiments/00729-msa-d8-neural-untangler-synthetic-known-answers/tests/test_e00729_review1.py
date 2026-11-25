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
