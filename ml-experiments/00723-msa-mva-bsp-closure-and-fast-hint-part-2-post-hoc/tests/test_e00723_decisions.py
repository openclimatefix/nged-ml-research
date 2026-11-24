# ruff: noqa: ANN001, ANN003
"""The bars, the verdict rules, and the stop rules of experiment 723 on synthetic inputs."""

import e00723_core as core
import e00723_score as score
import numpy as np
import pytest


def inputs(**kw) -> core.W2Inputs:
    base = {
        "correctness_ok": True, "alignment_ok": True, "timestamp_defect": False,
        "n_pairs_keeping_sensor": 12, "n_powered_pairs_level_2": 8, "n_powered_hosts": 3,
        "pooled_se": 0.03, "n_export_stretches": 100, "n_import_stretches": 100,
        "level_1_readable": True,
        "outcomes": {
            "area": "pass", "placebo": "pass", "false_export": "pass", "transfer": "pass",
            "d15_dead_band": "pass", "recall": "pass", "calibration": "void",
        },
    }  # fmt: skip
    base.update(kw)
    return core.W2Inputs(**base)


def with_outcome(**changes: str) -> core.W2Inputs:
    outcomes = {**inputs().outcomes, **changes}
    return inputs(outcomes=outcomes)


def test_named_constants_are_the_values_of_the_plan_page():
    assert core.AREA_BAR == 0.85
    assert core.AREA_BAR_LEVEL_1 == 0.80
    assert core.AREA_LOWER_BAR == 0.75
    assert core.PLACEBO_GAP_BAR == 0.15
    assert core.MAX_POOLED_SE == 0.036
    assert core.MAX_TRANSFER_SE == 0.05
    assert core.SIZE_OF_SKY_INDEX_CHANGE == 0.10
    assert core.SIZE_OF_READING_CHANGE_POINTS == 0.2
    assert core.DEAD_BAND == 0.25
    assert core.MIN_USABLE_PAIRS == 8
    assert core.POWERED_PAIR_NEED == 0.45
    assert core.ARC_REMOVAL_MAX == 0.1
    assert core.ARC_VOID_SHARE == 0.01
    assert core.D15_NO_EVIDENCE_SHARE == 0.90
    assert (core.MIN_PAIRS, core.MIN_HOSTS) == (4, 2)
    assert (core.MIN_STRETCHES, core.MIN_TRANSFER_STRETCHES) == (30, 100)
    assert core.N_LATER_DAYS == 413
    assert core.N_WORLDS == 32
    assert core.CAP_CORE_HOURS == 1.0
    assert core.CALIBRATION_TOLERANCE == 0.1
    assert core.SE_FACTOR == 1.25
    assert core.N_BOOT == 1000
    assert core.PLACEBO_SHIFT_DAYS == 28
    assert core.TRAILING_CAPACITY_DAYS == 91
    assert core.TRANSFER_MARGIN_HALF_HOURS == 2
    assert core.LEVELS == (1.0, 2.0)
    assert core.REACTIVE_FLOOR_POINTS == 2.0


@pytest.mark.parametrize(
    ("n", "allowed"), [(29, 0), (30, 0), (49, 0), (50, 1), (100, 2), (149, 2), (150, 3)]
)
def test_the_floor_rule_allows_floor_of_two_percent(n, allowed):
    assert core.floor_allowance(n=n) == allowed


def test_false_export_bar_is_void_under_30_and_uses_the_floor_rule():
    assert core.false_export_bar(n_export_evidence=0, n_scorable_import=29) == "void"
    assert core.false_export_bar(n_export_evidence=2, n_scorable_import=100) == "pass"
    assert core.false_export_bar(n_export_evidence=3, n_scorable_import=100) == "fail"
    assert core.false_export_bar(n_export_evidence=1, n_scorable_import=30) == "fail"
    assert core.false_export_bar(n_export_evidence=0, n_scorable_import=30) == "pass"


def test_evidence_classes_and_the_dead_band():
    assert core.evidence_of(c=0.26, n_usable=8) == "export"
    assert core.evidence_of(c=0.25, n_usable=8) == "none"
    assert core.evidence_of(c=-0.26, n_usable=8) == "import"
    assert core.evidence_of(c=0.9, n_usable=7) == "none"
    assert core.evidence_of(c=float("nan"), n_usable=20) == "none"


def test_truth_labels_use_80_and_20_percent():
    assert core.truth_of(share_export_half_hours=0.8) == "export"
    assert core.truth_of(share_export_half_hours=0.2) == "import"
    assert core.truth_of(share_export_half_hours=0.5) == "mixed"


def test_transfer_bar_is_void_under_100_stretches_or_with_a_large_error():
    assert core.transfer_bar(share_import_evidence=0.9, n_stretches=99, se=0.01) == "void"
    assert core.transfer_bar(share_import_evidence=0.9, n_stretches=100, se=0.051) == "void"
    assert core.transfer_bar(share_import_evidence=0.5, n_stretches=100, se=0.05) == "pass"
    assert core.transfer_bar(share_import_evidence=0.49, n_stretches=500, se=0.02) == "fail"


def test_dead_band_bar_binds_only_for_an_outside_farm_and_keeps_every_stretch():
    assert core.d15_dead_band_binds(farm_class="outside_or_added_back")
    assert not core.d15_dead_band_binds(farm_class="embedded_not_added_back")
    assert not core.d15_dead_band_binds(farm_class="unresolved")
    assert not core.d15_dead_band_binds(farm_class=None)
    assert core.dead_band_bar(n_no_evidence=10, n_stretches=50, binds=False) == "report"
    assert core.dead_band_bar(n_no_evidence=29, n_stretches=29, binds=True) == "void"
    assert core.dead_band_bar(n_no_evidence=27, n_stretches=30, binds=True) == "pass"
    assert core.dead_band_bar(n_no_evidence=26, n_stretches=30, binds=True) == "fail"


def test_recall_placebo_and_area_bars():
    assert core.recall_bar(n_export_evidence=10, n_scorable_export=29) == "void"
    assert core.recall_bar(n_export_evidence=15, n_scorable_export=30) == "pass"
    assert core.recall_bar(n_export_evidence=14, n_scorable_export=30) == "fail"
    assert core.placebo_bar(real_area=0.9, shifted_area=0.75) == "pass"
    assert core.placebo_bar(real_area=0.9, shifted_area=0.76) == "fail"
    assert core.area_bar(point=0.85, lo=0.76, point_level_1=0.80) == "pass"
    assert core.area_bar(point=0.85, lo=0.75, point_level_1=0.80) == "fail"
    assert core.area_bar(point=0.84, lo=0.8, point_level_1=0.9) == "fail"
    assert core.area_bar(point=0.9, lo=0.8, point_level_1=0.79) == "fail"
    assert core.area_bar(point=0.9, lo=0.8, point_level_1=float("nan")) == "void"


def test_powered_pair_rule_and_dilution():
    d = core.dilution(sd_injected=3.0, sd_host=4.0)
    assert d == pytest.approx(0.6)
    assert core.is_powered(coherence=0.75, dilution_value=0.6)
    assert not core.is_powered(coherence=0.74, dilution_value=0.6)
    assert core.expected_concordance(rho=0.0) == 0.0


def test_calibration_bar_needs_a_bin_of_100_and_checks_the_diagonal():
    p = np.full(100, 0.55)
    assert (
        core.calibration_bar(predicted=p, truth_export=np.r_[np.ones(55), np.zeros(45)]) == "pass"
    )
    assert (
        core.calibration_bar(predicted=p, truth_export=np.r_[np.ones(80), np.zeros(20)]) == "fail"
    )
    assert core.calibration_bar(predicted=p[:99], truth_export=np.ones(99)) == "void"


def test_verdict_rows_in_order():
    assert core.verdict_w2(x=inputs()) == ("supported", "row_4_supported")
    # Row 1 first, even when row 2 would also match.
    assert core.verdict_w2(x=inputs(correctness_ok=False, n_pairs_keeping_sensor=0))[0] == "invalid"
    assert core.verdict_w2(x=inputs(alignment_ok=False, timestamp_defect=True))[0] == "invalid"
    # A failed alignment without a time-stamp defect is not readable.
    assert core.verdict_w2(x=inputs(alignment_ok=False)) == ("inconclusive", "row_3_not_readable")
    assert core.verdict_w2(x=inputs(n_pairs_keeping_sensor=3)) == (
        "inconclusive",
        "row_2_void_before_run",
    )
    assert core.verdict_w2(x=inputs(n_powered_pairs_level_2=3))[1] == "row_2_void_before_run"
    assert core.verdict_w2(x=inputs(n_powered_hosts=1))[1] == "row_2_void_before_run"
    assert core.verdict_w2(x=inputs(pooled_se=0.0361))[1] == "row_3_not_readable"
    assert core.verdict_w2(x=inputs(n_import_stretches=29))[1] == "row_3_not_readable"
    assert core.verdict_w2(x=inputs(n_export_stretches=29))[1] == "row_3_not_readable"
    assert core.verdict_w2(x=inputs(level_1_readable=False))[1] == "row_3_not_readable"
    assert core.verdict_w2(x=with_outcome(false_export="void"))[1] == "row_3_not_readable"
    # Row 3 outranks the bars: an unreadable run is never `not supported`.
    assert (
        core.verdict_w2(x=inputs(pooled_se=0.05, outcomes={**inputs().outcomes, "area": "fail"}))[0]
        == "inconclusive"
    )


def test_partly_supported_needs_only_the_transfer_check_or_the_dead_band_to_fail():
    assert core.verdict_w2(x=with_outcome(transfer="fail")) == (
        "partly supported",
        "row_5_partly_supported",
    )
    assert core.verdict_w2(x=with_outcome(transfer="void"))[0] == "partly supported"
    assert core.verdict_w2(x=with_outcome(d15_dead_band="fail"))[0] == "partly supported"
    assert (
        core.verdict_w2(x=with_outcome(transfer="fail", d15_dead_band="fail"))[0]
        == "partly supported"
    )
    # A dead band that does not bind (report) or is void is neutral.
    assert core.verdict_w2(x=with_outcome(d15_dead_band="report"))[0] == "supported"
    assert core.verdict_w2(x=with_outcome(d15_dead_band="void"))[0] == "supported"


def test_not_supported_when_any_other_bar_fails():
    for bar in ("area", "placebo", "false_export", "recall"):
        assert core.verdict_w2(x=with_outcome(**{bar: "fail"})) == (
            "not supported",
            "row_6_not_supported",
        )
    assert core.verdict_w2(x=with_outcome(recall="void", calibration="void"))[0] == "supported"
    assert core.verdict_w2(x=with_outcome(calibration="fail"))[0] == "not supported"
    # Failed transfer plus a failed binding bar is not `partly supported`.
    assert core.verdict_w2(x=with_outcome(transfer="fail", area="fail"))[0] == "not supported"


def test_void_before_run_rule():
    assert core.void_before_run(
        n_pairs_keeping_sensor=3, n_powered_pairs_level_2=9, n_powered_hosts=3
    )
    assert core.void_before_run(
        n_pairs_keeping_sensor=12, n_powered_pairs_level_2=3, n_powered_hosts=3
    )
    assert core.void_before_run(
        n_pairs_keeping_sensor=12, n_powered_pairs_level_2=9, n_powered_hosts=1
    )
    assert not core.void_before_run(
        n_pairs_keeping_sensor=4, n_powered_pairs_level_2=4, n_powered_hosts=2
    )


def test_job_w3_verdict_rows():
    ok = {"a": "pass", "b": "pass"}
    assert (
        core.verdict_w3(control_outcomes=ok, masked=True)[0] == "controls pass, the read-outs stand"
    )
    assert core.verdict_w3(control_outcomes=ok, masked=False)[0] == "void"
    assert core.verdict_w3(control_outcomes={"a": "pass", "b": "void"}, masked=True)[0] == "void"
    fail = core.verdict_w3(control_outcomes={"a": "fail", "b": "void"}, masked=True)
    assert fail == ("a control fails, the hint is withdrawn at D12", "row_1")
    assert (
        "transfers not masked" in core.verdict_w3(control_outcomes={"a": "fail"}, masked=False)[0]
    )


def test_timing_gate_counts_the_worlds_it_will_score():
    pairs = [(h, f) for h in ("D10", "D14", "D16") for f in ("F1", "F2", "F3", "F4")][:12]
    specs = score.world_specs(qualifying=pairs)
    assert len(specs) == core.N_WORLDS
    assert len({s.name for s in specs}) == len(specs)
    projected = core.project_w2_seconds(
        seconds_per_world=10.0, seconds_per_draw=0.5, n_worlds=len(specs)
    )
    assert projected == pytest.approx(10.0 * 32 + 0.5 * 1000)
    assert not core.over_cap(projected_seconds=projected)
    assert core.over_cap(
        projected_seconds=core.project_w2_seconds(seconds_per_world=100.0, seconds_per_draw=1.0)
    )
    fewer = score.world_specs(qualifying=pairs[:4])
    assert len(fewer) == 4 * 2 + 8


def test_permutation_rule_for_future_engine_records():
    assert core.permutation_p(main_slope=-5.0, shifted_slopes=[0.0] * 39) == pytest.approx(1 / 40)
    shifts = {d: 0.1 * ((-1) ** d) for d in range(3, 41)}
    assert (
        core.engine_rule_outcome(
            detection_limit_main=0.54, main_slope=-1.0, ci=(-1.2, -0.8), shifted_by_day=shifts
        )
        == "void"
    )
    assert (
        core.engine_rule_outcome(
            detection_limit_main=0.4, main_slope=-1.0, ci=(-1.2, -0.8), shifted_by_day=shifts
        )
        == "seen"
    )
    assert (
        core.engine_rule_outcome(
            detection_limit_main=0.4, main_slope=-1.0, ci=(-0.9, -0.4), shifted_by_day=shifts
        )
        == "seen, size differs"
    )
    assert (
        core.engine_rule_outcome(
            detection_limit_main=0.4, main_slope=0.5, ci=(0.0, 1.0), shifted_by_day=shifts
        )
        == "not seen"
    )
    weekly = {d: (1.0 if d % 7 == 0 else 0.0) + 0.01 * d for d in range(3, 41)}
    assert (
        core.engine_rule_outcome(
            detection_limit_main=0.4, main_slope=-1.0, ci=(-1.2, -0.8), shifted_by_day=weekly
        )
        == "not interpretable"
    )


def test_d15_fit_readings():
    assert "same-sky" in core.fit_a_reading(lo=0.7, hi=1.4)
    assert "add-back" in core.fit_a_reading(lo=2.1, hi=3.0)
    assert core.fit_a_reading(lo=1.0, hi=2.5) == "unresolved"
    assert core.fit_b_supports_addback(slope=1.0, farm_coefficient=-0.5)
    assert not core.fit_b_supports_addback(slope=1.0, farm_coefficient=0.5)
    assert core.meter_blind_contradicts(area=0.49)
    assert not core.meter_blind_contradicts(area=0.5)


def test_predicted_standard_error_uses_the_1_25_factor():
    assert core.predicted_se(sd_response=2.0, sd_regressor=1.0, n_rows=100) == pytest.approx(
        1.25 * 2.0 / 10.0
    )
    assert core.detection_limit(se=0.1) == pytest.approx(0.28)
