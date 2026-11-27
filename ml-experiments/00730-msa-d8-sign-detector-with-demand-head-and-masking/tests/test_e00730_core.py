"""Known-answer tests of the constants, the limits files, the choices, and the verdict table."""

from pathlib import Path

import e00729_core as core729
import e00730_core as core
import numpy as np
import pytest


def reading(point: float, half_width: float) -> core729.Reading:
    return core729.Reading(point, point - half_width, point + half_width)


def test_constants() -> None:
    assert core.MIN_OUTPUT_SHARE == 0.99
    assert pytest.approx(1 - 0.05 / 7) == core.LEVEL
    assert core.N_CONTRASTS == 7
    assert core.LAMBDA == 0.1
    assert core.DETECTION_BAR == 0.75


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0123, 0.012), (0.0199, 0.019), (0.25, 0.25), (0.00349, 0.0034)]
)
def test_round_down_two_significant_figures(value: float, expected: float) -> None:
    assert core.round_down_2sf(value=value) == pytest.approx(expected)


def synthetic_truth(*, s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    x = np.where(rng.random(100_000) < s, -1.0, 5.0)
    r = np.where(x < 0, rng.uniform(0.5, 6.0, x.size), rng.uniform(0.5, 30.0, x.size))
    return x * 1.0, r


def test_limits_are_computed_from_truth_and_reading_alone() -> None:
    x, r = synthetic_truth()
    limits = core.compute_limits(x=x, r=r)
    assert limits.s == pytest.approx(np.mean(x < 0))
    assert limits.floor == core.round_down_2sf(value=0.02 * limits.s)
    expected_l = core.round_down_2sf(value=0.25 * limits.s1 / (1 - limits.s1))
    assert limits.harm_limit == expected_l
    assert limits.reliability_bar == pytest.approx(0.1 * limits.s)


def test_limit_range_guard() -> None:
    assert core.limit_in_range(limit=0.01)
    assert not core.limit_in_range(limit=0.001)
    assert not core.limit_in_range(limit=0.05)


def test_limits_file_is_write_once(tmp_path: Path) -> None:
    x, r = synthetic_truth()
    limits = core.compute_limits(x=x, r=r)
    path = tmp_path / "limits.json"
    core.write_limits_file(path=path, limits=limits)
    assert core.read_limits_file(path=path) == limits
    with pytest.raises(FileExistsError):
        core.write_limits_file(path=path, limits=limits)


def test_margin_file_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "margin.json"
    out = core.write_margin_file(
        path=path, floor=0.001, val_brier_s_by_seed=[0.02, 0.021, 0.019, 0.02, 0.022]
    )
    assert core.read_margin_file(path=path) == out["margin"]
    assert out["margin"] >= 0.001
    with pytest.raises(FileExistsError):
        core.write_margin_file(path=path, floor=0.001, val_brier_s_by_seed=[0.02] * 5)


def test_choose_architecture_prefers_the_lighter_one_within_the_tie() -> None:
    scores = {"bilstm": 0.0300, "unet": 0.0301, "transformer": 0.0299}
    weights = {"bilstm": 100, "unet": 50, "transformer": 900}
    assert core.choose_architecture(scores=scores, weights=weights, floor=0.01) == "unet"
    assert core.choose_architecture(scores=scores, weights=weights, floor=0.0001) == "transformer"


def test_choose_style_prefers_the_better_demand_head_in_the_tie() -> None:
    brier = {"0": 0.0300, "K": 0.0301, "W": 0.0305, "R": 0.0400}
    demand = {"0": 2.0, "K": 1.0, "W": 0.5, "R": 0.1}
    assert core.choose_style(val_brier=brier, whole_day_demand_mae=demand, floor=0.0006) == "K"
    assert core.choose_style(val_brier=brier, whole_day_demand_mae=demand, floor=0.00001) == "0"


def inputs(**changes: object) -> core.VerdictInputs:
    base = {
        "stopped": False, "invalid": False, "failed": False, "h1_pass": True, "h2_pass": True,
        "c1": reading(-0.01, 0.003), "margin": 0.005, "calibration_pass": True,
        "detection_pass": True,
    }  # fmt: skip
    return core.VerdictInputs(**{**base, **changes})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("changes", "row"),
    [
        ({"stopped": True, "invalid": True}, 0),
        ({"invalid": True, "failed": True}, 1),
        ({"failed": True}, 2),
        ({"h1_pass": False}, 3),
        ({"h2_pass": False}, 3),
        ({}, 4),
        ({"detection_pass": False}, 5),
        ({"calibration_pass": False}, 5),
        ({"c1": reading(-0.002, 0.001)}, 6),
        ({"c1": reading(-0.002, 0.01)}, 7),
        ({"c1": reading(0.01, 0.001)}, 8),
    ],
)
def test_verdict_table_first_match(changes: dict, row: int) -> None:
    assert core.verdict_row(v=inputs(**changes)) == row


def test_timing_gate_blocks_on_memory_and_on_the_stop_line() -> None:
    fast = {"bilstm": 0.2, "unet": 0.2, "transformer": 0.3, "gru_transformer": 0.3}
    ok = core.timing_gate_report(seconds_per_epoch=fast, peak_memory_gb=10.0)
    assert core.timing_gate_passes(report=ok, manager_approved=False)
    heavy = core.timing_gate_report(seconds_per_epoch=fast, peak_memory_gb=15.0)
    assert not core.timing_gate_passes(report=heavy, manager_approved=True)
    slow = dict.fromkeys(fast, 40.0)
    over = core.timing_gate_report(seconds_per_epoch=slow, peak_memory_gb=10.0)
    assert not core.timing_gate_passes(report=over, manager_approved=False)
    assert core.timing_gate_passes(report=over, manager_approved=True)
