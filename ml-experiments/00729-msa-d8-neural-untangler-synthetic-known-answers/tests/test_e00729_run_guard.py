"""Tests of the guard that keeps a stage from reading the outputs of a superseded run."""

import json
from pathlib import Path

import e00729_core as core
import e00729_paths as paths
import e00729_stages as stages
import pytest


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the data folder at an empty temporary folder and return the experiment's root."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    out = paths.data_root()
    out.mkdir(parents=True)
    return out


def write_marker(root: Path, **changes: object) -> None:
    (root / stages.RUN_MARKER).write_text(json.dumps({**stages.run_marker_payload(), **changes}))


def test_banks_may_start_in_an_empty_folder(root: Path) -> None:
    stages.guard_run(stage="banks")


def test_banks_refuse_to_start_over_an_earlier_run_without_a_marker(root: Path) -> None:
    (root / "banks").mkdir()
    (root / "banks" / "test.pkl").write_text("x")
    (root / "margin.json").write_text("{}")
    with pytest.raises(RuntimeError, match="superseded_run_1") as error:
        stages.guard_run(stage="banks")
    assert "banks" in str(error.value)
    assert "margin.json" in str(error.value)


UNGATED = ("gate", "events", "report")


def write_gate(root: Path, *, passed: bool, n: int = 1, marker: dict | None = None) -> None:
    scores = root / "scores"
    scores.mkdir(exist_ok=True)
    record = {"pass": passed, "marker": marker or stages.run_marker_payload()}
    for k in range(1, n + 1):
        (scores / f"gate_attempt{k}.json").write_text(json.dumps(record))
    (scores / "gate.json").write_text(json.dumps(record))


def test_every_other_stage_needs_a_matching_marker(root: Path) -> None:
    for stage in (*UNGATED, *stages.GATED_STAGES):
        with pytest.raises(RuntimeError, match="run marker"):
            stages.guard_run(stage=stage)
    write_marker(root)
    for stage in UNGATED:
        stages.guard_run(stage=stage)


def test_the_stages_from_timing_on_need_a_passed_gate_on_the_current_banks(root: Path) -> None:
    assert {"timing", "screen", "fit", "ladder", "score", "series", "components"} <= set(
        stages.GATED_STAGES
    )
    write_marker(root)
    for stage in stages.GATED_STAGES:
        with pytest.raises(RuntimeError, match="pass"):
            stages.guard_run(stage=stage)
    write_gate(root, passed=False)
    with pytest.raises(RuntimeError, match="pass"):
        stages.guard_run(stage="timing")
    write_gate(root, passed=True, marker={**stages.run_marker_payload(), "ranges": "0" * 16})
    with pytest.raises(RuntimeError, match="pass"):
        stages.guard_run(stage="timing")
    write_gate(root, passed=True)
    for stage in stages.GATED_STAGES:
        stages.guard_run(stage=stage)


def test_a_marker_of_another_run_sky_rule_or_ranges_is_refused(root: Path) -> None:
    for change in ({"run": "1"}, {"sky_share": 1.0}, {"ranges": "0" * 16}):
        write_marker(root, **change)
        with pytest.raises(RuntimeError, match="run marker"):
            stages.guard_run(stage="events")


def test_banks_may_rebuild_only_after_one_failed_attempt_and_only_for_a_ranges_change(
    root: Path,
) -> None:
    old = {**stages.run_marker_payload(), "ranges": "0" * 16}
    write_marker(root, ranges="0" * 16)
    with pytest.raises(RuntimeError, match="0 gate attempts"):
        stages.guard_run(stage="banks")
    write_gate(root, passed=False, marker=old)
    stages.guard_run(stage="banks")
    write_gate(root, passed=True, marker=old)
    with pytest.raises(RuntimeError, match="did not fail"):
        stages.guard_run(stage="banks")
    write_gate(root, passed=False, n=2, marker=old)
    with pytest.raises(RuntimeError, match="2 gate attempts"):
        stages.guard_run(stage="banks")


def test_banks_refuse_a_rebuild_when_the_attempt_saw_other_banks_or_the_rule_changed(
    root: Path,
) -> None:
    write_marker(root, ranges="0" * 16)
    write_gate(root, passed=False)  # the attempt carries the marker of the current code
    with pytest.raises(RuntimeError, match="did not fail on the banks now on disk"):
        stages.guard_run(stage="banks")
    for change in ({"run": "1"}, {"sky_share": 1.0}):
        old = {**stages.run_marker_payload(), "ranges": "0" * 16, **change}
        write_marker(root, ranges="0" * 16, **change)
        write_gate(root, passed=False, marker=old)
        with pytest.raises(RuntimeError, match="only the ranges may change"):
            stages.guard_run(stage="banks")


def test_banks_refuse_a_rebuild_after_a_fit_exists(root: Path) -> None:
    old = {**stages.run_marker_payload(), "ranges": "0" * 16}
    write_marker(root, ranges="0" * 16)
    write_gate(root, passed=False, marker=old)
    (root / "fits").mkdir()
    (root / "fits" / "A_bilstm_s0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="superseded_run_N"):
        stages.guard_run(stage="banks")


def test_a_crash_during_banks_gets_a_message_about_an_unfinished_build(root: Path) -> None:
    (root / "banks").mkdir()
    (root / "banks" / "test.pkl").write_text("x")
    with pytest.raises(RuntimeError, match="unfinished build"):
        stages.guard_run(stage="banks")


def test_figures_and_the_private_figure_folders_are_superseded_but_splits_md_is_not(
    root: Path,
) -> None:
    (root / "figures").mkdir()
    (root / "figures" / "a.svg").write_text("x")
    private = root / "PRIVATE"
    (private / "components").mkdir(parents=True)
    (private / "components" / "b.svg").write_text("x")
    (private / "splits.md").write_text("x")
    assert stages.existing_outputs(root=root) == ["figures", "PRIVATE/components"]
    with pytest.raises(RuntimeError, match="figures"):
        stages.guard_run(stage="banks")


def test_the_marker_changes_when_a_range_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    before = stages.run_marker_payload()
    monkeypatch.setitem(core.RANGES, "solar_capacity_points", (60.0, 400.0))
    assert stages.run_marker_payload()["ranges"] != before["ranges"]
    assert before["run"] == core.RUN_ID


def test_the_gate_refuses_a_third_run_and_a_run_after_a_pass(root: Path) -> None:
    scores = root / "scores"
    scores.mkdir()
    (scores / "gate_attempt1.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(RuntimeError, match="already passed"):
        stages.stage_gate(ctx=None, log=print)  # ty: ignore[invalid-argument-type]
    (scores / "gate_attempt1.json").write_text(json.dumps({"pass": False}))
    (scores / "gate_attempt2.json").write_text(json.dumps({"pass": False}))
    with pytest.raises(RuntimeError, match="used its 2 attempts"):
        stages.stage_gate(ctx=None, log=print)  # ty: ignore[invalid-argument-type]
