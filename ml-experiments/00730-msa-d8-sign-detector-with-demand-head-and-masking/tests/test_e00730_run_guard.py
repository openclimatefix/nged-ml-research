"""Tests of the guard that keeps a stage from running on another run's outputs or before a gate."""

import json
from pathlib import Path

import e00730_banks as banks
import e00730_core as core
import pytest

UNGATED = ("gate",)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    out = tmp_path / "exp"
    out.mkdir()
    return out


def write_marker(root: Path, **changes: object) -> None:
    (root / banks.RUN_MARKER).write_text(json.dumps({**banks.run_marker_payload(), **changes}))


def write_gate(root: Path, *, passed: bool) -> None:
    (root / "scores").mkdir(exist_ok=True)
    record = {"pass": passed, "marker": banks.run_marker_payload()}
    (root / "scores" / "gate.json").write_text(json.dumps(record))


def write_gallery(root: Path, *, passed: bool) -> None:
    folder = root / "PRIVATE" / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    panels = dict.fromkeys(core.PANEL_REQUIREMENTS, passed)
    record = {"panels": panels, "banks_digest": banks.banks_digest_id(root=root)}
    (folder / "panel_check.json").write_text(json.dumps(record))


def test_the_marker_names_this_experiment_and_the_required_run_id() -> None:
    payload = banks.run_marker_payload()
    assert payload["experiment"] == "00730"
    assert payload["run"] == core.REQUIRED_RUN_ID == "2"


def test_banks_may_start_in_an_empty_folder(root: Path) -> None:
    banks.guard_run(root=root, stage="banks")


def test_banks_refuse_over_an_earlier_run_without_a_marker(root: Path) -> None:
    (root / "banks").mkdir()
    (root / "banks" / "summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="superseded_run_1"):
        banks.guard_run(root=root, stage="banks")


def test_a_stage_without_a_marker_refuses(root: Path) -> None:
    with pytest.raises(RuntimeError, match="no run marker"):
        banks.guard_run(root=root, stage="gate")


def test_a_marker_of_another_run_refuses(root: Path) -> None:
    write_marker(root, run="1")
    with pytest.raises(RuntimeError, match="no run marker"):
        banks.guard_run(root=root, stage="gate")


def test_the_gate_stage_needs_only_the_marker(root: Path) -> None:
    write_marker(root)
    banks.guard_run(root=root, stage="gate")


@pytest.mark.parametrize("stage", sorted(set(banks.GATED_STAGES)))
def test_gated_stages_refuse_before_the_gate_passes(root: Path, stage: str) -> None:
    write_marker(root)
    with pytest.raises(RuntimeError, match="gate"):
        banks.guard_run(root=root, stage=stage)
    write_gate(root, passed=False)
    with pytest.raises(RuntimeError, match="gate"):
        banks.guard_run(root=root, stage=stage)


@pytest.mark.parametrize("stage", sorted(set(banks.GALLERY_GATED_STAGES)))
def test_fit_stages_refuse_before_the_gallery_check(root: Path, stage: str) -> None:
    write_marker(root)
    write_gate(root, passed=True)
    with pytest.raises(RuntimeError, match="panel_check"):
        banks.guard_run(root=root, stage=stage)
    write_gallery(root, passed=False)
    with pytest.raises(RuntimeError, match="panel_check"):
        banks.guard_run(root=root, stage=stage)
    write_gallery(root, passed=True)
    banks.guard_run(root=root, stage=stage)


def test_the_gate_record_must_match_the_banks_in_use(root: Path) -> None:
    write_marker(root)
    (root / "scores").mkdir()
    (root / "scores" / "gate.json").write_text(json.dumps({"pass": True, "marker": {"x": 1}}))
    with pytest.raises(RuntimeError, match="gate"):
        banks.guard_run(root=root, stage="score")


def test_a_changed_marker_after_a_fit_refuses_a_rebuild(root: Path) -> None:
    write_marker(root, run="1")
    (root / "fits").mkdir()
    (root / "fits" / "x.json").write_text("{}")
    with pytest.raises(RuntimeError, match="superseded_run_N"):
        banks.guard_run(root=root, stage="banks")


def test_the_gallery_check_lists_every_required_panel(root: Path) -> None:
    folder = root / "PRIVATE" / "gallery"
    folder.mkdir(parents=True)
    panels = dict.fromkeys(list(core.PANEL_REQUIREMENTS)[:-1], True)
    (folder / "panel_check.json").write_text(json.dumps({"panels": panels}))
    assert not banks.gallery_passed(root=root)
