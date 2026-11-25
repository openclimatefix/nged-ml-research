# ruff: noqa: E501
"""Tests of experiment 725's check-hash rule and phase gates on temporary files (no data, no torch)."""

import json
from pathlib import Path

import e00725_common as cm
import e00725_phases as ph
import pytest


def test_check_result_carries_the_current_hash_and_a_stale_one_is_rejected(tmp_path: Path) -> None:
    """A check result written by other check code is not current, and the runner reruns that job."""
    path = tmp_path / "kl8.json"
    assert not cm.check_output_current(path=path)
    cm.write_check_json(obj={"pass": True}, path=path)
    assert json.loads(path.read_text())["checks_hash"] == cm.checks_hash()
    assert cm.check_output_current(path=path)
    job = ph.Job("check", "e00725_checks.py", ("kl8",), path, gpu=False)
    assert job.done()
    path.write_text(json.dumps({"pass": True, "checks_hash": "stale"}))
    assert not cm.check_output_current(path=path)
    assert not job.done()
    path.write_text(json.dumps({"pass": True}))
    assert not job.done()


def test_non_check_job_is_done_when_its_output_exists(tmp_path: Path) -> None:
    """A fit or cache job does not carry a checks hash."""
    out = tmp_path / "meta.json"
    out.write_text("{}")
    assert ph.Job("cpu", "e00725_sources.py", ("train",), out, gpu=False).done()


def test_phase_one_gate_needs_a_projection_at_or_under_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 1 refuses a missing projection and a projection above 12 GPU-process-hours."""
    monkeypatch.setattr(cm, "ROOT", tmp_path)
    (tmp_path / "decisions").mkdir()
    (tmp_path / "decisions" / "repro.json").write_text(json.dumps({"outcome": "pass"}))
    projection = tmp_path / "projection.json"
    monkeypatch.setattr(ph, "projection_file", lambda: projection)
    with pytest.raises(AssertionError, match="project"):
        ph.gate_for(phase=1)
    projection.write_text(json.dumps({"total": 12.5, "cut_arms": []}))
    with pytest.raises(AssertionError, match="ceiling"):
        ph.gate_for(phase=1)
    projection.write_text(json.dumps({"total": 11.7, "cut_arms": []}))
    ph.gate_for(phase=1)


def test_phase_four_gate_needs_the_t3_known_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 4 refuses until every fold check records a passing T3 known-answer check."""
    monkeypatch.setattr(ph, "ML_EXPERIMENTS_DATA_DIR", tmp_path)
    folder = cm.work(data_dir=tmp_path) / "checks"
    folder.mkdir(parents=True)
    for month in cm.ALL_FOLDS:
        for phase in ("train", "predict"):
            (folder / f"{month}_{phase}.json").write_text(
                json.dumps({"T3_known_answer": {"pass": True}})
            )
    ph.gate_for(phase=4)
    (folder / f"{cm.ALL_FOLDS[0]}_train.json").write_text(
        json.dumps({"T3_known_answer": {"pass": False}})
    )
    with pytest.raises(AssertionError, match="T3"):
        ph.gate_for(phase=4)
