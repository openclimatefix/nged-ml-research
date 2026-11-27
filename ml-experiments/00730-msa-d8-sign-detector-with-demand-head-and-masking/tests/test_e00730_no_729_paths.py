"""The bank stage runs on synthetic arrays and never asks experiment 729 for its data folder."""

import contextlib
from pathlib import Path

import e00729_core as core729
import e00729_paths as paths729
import e00730_banks as banks
import e00730_core as core
import numpy as np
import pytest
from world730 import make_calendar


@pytest.fixture
def no_729_root(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Path:
        msg = "experiment 730 asked experiment 729 for its data folder"
        raise AssertionError(msg)

    monkeypatch.setattr(paths729, "data_root", refuse)
    monkeypatch.setattr(core729, "SEGMENT_DRAWS_TRAIN", 3)
    monkeypatch.setattr(core729, "SEGMENT_DRAWS_EVAL", 3)


def inputs() -> banks.BankInputs:
    cal = make_calendar(n_days=62)
    keys = np.array([str(d)[:7] for d in cal.dates])
    months = {"training": ["2022-03"], "validation": ["2022-04"], "test": ["2022-04"]}
    use = {
        "training": {"2022-03": keys == "2022-03"},
        "validation": {"2022-04": keys == "2022-04"},
        "test": {"2022-04": keys == "2022-04"},
    }
    return banks.BankInputs(cal=cal, months=months, use=use)


def test_the_bank_stage_never_reads_729s_folder(tmp_path: Path, no_729_root: None) -> None:
    root = tmp_path / "exp"
    root.mkdir()
    lines: list[str] = []
    with contextlib.suppress(banks.StoppedError):
        banks.stage_banks(root=root, inputs=inputs(), train_seeds=[0], log=lines.append)
    assert (root / "banks").exists()
    assert not (root / "limits.json").exists() or core.read_limits_file(path=root / "limits.json")
    assert not any("2022" in line for line in lines)


def test_a_stopped_bank_build_leaves_no_run_marker(tmp_path: Path, no_729_root: None) -> None:
    root = tmp_path / "exp"
    root.mkdir()
    with pytest.raises(banks.StoppedError):
        banks.stage_banks(root=root, inputs=inputs(), train_seeds=[0], log=lambda _: None)
    assert not (root / banks.RUN_MARKER).exists()
    with pytest.raises(RuntimeError, match="marker"):
        banks.guard_run(root=root, stage="gate")
