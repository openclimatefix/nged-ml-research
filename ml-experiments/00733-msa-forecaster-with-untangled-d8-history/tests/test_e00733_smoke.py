# ruff: noqa: PT018
"""Tests of `e00733_smoke`: every arm gets a fit, a prediction, and a read; every later stage runs
on validation rows; failures are recorded."""

import json
from pathlib import Path

import e00733_core as core
import e00733_smoke as sm
import e00733_stages as st
import pytest


def _steps(tmp: Path, *, fail: str | None = None, seen: list | None = None) -> sm.SmokeSteps:
    log: list = [] if seen is None else seen

    def fit_real(arm: str, fold: str, seed: int) -> dict:
        log.append(("fit", arm, fold))
        if fail == f"fit {arm}":
            msg = "device error"
            raise RuntimeError(msg)
        return {"stages": [{"stored_pass": 1}, {"stored_pass": 1}]}

    def fit_synth(arm: str, seed: int) -> dict:
        log.append(("fit", arm, None))
        return {"stored_pass": 1}

    def predict(arm: str, fold: str | None, seed: int) -> Path:
        p = tmp / f"{arm}.parquet"
        p.write_text("x")
        return p

    later = {
        n: (lambda validation_only=False, n=n: log.append((n, validation_only)))
        for n in sm.LATER_STAGES
    }
    return sm.SmokeSteps(
        fit_real=fit_real,
        fit_synthetic=fit_synth,
        predict=predict,
        read_stored=lambda arm, p: 5,
        checks={"rerun one origin": lambda: "ok"},
        later=later,
        log=lambda _s: None,
    )


def test_smoke_runs_every_arm_and_every_later_stage_on_validation_rows(tmp_path: Path) -> None:
    """Seven arms are fitted for one pass, predicted, and read; each later stage is called with
    validation_only; the record passes."""
    root, scratch = tmp_path / "data", tmp_path / "scratch"
    root.mkdir()
    seen: list = []
    rec = sm.run_smoke(root=root, scratch=scratch, steps=_steps(scratch, seen=seen), device="cpu")
    assert rec["pass"] and set(sm.SMOKE_ARMS) == {"F0r", "FU", "FU-S", "FS", "Y0", "YU", "YT"}
    assert [x[1] for x in seen if x[0] == "fit"] == list(sm.SMOKE_ARMS)
    assert [x for x in seen if x[0] in sm.LATER_STAGES] == [(n, True) for n in sm.LATER_STAGES]
    assert json.loads((root / st.SMOKE_CPU_FILE).read_text())["pass"] and not st.marker_exists(
        root=root
    )
    assert core.ARMS["FU"].folds[0] in {x[2] for x in seen if x[0] == "fit" and x[1] == "FU"}


def test_a_failed_step_is_recorded_with_its_traceback_and_the_run_goes_on(tmp_path: Path) -> None:
    """A fit that raises fails the record, skips that arm's prediction, and the other arms still
    run."""
    root, scratch = tmp_path / "data", tmp_path / "scratch"
    root.mkdir()
    rec = sm.run_smoke(
        root=root, scratch=scratch, steps=_steps(scratch, fail="fit FU"), device="cpu"
    )
    assert (
        not rec["pass"]
        and "device error" in rec["steps"]["fit FU"]["detail"]
        and "Traceback" in rec["steps"]["fit FU"]["traceback"]
    )
    assert "predict FU" not in rec["steps"] and rec["steps"]["read YU"]["ok"]
    assert not st.smoke_passed(root=root)


def test_scratch_inside_the_data_folder_or_a_taken_test_score_is_refused(tmp_path: Path) -> None:
    """The scratch folder may not lie in the data folder, and no smoke test runs after the first
    test score."""
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(sm.SmokeError):
        sm.run_smoke(root=root, scratch=root / "scratch", steps=_steps(tmp_path), device="cpu")
    (root / "scores").mkdir()
    (root / "scores" / st.TEST_MARKER).touch()
    with pytest.raises(sm.SmokeError):
        sm.run_smoke(root=root, scratch=tmp_path / "s", steps=_steps(tmp_path), device="cpu")


def test_a_missing_later_stage_fails_the_smoke(tmp_path: Path) -> None:
    """A later stage with no function is a failure, so no stage escapes the smoke."""
    root = tmp_path / "data"
    root.mkdir()
    steps = _steps(tmp_path)
    steps.later = {k: v for k, v in steps.later.items() if k != "attention"}
    assert not sm.run_smoke(root=root, scratch=tmp_path / "s", steps=steps, device="cpu")["pass"]


def test_a_cpu_smoke_never_opens_the_chain(tmp_path: Path) -> None:
    """A passing CPU smoke test writes `smoke_cpu.json` only, so `smoke_passed` stays false."""
    root = tmp_path / "data"
    root.mkdir()
    assert sm.run_smoke(root=root, scratch=tmp_path / "s", steps=_steps(tmp_path), device="cpu")[
        "pass"
    ]
    assert (root / st.SMOKE_CPU_FILE).exists() and not (root / st.SMOKE_FILE).exists()
    assert not st.smoke_passed(root=root)
    st.write_json(  # a CPU record copied into the chain's file is refused as well
        path=root / st.SMOKE_FILE,
        payload={"pass": True, "code_version": st.CODE_VERSION, "device": "cpu"},
    )
    assert not st.smoke_passed(root=root)


def test_a_cuda_smoke_opens_the_chain_only_with_a_gpu_name_and_a_peak(tmp_path: Path) -> None:
    """The GPU record names the GPU and its peak memory; a record without either does not pass."""
    root = tmp_path / "data"
    root.mkdir()
    steps = _steps(tmp_path)
    steps.gpu_info = lambda: {"gpu_name": "stub GPU", "peak_gpu_gb": 3.1}
    rec = sm.run_smoke(root=root, scratch=tmp_path / "s", steps=steps, device="cuda")
    assert rec["gpu_name"] == "stub GPU" and rec["peak_gpu_gb"] == 3.1
    assert st.smoke_passed(root=root) and not (root / st.SMOKE_CPU_FILE).exists()
    for bad in ({"gpu_name": "", "peak_gpu_gb": 3.1}, {"gpu_name": "stub GPU", "peak_gpu_gb": 0.0}):
        steps.gpu_info = lambda bad=bad: bad
        sm.run_smoke(root=root, scratch=tmp_path / "s", steps=steps, device="cuda")
        assert not st.smoke_passed(root=root)
