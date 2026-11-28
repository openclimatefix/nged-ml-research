"""Tests of the guards of the stage runner: code hash, review marker, budget, done markers."""
# ruff: noqa: D103, E402

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _n in ("732", "734", "653", "660"):
    for _d in HERE.parent.glob(f"00{_n}-*"):
        sys.path.append(str(_d))
import e00736_common as common
import e00736_run as run


def test_code_hash_changes_with_a_module_and_ignores_tests(tmp_path: Path) -> None:
    (tmp_path / "e00736_a.py").write_text("x = 1\n")
    (tmp_path / "test_e00736_a.py").write_text("y = 1\n")
    first = run.code_hash(folder=tmp_path)
    (tmp_path / "test_e00736_a.py").write_text("y = 2\n")
    assert run.code_hash(folder=tmp_path) == first
    (tmp_path / "e00736_a.py").write_text("x = 2\n")
    assert run.code_hash(folder=tmp_path) != first


def test_require_code_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run, "CODE_SHA", tmp_path / "code.sha256")
    with pytest.raises(SystemExit):
        run.require_code_frozen()
    (tmp_path / "code.sha256").write_text("deadbeef  code\n")
    with pytest.raises(SystemExit):
        run.require_code_frozen()
    (tmp_path / "code.sha256").write_text(f"{run.code_hash()}  code\n")
    run.require_code_frozen()


def test_review_marker_and_gates_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    with pytest.raises(SystemExit):
        run.require_review(marker=run.MARKER_REVIEW)
    (tmp_path / run.MARKER_REVIEW).write_text("ok\n")
    with pytest.raises(SystemExit):  # a marker that holds no code hash does not pass
        run.require_review(marker=run.MARKER_REVIEW)
    (tmp_path / run.MARKER_REVIEW).write_text(f"{run.code_hash()}  reviewed code\n")
    run.require_review(marker=run.MARKER_REVIEW)
    with pytest.raises(SystemExit):
        run.require_gates()
    (tmp_path / "gates.json").write_text('{"G1": {"passed": true}, "G2": {"passed": true}}')
    with pytest.raises(SystemExit):
        run.require_gates()
    passed = {g: {"passed": True} for g in ("G1", "G2", "G3", "G4")}
    (tmp_path / "gates.json").write_text(json.dumps({**passed, "code_hash": "stale"}))
    with pytest.raises(SystemExit):  # gates written by different code do not pass
        run.require_gates()
    (tmp_path / "gates.json").write_text(json.dumps({**passed, "code_hash": run.code_hash()}))
    run.require_gates()


def log_line(*, stage: str, minutes: float) -> str:
    hh, mm = divmod(int(minutes), 60)
    return f"{stage},t,cpu,2026-11-28T00:00:00,2026-11-28T{hh:02d}:{mm:02d}:00\n"


def test_budget_counts_spent_and_projected_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    assert run.STOP_CPU_HOURS == 8.0
    run.require_budget()
    (tmp_path / "time_log.csv").write_text(log_line(stage="decode", minutes=30))
    assert run.spent_cpu_hours() == pytest.approx(0.5)
    assert run.projected_cpu_hours() == pytest.approx(0.5 * run.N_TASKS)
    run.require_budget()
    # 16 tasks at 40 minutes project over 8 hours
    (tmp_path / "time_log.csv").write_text(log_line(stage="decode", minutes=40))
    with pytest.raises(SystemExit):
        run.require_budget()
    (tmp_path / run.BUDGET_MARKER).write_text("ok\n")
    run.require_budget()
    (tmp_path / run.BUDGET_MARKER).unlink()
    (tmp_path / "time_log.csv").write_text(
        "".join(log_line(stage="stage0", minutes=59) for _ in range(9))
    )
    with pytest.raises(SystemExit):
        run.require_budget()


def test_run_task_writes_done_marker_with_peak_memory_and_skips_a_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    calls: list[int] = []
    assert run.run_task(name="a", fn=lambda: calls.append(1))
    text = (tmp_path / "done" / "a.ok").read_text()
    assert "peak_rss_gb=" in text
    assert f"code_hash={run.code_hash()} " in text
    assert run.run_task(name="a", fn=lambda: calls.append(1))
    assert calls == [1]


def test_a_done_marker_of_other_code_does_not_skip_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / "a.ok").write_text(
        "code_hash=deadbeef 2026-11-28T00:00:00Z peak_rss_gb=1\n"
    )
    calls: list[int] = []
    assert run.run_task(name="a", fn=lambda: calls.append(1))
    assert calls == [1]
    assert f"code_hash={run.code_hash()} " in (tmp_path / "done" / "a.ok").read_text()


def test_the_code_hash_covers_the_launch_scripts(tmp_path: Path) -> None:
    (tmp_path / "e00736_a.py").write_text("x = 1\n")
    first = run.code_hash(folder=tmp_path)
    (tmp_path / "run.sh").write_text("echo 1\n")
    second = run.code_hash(folder=tmp_path)
    assert second != first
    (tmp_path / "run_workers.sh").write_text("echo 2\n")
    assert run.code_hash(folder=tmp_path) != second


def test_freeze_code_needs_a_review_marker_of_the_current_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "CODE_SHA", tmp_path / "code.sha256")
    freeze = run.parse_args(argv=["freeze-code"])
    with pytest.raises(SystemExit):
        run.dispatch(args=freeze)
    (tmp_path / run.MARKER_REVIEW).write_text("deadbeef  reviewed code\n")
    with pytest.raises(SystemExit):
        run.dispatch(args=freeze)
    assert not (tmp_path / "code.sha256").exists()
    run.dispatch(args=run.parse_args(argv=["review-mark"]))
    run.dispatch(args=freeze)
    assert (tmp_path / "code.sha256").read_text().split()[0] == run.code_hash()
    run.require_code_frozen()


def test_world_and_later_stages_need_the_review_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    for stage in ("world", "decode", "score"):
        with pytest.raises(SystemExit):
            run.dispatch(args=run.parse_args(argv=[stage]))


def test_a_failing_stage_writes_a_record_and_exits_with_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)
    for var in run.BLAS_VARIABLES:
        monkeypatch.setenv(var, "1")

    def boom(*, args: object) -> None:
        msg = "stage boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(run, "dispatch", boom)
    with pytest.raises(SystemExit) as exc:
        run.main(argv=["report"])
    assert exc.value.code == run.EXIT_FAILED == 3
    record = json.loads((tmp_path / "failed" / "stage_report_worker_0.json").read_text())
    assert record["message"] == "stage boom"
    assert "Traceback" in record["traceback"]


def test_blas_variables_must_equal_one(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in run.BLAS_VARIABLES:
        monkeypatch.setenv(var, "1")
    run.assert_blas_pinned()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(SystemExit):
        run.assert_blas_pinned()
    monkeypatch.delenv("OMP_NUM_THREADS")
    with pytest.raises(SystemExit):
        run.assert_blas_pinned()


def test_an_output_override_must_be_a_scratch_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E00736_OUT", raising=False)
    monkeypatch.delenv("E00736_SCRATCH", raising=False)
    assert common._output_folder().name == common.FOLDER
    monkeypatch.setenv("E00736_OUT", "/somewhere/else")
    with pytest.raises(RuntimeError):
        common._output_folder()
    monkeypatch.setenv("E00736_OUT", "/x/.claude/scratch/fix/out")
    assert common._output_folder() == Path("/x/.claude/scratch/fix/out")
    monkeypatch.setenv("E00736_OUT", "/somewhere/else")
    monkeypatch.setenv("E00736_SCRATCH", "1")
    assert common._output_folder() == Path("/somewhere/else")


def test_the_pinned_params_digest_and_the_head_digests_are_sha256_strings() -> None:
    import e00736_heads as heads

    assert len(common.PARAMS_734_SHA256) == 64
    assert len(heads.STORED_DIGESTS) == 6
    assert all(len(d) == 64 for d in heads.STORED_DIGESTS.values())


def test_a_failed_task_leaves_a_record_and_the_shard_exits_with_code_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "OUT", tmp_path)

    def boom() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    assert not run.run_task(name="b", fn=boom)
    record = (tmp_path / "failed" / "b.json").read_text()
    assert "boom" in record
    assert "Traceback" in record
    with pytest.raises(SystemExit) as exc:
        run.run_shard(names_and_kwargs=[("c", {})], fn=boom)
    assert exc.value.code == run.EXIT_FAILED == 3


def test_sharding_splits_tasks_without_overlap() -> None:
    items = list(range(16))
    parts = [run.shard(items=items, worker=i, of=3) for i in range(3)]
    assert sorted(x for p in parts for x in p) == items


def test_every_attribute_taken_from_an_imported_module_exists() -> None:
    import e00736_gates as gates

    checked = 0
    importlib.import_module("e00732_paths")  # puts experiments 729 and 730 on sys.path
    for module, names in gates.HASATTR.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            pytest.skip(
                f"{module} needs the experiments' private folders or a package not installed"
            )
        for name in names:
            assert hasattr(mod, name), f"{module}.{name}"
            checked += 1
    for (module, cls), names in gates.HASATTR_CLASS.items():
        try:
            klass = getattr(importlib.import_module(module), cls)
        except ImportError:
            pytest.skip(f"{module} needs the experiments' private folders")
        for name in names:
            assert hasattr(klass, name), f"{module}.{cls}.{name}"
    assert checked > 10
