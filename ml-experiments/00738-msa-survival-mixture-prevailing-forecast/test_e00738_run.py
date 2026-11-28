"""Tests of the run guards, the phases, the job lists, the report, and the attribute check."""
# ruff: noqa: D103, E402, PT018, ANN001

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

for _k in ("NGED_S3_BUCKET_URL", "NGED_S3_BUCKET_ACCESS_KEY", "NGED_S3_BUCKET_SECRET"):
    os.environ.setdefault(_k, "dummy")
# the tests never write to the experiment's data folder
os.environ.setdefault("E00738_OUT", str(Path(tempfile.gettempdir()) / "scratch" / "e738_tests"))
os.environ.setdefault("E00738_SCRATCH", "1")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _d in HERE.parent.glob("*"):
    if _d.is_dir() and _d.name[:5].isdigit() and int(_d.name[:5]) in {
        11, 87, 377, 378, 381, 391, 397, 400, 402, 404, 407, 410, 418, 422, 425, 428, 435, 437, 448,
        454, 470, 510, 511, 513, 515, 521, 545, 558, 559, 560, 562, 565, 568, 569, 572, 574, 578,
        581, 632, 667, 668, 709, 647, 650, 652, 654, 713, 726,
    }:  # fmt: skip
        sys.path.append(str(_d))

import json
import subprocess

import e00738_common as common
import e00738_phases as phases
import e00738_report as report
import e00738_run as run
import e00738_score as score

FOLDER = Path(__file__).parent
REAL_ATTRIBUTE_MODULES = (
    "e00738_common", "e00738_survival", "e00738_world", "e00738_score", "e00738_phases",
    "e00738_gate", "e00738_members", "e00738_report", "e00738_run", "e00738_predict",
    "e00738_panel", "e00738_fit", "e00738_gallery", "e00738_figures",
)  # fmt: skip


def test_scratch_guard_refuses_an_output_folder_that_is_not_scratch(monkeypatch) -> None:
    monkeypatch.setenv("E00738_OUT", "/data/ml-experiments/00738-real")
    monkeypatch.delenv("E00738_SCRATCH", raising=False)
    with pytest.raises(RuntimeError, match="not a scratch folder"):
        common._output_folder()
    monkeypatch.setenv("E00738_SCRATCH", "1")
    assert common._output_folder() == Path("/data/ml-experiments/00738-real")


def test_this_run_writes_only_to_a_scratch_folder() -> None:
    assert "scratch" in common.OUT.parts or os.environ.get("E00738_SCRATCH") == "1"


# --- Job lists ----------------------------------------------------------------------------------


def test_the_full_plan_has_120_fits_and_24_prediction_jobs() -> None:
    screen = run.fit_jobs(screen=True)
    rest = run.fit_jobs(screen=False)
    assert len(screen) == 20
    assert len(rest) == 100
    assert run.count_fits() == 120
    names = [n for n, _ in (*screen, *rest)]
    assert len(set(names)) == 120
    assert len(run.predict_jobs(screen=True)) == 4
    assert len(run.predict_jobs(screen=False)) == 20


def test_cut_list_drops_ld_and_keeps_two_seeds() -> None:
    assert len(run.fit_jobs(screen=False, drop_ld=True)) == 100 - 20
    two = run.fit_jobs(screen=False, seeds=(0, 1))
    assert {kw["seed"] for _, kw in two} == {0, 1}
    assert any(kw["arm"] == "LD" for _, kw in run.fit_jobs(screen=True, drop_ld=True)), (
        "the screen keeps LD"
    )


def test_screen_fits_are_four_folds_at_seed_zero() -> None:
    kws = [kw for _, kw in run.fit_jobs(screen=True)]
    assert {kw["month"] for kw in kws} == set(common.SCREEN_FOLDS)
    assert {kw["seed"] for kw in kws} == {0}
    assert {kw["arm"] for kw in kws} == set(common.FIT_ARMS)


def test_every_stage_has_a_dispatch_entry() -> None:
    ns = run.parse_args(argv=["gate"])
    assert ns.stage == "gate" and ns.worker == 0 and ns.of == 1
    with pytest.raises(SystemExit):
        run.parse_args(argv=["not-a-stage"])
    assert set(run.STAGES) >= {"phase0", "review-mark", "screen-fits", "gate", "report"}


def test_exit_codes_are_distinct() -> None:
    codes = {common.EXIT_FAILED, common.EXIT_CHAIN_FAILED, common.EXIT_GPU_BUSY, common.EXIT_REMEDY}
    assert len(codes) == 4
    assert common.EXIT_FAILED == 3 and common.EXIT_CHAIN_FAILED == 5 and common.EXIT_GPU_BUSY == 75


# --- Phases -------------------------------------------------------------------------------------


def test_code_hash_changes_with_the_code(tmp_path: Path) -> None:
    (tmp_path / "e00738_a.py").write_text("x = 1\n")
    (tmp_path / "run.sh").write_text("echo\n")
    h1 = phases.code_hash(folder=tmp_path)
    (tmp_path / "e00738_a.py").write_text("x = 2\n")
    assert phases.code_hash(folder=tmp_path) != h1
    (tmp_path / "test_e00738_a.py").write_text("t")  # a test is not part of the hash
    h2 = phases.code_hash(folder=tmp_path)
    (tmp_path / "test_e00738_a.py").write_text("changed")
    assert phases.code_hash(folder=tmp_path) == h2


def test_review_marker_is_tied_to_the_code_hash(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "e00738_a.py").write_text("x = 1\n")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit):
        phases.require_review(out=out, folder=code)
    h = phases.write_review_marker(out=out, folder=code)
    assert h == phases.code_hash(folder=code)
    phases.require_review(out=out, folder=code)
    (code / "e00738_a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit):
        phases.require_review(out=out, folder=code)


def test_a_stage_needs_the_marker_of_the_stage_before_it(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        phases.require_stage(stage="screen_fits", out=tmp_path)
    (tmp_path / phases.STAGE_MARKERS["smoke"]).write_text("x")
    (tmp_path / phases.MARKER_REVIEW).write_text("zz")
    with pytest.raises(SystemExit):  # the review marker holds another hash
        phases.require_stage(stage="screen_fits", out=tmp_path)
    phases.require_stage(stage="phase0", out=tmp_path)  # the first stage needs nothing


def test_every_stage_of_the_order_has_a_marker() -> None:
    assert set(phases.STAGE_ORDER) == set(phases.STAGE_MARKERS)


def test_run_task_writes_a_done_marker_with_hash_and_peak_memory(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    calls: list[int] = []
    assert phases.run_task(name="t1", fn=lambda **kw: calls.append(kw["x"]), kwargs={"x": 3})
    text = (tmp_path / "done" / "t1.ok").read_text()
    assert text.startswith(f"code_hash={phases.code_hash()} ")
    assert "peak_rss_gb=" in text
    assert phases.run_task(name="t1", fn=lambda **kw: calls.append(9), kwargs={"x": 4})
    assert calls == [3], "a finished task must not run again"
    assert (tmp_path / "time_log.csv").exists()


def test_run_task_failure_record_holds_message_and_traceback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)

    def boom(**kw: object) -> None:
        msg = "the fit blew up"
        raise ValueError(msg)

    assert not phases.run_task(name="bad", fn=boom, kwargs={})
    rec = json.loads((tmp_path / "failed" / "task_bad.json").read_text())
    assert rec["message"] == "the fit blew up"
    assert "Traceback" in rec["traceback"] and "ValueError" in rec["traceback"]
    assert not (tmp_path / "done" / "bad.ok").exists()
    assert (tmp_path / "time_log.csv").exists(), "a crashed attempt still writes its time"


def test_run_task_keyword_names_do_not_collide(monkeypatch, tmp_path) -> None:
    """A task may take keywords called `name`, `fn`, `device`, and `stage`."""
    monkeypatch.setattr(common, "OUT", tmp_path)
    seen: dict[str, object] = {}

    def task(*, name: str, fn: str, device: str, stage: str) -> None:
        seen.update(name=name, fn=fn, device=device, stage=stage)

    ok = phases.run_task(
        name="collide", fn=task, kwargs={"name": "n", "fn": "f", "device": "d", "stage": "s"},
        device="cpu", stage="fit",
    )  # fmt: skip
    assert ok
    assert seen == {"name": "n", "fn": "f", "device": "d", "stage": "s"}


def test_chain_exits_with_the_given_code_when_a_task_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)

    def boom(**kw: object) -> None:
        raise RuntimeError

    with pytest.raises(SystemExit) as e:
        phases.run_chain(jobs=[("c1", {})], fn=boom, device="cpu", stage="x", exit_code=5)
    assert e.value.code == 5


def test_require_finished_and_failure_records(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(common, "OUT", tmp_path)
    assert phases.run_task(name="ok1", fn=lambda **kw: None, kwargs={})
    phases.require_finished(names=["ok1"], out=tmp_path)
    with pytest.raises(RuntimeError, match="no done marker"):
        phases.require_finished(names=["ok1", "missing"], out=tmp_path)
    phases.write_failure(name="f", exc=ValueError("x"))
    with pytest.raises(RuntimeError, match="failure records"):
        phases.require_finished(names=["ok1"], out=tmp_path)


def test_budget_stops_at_the_hour_line_unless_extended(tmp_path) -> None:
    log = tmp_path / "time_log.csv"
    log.write_text("fit,t,gpu,2026-01-01T00:00:00,2026-01-01T08:00:00\n")
    assert phases.spent_hours(device="gpu", log=log) == pytest.approx(8.0)
    assert phases.spent_hours(device="cpu", log=log) == 0.0
    with pytest.raises(RuntimeError, match="stop"):
        phases.require_budget(device="gpu", out=tmp_path, log=log)
    (tmp_path / common.BUDGET_MARKER).write_text("ok")
    phases.require_budget(device="gpu", out=tmp_path, log=log)


def test_gpu_busy_detection_ignores_this_process() -> None:
    listing = "123\n456\n"
    assert phases.other_compute_processes(listing=listing, own_pids={123}) == [456]
    assert phases.other_compute_processes(listing="", own_pids=set()) == []


def test_shards_partition_the_items() -> None:
    items = list(range(10))
    got = sorted(x for w in range(3) for x in phases.shard(items=items, worker=w, of=3))
    assert got == items


def test_blas_guard(monkeypatch) -> None:
    for k in phases.BLAS_VARIABLES:
        monkeypatch.setenv(k, "1")
    phases.assert_blas_pinned()
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(SystemExit):
        phases.assert_blas_pinned()


def test_wrapper_scripts_name_no_absolute_home_path() -> None:
    for f in [*FOLDER.glob("*.py"), *FOLDER.glob("*.sh")]:
        assert "/home/jack" not in f.read_text() or f.name.startswith("test_"), f.name


def test_run_sh_refuses_without_go() -> None:
    r = subprocess.run(
        ["bash", str(FOLDER / "run.sh")], capture_output=True, text=True, check=False
    )
    assert r.returncode == 64
    assert "nothing was started" in r.stderr


# --- The report on synthetic tables ----------------------------------------------------------


def _synthetic_tables(*, months: tuple[str, ...], seeds: tuple[int, ...], gain: float) -> dict:
    """Return tables in which `LD`, `LM`, `LR`, and `LP` beat `M1` in force by `gain` points."""
    rng = np.random.default_rng(0)
    slices = [
        "all", "in_force", "far", "middle", "near", "ended_24h", "ended_1_7", "event_free",
        "onset_ahead", "in_force_clipped", "event_free_ended_8_15",
        *(f"in_force_{n}" for n, _, _ in common.AGE_BANDS),
        *(f"in_force_lead_{d}" for d in range(1, 10)),
    ]  # fmt: skip
    metrics = [
        "crps", "crps_scaled", "nmae", "cover90", "cover98", "follow_num", "follow_den", "departs",
    ]  # fmt: skip
    tables: dict = {}
    for m in months:
        for s in seeds:
            t: dict = {}
            for arm in common.ALL_ARMS:
                t[arm] = {"scale_factor": 1.2}
                for wld in common.WORLDS:
                    t[arm][wld] = {}
                    for sl in slices:
                        n = 100.0
                        body = {"n": np.full(16, n)}
                        for metric in metrics:
                            level = 2.0 + rng.normal(0, 0.01)
                            if (
                                arm != common.ARM_M1
                                and sl.startswith("in_force")
                                and metric.startswith("crps")
                            ):
                                level -= gain
                            body[metric] = np.full(16, level * n)
                        t[arm][wld][sl] = body
            tables[(m, s)] = t
    return tables


def test_compute_report_on_synthetic_tables_gives_a_verdict_and_renders() -> None:
    labels = [*common.GROUP_OF_PRIMARY["all14"], *common.GROUP_OF_PRIMARY["short_history_two"]]
    months = common.FOLDS[:6]
    seeds = (0, 1, 2)
    tables = _synthetic_tables(months=months, seeds=seeds, gain=0.5)
    rep = report.compute_report(
        tables=tables, labels=labels, s_star="LD", months=months, seeds=seeds, n_resamples=200
    )
    assert rep["verdict"] in {r[1] for r in score.ROWS} | {"partly supported (duration-dependent)"}
    assert rep["C1"]["point"] < 0
    text = report.render(report=rep)
    assert "C1" in text and "N1" in text
    assert isinstance(report.audit_triggers(report=rep), dict)


def test_scenario_table_keeps_the_four_scenarios_in_words() -> None:
    import e00738_protocol as protocol
    import polars as pl

    slices = ["all", "outside_events", "event_rows", "event_following_7_days", "winter"]
    scores = pl.DataFrame(
        {
            "group": ["g"] * 5,
            "slice": slices,
            "metric": ["crps"] * 5,
            "arm": ["M1"] * 5,
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    table = protocol.scenario_table(scores=scores)
    assert sorted(table["scenario"].to_list()) == sorted(
        ["all rows", "outside events", "inside events", "7 days after an event"]
    )
    assert "winter" not in table["slice"].to_list()
