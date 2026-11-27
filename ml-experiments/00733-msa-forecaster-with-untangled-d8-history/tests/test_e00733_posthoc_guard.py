# ruff: noqa: PT018, E501
"""Regression tests of the guards around the post hoc stages.

The main-run guard `guard_run` stays as strict as before: every stage that starts a fit refuses once
`scores/test_started` exists. The post hoc stages have a guard of their own, are legal only after the
first test score, and write under `<root>/posthoc/` alone.
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import e00733_cli as cli
import e00733_core as core
import e00733_fit as fit
import e00733_leakage as leak
import e00733_runtime as rt
import e00733_stages as st
import pytest
from test_e00733_stages import DIGEST, _ready

pytest.importorskip("torch")


def _args(stage: str, **kw: object) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "stage": stage, "interim": False, "measured": None, "device": "cpu", "data_root": None,
            "max_passes": None, "first_job_per_family": False, "skip_gates": False, **kw,
        }
    )  # fmt: skip


def _finished_main_run(root: Path) -> Path:
    """Make `root` look like a finished main run: every gate passed, a complete chain, a verdict."""
    root = root / "ml-experiments" / "00733-test"
    root.mkdir(parents=True)
    _ready(root, through="timing")
    (root / "world").mkdir()
    st.write_chain_status(root=root, state="complete")
    (root / "scores" / st.TEST_MARKER).touch()
    return root


def _listing(root: Path) -> dict[str, tuple[int, int]]:
    """Return every file of `root` outside `posthoc/` with its size and modification time."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "posthoc" not in p.relative_to(root).parts
    }


# --------------------------------------------------------------------------- the main guard


@pytest.mark.parametrize("stage", ["chain", "fit", "timing", "smoke_fit"])
def test_every_fit_stage_of_the_main_run_still_refuses_once_the_first_test_score_exists(
    tmp_path: Path, stage: str
) -> None:
    """`guard_run` raises for the four gated stages when `scores/test_started` exists."""
    _ready(tmp_path, through="timing")
    st.guard_run(root=tmp_path, stage="fit", banks_digest=DIGEST)
    (tmp_path / "scores" / st.TEST_MARKER).touch()
    with pytest.raises(st.GuardError, match="first test score"):
        st.guard_run(root=tmp_path, stage=stage, banks_digest=DIGEST)


@pytest.mark.parametrize("stage", ["chain", "fit", "timing"])
def test_the_command_line_refuses_the_main_fit_stages_after_the_first_test_score(
    tmp_path: Path, stage: str
) -> None:
    """The stage exits 2 through `run_stage`, and no post hoc folder appears."""
    _ready(tmp_path, through="timing")
    (tmp_path / "scores" / st.TEST_MARKER).touch()
    assert cli.run_stage(args=_args(stage), root=tmp_path, digest=DIGEST) == 2
    assert not (tmp_path / st.POSTHOC_DIR).exists()


def test_the_main_guard_has_no_escape_and_no_post_hoc_stage() -> None:
    """`guard_run` takes the arguments it always took, and the gated stages are unchanged."""
    assert tuple(inspect.signature(st.guard_run).parameters) == (
        "root", "stage", "banks_digest", "validation_only",
    )  # fmt: skip
    assert st.GATED_STAGES == ("smoke_fit", "timing", "fit", "chain")
    assert not set(st.POSTHOC_STAGES) & set(st.GATED_STAGES)
    assert not set(st.POSTHOC_STAGES) & set(st.FAMILIES)
    assert st.FAMILIES == ("F0r", "FU-screen", "FU", "FU-S", "Y0", "YU", "FS", "YT")


# --------------------------------------------------------------------------- the post hoc guard


def test_posthoc_passes_after_a_finished_main_run(tmp_path: Path) -> None:
    """With the verdict out and every gate passed, both post hoc stages may start."""
    root = _finished_main_run(tmp_path)
    st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)
    st.write_posthoc_status(root=root, state="complete")
    st.guard_posthoc(root=root, stage="posthoc_score", banks_digest=DIGEST)


def test_posthoc_refuses_before_the_first_test_score(tmp_path: Path) -> None:
    """A post hoc run is legal only after the verdict."""
    root = _finished_main_run(tmp_path)
    (root / "scores" / st.TEST_MARKER).unlink()
    with pytest.raises(st.GuardError, match="only after the verdict"):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)
    assert cli.run_stage(args=_args("posthoc"), root=root, digest=DIGEST) == 2
    assert not (root / st.POSTHOC_DIR).exists()


@pytest.mark.parametrize("state", [None, "running", "failed", "stopped"])
def test_posthoc_refuses_while_the_main_chain_is_not_complete(
    tmp_path: Path, state: str | None
) -> None:
    """The main chain must say complete."""
    root = _finished_main_run(tmp_path)
    if state is None:
        (root / st.CHAIN_STATUS).unlink()
    else:
        st.write_chain_status(root=root, state=state)
    with pytest.raises(st.GuardError, match="main chain complete"):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)


def test_posthoc_refuses_while_another_posthoc_run_is_running(tmp_path: Path) -> None:
    """`posthoc_status.json` that says running closes the stage; `posthoc_score` waits too."""
    root = _finished_main_run(tmp_path)
    st.write_posthoc_status(root=root, state="running")
    with pytest.raises(st.GuardError, match="is running"):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)
    with pytest.raises(st.GuardError, match="complete, failed, or stopped"):
        st.guard_posthoc(root=root, stage="posthoc_score", banks_digest=DIGEST)
    assert cli.run_stage(args=_args("posthoc"), root=root, digest=DIGEST) == 2
    assert st.posthoc_state(root=root) == "running"


def test_posthoc_score_needs_a_finished_posthoc_run(tmp_path: Path) -> None:
    """Without a status file the readings have nothing to read."""
    root = _finished_main_run(tmp_path)
    with pytest.raises(st.GuardError, match="complete, failed, or stopped"):
        st.guard_posthoc(root=root, stage="posthoc_score", banks_digest=DIGEST)


@pytest.mark.parametrize(
    "missing", [st.SMOKE_FILE, st.TIMING_FILE, st.PRE_FIT_GUARDS, st.REVIEW_GATE, st.PANEL_CHECK]
)
def test_posthoc_refuses_without_the_gates_of_the_main_run(tmp_path: Path, missing: str) -> None:
    """The pre-fit gates, the smoke test, and the timing gate must have passed."""
    root = _finished_main_run(tmp_path)
    (root / missing).unlink()
    with pytest.raises(st.GuardError):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)


def test_posthoc_refuses_after_a_correctness_failure_or_another_banks_digest(
    tmp_path: Path,
) -> None:
    """Verdict row 1 and a run marker of other banks close the stage."""
    root = _finished_main_run(tmp_path)
    with pytest.raises(st.GuardError, match="run marker"):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest="other")
    st.record_invalid(root=root, rule="x")
    with pytest.raises(st.GuardError, match="correctness"):
        st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)


def test_a_refusal_writes_no_file(tmp_path: Path) -> None:
    """A refused post hoc stage leaves the data folder as it was."""
    root = _finished_main_run(tmp_path)
    st.write_chain_status(root=root, state="running")
    before = _listing(root)
    assert cli.run_stage(args=_args("posthoc"), root=root, digest=DIGEST) == 2
    assert _listing(root) == before and not (root / st.POSTHOC_DIR).exists()


# --------------------------------------------------------------------------- what the stage writes


def _fake_process(calls: list[dict[str, Any]], *, fail: tuple[str, ...] = ()) -> Any:
    """Return a stand-in for `run_stage_in_new_process` that writes the files a process would."""

    def fake(**kw: Any) -> None:
        folder = kw["root"].parent
        calls.append({**kw, "marker_seen": (folder.parent / st.POSTHOC_STARTED).exists()})
        job = fit.job_name(arm=kw["arm"], fold=kw["fold"], seed=kw["seed"])
        if job in fail:
            msg = "the device ran out of memory"
            raise MemoryError(msg)
        if kw["stage"] == "fit":
            fit.write_json(
                path=fit.record_path(
                    root=kw["root"], arm=kw["arm"], fold=kw["fold"], seed=kw["seed"]
                ),
                payload={"seconds": 3.0, "peak_rss_gb": 2.0},
            )
        else:
            st.write_json(
                path=rt.predict_record_path(
                    root=kw["root"], arm=kw["arm"], fold=kw["fold"], seed=kw["seed"]
                ),
                payload={"seconds": 4.0, "peak_rss_gb": 1.5},
            )

    return fake


def _run(
    root: Path, monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], **kw: Any
) -> int:
    monkeypatch.setattr(rt, "run_stage_in_new_process", _fake_process(calls, **kw))
    monkeypatch.setattr(leak, "assert_limits_before_fits", lambda **_: None)
    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "posthoc", rt.stage_posthoc)
    args = _args(
        "posthoc", data_root=str(root), max_passes=1, first_job_per_family=True, limit_segments=3
    )
    return cli.run_stage(args=args, root=root, digest=DIGEST)


def test_posthoc_writes_nothing_outside_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny run on a scratch root adds files under `posthoc/` only and leaves the rest as it was."""
    root = _finished_main_run(tmp_path)
    before = _listing(root)
    calls: list[dict[str, Any]] = []
    assert _run(root, monkeypatch, calls) == 0
    assert _listing(root) == before
    posthoc = root / st.POSTHOC_DIR
    assert {p.name for p in posthoc.iterdir()} == {
        "posthoc_started.json", "posthoc_status.json", root.name,
    }  # fmt: skip
    assert {c["arm"] for c in calls} == {"YO", "FS", "YS"} and {c["stage"] for c in calls} == {
        "fit",
        "predict",
    }
    assert {str(c["root"]) for c in calls} == {str(st.posthoc_fit_root(root=root))}
    assert {c["data_root"] for c in calls} == {str(st.posthoc_fit_root(root=root))}
    assert all(c["marker_seen"] for c in calls), "the marker is written before the first fit"
    started = json.loads((root / st.POSTHOC_STARTED).read_text())
    assert started["post_hoc"] is True and started["set_after_the_first_test_score"] is True
    assert started["arms"] == ["YO", "FS", "YS"] and "maintainer" in started["requested_by"]
    assert json.loads((root / st.POSTHOC_STATUS).read_text())["state"] == "complete"
    for name in ("verdict.json", "readings.json", "failure_posthoc.json"):
        assert not (root / "scores" / name).exists()
    assert st.chain_state(root=root) == "complete" and not st.invalid_recorded(root=root)


def test_the_fit_root_is_its_own_folder_with_copies_of_the_gate_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fit processes read the post hoc root, which links the inputs and copies the gates."""
    root = _finished_main_run(tmp_path)
    assert _run(root, monkeypatch, []) == 0
    fit_root = st.posthoc_fit_root(root=root)
    assert fit_root.parent == root / st.POSTHOC_DIR and fit_root.name == root.name
    for name in ("limits.json", "channels.json", "channels", "world"):
        assert (fit_root / name).is_symlink()
    assert (fit_root / st.RUN_MARKER).read_text() == (root / st.RUN_MARKER).read_text()
    assert not st.marker_exists(root=fit_root)
    for name in st.POSTHOC_GATE_RECORDS:
        assert (fit_root / name).read_text() == (root / name).read_text()
    st.guard_run(root=fit_root, stage="fit", banks_digest=DIGEST)  # the fit process's own guard


def test_a_failed_posthoc_fit_is_report_only_and_the_run_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed job leaves a record, the other family still runs, the exit code is 5."""
    root = _finished_main_run(tmp_path)
    first = core.ARMS["FS"]
    job = fit.job_name(arm="FS", fold=first.folds[0], seed=first.seeds[0])
    calls: list[dict[str, Any]] = []
    assert _run(root, monkeypatch, calls, fail=(job,)) == 5
    status = json.loads((root / st.POSTHOC_STATUS).read_text())
    assert status["state"] == "failed" and status["failed"] == [job]
    assert {c["arm"] for c in calls} == {"YO", "FS", "YS"}
    assert (st.posthoc_fit_root(root=root) / "fits" / f"{job}.failed.json").exists()


@pytest.mark.parametrize(
    ("exc", "code", "state"),
    [(st.StoppedError("x"), 3, "stopped"), (st.CorrectnessFailureError("x"), 4, "failed")],
)
def test_a_stop_or_a_correctness_failure_ends_the_posthoc_run_without_touching_the_main_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception, code: int, state: str
) -> None:
    """Exit 3 and 4 map as in the chain, and `invalid.json` of the main run is never written."""
    root = _finished_main_run(tmp_path)
    before = _listing(root)

    def boom(**_: Any) -> None:
        raise exc

    monkeypatch.setattr(rt, "run_stage_in_new_process", boom)
    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "posthoc", rt.stage_posthoc)
    args = _args("posthoc", data_root=str(root), max_passes=1, first_job_per_family=True)
    assert cli.run_stage(args=args, root=root, digest=DIGEST) == code
    assert json.loads((root / st.POSTHOC_STATUS).read_text())["state"] == state
    assert _listing(root) == before and not st.invalid_recorded(root=root)


def test_an_exception_outside_the_run_leaves_its_record_under_posthoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure of the stage itself writes `posthoc/failure_posthoc.json`, never `scores/`."""
    root = _finished_main_run(tmp_path)

    def boom(_a: argparse.Namespace, _r: Path) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setitem(cli.STAGE_FUNCTIONS, "posthoc", boom)
    assert cli.run_stage(args=_args("posthoc"), root=root, digest=DIGEST) == 5
    assert (root / st.POSTHOC_DIR / "failure_posthoc.json").exists()
    assert not (root / "scores" / "failure_posthoc.json").exists()


def test_first_job_per_family_and_max_passes_are_for_a_scratch_root_only(tmp_path: Path) -> None:
    """The debug options of the chain keep their rule on the post hoc stage."""
    root = _finished_main_run(tmp_path)
    with pytest.raises(rt.RuntimeError733):
        rt.stage_posthoc(_args("posthoc", first_job_per_family=True), root)
    with pytest.raises(rt.RuntimeError733):
        rt.stage_posthoc(_args("posthoc", max_passes=1), root)


def test_the_posthoc_stages_are_in_the_cli_and_only_the_scoring_stage_is_cpu_only() -> None:
    """`posthoc` fits on the GPU like `chain`; `posthoc_score` reads files on the CPU."""
    assert {"posthoc", "posthoc_score"} <= set(cli.STAGES)
    assert "posthoc_score" in st.CPU_ONLY_STAGES and "posthoc" not in st.CPU_ONLY_STAGES
    assert cli.parser().parse_args(["posthoc"]).device is None


# --------------------------------------------------------------------------- the cost cap


def _cap_steps(
    root: Path, *, main: float, hours_of: dict[str, float], reference: float = 1.0
) -> tuple[st.PosthocSteps, list[tuple[str, str | None, int]], dict[str, float]]:
    """Return steps whose fits take `hours_of[arm]` hours each, with the jobs they started."""
    started: list[tuple[str, str | None, int]] = []
    measured: dict[str, float] = {}

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        started.append((arm, fold, seed))
        measured[fit.job_name(arm=arm, fold=fold, seed=seed)] = hours_of[arm]
        return {}

    steps = st.PosthocSteps(
        run_job=run_job, fit_root=st.posthoc_fit_root(root=root), log=lambda _m: None,
        main_hours=lambda: main,
        job_hours=lambda a, f, s: measured.get(fit.job_name(arm=a, fold=f, seed=s)),
        reference_hours=lambda _arm: reference,
    )  # fmt: skip
    return steps, started, measured


def test_the_cost_cap_stops_the_run_before_the_fit_that_would_pass_it(tmp_path: Path) -> None:
    """With 7.2 hours in the main chain and 1 hour per fit, the run stops with the last fits skipped."""
    root = _finished_main_run(tmp_path)
    steps, started, measured = _cap_steps(
        root, main=7.2, hours_of={"YO": 1.0, "FS": 1.0, "YS": 1.0}
    )
    status = st.run_posthoc(root=root, steps=steps)
    assert status["state"] == "stopped" and st.exit_code(status=status) == 3
    assert status["reason"] == "stopped at the cost cap; ask the manager"
    assert [a for a, _, _ in started] == ["YO"] * 3 + [
        "FS"
    ] * 1  # 7.2 + 4 = 11.2; one more passes 11.9
    assert len(measured) == 4 and status["hours"]["running_total"] == pytest.approx(11.2)
    skipped = [x["job"] for x in status["skipped"]]
    assert len(skipped) == 23 + 3 and skipped[-3:] == ["YS_s0", "YS_s1", "YS_s2"]
    assert all("cap of 11.9" in x["reason"] for x in status["skipped"])
    assert not set(skipped) & set(measured)


def test_the_cost_cap_skips_the_oracle_last_when_it_alone_passes_the_cap(tmp_path: Path) -> None:
    """Every fit not started is skipped, YS first in the order of dropping, YO last."""
    root = _finished_main_run(tmp_path)
    steps, started, _ = _cap_steps(root, main=11.5, hours_of={"YO": 1.0, "FS": 1.0, "YS": 1.0})
    status = st.run_posthoc(root=root, steps=steps)
    assert started == [] and status["state"] == "stopped"
    skipped = [x["job"] for x in status["skipped"]]
    assert skipped[:3] == ["YO_s0", "YO_s1", "YO_s2"] and skipped[-3:] == [
        "YS_s0",
        "YS_s1",
        "YS_s2",
    ]


def test_no_fit_starts_when_the_total_has_reached_the_stop_line(tmp_path: Path) -> None:
    """A total of 12.0 stops the run even when the expected hours of the next fit are zero."""
    root = _finished_main_run(tmp_path)
    steps, started, _ = _cap_steps(root, main=12.0, hours_of={}, reference=0.0)
    status = st.run_posthoc(root=root, steps=steps)
    assert started == [] and status["state"] == "stopped"


@pytest.mark.parametrize("main", [0.0, 6.0, 7.2, 9.0])
def test_a_fit_never_starts_past_the_stop_line_whatever_the_rates(
    tmp_path: Path, main: float
) -> None:
    """Over runs with several rates, the running total at the start of every fit stays below 12."""
    root = _finished_main_run(tmp_path / str(main))
    totals: list[float] = []
    steps, _, measured = _cap_steps(root, main=main, hours_of={"YO": 0.9, "FS": 0.4, "YS": 0.9})
    inner = steps.run_job

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        totals.append(main + sum(measured.values()))
        return inner(arm, fold, seed)

    steps.run_job = run_job
    status = st.run_posthoc(root=root, steps=steps)
    assert totals and max(totals) < core.STOP_LINE_HOURS
    assert main + sum(measured.values()) <= st.POSTHOC_COST_CAP_HOURS
    assert status["state"] in ("complete", "stopped")


def test_a_run_inside_the_cap_completes_and_a_measured_job_is_not_run_again(tmp_path: Path) -> None:
    """Jobs measured by an earlier run count towards the total and are not fitted again."""
    root = _finished_main_run(tmp_path)
    steps, started, measured = _cap_steps(
        root, main=1.0, hours_of={"YO": 0.1, "FS": 0.1, "YS": 0.1}
    )
    measured["YO_s0"] = 0.1
    status = st.run_posthoc(root=root, steps=steps)
    assert status["state"] == "complete" and status["skipped"] == []
    assert ("YO", None, 0) not in started and len(started) == 29
    assert status["hours"]["per_arm"]["YO"] == pytest.approx(0.3)


def test_the_measured_cost_is_read_from_the_job_records(tmp_path: Path) -> None:
    """The main chain's job records and the post hoc fits' give the running total."""
    root = _finished_main_run(tmp_path)
    rec = {
        "fit_seconds": 3600.0,
        "predict_seconds": 1800.0,
        "fit_peak_rss_gb": 1.0,
        "predict_peak_rss_gb": 1.0,
    }
    st.write_json(path=root / "fits" / "YU_s0.job.json", payload=rec)
    fit_root = st.posthoc_fit_root(root=root)
    st.write_json(path=fit_root / "fits" / "YO_s0.job.json", payload=rec)
    st.write_json(path=fit_root / "fits" / "FS_2025-11_s0.job.json", payload=rec)
    cost = st.posthoc_cost(root=root)
    assert cost["main_chain"] == 1.5 and cost["per_arm"] == {"YO": 1.5, "FS": 1.5}
    assert cost["posthoc"] == 3.0 and cost["running_total"] == 4.5
    assert st.job_hours_of(root=fit_root, arm="YO", fold=None, seed=0) == 1.5
    assert st.job_hours_of(root=fit_root, arm="YS", fold=None, seed=0) is None


# --------------------------------------------------------------------------- stopped runs, failed jobs, Ctrl-C


def test_posthoc_score_accepts_a_stopped_run(tmp_path: Path) -> None:
    """A cost-cap stop leaves partial results, and the readings may read them."""
    root = _finished_main_run(tmp_path)
    st.write_posthoc_status(root=root, state="stopped", reason=st.COST_CAP_MESSAGE)
    st.guard_posthoc(root=root, stage="posthoc_score", banks_digest=DIGEST)
    st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)


def test_a_failed_job_adds_its_wall_seconds_to_the_total_and_a_resume_keeps_them(
    tmp_path: Path,
) -> None:
    """A failing job writes no job record, so its seconds are recorded and count towards the cap."""
    root = _finished_main_run(tmp_path)
    steps, started, _ = _cap_steps(root, main=1.0, hours_of={"YO": 0.01, "FS": 0.01, "YS": 0.01})
    clock = [0.0]
    inner = steps.run_job

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        clock[0] += 7200.0 if (arm, seed) == ("YO", 0) else 1.0
        if (arm, seed) == ("YO", 0):
            msg = "boom"
            raise RuntimeError(msg)
        return inner(arm, fold, seed)

    steps.run_job = run_job
    steps.clock = lambda: clock[0]
    status = st.run_posthoc(root=root, steps=steps)
    assert status["failed"] == ["YO_s0"]
    assert st.failed_hours(root=root) == pytest.approx(2.0)
    assert status["hours"]["failed"] == pytest.approx(2.0)
    assert status["hours"]["running_total"] == pytest.approx(1.0 + 2.0 + 0.01 * (len(started)))
    assert st.posthoc_cost(root=root)["failed"] == pytest.approx(2.0)
    # A resume starts from the recorded seconds: with the main chain at 9.9 hours, 2 more hours of
    # failed time pass the cap before any job starts.
    steps2, started2, _ = _cap_steps(root, main=9.9, hours_of={"YO": 0.01, "FS": 0.01, "YS": 0.01})
    status2 = st.run_posthoc(root=root, steps=steps2)
    assert status2["state"] == "stopped" and started2 == []


def test_a_clean_ctrl_c_writes_stopped_with_the_reason_interrupted_and_reraises(
    tmp_path: Path,
) -> None:
    """KeyboardInterrupt ends the run with the status `stopped`, so `posthoc` can start again."""
    root = _finished_main_run(tmp_path)
    steps, _, _ = _cap_steps(root, main=1.0, hours_of={"YO": 0.01, "FS": 0.01, "YS": 0.01})

    def run_job(_arm: str, _fold: str | None, _seed: int) -> dict:
        raise KeyboardInterrupt

    steps.run_job = run_job
    with pytest.raises(KeyboardInterrupt):
        st.run_posthoc(root=root, steps=steps)
    status = st.read_json(path=root / st.POSTHOC_STATUS)
    assert status["state"] == "stopped" and status["reason"] == "interrupted"
    st.guard_posthoc(root=root, stage="posthoc", banks_digest=DIGEST)
