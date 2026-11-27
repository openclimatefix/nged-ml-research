# ruff: noqa: PT018
"""Tests of `e00733_stages`: the guard, the gates, the timing gate, the drop rule, the screen rule,
the chain, the exit codes, and the test marker."""

import json
import os
import time
from pathlib import Path

import e00733_core as core
import e00733_fit as fit
import e00733_paths as paths
import e00733_stages as st
import pytest

DIGEST = "banks-digest"


def _ready(root: Path, *, through: str = "review") -> None:
    """Write the files of the gates up to `through` (channels, limits, gallery, review, smoke,
    timing)."""
    st.write_run_marker(root=root, banks_digest=DIGEST)
    order = ["channels", "limits", "gallery", "review", "smoke", "timing"]
    for step in order[: order.index(through) + 1]:
        if step == "channels":
            paths.channels_dir(root=root).mkdir(parents=True, exist_ok=True)
            paths.channels_json(root=root).write_text("{}")
            paths.table_path(root=root).write_text("")
            for name in st.PRE_FIT_GUARD_NAMES:
                st.record_pre_fit_guard(root=root, name=name, payload={})
        elif step == "limits":
            (root / "limits.json").write_text("{}")
        elif step == "gallery":
            st.write_json(
                path=root / st.PANEL_CHECK,
                payload={
                    "banks_digest": DIGEST,
                    "panels": dict.fromkeys(core.GALLERY_PANELS, True),
                },
            )
        elif step == "review":
            st.write_json(
                path=root / st.REVIEW_GATE, payload={"opus_code_review": True, "fix_review": True}
            )
        elif step == "smoke":
            st.write_json(
                path=root / st.SMOKE_FILE,
                payload={
                    "pass": True,
                    "code_version": st.CODE_VERSION,
                    "device": "cuda",
                    "gpu_name": "stub GPU",
                    "peak_gpu_gb": 2.5,
                },
            )
        elif step == "timing":
            st.write_json(
                path=root / st.TIMING_FILE,
                payload={
                    "pass": True,
                    "seconds_per_real_fit": 300.0,
                    "seconds_per_synthetic_fit": 1000.0,
                },
            )


def test_guard_needs_the_run_marker_and_every_gate_before_a_fit(tmp_path: Path) -> None:
    """A stage refuses without a matching marker; a fit stage needs channels, limits, gallery,
    review, smoke, and timing, each in turn."""
    with pytest.raises(st.GuardError, match="run marker"):
        st.guard_run(root=tmp_path, stage="limits", banks_digest=DIGEST)
    st.write_run_marker(root=tmp_path, banks_digest="other")
    with pytest.raises(st.GuardError, match="run marker"):
        st.guard_run(root=tmp_path, stage="banks", banks_digest=DIGEST)
    for through, needle in [
        ("channels", "limits.json"),
        ("limits", "panel_check"),
        ("gallery", "review"),
        ("review", "smoke"),
        ("smoke", "timing"),
    ]:
        root = tmp_path / through
        root.mkdir()
        _ready(root, through=through)
        with pytest.raises(st.GuardError, match=needle):
            st.guard_run(root=root, stage="chain", banks_digest=DIGEST)
    full = tmp_path / "full"
    full.mkdir()
    _ready(full, through="timing")
    st.guard_run(root=full, stage="chain", banks_digest=DIGEST)


def test_a_gallery_panel_that_failed_closes_the_gate(tmp_path: Path) -> None:
    """One false panel, a missing panel, or the digest of other banks keeps the gallery gate
    closed."""
    _ready(tmp_path, through="review")
    panels = dict.fromkeys(core.GALLERY_PANELS, True)
    panels["G12"] = False
    st.write_json(
        path=tmp_path / st.PANEL_CHECK, payload={"banks_digest": DIGEST, "panels": panels}
    )
    assert not st.gallery_passed(root=tmp_path, banks_digest=DIGEST)
    st.write_json(
        path=tmp_path / st.PANEL_CHECK, payload={"banks_digest": DIGEST, "panels": {"G10": True}}
    )
    assert not st.gallery_passed(root=tmp_path, banks_digest=DIGEST)
    st.write_json(
        path=tmp_path / st.PANEL_CHECK,
        payload={"banks_digest": "x", "panels": dict.fromkeys(core.GALLERY_PANELS, True)},
    )
    assert not st.gallery_passed(root=tmp_path, banks_digest=DIGEST)


def test_an_invalid_record_or_a_test_marker_stops_every_fit_stage(tmp_path: Path) -> None:
    """After `invalid.json` or the first test score no fit may start; `invalid.json` is
    write-once."""
    _ready(tmp_path, through="timing")
    st.record_invalid(root=tmp_path, rule="first")
    st.record_invalid(root=tmp_path, rule="second")
    assert json.loads((tmp_path / st.INVALID_FILE).read_text()) == {"rule": "first"}
    with pytest.raises(st.GuardError, match="correctness"):
        st.guard_run(root=tmp_path, stage="fit", banks_digest=DIGEST)
    other = tmp_path / "o"
    other.mkdir()
    _ready(other, through="timing")
    (other / "scores" / st.TEST_MARKER).touch()
    with pytest.raises(st.GuardError, match="first test score"):
        st.guard_run(root=other, stage="chain", banks_digest=DIGEST)


def test_score_waits_for_the_chain_and_reading_stages_wait_while_it_runs(tmp_path: Path) -> None:
    """`score` needs stored predictions and a finished chain; no reading stage runs while the chain
    says running."""
    _ready(tmp_path, through="timing")
    with pytest.raises(st.GuardError, match="stored predictions"):
        st.guard_run(root=tmp_path, stage="score", banks_digest=DIGEST)
    (tmp_path / "predictions" / "FU").mkdir(parents=True)
    (tmp_path / "predictions" / "FU" / "a.parquet").write_text("")
    st.write_chain_status(root=tmp_path, state="running")
    with pytest.raises(st.GuardError, match="running"):
        st.guard_run(root=tmp_path, stage="figures", banks_digest=DIGEST)
    st.write_chain_status(root=tmp_path, state="complete")
    st.guard_run(root=tmp_path, stage="score", banks_digest=DIGEST)


@pytest.mark.parametrize("stage", ["figures", "series", "attention"])
def test_a_stage_that_draws_test_rows_needs_the_first_test_score_marker(
    tmp_path: Path, stage: str
) -> None:
    """Without `scores/test_started`, a stage that reads test rows is refused; validation rows and
    the marker both open it, and `loss_curves` (stored fit records only) needs neither."""
    _ready(tmp_path, through="timing")
    with pytest.raises(st.GuardError, match="test_started"):
        st.guard_run(root=tmp_path, stage=stage, banks_digest=DIGEST)
    st.guard_run(root=tmp_path, stage=stage, banks_digest=DIGEST, validation_only=True)
    st.guard_run(root=tmp_path, stage="loss_curves", banks_digest=DIGEST)
    (tmp_path / "scores" / st.TEST_MARKER).touch()
    st.guard_run(root=tmp_path, stage=stage, banks_digest=DIGEST)


def test_test_marker_is_written_only_after_the_choices_were_fixed(tmp_path: Path) -> None:
    """The marker needs `limits.json` and the stored predictions to be older than it; a late
    rewrite of the limits fails."""
    from e00733_leakage import CorrectnessFailureError

    (tmp_path / "limits.json").write_text("{}")
    with pytest.raises(CorrectnessFailureError):
        st.write_test_marker(root=tmp_path)
    assert not st.marker_exists(root=tmp_path)
    (tmp_path / "predictions" / "FU").mkdir(parents=True)
    (tmp_path / "predictions" / "FU" / "a.parquet").write_text("")
    now = time.time()
    os.utime(tmp_path / "limits.json", (now - 20, now - 20))
    os.utime(tmp_path / "predictions" / "FU" / "a.parquet", (now - 10, now - 10))
    st.write_test_marker(root=tmp_path)
    assert st.marker_exists(root=tmp_path)
    os.utime(tmp_path / "limits.json", (now + 5, now + 5))
    assert (
        st.choice_events(root=tmp_path)[0] != "limits"
        or st.choice_events(root=tmp_path)[-1] == "limits"
    )


def test_timing_decision_drops_yt_then_fs_and_stops_when_counted_work_is_over() -> None:
    """The projection at the plan's rates passes; at slower rates YT then FS go; at much slower
    rates the line stops at row 0."""
    ok = st.timing_decision(
        seconds_per_real_fit=0.08 * 3600,
        seconds_per_synthetic_fit=0.3 * 3600,
        peak_gpu_gb=3.5,
        peak_rss_gb=8,
    )
    assert ok["dropped"] == [] and ok["projected_hours"] == pytest.approx(
        core.PLANNED_HOURS, abs=0.05
    )
    slow = st.timing_decision(
        seconds_per_real_fit=0.1 * 3600,
        seconds_per_synthetic_fit=0.3 * 3600,
        peak_gpu_gb=3.5,
        peak_rss_gb=8,
    )
    assert slow["dropped"] == ["YT"] or slow["dropped"] == ["YT", "FS"]
    assert slow["projected_hours"] <= core.STOP_LINE_HOURS
    slower = st.timing_decision(
        seconds_per_real_fit=0.11 * 3600,
        seconds_per_synthetic_fit=0.3 * 3600,
        peak_gpu_gb=3.5,
        peak_rss_gb=8,
    )
    assert slower["dropped"] == ["YT", "FS"]
    same = st.timing_decision(
        seconds_per_real_fit=0.08 * 3600,
        seconds_per_synthetic_fit=0.3 * 3600,
        peak_gpu_gb=3.5,
        peak_rss_gb=8,
        fu_equals_fs=True,
    )
    assert same["not_fitted"] == ["FS"] and same["projected_hours"] < ok["projected_hours"]
    with pytest.raises(st.StoppedError):
        st.timing_decision(
            seconds_per_real_fit=0.2 * 3600,
            seconds_per_synthetic_fit=0.3 * 3600,
            peak_gpu_gb=3.5,
            peak_rss_gb=8,
        )
    with pytest.raises(st.StoppedError, match="GPU memory"):
        st.timing_decision(
            seconds_per_real_fit=1, seconds_per_synthetic_fit=1, peak_gpu_gb=7.0, peak_rss_gb=1
        )


def test_family_jobs_match_the_arm_table() -> None:
    """The families cover every fit of every arm exactly once (the screen is part of FU)."""
    total = sum(len(st.family_jobs(family=f)) for f in st.FAMILIES)
    assert total == sum(a.n_fits for a in core.ARMS.values())
    screen = st.family_jobs(family="FU-screen")
    assert len(screen) == 8 and not set(screen) & set(st.family_jobs(family="FU"))
    assert len(st.family_jobs(family="FU")) == 32


def _steps(
    *,
    fail: set[str] | None = None,
    point: float = -0.01,
    log: list[str] | None = None,
    calls: list | None = None,
) -> st.ChainSteps:
    fail = fail or set()
    out = log if log is not None else []
    seen = calls if calls is not None else []

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        seen.append((arm, fold, seed))
        if arm in fail:
            msg = "boom"
            raise RuntimeError(msg)
        return {}

    return st.ChainSteps(
        run_job=run_job,
        f0r_gate=lambda: None,
        screen_point=lambda: point,
        log=out.append,
        now=lambda: "T",
    )


def test_chain_runs_every_family_in_order_with_its_lines(tmp_path: Path) -> None:
    """A complete chain runs the families in the plan's order and prints a start line and an end
    line for each; exit 0."""
    _ready(tmp_path, through="timing")
    log: list[str] = []
    calls: list = []
    status = st.run_chain(root=tmp_path, steps=_steps(log=log, calls=calls))
    assert status["state"] == "complete" and st.exit_code(status=status) == 0
    arms = [c[0] for c in calls]
    assert (
        arms[:4] == ["F0r"] * 4 and arms.index("FU") > arms.index("F0r") and arms[-3:] == ["YT"] * 3
    )
    assert sum("fits starting" in line for line in log) == len(st.FAMILIES)
    assert sum("last fit ended" in line for line in log) == len(st.FAMILIES)
    assert len(calls) == sum(a.n_fits for a in core.ARMS.values())


def test_screen_rule_skips_the_rest_of_fu_and_the_placebo_but_not_the_synthetic_arms(
    tmp_path: Path,
) -> None:
    """A screen point above +2% stops FU and FU-S and keeps Y0, YU, FS, YT."""
    _ready(tmp_path, through="timing")
    calls: list = []
    status = st.run_chain(root=tmp_path, steps=_steps(point=0.05, calls=calls))
    assert status["screen_stopped"] and status["state"] == "complete"
    assert len([c for c in calls if c[0] == "FU"]) == 8 and not [c for c in calls if c[0] == "FU-S"]
    assert {c[0] for c in calls} >= {"Y0", "YU", "FS", "YT"}
    assert json.loads((tmp_path / st.SCREEN_FILE).read_text())["interim"] is True


def test_chain_stops_on_a_failed_counted_fit_with_exit_5_and_records_the_traceback(
    tmp_path: Path,
) -> None:
    """A failed counted fit ends the chain as `failed`, exit 5 (never 1), with the message and
    traceback stored; a failed report-only fit does not."""
    _ready(tmp_path, through="timing")
    calls: list = []
    status = st.run_chain(root=tmp_path, steps=_steps(fail={"FU"}, calls=calls))
    assert status["state"] == "failed" and st.exit_code(status=status) == 5
    assert not [c for c in calls if c[0] == "Y0"]
    assert len([c for c in calls if c[0] == "FU"]) == 1  # the chain stops at the first failure
    assert status["failed"] == [st.fit.job_name(arm="FU", fold=calls[-1][1], seed=calls[-1][2])]
    failed = json.loads(next((tmp_path / "fits").glob("FU_*.failed.json")).read_text())
    assert failed["message"] == "boom" and "RuntimeError" in failed["traceback"]
    other = tmp_path / "o"
    other.mkdir()
    _ready(other, through="timing")
    status = st.run_chain(root=other, steps=_steps(fail={"FS"}))
    assert status["state"] == "complete" and len(status["failed"]) == 24


def test_chain_skips_dropped_arms_and_turns_a_stop_into_state_stopped(tmp_path: Path) -> None:
    """Arms dropped by the timing gate are skipped; an F0r difference stops the chain (exit 3)
    before any FU fit."""
    _ready(tmp_path, through="timing")
    st.write_json(
        path=tmp_path / st.TIMING_FILE,
        payload={
            "pass": True,
            "dropped": ["YT", "FS"],
            "not_fitted": [],
            "seconds_per_real_fit": 1,
            "seconds_per_synthetic_fit": 1,
        },
    )
    calls: list = []
    st.run_chain(root=tmp_path, steps=_steps(calls=calls))
    assert not {c[0] for c in calls} & {"YT", "FS"}
    other = tmp_path / "o"
    other.mkdir()
    _ready(other, through="timing")
    calls = []

    def stop() -> None:
        msg = "F0r differs"
        raise st.StoppedError(msg)

    steps = _steps(calls=calls)
    steps.f0r_gate = stop
    status = st.run_chain(root=other, steps=steps)
    assert status["state"] == "stopped" and st.exit_code(status=status) == 3
    assert {c[0] for c in calls} == {"F0r"}


def test_exit_codes_are_never_the_slot_busy_code() -> None:
    """Stopped 3, invalid 4, guard 2, any other failure 5; none is 1."""
    from e00733_leakage import CorrectnessFailureError

    codes = [
        st.exit_code(exc=e)
        for e in (
            st.StoppedError("x"),
            CorrectnessFailureError("x"),
            st.GuardError("x"),
            RuntimeError("x"),
        )
    ]
    assert codes == [3, 4, 2, 5] and 1 not in codes
    assert fit.chain_exit_code(status={"state": "failed"}) == 5


def test_a_fit_stage_needs_the_base_day_and_test_segment_guards(tmp_path: Path) -> None:
    """The chain refuses until the base-day and test-segment guards have been recorded."""
    _ready(tmp_path, through="timing")
    (tmp_path / st.PRE_FIT_GUARDS).unlink()
    for recorded in ((), ("base_day_guards",), ("test_segment_coverage",)):
        (tmp_path / st.PRE_FIT_GUARDS).unlink(missing_ok=True)
        for name in recorded:
            st.record_pre_fit_guard(root=tmp_path, name=name, payload={"n": 1})
        with pytest.raises(st.GuardError, match="pre_fit_guards"):
            st.guard_run(root=tmp_path, stage="chain", banks_digest=DIGEST)
    for name in st.PRE_FIT_GUARD_NAMES:
        st.record_pre_fit_guard(root=tmp_path, name=name, payload={"n": 1})
    st.guard_run(root=tmp_path, stage="chain", banks_digest=DIGEST)
    with pytest.raises(st.GuardError):
        st.record_pre_fit_guard(root=tmp_path, name="other", payload={})


@pytest.mark.parametrize(
    "module_error",
    ["e00733_inputs.InputsError", "e00733_rerun.RerunError", "e00733_synth.SynthError",
     "e00733_world.WorldError", "e00733_real.RealWindowError", "e00733_score.ScoreError"],
)  # fmt: skip
def test_a_correctness_failure_inside_the_chain_ends_it_with_exit_4_not_5(
    tmp_path: Path, module_error: str
) -> None:
    """A module error documented as verdict row 1 ends the chain as `invalid` with exit 4."""
    import importlib

    module, name = module_error.split(".")
    error = getattr(importlib.import_module(module), name)
    assert issubclass(error, st.CorrectnessFailureError)
    _ready(tmp_path, through="timing")

    def run_job(arm: str, fold: str | None, seed: int) -> dict:
        raise error("a check failed")

    steps = st.ChainSteps(
        run_job=run_job, f0r_gate=lambda: None, screen_point=lambda: 0.0, log=lambda _m: None,
        now=lambda: "T",
    )  # fmt: skip
    status = st.run_chain(root=tmp_path, steps=steps)
    assert status["state"] == "failed" and st.invalid_recorded(root=tmp_path)
    assert st.exit_code(status=status) == st.EXIT_INVALID == 4
