# ruff: noqa: E501
"""Experiment 733: the stage order, the run marker, the gates, the timing gate, and the chain.

(through `e00733_fit`, which imports torch).

A stage runs `guard_run` first, so a stage on the outputs of another run, before a gate, or after a
correctness failure refuses to start. A gate that fails raises `StoppedError` (verdict row 0) or
`CorrectnessFailureError` (verdict row 1); the command line turns either into a status file and an
exit code (stopped 3, invalid 4, a failed chain 5, never 1, which is the launcher's "slot busy").

The chain takes its work as callables (`ChainSteps`), so a test runs the order, the screen rule, the
drop rule, and the failure handling without a GPU. No stage prints a date, a series identifier, or a
generator value.
"""

import hashlib
import json
import math
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00733_channels as channels
import e00733_core as core
import e00733_fit as fit
import e00733_leakage as leak
import e00733_paths as paths

Log = Callable[[str], None]
StoppedError = channels.StoppedError
CorrectnessFailureError = leak.CorrectnessFailureError
GuardError = fit.GuardError

CODE_VERSION: Final = core.CODE_VERSION
"""The version of this experiment's code that the run marker records. A change that alters a stored
output raises `core.CODE_VERSION`."""
RUN_MARKER: Final = "run_marker.json"
INVALID_FILE: Final = "scores/invalid.json"
TEST_MARKER: Final = "test_started"
CHAIN_STATUS: Final = "chain_status.json"
CHAIN_DONE_STATES: Final = ("complete", "failed", "stopped")
"""The chain states after which the test outputs may be scored."""
REVIEW_GATE: Final = "scores/review_gate.json"
"""Written by the session that runs the experiment once the Opus code review and the review of its
fixes have passed."""
PRE_FIT_GUARDS: Final = "scores/pre_fit_guards.json"
"""The record of the guards that must pass before any fit: the base days precede every fold's history
with no overlap, and the channel coverage including the re-run reaches 90% (both written by
`channels`), and the synthetic test segments are covered by ensemble runs (written by `banks`)."""
PRE_FIT_GUARD_NAMES: Final = ("base_day_guards", "rerun_coverage", "test_segment_coverage")
SMOKE_FILE: Final = "scores/smoke.json"
SMOKE_CPU_FILE: Final = "scores/smoke_cpu.json"
"""The record of a smoke test on the CPU, which never opens the chain."""
TIMING_FILE: Final = "scores/timing.json"
SCREEN_FILE: Final = "scores/screen.json"
PANEL_CHECK: Final = "PRIVATE/gallery/panel_check.json"
EXIT_OK: Final = 0
EXIT_GUARD: Final = 2
EXIT_STOPPED: Final = 3
EXIT_INVALID: Final = 4
EXIT_FAILED: Final = core.CHAIN_FAILED_EXIT
READING_STAGES: Final = ("score", "report", "figures", "loss_curves", "attention", "series")
"""The stages that read the test outputs; none may run while the chain is fitting."""
GATED_STAGES: Final = ("smoke_fit", "timing", "fit", "chain")
"""The stages that start a fit: they need the channels, the limits, the gallery check, the review
gate, and no recorded failure."""
FIT_STAGES: Final = ("fit", "chain")
"""The stages that also need the smoke test and the timing gate."""
FAMILIES: Final = ("F0r", "FU-screen", "FU", "FU-S", "Y0", "YU", "FS", "YT")


# --------------------------------------------------------------------------- files


def write_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file atomically."""
    fit.write_json(path=path, payload=payload)


def read_json(*, path: Path) -> dict:
    """Read a JSON file."""
    return json.loads(path.read_text())


def write_once_json(*, path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON file that may never be written twice.

    (a second write raises `FileExistsError`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=1, default=str)


def file_digest(*, path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_invalid(*, root: Path, rule: str) -> None:
    """Write `scores/invalid.json` once (verdict row 1); a later call leaves the first record."""
    try:
        write_once_json(path=root / INVALID_FILE, payload={"rule": rule})
    except FileExistsError:
        return


def invalid_recorded(*, root: Path) -> bool:
    """Return whether a correctness check has failed in this run."""
    return (root / INVALID_FILE).exists()


def marker_exists(*, root: Path) -> bool:
    """Return whether the first test score has been taken (the marker exists)."""
    return (root / "scores" / TEST_MARKER).exists()


def prediction_files(*, root: Path) -> list[Path]:
    """Return the stored prediction files of this experiment's fits.

    (real parquet files and synthetic archives).
    """
    folder = root / "predictions"
    return sorted([*folder.glob("*/*.parquet"), *folder.glob("*/*.npz"), *folder.glob("*.npz")])


def chain_state(*, root: Path) -> str | None:
    """Return the state in `chain_status.json`, or None."""
    path = root / CHAIN_STATUS
    return read_json(path=path).get("state") if path.exists() else None


def chain_finished(*, root: Path) -> bool:
    """Return whether stored predictions exist and the chain ended (complete, failed.

    Or stopped).
    """
    return bool(prediction_files(root=root)) and chain_state(root=root) in CHAIN_DONE_STATES


def chain_running(*, root: Path) -> bool:
    """Return whether `chain_status.json` says that the chain is running."""
    return chain_state(root=root) == "running"


def choice_events(*, root: Path) -> list[str]:
    """Return the choices of the run and the first test score, in order of file time.

    The events are `limits` (`limits.json`), `fits` (the newest stored prediction file), and
    `test_score` (the marker, or now when it does not exist yet). A choice rewritten after the first
    test score sorts after `test_score`, and `leak.assert_no_test_choice` fails.
    """
    marker = root / "scores" / TEST_MARKER
    stamps: dict[str, int] = {
        "test_score": marker.stat().st_mtime_ns if marker.exists() else time.time_ns()
    }
    if (root / "limits.json").exists():
        stamps["limits"] = (root / "limits.json").stat().st_mtime_ns
    fits = prediction_files(root=root)
    if fits:
        stamps["fits"] = max(p.stat().st_mtime_ns for p in fits)
    return sorted(stamps, key=lambda k: (stamps[k], k == "test_score"))


def write_test_marker(*, root: Path) -> None:
    """Write the marker of the first test score.

    Only after `assert_no_test_choice` passes on the choice events.

    Args:
        root: The data folder.

    Raises:
        CorrectnessFailureError: When the limits or the fits were not fixed before the first test
            score.
    """
    if marker_exists(root=root):
        return
    leak.assert_no_test_choice(events=choice_events(root=root))
    marker = root / "scores" / TEST_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


# --------------------------------------------------------------------------- the run marker and the gates


def run_marker_payload(*, banks_digest: str) -> dict:
    """Return what the run marker holds: the experiment, the row sets, the code version.

    And the banks' digest.
    """
    return {
        "experiment": core.EXPERIMENT_ID, "row_set_id_real": core.ROW_SET_ID_REAL,
        "row_set_id_synthetic": core.ROW_SET_ID_SYNTHETIC, "code_version": CODE_VERSION,
        "banks_digest": banks_digest,
    }  # fmt: skip


def write_run_marker(*, root: Path, banks_digest: str) -> None:
    """Write the run marker (a marker that differs is refused by `guard_run`)."""
    write_json(path=root / RUN_MARKER, payload=run_marker_payload(banks_digest=banks_digest))


def _passed(*, path: Path) -> bool:
    return path.exists() and bool(read_json(path=path).get("pass"))


def gallery_passed(*, root: Path, banks_digest: str) -> bool:
    """Return whether `PRIVATE/gallery/panel_check.json` records every new panel as passed.

    For the banks in use.
    """
    path = root / PANEL_CHECK
    if not path.exists():
        return False
    record = read_json(path=path)
    panels = record.get("panels", {})
    return (
        record.get("banks_digest") == banks_digest
        and set(core.GALLERY_PANELS) <= set(panels)
        and all(bool(v) for v in panels.values())
    )


def review_gate_passed(*, root: Path) -> bool:
    """Return whether the review gate file records both reviews as passed.

    The reviews are the Opus code review and the review of its fixes.
    """
    path = root / REVIEW_GATE
    if not path.exists():
        return False
    record = read_json(path=path)
    return bool(record.get("opus_code_review")) and bool(record.get("fix_review"))


def smoke_file(*, device: str) -> str:
    """Return the record file of a smoke test on `device`: only the GPU's opens the chain."""
    return SMOKE_FILE if device == "cuda" else SMOKE_CPU_FILE


def smoke_passed(*, root: Path) -> bool:
    """Return whether `scores/smoke.json` records a pass on the GPU under this code version.

    The smoke test is the real-environment one: the record must name the device `cuda`, the GPU,
    and a positive peak GPU memory.
    """
    path = root / SMOKE_FILE
    if not _passed(path=path):
        return False
    record = read_json(path=path)
    peak = record.get("peak_gpu_gb")
    return (
        record.get("code_version") == CODE_VERSION
        and record.get("device") == "cuda"
        and bool(record.get("gpu_name"))
        and isinstance(peak, int | float)
        and math.isfinite(peak)
        and peak > 0
    )


def timing_passed(*, root: Path) -> bool:
    """Return whether `scores/timing.json` records a pass."""
    return _passed(path=root / TIMING_FILE)


def channels_passed(*, root: Path) -> bool:
    """Return whether `channels.json` and the channel table exist.

    (the coverage check passed when the table was written).
    """
    return paths.channels_json(root=root).exists() and paths.table_path(root=root).exists()


def record_pre_fit_guard(*, root: Path, name: str, payload: Mapping[str, Any]) -> None:
    """Add one passed guard to `scores/pre_fit_guards.json`.

    Args:
        root: The data folder of this experiment.
        name: One of `PRE_FIT_GUARD_NAMES`.
        payload: What the guard counted.

    Raises:
        GuardError: When `name` is not a pre-fit guard.
    """
    if name not in PRE_FIT_GUARD_NAMES:
        msg = f"{name} is not a pre-fit guard"
        raise GuardError(msg)
    path = root / PRE_FIT_GUARDS
    record = read_json(path=path) if path.exists() else {}
    record[name] = dict(payload)
    write_json(path=path, payload=record)


def pre_fit_guards_passed(*, root: Path) -> bool:
    """Return whether every guard of `PRE_FIT_GUARD_NAMES` has been recorded as passed."""
    path = root / PRE_FIT_GUARDS
    if not path.exists():
        return False
    record = read_json(path=path)
    return all(name in record for name in PRE_FIT_GUARD_NAMES)


def guard_run(*, root: Path, stage: str, banks_digest: str) -> None:  # noqa: PLR0912
    """Refuse to run a stage on the outputs of another run, or without its gates.

    Args:
        root: The data folder of this experiment.
        stage: The stage name.
        banks_digest: The digest id of experiment 730's banks in use.

    Raises:
        GuardError: When a gate is open, a correctness check has failed, or the folder holds the
            outputs of another run.
    """
    expected = run_marker_payload(banks_digest=banks_digest)
    marker = root / RUN_MARKER
    current = read_json(path=marker) if marker.exists() else None
    if stage == "banks":
        if current is not None and current != expected:
            msg = "the run marker differs from the code and the banks in use: move the outputs to superseded_run_N yourself"
            raise GuardError(msg)
        return
    if current != expected:
        msg = "no run marker matches the banks and the code in use: run `banks` first"
        raise GuardError(msg)
    needs: dict[str, tuple[tuple[str, bool], ...]] = {
        "limits": (("channels.json and the channel table", channels_passed(root=root)),),
        "gallery": (("limits.json", (root / "limits.json").exists()),),
    }
    for what, ok in needs.get(stage, ()):
        if not ok:
            msg = f"`{stage}` needs {what} first"
            raise GuardError(msg)
    if stage in GATED_STAGES:
        if marker_exists(root=root):
            msg = f"`{stage}` starts a fit, but the first test score was already taken: a fit now would be a choice made after the test (verdict row 1)"
            raise GuardError(msg)
        if invalid_recorded(root=root):
            msg = f"`{stage}` starts a fit, but {INVALID_FILE} records a failed correctness check (verdict row 1): the run is over"
            raise GuardError(msg)
        for what, ok in (
            ("channels.json and the channel table", channels_passed(root=root)),
            ("limits.json, written before any fit", (root / "limits.json").exists()),
            (
                f"{PRE_FIT_GUARDS} with the base-day and test-segment guards passed",
                pre_fit_guards_passed(root=root),
            ),
            (
                f"{PANEL_CHECK} with every panel passed (the gallery gate of the plan)",
                gallery_passed(root=root, banks_digest=banks_digest),
            ),
            (
                f"{REVIEW_GATE} with the Opus code review and the review of its fixes passed",
                review_gate_passed(root=root),
            ),
        ):
            if not ok:
                msg = f"`{stage}` needs {what}"
                raise GuardError(msg)
    if stage in ("timing", *FIT_STAGES) and not smoke_passed(root=root) and stage != "smoke_fit":
        msg = f"`{stage}` needs {SMOKE_FILE} with a pass under the current code version: run the smoke test first"
        raise GuardError(msg)
    if stage in FIT_STAGES and not timing_passed(root=root):
        msg = f"`{stage}` needs {TIMING_FILE} with a pass: run `timing` first"
        raise GuardError(msg)
    if stage in READING_STAGES:
        if not (root / "limits.json").exists():
            msg = f"`{stage}` needs limits.json"
            raise GuardError(msg)
        if chain_running(root=root):
            msg = f"`{stage}` reads the test outputs, but the chain is running (`chain_status.json` says running): wait for the last fit"
            raise GuardError(msg)
    if stage == "score" and not chain_finished(root=root):
        msg = "`score` needs stored predictions and a `chain_status.json` that says complete, failed, or stopped: the first test score ends every later fit"
        raise GuardError(msg)


# --------------------------------------------------------------------------- the jobs and the timing gate


def family_jobs(*, family: str) -> list[tuple[str, str | None, int]]:
    """Return the (arm, fold, seed) jobs of one family of `FAMILIES`.

    `FU-screen` is FU on the four screening folds at seeds 0 and 1; `FU` is the rest of FU.
    """
    if family == "FU-screen":
        return [("FU", f, s) for f in core.SCREEN_FOLDS for s in core.SCREEN_SEEDS]
    spec = core.ARMS["FU" if family == "FU" else family]
    folds: Sequence[str | None] = spec.folds or (None,)
    jobs = [(spec.name, f, s) for f in folds for s in spec.seeds]
    if family == "FU":
        done = set(family_jobs(family="FU-screen"))
        jobs = [j for j in jobs if j not in done]
    return jobs


def timing_decision(
    *, seconds_per_real_fit: float, seconds_per_synthetic_fit: float, peak_gpu_gb: float, peak_rss_gb: float,
    fu_equals_fs: bool = False, smoke_seconds: float = 0.0, timing_gate_seconds: float = 0.0,
) -> dict[str, Any]:  # fmt: skip
    """Return the timing gate's decision: the projection, the arms dropped from the end.

    And whether the counted work fits.

    The projection adds the planned 0.5 hours of the smoke test, the timing gate, and the re-runs
    to the
    fits at their measured rate. Report-only arms are dropped in `core.DROP_ORDER` (YT, then FS)
    until
    the projection is at or under the stop line; FS is not fitted when it equals FU.

    Args:
        seconds_per_real_fit: The measured seconds of one whole real-fold job (FU, 1 fold, seed 0):
            the fit, every prediction set, and the target-poison check.
        seconds_per_synthetic_fit: The measured seconds of one whole synthetic job (5 passes of the
            fit scaled to 80, plus the prediction).
        peak_gpu_gb: The measured peak GPU memory.
        peak_rss_gb: The measured peak resident memory.
        fu_equals_fs: Whether FS is the same arm as FU (then FS is not fitted).
        smoke_seconds: The measured seconds of the smoke test (0 when not measured).
        timing_gate_seconds: The measured seconds of the timing gate itself (0 when not measured).

    Returns:
        A dictionary with `pass`, `projected_hours`, `dropped`, `projected_counted_hours`, and the
        measurements.

    Raises:
        StoppedError: When the counted work alone exceeds the stop line, the peak GPU memory is
            above the slot's cap, or the peak resident memory is above `core.RSS_STOP_GB`.
    """
    fixed = (
        core.planned_hours(names=("reruns",))
        + max(core.planned_hours(names=("smoke",)), smoke_seconds / 3600.0)
        + max(core.planned_hours(names=("timing_gate",)), timing_gate_seconds / 3600.0)
    )
    already = ("FS",) if fu_equals_fs else ()

    def hours(dropped: tuple[str, ...]) -> float:
        return fixed + core.projected_hours(
            seconds_per_real_fit=seconds_per_real_fit,
            seconds_per_synthetic_fit=seconds_per_synthetic_fit, dropped=dropped,
        )  # fmt: skip

    if peak_gpu_gb > core.GPU_GB:
        msg = f"the peak GPU memory {peak_gpu_gb:.1f} GB is above the slot's {core.GPU_GB} GB (verdict row 0)"
        raise StoppedError(msg)
    if peak_rss_gb > core.RSS_STOP_GB:
        msg = f"the peak resident memory {peak_rss_gb:.1f} GB is above the stop line of {core.RSS_STOP_GB} GB (verdict row 0)"
        raise StoppedError(msg)
    dropped = already
    for arm in core.DROP_ORDER:
        if hours(dropped) <= core.STOP_LINE_HOURS:
            break
        if arm not in dropped:
            dropped = (*dropped, arm)
    counted = hours(tuple(dict.fromkeys((*already, *core.REPORT_ONLY_ARMS))))
    if counted > core.STOP_LINE_HOURS:
        msg = f"the counted work alone projects to {counted:.2f} GPU-process-hours, above the stop line of {core.STOP_LINE_HOURS} (verdict row 0)"
        raise StoppedError(msg)
    return {
        "pass": True, "code_version": CODE_VERSION, "projected_hours": round(hours(dropped), 3),
        "projected_counted_hours": round(counted, 3), "dropped": [a for a in dropped if a not in already],
        "not_fitted": list(already), "seconds_per_real_fit": seconds_per_real_fit,
        "seconds_per_synthetic_fit": seconds_per_synthetic_fit, "peak_gpu_gb": peak_gpu_gb,
        "peak_rss_gb": peak_rss_gb, "stop_line_hours": core.STOP_LINE_HOURS,
        "smoke_seconds": smoke_seconds, "timing_gate_seconds": timing_gate_seconds,
    }  # fmt: skip


def dropped_arms(*, root: Path) -> tuple[str, ...]:
    """Return the arms that the timing gate dropped or that are not fitted.

    (empty before the gate).
    """
    path = root / TIMING_FILE
    if not path.exists():
        return ()
    record = read_json(path=path)
    return (*record.get("dropped", ()), *record.get("not_fitted", ()))


def screen_stops_rest(*, point_relative: float) -> bool:
    """Return whether the screen rule stops the rest of FU and FU-S.

    The screen's point above +2% of M*'s CRPS at D8.
    """
    return point_relative > core.SCREEN_STOP_POINT


# --------------------------------------------------------------------------- the chain


@dataclass
class ChainSteps:
    """What the chain calls, so a test can replace each step.

    Attributes:
        run_job: Runs one fit (arm, fold, seed) and returns its record.
        f0r_gate: Checks the F0r forecasts against M* (raises `StoppedError` on a difference).
        screen_point: Returns the screen's relative CRPS point of FU minus M* (after the screen
            fits).
        log: Receives the progress lines.
        select_jobs: Keeps a subset of a family's jobs (a debug run on a scratch root; the default
            keeps every job).
        now: Returns the time stamp of a line.
    """

    run_job: Callable[[str, str | None, int], dict]
    f0r_gate: Callable[[], None]
    screen_point: Callable[[], float]
    log: Log = print
    select_jobs: Callable[
        [str, list[tuple[str, str | None, int]]], list[tuple[str, str | None, int]]
    ] = field(default=lambda _family, jobs: jobs)
    now: Callable[[], str] = field(
        default=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


def write_chain_status(*, root: Path, state: str, **extra: object) -> None:
    """Write `chain_status.json` (state running, complete, failed, or stopped)."""
    write_json(
        path=root / CHAIN_STATUS,
        payload={"state": state, "code_version": CODE_VERSION, "pid": os.getpid(), **extra},
    )


def run_chain(*, root: Path, steps: ChainSteps) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
    """Run the fit families in the plan's order and write the chain status.

    The order is F0r (then the F0r gate), FU on the screening folds, the screen rule, the rest of
    FU, FU-S,
    Y0, YU, and the report-only arms FS and YT. An arm that the timing gate dropped (or that equals
    FU) is
    skipped. A failed counted fit stops the chain at once; a failed report-only
    fit leaves the
    verdict untouched and the chain goes on. Each family sends a "fits starting" line first and a
    "last fit
    ended" line after.

    Args:
        root: The data folder.
        steps: The fit steps.

    Returns:
        The final status, `state` being "complete" when every fit succeeded, else "failed" or
        "stopped".
    """
    drop = set(dropped_arms(root=root))
    timing = read_json(path=root / TIMING_FILE) if (root / TIMING_FILE).exists() else {}
    per_fit = {
        "real": timing.get("seconds_per_real_fit", 0.0),
        "synthetic": timing.get("seconds_per_synthetic_fit", 0.0),
    }
    write_chain_status(root=root, state="running")
    failed: list[str] = []
    screen_stopped = False
    state = "complete"
    try:
        for family in FAMILIES:
            arm = "FU" if family.startswith("FU") and family != "FU-S" else family
            if arm in drop:
                steps.log(f"family {family} dropped by the timing gate or not fitted: skipped")
                continue
            if family in ("FU", "FU-S") and screen_stopped:
                steps.log(f"family {family} skipped by the screen rule")
                continue
            jobs = steps.select_jobs(family, family_jobs(family=family))
            where = core.ARMS[arm].where
            hours = len(jobs) * per_fit[where] / 3600.0
            steps.log(
                fit.fits_starting_line(
                    family=family, hours=hours, cores=core.SLOT_CORES, now=steps.now()
                )
            )
            counts = {"run": 0, "failed": 0}
            for job in jobs:
                try:
                    steps.run_job(*job)
                    (
                        root
                        / "fits"
                        / f"{fit.job_name(arm=job[0], fold=job[1], seed=job[2])}.failed.json"
                    ).unlink(missing_ok=True)
                    counts["run"] += 1
                except CorrectnessFailureError, StoppedError, GuardError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    counts["failed"] += 1
                    failed.append(fit.job_name(arm=job[0], fold=job[1], seed=job[2]))
                    write_json(
                        path=root
                        / "fits"
                        / f"{fit.job_name(arm=job[0], fold=job[1], seed=job[2])}.failed.json",
                        payload=fit.failure_payload(exc=exc),
                    )
                    steps.log(f"fit {job} failed: {type(exc).__name__}: {exc}")
                    if arm not in core.REPORT_ONLY_ARMS:
                        steps.log(f"counted fit {job} failed: the chain stops at once")
                        break
            steps.log(fit.last_fit_ended_line(family=family, counts=counts, now=steps.now()))
            if counts["failed"] and arm not in core.REPORT_ONLY_ARMS:
                state = "failed"
                break
            if family == "F0r":
                steps.f0r_gate()
            if family == "FU-screen":
                point = steps.screen_point()
                write_json(
                    path=root / SCREEN_FILE,
                    payload={
                        "point_relative": point,
                        "stops_rest": screen_stops_rest(point_relative=point),
                        "interim": True,
                    },
                )
                screen_stopped = screen_stops_rest(point_relative=point)
    except StoppedError as exc:
        write_chain_status(root=root, state="stopped", reason=str(exc), failed=failed)
        return read_json(path=root / CHAIN_STATUS)
    except CorrectnessFailureError as exc:
        record_invalid(root=root, rule=str(exc))
        write_chain_status(
            root=root,
            state="failed",
            reason="correctness failure",
            traceback=traceback.format_exc(),
            failed=failed,
        )
        return read_json(path=root / CHAIN_STATUS)
    except GuardError as exc:
        write_chain_status(
            root=root,
            state="failed",
            reason=f"guard: {exc}",
            traceback=traceback.format_exc(),
            failed=failed,
        )
        raise
    except Exception as exc:  # noqa: BLE001
        write_chain_status(
            root=root,
            state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            failed=failed,
        )
        return read_json(path=root / CHAIN_STATUS)
    write_chain_status(root=root, state=state, failed=failed, screen_stopped=screen_stopped)
    return read_json(path=root / CHAIN_STATUS)


def exit_code(  # noqa: PLR0911
    *, exc: BaseException | None = None, status: Mapping[str, Any] | None = None
) -> int:
    """Return the exit code of a stage or a chain: stopped 3, invalid 4, a failed chain 5.

    A guard refusal 2, else 0. Never 1.
    """
    if exc is not None:
        if isinstance(exc, StoppedError):
            return EXIT_STOPPED
        if isinstance(exc, CorrectnessFailureError):
            return EXIT_INVALID
        if isinstance(exc, GuardError):
            return EXIT_GUARD
        return EXIT_FAILED
    if status is not None:
        state = status.get("state")
        if state == "failed" and status.get("reason") == "correctness failure":
            return EXIT_INVALID
        return (
            EXIT_OK if state == "complete" else EXIT_STOPPED if state == "stopped" else EXIT_FAILED
        )
    return EXIT_OK
