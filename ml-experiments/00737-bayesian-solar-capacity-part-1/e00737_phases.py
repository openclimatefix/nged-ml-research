"""The guards that decide whether a stage of experiment 737 may start.

No stage runs before its review marker exists, and a marker holds the hash of the code that was
reviewed. This module holds the code hash, the review marker and the frozen-code check, the gate and
tuning records that the bulk stages need, the CPU-hour budget, the done markers of finished tasks,
and the task and shard runners. `e00737_run.py` is the runner that calls it.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Final

from e00737_common import EXIT_FAILED, OUT, Tuned, atomic_json, sha256_of

CODE_SHA: Final[Path] = OUT / "code.sha256"
"""Hash of the scoring code, written by `freeze-code`."""
TUNED_SHA_NAME: Final[str] = "tuned.sha256"
"""The sidecar that holds the SHA-256 of `tuned.json`, written by `tune-collect`."""
MARKER_REVIEW: Final[str] = "review_stage1.ok"
BUDGET_MARKER: Final[str] = "cpu_budget_extended.ok"
"""Written by the manager to allow a run past `STOP_CPU_HOURS`."""
STOP_CPU_HOURS: Final[float] = 26.0
BLAS_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
"""The variables that `run.sh` sets to 1, one thread per numerical library."""
GATES_REQUIRED: Final[tuple[str, ...]] = ("G1", "G1b", "G2", "G3", "G4")
"""The gates of `gates.json` that must hold before the bulk fits (G5 is read at scoring)."""
GATES_BEFORE_TUNING: Final[tuple[str, ...]] = ("G2", "G3", "G4")
"""The gates that must hold before the tuning. G1 and G1b run after it, with the tuned values."""


def assert_blas_pinned() -> None:
    """Stop unless every numerical library is limited to one thread by the environment."""
    wrong = {k: os.environ.get(k) for k in BLAS_VARIABLES if os.environ.get(k) != "1"}
    if wrong:
        sys.exit(f"refusing to run: these variables must equal 1 (use run.sh): {wrong}")


def code_hash(*, folder: Path | None = None) -> str:
    """Return the SHA-256 over the names and bytes of every non-test module and `run*.sh` script.

    Args:
        folder: The folder to hash (this file's folder by default).

    Returns:
        The hex digest.
    """
    here = folder if folder is not None else Path(__file__).parent
    h = hashlib.sha256()
    for f in [*sorted(here.glob("e00737_*.py")), *sorted(here.glob("run*.sh"))]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def marker_hash(*, marker: str) -> str | None:
    """Return the code hash a marker file holds, or None when the file does not exist.

    Args:
        marker: The marker's file name under `OUT`.

    Returns:
        The first word of the file.
    """
    path = OUT / marker
    text = path.read_text().split() if path.exists() else []
    return text[0] if text else None


def require_review(*, marker: str) -> None:
    """Stop unless the review marker holds the hash of the current code.

    Args:
        marker: The marker's file name under `OUT`.
    """
    held = marker_hash(marker=marker)
    if held is None:
        sys.exit(f"refusing to run: {OUT / marker} is missing (the review has not passed)")
    if held != code_hash():
        sys.exit(
            f"refusing to run: {OUT / marker} holds the hash of different code (run review-mark)"
        )


def require_code_frozen() -> None:
    """Stop unless the code hash equals the one frozen with the review."""
    if not CODE_SHA.exists() or code_hash() != CODE_SHA.read_text().split()[0]:
        sys.exit("refusing to run: the code differs from code.sha256 (run freeze-code)")


def read_record(*, name: str) -> dict:
    """Read a JSON record under `OUT`, or stop when it is missing or written by other code.

    Args:
        name: The record's file name.

    Returns:
        The record.
    """
    path = OUT / name
    if not path.exists():
        sys.exit(f"refusing to run: {path} does not exist (run the stage that writes it)")
    record = json.loads(path.read_text())
    if record.get("code_hash") != code_hash():
        sys.exit(f"refusing to run: {name} was written by different code (rerun its stage)")
    return record


def require_gates(*, required: tuple[str, ...] = GATES_REQUIRED) -> None:
    """Stop unless `gates.json` records a pass for the gates named and for the other checks.

    Args:
        required: The gates that must have passed (all of `GATES_REQUIRED` by default).
    """
    record = read_record(name="gates.json")
    failed = [g for g in required if record.get(g, {}).get("passed") is not True]
    failed.extend(
        check
        for check in ("attributes", "known_answers", "quarters")
        if record.get(check, {}).get("passed") is not True
    )
    if failed:
        sys.exit(f"refusing to run: gates {failed} have not passed")


def require_tuned() -> Tuned:
    """Stop unless `tuned.json` holds the values chosen by the current code, with its digest.

    The digest in `tuned.sha256` must equal the SHA-256 of the file, so a hand edit of the chosen
    values stops every stage that reads them.

    Returns:
        The chosen growth standard deviation and ridge scale.
    """
    record = read_record(name="tuned.json")
    digest_file = OUT / TUNED_SHA_NAME
    held = digest_file.read_text().split()[0] if digest_file.exists() else None
    if held != sha256_of(path=OUT / "tuned.json"):
        sys.exit(
            f"refusing to run: tuned.json does not match {TUNED_SHA_NAME} (rerun tune-collect)"
        )
    return Tuned(growth_sd=float(record["growth_sd"]), ridge_scale=float(record["ridge_scale"]))


def require_ridge_scale() -> float:
    """Stop unless `ridge.json` holds the ridge scale chosen by the current code.

    Returns:
        The chosen ridge scale.
    """
    return float(read_record(name="ridge.json")["ridge_scale"])


def require_unit() -> dict:
    """Stop unless `unit_timing.json` holds the first-unit projection of the current code.

    Returns:
        The record.
    """
    return read_record(name="unit_timing.json")


def spent_cpu_hours(*, log: Path | None = None, stage: str | None = None) -> float:
    """Return the CPU-hours in the per-phase time log (one thread per task).

    Args:
        log: The log file (`OUT/time_log.csv` by default).
        stage: Count only this phase when given.

    Returns:
        Hours between each line's start and end, summed.
    """
    path = log if log is not None else OUT / "time_log.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5 and (stage is None or parts[0] == stage):
            a, b = (calendar.timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%S")) for v in parts[3:5])
            total += (b - a) / 3600.0
    return total


def require_budget(*, check_projection: bool = False) -> None:
    """Stop at the CPU-hour line, spent or projected by the first-unit stage.

    The marker `cpu_budget_extended.ok` in `OUT` lets a run past the line (the manager writes it).
    A refusal raises, so that the runner writes a failure record and exits with code 3.

    Args:
        check_projection: Also stop when the first-unit projection is over the line.

    Raises:
        RuntimeError: When the spent or projected CPU-hours are over the stop.
    """
    if (OUT / BUDGET_MARKER).exists():
        return
    spent = spent_cpu_hours()
    if spent > STOP_CPU_HOURS:
        msg = (
            f"refusing to run: {spent:.1f} CPU-hours spent, over the {STOP_CPU_HOURS:.0f} stop "
            f"(ask the manager; marker {BUDGET_MARKER})"
        )
        raise RuntimeError(msg)
    if check_projection:
        projected = float(require_unit()["projected_total_cpu_hours"])
        if projected > STOP_CPU_HOURS:
            msg = (
                f"refusing to run: the first unit projects {projected:.1f} CPU-hours, over the "
                f"{STOP_CPU_HOURS:.0f} stop (ask the manager; marker {BUDGET_MARKER})"
            )
            raise RuntimeError(msg)


def require_finished(*, names: list[str]) -> None:
    """Stop unless every task has a done marker of the current code and `OUT/failed/` is empty.

    Args:
        names: The names of the tasks that must have finished.

    Raises:
        RuntimeError: When a done marker is missing or holds another code hash, or a failure
            record exists.
    """
    here = code_hash()
    unfinished = [
        n
        for n in names
        if not (OUT / "done" / f"{n}.ok").exists()
        or not (OUT / "done" / f"{n}.ok").read_text().startswith(f"code_hash={here} ")
    ]
    failures = (
        sorted(p.name for p in (OUT / "failed").glob("*.json")) if (OUT / "failed").exists() else []
    )
    if unfinished or failures:
        msg = (
            f"refusing to score: {len(unfinished)} of {len(names)} tasks have no done marker for "
            f"the current code (such as {unfinished[:3]}), and {len(failures)} failure records "
            f"exist (such as {failures[:3]})"
        )
        raise RuntimeError(msg)


def shard(*, items: list, worker: int, of: int) -> list:
    """Return the items of worker `worker` out of `of` workers.

    Args:
        items: Every item.
        worker: This worker's index.
        of: The number of workers.

    Returns:
        This worker's items.
    """
    return items[worker::of]


def run_task(*, name: str, fn: Callable[..., None], **kwargs: object) -> bool:
    """Run one task unless its done marker exists; record a failure and return False on error.

    A finished task writes `OUT/done/<name>.ok`, holding the code hash, the time, and the peak
    resident memory of this process. A re-run after a crash skips every task whose marker holds
    the current code hash, and runs again a task whose marker holds another hash.

    Args:
        name: The task's name, unique within the experiment.
        fn: The task.
        **kwargs: The task's keyword arguments.

    Returns:
        True when the task finished now or earlier, False when it failed.
    """
    done = OUT / "done" / f"{name}.ok"
    here = code_hash()
    if done.exists() and done.read_text().startswith(f"code_hash={here} "):
        print(f"skip {name}: already done", flush=True)
        return True
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        atomic_json(
            obj={"task": name, "message": str(exc), "traceback": traceback.format_exc()},
            path=OUT / "failed" / f"{name}.json",
        )
        print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
        return False
    done.parent.mkdir(parents=True, exist_ok=True)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    stamp = time.strftime("%FT%TZ", time.gmtime())
    done.write_text(f"code_hash={here} {stamp} peak_rss_gb={peak_gb:.2f}\n")
    print(f"peak resident memory of this worker so far: {peak_gb:.2f} GB", flush=True)
    return True


def run_shard(*, names_and_kwargs: list[tuple[str, dict]], fn: Callable[..., None]) -> None:
    """Run a worker's tasks in turn, stopping at the CPU budget; exit with code 3 on any failure.

    Args:
        names_and_kwargs: Each task's name and keyword arguments.
        fn: The task function.
    """
    failed = 0
    for name, kwargs in names_and_kwargs:
        require_budget()
        failed += not run_task(name=name, fn=fn, **kwargs)
    if failed:
        sys.exit(EXIT_FAILED)


def write_stage_failure(*, stage: str, worker: int, exc: BaseException) -> None:
    """Write a failed stage's message and traceback under `OUT/failed/`.

    Args:
        stage: The stage's name.
        worker: The worker's index.
        exc: The exception that ended the stage.
    """
    atomic_json(
        obj={
            "stage": stage,
            "worker": worker,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
        path=OUT / "failed" / f"stage_{stage}_worker_{worker}.json",
    )
