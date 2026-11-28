"""The guards that decide whether a stage of experiment 738 may start, and the task runners.

No stage runs before the marker of the stage before it exists, and a review marker holds the hash of
the code that was reviewed. This module holds the code hash, the review and freeze markers, the
done markers (each with the code hash and the peak resident memory), the hour ledgers with the stop
lines, the GPU-busy check, and the task and shard runners. `e00738_run.py` calls it.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from calendar import timegm
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00738_common as common

MARKER_REVIEW: Final[str] = "review_1.ok"
CODE_SHA_NAME: Final[str] = "code.sha256"
BLAS_VARIABLES: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
STAGE_ORDER: Final[tuple[str, ...]] = (
    "phase0",
    "review",
    "smoke",
    "screen_fits",
    "gate",
    "screen_score",
    "full_fits",
    "predict",
    "score",
    "report",
    "figures",
)
"""The stages in run order. A stage needs the marker of the stage before it."""
STAGE_MARKERS: Final[dict[str, str]] = {
    "phase0": "phase0.done",
    "review": MARKER_REVIEW,
    "smoke": "smoke.ok",
    "screen_fits": "screen_fits.done",
    "gate": "gate_passed.json",
    "screen_score": "screen.done",
    "full_fits": "full_fits.done",
    "predict": "predict.done",
    "score": "score.done",
    "report": "report.done",
    "figures": "figures.done",
}


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
    for f in [*sorted(here.glob("e00738_*.py")), *sorted(here.glob("run*.sh"))]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def marker_hash(*, marker: str, out: Path | None = None) -> str | None:
    """Return the code hash that a marker file holds, or None when it does not exist.

    Args:
        marker: The marker's file name under the output folder.
        out: The output folder (`OUT` by default).

    Returns:
        The first word of the file, a JSON marker's `code_hash` field, or None.
    """
    path = (out if out is not None else common.OUT) / marker
    if not path.exists():
        return None
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text).get("code_hash")
    words = text.split()
    return words[0].removeprefix("code_hash=") if words else None


def require_review(*, out: Path | None = None, folder: Path | None = None) -> None:
    """Stop unless the review marker holds the hash of the current code.

    Args:
        out: The output folder (`OUT` by default).
        folder: The code folder (this file's folder by default).
    """
    root = out if out is not None else common.OUT
    held = marker_hash(marker=MARKER_REVIEW, out=root)
    if held is None:
        sys.exit(f"refusing to run: {root / MARKER_REVIEW} is missing (the review has not passed)")
    if held != code_hash(folder=folder):
        sys.exit(
            f"refusing to run: {MARKER_REVIEW} holds the hash of different code "
            "(re-review the diff)"
        )


def write_review_marker(*, out: Path | None = None, folder: Path | None = None) -> str:
    """Write the review marker with the current code hash and return the hash.

    Args:
        out: The output folder (`OUT` by default).
        folder: The code folder (this file's folder by default).

    Returns:
        The hash.
    """
    root = out if out is not None else common.OUT
    h = code_hash(folder=folder)
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER_REVIEW).write_text(f"{h} {time.strftime('%FT%TZ', time.gmtime())}\n")
    return h


def require_stage(*, stage: str, out: Path | None = None, folder: Path | None = None) -> None:
    """Stop unless the previous stage's marker exists and holds the current code hash.

    Args:
        stage: A key of `STAGE_MARKERS`.
        out: The output folder (`OUT` by default).
        folder: The code folder (this file's folder by default).
    """
    root = out if out is not None else common.OUT
    i = STAGE_ORDER.index(stage)
    if i == 0:
        return
    before = STAGE_ORDER[i - 1]
    marker = STAGE_MARKERS[before]
    if not (root / marker).exists():
        sys.exit(f"refusing to start {stage}: {marker} of the stage {before} is missing")
    if stage != "review":
        require_review(out=root, folder=folder)


def done_path(*, name: str, out: Path | None = None) -> Path:
    """Return the done marker of a task."""
    return (out if out is not None else common.OUT) / "done" / f"{name}.ok"


def is_done(*, name: str, out: Path | None = None, folder: Path | None = None) -> bool:
    """Return whether the task's done marker exists and holds the current code hash."""
    path = done_path(name=name, out=out)
    return path.exists() and path.read_text().startswith(f"code_hash={code_hash(folder=folder)} ")


def spent_hours(*, device: str, log: Path | None = None, stage: str | None = None) -> float:
    """Return the hours of one device in the per-phase time log.

    Args:
        device: `cpu` or `gpu`.
        log: The log file (`OUT/time_log.csv` by default).
        stage: Count only this phase when given.

    Returns:
        Hours between each line's start and end, summed. A crashed attempt and a wait under
        contention each have a line, so both count.
    """
    path = log if log is not None else common.OUT / "time_log.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5 and parts[2] == device and (stage is None or parts[0] == stage):
            a, b = (timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%S")) for v in parts[3:5])
            total += (b - a) / 3600.0
    return total


def require_budget(*, device: str, out: Path | None = None, log: Path | None = None) -> None:
    """Stop at the hour line of the device, unless the manager wrote `budget_extended.ok`.

    The launcher calls this between fits, so a run stops between fits and never inside one.

    Args:
        device: `cpu` or `gpu`.
        out: The output folder (`OUT` by default).
        log: The time log (`OUT/time_log.csv` by default).

    Raises:
        RuntimeError: When the hours spent are over the stop.
    """
    root = out if out is not None else common.OUT
    if (root / common.BUDGET_MARKER).exists():
        return
    line = common.STOP_GPU_HOURS if device == "gpu" else common.STOP_CPU_HOURS
    spent = spent_hours(device=device, log=log if log is not None else root / "time_log.csv")
    if spent > line:
        msg = (
            f"refusing to run: {spent:.2f} {device} hours spent, over the {line:.1f} stop "
            f"(ask the manager; marker {common.BUDGET_MARKER})"
        )
        raise RuntimeError(msg)


def other_compute_processes(*, listing: str, own_pids: set[int]) -> list[int]:
    """Return the pids in an `nvidia-smi --query-compute-apps=pid` listing that are not ours.

    Args:
        listing: The command's output, one pid per line.
        own_pids: The pids of this session's own processes.

    Returns:
        The other pids.
    """
    pids = [int(w) for w in listing.split() if w.strip().isdigit()]
    return [p for p in pids if p not in own_pids]


def gpu_busy() -> bool:
    """Return whether `nvidia-smi` lists a compute process of another session."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired:
        return False
    return bool(other_compute_processes(listing=listing, own_pids={os.getpid(), os.getppid()}))


def wait_for_gpu(
    *,
    stage: str,
    task: str,
    busy: Callable[[], bool] = gpu_busy,
    sleep: Callable[[float], None] = time.sleep,
    max_waits: int | None = None,
) -> int:
    """Wait while another session uses the GPU, and write each wait to the ledger as GPU time.

    Args:
        stage: The phase for the ledger.
        task: The task for the ledger.
        busy: Whether the GPU is busy.
        sleep: The sleep function (replaced in tests).
        max_waits: Give up after this many back-offs (None to wait for ever).

    Returns:
        The number of back-offs, or -1 if it gave up.
    """
    n = 0
    while busy():
        if max_waits is not None and n >= max_waits:
            return -1
        start = time.time()
        sleep(common.GPU_BACKOFF_SECONDS)
        common.time_log(stage=stage, task=f"wait_{task}", start=start, device="gpu")
        n += 1
    return n


def write_failure(*, name: str, exc: BaseException, kind: str = "task") -> None:
    """Write a failed task's or stage's message and traceback under `OUT/failed/`.

    Args:
        name: The task or stage name.
        exc: The exception.
        kind: `task` or `stage`.
    """
    common.atomic_json(
        obj={
            kind: name,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
        path=common.OUT / "failed" / f"{kind}_{name}.json",
    )


def run_task(
    *,
    name: str,
    fn: Callable[..., None],
    kwargs: dict[str, object],
    device: str = "cpu",
    stage: str = "task",
    folder: Path | None = None,
) -> bool:
    """Run one task unless its done marker holds the current code hash; record a failure on error.

    The task's keyword arguments travel in the `kwargs` dict, so a task may take a keyword named
    `name`, `fn`, `device`, or `stage` without a collision. A finished task writes
    `OUT/done/<name>.ok` with the code hash, the time, and this process's peak resident memory so
    far.
    A crashed attempt still writes its time to the ledger.

    Args:
        name: The task's name, unique within the experiment.
        fn: The task.
        kwargs: The task's keyword arguments.
        device: `cpu` or `gpu`, for the ledger.
        stage: The phase for the ledger.
        folder: The code folder whose hash the marker holds (this file's folder by default).

    Returns:
        True when the task finished now or earlier, False when it failed.
    """
    if is_done(name=name, folder=folder):
        print(f"skip {name}: already done", flush=True)
        return True
    start = time.time()
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure record is the point of this boundary
        write_failure(name=name, exc=exc)
        print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
        return False
    finally:
        common.time_log(stage=stage, task=name, start=start, device=device)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    path = done_path(name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%FT%TZ", time.gmtime())
    path.write_text(f"code_hash={code_hash(folder=folder)} {stamp} peak_rss_gb={peak_gb:.2f}\n")
    print(f"peak resident memory of this worker so far: {peak_gb:.2f} GB", flush=True)
    return True


def shard(*, items: list, worker: int, of: int) -> list:
    """Return the items of worker `worker` out of `of` workers."""
    return items[worker::of]


def run_chain(
    *,
    jobs: list[tuple[str, dict[str, object]]],
    fn: Callable[..., None],
    device: str,
    stage: str,
    exit_code: int = common.EXIT_FAILED,
) -> None:
    """Run a chain of tasks in turn, stopping at the hour line, and exit on any failure.

    Args:
        jobs: Each task's name and keyword arguments.
        fn: The task function.
        device: `cpu` or `gpu`.
        stage: The phase for the ledger.
        exit_code: The code to exit with when any task failed (3 for a CPU stage, 5 for a GPU
        chain).
    """
    failed = 0
    for name, kwargs in jobs:
        require_budget(device=device)
        if device == "gpu" and not is_done(name=name):
            wait_for_gpu(stage=stage, task=name)
        failed += not run_task(name=name, fn=fn, kwargs=kwargs, device=device, stage=stage)
    if failed:
        sys.exit(exit_code)


def require_finished(*, names: list[str], out: Path | None = None) -> None:
    """Stop unless every task has a done marker of the current code and `OUT/failed/` is empty.

    Args:
        names: The names of the tasks that must have finished.
        out: The output folder (`OUT` by default).

    Raises:
        RuntimeError: When a done marker is missing or holds another code hash, or a failure
            record exists.
    """
    root = out if out is not None else common.OUT
    unfinished = [n for n in names if not is_done(name=n, out=root)]
    failures = (
        sorted(p.name for p in (root / "failed").glob("*.json"))
        if (root / "failed").exists()
        else []
    )
    if unfinished or failures:
        msg = (
            f"refusing to continue: {len(unfinished)} of {len(names)} tasks have no done marker "
            "for "
            f"the current code (such as {unfinished[:3]}), and {len(failures)} failure records "
            f"exist (such as {failures[:3]})"
        )
        raise RuntimeError(msg)
