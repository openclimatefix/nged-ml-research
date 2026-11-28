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
import shutil
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


FIT_MODULES: Final[tuple[str, ...]] = (
    "common",
    "survival",
    "world",
    "model",
    "train",
    "fit",
    "panel",
)
"""The modules whose code decides what a fitted state holds."""
FORECAST_MODULES: Final[tuple[str, ...]] = (*FIT_MODULES, "predict")
"""The modules whose code decides what a forecast file holds."""
RECIPE_SUFFIX: Final[str] = ".recipe"
WAIT_PREFIX: Final[str] = "wait_"
"""A ledger task that starts with this prefix is a wait, which the stop line does not count."""


def recipe_hash(*, modules: tuple[str, ...], folder: Path | None = None) -> str:
    """Return the SHA-256 over the names and bytes of the named `e00738_<name>.py` modules.

    Args:
        modules: The module names without the `e00738_` prefix.
        folder: The folder of the modules (this file's folder by default).

    Returns:
        The hex digest.
    """
    here = folder if folder is not None else Path(__file__).parent
    h = hashlib.sha256()
    for name in modules:
        f = here / f"e00738_{name}.py"
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def recipe_path(*, path: Path) -> Path:
    """Return the sidecar file that holds the recipe hash of a state or forecast file."""
    return path.with_name(path.name + RECIPE_SUFFIX)


def file_digest(*, path: Path) -> str:
    """Return the SHA-256 of a file's bytes, or `none` when the file does not exist."""
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(*, arm: str, stem: str, out: Path | None = None) -> str:
    """Return the digest of the state that an arm's forecasts of one fold and seed come from.

    Args:
        arm: The arm.
        stem: `<month>_s<seed>`.
        out: The output folder (`OUT` by default).

    Returns:
        The digest of the stored stage 3 state of `M1` (experiment 713's, read only), or of the
        fitted
        state of any other arm.
    """
    if arm == common.ARM_M1:
        return file_digest(path=common.ROOT_713 / "work" / "models" / "M1" / f"{stem}_stage3.pt")
    root = out if out is not None else common.OUT
    return file_digest(path=root / "networks" / arm / f"{stem}.pt")


def write_recipe(
    *,
    path: Path,
    modules: tuple[str, ...],
    folder: Path | None = None,
    link: str | None = None,
) -> None:
    """Write a file's sidecar atomically: the recipe hash, then the digest of its source state.

    Callers write the sidecar before the file, so a file that exists always has a sidecar.
    """
    side = recipe_path(path=path)
    side.parent.mkdir(parents=True, exist_ok=True)
    tmp = side.with_name(f".{side.name}.{os.getpid()}.tmp")
    tmp.write_text(recipe_hash(modules=modules, folder=folder) + "\n" + (link or "") + "\n")
    tmp.rename(side)


def recipe_link(*, path: Path) -> str:
    """Return the state digest that a sidecar records (`none` when it records no state)."""
    side = recipe_path(path=path)
    lines = side.read_text().splitlines() if side.exists() else []
    return lines[1] if len(lines) > 1 and lines[1] else "none"


def recipe_matches(*, path: Path, modules: tuple[str, ...], folder: Path | None = None) -> bool:
    """Return whether the sidecar of a file holds the current recipe hash of the modules."""
    side = recipe_path(path=path)
    lines = side.read_text().splitlines() if side.exists() else []
    return bool(lines) and lines[0].strip() == recipe_hash(modules=modules, folder=folder)


def require_recipe(*, path: Path, modules: tuple[str, ...], folder: Path | None = None) -> None:
    """Raise unless the file's recipe hash is the current one.

    Raises:
        RuntimeError: When the sidecar is missing or holds another hash.
    """
    if not recipe_matches(path=path, modules=modules, folder=folder):
        msg = (
            f"refusing to use {path}: it was written by other fitting code than the current "
            "code (or has no recipe file). Delete it and its dependants, and rerun."
        )
        raise RuntimeError(msg)


def require_all_recipes(*, out: Path | None = None, folder: Path | None = None) -> int:
    """Check every state file and forecast file under the output folder, and return how many.

    A forecast also records the digest of the state it was made from, and the check compares it with
    the digest of the state that exists now.

    Args:
        out: The output folder (`OUT` by default).
        folder: The code folder (this file's folder by default).

    Raises:
        RuntimeError: When any file has no recipe file or another recipe hash, or a forecast came
            from another state.
    """
    root = out if out is not None else common.OUT
    n = 0
    for pattern, modules in (
        ("networks/*/*.pt", FIT_MODULES),
        ("predictions/*/*/*.parquet", FORECAST_MODULES),
    ):
        for path in sorted(root.glob(pattern)):
            require_recipe(path=path, modules=modules, folder=folder)
            if pattern.startswith("predictions"):
                want = state_digest(arm=path.parent.parent.name, stem=path.parent.name, out=root)
                if recipe_link(path=path) != want:
                    msg = (
                        f"refusing to use {path}: it was made from another state of "
                        f"{path.parent.parent.name} than the one now in {root / 'networks'}"
                    )
                    raise RuntimeError(msg)
            n += 1
    return n


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
        Hours between each line's start and end, summed. A crashed attempt has a line, so it
        counts. A wait for another session's GPU process (a task named `wait_...`) does not
        count, because the stop line limits the hours this experiment works, and `waited_hours`
        reports the waits.
    """
    path = log if log is not None else common.OUT / "time_log.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if (
            len(parts) == 5
            and parts[2] == device
            and (stage is None or parts[0] == stage)
            and not parts[1].startswith(WAIT_PREFIX)
        ):
            a, b = (timegm(time.strptime(v, "%Y-%m-%dT%H:%M:%S")) for v in parts[3:5])
            total += (b - a) / 3600.0
    return total


def waited_hours(*, device: str = "gpu", log: Path | None = None) -> float:
    """Return the hours the ledger records as waiting for another session's process."""
    path = log if log is not None else common.OUT / "time_log.csv"
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) == 5 and parts[2] == device and parts[1].startswith(WAIT_PREFIX):
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


def require_gpu_free(*, busy: Callable[[], bool] | None = None) -> None:
    """Exit with code 75 when another session's process is on the GPU, before any job starts.

    A launcher calls this once. Between the jobs of one chain the chain waits instead
    (`wait_for_gpu`), which prints each wait and writes it to the ledger apart from the work.

    Args:
        busy: Whether the GPU is busy (`gpu_busy` by default; replaced in tests).
    """
    if (busy if busy is not None else gpu_busy)():
        print("refusing to start: the GPU is busy (exit 75)", file=sys.stderr, flush=True)
        sys.exit(common.EXIT_GPU_BUSY)


def wait_for_gpu(
    *,
    stage: str,
    task: str,
    busy: Callable[[], bool] = gpu_busy,
    sleep: Callable[[float], None] = time.sleep,
    max_waits: int | None = None,
    max_seconds: float | None = None,
) -> int:
    """Wait while another session uses the GPU, and write each wait to the ledger as a wait.

    The stop line does not count a wait, and each wait is printed.

    Args:
        stage: The phase for the ledger.
        task: The task for the ledger.
        busy: Whether the GPU is busy.
        sleep: The sleep function (replaced in tests).
        max_waits: Give up after this many back-offs (None to wait for ever).
        max_seconds: Exit with code 75 when the waits of this call reach this many seconds.

    Returns:
        The number of back-offs, or -1 if it gave up.
    """
    n = 0
    while busy():
        if max_waits is not None and n >= max_waits:
            return -1
        if max_seconds is not None and n * common.GPU_BACKOFF_SECONDS >= max_seconds:
            print("giving up: the GPU stayed busy for the whole wait ceiling", file=sys.stderr)
            sys.exit(common.EXIT_GPU_BUSY)
        start = time.time()
        print(
            f"waiting for the GPU before {task}: another session's process is on it "
            f"(wait {n + 1}, {common.GPU_BACKOFF_SECONDS} s)",
            flush=True,
        )
        sleep(common.GPU_BACKOFF_SECONDS)
        common.time_log(stage=stage, task=f"{WAIT_PREFIX}{task}", start=start, device="gpu")
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


def archive_failures(*, out: Path | None = None, folder_code: Path | None = None) -> list[str]:
    """Move the failure records of tasks that have since finished into `failed/archive/`.

    A task that failed once and finished on a rerun leaves a record behind, and `require_finished`
    refuses while any record is left in `failed/`. The archive keeps the record and the traceback.

    Args:
        out: The output folder (`OUT` by default).
        folder_code: The code folder whose hash a done marker must hold (this file's by default).

    Returns:
        The names of the tasks whose records were moved.
    """
    root = out if out is not None else common.OUT
    folder = root / "failed"
    moved: list[str] = []
    if not folder.exists():
        return moved
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for path in sorted(folder.glob("*.json")):
        task = json.loads(path.read_text()).get("task")
        if task is not None and is_done(name=task, out=root, folder=folder_code):
            (folder / "archive").mkdir(exist_ok=True)
            shutil.move(path, folder / "archive" / f"{path.stem}_{stamp}.json")
            moved.append(task)
    return moved


def archive_stage_failures(*, stage: str, out: Path | None = None) -> None:
    """Move the failure records of a stage that has now finished into `failed/archive/`."""
    root = out if out is not None else common.OUT
    folder = root / "failed"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for path in sorted(folder.glob(f"stage_{stage}_worker_*.json")) if folder.exists() else []:
        (folder / "archive").mkdir(exist_ok=True)
        shutil.move(path, folder / "archive" / f"{path.stem}_{stamp}.json")


def child_env(
    *, month: str | None = None, phase: str | None = None, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment of a child process that runs one fit or predict job.

    Experiment 713's pipeline reads `E00667_MONTH` and `E00667_PHASE` once, when it is imported, so
    a job runs in a child process whose environment already holds its own month and phase.

    Args:
        month: The fold month.
        phase: `train` or `predict`.
        extra: More variables for the child only (such as the smoke test's pass cap).

    Returns:
        A copy of this process's environment with the extra variables set.
    """
    env = dict(os.environ)
    if month is not None:
        env["E00667_MONTH"] = month
    if phase is not None:
        env["E00667_PHASE"] = phase
    env.update(extra or {})
    return env


def run_child(*, argv: list[str], env: dict[str, str]) -> None:
    """Run a command in a child process and raise when it fails, with the tail of its stderr.

    Args:
        argv: The command.
        env: The child's environment.

    Raises:
        RuntimeError: When the child exits with a non-zero code.
    """
    done = subprocess.run(argv, env=env, stderr=subprocess.PIPE, text=True, check=False)
    if done.stderr:
        print(done.stderr, file=sys.stderr, end="", flush=True)
    if done.returncode != 0:
        msg = f"child process {argv[1:]} exited with code {done.returncode}: {done.stderr[-2000:]}"
        raise RuntimeError(msg)


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
    peak_gb = (
        max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        / 1024**2
    )
    path = done_path(name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%FT%TZ", time.gmtime())
    path.write_text(f"code_hash={code_hash(folder=folder)} {stamp} peak_rss_gb={peak_gb:.2f}\n")
    print(f"peak resident memory of this worker so far: {peak_gb:.2f} GB", flush=True)
    archive_failures(folder_code=folder)
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
    waited = 0.0
    for name, kwargs in jobs:
        require_budget(device=device)
        if device == "gpu" and not is_done(name=name):
            n = wait_for_gpu(
                stage=stage, task=name, max_seconds=common.MAX_GPU_WAIT_SECONDS - waited
            )
            waited += n * common.GPU_BACKOFF_SECONDS
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
    archive_failures(out=root)
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
