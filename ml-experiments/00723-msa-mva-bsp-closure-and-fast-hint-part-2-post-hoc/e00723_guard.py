# ruff: noqa: E501
"""Experiment 723: where the outputs go, the anonymisation guard, and the run log.

The guard, the order-of-reads log, and the timing gate are experiment 722's (`e00722_guard`),
imported and not copied. This module sets experiment 723's own output folders, writes every file
atomically, and writes a run log that holds a fingerprint of each input table.

**Nothing dated leaves `PRIVATE/`, and a generator's output lives in `PRIVATE/` only.** The public
writers refuse a frame, a value, or a printed line that holds a date, and refuse a column that
names a generator's identifier. Farms carry the labels F1 to F6.
"""

import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import e00722_guard as guard722  # ty: ignore[unresolved-import]
import e00723_core as core
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

AnonymisationError = guard722.AnonymisationError
ReadLog = guard722.ReadLog
TimingGate = guard722.TimingGate
TimingGateError = guard722.TimingGateError
say = guard722.say
atomic_write = guard722.atomic_write
file_fingerprint = guard722.file_fingerprint

EXPERIMENT_FOLDER: Final[str] = "00723-msa-mva-bsp-closure-and-fast-hint-part-2-post-hoc"
"""The experiment's folder name under `data/ml-experiments/`."""

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_FOLDER
"""Where every aggregate output goes."""

PRIVATE_DIR: Final[Path] = OUT_DIR / "PRIVATE"
"""Where anything dated, and every generator's output, goes."""

OUT_DIR_722: Final[Path] = guard722.OUT_DIR
"""Experiment 722's output folder, which this experiment reads."""

PRIVATE_DIR_722: Final[Path] = guard722.PRIVATE_DIR
"""Experiment 722's private folder, which this experiment reads."""

FOLDER: Final[Path] = Path(__file__).resolve().parent
"""This experiment's code folder."""


def write_public_csv(*, frame: pl.DataFrame, name: str, out_dir: Path = OUT_DIR) -> Path:
    """Write an aggregate table to `out_dir/name` after the guard checks it."""
    return guard722.write_public_csv(frame=frame, name=name, out_dir=out_dir)


def write_public_json(*, payload: Mapping[str, Any], name: str, out_dir: Path = OUT_DIR) -> Path:
    """Write a JSON file to `out_dir/name`; refuses a value that holds a calendar date."""
    return guard722.write_public_json(payload=payload, name=name, out_dir=out_dir)


def write_private_parquet(
    *, frame: pl.DataFrame, name: str, private_dir: Path = PRIVATE_DIR
) -> Path:
    """Write dated detail or a generator's output under `PRIVATE/`."""
    return guard722.write_private_parquet(frame=frame, name=name, private_dir=private_dir)


def invalidate_outputs(*, names: Sequence[str], out_dir: Path = OUT_DIR) -> list[str]:
    """Delete earlier outputs by name, so that a crashed re-run leaves no stale file to trust."""
    return guard722.invalidate_outputs(names=names, out_dir=out_dir)


RERUN_APPROVAL_ENV: Final[str] = "E723_RERUN_APPROVED_BY_MANAGER"
"""The manager's approval of one rerun, written `<job>:<attempt>` (for example `w0:2`), the attempt being the one to start. Any other value refuses the start."""


def archive_outputs(
    *,
    names: Sequence[str],
    private_names: Sequence[str] = (),
    out_dir: Path = OUT_DIR,
    private_dir: Path | None = None,
    folder_name: str | None = None,
) -> Path | None:
    """Move a job's earlier outputs into `attempt_<n>/` (or `folder_name`), keeping all outputs.

    Rerun condition 4 of the plan page keeps the outputs of every attempt. Nothing is deleted. A
    file that does not exist is skipped, and no folder is made when no file exists. The numbering
    counts the existing `attempt_*` folders. Files of `private_names` are moved from `private_dir`
    (default: `out_dir/PRIVATE`) into `attempt_<n>/PRIVATE/`.

    Args:
        names: The public output files to move.
        private_names: The output files under the private folder to move.
        out_dir: The experiment's output folder.
        private_dir: The private folder (default: `out_dir/PRIVATE`).
        folder_name: The name of the attempt folder; the default numbers the existing
            `attempt_*` folders.

    Returns:
        The attempt folder, or None when there was nothing to move.
    """
    private = out_dir / "PRIVATE" if private_dir is None else private_dir
    present = [out_dir / n for n in names if (out_dir / n).is_file()]
    present_private = [private / n for n in private_names if (private / n).is_file()]
    if not present and not present_private:
        return None
    taken = [
        int(m.group(1))
        for p in out_dir.glob("attempt_*")
        if (m := re.fullmatch(r"attempt_(\d+)", p.name))
    ]
    attempt = out_dir / (folder_name or f"attempt_{max(taken, default=0) + 1}")
    attempt.mkdir(parents=True)
    for path in present:
        shutil.move(path, attempt / path.name)
    if present_private:
        (attempt / "PRIVATE").mkdir()
        for path in present_private:
            shutil.move(path, attempt / "PRIVATE" / path.name)
    return attempt


def attempt_marker_name(*, job: str) -> str:
    """Return the file name of a job's attempt marker."""
    return f"marker_attempt_{job}.json"


def begin_attempt(
    *,
    job: str,
    names: Sequence[str],
    private_names: Sequence[str] = (),
    out_dir: Path = OUT_DIR,
    env: Mapping[str, str] | None = None,
) -> int:
    """Start a job's counted attempt: refuse an unapproved rerun, archive earlier outputs, write the marker.

    The marker `marker_attempt_<job>.json` is written before the job reads anything, so a job that
    crashes, ends `invalid`, or ends with any other verdict leaves the marker, and the next run is
    refused unless `RERUN_APPROVAL_ENV` equals `<job>:<attempt>` for the attempt to start. One
    approval therefore permits one rerun. With the approval the earlier outputs and the old marker
    move into `attempt_<job>_<n>/`, `n` being the attempt they belong to, and the approval text is
    recorded in the new marker.

    Args:
        job: `w0`, `w2`, or `w3`.
        names: The job's public outputs, including its run log, which are archived on a rerun.
        private_names: The job's outputs under `PRIVATE/`, archived on a rerun.
        out_dir: The experiment's output folder.
        env: The environment (default `os.environ`), for a test.

    Returns:
        The attempt number, 1 for the first attempt.

    Raises:
        JobRefusedError: An earlier attempt of this job began and the approval does not name this
            job and the attempt to start.
    """
    marker = out_dir / attempt_marker_name(job=job)
    environment = os.environ if env is None else env
    previous = 0
    approval = ""
    if marker.is_file():
        try:
            previous = int(json.loads(marker.read_text())["attempt"])
        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            msg = (
                f"the attempt marker of job {job} is unreadable or has no attempt number; "
                "the manager must repair it before the job can start"
            )
            raise JobRefusedError(msg) from exc
        approval = environment.get(RERUN_APPROVAL_ENV, "").strip()
        if approval != f"{job}:{previous + 1}":
            msg = (
                f"attempt {previous} of job {job} already began (it may have crashed, or ended with "
                f"any verdict); attempt {previous + 1} needs the manager's approval, the value "
                f"'{job}:{previous + 1}' in the environment variable {RERUN_APPROVAL_ENV}"
            )
            raise JobRefusedError(msg)
    archive = out_dir / f"attempt_{job}_{previous}"
    if marker.is_file() and archive.exists():
        msg = (
            f"the archive folder {archive.name} of job {job} already exists (an earlier start "
            "stopped part-way); the manager must inspect it before the job can start"
        )
        raise JobRefusedError(msg)
    moved = archive_outputs(
        names=names, private_names=private_names, out_dir=out_dir,
        private_dir=out_dir / "PRIVATE", folder_name=archive.name,
    )  # fmt: skip
    if marker.is_file():
        # Copy, never move, the old marker, so that a crash before the new marker is written
        # leaves the counter in place.
        folder = moved if moved is not None else archive
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(marker, folder / marker.name)
    attempt = previous + 1
    payload = json.dumps({"job": job, "attempt": attempt, "approval": approval}, sort_keys=True)
    atomic_write(path=marker, write=lambda tmp: tmp.write_text(payload))
    return attempt


def known_answer_tests(
    *, tests_dir: Path | None = None, extra_args: Sequence[str] = ()
) -> dict[str, Any]:
    """Run this folder's tests in a child process and return `passed`, `n_tests`, and the code hash.

    The tests run on synthetic arrays only. Row 1 of the verdict table reads `passed`
    (the plan page: "a known-answer test on synthetic arrays"). `extra_args` is for a test of this
    function.
    """
    folder = FOLDER / "tests" if tests_dir is None else tests_dir
    args = [
        sys.executable,
        "-m",
        "pytest",
        str(folder),
        "-q",
        "-p",
        "no:cacheprovider",
        *extra_args,
    ]
    if tests_dir is None:
        args.append("--run-studies")
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    passed_match = re.search(r"(\d+) passed", done.stdout)
    n_tests = int(passed_match.group(1)) if passed_match else 0
    return {
        "passed": bool(done.returncode == 0 and n_tests > 0),
        "n_tests": n_tests,
        "code_hash": code_hash(),
    }


def code_hash(*, folder: Path = FOLDER) -> str:
    """Return a SHA-256 over every `e00723_*.py` file of the folder and `run.sh`, in name order."""
    digest = hashlib.sha256()
    for path in [*sorted(folder.glob("e00723_*.py")), folder / "run.sh"]:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprints(*, inputs: Sequence[Path]) -> dict[str, str]:
    """Return a fingerprint for each input table that exists (a hash of the file, or of a large file's head)."""
    return {
        guard722._relative(path): file_fingerprint(path=path) for path in inputs if path.is_file()
    }


def write_run_log(
    *,
    job: str,
    inputs: Sequence[Path],
    gate: TimingGate,
    extra: Mapping[str, Any] | None = None,
    out_dir: Path = OUT_DIR,
    name: str | None = None,
) -> Path:
    """Write `run_log_<job>.json`: code hash, input fingerprints, peak memory, and core-seconds.

    The log holds no data value. Input paths are written relative to the data folder.
    """
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload: dict[str, Any] = {
        "job": job,
        "code_hash": code_hash(),
        "git_commit": guard722.git_commit(folder=FOLDER),
        "input_fingerprints": fingerprints(inputs=inputs),
        "inputs_missing": [guard722._relative(p) for p in inputs if not p.is_file()],
        "peak_memory_mb": round(peak_kib / 1024.0, 1),
        "core_seconds": round(gate.elapsed(), 1),
        "core_second_cap": gate.cap_core_seconds,
        "n_boot": core.N_BOOT,
        "seed": core.SEED,
        **(extra or {}),
    }
    return write_public_json(payload=payload, name=name or f"run_log_{job}.json", out_dir=out_dir)


@contextmanager
def run_log(
    *,
    job: str,
    gate: TimingGate,
    inputs: list[Path],
    info: dict[str, Any],
    out_dir: Path = OUT_DIR,
) -> Generator[None]:
    """Write the run log when the block ends, whether it finished or crashed.

    The block appends to `inputs` and `info` as it reads sources, so a crashed job still records
    what it read. The log carries `completed`, which is False after a crash. The job puts its
    attempt number in `info["attempt"]` once `begin_attempt` has returned. A block that ends
    without it (a refused or mis-launched start) never began an attempt, so its log goes to
    `run_log_<job>_refused_<stamp>.json` and the earlier attempt's log stays as it was.
    """
    completed = False
    try:
        yield
        completed = True
    finally:
        began = "attempt" in info
        write_run_log(
            job=job,
            inputs=inputs,
            gate=gate,
            extra={**info, "completed": completed, **({} if began else {"refused": True})},
            out_dir=out_dir,
            name=None
            if began
            else f"run_log_{job}_refused_{datetime.now(UTC).strftime('%H%M%S%f')}.json",
        )


def read_json(*, name: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    """Return a JSON file of this experiment, or raise `JobRefusedError` when it is missing."""
    path = out_dir / name
    if not path.is_file():
        msg = f"{name} is missing: run the earlier job first"
        raise JobRefusedError(msg)
    return json.loads(path.read_text())


class JobRefusedError(RuntimeError):
    """Raised when an earlier job's output, or the environment, forbids this job."""
