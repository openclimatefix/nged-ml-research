# ruff: noqa: E501
"""Experiment 722: the anonymisation guard, the order-of-reads log, the timing gate, and the run log.

**Nothing dated leaves `PRIVATE/`.** `write_public_csv` and `write_public_json` refuse a frame or
a value that holds a timestamp, a date, or a column whose name says it is one. `say` is the only
way a job prints, and it refuses a string that looks like a date. The jobs never write a path
under `/home`: every location comes from `studies.sources`.
"""

import hashlib
import json
import re
import resource
import subprocess
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import e00722_core as core
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_FOLDER: Final[str] = "00722-msa-mva-bsp-closure-and-fast-hint"
"""The experiment's folder name under `data/ml-experiments/`."""

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_FOLDER
"""Where every output goes."""

PRIVATE_DIR: Final[Path] = OUT_DIR / "PRIVATE"
"""Where anything dated goes."""

DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d{4}-\d{2}-\d{2}")
"""A printed or published value may not hold a calendar date."""

DATED_COLUMN_PARTS: Final[tuple[str, ...]] = ("time", "date", "stamp", "start")
"""A public column whose name holds one of these parts is refused, apart from `time_series_id`."""

HASH_FULL_LIMIT_BYTES: Final[int] = 1_000_000_000
"""A file up to this size is hashed in full; a larger file by its first 64 MB, its size, and its mtime."""

HEAD_BYTES: Final[int] = 64 * 1024 * 1024
"""How much of a large file the fingerprint reads."""


class AnonymisationError(ValueError):
    """Raised when a public output or a printed line would carry a dated value."""


class TimingGateError(RuntimeError):
    """Raised when a job's projected cost exceeds its cap."""


class ReadOrderError(RuntimeError):
    """Raised when a Primary reading was read before the class audit finished."""


def say(*parts: object) -> None:
    """Print one line of aggregates: ratios, counts, shares, and intervals only."""
    text = " ".join(str(p) for p in parts)
    if DATE_PATTERN.search(text):
        msg = "a printed line may not hold a calendar date"
        raise AnonymisationError(msg)
    print(text)


def check_public_frame(*, frame: pl.DataFrame, name: str) -> None:
    """Raise when a frame about to be written outside `PRIVATE/` holds a timestamp or a generator id.

    A column named `generator_id` is refused, and so is a `time_series_id` column in a frame
    whose `kind` column holds a generator row. A public table names a generator by its anonymous
    label (`GA1`, `GB3`), and the map from label to id lives under `PRIVATE/`.
    """
    if "generator_id" in frame.columns:
        msg = f"{name}: a generator id may not be written outside PRIVATE/; use the anonymous label"
        raise AnonymisationError(msg)
    if (
        "time_series_id" in frame.columns
        and "kind" in frame.columns
        and frame.filter(pl.col("kind").cast(pl.String) == "generator")["time_series_id"]
        .is_not_null()
        .any()
    ):
        msg = f"{name}: a generator's time series id may not be written outside PRIVATE/"
        raise AnonymisationError(msg)
    for column, dtype in frame.schema.items():
        dated = dtype in (pl.Datetime, pl.Date, pl.Time) or isinstance(
            dtype, pl.Datetime | pl.Date | pl.Time
        )
        lowered = column.lower().replace("time_series", "")
        named = any(part in lowered for part in DATED_COLUMN_PARTS)
        if dated or named:
            msg = f"{name}: column {column!r} is dated and may not be written outside PRIVATE/"
            raise AnonymisationError(msg)
        if dtype == pl.String and frame[column].str.contains(DATE_PATTERN.pattern).any():
            msg = f"{name}: column {column!r} holds a calendar date"
            raise AnonymisationError(msg)


def atomic_write(*, path: Path, write: Callable[[Path], object]) -> Path:
    """Write through `write` to `path` plus `.tmp` in the same folder, then rename onto `path`.

    A job that crashes mid-write leaves no half-written file under the real name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)
    return path


def invalidate_outputs(*, names: Sequence[str], out_dir: Path = OUT_DIR) -> list[str]:
    """Delete earlier outputs by name, so that a crashed re-run leaves no stale file to trust."""
    removed = []
    for name in names:
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def write_public_csv(*, frame: pl.DataFrame, name: str, out_dir: Path = OUT_DIR) -> Path:
    """Write an aggregate table to `out_dir/name` after the guard checks it."""
    check_public_frame(frame=frame, name=name)
    return atomic_write(path=out_dir / name, write=frame.write_csv)


def write_public_parquet(*, frame: pl.DataFrame, name: str, out_dir: Path = OUT_DIR) -> Path:
    """Write an aggregate table as parquet to `out_dir/name` after the guard checks it."""
    check_public_frame(frame=frame, name=name)
    return atomic_write(path=out_dir / name, write=frame.write_parquet)


def write_public_json(*, payload: Mapping[str, Any], name: str, out_dir: Path = OUT_DIR) -> Path:
    """Write a JSON verdict or check file to `out_dir/name`; refuses a value that holds a date."""
    text = json.dumps(payload, indent=1, sort_keys=True, default=float)
    if DATE_PATTERN.search(text):
        msg = f"{name}: the payload holds a calendar date"
        raise AnonymisationError(msg)
    return atomic_write(path=out_dir / name, write=lambda tmp: tmp.write_text(text))


def write_private_parquet(
    *, frame: pl.DataFrame, name: str, private_dir: Path = PRIVATE_DIR
) -> Path:
    """Write dated detail under `PRIVATE/`."""
    return atomic_write(path=private_dir / name, write=frame.write_parquet)


# ---------------------------------------------------------------------------------------------
# Order of reads
# ---------------------------------------------------------------------------------------------
@dataclass
class ReadLog:
    """The sources a job read, in order, so that the order can be proved.

    A source's `kind` is `sky` (CAMS and sun geometry), `farm` (a farm's recorded output), or
    `primary` (any Primary's reading, the BSP meter, or a generator that sits below a Primary).
    """

    entries: list[tuple[int, str, str]] = field(default_factory=list)
    holdout: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_holdout(self, *, name: str, read_after: int, dropped: int) -> None:
        """Record, for one input, the rows read on or after the holdout and the rows the filter dropped."""
        self.holdout[name] = {
            "read_on_or_after_holdout": int(read_after),
            "dropped_by_filter": int(dropped),
        }

    def holdout_clean(self) -> bool:
        """Return True when no input read a row on or after the holdout."""
        return all(v["read_on_or_after_holdout"] == 0 for v in self.holdout.values())

    def record(self, *, name: str, kind: str) -> None:
        """Append a read."""
        self.entries.append((len(self.entries), name, kind))

    def mark_audit_done(self) -> None:
        """Mark the end of the class audit and the same-sky pairs."""
        self.record(name="class_audit_done", kind="marker")

    def assert_audit_before_primary_reads(self) -> None:
        """Raise unless the audit marker exists and no `primary` read comes before it."""
        marks = [i for i, _, kind in self.entries if kind == "marker"]
        if not marks:
            msg = "the class audit never finished"
            raise ReadOrderError(msg)
        first_mark = marks[0]
        early = [name for i, name, kind in self.entries if kind == "primary" and i < first_mark]
        if early:
            msg = f"Primary readings were read before the class audit: {early}"
            raise ReadOrderError(msg)

    def as_list(self) -> list[dict[str, Any]]:
        """Return the entries for a check file."""
        return [{"order": i, "source": name, "kind": kind} for i, name, kind in self.entries]


# ---------------------------------------------------------------------------------------------
# Timing gate and run log
# ---------------------------------------------------------------------------------------------
@dataclass
class TimingGate:
    """Stops a job whose projected cost exceeds `cap_core_seconds`.

    `core_seconds` is the process CPU time, which is the core-seconds of a one-core job. The
    projection is the setup time (start to `mark`) plus the measured time per unit since `mark`
    times the number of units, so loading data is counted once and never multiplied.
    """

    cap_core_seconds: float = core.CORE_SECOND_CAP
    start: float = field(default_factory=time.process_time)
    loop_start: float | None = None

    def elapsed(self) -> float:
        """Return the core-seconds used since the gate started."""
        return time.process_time() - self.start

    def mark(self, *, now: float | None = None) -> None:
        """Mark the end of the setup and the start of a loop of equal units."""
        self.loop_start = time.process_time() if now is None else now

    def check(self, *, done: int, total: int, now: float | None = None) -> float:
        """Project the total cost from `done` of `total` equal units since `mark`, and raise over the cap.

        Args:
            done: The number of units finished since `mark`.
            total: The number of units in the loop.
            now: The process time to use as the present, for a test; the clock when None.

        Returns:
            The projected core-seconds: setup plus the time per unit times `total`.
        """
        current = time.process_time() if now is None else now
        loop_start = self.start if self.loop_start is None else self.loop_start
        setup = loop_start - self.start
        projected = setup + (current - loop_start) / max(done, 1) * total
        if projected > self.cap_core_seconds:
            msg = f"projected {projected:.0f} core-seconds exceeds the cap of {self.cap_core_seconds:.0f}"
            raise TimingGateError(msg)
        return projected


def file_fingerprint(*, path: Path) -> str:
    """Return a SHA-256 of a file, or of its first 64 MB with the size and mtime when it is large."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if size <= HASH_FULL_LIMIT_BYTES:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
            return digest.hexdigest()
        digest.update(handle.read(HEAD_BYTES))
    digest.update(f"{size}:{int(path.stat().st_mtime)}".encode())
    return "partial:" + digest.hexdigest()


def code_hash(*, folder: Path) -> str:
    """Return a SHA-256 over every `e00722_*.py` file of the experiment folder, in name order."""
    digest = hashlib.sha256()
    for path in [*sorted(folder.glob("e00722_*.py")), folder / "run.sh"]:
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(*, folder: Path) -> str:
    """Return the current commit, or `unknown` outside a repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def write_run_log(
    *,
    job: str,
    folder: Path,
    inputs: Sequence[Path],
    gate: TimingGate,
    extra: Mapping[str, Any] | None = None,
    out_dir: Path = OUT_DIR,
) -> Path:
    """Write `run_log_<job>.json`: code hash, input hashes, peak memory, and core-seconds.

    The log holds no data value. Input paths are written relative to the data folder.
    """
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload: dict[str, Any] = {
        "job": job,
        "code_hash": code_hash(folder=folder),
        "git_commit": git_commit(folder=folder),
        "inputs": {
            _relative(path): file_fingerprint(path=path) for path in inputs if path.is_file()
        },
        "inputs_missing": [_relative(path) for path in inputs if not path.is_file()],
        "peak_memory_mb": round(peak_kib / 1024.0, 1),
        "core_seconds": round(gate.elapsed(), 1),
        "core_second_cap": gate.cap_core_seconds,
        "n_boot": core.N_BOOT,
        "seed": core.SEED,
        **(extra or {}),
    }
    return write_public_json(payload=payload, name=f"run_log_{job}.json", out_dir=out_dir)


def _relative(path: Path) -> str:
    """Return the path relative to the data folder, with any date in a folder name masked."""
    try:
        text = str(path.relative_to(ML_EXPERIMENTS_DATA_DIR.parent))
    except ValueError:
        text = path.name
    return DATE_PATTERN.sub("DATE", text)


@contextmanager
def run_log(
    *,
    job: str,
    folder: Path,
    gate: TimingGate,
    inputs: list[Path],
    info: dict[str, Any],
    out_dir: Path = OUT_DIR,
) -> Generator[None]:
    """Write the run log when the block ends, whether it finished or crashed.

    The block appends to `inputs` and `info` as it reads sources, so a crashed job still records
    what it read. The log carries `completed`, which is False after a crash.
    """
    completed = False
    try:
        yield
        completed = True
    finally:
        write_run_log(
            job=job,
            folder=folder,
            inputs=inputs,
            gate=gate,
            extra={**info, "completed": completed},
            out_dir=out_dir,
        )
