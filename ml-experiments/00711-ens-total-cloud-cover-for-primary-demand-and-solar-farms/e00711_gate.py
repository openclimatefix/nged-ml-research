"""The gate marker and the manifest check that guard the fits of experiment 711.

**The fits refuse to run until both gates have passed.** The fit gate compares the new code's
refit of `R_F` (arm `D0n`) with the stored `R_F`. The prediction gate reloads the stored `R_F` state
and compares the forecasts. Each gate result is recorded in `work/gate_passed.json` with the code
commit that produced it. `e00711_score.py gate` records the fit gate, and `e00711_fit.py
--gate-predict` records the prediction gate.

**A gate result counts only while the code is unchanged.** The marker records each result's commit.
A result is current if that commit is `HEAD`, or an ancestor of `HEAD` with no change under the
experiment's folder. Two gate commands that run at once take a file lock, so neither loses the
other's result.

**A prediction set written from an uncommitted tree or from older code is not reused.**
`check_existing_set` reads the set's manifest and raises if `dirty` is true, if the commit that
wrote the set is not an ancestor of the checked-out commit, or if it is older than `FLOOR_COMMIT`.
The fit and the scorer both apply the check.
"""

import fcntl
import json
import subprocess
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

import e00711_arms as arms
import e00711_common as common

from studies import experiment_store

GATE_PATH: Final[Path] = common.WORK_DIR / "work" / "gate_passed.json"
"""The marker: `{"fit": {month: {...}}, "predict": {month: {...}}}`."""
FLOOR_COMMIT: Final[str] = "f4b8d3b9b31d53bc8789e5a8e0ae446089537caf"
"""The oldest code whose prediction sets the experiment reuses: the commit of the stage 2 fixes.
A set written by an older commit is rejected, so a set from before a code change is refit."""
EXPERIMENT_FOLDER: Final[Path] = Path(__file__).resolve().parent
GateKindType = Literal["fit", "predict"]
GATE_KINDS: Final[tuple[GateKindType, ...]] = ("fit", "predict")
GATE_FREE_ARMS: Final[frozenset[str]] = frozenset({"D0n"})
"""The arms that run before the gate has passed: the gate's own fit."""


def head_commit() -> str:
    """Return the commit that is checked out."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def is_ancestor(*, commit: str, descendant: str) -> bool:
    """Return whether `commit` is `descendant` or one of its ancestors."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, descendant],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def read_marker(*, path: Path = GATE_PATH) -> dict[str, dict[str, dict]]:
    """Return the marker's content, or an empty marker."""
    if not path.exists():
        return {kind: {} for kind in GATE_KINDS}
    marker = json.loads(path.read_text())
    return {kind: marker.get(kind, {}) for kind in GATE_KINDS}


@contextmanager
def locked(*, path: Path) -> Generator[None]:
    """Hold an exclusive file lock beside `path`, so two gate commands keep both results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def record(
    *,
    kind: GateKindType,
    month: str,
    passed: bool,
    detail: str,
    commit: str,
    path: Path = GATE_PATH,
) -> None:
    """Record one gate result in the marker, replacing an earlier result for the same month.

    The read, edit, and rename happen under a file lock, and the temporary file has a unique name.
    """
    with locked(path=path):
        marker = read_marker(path=path)
        marker[kind][month] = {"passed": passed, "detail": detail, "commit": commit}
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(marker, indent=2) + "\n")
        temporary.rename(path)


def commit_is_current(*, commit: str, folder: Path = EXPERIMENT_FOLDER) -> bool:
    """Return whether a recorded commit is `HEAD`, or an ancestor with no change under `folder`."""
    head = head_commit()
    if commit == head:
        return True
    if not is_ancestor(commit=commit, descendant=head):
        return False
    changed = subprocess.run(
        ["git", "diff", "--quiet", commit, head, "--", str(folder)],
        check=False,
        capture_output=True,
    )
    return changed.returncode == 0


def missing_gates(*, path: Path = GATE_PATH) -> list[str]:
    """Return which gate results are absent or failed, as one line each (empty if both passed)."""
    marker = read_marker(path=path)
    missing = []
    for kind in GATE_KINDS:
        for month in arms.GATE_MONTHS:
            result = marker[kind].get(month)
            if result is None:
                missing.append(f"{kind} gate at {month}: not run")
            elif not result["passed"]:
                missing.append(f"{kind} gate at {month}: failed ({result['detail']})")
            elif not commit_is_current(commit=result["commit"]):
                missing.append(
                    f"{kind} gate at {month}: passed at commit {result['commit'][:9]}, but the "
                    "experiment's code has changed since; run the gate again"
                )
    return missing


def gate_state_lines(*, path: Path = GATE_PATH) -> list[str]:
    """Return the gate state as printable lines, for a dry run that must not raise.

    A real fit raises if any gate result is missing or stale. A dry run prints these lines instead,
    so that a launcher sees the refusal before the queue starts.
    """
    missing = missing_gates(path=path)
    if not missing:
        return ["gate: both gates passed at both gate months, current at HEAD"]
    return [f"gate: a real fit would be refused; {line}" for line in missing]


def refuse_without_gate(*, arm: str, measure: bool, path: Path = GATE_PATH) -> None:
    """Raise unless the gate marker holds both gates at both gate months.

    Args:
        arm: The arm to fit; the gate's own arm `D0n` always runs.
        measure: The explicit override for the measuring fit.
        path: The marker file.

    Raises:
        ValueError: If a gate result is absent or failed, and the arm is not `D0n` and `measure`
            is false.
    """
    if measure or arm in GATE_FREE_ARMS:
        return
    missing = missing_gates(path=path)
    if missing:
        msg = f"arm {arm} is refused until both gates pass: {'; '.join(missing)}"
        raise ValueError(msg)


def check_existing_set(*, folder: Path, head: str | None = None) -> None:
    """Raise if a prediction set was written from an uncommitted tree or an unrelated commit.

    Args:
        folder: The set's folder.
        head: The commit to compare with; the checked-out commit if not given.

    Raises:
        ValueError: If the manifest says `dirty`, its commit is not an ancestor of `head`, or its
            commit is older than `FLOOR_COMMIT`.
    """
    manifest = json.loads((folder / experiment_store.MANIFEST_NAME).read_text())
    if manifest.get("dirty"):
        msg = f"{folder.name} was written from an uncommitted tree; delete it and fit again"
        raise ValueError(msg)
    current = head_commit() if head is None else head
    if not is_ancestor(commit=manifest["commit"], descendant=current):
        msg = f"{folder.name} was written by commit {manifest['commit']}, not an ancestor of HEAD"
        raise ValueError(msg)
    if not is_ancestor(commit=FLOOR_COMMIT, descendant=manifest["commit"]):
        msg = (
            f"{folder.name} was written by commit {manifest['commit']}, older than the floor commit"
        )
        raise ValueError(msg)


def usable_sets(*, names: list[str], folder: Path) -> tuple[set[str], list[str]]:
    """Return the sets in `folder` that `check_existing_set` accepts, and the ones it rejects.

    Args:
        names: Candidate set names.
        folder: The folder of the prediction sets.

    Returns:
        The usable names, and one line per rejected set.
    """
    head = head_commit()
    usable: set[str] = set()
    rejected: list[str] = []
    for name in names:
        if not (folder / name).exists():
            continue
        try:
            check_existing_set(folder=folder / name, head=head)
        except ValueError as error:
            rejected.append(str(error))
        else:
            usable.add(name)
    return usable, rejected
