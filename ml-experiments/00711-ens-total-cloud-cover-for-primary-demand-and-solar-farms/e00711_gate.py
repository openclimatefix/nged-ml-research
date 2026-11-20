"""The gate marker and the manifest check that guard the fits of experiment 711.

**The fits refuse to run until both gates have passed.** The fit gate compares the new code's
refit of `R_F` (arm `D0n`) with the stored `R_F`. The prediction gate reloads the stored `R_F` state
and compares the forecasts. Each gate result is recorded in `work/gate_passed.json` with the code
commit that produced it. `e00711_score.py gate` records the fit gate, and `e00711_fit.py
--gate-predict` records the prediction gate.

**A prediction set written from an uncommitted tree or from an unrelated commit is not reused.**
`check_existing_set` reads the set's manifest and raises if `dirty` is true or if the commit that
wrote the set is not an ancestor of the commit that is checked out.
"""

import json
import subprocess
from pathlib import Path
from typing import Final, Literal

import e00711_arms as arms
import e00711_common as common

from studies import experiment_store

GATE_PATH: Final[Path] = common.WORK_DIR / "work" / "gate_passed.json"
"""The marker: `{"fit": {month: {...}}, "predict": {month: {...}}}`."""
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


def record(
    *,
    kind: GateKindType,
    month: str,
    passed: bool,
    detail: str,
    commit: str,
    path: Path = GATE_PATH,
) -> None:
    """Record one gate result in the marker, replacing an earlier result for the same month."""
    marker = read_marker(path=path)
    marker[kind][month] = {"passed": passed, "detail": detail, "commit": commit}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    temporary.rename(path)


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
    return missing


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
        ValueError: If the manifest says `dirty`, or its commit is not an ancestor of `head`.
    """
    manifest = json.loads((folder / experiment_store.MANIFEST_NAME).read_text())
    if manifest.get("dirty"):
        msg = f"{folder.name} was written from an uncommitted tree; delete it and fit again"
        raise ValueError(msg)
    current = head_commit() if head is None else head
    if not is_ancestor(commit=manifest["commit"], descendant=current):
        msg = f"{folder.name} was written by commit {manifest['commit']}, not an ancestor of HEAD"
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
