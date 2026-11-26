# ruff: noqa: E501
"""Experiment 716 and 717: the arms, the folds, the file layout, the job list, and the code hash.

This module is WP4. Every arm of both parts is defined here, so part 2's folder imports this module.
Part 1's arms (`Z`, `H`, `H-S`, `H-X`) write under `00716-*/work/`, and part 2's (`HT`, `HT-S`, `T`)
under `00717-*/work/`. Every arm starts stages 2 and 3 from the stored stage 1 state of `M*` of
experiment 713.
"""

import functools
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00713_arms as arms713  # ty: ignore[unresolved-import]

NAME_1: Final = "00716-msa-unmetered-pv-part-1-history-estimate"
NAME_2: Final = "00717-msa-unmetered-pv-part-2-target-row-forecast"
ALL_FOLDS: Final = arms713.ALL_FOLDS
SCREEN_FOLDS: Final = ("2025-11", "2026-01", "2026-05", "2026-06")
"""The four folds of part 1's screen: one per season, two of them sunny."""
OTHER_FOLDS: Final = tuple(m for m in ALL_FOLDS if m not in SCREEN_FOLDS)
Z_FOLDS: Final = ("2025-11", "2026-01", "2026-04", "2026-05", "2026-06")
"""The five folds on which arm `Z` is refitted."""
TWO_SEEDS: Final = (0, 1)
HEADLINE_SEEDS: Final = (0, 1, 2, 3, 4)
EXTRA_SEEDS: Final = (2, 3, 4)
ROW_SET_ID: Final = arms713.ROW_SET_ID
HistoryKind = Literal["none", "real", "shifted", "partner"]
TargetKind = Literal["none", "real", "shifted"]


@dataclass(frozen=True)
class PvArmSpec:
    """What an arm feeds the network.

    Attributes:
        name: The arm's name (`Z`, `H`, `H-S`, `H-X`, `HT`, `HT-S`, `T`).
        net: The network of `e00716_model.PV_SWITCHES` (`Z`, `H`, `HT`, or `T`).
        history: The history estimate: `none`, `real` (CAMS), `shifted` (the sky 28 days earlier),
            or `partner` (another Primary's).
        target: The target-row estimate: `none`, `real` (the origin's ENS run), or `shifted` (the
            run 28 days earlier).
        part: 1 or 2.
    """

    name: str
    net: str
    history: HistoryKind
    target: TargetKind
    part: int


ARMS: Final[dict[str, PvArmSpec]] = {
    "Z": PvArmSpec("Z", "Z", "none", "none", 1),
    "H": PvArmSpec("H", "H", "real", "none", 1),
    "H-S": PvArmSpec("H-S", "H", "shifted", "none", 1),
    "H-X": PvArmSpec("H-X", "H", "partner", "none", 1),
    "HT": PvArmSpec("HT", "HT", "real", "real", 2),
    "HT-S": PvArmSpec("HT-S", "HT", "real", "shifted", 2),
    "T": PvArmSpec("T", "T", "none", "real", 2),
}


FIT_CODE_FILES: Final = (
    "e00716_arms.py", "e00716_batch.py", "e00716_fit.py", "e00716_model.py", "e00716_pv.py",
    "e00716_train.py", "e00716_worlds.py", "e00716_world_rung.py", "e00716_known_answer.py",
)  # fmt: skip
"""The modules of this folder whose edits can change a fitted state (prediction has its own record)."""


@functools.cache
def code_hash() -> str:
    """Return a hash of this folder's fit modules and of experiment 713's own code hash.

    That code hash covers its fit modules and recipe.
    """
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in FIT_CODE_FILES:
        h.update((here / name).read_bytes())
    h.update(f"e00713:{arms713.code_hash()}".encode())
    return h.hexdigest()[:16]


@functools.cache
def file_hashes() -> dict[str, str]:
    """Return the first 12 hex digits of the sha256 of each fit module.

    The known-answer files are included.
    """
    here = Path(__file__).parent
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest()[:12] for n in FIT_CODE_FILES}


def _git(*args: str) -> str:
    """Return the output of a git command run in this folder, or `unavailable`."""
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unavailable"


WORLD_FILES: Final = ("e00716_worlds.py", "e00716_world_rung.py")
"""`world_rung.json` and its rung files depend on these two modules alone."""


def code_record() -> dict:
    """Return what a log records about the code.

    The record holds this folder's hash, 713's hash, the git commit, and a dirty flag.
    """
    return {
        "code_hash": code_hash(),
        "e00713_code_hash": arms713.code_hash(),
        "file_sha256": file_hashes(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
    }


def check_code(
    *, earlier: dict, accept: Sequence[str], record: dict, what: str, kind: str = "state"
) -> None:
    """Refuse a state or a gate written by other code, unless an override names its exact old hash.

    The override is appended to `record["code_overrides"]` with both hashes and both commits.
    """
    now = code_hash()
    old = earlier.get("code_hash")
    if old is None:
        msg = f"{what} carries no code record: refit it"
        raise AssertionError(msg)
    if old == now:
        return
    flag = "--accept-gate-change" if kind == "gate" else "--accept-code-change"
    if old not in accept:
        msg = (
            f"{what} was written by code with hash {old}, and the code now has hash {now}. "
            f"After an edit that cannot change a fitted state, rerun with `{flag} {old}`; the override is logged."
        )
        raise AssertionError(msg)
    entry = {
        "what": what, "kind": kind, "old_hash": old, "new_hash": now,
        "old_commit": earlier.get("git_commit", "unavailable"), "new_commit": _git("rev-parse", "HEAD"),
    }  # fmt: skip
    if entry not in record.setdefault("code_overrides", []):
        record["code_overrides"].append(entry)
        print("code override accepted:", json.dumps(entry), flush=True)


def check_713_warm_start(
    *, stage1_log: dict, accept_713: Sequence[str], record: dict, what: str
) -> None:
    """Assert that the logged code hash of the stage 1 state of `M*` equals 713's `code_hash()`.

    The logged hash is the one stored with that state, and `code_hash()` is as this
    checkout computes it.

    An override names the exact old hash with `--accept-713-hash`, and is logged.
    """
    old = stage1_log.get("code_hash")
    now = arms713.code_hash()
    if old == now:
        return
    if old is None or old not in accept_713:
        msg = (
            f"{what}: the stage 1 state of M* was fitted by experiment 713 code with hash {old}, and this "
            f"checkout computes {now}. Read the diff, then rerun with `--accept-713-hash {old}` if it cannot change a state."
        )
        raise AssertionError(msg)
    entry = {"what": what, "kind": "713 warm start", "old_hash": old, "new_hash": now}
    if entry not in record.setdefault("code_overrides", []):
        record["code_overrides"].append(entry)
        print("713 code override accepted:", json.dumps(entry), flush=True)


def root(*, data_dir: Path, arm: str) -> Path:
    """Return the data folder of the experiment that owns `arm`."""
    return data_dir / (NAME_1 if ARMS[arm.removesuffix("-nohist")].part == 1 else NAME_2)


def gate_file(*, data_dir: Path) -> Path:
    """Return the known-answer gate file (part 1's folder holds it for both parts)."""
    return data_dir / NAME_1 / "work" / "known_answers_passed.json"


def model_file(*, data_dir: Path, arm: str, month: str, seed: int, stage: int) -> Path:
    """Return the saved state after `stage` (2 or 3)."""
    return (
        root(data_dir=data_dir, arm=arm)
        / "work"
        / "models"
        / arm
        / f"{month}_s{seed}_stage{stage}.pt"
    )


def fit_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file."""
    return root(data_dir=data_dir, arm=arm) / "work" / "fits" / arm / f"{month}_s{seed}.parquet"


def log_file(*, data_dir: Path, arm: str, month: str, seed: int, kind: str) -> Path:
    """Return a job's JSON log; `kind` is `fit` or `predict`."""
    return root(data_dir=data_dir, arm=arm) / "work" / "logs" / arm / f"{month}_s{seed}_{kind}.json"


def m_star_stage1_state(*, data_dir: Path, month: str, seed: int) -> Path:
    """Return the stored stage 1 state of experiment 713's `M*` for (`month`, `seed`)."""
    star = arms713.m_star(data_dir=data_dir)
    return arms713.model_file(data_dir=data_dir, arm=star, month=month, seed=seed, stage=1)


def m_star_stage1_log(*, data_dir: Path, month: str, seed: int) -> Path:
    """Return the stage 1 log of experiment 713's `M*`."""
    star = arms713.m_star(data_dir=data_dir)
    path = arms713.log_file(data_dir=data_dir, arm=star, month=month, seed=seed, kind="fit")
    return path.with_name(f"{path.stem}_stage1.json")


def jobs(*, phase: int) -> list[tuple[str, str, int]]:
    """Return the (arm, month, seed) jobs of a phase of design section 10.

    Phase 2: `Z` on 5 folds, seeds 0 and 1. Phase 3: `H` and `H-S` on the 4 screen folds, seeds 0
    and 1. Phase 4: `H` and `H-S` on the other 4 folds. Phase 5: `HT` and `HT-S` on 8 folds, seeds 0
    and 1. Phase 6: `H`, `H-S`, `HT`, and `HT-S` on 8 folds, seeds 2 to 4. Phase 7: `T` and `H-X` on
    8 folds, seeds 0 and 1.
    """
    if phase == 2:
        return [("Z", m, s) for m in Z_FOLDS for s in TWO_SEEDS]
    if phase == 3:
        return [(a, m, s) for a in ("H", "H-S") for m in SCREEN_FOLDS for s in TWO_SEEDS]
    if phase == 4:
        return [(a, m, s) for a in ("H", "H-S") for m in OTHER_FOLDS for s in TWO_SEEDS]
    if phase == 5:
        return [(a, m, s) for a in ("HT", "HT-S") for m in ALL_FOLDS for s in TWO_SEEDS]
    if phase == 6:
        return [
            (a, m, s) for a in ("H", "H-S", "HT", "HT-S") for m in ALL_FOLDS for s in EXTRA_SEEDS
        ]
    if phase == 7:
        return [(a, m, s) for a in ("T", "H-X") for m in ALL_FOLDS for s in TWO_SEEDS]
    msg = f"phase {phase} has no fit jobs (0 and 1 are the smoke fit and the known-answer worlds; 8 is the read-outs)"
    raise ValueError(msg)


def gate713_summary(*, data_dir: Path) -> dict:
    """Return experiment 713's gate record, and refuse an unwaived failure.

    The record has the form `{waived, failed_checks, ...}`. The file is
    `00713-*/work/known_answers_passed.json`. A record with `waived` true must name the failed
    checks (KA3 and KA12) and `thresholds_changed` false.
    """
    path = arms713.root(data_dir=data_dir) / "work" / "known_answers_passed.json"
    assert path.exists(), f"{path} is missing: experiment 713's known-answer gate has not run"
    record = json.loads(path.read_text())
    waived = bool(record.get("waived", False))
    failed = [
        k for k in ("KA2", "KA3", "KA4", "KA10", "KA12") if not record.get(k, {}).get("pass", False)
    ]
    if failed:
        assert waived, f"experiment 713's gate has failed checks {failed} and no waiver"
        waiver = record["waiver"]
        assert sorted(waiver["failed_checks"]) == sorted(failed), (waiver["failed_checks"], failed)
        assert waiver.get("thresholds_changed") is False, "the waiver changed a threshold"
    return {
        "waived": waived, "failed_checks": failed,
        "waiver_date": (record.get("waiver") or {}).get("date"),
        "decision_quoted": (record.get("waiver") or {}).get("decision_quoted"),
        "e00713_code_hash": (record.get("code") or {}).get("code_hash"),
    }  # fmt: skip


def own_cache(*, data_dir: Path | None = None) -> Path:
    """Return this experiment's own dense-array cache folder (part 1's `work/cache`)."""
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    return (ML_EXPERIMENTS_DATA_DIR if data_dir is None else data_dir) / NAME_1 / "work" / "cache"


def install_own_cache() -> Path:
    """Point experiment 713's loaders at this experiment's own cache, seeded by copies of 713's files.

    The loaders write the dense ERA5 and B1 arrays into `e00713_inputs.CACHE` the first time they build
    them. The folder is redirected before the first call, so nothing is written into experiment 713's
    or 632's data. The files are copied and not hard-linked, because a hard link would let a rewrite
    change experiment 713's file.
    """
    import shutil

    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    own = own_cache()
    own.mkdir(parents=True, exist_ok=True)
    old = Path(inp.CACHE)
    if old.exists() and old != own:
        for src in sorted(old.glob("*.npy")):
            dest = own / src.name
            if not dest.exists():
                tmp = dest.with_name(dest.name + ".tmp")
                shutil.copy2(src, tmp)
                tmp.rename(dest)
    inp.CACHE = own
    return own


def watched_mtimes(*, data_dir: Path) -> dict[str, int]:
    """Return the modification times of the files of experiment 713 that this experiment reads.

    This experiment must not change those files.
    """
    base = arms713.root(data_dir=data_dir)
    files = [
        base / "work" / "selection.json",
        *sorted((base / "work" / "models").glob("*/*_stage1.pt")),
        *sorted((base / "work" / "logs").glob("*/*_stage1.json")),
        *sorted((base / "work" / "standardiser").glob("*.sha256")),
        *sorted((base / "cache").glob("*")),
    ]
    return {str(f): f.stat().st_mtime_ns for f in files if f.exists()}


def wiring_file(*, data_dir: Path, name: str) -> Path:
    """Return the record of one wiring check (`KF1` to `KF14`).

    The command `e00716_known_answer.py finish` reads the record.
    """
    return root(data_dir=data_dir, arm="Z") / "work" / "wiring" / f"{name}.json"


CHECK_FILES: Final = (
    "e00716_check_batch.py", "e00716_check_pv.py", "e00716_check_model.py", "e00716_check_train.py",
    "e00716_predict.py",
)  # fmt: skip
"""The scripts that write the wiring records (and `e00716_predict.py`, which runs KF7); their sha256 is pinned in every record."""
FOLD_CHECKS: Final = ("KF2", "KF4", "KF7")
"""The wiring checks that run on a real fold and must record which one."""


def check_file_hashes() -> dict[str, str]:
    """Return the first 12 hex digits of the sha256 of each check script."""
    here = Path(__file__).parent
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest()[:12] for n in CHECK_FILES}


def seal_of(*, body: dict) -> str:
    """Return the sha256 of a record's contents and its commit.

    A hand edit of any field breaks the seal. The seal is not a secret, so it shows an edit rather
    than stops one.
    """
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def write_wiring_record(
    *, data_dir: Path, name: str, passed: bool, detail: dict | None = None, fold: str | None = None
) -> Path:
    """Write the sealed record of a wiring check.

    The record holds the code hash, the check scripts' hashes, 713's hash, the git state, and the fold
    the check ran on.
    """
    assert name not in FOLD_CHECKS or fold, f"{name} runs on a real fold: pass `fold`"
    path = wiring_file(data_dir=data_dir, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "check": name, "pass": bool(passed), "detail": detail or {}, "fold": fold,
        "check_file_sha256": check_file_hashes(), **code_record(),
    }  # fmt: skip
    record = {**body, "seal": seal_of(body=body)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.rename(path)
    return path


def wiring_problems(*, name: str, record: dict) -> list[str]:
    """Return why a wiring record cannot be accepted now (empty when it can)."""
    body = {k: v for k, v in record.items() if k != "seal"}
    out = []
    if record.get("seal") != seal_of(body=body):
        out.append("the record was edited after it was written (its seal does not match)")
    if not record.get("pass"):
        out.append("the check did not pass")
    if record.get("code_hash") != code_hash():
        out.append("written by other fit code")
    if record.get("check_file_sha256") != check_file_hashes():
        out.append("written by other check scripts")
    if record.get("e00713_code_hash") != arms713.code_hash():
        out.append("written against other experiment 713 code")
    if record.get("git_dirty"):
        out.append("written from a dirty tree")
    if name in FOLD_CHECKS and not record.get("fold"):
        out.append("no fold recorded")
    return [f"wiring record {name}: {x}" for x in out]


def artefact_problems(*, what: str, record: dict, files: Sequence[str] | None = None) -> list[str]:
    """Return why a stored artefact (an `M` log, a job, a pre-check, `world_rung.json`) cannot be used now.

    With `files` the check compares only those modules' sha256 (`world_rung.json` depends on the two
    world modules alone); otherwise the whole fit-code hash must match. A record from a dirty tree, or
    with no code record, is refused.
    """
    code = record.get("code", record)
    if "code_hash" not in code:
        return [f"{what} carries no code record: rebuild it"]
    out = []
    if files is None:
        if code["code_hash"] != code_hash():
            out.append("written by other fit code")
    else:
        now, then = file_hashes(), code.get("file_sha256", {})
        out += [f"{f} changed since it was written" for f in files if then.get(f) != now[f]]
    if code.get("git_dirty"):
        out.append("written from a dirty tree")
    return [f"{what}: {x}" for x in out]
