# ruff: noqa: E501
"""Experiment 718: arm specifications, the file layout, the job list, and the code hash (design sections 4, 6, 11, 15).

Part 1 owns arms `Z`, `R`, `R-A`, and `P90` (experiment 718), and part 2 owns `U`, `U-A`, `U27`,
`U27-A`, and `UX` (experiment 719). Both parts run through this module, and each arm writes under
its own part's data folder. Every arm uses `M*` of experiment 713, which `work/selection.json` of
experiment 713 names: its encoders are loaded from the Candidate chain and frozen (`M1`).
"""

import functools
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00718_layout as lay

NAME_PART_1: Final = "00718-msa-similar-day-retrieval-part-1-weather-distance"
NAME_PART_2: Final = "00719-msa-similar-day-retrieval-part-2-union-of-measures"
ALL_FOLDS: Final = a713.ALL_FOLDS
SCREEN_FOLDS: Final = a713.SCREEN_FOLDS
OTHER_FOLDS: Final = a713.OTHER_FOLDS
UX_FOLDS: Final = ("2025-12", "2026-01")
SCREEN_SEEDS: Final = (0, 1)
HEADLINE_SEEDS: Final = (0, 1, 2, 3, 4)
ROW_SET_ID: Final = a713.ROW_SET_ID
CEILING_GPU_HOURS: Final = 23.5
CEILING_GPU_HOURS_WITH_SEEDS_2_TO_4: Final = 36.0
FIT_HOURS_15_ROWS: Final = 0.13
FIT_HOURS_28_ROWS: Final = 0.24
PART_OF_ARM: Final = {
    "Z": 1,
    "R": 1,
    "R-A": 1,
    "P90": 1,
    "U": 2,
    "U-A": 2,
    "U27": 2,
    "U27-A": 2,
    "UX": 2,
}
CODE_FILES: Final = (
    "e00718_arms.py", "e00718_batch.py", "e00718_fit.py", "e00718_layout.py", "e00718_model.py",
    "e00718_retrieval.py", "e00718_tables.py", "e00718_train.py",
)  # fmt: skip
"""The modules of this experiment whose edits can change a fitted state."""
IMPORTED_713_FILES: Final = (
    "e00713_arms.py", "e00713_attention.py", "e00713_batch.py", "e00713_inputs.py",
    "e00713_layout.py", "e00713_model.py", "e00713_train.py",
)  # fmt: skip
"""Modules of experiment 713 that F9 imports unchanged and that its code hash covers."""


@dataclass(frozen=True)
class ArmSpec:
    """What an arm does.

    Attributes:
        name: The arm name (`Z`, `R`, `R-A`, `P90`, `U`, `U-A`, `U27`, `U27-A`, or `UX`).
        n_rows: The history rows.
        part: 1 or 2, which experiment's folder holds the arm's files.
        accumulate: Half-batches per optimiser step (2 for the 28-row arms if the memory rule fires).
    """

    name: str
    n_rows: int
    part: int
    accumulate: int = 1


def resolve(*, arm: str, accumulate: int = 1) -> ArmSpec:
    """Return the specification of `arm`."""
    assert arm in PART_OF_ARM, arm
    return ArmSpec(
        name=arm, n_rows=lay.ROWS_PER_ARM[arm], part=PART_OF_ARM[arm], accumulate=accumulate
    )


def encoders_mode(*, data_dir: Path) -> a713.EncoderMode:
    """Return `M*`'s encoder mode from experiment 713's selection (`frozen` for `M1`)."""
    return a713.SCREEN_ARMS[a713.m_star(data_dir=data_dir)][1]


def root(*, data_dir: Path, part: int) -> Path:
    """Return the data folder of part 1 or part 2."""
    return data_dir / (NAME_PART_1 if part == 1 else NAME_PART_2)


def _file(*, data_dir: Path, arm: str, kind: str, name: str, smoke: bool = False) -> Path:
    base = (
        root(
            data_dir=data_dir,
            part=PART_OF_ARM[arm.split("-nohist", maxsplit=1)[0].split("-swap", maxsplit=1)[0]],
        )
        / "work"
    )
    return base / ("smoke" if smoke else "") / kind / arm / name


def model_file(
    *, data_dir: Path, arm: str, month: str, seed: int, stage: int, smoke: bool = False
) -> Path:
    """Return the saved state after `stage` (1 to 3)."""
    return _file(
        data_dir=data_dir,
        arm=arm,
        kind="models",
        name=f"{month}_s{seed}_stage{stage}.pt",
        smoke=smoke,
    )


def fit_file(*, data_dir: Path, arm: str, month: str, seed: int, smoke: bool = False) -> Path:
    """Return the per-seed forecast file."""
    return _file(
        data_dir=data_dir, arm=arm, kind="fits", name=f"{month}_s{seed}.parquet", smoke=smoke
    )


def log_file(
    *, data_dir: Path, arm: str, month: str, seed: int, kind: str, smoke: bool = False
) -> Path:
    """Return a job's JSON log; `kind` is `fit` or `predict`."""
    return _file(
        data_dir=data_dir, arm=arm, kind="logs", name=f"{month}_s{seed}_{kind}.json", smoke=smoke
    )


def table_file(*, data_dir: Path, arm: str, month: str, version: str) -> Path:
    """Return the row table of an arm and fold; `version` is `era5`, `ens`, or `swap`."""
    return _file(data_dir=data_dir, arm=arm, kind="tables", name=f"{month}_{version}.npz")


def gate_file(*, data_dir: Path) -> Path:
    """Return the known-answer record that part 1 writes and every real fit requires."""
    return root(data_dir=data_dir, part=1) / "work" / "known_answers_passed.json"


def own_cache(*, data_dir: Path) -> Path:
    """Return F9's own dense-array cache, so that no file is written into experiment 713's folder."""
    return root(data_dir=data_dir, part=1) / "cache"


def jobs(*, phase: int) -> list[tuple[str, str, int]]:
    """Return the (arm, month, seed) fit jobs of a phase of design section 11 (phases 2 to 6 and 9)."""
    two = SCREEN_SEEDS
    if phase == 2:
        return [("Z", m, s) for m in SCREEN_FOLDS for s in two]
    if phase == 3:
        return [
            (a, m, s) for a in ("R", "R-A", "P90", "U", "U-A") for m in SCREEN_FOLDS for s in two
        ]
    if phase == 4:
        return [
            (a, m, s) for a in ("R", "R-A", "P90", "U", "U-A") for m in OTHER_FOLDS for s in two
        ]
    if phase == 5:
        return [(a, m, s) for a in ("U27", "U27-A") for m in SCREEN_FOLDS for s in two]
    if phase == 6:
        return [("UX", m, s) for m in UX_FOLDS for s in two]
    if phase == 9:
        return [(a, m, s) for a in ("R", "R-A", "U", "U-A") for m in ALL_FOLDS for s in (2, 3, 4)]
    msg = f"phase {phase} has no fit jobs (0 is the smoke fit, 1 the known-answer worlds, 7 the read-outs, 8 the reserve)"
    raise ValueError(msg)


FITS_BY_PHASE: Final = {0: 2, 1: 33, 2: 8, 3: 40, 4: 40, 5: 16, 6: 4, 7: 0, 8: 0, 9: 96}
HOURS_BY_PHASE: Final = {
    0: 0.4,
    1: 3.4,
    2: 1.0,
    3: 5.2,
    4: 5.2,
    5: 3.8,
    6: 0.5,
    7: 1.5,
    8: 1.5,
    9: 12.5,
}
"""Fits and GPU-process-hours per phase of design section 11 (phases 0 to 8 total 143 fits and 22.5 hours; phase 9 adds 96 and 12.5)."""


def projected_gpu_hours(
    *, phases: Sequence[int], measured_15_row_hours: float | None = None
) -> float:
    """Return the projected GPU-process-hours of `phases`, scaled by a measured 15-row fit time if given."""
    scale = 1.0 if measured_15_row_hours is None else measured_15_row_hours / FIT_HOURS_15_ROWS
    return round(scale * sum(HOURS_BY_PHASE[p] for p in phases), 2)


# ---------------------------------------------------------------------------------- code record


@functools.cache
def recipe_hashes() -> dict[str, str]:
    """Return the recipe hashes of experiment 713 (every recipe experiment's scripts)."""
    return a713.recipe_hashes()


def _module_path(*, name: str) -> Path:
    return (
        Path(__file__).parent / name
        if name.startswith("e00718")
        else Path(a713.__file__).parent / name
    )


@functools.cache
def code_hash() -> str:
    """Return a hash of F9's fit modules, the modules of experiment 713 that F9 imports, and the recipe modules."""
    h = hashlib.sha256()
    for name in (*CODE_FILES, *IMPORTED_713_FILES):
        h.update(_module_path(name=name).read_bytes())
    for name, digest in sorted(recipe_hashes().items()):
        h.update(f"{name}:{digest}".encode())
    return h.hexdigest()[:16]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unavailable"


def code_record() -> dict:
    """Return what a log records about the code: the hash, the git commit, whether the folder is dirty."""
    return {
        "code_hash": code_hash(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
    }


def check_code(
    *, earlier: dict, accept: Sequence[str], record: dict, what: str, kind: str = "state"
) -> None:
    """Refuse a state or a gate written by other code, unless an override names its exact old hash.

    `accept` holds the values of `--accept-code-change` (stage states, repeatable) or
    `--accept-gate-change` (the gate record). An override is appended to `record["code_overrides"]`
    with both hashes and both git commits. A record without a code hash cannot be overridden.
    """
    now, old = code_hash(), earlier.get("code_hash")
    if old is None:
        msg = f"{what} carries no code record: refit it"
        raise AssertionError(msg)
    if old == now:
        return
    flag = "--accept-gate-change" if kind == "gate" else "--accept-code-change"
    if old not in accept:
        msg = f"{what} was written by code with hash {old}, and the code now has hash {now}. After an edit that cannot change a fitted state, rerun with `{flag} {old}`."
        raise AssertionError(msg)
    entry = {
        "what": what,
        "kind": kind,
        "old_hash": old,
        "new_hash": now,
        "old_commit": earlier.get("git_commit", "unavailable"),
        "new_commit": _git("rev-parse", "HEAD"),
    }
    if entry not in record.setdefault("code_overrides", []):
        record["code_overrides"].append(entry)
        print("code override accepted:", json.dumps(entry), flush=True)


def install_own_cache(*, data_dir: Path) -> Path:
    """Point experiment 713's loaders at F9's own cache, seeded by copies of 713's files (copies, not hard links)."""
    import shutil

    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    own = own_cache(data_dir=data_dir)
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


def watched_713_files(*, data_dir: Path) -> dict[str, float]:
    """Return the modification times of experiment 713's `work` and `cache` files, for the unchanged-files assertion."""
    base = a713.root(data_dir=data_dir)
    out: dict[str, float] = {}
    for folder in (base / "work", base / "cache"):
        if folder.exists():
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    out[str(path)] = path.stat().st_mtime
    return out
