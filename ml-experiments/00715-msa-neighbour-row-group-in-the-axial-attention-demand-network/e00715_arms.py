# ruff: noqa: E501, PT018, D205, D209
"""Experiment 715: the arms, the folds, the file layout, the job list, and the code hash (design
sections 4, 6, and 11.1).

An arm name is one of `F4` (the real neighbours), `F4-R` (the rewired placebo), `F4-T` (the
time-shuffled placebo), `F4-W` and `F4-WR` (the real and rewired neighbours trained in the mirror
world), `F4-0` (F4's training loop with the neighbour path absent), and `F4E` (the real neighbours with
arm `E`'s convolution stem on the host rows). `M*` and the Candidate `C` are experiment 713's stored
arms
and are never refitted here.

**The code hash covers every module that F4 imports, not only the fit modules.** It joins the hash of
F4's own fit modules, experiment 713's `code_hash()` (its nine fit modules and the 41 recipe folders),
experiment 713's `e00713_score.py`, `e00713_protocol.py`, `e00713_era.py`, and `e00713_attention.py`,
and experiment 714's `e00714_generator2.py` and `e00714_common.py`, which the second generator's adapter
imports. Before a warm start, F4 asserts that the hash logged with the stage 1 state of `M*` equals
experiment 713's `code_hash()` as F4's checkout computes it. A later commit to experiment 713
therefore cannot change F4 silently. Any merge to `main` that touches one of the 41 recipe folders
changes the hash and refuses the warm start: read the diff, and name the override only if the diff
cannot change a fitted state.
"""

import functools
import hashlib
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00715_batch as fb
import e00715_neighbours as nb
import numpy as np

NAME: Final = "00715-msa-neighbour-row-group-in-the-axial-attention-demand-network"
WorldKind = Literal["real", "mirror"]
ALL_FOLDS: Final = a713.ALL_FOLDS
SCREEN5_FOLDS: Final = ("2025-11", "2026-01", "2026-03", "2026-04", "2026-06")
"""Experiment 713's four screen folds plus 2026-04, because the injected worlds need the four bank folds."""
OTHER3_FOLDS: Final = tuple(m for m in ALL_FOLDS if m not in SCREEN5_FOLDS)
BANK_FOLDS: Final = a713.BANK_FOLDS
SEEDS: Final = (0, 1)
ROW_SET_ID: Final = a713.ROW_SET_ID
CEILING_GPU_HOURS: Final = 20.0
"""Stop and ask the manager before crossing this projected total (design section 6)."""

OWN_FIT_FILES: Final = (
    "e00715_arms.py", "e00715_batch.py", "e00715_fit.py", "e00715_model.py", "e00715_neighbours.py",
    "e00715_predict.py", "e00715_train.py", "e00715_worlds.py", "e00715_era.py",
)  # fmt: skip
"""The modules of this experiment whose edits can change a fitted state."""
IMPORTED_713_FILES: Final = (
    "e00713_score.py", "e00713_protocol.py", "e00713_era.py", "e00713_attention.py",
)  # fmt: skip
IMPORTED_714_FILES: Final = ("e00714_generator2.py", "e00714_common.py")


@dataclass(frozen=True)
class ArmSpec:
    """What an arm does.

    Attributes:
        name: The arm's name.
        slots: What the six slots hold: `real`, `rewired`, `shuffled`, or `none`.
        world: The training world of stage 3: `real` (experiment 713's) or `mirror` (design section 5.3).
        neighbour_path: Whether the network has the neighbour path (False only for `F4-0`).
        stem: Arm `E`'s convolution stem on the host rows (`F4E`).
        folds: The fold months the arm is fitted on.
    """

    name: str
    slots: nb.SlotMode
    world: WorldKind
    neighbour_path: bool
    stem: bool
    folds: tuple[str, ...]

    @property
    def switches(self) -> mdl.ArmSwitches:
        """Return the architectural switches of the host rows (arm `M`, or arm `E` with its stem)."""
        return mdl.ARM_SWITCHES["E" if self.stem else "M"]

    def start_arm(self, *, data_dir: Path) -> str:
        """Return experiment 713's arm whose stage 1 state starts the fit: `M*` (resolved), or `E`."""
        return "E" if self.stem else a713.m_star(data_dir=data_dir)


ARMS: Final[dict[str, ArmSpec]] = {
    "F4": ArmSpec("F4", "real", "real", True, False, ALL_FOLDS),
    "F4-R": ArmSpec("F4-R", "rewired", "real", True, False, ALL_FOLDS),
    "F4-T": ArmSpec("F4-T", "shuffled", "real", True, False, SCREEN5_FOLDS),
    "F4-W": ArmSpec("F4-W", "real", "mirror", True, False, BANK_FOLDS),
    "F4-WR": ArmSpec("F4-WR", "rewired", "mirror", True, False, BANK_FOLDS),
    "F4-0": ArmSpec("F4-0", "none", "real", False, False, SCREEN5_FOLDS),
    "F4E": ArmSpec("F4E", "real", "real", True, True, SCREEN5_FOLDS),
}


def resolve(*, arm: str) -> ArmSpec:
    """Return the specification of `arm`."""
    assert arm in ARMS, f"unknown arm {arm!r}: one of {sorted(ARMS)}"
    return ARMS[arm]


# ---------------------------------------------------------------------- files


def root(*, data_dir: Path) -> Path:
    """Return the experiment's data folder under `data_dir`."""
    return data_dir / NAME


def model_file(*, data_dir: Path, arm: str, month: str, seed: int, stage: int) -> Path:
    """Return the saved state after `stage` (2 or 3); stage 1 is experiment 713's."""
    assert stage in {2, 3}, stage
    return root(data_dir=data_dir) / "work" / "models" / arm / f"{month}_s{seed}_stage{stage}.pt"


def fit_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file (`arm` may be `F4-nohist` or `F4-nonbr`, the read-outs of section 7.4)."""
    return root(data_dir=data_dir) / "work" / "fits" / arm / f"{month}_s{seed}.parquet"


def log_file(*, data_dir: Path, arm: str, month: str, seed: int, kind: str) -> Path:
    """Return a job's JSON log; `kind` is `fit` or `predict`."""
    return root(data_dir=data_dir) / "work" / "logs" / arm / f"{month}_s{seed}_{kind}.json"


def stage_log_file(*, data_dir: Path, arm: str, month: str, seed: int, stage: int) -> Path:
    """Return the log of one stage of one of this experiment's fits."""
    path = log_file(data_dir=data_dir, arm=arm, month=month, seed=seed, kind="fit")
    return path.with_name(f"{path.stem}_stage{stage}.json")


def stage1_state_file(*, data_dir: Path, start_arm: str, month: str, seed: int) -> Path:
    """Return experiment 713's stage 1 state of `start_arm` (`M*` resolved, or `E`), the start of every F4 fit."""
    return a713.model_file(data_dir=data_dir, arm=start_arm, month=month, seed=seed, stage=1)


def stage1_log_file(*, data_dir: Path, start_arm: str, month: str, seed: int) -> Path:
    """Return the log of experiment 713's stage 1 fit (the code record that the warm start checks)."""
    path = a713.log_file(data_dir=data_dir, arm=start_arm, month=month, seed=seed, kind="fit")
    return path.with_name(f"{path.stem}_stage1.json")


def gate_file(*, data_dir: Path) -> Path:
    """Return the file that `e00715_known_answer.py` writes when its checks pass."""
    return root(data_dir=data_dir) / "work" / "known_answers_passed.json"


def own_cache() -> Path:
    """Return F4's dense-array cache folder (nothing is written into experiment 713's or 632's)."""
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    return root(data_dir=ML_EXPERIMENTS_DATA_DIR) / "cache"


# ---------------------------------------------------------------------- jobs


def jobs(*, phase: int, data_dir: Path | None = None) -> list[tuple[str, str, int]]:
    """Return the (arm, month, seed) fit jobs of a phase of design section 6.

    Phase 0 is the timing fit (`F4`, 2025-11, seed 0). Phase 1 is the screen (`F4`, `F4-R`, `F4-T`
    on the 5 screen folds). Phase 11 is phase 1b (`F4-0` on the same 5 folds). Phase 2 is `F4` and
    `F4-R` on the other 3 folds. Phase 3 is `F4-W` and `F4-WR` on the 4 bank folds. Phase 6 is `F4E`.
    """
    del data_dir
    table = {
        0: [("F4", "2025-11", 0)],
        1: [(a, m, s) for a in ("F4", "F4-R", "F4-T") for m in SCREEN5_FOLDS for s in SEEDS],
        11: [("F4-0", m, s) for m in SCREEN5_FOLDS for s in SEEDS],
        2: [(a, m, s) for a in ("F4", "F4-R") for m in OTHER3_FOLDS for s in SEEDS],
        3: [(a, m, s) for a in ("F4-W", "F4-WR") for m in BANK_FOLDS for s in SEEDS],
        6: [("F4E", m, s) for m in SCREEN5_FOLDS for s in SEEDS],
    }
    if phase not in table:
        msg = f"phase {phase} has no fit jobs; one of {sorted(table)} (phases 4 and 5 are prediction-only)"
        raise ValueError(msg)
    return table[phase]


FITS_BY_PHASE: Final = {0: 1, 1: 30, 11: 10, 2: 12, 3: 16, 6: 10}
"""The fit counts of design section 6."""


def projected_gpu_hours(
    *, seconds_per_fit: float, include_optional: bool = False
) -> dict[str, float]:
    """Return the projected GPU-process-hours from the measured time of one fit (design section 6).

    Phases 1, 11, 2, and 3 are fits; phase 4 is 430,080 forecasts at 2 minutes per 6,944 plus 0.5 for
    192 panel loads; phase 5 is 0.5. The reruns of stages that hit the cap are kept at 2.3.
    """
    per_fit = seconds_per_fit / 3600.0
    fits = {
        p: n * per_fit for p, n in FITS_BY_PHASE.items() if p != 0 and (include_optional or p != 6)
    }
    phase4 = 430_080 / 6_944 * 2.0 / 60.0 + 0.5
    total = sum(fits.values()) + phase4 + 0.5
    return {"fits": sum(fits.values()), "phase4": phase4, "phase5": 0.5, "total": total, "reruns_kept": 2.3,
            "ceiling": CEILING_GPU_HOURS, "above_ceiling": total + 0.0 > CEILING_GPU_HOURS}  # fmt: skip


# ---------------------------------------------------------------------- code hash


def _module_file(*, name: str) -> Path:
    """Return the path of a top-level module on `sys.path` without importing it."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, (
        f"module {name} is not on PYTHONPATH: the code hash needs it (see run.sh, EXTRA_PYTHONPATH)"
    )
    return Path(spec.origin)


@functools.cache
def resolved_recipe_hashes() -> dict[str, str]:
    """Return the sha256 (16 hex digits) of the recipe files that `importlib` resolves in this process.

    Experiment 713's `recipe_hashes` reads the folders of its own checkout, while this process imports
    the copies on `PYTHONPATH`, which can be another checkout. A recipe module that cannot be found
    is recorded as `missing`.
    """
    out = {}
    for name in a713.recipe_hashes():
        if not name.endswith(".py"):
            continue
        spec = importlib.util.find_spec(name.removesuffix(".py"))
        origin = None if spec is None else spec.origin
        out[name] = (
            "missing"
            if origin is None
            else hashlib.sha256(Path(origin).read_bytes()).hexdigest()[:16]
        )
    return out


def check_recipe_paths() -> None:
    """Assert that the recipe files this process imports equal those that experiment 713's hash covers."""
    theirs = {k: v for k, v in a713.recipe_hashes().items() if k.endswith(".py")}
    mine = resolved_recipe_hashes()
    differing = sorted(k for k in theirs if mine.get(k) != theirs[k])
    assert not differing, (
        f"the recipe modules imported here differ from the files experiment 713's hash covers: {differing}"
    )


@functools.cache
def code_hash() -> str:
    """Return the hash of F4's fit modules, experiment 713's hash, and the imported modules of 713 and 714."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in OWN_FIT_FILES:
        h.update((here / name).read_bytes())
    h.update(f"e00713:{a713.code_hash()}".encode())
    for name, digest in sorted(resolved_recipe_hashes().items()):
        h.update(f"{name}:{digest}".encode())
    for name in IMPORTED_713_FILES:
        h.update(_module_file(name=name.removesuffix(".py")).read_bytes())
    for name in IMPORTED_714_FILES:
        h.update(_module_file(name=name.removesuffix(".py")).read_bytes())
    return h.hexdigest()[:16]


def _git(*args: str) -> str:
    """Return the output of a git command run in this folder, or `unavailable`."""
    import subprocess

    try:
        return subprocess.run(
            ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unavailable"


def code_record() -> dict:
    """Return what a log records about the code: the hash, experiment 713's hash, the commit,
    whether the folder is dirty."""
    return {
        "code_hash": code_hash(),
        "e00713_code_hash": a713.code_hash(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
    }


def check_code(
    *, earlier: dict, accept: Sequence[str], record: dict, what: str, kind: str = "state"
) -> None:
    """Refuse a state or a gate written by other code, unless an override names its exact old hash.

    `accept` holds the values of `--accept-code-change` (for stage states, repeatable) or the value of
    `--accept-gate-change` (for the gate record). **No override is accepted when experiment 713's code
    hash differs from the one the record holds.** An override is appended to `record["code_overrides"]`
    with both hashes, both git commits, and the diff stat of the fit modules. A state or gate without
    a code record cannot be overridden.
    """
    now = code_hash()
    old = earlier.get("code_hash")
    if old is None:
        msg = f"{what} carries no code record: " + (
            "rerun e00715_known_answer.py" if kind == "gate" else "refit it"
        )
        raise AssertionError(msg)
    if old == now:
        return
    flag = "--accept-gate-change" if kind == "gate" else "--accept-code-change"
    old_713, now_713 = earlier.get("e00713_code_hash"), a713.code_hash()
    if old_713 != now_713:
        msg = (
            f"{what} was written with experiment 713's code at hash {old_713}, and that code now has hash "
            f"{now_713}. A change to experiment 713's code can change an F4 network, so no override "
            f"(`{flag}`) is accepted: refit, or rerun the gate."
        )
        raise AssertionError(msg)
    if old not in accept:
        msg = (
            f"{what} was written by code with hash {old}, and the code now has hash {now}. "
            f"After an edit that cannot change a fitted state, rerun with `{flag} {old}`; "
            "the override is logged with a diff stat."
        )
        raise AssertionError(msg)
    old_commit = earlier.get("git_commit", "unavailable")
    stat = (
        _git("diff", "--stat", old_commit, "--", *OWN_FIT_FILES)
        if old_commit != "unavailable"
        else "unavailable"
    )
    entry = {
        "what": what, "kind": kind, "old_hash": old, "new_hash": now, "old_commit": old_commit,
        "new_commit": _git("rev-parse", "HEAD"), "diff_stat": stat,
    }  # fmt: skip
    if entry not in record.setdefault("code_overrides", []):
        record["code_overrides"].append(entry)
        print("code override accepted:", json.dumps(entry), flush=True)


def check_warm_start(
    *, stage1_log: dict, accept: Sequence[str], record: dict, what: str
) -> None:  # fmt: skip
    """Assert that the stage 1 state of experiment 713 was fitted by the code that F4's checkout computes.

    The state's log carries experiment 713's `code_hash`; it must equal `e00713_arms.code_hash()` as
    this checkout computes it. A state fitted by other code is refused unless `accept` names its hash.
    """
    check_recipe_paths()
    a713.check_code(earlier=stage1_log, accept=accept, record=record, what=what)


# ---------------------------------------------------------------------- sources


def capacity_mw() -> np.ndarray:
    """Return the effective capacity of each Primary in megawatts, in Primary-index order (`site_map.csv`)."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    sites = k578.sites().sort("primary")
    assert sites["label"].to_list() == list(nb.LABELS), sites["label"].to_list()
    return sites["capacity_mw"].to_numpy().astype(np.float64)


def install_own_cache() -> Path:
    """Point experiment 713's loaders at F4's own dense-array cache, seeded by copies of 713's files.

    The loaders write the dense ERA5 and B1 arrays into `e00713_inputs.CACHE` the first time they build
    them. F4 redirects the folder before the first call, so nothing is written into experiment 713's
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


def f4_source(
    *,
    data: inp.FoldData,
    table: nb.NeighbourTable,
    mode: nb.SlotMode,
    rewire_seed: int,
) -> fb.F4Source:
    """Return the source that reads the cleaned power as host history and as neighbour power (modes
    `stage2` and `predict_real`)."""
    scale = data.target_std.astype(np.float64) * capacity_mw()
    return fb.F4Source(
        host=data.source, neighbour_power=data.source.power_z, mw_scale=scale, table=table,
        mode=mode, rewire_seed=rewire_seed, label="cleaned",
    )  # fmt: skip
