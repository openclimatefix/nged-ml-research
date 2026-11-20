# ruff: noqa: E501
"""Experiment 713: arm specifications, the fold lists, the file layout, and the job list (design 1.2, 5.2).

An arm name is one of `M1`, `M2`, `M3` (the screen arms), `P`, `A`, `K`, `N`, `E`, `G` (the arms
built on `M*`), or `C` (the Candidate, never refitted here except through `chain_seeds.sh`). `M*`
is read from `work/selection.json`, which `e00713_select.py` writes from the screen's fit logs.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import e00713_model as mdl

NAME: Final = "00713-msa-axial-attention-demand-network-part-1"
EncoderMode = Literal["frozen", "trained", "random"]

ALL_FOLDS: Final = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
SCREEN_FOLDS: Final = ("2025-11", "2026-01", "2026-03", "2026-06")
OTHER_FOLDS: Final = tuple(m for m in ALL_FOLDS if m not in SCREEN_FOLDS)
BANK_FOLDS: Final = ("2025-11", "2026-01", "2026-04", "2026-06")
"""The four fold months of experiment 709's stored injected worlds."""
SCREEN_SEEDS: Final = (0, 1)
HEADLINE_SEEDS: Final = (0, 1, 2, 3, 4)
ROW_SET_ID: Final = "backtest_2025_11_to_2026_06_daily_12utc_before_holdout"


@dataclass(frozen=True)
class ArmSpec:
    """What an arm does, with `M*` resolved.

    Attributes:
        name: `M1`, `M2`, `M3`, `P`, `A`, `K`, `N`, `E`, or `G`.
        letter: The key of `e00713_model.ARM_SWITCHES`.
        encoders: `frozen` (loaded, never trained), `trained` (loaded, trained), or `random`.
        placebo: Replace the 14 history rows with 14 random past days.
    """

    name: str
    letter: str
    encoders: EncoderMode
    placebo: bool

    @property
    def switches(self) -> mdl.ArmSwitches:
        """Return the architectural switches of the arm."""
        return mdl.ARM_SWITCHES[self.letter]


SCREEN_ARMS: Final[dict[str, tuple[str, EncoderMode]]] = {
    "M1": ("M", "frozen"),
    "M2": ("M", "trained"),
    "M3": ("M", "random"),
}


FIT_CODE_FILES: Final = (
    "e00713_arms.py", "e00713_batch.py", "e00713_fit.py", "e00713_inputs.py", "e00713_layout.py",
    "e00713_model.py", "e00713_train.py", "e00713_variants.py",
)  # fmt: skip
"""The modules of this experiment whose edits can change a fitted state."""
RECIPE_MODULES: Final = (
    "e00087_common", "e00397_models", "e00397_prep", "e00422_prep", "e00428_data", "e00448_run",
    "e00510_common", "e00558_lags", "e00559_model", "e00560_data", "e00574_fit", "e00574_model",
    "e00632_data", "e00667_cand", "e00667_fold",
)  # fmt: skip
"""Imported recipe modules of earlier experiments; their file hashes enter the code hash and the logs."""


def recipe_hashes() -> dict[str, str]:
    """Return the sha256 (first 16 hex digits) of the file of each imported recipe module, or `missing`."""
    import importlib.util

    out = {}
    for name in RECIPE_MODULES:
        spec = importlib.util.find_spec(name)
        out[name] = (
            hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()[:16]
            if spec is not None and spec.origin
            else "missing"
        )
    return out


def code_hash() -> str:
    """Return a hash of this experiment's fit modules and of the imported recipe modules."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in FIT_CODE_FILES:
        h.update((here / name).read_bytes())
    for name, digest in sorted(recipe_hashes().items()):
        h.update(f"{name}:{digest}".encode())
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
    """Return what a log records about the code: the hash, the git commit, whether the folder is dirty."""
    return {
        "code_hash": code_hash(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", ".")),
    }


def check_code(*, earlier: dict, accept: str | None, record: dict, what: str) -> None:
    """Refuse a state or a gate written by other code, unless the override names its old hash.

    `accept` is the value of `--accept-code-change`: the hash of the earlier code. When it equals the
    earlier hash, the override is appended to `record["code_overrides"]` with both hashes, both git
    commits, and the diff stat of the fit modules between the old commit and the working tree.
    """
    now = code_hash()
    old = earlier.get("code_hash")
    if old == now:
        return
    if accept is None or accept != old:
        msg = (
            f"{what} was written by code with hash {old}, and the code now has hash {now}. "
            "After an edit that cannot change a fitted state, rerun with `--accept-code-change "
            f"{old}`; the override is logged with a diff stat."
        )
        raise AssertionError(msg)
    old_commit = earlier.get("git_commit", "unavailable")
    stat = (
        _git("diff", "--stat", old_commit, "--", *FIT_CODE_FILES)
        if old_commit != "unavailable"
        else "unavailable"
    )
    entry = {
        "what": what, "old_hash": old, "new_hash": now, "old_commit": old_commit,
        "new_commit": _git("rev-parse", "HEAD"), "diff_stat": stat,
    }  # fmt: skip
    record.setdefault("code_overrides", []).append(entry)
    print("code override accepted:", json.dumps(entry), flush=True)


def root(*, data_dir: Path) -> Path:
    """Return the experiment's data folder under `data_dir`."""
    return data_dir / NAME


def selection_file(*, data_dir: Path) -> Path:
    """Return the file that holds the choice of `M*`."""
    return root(data_dir=data_dir) / "work" / "selection.json"


def m_star(*, data_dir: Path) -> str:
    """Return the selected screen arm (`M1`, `M2`, or `M3`), read from the selection file."""
    path = selection_file(data_dir=data_dir)
    assert path.exists(), "run e00713_select.py on the screen's fit logs first"
    return json.loads(path.read_text())["m_star"]


def resolve(*, arm: str, data_dir: Path) -> ArmSpec:
    """Return the specification of `arm`; arms other than the screen arms inherit `M*`'s encoders."""
    if arm in SCREEN_ARMS:
        letter, encoders = SCREEN_ARMS[arm]
        return ArmSpec(name=arm, letter=letter, encoders=encoders, placebo=False)
    assert arm in {"P", "A", "K", "N", "E", "G"}, arm
    _, encoders = SCREEN_ARMS[m_star(data_dir=data_dir)]
    return ArmSpec(name=arm, letter=arm, encoders=encoders, placebo=arm == "P")


def model_file(*, data_dir: Path, arm: str, month: str, seed: int, stage: int) -> Path:
    """Return the saved state after `stage` (1 to 3)."""
    return root(data_dir=data_dir) / "work" / "models" / arm / f"{month}_s{seed}_stage{stage}.pt"


def fit_file(*, data_dir: Path, arm: str, month: str, seed: int) -> Path:
    """Return the per-seed forecast file."""
    return root(data_dir=data_dir) / "work" / "fits" / arm / f"{month}_s{seed}.parquet"


def log_file(*, data_dir: Path, arm: str, month: str, seed: int, kind: str) -> Path:
    """Return a job's JSON log; `kind` is `fit` or `predict`."""
    return root(data_dir=data_dir) / "work" / "logs" / arm / f"{month}_s{seed}_{kind}.json"


def jobs(*, phase: int, data_dir: Path) -> list[tuple[str, str, int]]:
    """Return the (arm, month, seed) jobs of a phase of section 5.2.

    Phases 1 (screen), 2 (`P` on the screen folds), 3 (`M*` and `P` on the other folds), 4 to 7 (`A`,
    `K`, `N`, `E`), 9 (`M*` and `P`, seeds 2 to 4, all folds), and 13 (`G`) are fit jobs. `M*` is
    stored under the name of the screen arm it resolves to.
    """
    two = SCREEN_SEEDS
    if phase == 1:
        return [(a, m, s) for a in SCREEN_ARMS for m in SCREEN_FOLDS for s in two]
    if phase == 2:
        return [("P", m, s) for m in SCREEN_FOLDS for s in two]
    star = m_star(data_dir=data_dir)
    if phase == 3:
        return [(a, m, s) for a in (star, "P") for m in OTHER_FOLDS for s in two]
    letters = {4: "A", 5: "K", 6: "N", 7: "E", 13: "G"}
    if phase in letters:
        return [(letters[phase], m, s) for m in ALL_FOLDS for s in two]
    if phase == 9:
        return [(a, m, s) for a in (star, "P") for m in ALL_FOLDS for s in (2, 3, 4)]
    msg = f"phase {phase} has no fit jobs (8 is chain_seeds.sh; 10 to 12 are prediction-only)"
    raise ValueError(msg)
