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
"""The modules whose edits change a fitted state."""


def code_hash() -> str:
    """Return a hash of the modules that decide a fitted state, written into each stage log and checked on resume."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for name in FIT_CODE_FILES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


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
