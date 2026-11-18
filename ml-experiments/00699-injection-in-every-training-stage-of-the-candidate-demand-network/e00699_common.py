# ruff: noqa: E501
"""Experiment 699: constants shared by the scripts of this folder (folds, seeds, arms, data folders)."""

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00699-injection-in-every-training-stage-of-the-candidate-demand-network"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
"""The data folder of this experiment; every output lives under it."""
NAME_677: Final = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT_677: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME_677
"""Experiment 677's data folder: read only, never written."""
HOLDOUT: Final = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time on or after this instant is read by any score."""
FOLDS: Final = (
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
PHASE_A_FOLDS: Final = ("2025-11", "2026-01", "2026-04", "2026-06")
PHASE_B_FOLDS: Final = tuple(m for m in FOLDS if m not in PHASE_A_FOLDS)
A0_FOLD: Final = "2026-01"
SEEDS: Final = tuple(range(8))
STORED_SEEDS: Final = (0, 1, 2)
STAGE_SEED_BASE: Final = {"stage1": 5_000_000, "stage2": 2_000_000, "stage3": 0}
"""Each stage's world seed is this base plus the fold month as a number (such as 202511)."""
SEED_RANGES: Final = {
    "stage3": (202511, 202606),
    "bank660": (862511, 862606),
    "stage2": (2202511, 2202606),
    "stage1": (5202511, 5202606),
    "bank581": (9202511, 9202606),
}
BANK_BASE_581: Final = 9_000_000
BANK_BASE_660: Final = 660_000
SHARE_RANGE: Final = (0.04, 0.10)
VERSION: Final = {"5w": "with_injected", "5wo": "without_injected"}
MARGIN_COPIED_SHARE: Final = 0.012
LEVEL_FAMILY: Final = 100.0 - 5.0 / 3.0
"""The level of each family of three contrasts, in percent (98.33)."""


@dataclass(frozen=True)
class Arm:
    """One arm: the sub-folder of `roots/` that holds it and the ladder step name inside the folder."""

    key: str
    label: str
    root: str
    step: str
    stored: bool = False

    @property
    def file_stem(self) -> str:
        """Return the stem of the exported forecast file (without the seed tag)."""
        return f"step{self.step}_{VERSION[self.step]}_sp1"

    @property
    def folder(self) -> Path:
        """Return the arm's sub-folder of `roots/`."""
        return ROOT / "roots" / self.root


ARMS: Final[dict[str, Arm]] = {
    "none": Arm("none", "No injection (refitted)", "clean", "5wo"),
    "s3": Arm("s3", "Stage 3 only (refitted)", "clean", "5w"),
    "s23": Arm("s23", "Stages 2 and 3", "stage2", "5w"),
    "s12": Arm("s12", "Stages 1 and 2", "early", "5wo"),
    "s123": Arm("s123", "Stages 1 to 3", "early", "5w"),
    "none_stored": Arm("none_stored", "No injection (stored)", "stored", "5wo", stored=True),
    "s3_stored": Arm("s3_stored", "Stage 3 only (stored)", "stored", "5w", stored=True),
}
NEW_ARMS: Final = ("none", "s3", "s23", "s12", "s123")


def seeds_of(*, arm: Arm) -> tuple[int, ...]:
    """Return the seeds that the arm holds."""
    return STORED_SEEDS if arm.stored else SEEDS


def bank_folder(*, arm: Arm, month: str, suffix: str = "") -> Path:
    """Return the folder of an arm's bank files for one fold (`suffix` is `_b660` for the second bank)."""
    return arm.folder / f"P0_D50_{month}_{arm.step}_sp1{suffix}" / "work" / "bank"


def data_root_check() -> None:
    """Assert that the data folder is a private one outside the repository."""
    assert os.environ.get("DATA_PATH_INTERNAL"), "set DATA_PATH_INTERNAL"
