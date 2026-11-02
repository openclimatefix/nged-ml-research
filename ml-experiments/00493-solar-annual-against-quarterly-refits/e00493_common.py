"""Shared constants of experiment 493 (annual against quarterly refits for the solar network)."""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00493-solar-annual-against-quarterly-refits"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
SCORE_DIR: Final[Path] = DATA_DIR / "scores"
E430_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
)
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
"""The six seeds experiment 430 stored for `g0_onepass` and `a2_member`."""
BASE_ARMS: Final[tuple[str, ...]] = ("g0_onepass", "a2_member")
ANNUAL_MONTHS: Final[int] = 12
"""One refit predicts the runs of all 12 scored months."""
