# ruff: noqa: E501
"""Experiment 595: cases, worlds, and paths for the target-side absorption of arms C1 and A1.

The cases, worlds, designs, and event helpers are experiment 587's (`e00587_common`, imported
unchanged): the 12 injected designs, the verified D5/D16 transfer, and the event world `E` and
no-event world `N` of each case. What this module adds is the two arms and their own output
folder. The reference arms (M0, D5, and P5) are read from experiment 587's stored forecasts.
"""

from pathlib import Path
from typing import Final

import e00587_common as c587  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00595-nra-target-side-absorption-for-p5-lag-encoder-with-synthetic-shifts"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME
NEW_ARMS: Final[dict[str, tuple[str, str]]] = {"C1": ("584", "P5"), "A1": ("591", "P5")}
"""Arm name to (the experiment whose arm code is wrapped, the fit folder's tag under `work/fits`)."""
REFERENCE_ARMS: Final[tuple[str, ...]] = ("M0", "D5", "P5")
"""Arms whose forecasts are read from experiment 587's data folder."""
ALL_ARMS: Final[tuple[str, ...]] = (*REFERENCE_ARMS, *NEW_ARMS)


def world_dir(*, arm: str, world: str, on: str | None = None) -> Path:
    """Return the folder of an arm's fits in a world: this experiment's for C1 and A1, 587's for the rest."""
    root = OUT if arm in NEW_ARMS else c587.OUT
    return root / "work" / arm / (world if on is None else f"{world}_on_{on}")


def tag(*, arm: str) -> str:
    """Return the fit folder tag of an arm."""
    return NEW_ARMS[arm][1] if arm in NEW_ARMS else c587.ARMS[arm][1]
