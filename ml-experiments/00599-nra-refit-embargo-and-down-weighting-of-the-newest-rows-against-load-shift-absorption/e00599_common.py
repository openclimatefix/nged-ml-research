# ruff: noqa: E501
"""Experiment 599: arms, worlds, and paths for the refit embargo and down-weighting test.

The cases, worlds, designs, and event helpers are experiment 587's (`e00587_common`, imported
unchanged). The base arm is experiment 595's A1; E0 is that arm as stored by 595. The three new arms
change one training-side setting of the daily fine-tune each (see `VARIANTS`).
"""

from pathlib import Path
from typing import Final

import e00595_common as c595  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = (
    "00599-nra-refit-embargo-and-down-weighting-of-the-newest-rows-against-load-shift-absorption"
)
OUT = ML_EXPERIMENTS_DATA_DIR / NAME
TAG: Final[str] = "P5"
"""The fit folder tag of every arm (experiment 591's A1 fits under experiment 574's arm tag)."""
REFERENCE_ARM: Final[str] = "E0"
"""A1 as stored by experiment 595."""
DOWN_WEIGHT: Final[float] = 0.5
"""Multiplier of the fine-tune weight of the rows in the down-weighted window."""
DOWN_WEIGHT_DAYS: Final[int] = 7
"""Length of the down-weighted window, which starts where the embargo ends."""
VARIANTS: Final[dict[str, tuple[int, bool]]] = {
    "E1": (7, False),
    "E2w": (3, True),
    "E3": (7, True),
    "E0chk": (3, False),
}
"""Arm name to (embargo days of the daily fine-tune, whether the newest 7 remaining days are halved).
`E0chk` is a no-change smoke run that must reproduce experiment 595's A1."""
NEW_ARMS: Final[tuple[str, ...]] = ("E1", "E2w", "E3")
ALL_ARMS: Final[tuple[str, ...]] = (REFERENCE_ARM, *NEW_ARMS)


def world_dir(*, arm: str, world: str, on: str | None = None) -> Path:
    """Return the folder of an arm's fits in a world: 595's `A1` folder for E0, this experiment's otherwise."""
    if arm == REFERENCE_ARM:
        return c595.world_dir(arm="A1", world=world, on=on)
    return OUT / "work" / arm / (world if on is None else f"{world}_on_{on}")
