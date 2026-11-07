# ruff: noqa: E501
"""Shared pieces of experiment 643: paths, arms, and the rearranged capacity parameters of the permutations.

Experiment 639's rows, networks, stored free and `cal` predictions, and training loop are reused unchanged.
"""

from pathlib import Path
from typing import Final

import e00634_common as k634  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = (
    "00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations"
)
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
SHIFTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
STAGE1_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
STAGE2_SEEDS: Final[tuple[int, ...]] = (5, 6, 7)
MDE: Final[dict[str, float]] = {"real": 0.067, "step": 0.181}
"""Experiment 639's minimum detectable effects (CRPS points, sunny middays). The bars do not change."""
N_CONTRASTS: Final = 5


def prediction_path(*, cutoff: str, world: str, arm: str, seed: int) -> Path:
    """Return where an arm's test-window quantiles are stored.

    Arms `free` and `cal` (seeds 0 to 4) and `perm2` (seeds 0 to 4, experiment 639's `perm`) live in experiment 639's folder.
    """
    own = WORK / "predictions" / cutoff / f"{world}_{arm}_s{seed}.npz"
    if own.exists():
        return own
    old_arm = "perm" if arm == "perm2" else arm
    old = k639.prediction_path(cutoff=cutoff, world=world, arm=old_arm, seed=seed)
    if arm in {"free", "cal", "perm2"} and old.exists():
        return old
    return own


def weights_path(*, cutoff: str, world: str, arm: str, seed: int) -> Path:
    """Return where a network (and its `.json` statistics) is stored."""
    return WORK / "pretrained" / cutoff / f"{world}_{arm}_s{seed}.pt"


def shifted_parameters(*, cutoff: str, shift: int) -> dict:
    """Return experiment 639's `cal` parameters with the Primaries' timelines cyclically shifted by `shift` places.

    The same rearrangement as experiment 639's `perm` arm: Primary `s` gets the megawatt timeline of the
    Primary `shift` places on, by rescaling `k` with the ratio of the two Primaries' table values.
    """
    base = k639.capacity_parameters(cutoff=cutoff, arm="cal")
    table = k639.r4_table()
    ref = k639.last_complete_season_year(cutoff=cutoff)
    sites = list(k634.SIX)
    out = {}
    for i, s in enumerate(sites):
        other = sites[(i + shift) % len(sites)]
        o = base[other]
        k_own = float(np.clip(o.k * table[(other, ref)] / table[(s, ref)], 0.0, None))
        out[s] = k639.SiteCapacity(k=k_own, d=o.d, rel_sd_k=o.rel_sd_k, rel_sd_d=o.rel_sd_d)
    return out
