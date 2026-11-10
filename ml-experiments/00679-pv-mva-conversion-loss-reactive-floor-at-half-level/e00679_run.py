"""Experiment 679: run experiment 662's fit and probe code with this experiment's own work folder.

Experiment 662's modules are imported and edited not at all. This script points `e00662_common.WORK`
at experiment 679's private folder, links the 3-seed networks of the conditions that experiment 662
already trained (the twins, the signed world, and the power-factor world), and then calls the
requested 662 script's `main`.

Usage: python e00679_run.py link | fit ARGS... | probe ARGS...
"""

import sys
from pathlib import Path
from typing import Final

import e00662_common as k  # ty: ignore[unresolved-import]
import e00662_fit as fit  # ty: ignore[unresolved-import]
import e00662_probe as probe  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00679-pv-mva-conversion-loss-reactive-floor-at-half-level"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
REUSED: Final[tuple[str, ...]] = (
    "sig_twin", "sig_fleet_l05_r0", "sig_fleet_l1_r0",
    "pf_twin", "pf_fleet_l05_r0", "pf_fleet_l1_r0",
    "floor_twin", "q0_twin",
)  # fmt: skip
"""Conditions whose networks experiment 662 trained at 2026-06 (seeds 0 to 2 are reused)."""
NEW: Final[tuple[str, ...]] = ("floor_fleet_l05_r0", "q0_fleet_l05_r0", "q0_fleet_l1_r0")
"""Conditions this experiment trains."""


def link() -> None:
    """Symlink experiment 662's networks for the reused conditions into this experiment's folder."""
    for cond in REUSED:
        for seed in SEEDS:
            src = k.weights_path(cutoff="2026-06", condition=cond, seed=seed)
            dst = WORK / "pretrained" / "2026-06" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            assert src.exists(), src
            if not dst.is_symlink():
                dst.symlink_to(src)


def main() -> None:
    """Dispatch to experiment 662's fit or probe with the work folder replaced."""
    command, *rest = sys.argv[1:]
    if command == "link":
        link()
        return
    k.WORK = WORK
    sys.argv = [sys.argv[0], *rest]
    if command == "fit":
        fit.main()
    elif command == "probe":
        probe.main()
    else:
        raise SystemExit(command)


if __name__ == "__main__":
    main()
