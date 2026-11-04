"""Pretrain the W-T network for the folds with no stored weights.

The network reads 37 lag-free columns plus 3 trailing temperature windows.

Experiments 515 and 526 stored W-T pretrained weights for 2025-12, 2026-01, and 2026-03 (seeds 0 to
2). Experiment 511's folds 2025-10 and 2026-06 have none, so this script pretrains them with
experiment 515's code (`e00515_fit.run`), unchanged except that the output folder is this
experiment's. ERA5 and CAMS rows only, cut at the fold's refit date, as in experiment 515.
"""

import sys
from pathlib import Path
from typing import Final

import e00515_fit as f515  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00548-lag-free-release-candidate-assembly-with-trailing-temperature-windows"
    / "work"
)
MISSING_FOLDS: Final[tuple[str, ...]] = ("2025-10", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)


def main() -> None:
    """Pretrain the W-T network on the folds without stored weights."""
    f515.W = W
    f515.run(
        months=list(MISSING_FOLDS),
        seeds=list(SEEDS),
        arms=["W-T"],
        do_pretrain=True,
        do_fit=False,
        device=sys.argv[1] if len(sys.argv) > 1 else "cuda",
    )


if __name__ == "__main__":
    main()
