# ruff: noqa: E501
"""Pretrain the W-T and W-Tnull lag-free networks for the folds that have no stored weights.

Experiments 515 and 526 stored W-T (and 526 W-Tnull) pretrained weights for 2025-12, 2026-01, and
2026-03; experiment 548 stored W-T for 2025-10 and 2026-06. This script pretrains W-Tnull for 2025-10
and 2026-06 with experiment 526's `pretrain`, unchanged except that the output folder is this
experiment's. ERA5 and CAMS rows only, cut at the fold's refit date.
"""

import sys
from pathlib import Path
from typing import Final

import e00510_common as c  # ty: ignore[unresolved-import]
import e00526_fit as f526  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00550-lag-light-network-trailing-temperature-windows-with-placebo"
    / "work"
)
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1)


def main() -> None:
    """Pretrain W-Tnull on the folds without stored weights."""
    f526.W = W
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    clean = c.clean_stack()
    for month in FOLDS:
        todo = [
            s
            for s in SEEDS
            if f526.pretrained_source(arm=f526.SPEC_NULL_T, month=month, seed=s) is None
        ]
        if not todo:
            continue
        fold = f526.Fold(
            month=month, clean=clean, need_pre=True, arms=[f526.SPEC_NULL_T], seam=False
        )
        for seed in todo:
            print(
                f526.pretrain(fold=fold, arm=f526.SPEC_NULL_T, seed=seed, device=device), flush=True
            )
        del fold


if __name__ == "__main__":
    main()
