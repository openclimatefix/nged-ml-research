# ruff: noqa: E501
"""Experiment 594: score arms PS and PP with experiment 584's scoring (which wraps experiment 583's).

No fit and no GPU. Experiment 584's `ARMS` table is extended with the three new arms and `ROOT` is
pointed at this experiment's data folder, so the symlink views and score folders land here. Arm T0 is
experiment 584's C1, T3 and P5 are experiment 590's: their score folders are read in place by
`e00594_report.py`.

Usage: python e00594_score.py ARM   (ARM: ps or pp); run one arm per process (module state).
"""

import sys

import e00584_score as s  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00594-persistent-and-share-matched-placebos-for-synthetic-load-shifts"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
for arm in ("ps", "pp"):
    s.ARMS[arm] = {
        "fits": ROOT / arm / "work/fits",
        "fit_tag": "P5",
        "inject": ROOT / arm / "work/inject",
    }
s.ROOT = ROOT

if __name__ == "__main__":
    s.main(sys.argv[1])
