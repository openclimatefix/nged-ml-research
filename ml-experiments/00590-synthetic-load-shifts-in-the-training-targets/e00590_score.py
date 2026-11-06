# ruff: noqa: E501
"""Experiment 590: score arms T1 to T3 with experiment 584's scoring (which wraps experiment 583's).

No fit and no GPU. Experiment 584's `ARMS` table is extended with the three new arms and `ROOT` is
pointed at this experiment's data folder, so the symlink views and score folders land here. Arm T0 is
experiment 584's C1: its score folder (`score_c1`) is read in place by `e00590_report.py`.

Usage: python e00590_score.py ARM   (ARM: t1, t2, t3, or p5); run one arm per process (module state).
"""

import sys

import e00584_score as s  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00590-synthetic-load-shifts-in-the-training-targets"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
for arm in ("t1", "t2", "t3"):
    s.ARMS[arm] = {
        "fits": ROOT / arm / "work/fits",
        "fit_tag": "P5",
        "inject": ROOT / arm / "work/inject",
    }
s.ARMS["p5"] = {
    "fits": s.ARMS["p5"]["fits"],
    "fit_tag": "P5",
    "inject": ROOT / "p5/work/inject",
}  # experiment 574's P5 (clean lags)
s.ROOT = ROOT

if __name__ == "__main__":
    s.main(sys.argv[1])
