# ruff: noqa: E501
"""Experiment 591: score arms A1 to A5 with experiment 584's scoring (no fit, no GPU).

Experiment 584's `e00584_score` builds a symlink view of one arm's stored forecasts (fits, lags-removed
twin, injected bank) in the layout experiment 583's scoring reads, then runs that scoring and the
width-matched controls. This script adds this experiment's arms to its `ARMS` table and points its output
folder at this experiment's data folder. Arm A0 is `c1` and C0 is `c0` of experiment 584 (already scored
there; re-scored here into this experiment's folder so that all tables sit side by side).

Usage: python e00591_score.py ARM   (ARM: a0, c0, a1, a2, a3, a4, a5)
"""

import sys

import e00584_score as s584  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00591-shared-lag-encoder-auxiliary-shift-depth-loss"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
s584.ROOT = ROOT
C1 = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/c1/work"
s584.ARMS["a0"] = {"fits": C1 / "fits", "fit_tag": "P5", "inject": C1 / "inject"}
for arm in ("a1", "a2", "a3", "a4", "a5"):
    work = ROOT / arm.upper() / "work"
    s584.ARMS[arm] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}

if __name__ == "__main__":
    s584.main(sys.argv[1])
