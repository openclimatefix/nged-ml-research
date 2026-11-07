# ruff: noqa: ANN001, ANN201, E402, D103, E501
"""Experiment 612: score one arm with its G4 band, on the months it was fitted on (CPU).

Wraps experiment 597's scoring unchanged, with three changes:

- the months are restricted to the arm's fitted months (`--months`);
- the arm's forecasts and injected bank come from `arms/NAME/work` (`--arm NAME`), or from the stored
  experiment-591 A1 (`--arm a1`);
- the lag-free twin that the G4 gate reads (experiment 592's `lag_free_median`) is the arm's own twin
  `arms/TWIN/work/fits/D5` (`--twin TWIN`); without `--twin` it is experiment 565's stored D5. The gate
  state is therefore always recomputed (never experiment 592's `gate_state.npz`), and its
  false-trigger rate is written to `false_triggers.csv`.

Usage: python e00612_score.py OUTNAME --arm NAME [--twin TWIN] [--months ...]
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("outname")
_p.add_argument("--arm", required=True)
_p.add_argument("--twin", default=None)
_p.add_argument("--twin-tag", default="D5")
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
MINE = _p.parse_args()

import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_score as s597  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
c.MONTHS = tuple(MINE.months)
k597.WORK = ROOT / "score" / MINE.outname
if MINE.arm != "a1":
    work = ROOT / "arms" / MINE.arm / "work"
    k584.ARMS[MINE.arm] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
if MINE.twin:
    twin_root = ROOT / "arms" / MINE.twin / "work"
    _orig = g.a._load

    def load(*, root, tag, months):
        return (
            _orig(root=twin_root, tag=MINE.twin_tag, months=months)
            if tag == "D5"
            else _orig(root=root, tag=tag, months=months)
        )

    g.a._load = load
sys.argv = ["e00597_score.py", MINE.arm]
s597.main(MINE.arm)
