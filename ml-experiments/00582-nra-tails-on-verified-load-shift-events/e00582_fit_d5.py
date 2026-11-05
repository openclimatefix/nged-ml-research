# ruff: noqa: E402
"""Experiment 582: experiment 565's lag-free D5 network on the fold month 2026-05.

A thin wrapper around experiment 565's `e00565_fit.py` (imported, unchanged) that re-points its
output folder to this experiment's own work folder, so experiment 565's files are never written.
Usage: python e00582_fit_d5.py {pretrain-wt|fit|assemble} --months 2026-05 [--seeds 0 1].
"""

import sys

sys.argv = ["e00565_fit.py", sys.argv[1], "--arm", "D5", *sys.argv[2:]]

import e00565_fit as e  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

e.WORK = ML_EXPERIMENTS_DATA_DIR / k.NAME
e.W = e.WORK / "work"

if __name__ == "__main__":
    if e.ARGS.command == "assemble":
        e.assemble()
    elif e.ARGS.command == "pretrain-wt":
        e.pretrain_wt(device="cuda")
    else:
        e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
