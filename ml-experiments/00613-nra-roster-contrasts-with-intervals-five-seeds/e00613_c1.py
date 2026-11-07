# ruff: noqa: ANN201, E402, D103, E501, ANN003
"""Experiment 613: fit or inject the C1 wiring (separate lag encoders) with E3 (GPU).

A wrapper over experiment 584's `e00584_arm.py`, with the E3 refit rule of `e00613_arm.py` applied to
the daily fine-tune. Experiment 574's stored step-2 weights and B1 statistics are read only.

Usage: python e00613_c1.py NAME {fit|inject} [--months ...] [--seeds ...]
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "inject"))
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
sys.argv = [
    "e00584_arm.py",
    MINE.command,
    "--months",
    *MINE.months,
    "--seeds",
    *map(str, MINE.seeds),
]

import e00584_arm as arm  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00613-nra-roster-contrasts-with-intervals-five-seeds"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
e = arm.e
assert e.EMBARGO_DAYS == 3
e.WORK = OUT
e.W = OUT / "work"
E3_DAYS, HALF, HALF_DAYS = 7, 0.5, 7
e.EMBARGO_DAYS = E3_DAYS
_select_rows = e.c.select_rows


def select_rows(**kwargs):
    refit = _select_rows(**kwargs)
    rows, cutoff = kwargs["rows"], kwargs["cutoff"]
    age_days = (e.c.us(t=cutoff) - rows.valid_us[refit.index]) / e.c.DAY_US
    window = (age_days >= E3_DAYS) & (age_days < E3_DAYS + HALF_DAYS)
    weight = np.where(window, refit.weight * HALF, refit.weight).astype(np.float32)
    report = {**refit.report, "downweighted_rows": int(window.sum()), "embargo_days": E3_DAYS}
    return e.c.Refit(refit.index, refit.y, weight, refit.is_val, report)


e.c.select_rows = select_rows
if MINE.command == "inject":
    arm.run_inject()
else:
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
