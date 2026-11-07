# ruff: noqa: ANN201, E402, E501, ANN003, PT018
"""Experiment 612: fit the lag-free twin with the O2 offset, with or without E3 (GPU).

A wrapper over experiment 577's `e00577_fit.py` (arm O2: the lag-free network with the steadier
output offset, one-step recipe: experiment 577's pretrained weights, a fresh mixer, frozen encoders,
refits on days 0 and 14). Experiment 577's pretrained weights are read only. `--e3 1` (default)
applies the same refit rule as `e00612_arm.py`: no rows from the newest 7 days before the cutoff,
and half the weight for the 7 days before that. `--e3 0` is experiment 577's own recipe (a
reproduction check).

Usage: python e00612_free.py NAME [--e3 0|1] [--months ...] [--seeds ...]
Outputs: `data/ml-experiments/00612-*/arms/NAME/work/{fits,models}`.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
sys.argv = [
    "e00577_fit.py", "fit", "--arm", "O2", "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00577_fit as f577  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
E3_EMBARGO_DAYS = 7
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7

W577 = f577.W
f577.pretrained_dir = lambda: W577 / "pretrained_O2" / "real"
f577.WORK = OUT
f577.W = OUT / "work"
assert f577.SPEC.tag == "O2" and f577.SPEC.variant == "tv2"
assert (f577.pretrained_dir() / f"{MINE.months[0]}_s{MINE.seeds[0]}.pt").exists()

if MINE.e3:
    c = f577.c
    _select_rows = c.select_rows

    def select_rows(**kwargs):
        """Return experiment 510's rows without the newest 7 days, the 7 days before them at half weight."""
        refit = _select_rows(**kwargs)
        rows, cutoff = kwargs["rows"], kwargs["cutoff"]
        age_days = (c.us(t=cutoff) - rows.valid_us[refit.index]) / c.DAY_US
        keep = age_days >= E3_EMBARGO_DAYS
        window = (age_days >= E3_EMBARGO_DAYS) & (age_days < E3_EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
        weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
        report = {
            **refit.report,
            "downweighted_rows": int(window.sum()),
            "embargo_days": E3_EMBARGO_DAYS,
        }
        # the early-stopping rows are kept or dropped with the same age rule
        return c.Refit(refit.index[keep], refit.y[keep], weight[keep], refit.is_val[keep], report)

    c.select_rows = select_rows

f577.fit_arm(device="cuda")
(OUT / "checks_fit.json").write_text(__import__("json").dumps(f577.CHECK, indent=1, default=str))
print("CHECKS PASSED", {a: v for a, v in f577.CHECK.items() if a != "months"})
