# ruff: noqa: ANN001, ANN201, E402, D103, E501
"""Experiment 608: fit the lag-free twin (arm K3 with D5) with or without the E3 refit rule (GPU).

A wrapper over experiment 565's `e00565_fit.py` (arm D5: the lag-free network with the slow output
offset, one-step recipe: experiment 565's pretrained weights, a fresh mixer, frozen encoders, refits on
days 0 and 14). Experiment 565's pretrained weights are read only. `--e3 1` (default) applies the same
refit rule as `e00608_arm.py`: no rows from the newest 7 days before the cutoff, and half the weight
for the 7 days before that. `--e3 0` is experiment 565's own recipe (a reproduction check).

Usage: python e00608_free.py NAME [--e3 0|1] [--months ...] [--seeds ...]
Outputs: `data/ml-experiments/00608-*/arms/NAME/work/{fits,models}`.
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
    "e00565_fit.py", "fit", "--arm", "D5", "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00565_fit as f565  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00608-nra-combined-networks-per-class-and-leave-one-out"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
E3_EMBARGO_DAYS = 7
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7

W565 = f565.W
_pretrained_dir = f565.pretrained_dir
f565.pretrained_dir = lambda: W565 / "pretrained_D5" / f565.SPEC.columns
f565.WORK = OUT
f565.W = OUT / "work"
assert f565.SPEC.tag == "D5" and f565.SPEC.start == "fresh" and f565.SPEC.frozen
assert f565.pretrained_file(month=MINE.months[0], seed=MINE.seeds[0]).exists()

if MINE.e3:
    c = f565.c
    _select_rows = c.select_rows

    def select_rows(**kwargs):
        """Return experiment 510's rows without the newest 7 days, the 7 days before them at half weight."""
        refit = _select_rows(**kwargs)
        rows, cutoff = kwargs["rows"], kwargs["cutoff"]
        age_days = (c.us(t=cutoff) - rows.valid_us[refit.index]) / c.DAY_US
        keep = age_days >= E3_EMBARGO_DAYS
        window = (age_days >= E3_EMBARGO_DAYS) & (age_days < E3_EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
        weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
        report = {**refit.report, "downweighted_rows": int(window.sum()), "embargo_days": E3_EMBARGO_DAYS}
        # the early-stopping rows are kept or dropped with the same age rule
        return c.Refit(refit.index[keep], refit.y[keep], weight[keep], refit.is_val[keep], report)

    c.select_rows = select_rows

f565.fit_arm(months=f565.ARGS.months, seeds=f565.ARGS.seeds, stress=False, limit=None, device="cuda")
(OUT / "checks_fit.json").write_text(__import__("json").dumps(f565.CHECK, indent=1, default=str))
print("CHECKS PASSED", {a: v for a, v in f565.CHECK.items() if a != "months"})
