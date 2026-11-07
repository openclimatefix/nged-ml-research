# ruff: noqa: ANN201, E402, E501, ANN003, PT018
"""Experiment 620: pre-train and fit K3 (class 3, lag-free, D5, E3) on the final-test fold 2026-07 (GPU).

A copy of experiment 613's `e00613_free.py` with `e00620_fold` patched onto experiment 565's module.
Pre-training writes the fold's own weights to this experiment's chain folder. Original docstring:

A wrapper over experiment 565's `e00565_fit.py` (arm D5: the lag-free network with the slow output
offset, one-step recipe: experiment 565's pretrained weights, a fresh mixer, frozen encoders, refits
on days 0 and 14). Experiment 565's pretrained weights are read only. `--e3 1` (default) applies the
same refit rule as `e00613_arm.py`: no rows from the newest 7 days before the cutoff, and half the
weight for the 7 days before that. `--e3 0` is experiment 565's own recipe (a reproduction check).

Usage: python e00620_k3.py NAME {pretrain|fit} [--months ...] [--seeds ...]
Outputs: `data/ml-experiments/00613-*/arms/NAME/work/{fits,models}`.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("pretrain", "fit"))
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
assert set(MINE.months) <= {"2025-10", "2025-12", "2026-01", "2026-03", "2026-06", "2026-07"}
assert MINE.e3 == 1
sys.argv = [
    "e00565_fit.py", {"pretrain": "pretrain-wt", "fit": "fit"}[MINE.command], "--arm", "D5", "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00565_fit as f565  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import os
from pathlib import Path

import e00620_fold as fold

NAME = "00620-bakeoff-final-test-demand-networks-at-primaries"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
E3_EMBARGO_DAYS = 7
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7

CHAIN = Path(os.environ["CHAIN620"])
"""This experiment's chain folder for K3: pre-training weights of the final fold."""
fold.install(f565, months=MINE.months)
f565.pretrained_dir = lambda: CHAIN / "pretrained_D5" / f565.SPEC.columns
f565.WORK = OUT
f565.W = OUT / "work"
assert f565.SPEC.tag == "D5" and f565.SPEC.start == "fresh" and f565.SPEC.frozen

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
        report = {
            **refit.report,
            "downweighted_rows": int(window.sum()),
            "embargo_days": E3_EMBARGO_DAYS,
        }
        # the early-stopping rows are kept or dropped with the same age rule
        return c.Refit(refit.index[keep], refit.y[keep], weight[keep], refit.is_val[keep], report)

    c.select_rows = select_rows

if MINE.command == "pretrain":
    f565.pretrain_wt(device="cuda")
else:
    assert f565.pretrained_file(month=MINE.months[0], seed=MINE.seeds[0]).exists()
    f565.fit_arm(
        months=f565.ARGS.months, seeds=f565.ARGS.seeds, stress=False, limit=None, device="cuda"
    )
    (OUT / "checks_fit.json").write_text(__import__("json").dumps(f565.CHECK, indent=1, default=str))
    print("CHECKS PASSED", {a: v for a, v in f565.CHECK.items() if a != "months"})
