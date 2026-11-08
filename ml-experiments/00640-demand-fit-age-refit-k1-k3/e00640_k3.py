# ruff: noqa: ANN003, ANN201, E402, E501, I001, PT018
"""Experiment 640: pre-train and fit K3 (class 3, lag-free, D5, E3) on fit-age folds (GPU).

A copy of experiment 620's `e00620_k3.py` with `e00640_fold` patched onto experiment 565's module: a
fold `YYYY-MM-AA` has one cutoff `AA` weeks before the month start and is scored over the whole
month. Pre-training writes each fold's own weights to the chain folder `$CHAIN640`.
`E00640_PHASE=train` hides the cleaned power from the cutoff and builds no test rows;
`E00640_PHASE=predict` forecasts the real month with the stored models and refits nothing.

Usage: python e00640_k3.py NAME {pretrain|fit} --folds 2025-10-08 --seeds 0 1 2
Outputs: `data/ml-experiments/00640-*/arms/NAME/work/{fits,models}`.
"""

import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("pretrain", "fit"))
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--folds", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
MINE = _p.parse_args()
assert MINE.e3 == 1
sys.argv = [
    "e00565_fit.py", {"pretrain": "pretrain-wt", "fit": "fit"}[MINE.command], "--arm", "D5", "--months", *MINE.folds, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00565_fit as f565  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import os
from pathlib import Path

import e00640_fold as fold

NAME = "00640-demand-fit-age-refit-k1-k3"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
E3_EMBARGO_DAYS = 7
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7

CHAIN = Path(os.environ["CHAIN640"])
"""This experiment's chain folder for K3: the pre-training weights of every fold."""
fold.install(f565, folds=MINE.folds)
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
    assert f565.pretrained_file(month=MINE.folds[0], seed=MINE.seeds[0]).exists()
    f565.fit_arm(
        months=f565.ARGS.months, seeds=f565.ARGS.seeds, stress=False, limit=None, device="cuda"
    )
    (OUT / f"checks_{fold.PHASE}_{'_'.join(MINE.folds)}.json").write_text(
        __import__("json").dumps(f565.CHECK, indent=1, default=str)
    )
    print("CHECKS PASSED", {a: v for a, v in f565.CHECK.items() if a != "months"})
