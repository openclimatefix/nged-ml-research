# ruff: noqa: ANN003, ANN201, E402, E501, I001, PT018
"""Experiment 640: pre-train, step-2, and fit K1 (class 1) on fit-age folds (GPU).

A copy of experiment 620's `e00620_k1.py` (itself a copy of experiment 613's `e00613_arm.py`, K1 with
D5 and E3 only) with `e00640_fold` patched on instead of `e00620_fold`: a fold `YYYY-MM-AA` has one
cutoff `AA` weeks before the month start and is scored over the whole month. `E00640_PHASE=train`
hides the cleaned power from the cutoff and builds no test rows; `E00640_PHASE=predict` forecasts the
real month with the stored models and refits nothing.

Usage: python e00640_k1.py NAME {pretrain|step2|fit} --folds 2025-10-08 --seeds 0 1 2
Outputs: `data/ml-experiments/00640-*/arms/NAME/work/{fits,models}`; the chain (pre-training,
step-2 weights, B1 statistics) is under `$CHAIN640`.
"""

import argparse
import os
import sys
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "pretrain", "step2"))
_p.add_argument("--offset", choices=("d5", "o2", "none"), default="d5")
_p.add_argument("--drop", nargs="*", default=[], choices=("id", "age", "expected", "spread"))
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--synthetic", type=int, default=1)
_p.add_argument("--folds", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
assert MINE.offset == "d5" and MINE.e3 == 1 and not MINE.drop and MINE.synthetic == 1
sys.argv = [
    "e00591_arm.py", "A1", MINE.command, "--months", *MINE.folds, "--seeds", *map(str, MINE.seeds),
    *(["--passes", *map(str, MINE.passes)] if MINE.passes else []),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00640_fold as fold

NAME = "00640-demand-fit-age-refit-k1-k3"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7
E3_EMBARGO_DAYS = 7

e = arm.e
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
CHAIN_ENV = os.environ.get("CHAIN640")
"""The chain folder (`.../work`) holding pre-training and step-2 weights of every fold."""
assert CHAIN_ENV, "set CHAIN640 to the chain's work folder"
CHAIN = Path(CHAIN_ENV)
assert "00591" not in str(CHAIN.resolve()), "experiment 591's chain is read-only"
arm.CHAIN = CHAIN
fold.install(e, folds=MINE.folds)
# a fold has no stored B1 statistics: pre-training derives them into the chain folder and every
# later command reads them from there
e.stats_file = lambda *, month: CHAIN / "stats" / e.SPEC.b1 / f"{month}.npz"
arm.check_stats = lambda: None
if MINE.command in {"pretrain", "step2"}:
    e.WORK, e.W = CHAIN.parent, CHAIN
else:
    e.WORK = OUT
    e.W = OUT / "work"

if MINE.e3 and MINE.command in {"fit", "inject"}:
    e.EMBARGO_DAYS = E3_EMBARGO_DAYS
    _select_rows = e.c.select_rows

    def select_rows(**kwargs):
        """Return experiment 510's rows, with the weights of the newest 7 days left after the embargo halved."""
        refit = _select_rows(**kwargs)
        rows, cutoff = kwargs["rows"], kwargs["cutoff"]
        age_days = (e.c.us(t=cutoff) - rows.valid_us[refit.index]) / e.c.DAY_US
        window = (age_days >= E3_EMBARGO_DAYS) & (age_days < E3_EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
        weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
        report = {
            **refit.report,
            "downweighted_rows": int(window.sum()),
            "embargo_days": E3_EMBARGO_DAYS,
        }
        return e.c.Refit(refit.index, refit.y, weight, refit.is_val, report)

    e.c.select_rows = select_rows

if MINE.command == "pretrain":
    e.pretrain_wt(device="cuda")
elif MINE.command == "step2":
    e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
elif MINE.command == "fit":
    e.row_nll = arm.row_nll
    if MINE.synthetic:
        # experiment 613 seeds the synthetic events with the month's digits; seed with the development
        # month's digits so that every fit age of one month shares its events
        _synthetic_clean = arm.bank.synthetic_clean
        _dev_seed = int(fold.parse(MINE.folds[0])[0].replace("-", ""))
        arm.bank.synthetic_clean = lambda **kw: _synthetic_clean(**{**kw, "seed": _dev_seed})
        e.k.build_month = arm.build_month_syn
        e.k.GpuMonth = arm.AuxGpuMonth
    # without synthetic shifts experiment 574's plain P5 rows and tensors (module defaults) are used
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    (OUT / f"checks_{fold.PHASE}_{'_'.join(MINE.folds)}.json").write_text(
        __import__("json").dumps(e.CHECK, indent=1, default=str)
    )
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
torch.cuda.empty_cache()
