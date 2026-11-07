# ruff: noqa: ANN001, ANN201, E402, D103, E501, ANN003
"""Experiment 613: fit or inject one arm of the leave-one-out table (GPU).

A wrapper over experiment 591's `e00591_arm.py` (arm A1: shared lag encoder with lag ID, no auxiliary
loss), imported with its module state patched as experiment 599 did. The pretrained and step-2
weights are experiment 591's stored A1 chain, read only. The arm sets the fine-tune recipe:

- `--e3 1` (default): the daily fine-tune embargo is 7 days and the newest 7 days left after the
  embargo get half weight (experiment 599's E3); `--e3 0` keeps experiment 591's 3-day embargo;
- `--synthetic 1` (default): lag-only synthetic load-shifts in the fine-tune rows (experiment 584's
  C1); `--synthetic 0` uses the plain P5 rows.

Usage: python e00613_arm.py NAME {fit|inject} [--e3 0|1] [--synthetic 0|1] [--months ...] [--seeds ...]
Outputs: `data/ml-experiments/00613-*/arms/NAME/work/{fits,inject,models}`.
"""

import argparse
import os
import sys
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "inject", "pretrain", "step2"))
_p.add_argument("--offset", choices=("d5", "o2", "none"), default="d5")
_p.add_argument("--drop", nargs="*", default=[], choices=("id", "age", "expected", "spread"))
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--synthetic", type=int, default=1)
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
sys.argv = [
    "e00591_arm.py", "A1", MINE.command, "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
    *(["--passes", *map(str, MINE.passes)] if MINE.passes else []),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00613_model as mymodel
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00613-nra-roster-contrasts-with-intervals-five-seeds"
OUT = ML_EXPERIMENTS_DATA_DIR / NAME / "arms" / MINE.name
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7
E3_EMBARGO_DAYS = 7

e = arm.e
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
DROPS = mymodel.Drops(**{f"drop_{d}": True for d in MINE.drop})
CHAIN_ENV = os.environ.get("CHAIN613")
"""A shared chain folder (`.../work`) holding pre-training and step-2 weights for all five seeds."""
CHAIN = Path(CHAIN_ENV) if CHAIN_ENV else OUT / "chain" / "work"
OWN_NETWORK = bool(MINE.drop) or MINE.offset != "d5"
assert "00591" not in str(CHAIN.resolve()), "experiment 591's chain is read-only"
assert not CHAIN_ENV or Path(CHAIN_ENV).parent.name.startswith(
    "o2" if MINE.offset == "o2" else "d5"
)
if CHAIN_ENV and not OWN_NETWORK:  # K1 with D5: stock network, own seeds 2-4 chain
    arm.CHAIN = CHAIN
if OWN_NETWORK:  # a removed input, O2, or no offset needs its own pre-training and step-2 weights
    arm.CHAIN = CHAIN
    cls = {"o2": mymodel.O2Mdn, "none": mymodel.NoOffsetMdn}.get(MINE.offset, mymodel.DropMdn)

    def new_model(*, n_primaries: int, stage: str = "ft"):
        model = cls(n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, drops=DROPS)
        model.table = e.CTX.get(f"table_{stage}")
        return model

    e.new_model = new_model
    if MINE.offset == "o2":
        e.m572.load_lag_pre_into_ft = mymodel.load_o2_pre_into_ft
        _train = e.train

        def train(*, model, is_val, **kwargs):
            """Set the penalty's row count (experiment 577's scaling), then run experiment 574's loop."""
            model.n_rows = float((~is_val).sum())
            return _train(model=model, is_val=is_val, **kwargs)

        e.train = train
if MINE.command in {"pretrain", "step2"}:
    assert OWN_NETWORK or CHAIN_ENV, "the stored chain of experiment 591 is read-only"
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
    arm.check_stats()
elif MINE.command == "step2":
    e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
elif MINE.command == "fit":
    e.row_nll = arm.row_nll
    if MINE.synthetic:
        e.k.build_month = arm.build_month_syn
        e.k.GpuMonth = arm.AuxGpuMonth
    # without synthetic shifts experiment 574's plain P5 rows and tensors (module defaults) are used
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    (OUT / "checks_fit.json").write_text(__import__("json").dumps(e.CHECK, indent=1, default=str))
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
else:
    arm.run_inject()
torch.cuda.empty_cache()
