# ruff: noqa: ANN003, ANN201, ANN202, E402, E501, I001, PT018
"""Experiment 620: pre-train, step-2, and fit K1 (class 1) on the final-test fold 2026-07 (GPU).

A copy of experiment 613's `e00613_arm.py` (K1 with D5 and E3 only), with `e00677_fold` patched
onto it so that the fold `2026-07` has one cutoff (2026-07-01) and the held-out power is hidden
in the training phase. Development months still run unchanged. Original docstring:

A wrapper over experiment 591's `e00591_arm.py` (arm A1: shared lag encoder with lag ID, no
auxiliary loss), imported with its module state patched as experiment 599 did. The pretrained and
step-2 weights are experiment 591's stored A1 chain, read only. The arm sets the fine-tune
recipe:

- `--e3 1` (default): the daily fine-tune embargo is 7 days and the newest 7 days left after the
  embargo get half weight (experiment 599's E3); `--e3 0` keeps experiment 591's 3-day embargo;
- `--synthetic 1` (default): lag-only synthetic load-shifts in the fine-tune rows (experiment
  584's C1); `--synthetic 0` uses the plain P5 rows.

Usage: python e00613_arm.py NAME {fit|inject} [--e3 0|1] [--synthetic 0|1] [--months ...]
[--seeds ...] Outputs: `data/ml-experiments/00613-*/arms/NAME/work/{fits,inject,models}`.
"""

import argparse
import os
import sys
from pathlib import Path

import e00677_model as ladder  # ty: ignore[unresolved-import]

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "pretrain", "step2"))
_p.add_argument("--step", choices=tuple(ladder.STEP_MASKS), default="5")
_p.add_argument("--ens-only", type=int, default=0)
_p.add_argument("--offset", choices=("d5", "o2", "none"), default="d5")
_p.add_argument("--drop", nargs="*", default=[], choices=("id", "age", "expected", "spread"))
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--synthetic", type=int, default=1)
_p.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
MINE = _p.parse_args()
assert set(MINE.months) <= {
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
    "2026-08",
    "2026-09",
    "2026-10",
}
assert MINE.offset == "d5" and MINE.e3 == 1 and not MINE.drop
sys.argv = [
    "e00591_arm.py", "A1", MINE.command, "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
    *(["--passes", *map(str, MINE.passes)] if MINE.passes else []),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00677_fold as fold

assert list(MINE.months) == [fold.FINAL_MONTH], "one process per fold month: set E00677_MONTH"

NAME = "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
ROOT = (
    Path(os.environ["E00677_ROOT"])
    if os.environ.get("E00677_ROOT")
    else ML_EXPERIMENTS_DATA_DIR / NAME
)
OUT = ROOT / "arms" / f"{MINE.name}_sp{fold.SPAN_MONTHS}"
"""One folder per scored span, so the monthly pilot and the quarterly fold of 2026-05 never share fits."""
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7
E3_EMBARGO_DAYS = 7

e = arm.e
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
CHAIN_ENV = os.environ.get("CHAIN677")
"""The chain folder (`.../work`) holding pre-training and step-2 weights for all five seeds."""
assert CHAIN_ENV, "set CHAIN677 to the chain's work folder"
CHAIN = Path(CHAIN_ENV)
assert "00591" not in str(CHAIN.resolve()), "experiment 591's chain is read-only"
arm.CHAIN = CHAIN
fold.install(e, months=MINE.months)
if MINE.step != "5":  # steps 1 to 4: whole input groups removed; the layer shapes stay those of A1

    def new_model(*, n_primaries: int, stage: str = "ft"):
        model = ladder.LadderMdn(
            n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1,
            masks=ladder.STEP_MASKS[MINE.step],
        )  # fmt: skip
        model.table = e.CTX.get(f"table_{stage}")
        return model

    e.new_model = new_model
if MINE.ens_only:  # no ERA5 or CAMS pre-training: random initialisation, nothing frozen, one rate
    import dataclasses

    assert MINE.command != "pretrain", "an ENS-only network has no pre-training stage"
    e.SPEC = dataclasses.replace(e.SPEC, frozen=False, lr=e.c.PRETRAIN_LR)

    def load_random_fresh(*, model, path):
        """Initialise every layer from the seed in the step-2 file name, then give the mixer its fixed fresh draw."""
        seed = int(Path(path).stem.rsplit("_s", 1)[1])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        e.fresh_mixer(model=model)

    e.load_pretrained_fresh = load_random_fresh
    e.pretrained_file = lambda *, month, seed: (
        e.W / "no_pretraining" / f"{month}_s{seed}.pt"
    )  # a name only: it carries the seed
if fold.is_final(months=MINE.months):
    # the final fold has no stored B1 statistics: pre-training derives them into the chain folder
    # and every later command reads them from there
    def _final_stats_file(*, month: str):
        return Path(os.environ.get("STATS677", CHAIN)) / "stats" / e.SPEC.b1 / f"{month}.npz"

    _dev_stats_file = arm._stats_file
    e.stats_file = lambda *, month: (
        _final_stats_file(month=month)
        if month == fold.FINAL_MONTH
        else _dev_stats_file(month=month)
    )
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
torch.cuda.empty_cache()
