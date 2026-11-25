# ruff: noqa: ANN001, ANN003, ANN201, ANN202, ANN204, D103, E402, E501, I001, PT018
"""Experiment 731: stage 3 (the fine-tune) and prediction of the step-5 network with one ablation.

README. This script runs step 3 ("fit": train phase, then "pfit": predict phase) of experiment 731.
It is a copy of experiment 677's `e00677_k1.py` (immutable) with marked patch blocks (search for
`E00731 PATCH`). The code outside the marked blocks is identical to the original, which
`e00731_tests_known_answer.py` checks. It relies on experiment 677 (`e00677_fold`, `e00677_model`,
`e00677_extend`, `e00677_cand`), 574 (`e00574_fit`: `train`, `fine_tune`, `fit_arm`), 510
(`e00510_common.select_rows`), 581 (`e00581_bank.DEPTHS`, `synthetic_clean`), 591
(`e00591_arm`: `build_month_syn`), and 710 (`e00710_weights.permuted_age_days`).

PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root (it puts every experiment
folder on `PYTHONPATH`). Set `DATA_PATH_INTERNAL`, `E00677_ROOT` (must be the folder of experiment
731 under `data/ml-experiments/`), `E00677_MONTH`, `E00677_SPAN=1`, `E00677_PHASE` (train or
predict), `CHAIN677` (677's `chains/5/<month>/work`, read only), `STATS677` (677's
`chains/3b/<month>/work`, read only), and `E00731_ARM` (one of R0, W0, W30, W270, WP, IP, B0). Use
`e00731_run.sh`, which sets all of them.

The arms (the only difference from arm 9a, which is R0):

- W0: equal training-row weights (half-life infinite; the E3 embargo and halving stay);
- W30, W270: half-life 30 or 270 days;
- WP: the 90-day weights computed from ages permuted among calendar months (experiment 710's
  `permuted_age_days`); the E3 window still uses the true ages;
- IP: injected events as in R0 with every depth set to zero;
- B0: equal weights and no injection (`--synthetic 0`).

Original docstring of `e00677_k1.py`:

Experiment 620: pre-train, step-2, and fit K1 (class 1) on the final-test fold 2026-07 (GPU).

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
# >>> E00731 PATCH BEGIN (A: root assertion)
import hashlib as _hashlib
from typing import Final as _Final

E00731_ROOT_OK: _Final = os.environ.get("E00677_ROOT", "")
E00731_CODE_HASH: _Final = _hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
"""The sha256 of this file: every fit, history, and world records it, and a changed file refuses old outputs."""
assert "00731" in E00731_ROOT_OK, (
    "experiment 731 writes only under its own data folder: set E00677_ROOT to it",
    E00731_ROOT_OK,
    str(ROOT),
)
assert "00731" in str(ROOT), str(ROOT)
assert "00677" not in str(ROOT), "experiment 677 is immutable: never write under its folder"
E00731_ARM: _Final = os.environ.get("E00731_ARM", "")
assert E00731_ARM in {"R0", "W0", "W30", "W270", "WP", "IP", "B0"}, ("set E00731_ARM", E00731_ARM)
assert MINE.command == "fit", "experiment 731 runs stage 3 only; stages 1 and 2 are 677's, frozen"
assert MINE.step == "5" and not MINE.ens_only
assert MINE.synthetic == (0 if E00731_ARM == "B0" else 1), (
    "only arm B0 has no injected events",
    E00731_ARM,
    MINE.synthetic,
)
# <<< E00731 PATCH END (A: root assertion)
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

# >>> E00731 PATCH BEGIN (B: the arm)
import dataclasses as _dc
import json as _json
import math as _math

import e00710_weights as _w710  # ty: ignore[unresolved-import]

if E00731_ARM in {"W0", "B0"}:
    e.SPEC = _dc.replace(e.SPEC, half_life=_math.inf)
elif E00731_ARM == "W30":
    e.SPEC = _dc.replace(e.SPEC, half_life=30.0)
elif E00731_ARM == "W270":
    e.SPEC = _dc.replace(e.SPEC, half_life=270.0)
else:
    assert e.SPEC.half_life == 90.0, e.SPEC.half_life
if E00731_ARM == "IP":
    # onsets, lengths, and signs come from the same random draws; every depth is zero
    assert arm.bank.DEPTHS == (0.15, 0.30, 0.50), arm.bank.DEPTHS
    arm.bank.DEPTHS = (0.0, 0.0, 0.0)
if E00731_ARM == "WP":
    _wp_inner = e.c.select_rows

    def select_rows_wp(**kwargs):
        """Return the rows of `select_rows` with the recency weights of ages permuted among calendar months.

        Wrapped inside experiment 677's E3 wrapper below, so that the E3 window still uses the true ages.
        """
        refit = _wp_inner(**kwargs)
        rows, cutoff = kwargs["rows"], kwargs["cutoff"]
        refit_us = e.c.us(t=cutoff)
        age = _w710.permuted_age_days(
            valid_us=rows.valid_us[refit.index],
            refit_us=refit_us,
            floor_us=refit_us - e.c.WINDOW_DAYS * e.c.DAY_US,
            fold_yyyymm=int(MINE.months[0].replace("-", "")),
        )
        weight = (0.5 ** (age / kwargs["half_life_days"])).astype(np.float32)
        assert weight.shape == refit.weight.shape
        report = {**refit.report, "placebo_total_weight": float(weight.sum())}
        return e.c.Refit(
            index=refit.index, y=refit.y, weight=weight, is_val=refit.is_val, report=report
        )

    e.c.select_rows = select_rows_wp
# <<< E00731 PATCH END (B: the arm)
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
# >>> E00731 PATCH BEGIN (C: diagnostics and per-epoch loss history; no change to the fit)
    _marker = OUT / "work" / f"code_hash_{MINE.months[0]}.txt"
    _existing = [
        *(OUT / "work" / "models" / "P5").glob(f"{MINE.months[0]}_d0_s*.pt"),
        *(OUT / "work" / "fits" / "P5").glob(f"{MINE.months[0]}_d0_s*.parquet"),
        *(OUT / "work" / "history").glob(f"{MINE.months[0]}_d0_s*.json"),
        *(OUT / "work" / "worlds").glob(f"{MINE.months[0]}.npz"),
    ]
    if _marker.exists():
        assert _marker.read_text().strip() == E00731_CODE_HASH, (
            "e00731_k1.py changed since this arm and month were fitted: delete its outputs first",
            str(_marker),
        )
    else:
        assert not _existing, ("outputs without a code-hash marker: delete them", str(_marker))
        _marker.parent.mkdir(parents=True, exist_ok=True)
        _marker.write_text(E00731_CODE_HASH + "\n")
    print("CODE HASH", E00731_CODE_HASH, flush=True)

    _outer_select_rows = e.c.select_rows

    def select_rows_report(**kwargs):
        """Return the rows of the wrapped `select_rows`, with the total weight and effective rows in the report."""
        refit = _outer_select_rows(**kwargs)
        w64 = refit.weight.astype(np.float64)
        refit.report["total_weight"] = float(w64.sum())
        refit.report["effective_rows"] = float(w64.sum() ** 2 / (w64**2).sum())
        refit.report["arm731"] = E00731_ARM
        return refit

    e.c.select_rows = select_rows_report

    class _History:
        """Per-epoch mean training and validation negative log-likelihood, recorded from the row loss."""

        def __init__(self):
            self.reset()

        def reset(self):
            self.rows, self.in_val = [], False
            self.tr = self.va = None
            self.tr_n = self.va_n = 0

        def add(self, *, nll):
            d = nll.detach().double()
            if torch.is_grad_enabled():  # a training batch; the validation pass runs under no_grad
                if self.in_val:
                    self.close()
                self.tr = d.sum() if self.tr is None else self.tr + d.sum()
                self.tr_n += d.numel()
            else:
                self.in_val = True
                self.va = d.sum() if self.va is None else self.va + d.sum()
                self.va_n += d.numel()

        def close(self):
            self.rows.append(
                {
                    "train_nll_unweighted": float(self.tr) / self.tr_n,
                    "val_nll": float(self.va) / self.va_n,
                }
            )
            self.in_val = False
            self.tr = self.va = None
            self.tr_n = self.va_n = 0

        def finish(self):
            if self.in_val:
                self.close()
            best, stale, chosen = float("inf"), 0, -1
            for i, r in enumerate(self.rows):  # experiment 574's early-stopping rule
                if r["val_nll"] < best - 1e-5:
                    best, stale, chosen = r["val_nll"], 0, i
                else:
                    stale += 1
            return self.rows, chosen + 1  # 1-based stored epoch

    HIST = _History()
    _inner_row_nll = e.row_nll

    def row_nll_recorded(*, out, z, p):
        nll = _inner_row_nll(out=out, z=z, p=p)
        HIST.add(nll=nll)
        return nll

    e.row_nll = row_nll_recorded
    _inner_fine_tune = e.fine_tune

    def fine_tune_recorded(**kwargs):
        """Fine-tune as before, and save the per-epoch loss history beside the fit's log."""
        HIST.reset()
        model, log = _inner_fine_tune(**kwargs)
        rows, stored_epoch = HIST.finish()
        assert len(rows) == log["epochs"], (len(rows), log["epochs"])
        assert abs(rows[stored_epoch - 1]["val_nll"] - log["val_nll"]) < 1e-4, (
            rows[stored_epoch - 1]["val_nll"],
            log["val_nll"],
        )
        day = e.f.day_of(month=kwargs["month"], cut=kwargs["cut"])
        path = OUT / "work" / "history" / f"{kwargs['month']}_d{day}_s{kwargs['seed']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm": E00731_ARM,
            "code_hash": E00731_CODE_HASH,
            "max_epochs": e.SPEC.max_epochs,
            "epochs_run": len(rows),
            "epochs_in_log": log["epochs"],
            "stored_epoch": stored_epoch,
            "val_nll_in_log": log["val_nll"],
            "history": rows,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json.dumps(payload, indent=1))
        tmp.rename(path)
        log["history_file"] = str(path)
        log["code_hash"] = E00731_CODE_HASH
        log["stored_epoch"] = stored_epoch
        return model, log

    e.fine_tune = fine_tune_recorded

    _inner_synthetic_clean = arm.bank.synthetic_clean

    def synthetic_clean_recorded(*, clean, first, last, seed):
        """Return experiment 581's injected world as before, and save its shifts once per month for the figures."""
        fixed, n_events = _inner_synthetic_clean(clean=clean, first=first, last=last, seed=seed)
        path = OUT / "work" / "worlds" / f"{MINE.months[0]}.npz"
        if not path.exists():
            shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp, shift=shift[:, first:last].astype(np.float32), first=first, last=last,
                n_events=n_events, seed=seed, code_hash=E00731_CODE_HASH,
            )  # fmt: skip
            tmp.rename(path)
        return fixed, n_events

    arm.bank.synthetic_clean = synthetic_clean_recorded
# <<< E00731 PATCH END (C: diagnostics and per-epoch loss history; no change to the fit)
    if MINE.synthetic:
        e.k.build_month = arm.build_month_syn
        e.k.GpuMonth = arm.AuxGpuMonth
    # without synthetic shifts experiment 574's plain P5 rows and tensors (module defaults) are used
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    (OUT / "checks_fit.json").write_text(__import__("json").dumps(e.CHECK, indent=1, default=str))
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
torch.cuda.empty_cache()
