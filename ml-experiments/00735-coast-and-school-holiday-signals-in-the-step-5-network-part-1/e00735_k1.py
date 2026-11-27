# ruff: noqa: ANN001, ANN003, ANN201, ANN202, ANN204, D103, E402, E501, I001, PT018
"""Experiment 735: stages 2 and 3 of the step-5 network with school-holiday and coast columns.

README. This script runs stage 2 ("step2"), stage 3 ("fit": train phase, then "pfit": predict
phase), and the smoke test's validation-row forecast ("valpred") of experiment 735. It is a copy of
experiment 731's `e00731_k1.py`, which is a copy of experiment 677's `e00677_k1.py` (both
immutable), with marked patch blocks (search for `E00735 PATCH`). The code outside the marked blocks
is the code of experiment 677, which `e00735_tests_known_answer.py` checks. It relies on experiment 677
(`e00677_fold`, `e00677_model`, `e00677_extend`), 574 (`e00574_fit`: `train`, `step2`, `fine_tune`,
`fit_arm`), 510 (`e00510_common.select_rows`), 581 (`e00581_bank`: `synthetic_clean`), 591
(`e00591_arm`: `build_month_syn`, `AuxGpuMonth`), 735's `e00735_columns`, and `e00735_model`.

PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root (it puts every experiment
folder on `PYTHONPATH`). Set `DATA_PATH_INTERNAL`, `E00677_ROOT` (must be the folder of experiment
735 under `data/ml-experiments/`, or a scratch folder whose name contains `00735`), `E00677_MONTH`,
`E00677_SPAN=1`, `E00677_PHASE` (train or predict), `CHAIN677` (677's `chains/5/<month>/work`, read
only: only `pretrained_P2` is read), `STATS677` (677's `chains/3b/<month>/work`, read only), and
`E00735_ARM` (one of R0, S, SP, SKW, SKWP, SE, SEP). Use `e00735_run.sh`, which sets all of them.
`E00735_SMOKE=1` caps both stages at one epoch.

Every arm trains from 677's stage-1 weights. Stage 2 writes its own step-2 weights under the arm's
own folder (`arms/k1_<ARM>_sp1/work/step2/P5`), because `step2` skips a seed whose file exists and
677's folder must never be read as step-2 output or written. The new input columns come after the
B1 columns (input index 60 onwards), enter the mixer with zero first-layer weights, and move the
label columns of stage 3 up by their width. The twins of the main frame (lags removed, lags aged)
are not forecast, because no score reads them.

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
_p.add_argument("command", choices=("fit", "pretrain", "step2", "valpred"))
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
_ORIGINAL_COMMAND = MINE.command
sys.argv = [
    "e00591_arm.py", "A1", "fit" if MINE.command == "valpred" else MINE.command,
    "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
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
# >>> E00735 PATCH BEGIN (A: root assertion, arm, and code hash)
import hashlib as _hashlib
from typing import Final as _Final

E00735_ROOT_OK: _Final = os.environ.get("E00677_ROOT", "")
E00735_CODE_HASH: _Final = _hashlib.sha256(
    b"".join(
        (Path(__file__).parent / n).read_bytes()
        for n in ("e00735_k1.py", "e00735_columns.py", "e00735_model.py", "e00735_diagnostic.py")
    )
).hexdigest()
"""The sha256 of this file, the column module, the model module, and the diagnostic module (the coast distances): every fit records it, and a changed code state refuses old outputs."""
assert "00735" in E00735_ROOT_OK, (
    "experiment 735 writes only under its own data folder: set E00677_ROOT to it",
    E00735_ROOT_OK,
    str(ROOT),
)
assert "00735" in str(ROOT), str(ROOT)
assert "00677" not in str(ROOT), "experiment 677 is immutable: never write under its folder"
E00735_ARM: _Final = os.environ.get("E00735_ARM", "")
assert E00735_ARM in {"R0", "S", "SP", "SKW", "SKWP", "SE", "SEP"}, ("set E00735_ARM", E00735_ARM)
E00735_SMOKE: _Final = os.environ.get("E00735_SMOKE", "") == "1"
assert MINE.command in {"fit", "step2", "valpred"}, "experiment 735 runs stages 2 and 3 only"
assert MINE.step == "5" and not MINE.ens_only
assert MINE.synthetic == 1, "every arm of experiment 735 has the injected events of arm 9a"
# <<< E00735 PATCH END (A: root assertion, arm, and code hash)
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
# >>> E00735 PATCH BEGIN (B: the chain is this arm's own folder; only pretrained_P2 is read from 677)
arm.CHAIN = OUT / "work"
assert "00735" in str(arm.CHAIN) and "00677" not in str(arm.CHAIN)
# <<< E00735 PATCH END (B: the chain is this arm's own folder; only pretrained_P2 is read from 677)
fold.install(e, months=MINE.months)
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
# >>> E00735 PATCH BEGIN (C: both commands write to this arm's own folder)
e.WORK = OUT
e.W = OUT / "work"
# <<< E00735 PATCH END (C: both commands write to this arm's own folder)

# >>> E00735 PATCH BEGIN (D: the arm's columns, network, stage-1 weights, and smoke cap)
import dataclasses as _dc

import e00735_columns as _cols735
import e00735_model as _model735

SPEC735: _Final = _cols735.ARMS[E00735_ARM]
BUILDER: _Final = _cols735.RowBuilder(
    spec=SPEC735,
    stats_file=lambda *, month: e.stats_file(month=month),
    expected_dir=e.W560 / "expected",
)
COAST: _Final = BUILDER.c_permuted if SPEC735.permuted else BUILDER.c_true
if E00735_SMOKE:
    e.SPEC = _dc.replace(e.SPEC, max_epochs=1)
assert e.SPEC.sched == "3sE" and e.SPEC.half_life == 90.0 and e.SPEC.pretrain_from == "P2"
e.pretrained_dir = lambda: CHAIN / "pretrained_P2"
"""Stage-1 weights come from experiment 677's chain, read only; `step2_file` reads this arm's own folder."""
e.fresh_mixer = _model735.fresh_mixer
e.load_pretrained_fresh = _model735.load_pretrained_fresh


def new_model_735(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return the control's network (R0) or the network with the arm's extra columns and coast term."""
    model = _model735.make_network(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, n_new=SPEC735.width,
        coast=COAST if SPEC735.embed else None,
    )  # fmt: skip
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model_735


def _with_columns(inner):
    """Wrap a month builder so that its rows carry the arm's new columns after the B1 columns."""

    def build_month_columns(*, month, clean):
        mr = inner(month=month, clean=clean)
        if SPEC735.width:
            assert mr.x.shape[1] == 60, mr.x.shape
            block = BUILDER.month_block(month=month, mr=mr)
            assert block.dtype == np.float32 and mr.x.dtype == np.float32, (block.dtype, mr.x.dtype)
            mr.x = np.concatenate([mr.x, block], axis=1)
            assert mr.x.dtype == np.float32 and mr.x.shape[1] == 60 + SPEC735.width, mr.x.shape
            share = (block != 0).mean(axis=0)
            e.CHECK["months"][month]["new_columns_train"] = dict(
                zip(SPEC735.names, share.round(4).tolist(), strict=True)
            )
            e.CHECK["months"][month]["new_columns_report"] = BUILDER.last_report
            print(
                month,
                "new columns appended; non-zero shares",
                e.CHECK["months"][month]["new_columns_train"],
                flush=True,
            )
        return mr

    return build_month_columns


e.k.build_month = _with_columns(e.k.build_month)
arm._build_month_p5 = _with_columns(arm._build_month_p5)
"""Stage 3 builds rows through `arm.build_month_syn`, which calls `_build_month_p5`, so this wrapper is the one that counts there."""
_inner_build_test = e.data.build_test


def build_test_columns(*, month, standardiser):
    """Return the month's test rows with the arm's columns appended; None while the scored window is closed."""
    test = _inner_build_test(month=month, standardiser=standardiser)
    e.CTX["variants"] = {}
    if test is None or not SPEC735.width:
        return test
    assert test.x.shape[1] == 60, test.x.shape
    block = BUILDER.test_block(
        month=month, test=test
    )  # asserts float32, K1, S1, and W1 to W2 are non-zero
    assert block.dtype == np.float32 and test.x.dtype == np.float32, (block.dtype, test.x.dtype)
    share = (block != 0).mean(axis=0)
    e.CHECK["months"].setdefault(month, {})["new_columns_test"] = dict(
        zip(SPEC735.names, share.round(4).tolist(), strict=True)
    )
    print(
        month,
        "test new columns; non-zero shares",
        e.CHECK["months"][month]["new_columns_test"],
        flush=True,
    )
    joined = np.concatenate([test.x, block], axis=1)
    assert joined.dtype == np.float32, joined.dtype
    return _dc.replace(test, x=joined)


e.data.build_test = build_test_columns
# <<< E00735 PATCH END (D: the arm's columns, network, stage-1 weights, and smoke cap)

if MINE.e3 and MINE.command in {"fit", "valpred"}:
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

# >>> E00735 PATCH BEGIN (E: code-hash marker, row-loss history, and per-epoch loss history)
import json as _json2

_marker = OUT / "work" / f"code_hash_{MINE.months[0]}.txt"
_existing = [
    *(OUT / "work" / "step2" / "P5").glob(f"{MINE.months[0]}_s*.pt"),
    *(OUT / "work" / "models" / "P5").glob(f"{MINE.months[0]}_d0_s*.pt"),
    *(OUT / "work" / "fits" / "P5").glob(f"{MINE.months[0]}_d0_s*.parquet"),
    *(OUT / "work" / "history").glob(f"{MINE.months[0]}_d0_s*.json"),
]
if _marker.exists():
    assert _marker.read_text().strip() == E00735_CODE_HASH, (
        "the experiment 735 code changed since this arm and month were fitted: delete its outputs first",
        str(_marker),
    )
else:
    assert not _existing, ("outputs without a code-hash marker: delete them", str(_marker))
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.write_text(E00735_CODE_HASH + "\n")
print(
    "CODE HASH",
    E00735_CODE_HASH,
    "ARM",
    E00735_ARM,
    "columns",
    SPEC735.names,
    "smoke",
    E00735_SMOKE,
    flush=True,
)


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
                "train_nll_unweighted": float(self.tr) / self.tr_n,  # ty: ignore[invalid-argument-type]
                "val_nll": float(self.va) / self.va_n,  # ty: ignore[invalid-argument-type]
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


def _write_history(*, path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_json2.dumps(payload, indent=1))
    tmp.rename(path)


HIST = _History()
# <<< E00735 PATCH END (E: code-hash marker, row-loss history, and per-epoch loss history)

if MINE.command == "step2":
    # >>> E00735 PATCH BEGIN (F: stage 2 records its per-epoch loss history)
    _inner_row_nll_s2 = e.row_nll

    def row_nll_recorded_s2(*, out, z, p):
        nll = _inner_row_nll_s2(out=out, z=z, p=p)
        HIST.add(nll=nll)
        return nll

    e.row_nll = row_nll_recorded_s2
    _inner_train = e.train

    def train_recorded(**kwargs):
        """Train as before, and save the per-epoch loss history of this stage-2 fit."""
        HIST.reset()
        epochs, best = _inner_train(**kwargs)
        rows, stored_epoch = HIST.finish()
        assert len(rows) == epochs, (len(rows), epochs)
        assert abs(rows[stored_epoch - 1]["val_nll"] - best) < 1e-4, (rows[stored_epoch - 1], best)
        month = f"{e.CTX['cut']:%Y-%m}"
        seed = e.CTX["seed"]
        _write_history(
            path=OUT / "work" / "history_step2" / f"{month}_s{seed}.json",
            payload={
                "arm": E00735_ARM, "stage": 2, "code_hash": E00735_CODE_HASH,
                "max_epochs": e.SPEC.max_epochs, "epochs_run": epochs, "stored_epoch": stored_epoch,
                "val_nll": best, "history": rows,
            },
        )  # fmt: skip
        return epochs, best

    e.train = train_recorded
    # <<< E00735 PATCH END (F: stage 2 records its per-epoch loss history)
    e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    print("STEP2 DONE", {a: v for a, v in e.CHECK.items() if a != "months"}, flush=True)
elif MINE.command == "fit":
    e.row_nll = arm.row_nll
    # >>> E00735 PATCH BEGIN (G: diagnostics and per-epoch loss history of stage 3; no change to the fit)
    _outer_select_rows = e.c.select_rows

    def select_rows_report(**kwargs):
        """Return the rows of the wrapped `select_rows`, with the total weight and effective rows in the report."""
        refit = _outer_select_rows(**kwargs)
        w64 = refit.weight.astype(np.float64)
        refit.report["total_weight"] = float(w64.sum())
        refit.report["effective_rows"] = float(w64.sum() ** 2 / (w64**2).sum())
        refit.report["arm735"] = E00735_ARM
        return refit

    e.c.select_rows = select_rows_report
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
        payload = {
            "arm": E00735_ARM,
            "stage": 3,
            "code_hash": E00735_CODE_HASH,
            "max_epochs": e.SPEC.max_epochs,
            "epochs_run": len(rows),
            "epochs_in_log": log["epochs"],
            "stored_epoch": stored_epoch,
            "val_nll_in_log": log["val_nll"],
            "history": rows,
        }
        _write_history(path=path, payload=payload)
        log["history_file"] = str(path)
        log["code_hash"] = E00735_CODE_HASH
        log["stored_epoch"] = stored_epoch
        return model, log

    e.fine_tune = fine_tune_recorded
    # <<< E00735 PATCH END (G: diagnostics and per-epoch loss history of stage 3; no change to the fit)
    e.k.build_month = arm.build_month_syn
    e.k.GpuMonth = arm.AuxGpuMonth
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    _checks_dir = OUT / "work" / "checks"
    _checks_dir.mkdir(parents=True, exist_ok=True)
    for _month in e.ARGS.months:
        for _seed in e.ARGS.seeds:
            _name = f"checks_fit_{_month}_{E00735_ARM}_s{_seed}_{os.environ.get('E00677_PHASE', 'train')}.json"
            (_checks_dir / _name).write_text(
                _json2.dumps(
                    {
                        "month": _month,
                        "arm": E00735_ARM,
                        "seed": _seed,
                        "code_hash": E00735_CODE_HASH,
                        "checks": {a: v for a, v in e.CHECK.items() if a != "months"},
                        "month_checks": e.CHECK["months"].get(_month, {}),
                    },
                    indent=1,
                    default=str,
                )
            )
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
elif MINE.command == "valpred":
    # >>> E00735 PATCH BEGIN (H: the smoke test's forecast of the stage-3 validation rows; the scored month stays closed)
    import polars as _pl

    e.k.build_month = arm.build_month_syn
    _clean = e.c.clean_stack()
    _month = MINE.months[0]
    _mr = e.k.build_month(month=_month, clean=_clean)
    _cut = e.c.month_start(month=_month)
    _refit = e.c.select_rows(rows=_mr, cutoff=_cut, half_life_days=e.SPEC.half_life)
    _refit = e.embargoed(mr=_mr, refit=_refit, cut=_cut)
    _val = _refit.index[_refit.is_val]
    print(f"{_month}: {len(_val):,} validation rows of stage 3", flush=True)
    for _seed in e.ARGS.seeds:
        _model = e.new_model(n_primaries=len(_mr.base.target_mean)).to("cuda")
        _model.load_state_dict(
            torch.load(
                e.model_file(tag=e.SPEC.tag, month=_month, day=0, seed=_seed), map_location="cuda"
            )
        )
        _model.eval()
        _model.table = e.CTX["table_ft"]
        _q = []
        for _a in range(0, len(_val), 65_536):
            _s = _val[_a : _a + 65_536]
            _x = torch.from_numpy(_mr.x[_s]).to("cuda")
            _p = torch.from_numpy(_mr.primary[_s]).to("cuda")
            _raw = e.c.serving_quantiles(model=_model, x=_x, p=_p, outlier=False)
            _mu = torch.from_numpy(_mr.base.target_mean).to("cuda")[_p][:, None]
            _sd = torch.from_numpy(_mr.base.target_std).to("cuda")[_p][:, None]
            _q.append(torch.sort(_raw * _sd + _mu, dim=1).values.float().cpu().numpy())
        _q = np.concatenate(_q)
        _frame = _pl.DataFrame(
            {
                "primary": _mr.primary[_val],
                "origin_us": _mr.origin_us[_val],
                "valid_us": _mr.valid_us[_val],
                "y": _mr.y[_val],
                **{
                    f"q{i:02d}": _q[:, j]
                    for j, i in enumerate((1, 5, 10, 25, 40, 50, 60, 75, 90, 95, 99))
                },
            }
        )
        _path = OUT / "work" / "valpred" / f"{_month}_d0_s{_seed}.parquet"
        _path.parent.mkdir(parents=True, exist_ok=True)
        _tmp = _path.with_suffix(".tmp")
        _frame.write_parquet(_tmp)
        _tmp.rename(_path)  # atomic, so `have_valpred` never sees a half-written file
        print("wrote", _path, _frame.height, "rows", flush=True)
        del _model
    # <<< E00735 PATCH END (H: the smoke test's forecast of the stage-3 validation rows; the scored month stays closed)
torch.cuda.empty_cache()
