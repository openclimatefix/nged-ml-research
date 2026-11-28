# ruff: noqa: ANN001, ANN003, ANN201, ANN202, D103, E402, E501, I001, PT018
"""Experiment 735 part 2: stages 1, 2, and 3 of the step-5 network with the date-time encoder's 8 columns.

README. This script runs stage 1 ("pretrain"), stage 2 ("step2"), stage 3 ("fit": train phase, then
"pfit": predict phase), and the smoke test's validation-row forecast ("valpred") of one arm (R1, SKE,
SKE-S21, or SKE-CP) at one refit month. It follows experiment 735 part 1's runner `e00735_k1.py`
(a copy of experiment 677's `e00677_k1.py`), and, like it, imports `e00591_arm` (experiment 591's arm
A1 wrapper around experiment 574's `e00574_fit`) with its module state patched, so it executes on
import and is run as a script. It relies on experiments 677 (`e00677_fold`), 574, 581, 591, 510,
on part 1 (`e00735_columns`, `e00735_diagnostic`), and on part 2's `e00735b_columns`,
`e00735b_model`, and `e00735b_stage1`.

PYTHONPATH: source `ml-experiments/00524*/env.sh` from the worktree root. Set `DATA_PATH_INTERNAL`,
`E00677_ROOT` (must be the folder of experiment 735 part 2 under `data/ml-experiments/`, or a scratch
folder whose name contains `00735`), `E00677_MONTH`, `E00677_SPAN=1`, `E00677_PHASE` (train or
predict), `STATS677` (experiment 677's `chains/3b/<month>/work`, read only: the stored B1
statistics), and `E00735B_ARM`. Use `e00735b_run.sh`, which sets all of them. `E00735B_SMOKE=1` caps
every stage at one epoch.

**Every arm trains stage 1 itself.** The date-time encoder of SKE, SKE-S21, and SKE-CP has 8 more
inputs than the encoder of experiment 677's stored stage 1, so no arm reads experiment 677's
pre-training weights. Stage 1 writes `arms/k1_<ARM>_sp1/work/pretrained_P2/<month>_s<seed>.pt` under
this experiment's own folder, stage 2 reads it and writes `step2/P5`, and stage 3 reads stage 2's
file. Stage 1 re-derives the B1 statistics into `work/stats/B1/<month>.npz` and asserts that they
equal experiment 677's stored file, which stages 2 and 3 read.

The 8 columns follow the 60 columns of a stage-2 or stage-3 row (input index 60 to 67), and stage 3's
two label columns follow them. The tables of the lag path (`NewColumnTables`) are built from the same
function, so a lag's columns equal the row path's columns at the same valid time. The twins of the
main frame (lags removed, lags aged) are not forecast, because no score reads them. Marked patch
blocks (search for `E00735B PATCH`) hold every change to the code of experiment 677.

Usage: python e00735b_k1.py NAME {pretrain|step2|fit|valpred} --months M --seeds S...
"""

import argparse
import hashlib as _hashlib
import json as _json2
import os
import sys
from pathlib import Path
from typing import Final as _Final

_p = argparse.ArgumentParser()
_p.add_argument("name")
_p.add_argument("command", choices=("fit", "pretrain", "step2", "valpred"))
_p.add_argument("--months", nargs="+", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
MINE = _p.parse_args()
_ORIGINAL_COMMAND = MINE.command
sys.argv = [
    "e00591_arm.py", "A1", "fit" if MINE.command == "valpred" else MINE.command,
    "--months", *MINE.months, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00677_fold as fold

assert list(MINE.months) == [fold.FINAL_MONTH], "one process per fold month: set E00677_MONTH"

ROOT = (
    Path(os.environ["E00677_ROOT"])
    if os.environ.get("E00677_ROOT")
    else ML_EXPERIMENTS_DATA_DIR / "00735-school-columns-in-the-date-time-encoder-part-2"
)
# >>> E00735B PATCH BEGIN (A: root assertion, arm, and code hash)
import dataclasses as _dc

import e00735_columns as _cols735_p1  # ty: ignore[unresolved-import]
import e00735b_columns as _cols
import e00735b_model as _model
import e00735b_stage1 as _stage1

E00735B_ROOT_OK: _Final = os.environ.get("E00677_ROOT", "")
_cols.check_output_root(
    root=E00735B_ROOT_OK
)  # set E00677_ROOT to experiment 735 part 2's own folder
_cols.check_output_root(root=str(ROOT))
E00735B_ARM: _Final = os.environ.get("E00735B_ARM", "")
assert E00735B_ARM in _cols.NEW_ARMS, ("set E00735B_ARM", E00735B_ARM)
assert MINE.name == f"k1_{E00735B_ARM}", (MINE.name, E00735B_ARM)
E00735B_SMOKE: _Final = os.environ.get("E00735B_SMOKE", "") == "1"
SPEC735: _Final = _cols.ARMS[E00735B_ARM]
HASHED_FILES: _Final = (
    Path(__file__).parent / "e00735b_k1.py",
    Path(__file__).parent / "e00735b_columns.py",
    Path(__file__).parent / "e00735b_model.py",
    Path(__file__).parent / "e00735b_stage1.py",
    Path(_cols735_p1.__file__),
    Path(_cols735_p1.__file__).parent / "e00735_diagnostic.py",
)
E00735B_CODE_HASH: _Final = _hashlib.sha256(
    b"".join(f.read_bytes() for f in HASHED_FILES)
).hexdigest()
"""The sha256 of this file, part 2's columns, model, and stage-1 modules, and part 1's column and diagnostic modules (the school calendar and the coast distances): every fit records it, and a changed code state refuses old outputs. `e00735b_run.sh` computes the same hash in the same order."""
# <<< E00735B PATCH END (A: root assertion, arm, and code hash)
OUT = ROOT / "arms" / f"{MINE.name}_sp{fold.SPAN_MONTHS}"
"""One folder per scored span, so the monthly pilot and the quarterly fold never share fits."""
DOWN_WEIGHT = 0.5
DOWN_WEIGHT_DAYS = 7
E3_EMBARGO_DAYS = 7

e = arm.e
assert arm.LAMBDA == 0.0, "A1 has no auxiliary loss"
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
# >>> E00735B PATCH BEGIN (B: the chain is this arm's own folder; nothing is read from experiment 677's stage 1)
arm.CHAIN = OUT / "work"
assert "00735" in str(arm.CHAIN) and "00677" not in str(arm.CHAIN)
STATS_READ: _Final = Path(os.environ["STATS677"]) / "stats" / "B1"
"""Experiment 677's stored B1 statistics of the month, read only."""
# <<< E00735B PATCH END (B: the chain is this arm's own folder; nothing is read from experiment 677's stage 1)
fold.install(e, months=MINE.months)
# >>> E00735B PATCH BEGIN (C: every command writes to this arm's own folder; stage 1 writes its own B1 statistics)
e.WORK = OUT
e.W = OUT / "work"
_OWN_STATS: _Final = e.W / "stats" / "B1"


def _stats_file(*, month: str):
    """Return where the month's B1 statistics are read: stage 1 writes its own, the later stages read experiment 677's."""
    folder = _OWN_STATS if MINE.command == "pretrain" else STATS_READ
    return folder / f"{month}.npz"


e.stats_file = _stats_file
arm.check_stats = lambda: None
# <<< E00735B PATCH END (C: every command writes to this arm's own folder; stage 1 writes its own B1 statistics)

# >>> E00735B PATCH BEGIN (D: the arm's columns, network, and smoke cap)
_C_TRUE, _C_PERMUTED = _cols.coast_arrays()
TABLES: _Final = _stage1.NewColumnTables(
    arm=SPEC735, first_us=e.c.prep422.GRID_START_US, step_us=e.c.prep422.STEP_US, device="cuda",
    c_true=_C_TRUE, c_permuted=_C_PERMUTED,
)  # fmt: skip
_GRID_FIRST_US: _Final = e.c.prep422.GRID_START_US
_GRID_STEP_US: _Final = e.c.prep422.STEP_US


def _check_lag_table(*, block, mr) -> None:
    """Check on 1,000 random rows that the lag table equals the row columns (same grid index and valid time).

    Uses the table that the network will read when the expected-input table is already attached;
    before that, a table of just the length the rows need, which holds the same values.
    """
    table = e.CTX.get("table_ft")
    if table is not None:
        new = TABLES.get(n_grid=table.shape[1])
    else:
        n_grid = int((mr.valid_us.max() - _GRID_FIRST_US) // _GRID_STEP_US) + 1
        new = torch.from_numpy(
            _cols.grid_table(
                arm=SPEC735, first_us=_GRID_FIRST_US, step_us=_GRID_STEP_US, n_grid=n_grid,
                c_true=_C_TRUE, c_permuted=_C_PERMUTED,
            )
        )  # fmt: skip
    _stage1.check_table_matches_rows(
        table=new, block=block, primary=mr.primary, valid_us=mr.valid_us,
        first_us=_GRID_FIRST_US, step_us=_GRID_STEP_US,
    )  # fmt: skip


if E00735B_SMOKE:
    e.SPEC = _dc.replace(e.SPEC, max_epochs=1)
assert e.SPEC.sched == "3sE" and e.SPEC.half_life == 90.0 and e.SPEC.pretrain_from == "P2"
assert e.SPEC.frozen, "stages 2 and 3 freeze the weather and date-time encoders"


def new_model_735(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return experiment 677's network (R1) or the network whose date-time encoder takes the 8 columns."""
    model = _model.make_network(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, n_new=SPEC735.width
    )  # fmt: skip
    table = e.CTX.get(f"table_{stage}")
    model.table = table  # ty: ignore[invalid-assignment]
    if SPEC735.width:
        assert table is not None, "attach the expected-input table before building the network"
        model.table_new = TABLES.get(n_grid=table.shape[1])
    return model


e.new_model = new_model_735


def _with_columns(inner):
    """Wrap a month builder so that its rows carry the arm's 8 columns after the B1 columns."""

    def build_month_columns(*, month, clean):
        mr = inner(month=month, clean=clean)
        if not SPEC735.width or e.PRETRAINING["on"]:
            return mr  # stage 1 builds its own matrix in `e00735b_stage1.pretrain_arm`
        assert mr.x.shape[1] == 60, mr.x.shape
        block = _cols.extra_columns(
            valid_us=mr.valid_us,
            primary=mr.primary,
            arm=SPEC735,
            c_true=_C_TRUE,
            c_permuted=_C_PERMUTED,
        )  # raises if a valid time is outside the school calendar
        _cols.check_block(arm=SPEC735, block=block, valid_us=mr.valid_us, kind="train")
        _check_lag_table(block=block, mr=mr)
        mr.x = np.concatenate([mr.x, block], axis=1)
        assert mr.x.dtype == np.float32 and mr.x.shape[1] == 60 + SPEC735.width, mr.x.shape
        share = (block != 0).mean(axis=0)
        e.CHECK["months"][month]["new_columns_train"] = dict(
            zip(SPEC735.names, share.round(4).tolist(), strict=True)
        )
        print(month, "new columns appended; non-zero shares", e.CHECK["months"][month]["new_columns_train"], flush=True)  # fmt: skip
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
    block = _cols.extra_columns(
        valid_us=test.valid,
        primary=test.primary,
        arm=SPEC735,
        c_true=_C_TRUE,
        c_permuted=_C_PERMUTED,
    )
    _cols.check_block(arm=SPEC735, block=block, valid_us=test.valid, kind="test")
    share = (block != 0).mean(axis=0)
    e.CHECK["months"].setdefault(month, {})["new_columns_test"] = dict(
        zip(SPEC735.names, share.round(4).tolist(), strict=True)
    )
    print(month, "test new columns; non-zero shares", e.CHECK["months"][month]["new_columns_test"], flush=True)  # fmt: skip
    joined = np.concatenate([test.x, block], axis=1)
    assert joined.dtype == np.float32, joined.dtype
    return _dc.replace(test, x=joined)


e.data.build_test = build_test_columns
# <<< E00735B PATCH END (D: the arm's columns, network, and smoke cap)

if MINE.command in {"fit", "valpred"}:
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

# >>> E00735B PATCH BEGIN (E: code-hash marker and per-epoch loss history)
_marker = OUT / "work" / f"code_hash_{MINE.months[0]}.txt"
_existing = [
    *(OUT / "work" / "pretrained_P2").glob(f"{MINE.months[0]}_s*.pt"),
    *(OUT / "work" / "step2" / "P5").glob(f"{MINE.months[0]}_s*.pt"),
    *(OUT / "work" / "models" / "P5").glob(f"{MINE.months[0]}_d0_s*.pt"),
    *(OUT / "work" / "fits" / "P5").glob(f"{MINE.months[0]}_d0_s*.parquet"),
    *(OUT / "work" / "history").glob(f"{MINE.months[0]}_d0_s*.json"),
]
if _marker.exists():
    assert _marker.read_text().strip() == E00735B_CODE_HASH, (
        "the experiment 735 part 2 code changed since this arm and month were fitted: delete its outputs first",
        str(_marker),
    )
else:
    assert not _existing, ("outputs without a code-hash marker: delete them", str(_marker))
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.write_text(E00735B_CODE_HASH + "\n")
print("CODE HASH", E00735B_CODE_HASH, "ARM", E00735B_ARM, "columns", SPEC735.names, "smoke", E00735B_SMOKE, flush=True)  # fmt: skip
HIST = _stage1.History()
_write_history = _stage1.write_json
# <<< E00735B PATCH END (E: code-hash marker and per-epoch loss history)


def _check_b1_stats(month: str) -> None:
    """Raise unless the B1 statistics that stage 1 wrote for the month equal experiment 677's stored statistics."""
    mine, stored = np.load(_OWN_STATS / f"{month}.npz"), np.load(STATS_READ / f"{month}.npz")
    assert np.array_equal(mine["mean"], stored["mean"]) and np.array_equal(mine["std"], stored["std"]), (
        "stage 1's B1 statistics differ from experiment 677's stored statistics",
        month,
    )  # fmt: skip


# >>> E00735B PATCH BEGIN (H: the real-run stage gate of stages 2 and 3)
_orig_refit_tensors = e.LagGpuMonth.refit_tensors


def _refit_tensors_stash(self, *, refit, shift_z=None):
    """Return the tensors of the wrapped method, and keep the selected rows' valid times for the stage gate."""
    t = _orig_refit_tensors(self, refit=refit, shift_z=shift_z)
    e.CTX["gate_valid_us"] = self.mr.valid_us[refit.index]
    return t


e.LagGpuMonth.refit_tensors = _refit_tensors_stash


def _gated(*, inner, stage: int):
    """Wrap `e.train` of stage 2 or 3 with the stage gate, the stage-1 tag check, and the frozen-encoder check."""

    def train_gated(**kwargs):
        model, x = kwargs["model"], kwargs["x"]
        month, seed = f"{e.CTX['cut']:%Y-%m}", e.CTX["seed"]
        valid_us = e.CTX["gate_valid_us"]
        tag = e.pretrained_dir() / f"{month}_s{seed}.tag.json"
        _stage1.check_stage_tag(
            path=tag, arm=SPEC735, model=model
        )  # after the load, before any training
        _stage1.gate_stage(
            model=model, x=x, primary=kwargs["p"], is_val=kwargs["is_val"], valid_us=valid_us, arm=SPEC735,
            stage=stage, month=month, seed=seed, out_dir=OUT / "work" / "checks",
            first_us=_GRID_FIRST_US, step_us=_GRID_STEP_US, c_true=_C_TRUE, c_permuted=_C_PERMUTED,
        )  # fmt: skip
        result = inner(**kwargs)
        _stage1.check_stage_tag(
            path=tag, arm=SPEC735, model=model
        )  # the frozen encoder did not move
        return result

    return train_gated


# <<< E00735B PATCH END (H: the real-run stage gate of stages 2 and 3)

if MINE.command == "pretrain":
    # >>> E00735B PATCH BEGIN (F1: stage 1 trains the arm's own network on ERA5 and CAMS rows)
    _stage1.pretrain_arm(
        e=e, months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda", arm=SPEC735,
        c_true=_C_TRUE, c_permuted=_C_PERMUTED, history_dir=OUT / "work" / "history_pre",
        smoke=E00735B_SMOKE, code_hash=E00735B_CODE_HASH,
        checks_dir=OUT / "work" / "checks", check_b1_stats=_check_b1_stats,
    )  # fmt: skip
    print("PRETRAIN DONE", {a: v for a, v in e.CHECK.items() if a != "months"}, flush=True)
    # <<< E00735B PATCH END (F1: stage 1 trains the arm's own network on ERA5 and CAMS rows)
elif MINE.command == "step2":
    # >>> E00735B PATCH BEGIN (F: stage 2 records its per-epoch loss history)
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
                "arm": E00735B_ARM, "stage": 2, "code_hash": E00735B_CODE_HASH,
                "max_epochs": e.SPEC.max_epochs, "epochs_run": epochs, "stored_epoch": stored_epoch,
                "val_nll": best, "history": rows,
            },
        )  # fmt: skip
        return epochs, best

    e.train = _gated(inner=train_recorded, stage=2)
    # <<< E00735B PATCH END (F: stage 2 records its per-epoch loss history)
    e.step2(months=e.ARGS.months, seeds=e.ARGS.seeds, device="cuda")
    print("STEP2 DONE", {a: v for a, v in e.CHECK.items() if a != "months"}, flush=True)
elif MINE.command == "fit":
    e.row_nll = arm.row_nll
    # >>> E00735B PATCH BEGIN (G: diagnostics and per-epoch loss history of stage 3; no change to the fit)
    _outer_select_rows = e.c.select_rows

    def select_rows_report(**kwargs):
        """Return the rows of the wrapped `select_rows`, with the total weight and effective rows in the report."""
        refit = _outer_select_rows(**kwargs)
        w64 = refit.weight.astype(np.float64)
        refit.report["total_weight"] = float(w64.sum())
        refit.report["effective_rows"] = float(w64.sum() ** 2 / (w64**2).sum())
        refit.report["arm735b"] = E00735B_ARM
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
            "arm": E00735B_ARM,
            "stage": 3,
            "code_hash": E00735B_CODE_HASH,
            "max_epochs": e.SPEC.max_epochs,
            "epochs_run": len(rows),
            "epochs_in_log": log["epochs"],
            "stored_epoch": stored_epoch,
            "val_nll_in_log": log["val_nll"],
            "history": rows,
        }
        _write_history(path=path, payload=payload)
        log["history_file"] = str(path)
        log["code_hash"] = E00735B_CODE_HASH
        log["stored_epoch"] = stored_epoch
        return model, log

    e.fine_tune = fine_tune_recorded
    e.train = _gated(inner=e.train, stage=3)
    # <<< E00735B PATCH END (G: diagnostics and per-epoch loss history of stage 3; no change to the fit)
    e.k.build_month = arm.build_month_syn
    e.k.GpuMonth = arm.AuxGpuMonth
    e.fit_arm(months=e.ARGS.months, seeds=e.ARGS.seeds, stress=False, limit=None, device="cuda")
    _checks_dir = OUT / "work" / "checks"
    _checks_dir.mkdir(parents=True, exist_ok=True)
    for _month in e.ARGS.months:
        for _seed in e.ARGS.seeds:
            _name = f"checks_fit_{_month}_{E00735B_ARM}_s{_seed}_{os.environ.get('E00677_PHASE', 'train')}.json"
            (_checks_dir / _name).write_text(
                _json2.dumps(
                    {
                        "month": _month,
                        "arm": E00735B_ARM,
                        "seed": _seed,
                        "code_hash": E00735B_CODE_HASH,
                        "checks": {a: v for a, v in e.CHECK.items() if a != "months"},
                        "month_checks": e.CHECK["months"].get(_month, {}),
                    },
                    indent=1,
                    default=str,
                )
            )
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})
elif MINE.command == "valpred":
    # >>> E00735B PATCH BEGIN (H: the smoke test's forecast of the stage-3 validation rows; the scored month stays closed)
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
        _model_net = e.new_model(n_primaries=len(_mr.base.target_mean)).to("cuda")
        _model_net.load_state_dict(
            torch.load(
                e.model_file(tag=e.SPEC.tag, month=_month, day=0, seed=_seed), map_location="cuda"
            )
        )
        _model_net.eval()
        _model_net.table = e.CTX["table_ft"]
        _q = []
        for _a in range(0, len(_val), 65_536):
            _s = _val[_a : _a + 65_536]
            _x = torch.from_numpy(_mr.x[_s]).to("cuda")
            _p = torch.from_numpy(_mr.primary[_s]).to("cuda")
            _raw = e.c.serving_quantiles(model=_model_net, x=_x, p=_p, outlier=False)
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
        del _model_net
    # <<< E00735B PATCH END (H: the smoke test's forecast of the stage-3 validation rows; the scored month stays closed)
torch.cuda.empty_cache()
