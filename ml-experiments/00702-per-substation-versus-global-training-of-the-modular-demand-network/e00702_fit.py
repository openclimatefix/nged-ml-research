# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, E402, E501, PLW0108, D102, PT018
"""Experiment 702: fit and forecast the per-substation arms of K1 for one fold (GPU).

A patched copy of experiment 677's `e00677_k1.py`: it imports experiment 591's arm A1 (K1 with the
shared lag encoder) with its module state patched, installs experiment 677's fold, and applies the
patches below. Nothing in an earlier experiment's folder is edited.

Usage (one process per command; `chain_arm.sh` runs them in order):

    E00677_MONTH=M E00677_SPAN=1 E00677_ROOT=<this experiment's data folder> E00677_PHASE=train|predict \
      python e00702_fit.py ARM COMMAND --month M [--primaries D5|all|D5,D9] [--seeds 0 1 2] \
        [--batch 512] [--suffix _b8192]

ARM and COMMAND:

- `split1` (every phase per Primary; one process per fold and Primary, `--primaries D5`):
  `pretrain`, `step2`, `fit`, `bank`, `isolation`. The process sees one Primary only (`e00702_world`).
- `split2` (stored pooled pre-training; step 2 and 3 per Primary): `step2`, `fit`, `bank`.
- `hybrid3` (as split2 plus two new encoders): `step2`, `fit`, `bank`.
- `split3` (stored pooled pre-training and step 2; step 3 per Primary): `fit`, `bank`, `spans`,
  `isolation`, `weather`.
- `global` (the control: all 16 Primaries together, batch 8,192, no output mask, written to
  `arms/global<suffix>/<month>/pooled/`): `pretrain`, `step2`, `fit`, `bank`. `bank --stored-models`
  runs the bank on experiment 677's stored step-3 networks (seed by seed) so that the output can be
  compared with experiment 677's own bank file (`e00702_score.py match-bank`).

A pooled arm builds the fold's rows once and loops over `--primaries` (default all 16). It keeps the
fold's pooled standardiser and restricts the training rows, the early-stopping rows, and the test
rows to one Primary through `select_rows(drop=...)` and a filter on the forecast rows.

Phases follow experiment 677: `E00677_PHASE=train` hides the held-out power and writes `.empty`
markers instead of forecasts; `predict` opens the window (after the markers are deleted).
"""

import argparse
import dataclasses
import json
import os
import resource
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00702_core as core
import e00702_world as world

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=core.FIT_ARMS)
_p.add_argument(
    "command", choices=("pretrain", "step2", "fit", "bank", "spans", "isolation", "weather")
)
_p.add_argument("--month", required=True)
_p.add_argument("--seeds", nargs="+", type=int, default=list(core.SEEDS))
_p.add_argument("--primaries", default="all")
_p.add_argument(
    "--batch",
    type=int,
    default=None,
    help="rows per step (default: 8,192 for global, 512 otherwise)",
)
_p.add_argument(
    "--stored-models",
    action="store_true",
    help="bank only: read the networks of experiment 677 (global arm)",
)
_p.add_argument("--suffix", default="")
MINE = _p.parse_args()
MONTH, ARM, CMD = MINE.month, MINE.arm, MINE.command
BATCH: Final[int] = MINE.batch or core.default_batch(arm=ARM)
ALLOWED = {
    "split1": {"pretrain", "step2", "fit", "bank", "isolation", "weather"},
    "split2": {"step2", "fit", "bank"},
    "hybrid3": {"step2", "fit", "bank"},
    "split3": {"fit", "bank", "spans", "weather"},
    "global": {"pretrain", "step2", "fit", "bank"},
}
assert CMD in ALLOWED[ARM], f"{ARM} has no command {CMD}"
assert not MINE.stored_models or (ARM == "global" and CMD == "bank"), (
    "--stored-models is for `global bank`"
)
assert os.environ.get("E00677_MONTH") == MONTH, "set E00677_MONTH to the fold month"
assert os.environ.get("E00677_SPAN") == "1", "monthly refits: set E00677_SPAN=1"
assert os.environ.get("E00677_PHASE") in {"train", "predict"}, "set E00677_PHASE"
assert Path(os.environ.get("E00677_ROOT", "")) == core.ROOT, (
    "set E00677_ROOT to this experiment's data folder, so that no file is written into experiment 677's"
)
POOLED = ARM != "split1"
GLOBAL = ARM == "global"
RESTRICT = POOLED and not GLOBAL
"""Whether a pooled arm keeps only one Primary's rows. The global arm keeps all 16."""
if not POOLED:
    assert "," not in MINE.primaries and MINE.primaries != "all", (
        "split1 takes one Primary per process"
    )
    import e00087_common as common  # ty: ignore[unresolved-import]

    ALL_LABELS = world.primary_labels(series_table=common.series_table)
    GLOBAL_INDEX = world.restrict_to_primary(common=common, label=MINE.primaries)
    """The Primary's index among the 16, which the shift schedule and the stored 16-row tables use."""

sys.argv = [
    "e00591_arm.py", "A1", "pretrain" if CMD == "pretrain" else "step2" if CMD == "step2" else "fit",
    "--months", MONTH, "--seeds", *map(str, MINE.seeds),
]  # fmt: skip

import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00677_fold as fold  # ty: ignore[unresolved-import]
import e00702_model as model702
import e00702_train as train702
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

e = arm.e
c, k, f, data = e.c, e.k, e.f, e.data
STATS_677: Final[Path] = core.ROOT_677 / "chains" / "3b" / MONTH / "work"
CHAIN_677: Final[Path] = core.ROOT_677 / "chains" / "5" / MONTH / "work"
OUT: Final[Path] = core.ROOT / "arms" / f"{ARM}{MINE.suffix}" / MONTH
assert e.SPEC.tag == "P5" and arm.LAMBDA == 0.0 and e.EMBARGO_DAYS == 3
assert (
    core.BATCH_GLOBAL, train702.SETTINGS.patience, train702.SETTINGS.row_fraction,
    train702.SETTINGS.weight_decay, train702.SETTINGS.grad_clip, train702.SETTINGS.validation_rows,
) == (c.BATCH, c.PATIENCE, c.ROW_FRACTION, c.WEIGHT_DECAY, c.GRAD_CLIP, c.VALIDATION_ROWS)  # fmt: skip
arm.CHAIN = CHAIN_677
fold.install(e, months=[MONTH])
assert fold.FINAL_MONTH == MONTH
FOLD_CUT_IDX: Final[int] = int(
    (c.us(t=c.month_start(month=MONTH)) - c.prep422.GRID_START_US) // c.prep422.STEP_US
)
"""Grid index of the fold's cutoff: in the train phase `e00677_fold` hides the cleaned power from here on."""

SPANS_FILE: Final[Path] = core.ROOT / "fold_spans" / f"{MONTH}.json"
"""The pooled rows' spans and shifted-row hashes of the fold, written by `split3 spans` (one file per fold, so that two folds' runs never write the same file)."""
DAY_US: Final[int] = 86_400 * 1_000_000
FIRST_RUN_US: Final[int] = c.us(t=e.prep.FIRST_RUN)
STATE: dict = {
    "step": None, "origin_day": None, "phases": [], "label": None, "index": None,
    "timings": [], "floored_columns": None, "arm_label": f"{ARM}{MINE.suffix}",
}  # fmt: skip
"""`step`: the schedule's step (1, 2, or 3) of the phase that is training. `origin_day`: the origin day of each selected row. `index`: the Primary's index among the 16 (pooled arms)."""
MEMO: dict = {}
BANK_SEED_BASE: Final[int] = 9_000_000
"""Experiment 677's seed base for the held-out bank (`e00677_cand.BANK_SEED_BASE`)."""


def first_idx() -> int:
    """Return the earliest training target's grid index, from experiment 397's first run and experiment 422's purge."""
    origin = e.prep.FIRST_RUN + e.common.PUBLICATION_DELAY
    origin_idx = int((c.us(t=origin) - c.prep422.GRID_START_US) // c.prep422.STEP_US)
    _, purged = c.prep422.purge_masks(origin_day=np.arange(c.prep422.CYCLE_DAYS))
    return core.first_target_idx(first_origin_idx=origin_idx, purged_day=purged)


def origin_day_of(*, origin_us: np.ndarray) -> np.ndarray:
    """Return the origin day of each origin time, as `e00428_data.build_fold` counts it."""
    return ((np.asarray(origin_us) - FIRST_RUN_US) // DAY_US).astype(np.int64)


# ------------------------------------------------------------------ one-Primary world (split-at-1)
if not POOLED:
    import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00560_data as d560  # ty: ignore[unresolved-import]
    import e00581_bank as bank  # ty: ignore[unresolved-import]

    prep422.cleaned_arrays = lambda: world.one_site_cleaned_arrays(
        prep422=prep422, fd=fd, common=common, flagged_keys=prep422.stale.FLAGGED_KEYS
    )
    d560.raw_table = world.sliced_raw_table(index=GLOBAL_INDEX)
    CUTOFF_IDX = int(
        (c.us(t=c.month_start(month=MONTH)) - c.prep422.GRID_START_US) // c.prep422.STEP_US
    )
    SPAN = core.fold_span(cutoff_idx=CUTOFF_IDX, first_idx=first_idx())
    assert SPANS_FILE.exists(), (
        "run `e00702_fit.py split3 spans` first: it records the pooled rows' spans"
    )
    POOLED_SPANS = json.loads(SPANS_FILE.read_text())
    assert POOLED_SPANS["train"] == list(SPAN), (
        "the fold's span differs from the pooled rows' span (`split3 spans`)"
    )
    ORIGINAL_SYNTHETIC = bank.synthetic_clean
    bank.synthetic_clean = world.schedule_trick(
        original=ORIGINAL_SYNTHETIC, index=GLOBAL_INDEX, n_primaries=core.N_PRIMARIES,
        first=SPAN[0], last=SPAN[1],
    )  # fmt: skip
    assert e.prep.primaries().height == 1 and fd._series_and_arrays()[0].height == 1
    assert e.prep.primaries()["primary"].to_list() == [0], "the one Primary must have index 0"
    assert e.prep.primaries()["site"].to_list() == [MINE.primaries], e.prep.primaries()["site"]
    OPENED = world.guard_readers(
        own=MINE.primaries, labels=ALL_LABELS,
        modules={
            "np.load": (np, "load"), "pl.read_parquet": (pl, "read_parquet"),
            "pl.scan_parquet": (pl, "scan_parquet"),
        },
    )  # fmt: skip
    """Every path the process opens through the guarded readers; opening another Primary's per-site file raises."""


# ------------------------------------------------------------------ the training loop
def patched_train(
    *, model, x, z, w, p, is_val, seed, lr, max_epochs, fresh_lr, frozen, anchor, lam, device
):
    """Run `e00702_train.train` with the batch, and the fixed epoch count where the held weeks are too few."""
    assert anchor is None and lam == 0.0
    step = STATE["step"]
    held = is_val.cpu().numpy()
    assert STATE["origin_day"] is not None and len(STATE["origin_day"]) == len(held), (
        "origin days not recorded for this phase"
    )
    fixed = None
    if not GLOBAL and core.needs_fixed_epochs(origin_day=STATE["origin_day"], is_val=held):
        fixed = core.FIXED_EPOCHS[step]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    epochs, best = train702.train(
        model=model, x=x, z=z, w=w, p=p, is_val=is_val, seed=seed, lr=lr, max_epochs=max_epochs,
        fresh_lr=fresh_lr, frozen=frozen, batch=BATCH, device=device,
        row_nll=lambda **kw: e.row_nll(**kw), fixed_epochs=fixed,
    )  # fmt: skip
    weeks = np.unique(STATE["origin_day"][held] // core.BLOCK_DAYS) if held.any() else []
    STATE["phases"].append({
        "label": STATE["label"], "step": step, "seed": seed, "epochs": epochs, "fixed_epochs": fixed,
        "val_nll": best, "held_rows": int(held.sum()), "held_weeks": len(weeks),
        "train_rows": int((~held).sum()), "batch": BATCH,
        "seconds": time.perf_counter() - start,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
        "host_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })  # fmt: skip
    return epochs, best


def phase(*, step: str, name: str) -> None:
    """Wrap `e.<name>` so that `STATE['step']` names the phase while it runs."""
    original = getattr(e, name)

    def wrapped(*args, **kwargs):
        STATE["step"] = step
        return original(*args, **kwargs)

    setattr(e, name, wrapped)


e.train = patched_train
for _step, _name in (("1", "pretrain_wt"), ("2", "step2"), ("3", "fine_tune")):
    phase(step=_step, name=_name)
_pretrain_rows = e.r448.pretrain_rows


def pretrain_rows(**kwargs):
    """Stash the origin days and the early-stopping flags of the pre-training rows for the fixed-epoch rule."""
    pre = _pretrain_rows(**kwargs)
    origin_us = c.prep422.GRID_START_US + pre.origin_idx.astype(np.int64) * c.prep422.STEP_US
    STATE["origin_day"] = origin_day_of(origin_us=origin_us)
    return pre


e.r448.pretrain_rows = pretrain_rows


# ------------------------------------------------------------------ networks and weight files
def new_model(*, n_primaries: int, stage: str = "ft"):
    if ARM == "hybrid3":
        assert stage == "ft"
        model = model702.Hybrid3Mdn(n_primaries=n_primaries, n_knots=e.KNOTS, n_b1=e.N_B1)
        model.table = e.CTX.get("table_ft")
    else:
        model = arm.new_model(n_primaries=n_primaries, stage=stage)
    if ARM == "split1":
        model702.zero_primary_embedding(model=model)
    if not STATE.get("reported"):
        STATE["reported"] = True
        names = [n for n, _ in model.named_parameters()]
        print(
            f"REPORT arm={ARM} month={MONTH} class={type(model).__name__} stage={stage} "
            f"mixer_ft_in={model.mixer_ft[0].in_features} n_primaries={n_primaries} "
            f"embedding_params={sum(n.startswith('embedding') for n in names)} "
            f"n_params={sum(q.numel() for q in model.parameters())} batch={BATCH} "
            "frozen_encoders=weather,datetime (steps 2 and 3)",
            flush=True,
        )
        print("REPORT parameters:", names, flush=True)
    return model


e.new_model = new_model
if ARM == "hybrid3":
    _fresh = e.load_pretrained_fresh

    def load_pretrained_fresh(*, model, path):
        raw = torch.load(path, map_location="cpu")
        seed = int(Path(path).stem.rsplit("_s", 1)[1])
        model702.load_pretrained_into_hybrid3(model=model, state=raw, seed=seed)
        e.fresh_mixer(model=model)

    e.load_pretrained_fresh = load_pretrained_fresh
CHECK_COMMANDS: Final[frozenset[str]] = frozenset({"spans", "isolation", "weather"})
"""The commands that build rows to check them and train nothing."""
if RESTRICT:
    e.pretrained_file = lambda *, month, seed: CHAIN_677 / "pretrained_P2" / f"{month}_s{seed}.pt"
    e.stats_file = lambda *, month: STATS_677 / "stats" / e.SPEC.b1 / f"{month}.npz"
elif MINE.stored_models or (not POOLED and CMD in CHECK_COMMANDS):
    # experiment 677's networks, or a split-1 check that may run before this Primary's pre-training:
    # experiment 677's B1 statistics (read only). The checks compare unstandardised blocks and the
    # first 37 columns, which the B1 statistics do not scale.
    e.stats_file = lambda *, month: STATS_677 / "stats" / e.SPEC.b1 / f"{month}.npz"
else:
    e.stats_file = lambda *, month: e.W / "stats" / e.SPEC.b1 / f"{month}.npz"
arm.check_stats = lambda: None


def step2_file(*, month: str, seed: int) -> Path:
    """Return the step-2 weights: the stored pooled ones for split-at-3, this arm's own otherwise."""
    if ARM == "split3":
        return CHAIN_677 / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    return e.W / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"


e.step2_file = step2_file

# ------------------------------------------------------------------ rows: restriction and weights
_select_rows = c.select_rows
E3 = CMD in {"fit", "bank"}
if E3:
    e.EMBARGO_DAYS = 7


def select_rows(**kwargs):
    """Return experiment 510's rows, restricted to one Primary (pooled arms) and with experiment 613's E3 weights."""
    rows = kwargs["rows"]
    if RESTRICT:
        other = rows.primary != STATE["index"]
        kwargs["drop"] = other if kwargs.get("drop") is None else (kwargs["drop"] | other)
    refit = _select_rows(**kwargs)
    if RESTRICT:
        assert set(np.unique(rows.primary[refit.index]).tolist()) <= {STATE["index"]}, (
            "another Primary's row was selected"
        )
    if not E3:
        return refit
    cutoff = kwargs["cutoff"]
    age_days = (c.us(t=cutoff) - rows.valid_us[refit.index]) / c.DAY_US
    window = (age_days >= 7) & (age_days < 14)
    weight = np.where(window, refit.weight * 0.5, refit.weight).astype(np.float32)
    report = {**refit.report, "downweighted_rows": int(window.sum()), "embargo_days": 7}
    return c.Refit(refit.index, refit.y, weight, refit.is_val, report)


c.select_rows = select_rows


def memoised(*, key: str, build, by=lambda **kw: None):
    """Return a wrapper that builds a fold-level object once per process (pooled arms loop over Primaries).

    Args:
        key: Names the kind of object.
        build: Builds the object from the keyword arguments of the call.
        by: Maps the keyword arguments to what distinguishes two objects of this kind.
    """

    def wrapper(**kwargs):
        name = (key, by(**kwargs))
        if name not in MEMO:
            MEMO[name] = build(**kwargs)
        return MEMO[name]

    return wrapper


def write_stats_record(*, mr) -> None:
    """Store, or check against the stored copy, the fold's statistics (standardiser, lag stats, spread table, target scale)."""
    if CMD in CHECK_COMMANDS:  # a check's rows are not the record that fit and bank compare
        return
    arrays = {
        "std_mean": mr.base.standardiser.mean, "std_std": mr.base.standardiser.std,
        "lag": np.asarray(e.CTX["stats"]), "spread": e.r448.spread_table(rows=mr.base)[0],
        "target_mean": mr.base.target_mean, "target_std": mr.base.target_std,
    }  # fmt: skip
    path = (
        OUT / (STATE["label"] or "pooled") / "work" / "stats_record.npz"
        if not POOLED
        else OUT / "stats_record_pooled.npz"
    )
    if path.exists():
        core.StatsRecord.check(path=path, arrays=arrays)
    else:
        core.StatsRecord(month=MONTH, primary=-1, arrays=arrays).save(path=path)


SYNTHETIC = core.uses_synthetic_rows(command=CMD)
"""Fit and bank build the month with injected lag shifts; pre-training and step 2 use the observed lags (as experiment 677)."""
_month_build = arm.build_month_syn if SYNTHETIC else arm._build_month_p5
FLOOR_STD: Final[float] = 1e-6
"""The floor that `Standardiser.fit` puts under a column's standard deviation."""


def log_floored_columns(*, mr) -> None:
    """Record which input columns sit at the standard-deviation floor in this world's standardiser.

    A per-Primary standardiser can floor a column that the pooled one does not (a weather mask that
    never varies at one Primary). The columns are logged and flagged, never changed.
    """
    std = np.asarray(mr.base.standardiser.std)
    floored = np.flatnonzero(std <= FLOOR_STD * (1 + 1e-6)).tolist()
    STATE["floored_columns"] = floored
    print(f"FLOORED_COLUMNS {STATE['arm_label']} {MONTH}: {floored}", flush=True)


def build_month(*, month, clean):
    start = time.perf_counter()
    mr = _month_build(month=month, clean=clean)
    STATE["timings"].append({"what": "build_month", "seconds": time.perf_counter() - start})
    assert mr.base.target_mean.shape == ((16,) if POOLED else (1,)), mr.base.target_mean.shape
    write_stats_record(mr=mr)
    log_floored_columns(mr=mr)
    return mr


class Gpu702(arm.AuxGpuMonth if SYNTHETIC else e.LagGpuMonth):
    """The month on the GPU (with the aux labels when the rows are synthetic); also records the selected rows' origin days."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        STATE["origin_day"] = origin_day_of(origin_us=self.mr.origin_us[refit.index])
        return t


assert issubclass(Gpu702, arm.AuxGpuMonth) == SYNTHETIC
e.k.build_month = (
    memoised(key="month", build=build_month, by=lambda *, month, clean: month)
    if POOLED
    else build_month
)
e.k.GpuMonth = (
    memoised(
        key="gpu",
        build=lambda *, mr, device, flag=None: Gpu702(mr=mr, device=device, flag=flag),
        by=lambda *, mr, **_: id(mr),
    )
    if POOLED
    else Gpu702
)
if MINE.stored_models:
    _model_file = e.model_file

    def model_file(**kwargs):
        """Return experiment 677's stored step-3 weights of the global arm (read only)."""
        folder = e.W
        e.W = core.ROOT_677 / "arms" / "k1_5w_sp1" / "work"
        try:
            return _model_file(**kwargs)
        finally:
            e.W = folder

    e.model_file = model_file
e.row_nll = arm.row_nll
if POOLED:
    _clean_stack = c.clean_stack
    c.clean_stack = memoised(key="clean", build=lambda: _clean_stack())
_build_test = data.build_test


def build_test(*, month, standardiser):
    test = _build_test(month=month, standardiser=standardiser)
    e.CTX["variants"] = {}  # the removed/aged twins are 677's slices, not this experiment's
    return test


data.build_test = (
    memoised(key="test", build=build_test, by=lambda *, month, standardiser: month)
    if POOLED
    else build_test
)
e.build_test = data.build_test
_window_frame = f.window_frame


def window_frame(*, model, test, mr, lo, hi, flag, device):
    """Forecast one Primary's rows of the window (pooled arms hold the other Primaries' rows too)."""
    if test is None or GLOBAL:
        return _window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    mask = test.primary == (STATE["index"] if POOLED else 0)
    n = len(test.primary)
    cut = dataclasses.replace(
        test,
        **{
            fld.name: getattr(test, fld.name)[mask]
            for fld in dataclasses.fields(test)
            if isinstance(getattr(test, fld.name), np.ndarray) and len(getattr(test, fld.name)) == n
        },
    )
    frame = _window_frame(model=model, test=cut, mr=mr, lo=lo, hi=hi, flag=flag, device=device)
    if frame is not None:
        assert frame["time_series_id"].n_unique() == 1, "a forecast holds more than one Primary"
    return frame


f.window_frame = window_frame
e.window_frame = window_frame


# ------------------------------------------------------------------ the held-out bank
def run_bank(*, device: str) -> None:
    """Forecast the scored window from the observed lags and from an injected bank of shifts (as experiment 677's `bank`).

    The bank is experiment 581's generator at seed `BANK_SEED_BASE` plus the month number. It shifts
    the lag inputs only. For each scored row whose 1-day lag falls inside a shift, the file holds the
    shift (a share of capacity) and the median forecast from the observed and from the shifted lags.
    """
    assert fold.PHASE == "predict", "the bank needs the scored window open"
    import e00581_bank as bank_module  # ty: ignore[unresolved-import]

    clean = c.clean_stack()
    mr = k.build_month(month=MONTH, clean=clean)
    test = data.build_test(month=MONTH, standardiser=mr.base.standardiser)
    first, last = int(test.target_idx.min()), int(test.target_idx.max())
    if not POOLED:
        first_pooled, last_pooled = POOLED_SPANS["test"]
        assert first_pooled <= first and last <= last_pooled, (
            "one Primary's scored span leaves the pooled span", (first, last), POOLED_SPANS["test"],
        )  # fmt: skip
        first, last = first_pooled, last_pooled
    synthetic = (
        bank_module.synthetic_clean
        if POOLED
        else world.schedule_trick(
            original=ORIGINAL_SYNTHETIC, index=GLOBAL_INDEX, n_primaries=core.N_PRIMARIES,
            first=first, last=last,
        )
    )  # fmt: skip
    fixed, n_events = synthetic(
        clean=clean, first=first, last=last, seed=BANK_SEED_BASE + int(MONTH.replace("-", ""))
    )
    shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0).astype(np.float32)
    blk = e.block_of(
        clean=fixed, primary=test.primary, target_idx=test.target_idx, origin_idx=test.origin_idx
    )
    lag_x = test.x.copy()
    lag_x[:, arm.LAG_FIRST : arm.LAG_FIRST + arm.LAG_WIDTH] = blk
    day_idx = test.x[:, 43].astype(np.int64)
    row_shift = np.where(
        day_idx >= 0, shift[test.primary, np.clip(day_idx, 0, shift.shape[1] - 1)], 0.0
    )
    ts_id = e.prep.primaries()["time_series_id"].to_numpy()
    keys = pl.DataFrame({
        "time_series_id": ts_id[test.primary].astype(np.int64),
        "power_fcst_init_time": test.origin, "valid_time": test.valid, "lag_shift": row_shift,
    }).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )  # fmt: skip
    print(f"{MONTH}: bank span {first}-{last}, {n_events} events", flush=True)
    for label in LABELS:
        use_primary(label=label)
        (e.W / "bank").mkdir(parents=True, exist_ok=True)
        (e.W / "bank" / "span.json").write_text(
            json.dumps({"first": first, "last": last, "events": n_events})
        )
        lo, hi = k.window_of(month=MONTH, cadence=e.CADENCE, index=0)
        for seed in MINE.seeds:
            mfile = e.model_file(tag=e.SPEC.tag, month=MONTH, day=0, seed=seed)
            assert mfile.exists(), mfile
            model = e.new_model(n_primaries=len(mr.base.target_mean)).to(device)
            model.load_state_dict(torch.load(mfile, map_location=device))
            model.eval()
            base = window_frame(
                model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=device
            )
            inj = window_frame(
                model=model, test=dataclasses.replace(test, x=lag_x), mr=mr, lo=lo, hi=hi, flag=False, device=device
            )  # fmt: skip
            k3 = ["time_series_id", "power_fcst_init_time", "valid_time"]
            joined = (
                base.select(*k3, pl.col("q50").alias("median_base"))
                .join(inj.select(*k3, pl.col("q50").alias("median_bank")), on=k3)
                .join(keys, on=k3)
            )
            assert joined.height == base.height == inj.height
            out = e.W / "bank" / f"{MONTH}_d0_s{seed}.parquet"
            joined.write_parquet(out.with_suffix(".tmp"))
            out.with_suffix(".tmp").rename(out)
            print(
                f"{label} s{seed}: {joined.height:,} rows, {int((joined['lag_shift'] != 0).sum()):,} shifted -> {out}",
                flush=True,
            )
            del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ the run
if GLOBAL:
    ALL = e.prep.primaries()["site"].to_list()
    LABELS = ["pooled"]
    assert MINE.primaries == "all", "the global arm trains on all 16 Primaries"
elif POOLED:
    ALL = e.prep.primaries()["site"].to_list()
    LABELS = ALL if MINE.primaries == "all" else MINE.primaries.split(",")
    assert set(LABELS) <= set(ALL), LABELS
else:
    ALL = [MINE.primaries]
    LABELS = ALL


def use_primary(*, label: str) -> None:
    """Point the shared module state at one Primary's own folder (and index, for a pooled arm)."""
    STATE["label"] = label
    STATE["index"] = None if GLOBAL else ALL.index(label) if POOLED else 0
    e.WORK = OUT / label
    e.W = e.WORK / "work"


# One label (split1's Primary, or the global arm's pool): set the folders before any row is built.
if not RESTRICT:
    use_primary(label=LABELS[0])


def finish() -> None:
    for label in LABELS:
        use_primary(label=label)
        mine = [p for p in STATE["phases"] if p["label"] == label]
        (e.W / "logs").mkdir(parents=True, exist_ok=True)
        (e.W / "logs" / f"phases_{CMD}_{fold.PHASE}_s{'-'.join(map(str, MINE.seeds))}.json").write_text(
            json.dumps(
                {
                    "phases": mine, "timings": STATE["timings"],
                    "floored_columns": STATE["floored_columns"],
                    "host_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "cpu_seconds": time.process_time(),
                }, indent=1, default=str,
            )
        )  # fmt: skip


def shifted_row_hashes(*, clean: np.ndarray, spans: dict, synthetic, labels: list[str]) -> dict:
    """Hash each Primary's cleaned row and its shifted rows (the training schedule and the bank's) from a generator `synthetic`.

    `clean_before` and `train_before` hash the grid before the fold's cutoff only, which the train
    phase and the predict phase both see. `clean`, `train`, and `bank` hash the whole row, which only
    the predict phase sees (the train phase hides the power from the cutoff on).
    """
    seed = int(MONTH.replace("-", ""))
    train = synthetic(clean=clean, first=spans["train"][0], last=spans["train"][1], seed=seed)[0]
    scored = synthetic(
        clean=clean, first=spans["test"][0], last=spans["test"][1], seed=BANK_SEED_BASE + seed
    )[0]
    kinds = {
        "clean_before": clean[:, :FOLD_CUT_IDX], "train_before": train[:, :FOLD_CUT_IDX],
        "clean": clean, "train": train, "bank": scored,
    }  # fmt: skip
    return {
        kind: {label: core.array_hash(arrays={"row": rows[i]}) for i, label in enumerate(labels)}
        for kind, rows in kinds.items()
    }


PHASE_KINDS: Final[dict[str, tuple[str, ...]]] = {
    "train": ("clean_before", "train_before"),
    "predict": ("clean_before", "train_before", "clean", "train", "bank"),
}
"""The hashes that each phase can compare: the train phase sees no power from the fold's cutoff on."""


def run_spans() -> None:
    """Write the pooled spans and the hashes of every Primary's cleaned and shifted rows to `fold_spans/<month>.json`, and check the formula."""
    assert fold.PHASE == "predict", (
        "run `spans` with E00677_PHASE=predict: in the train phase `build_test` returns None"
    )
    import e00581_bank as bank_module  # ty: ignore[unresolved-import]

    clean = c.clean_stack()
    mr = k.build_month(month=MONTH, clean=clean)
    test = data.build_test(month=MONTH, standardiser=mr.base.standardiser)
    spans = {
        "train": [int(mr.target_idx.min()), int(mr.target_idx.max())],
        "test": [int(test.target_idx.min()), int(test.target_idx.max())],
    }
    cutoff_idx = int(
        (c.us(t=c.month_start(month=MONTH)) - c.prep422.GRID_START_US) // c.prep422.STEP_US
    )
    formula = list(core.fold_span(cutoff_idx=cutoff_idx, first_idx=first_idx()))
    record = {
        **spans, "formula": formula,
        "shift_hash": shifted_row_hashes(
            clean=clean, spans=spans, synthetic=bank_module.synthetic_clean, labels=ALL
        ),
        "extra_hash": extra_column_hashes(mr=mr, test=test, labels=ALL),
    }  # fmt: skip
    SPANS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SPANS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(SPANS_FILE)
    print(MONTH, "pooled spans", spans, "formula", formula, flush=True)
    assert spans["train"] == formula, "fix core.fold_span: it disagrees with the pooled rows"


class IdentityStandardiser:
    """Returns the raw input matrix, so that `run_inputs` yields unstandardised columns."""

    @staticmethod
    def apply(*, x: np.ndarray) -> np.ndarray:
        return x


def has_member_frame(*, run: datetime) -> bool:
    """Return whether the run has a control-member demand frame, not experiment 677's two-column stand-in.

    The control member's frames stop at the run of 2026-06-29. For later runs
    `common.demand_frame_path` returns a stand-in that holds only `power_fcst_init_time` and
    `nwp_init_time`, and the stored test rows come from the ensemble summary alone
    (`e00677_extend.build_site`).
    """
    path = e.common.demand_frame_path(run=run)
    return "time_series_id" in pl.read_parquet_schema(path)


def summary_run_inputs(*, origin: datetime) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return what `prep.run_inputs` returns for the summary scheme, built from the run's summary file alone.

    A run with no control-member frame needs this path. The rows are built as
    `e00677_extend.build_site` builds them, from the one summary file of the run in
    `prep.SUMMARY_DIR`, so a summary file issued later cannot reach them. The summary scheme's
    weather columns come from the summary in both paths; the control member supplies only the keys.
    """
    prep, common, fd = e.prep, e.common, e.prep.fd
    run = origin - common.PUBLICATION_DELAY
    _series, arrays = fd._series_and_arrays()
    summary = pl.read_parquet(prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet")
    assert (summary["power_fcst_init_time"] == origin).all(), (
        f"the summary file of run {run} holds another origin"
    )
    out = []
    for row in prep.primaries().iter_rows(named=True):
        rows = summary.filter(pl.col("time_series_id") == row["time_series_id"])
        rows = rows.filter(pl.col("valid_time") > pl.col("power_fcst_init_time")).sort("valid_time")
        assert rows.height > 0, f"run {run} has no summary rows for {row['site']}"
        rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
        rows = rows.with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        rows = fd.add_lags(frame=rows, power=arrays[row["site"]].power, a7=False)
        rows = rows.with_columns(capacity=pl.lit(float(row["effective_capacity_mw"])))
        keys = rows.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        assert keys.height == rows.height, f"run {run} {row['site']}: duplicate valid times"
        out.append((row, keys, prep.feature_matrix(frame=rows, scheme="summary")))
    return out


def truncated_summary_dir(*, label: str, run: datetime) -> Path:
    """Return a folder of symbolic links to the ensemble summary files issued at or before `run`, and to none issued later."""
    folder = core.ROOT / "scratch" / f"truncated_{ARM}_{MONTH}_{label}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for path in sorted(e.prep.SUMMARY_DIR.glob("init=*.parquet")):
        issued = datetime.strptime(path.stem.removeprefix("init="), "%Y-%m-%d").replace(tzinfo=UTC)
        if issued <= run:
            (folder / path.name).symlink_to(path)
    return folder


def run_weather_test(*, n_origins: int = 20) -> None:  # noqa: PLR0915
    """Check the weather inputs of 20 sampled test origins against rebuilds that use no later forecast run.

    For each sampled origin and Primary the test asserts three things. (1) The weather, solar, and
    calendar columns of the stored test rows equal a rebuild by `prep.run_inputs` from a summary
    folder that holds no file issued after the origin's run. `run_inputs` reads only the origin's own
    summary file, so this is a check that the stored rows equal a rebuild from the origin's own
    summary, and removing later files cannot change the rebuild. A run after 2026-06-29 has no
    control-member frame, so `summary_run_inputs` rebuilds its rows from the run's summary file
    alone, which also asserts that the file's origin is the test origin. (2) The thermal-lag, solar-gain,
    and other B1 columns that experiment 437 stores for the origin equal a rebuild from a summary
    that holds no later run (experiment 437's own causality check, run for the sampled origins).
    This rebuild reads the whole summary, so removing later runs does test it. (3) Every
    expected-weather table entry that a test row reads at its day and week lag has a run origin at
    or before the lag time, itself at least 24 hours before the forecast origin (`check_expected`).
    """
    assert fold.PHASE == "predict", (
        "run `weather` with E00677_PHASE=predict (the test rows are needed)"
    )
    import e00437_features as f437  # ty: ignore[unresolved-import]

    prep = e.prep
    clean = c.clean_stack()
    mr = k.build_month(month=MONTH, clean=clean)
    test = data.build_test(month=MONTH, standardiser=mr.base.standardiser)
    keep_cols, _, _ = data.non_lag_layout()
    origin_tab = e.d560.raw_table(source="ft", cache=e.W560 / "expected")[1]
    origins = np.unique(test.origin)
    pick = np.sort(
        np.random.default_rng(0).choice(origins, size=min(n_origins, len(origins)), replace=False)
    )
    delay = e.common.PUBLICATION_DELAY
    local_date, daily = (
        f437.local_date_codes(),
        {"school": f437.school_columns(), "national": f437.national_daily()},
    )
    report = []
    rebuilt_runs: dict[int, list] = {}
    stand_in_origins: set[int] = set()  # runs with no control-member frame (after 2026-06-29)
    for label in LABELS:
        row = next(r for r in prep.primaries().iter_rows(named=True) if r["site"] == label)
        index = int(row["primary"])
        table = f437.load_table(site=label)
        full = prep.load_summary(series_id=int(row["time_series_id"]))
        mine = test.primary == index
        # (3) the expected-weather table entries that the rows read
        message = e.d560.check_expected(
            origin_tab=origin_tab, primary=test.primary[mine], x=test.x[mine],
            origin_us=test.origin[mine], label=f"{MONTH} {label}",
        )  # fmt: skip
        for o in pick:
            origin_dt = datetime.fromtimestamp(int(o) / 1e6, UTC)
            run_dt = origin_dt - delay
            folder = truncated_summary_dir(label=label, run=run_dt)
            assert not any(
                datetime.strptime(q.stem.removeprefix("init="), "%Y-%m-%d").replace(tzinfo=UTC)
                > run_dt
                for q in folder.glob("init=*.parquet")
            ), "a later summary file is still present"
            if int(o) not in rebuilt_runs:  # `run_inputs` rebuilds all 16 Primaries (about 30 s)
                saved = prep.SUMMARY_DIR
                prep.SUMMARY_DIR = folder
                try:
                    if has_member_frame(run=run_dt):
                        rebuilt_runs[int(o)] = prep.run_inputs(
                            origin=origin_dt, scheme="summary", standardiser=IdentityStandardiser()
                        )
                    else:
                        rebuilt_runs[int(o)] = summary_run_inputs(origin=origin_dt)
                        stand_in_origins.add(int(o))
                finally:
                    prep.SUMMARY_DIR = saved
            items = rebuilt_runs[int(o)]
            shutil.rmtree(folder)
            _, keys, raw = next(it for it in items if it[0]["site"] == label)
            rebuilt = mr.base.standardiser.apply(x=raw[:, keep_cols])
            valid = keys["valid_time"].dt.epoch("us").to_numpy()
            sel = np.flatnonzero(mine & (test.origin == o))
            where = {int(v): i for i, v in enumerate(valid)}
            assert all(int(test.valid[i]) in where for i in sel), "a test row has no rebuilt row"
            got = rebuilt[[where[int(test.valid[i])] for i in sel]]
            width = got.shape[1]
            diff = float(np.nanmax(np.abs(got - test.x[sel][:, :width])))
            assert diff <= 1e-6 and np.array_equal(
                np.isnan(got), np.isnan(test.x[sel][:, :width])
            ), (
                f"{label} origin {origin_dt}: the weather inputs change when later summary files are removed ({diff})"
            )
            # (2) experiment 437's columns of this origin, rebuilt from a summary of the runs up to this origin
            truncated = full.filter(pl.col("power_fcst_init_time") <= origin_dt)
            original = prep.load_summary
            prep.load_summary = lambda *, series_id, t=truncated: t
            try:
                again = f437.site_table(row=row, local_date=local_date, daily=daily)
            finally:
                prep.load_summary = original
            key = int(o)
            a = table.filter(pl.col("origin_us") == key).sort("valid_us")
            b = again.filter(pl.col("origin_us") == key).sort("valid_us")
            worst = 0.0
            for col in f437.INPUT_COLUMNS:
                va, vb = a[col].to_numpy(), b[col].to_numpy()
                assert np.array_equal(np.isnan(va), np.isnan(vb)), (
                    f"{label} origin {origin_dt}: {col} has different missing values after the rebuild"
                )
                worst = max(worst, float(np.nanmax(np.abs(va - vb), initial=0.0)))
            assert worst <= 1e-5, (
                f"{label} origin {origin_dt}: a B1 column depends on a later run ({worst})"
            )
            report.append(
                {
                    "label": label,
                    "origin": str(origin_dt),
                    "weather_max_diff": diff,
                    "b1_max_diff": worst,
                    "rows": len(sel),
                    "rebuilt_from": "summary file only"
                    if int(o) in stand_in_origins
                    else "control-member frame and summary file",
                }
            )
        print("WEATHER OK", label, message, flush=True)
    out = core.ROOT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"weather_{ARM}_{MONTH}.json").write_text(json.dumps(report, indent=1))
    print(
        f"WEATHER TEST OK: {len(pick)} origins x {len(LABELS)} Primaries, {ARM} world", flush=True
    )


def one_site_synthetic(*, clean, first, last, seed):
    """Return this Primary's shifted row for a span, as the pooled draw gives it (the schedule depends on the span, so each call builds its trick)."""
    trick = world.schedule_trick(
        original=ORIGINAL_SYNTHETIC, index=GLOBAL_INDEX, n_primaries=core.N_PRIMARIES,
        first=first, last=last,
    )  # fmt: skip
    return trick(clean=clean, first=first, last=last, seed=seed)


def check_one_site_world() -> None:
    """Assert that this process's cleaned row and shifted rows equal the Primary's rows in the pooled world (hashes from `split3 spans`)."""
    clean = c.clean_stack()
    assert clean.shape[0] == 1, clean.shape
    if fold.PHASE == "train":
        assert not np.isfinite(clean[:, FOLD_CUT_IDX:]).any(), "the train phase sees held-out power"
    mine = shifted_row_hashes(
        clean=clean, spans=POOLED_SPANS, synthetic=one_site_synthetic, labels=[MINE.primaries]
    )  # fmt: skip
    kinds = PHASE_KINDS[fold.PHASE]
    assert set(kinds) <= set(POOLED_SPANS["shift_hash"]), (
        "fold_spans holds no before-cutoff hashes: re-run `split3 spans` for this month"
    )
    for kind in kinds:
        by_label = mine[kind]
        stored = POOLED_SPANS["shift_hash"][kind][MINE.primaries]
        assert by_label[MINE.primaries] == stored, (
            f"the one-site {kind} row differs from the pooled world's row of {MINE.primaries}"
        )
    print(
        f"ISOLATION OK {MINE.primaries} {MONTH} ({fold.PHASE} phase): {', '.join(kinds)} equal the pooled world's",
        flush=True,
    )


def extra_column_hashes(*, mr, test, labels: list[str]) -> dict[str, dict[str, str]]:
    """Hash, per Primary, the unstandardised extra-column blocks that the scored rows read.

    The blocks are experiment 437's B1 columns (`FoldColumns._raw_real`), the drift and knot columns
    (`WT.rows`), the W-T columns (`WTC.rows`, only where `SPEC.wt`; P5 has none), and the
    Primary's row of the expected-weather table with its run origins. In a pooled world the Primary is at its own index among the 16; in the one-site
    world it is index 0.
    """
    fc = e.c562.FoldColumns(fit_until_idx=e.fit_until(month=MONTH))
    cols, origin = e.d560.raw_table(source="ft", cache=e.W560 / "expected")
    out = {}
    for label in labels:
        index = ALL.index(label) if POOLED else 0
        sel = test.primary == index
        keys = {
            "primary": test.primary[sel],
            "origin_idx": test.origin_idx[sel],
            "valid_idx": test.target_idx[sel],
        }
        blocks = {
            "b1": fc._raw_real(arm="B1", **keys),
            "drift": e.WT.rows(kind="ens", null=False, **keys),
            **({"wt": e.WTC.rows(kind="ens", null=False, **keys)} if e.SPEC.wt else {}),
            "expected_cols": np.asarray(cols[index]),
            "expected_origin": np.asarray(origin[index]),
        }
        out[label] = {
            name: core.array_hash(arrays={"x": np.asarray(block)}) for name, block in blocks.items()
        }
    return out


def check_extra_blocks() -> None:
    """Assert that this Primary's extra-column blocks equal the pooled world's (hashes from `split3 spans`)."""
    assert fold.PHASE == "predict", (
        "run `isolation` with E00677_PHASE=predict (the scored rows are needed)"
    )
    clean = c.clean_stack()
    mr = k.build_month(month=MONTH, clean=clean)
    test = data.build_test(month=MONTH, standardiser=mr.base.standardiser)
    mine = extra_column_hashes(mr=mr, test=test, labels=[MINE.primaries])[MINE.primaries]
    stored = POOLED_SPANS["extra_hash"][MINE.primaries]
    bad = [name for name in stored if mine[name] != stored[name]]
    assert not bad, f"{MINE.primaries}: extra-column blocks differ from the pooled world's: {bad}"
    print(
        f"ISOLATION OK {MINE.primaries} {MONTH}: extra-column blocks {sorted(stored)} equal the pooled world's",
        flush=True,
    )


def write_opened_log() -> None:
    """Write the audit log of every path the guarded readers opened (split-at-1 only)."""
    if POOLED:
        return
    log = OUT / MINE.primaries / "work" / "logs"
    log.mkdir(parents=True, exist_ok=True)
    (log / f"opened_files_{CMD}.json").write_text(json.dumps(sorted(set(OPENED)), indent=1))
    print("files opened through the guarded readers:", len(set(OPENED)), flush=True)


device = "cuda"
if not POOLED and CMD in {"fit", "bank", "isolation"}:
    check_one_site_world()
if CMD == "isolation":
    if fold.PHASE == "predict":
        check_extra_blocks()
    else:
        print(
            "isolation (train phase): the extra-column blocks need the scored rows; run it in the predict phase too",
            flush=True,
        )
    write_opened_log()
elif CMD == "spans":
    run_spans()
elif CMD == "weather":
    run_weather_test()
elif CMD == "bank":
    run_bank(device=device)
    write_opened_log()
else:
    for label in LABELS:
        use_primary(label=label)
        if CMD == "pretrain":
            e.pretrain_wt(device=device)
            rec = np.load(e.stats_file(month=MONTH))
            b1 = {"mean": rec["mean"], "std": rec["std"]}
            path = e.W / "stats" / "b1_record.npz"
            core.StatsRecord(
                month=MONTH,
                primary=-1 if STATE["index"] is None else int(STATE["index"]),
                arrays=b1,
            ).save(path=path)
            if GLOBAL:
                stored = np.load(STATS_677 / "stats" / e.SPEC.b1 / f"{MONTH}.npz")
                assert np.array_equal(rec["mean"], stored["mean"]) and np.array_equal(
                    rec["std"], stored["std"]
                ), "the global arm's B1 statistics differ from experiment 677's"
        elif CMD == "step2":
            e.step2(months=[MONTH], seeds=MINE.seeds, device=device)
        else:
            if not POOLED:
                rec_path = e.W / "stats" / "b1_record.npz"
                rec = np.load(e.stats_file(month=MONTH))
                core.StatsRecord.check(
                    path=rec_path, arrays={"mean": rec["mean"], "std": rec["std"]}
                )
            e.fit_arm(months=[MONTH], seeds=MINE.seeds, stress=False, limit=None, device=device)
            (e.W / "logs").mkdir(parents=True, exist_ok=True)
            (e.W / "logs" / "checks_fit.json").write_text(
                json.dumps(e.CHECK, indent=1, default=str)
            )
        torch.cuda.empty_cache()
    finish()
    write_opened_log()
    print(
        "DONE", ARM, CMD, MONTH, LABELS, "fixed-epoch phases:",
        [(p["label"], p["step"], p["seed"], p["fixed_epochs"]) for p in STATE["phases"] if p["fixed_epochs"]],
        datetime.now(UTC).isoformat(), flush=True,
    )  # fmt: skip
