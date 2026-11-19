# ruff: noqa: ANN001, ANN003, ANN201, ANN202, D102, D103, D107, E402, E501, I001, PLR0912, PLR0915, PT018
"""Experiment 709, phase B: the Candidate's stage 3 with an era signal (copy of 667's `e00667_cand.py`).

Experiment 667's script fine-tunes and scores the Candidate (arm `P7+D50`) one backtest fold at a
time. This copy runs stage 3 only. It reads 667's stage 1 and stage 2 weights (seeds 0 and 1), the
tables `T_<month>_s<seed>.npz`, and the extended inputs from 667's own data folder and never writes
there; everything it writes goes under `features.DATA_DIR` (experiment 709's folder).

Commands (positional): `fit` (stage 3 fit, then the forecasts of the fold's month in the predict
phase) and `bank` (inference on the injected bank, predict phase). Flags: `--era` (the arm),
`--fold` (the month), `--seed` (0 or 1), `--phase` (`train` hides the scored month, `predict`
opens it), `--pilot` (one fold, one seed, one arm, a small slice, for timing).

Arms: `none` is 667's Candidate. `aug` adds the curriculum with no era input. The other arms add the
era block (see `e00709_signal`) through `e00709_model.EraPermLagMdn`, the curriculum, the rows held
out of stage 3, and the lag translation. The curriculum gives half of the training passes (the odd
epochs) a second set of tensors in which persistent steps and switching events from the phase A
injector (CURRICULUM seeds) are added to the panel.
"""

import argparse
import dataclasses
import json
import os
import pickle
import resource
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

_p = argparse.ArgumentParser()
_p.add_argument("command", choices=("fit", "bank"))
_p.add_argument(
    "--era",
    required=True,
    choices=(
        "none",
        "aug",
        "oracle",
        "placebo",
        "oracle-enc",
        "all",
        "rule",
        "soft",
        "hmm",
        "stat",
    ),
)
_p.add_argument(
    "--fold",
    required=True,
    choices=(
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    ),
)
# experiment 667 has stage 1 and stage 2 weights for seeds 0 and 1 only; seeds 2 and above need new chains
_p.add_argument("--seed", type=int, required=True, choices=(0, 1))
_p.add_argument("--phase", choices=("train", "predict"), default="train")
_p.add_argument("--pilot", action="store_true")
_p.add_argument("--bank-passes", type=int, default=16)
_p.add_argument(
    "--rule", choices=("es", "b2"), default="es",
    help="stage 3 rule: es = phase B's odd-epoch alternation, b2 = each row reads the injected world with probability 0.5 in every epoch",
)  # fmt: skip
MINE = _p.parse_args()
# the fold module reads these at import, so set them before any experiment module is imported
os.environ["E00667_MONTH"] = MINE.fold
os.environ["E00667_PHASE"] = MINE.phase
ERA, CMD, SEED, MONTH = MINE.era, MINE.command, MINE.seed, MINE.fold
sys.argv = ["e00574_fit.py", "fit", "--arm", "P5", "--months", MONTH, "--seeds", str(SEED)]

import e00574_fit as e  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00613_nbmodel as mymodel  # ty: ignore[unresolved-import]
import e00613_nbsignals as sig  # ty: ignore[unresolved-import]
import e00709_classify as classify
import e00709_features as features
import e00709_inject as inject
import e00709_model as me
import e00709_signal as sg
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00667_fold as fold  # ty: ignore[unresolved-import]

assert fold.FINAL_MONTH == MONTH

SRC667 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
)
"""Experiment 667's data folder: chains (stage 1 and 2 weights), tables, and inputs; read only."""
ROOT = features.DATA_DIR
"""Everything this experiment writes."""
assert ROOT != SRC667
CHAIN = SRC667 / "chains" / MONTH / "work"
TABLES = SRC667 / "tables"
B2: Final = MINE.rule == "b2"
"""Phase B2: the injected world is mixed into every batch. Its fits and predictions carry a `b2` prefix."""
assert not B2 or ERA in ("aug", "oracle", "placebo", "stat"), "B2 runs four arms; A0 is phase B's"
PREFIX = "b2_" if B2 else ""
DIR = f"{PREFIX}{ERA.replace('-', '_')}_{MONTH}_s{SEED}" + ("_pilot" if MINE.pilot else "")
ROW_SET_ID: Final = "experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06"
"""Contract rows at 12:00 UTC on each day of a fold's month, 14 days of half-hours, 16 series, with
`valid_time` before 2026-07-01 (the rows after that date are the held-out finalist rows)."""
HOLDOUT_START: Final = datetime(2026, 7, 1, tzinfo=UTC)
MAX_SEEDS: Final = 10
"""World indices leave room for this many seeds per month, so no seed's curriculum collides with another's."""
WORLDS_DIR: Final = ROOT / "worlds"
"""Cached injected worlds (signals of every arm), by curriculum (month, seed) and bank (month, pass)."""
PILOT_DAYS: Final = 3
PILOT_EPOCHS: Final = 4
PILOT_ROW_FRACTION: Final = 0.1

USE_ERA = ERA not in ("none", "aug")
CURRICULUM = ERA != "none"
MIX_P: Final = 0.5
"""Phase B2: the share of rows in a batch that read the injected world."""
SIGNAL_NAME = "oracle" if ERA == "oracle-enc" else ERA
SOFT = ERA == "soft"

GROUPS = ["gap", "own:ew1,ew3,ew7", "exp", "nb:gap,ew1,ew3,ew7"]
"""Arm P7's stage 1 inputs; D50 below is the signal dropout."""
COLUMNS = sig.columns_of(groups=tuple(GROUPS))
N_EXTRA = len(COLUMNS)
DROP_P = 0.5
INPUTS = mymodel.InputSet(n_extra=N_EXTRA)
W574 = e.W
e.WORK = ROOT / "fits" / DIR
e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen
LAG_FIRST: Final = 37
LAG_WIDTH: Final = 8
CLIP: Final = float(e.prep.INPUT_CLIP_SD)
"""Standardised lag inputs are clipped to this many standard deviations, as experiment 397 clips them."""
LEAD_OF: Final = {"day": 43, "week": 44}
GRID, STEP = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
DAY_US = 86_400_000_000
PER_DAY = features.HALF_HOURS_PER_DAY
CUT_DAY = (fold.FINAL_START - features.GRID_START).days
"""Day index of the fold's cut on the grid of experiment 709."""
assert int(features.GRID_START.timestamp() * 1_000_000) == GRID, "grids differ"


def _step2_file(*, month: str, seed: int):
    p = CHAIN / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def _stats_file(*, month: str):
    p = CHAIN / "stats" / e.SPEC.b1 / f"{fold.FINAL_MONTH}.npz"
    assert p.exists(), p
    return p


e.step2_file, e.stats_file = _step2_file, _stats_file
fold.install(e, months=[MONTH], reference_month=fold.PREVIOUS_MONTH)
# experiment 667's install extends the grid to its own end (2026-11-20 23:30), so the grid is read after it
_grid = e.common.grid_times()
import e00667_extend as _ext667  # ty: ignore[unresolved-import]

GRID_END_TIME: Final = _ext667.GRID_END
assert _grid[0] == features.GRID_START and _grid[-1] == GRID_END_TIME, (_grid[0], _grid[-1])
assert len(_grid) == int((GRID_END_TIME - features.GRID_START) / timedelta(minutes=30)) + 1
GRID_SHAPE: Final = (16, len(_grid))
"""The 667 clean stack covers 2019-09-01 to 2026-11-20 23:30 (126,624 half-hours). Experiment 709's features,
signal tables, and injected panels end on 2026-06-30 23:30 (119,328 half-hours) on the same start, so a day or
half-hour index means the same instant in all of them."""

# ---------------------------------------------------------------- the era signal, rows held out, weights

E3_EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
e.EMBARGO_DAYS = E3_EMBARGO_DAYS
_select_rows = e.c.select_rows

SIGNAL: sg.Signal | None = sg.read_signal(arm=SIGNAL_NAME) if USE_ERA else None
HOLD: np.ndarray | None = (
    sg.hold_mask(signal=SIGNAL, cut_day=CUT_DAY) if SIGNAL is not None else None
)
SLOTS = (
    sg.era_slots(signal=SIGNAL, cut_day=CUT_DAY) if (SIGNAL is not None and ERA == "stat") else None
)
REPORT: dict = {}


def _select_rows_era(**kwargs):
    """Return experiment 510's rows with the E3 weights, minus the rows held out for the era signal.

    Rows whose valid-time day lies in the unknown period or inside a switching event are removed
    from the training and validation rows, and rows from before the current era keep weight 0.1.
    """
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
    if SIGNAL is None or HOLD is None:
        return e.c.Refit(refit.index, refit.y, weight, refit.is_val, report)
    site = rows.primary[refit.index]
    valid_day = ((rows.valid_us[refit.index] - GRID) // DAY_US).astype(np.int64)
    keep = ~HOLD[site, np.clip(valid_day, 0, HOLD.shape[1] - 1)]
    factor = sg.row_weight_factor(signal=SIGNAL, cut_day=CUT_DAY, site=site, valid_day=valid_day)
    weight = (weight * factor).astype(np.float32)
    report |= {
        "held_out_rows": int((~keep).sum()),
        "earlier_era_rows": int((factor < 1).sum()),
        "rows_before_hold": len(keep),
    }
    REPORT.update(report)
    return e.c.Refit(refit.index[keep], refit.y[keep], weight[keep], refit.is_val[keep], report)


e.c.select_rows = _select_rows_era


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    if USE_ERA:
        model = me.new_era_model(
            n_primaries=n_primaries, n_knots=e.KNOTS, n_b1=e.N_B1, n_signal=N_EXTRA, arm=ERA
        )
    else:
        model = mymodel.PermLagMdn(
            n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, inputs=INPUTS
        )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


# ---------------------------------------------------------------- neighbours, synthetic shifts (667's)


def world() -> s598.Neighbours:
    import e00400_blend as b400  # ty: ignore[unresolved-import]

    sm = k578.sites()
    labels = sm["label"].to_list()
    lab2p = {lab: i for i, lab in enumerate(labels)}
    of = {i: [] for i in range(len(labels))}
    for a, v in b400.nged_neighbours().items():
        if a in lab2p:
            of[lab2p[a]] = [lab2p[x] for x in v if x in lab2p]
    return s598.Neighbours(of=of, capacity=sm["capacity_mw"].to_numpy().astype(np.float64))


NB = world()
SYN: dict[int, tuple[np.ndarray, np.ndarray]] = {}
"""Per month-rows object: the synthetic power (16, grid) and the shift added to it (0 where none)."""
_build_month_p5 = e.k.build_month
import e00581_bank as bank  # ty: ignore[unresolved-import]


def origin_idx_of(*, origin_us: np.ndarray) -> np.ndarray:
    return ((origin_us - GRID) // STEP).astype(np.int64)


def build_month_syn(*, month, clean):
    """Build the month's P5 rows with experiment 591's synthetic lag-only load-shifts in the lag columns."""
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = origin_idx_of(origin_us=mr.origin_us)
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    fixed, n_events = bank.synthetic_clean(
        clean=clean, first=first, last=last, seed=int(month.replace("-", ""))
    )
    blk = e.block_of(
        clean=fixed, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
    )
    assert blk.shape[1] == LAG_WIDTH, blk.shape
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
    shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0).astype(np.float32)
    SYN[id(mr)] = (fixed, shift)
    print(f"{month}: {n_events} synthetic events", flush=True)
    return mr


# ---------------------------------------------------------------- the table T (read only) and the row blocks


def table_file(*, month: str, seed: int):
    if fold.PHASE == "predict":
        return TABLES / "predict" / f"T_{month}_s{seed}.npz"
    return TABLES / f"T_{month}_s{seed}.npz"


def require_tables(*, month: str, seed: int) -> None:
    """Refuse to build a table: experiment 667's tables are read only, and this experiment has none."""
    p = table_file(month=month, seed=seed)
    if not p.exists():
        msg = f"{p} is missing: tables come from experiment 667 and are never built here"
        raise FileNotFoundError(msg)


class FoldSignals:
    """The signal blocks of one fold month and seed (training rows with synthetic shifts, and test power)."""

    def __init__(self, *, mr, test, clean: np.ndarray, month: str, seed: int) -> None:
        z = np.load(table_file(month=month, seed=seed))
        assert int(z["n_train"]) == len(mr.y) and (
            test is None or int(z["n_test"]) == len(test.x)
        ), "stale table"
        self.mr, self.test, self.clean, self.month, self.seed = mr, test, clean, month, seed
        self.table = s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
        self.fixed, self.shift = SYN[id(mr)]
        self.choice = s598.mirror_choice(neighbours=NB.of, month=month)
        self.power_sig = s598.mirrored_power(
            base=self.fixed, shift=self.shift, choice=self.choice, capacity=NB.capacity
        )
        self.weather_origin = np.asarray(e.CTX["origin_ft"])
        self._train: np.ndarray | None = None
        self.plan: sig.NoisePlan | None = None
        self.report: dict = {}

    def _block(self, *, rows: dict, power_lag, power_sig) -> np.ndarray:
        lag_idx = {k: rows["x"][:, col].astype(np.int64) for k, col in LEAD_OF.items()}
        return sig.signal_columns(
            prim=rows["prim"], origin_idx=rows["origin"], day_idx=lag_idx["day"],
            week_idx=lag_idx["week"], day_missing=rows["x"][:, 38] > 0.5,
            week_missing=rows["x"][:, 41] > 0.5, power_lag=power_lag, power_sig=power_sig,
            table=self.table, nb=NB, weather_origin=self.weather_origin, columns=COLUMNS,
        )  # fmt: skip

    def train_rows(self) -> dict:
        return {
            "x": self.mr.x[:, :45], "prim": self.mr.primary,
            "origin": origin_idx_of(origin_us=self.mr.origin_us),
        }  # fmt: skip

    def test_rows(self, *, test=None) -> dict:
        t = test if test is not None else self.test
        return {"x": t.x[:, :45], "prim": t.primary, "origin": t.origin_idx}

    def train_block(self) -> np.ndarray:
        """Return the training rows' block (synthetic shifts, plus training noise), cached."""
        if self._train is not None:
            return self._train
        synth = self._block(rows=self.train_rows(), power_lag=self.fixed, power_sig=self.power_sig)
        real_test = self._block(rows=self.test_rows(), power_lag=self.clean, power_sig=self.clean)
        names = [sig.ALL_NAMES[c] for c in COLUMNS]
        self.plan = sig.noise_plan(
            train=synth, train_prim=self.mr.primary, test=real_test, test_prim=self.test.primary,
            n_primaries=16, names=names,
        )  # fmt: skip
        self.report = {
            "noised_columns": [
                names[c] for j, c in enumerate(self.plan.columns) if self.plan.sd[j].any()
            ]
        }
        sig.add_training_noise(block=synth, prim=self.mr.primary, plan=self.plan, seed=self.seed)
        assert synth.shape[0] == len(self.mr.x)
        self._train = synth
        return synth

    def test_block(self, *, test, power_lag, power_sig) -> np.ndarray:
        return self._block(rows=self.test_rows(test=test), power_lag=power_lag, power_sig=power_sig)


FOLD: dict[tuple[str, int], FoldSignals] = {}


def fold_signals(*, mr, test, clean, month: str, seed: int) -> FoldSignals:
    key = (month, seed)
    if key not in FOLD:
        FOLD.clear()
        FOLD[key] = FoldSignals(mr=mr, test=test, clean=clean, month=month, seed=seed)
    return FOLD[key]


# ---------------------------------------------------------------- era columns and lag translation for rows


def tau_of_days(*, n_days: int) -> np.ndarray:
    """Return the knot position of the start of each day of the signal's grid."""
    return e.extra.DriftColumns.tau(
        valid_us=GRID + np.arange(n_days, dtype=np.int64) * DAY_US
    ).astype(np.float64)  # ty: ignore[unresolved-attribute]


def era_columns(*, signal: sg.Signal, mr, prim, origin_idx, cut, slots=None) -> np.ndarray:
    """Return the era block of rows (the single as-of function for training and forecast rows)."""
    return sg.era_block(
        signal=signal, site=prim, origin_day=origin_idx // PER_DAY, cut_day=cut,
        target_std=np.asarray(mr.base.target_std, dtype=np.float64),
        tau_of_day=tau_of_days(n_days=signal.n_days), soft=SOFT, slots=slots,
    )  # fmt: skip


def translate_rows(
    *,
    signal: sg.Signal,
    fs: FoldSignals,
    rows: dict,
    target_idx,
    power_lag,
    power_sig,
    block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate the lags of rows inside an era's first days or inside a switching event.

    The own-site power is changed in the lag and signal panels (band changes added before an era's
    onset; the estimated depth subtracted inside a switching event; both for a soft-arm row that
    needs both) once per group of rows that share a translation, and the lag values and the signal
    block are rebuilt from the changed panels.

    Args:
        signal: The signal that the rows read.
        fs: The fold's signal builder.
        rows: `prim`, `origin`, and `x` (45 columns) of the rows.
        target_idx: Target index of each row on the grid.
        power_lag: The lag power panel.
        power_sig: The signal power panel.
        block: The signal block of the rows, changed in place.

    Returns:
        The translated rows' indices and their new (day, week) lag values (standardised).
    """
    tr = sg.translation_rows(
        signal=signal, site=rows["prim"], origin_day=rows["origin"] // PER_DAY, soft=SOFT
    )
    if len(tr.row) == 0:
        return tr.row, np.empty((0, 2), np.float32)
    values = np.empty((len(tr.row), 2), np.float32)
    n = signal.n_days * PER_DAY
    for grp in np.unique(tr.group):
        pick = np.flatnonzero(tr.group == grp)
        idx = tr.row[pick]
        site = int(rows["prim"][idx[0]])
        first = pick[0]
        onset, era_scale, bands = int(tr.onset[first]), float(tr.era_scale[first]), tr.bands[first]
        switch_scale, depth = float(tr.switch_scale[first]), float(tr.depth[first])

        def translate(
            power,
            *,
            onset=onset,
            era_scale=era_scale,
            bands=bands,
            switch_scale=switch_scale,
            depth=depth,
        ):
            return sg.translate_panel_row(
                power=power, onset_day=onset, era_scale=era_scale, bands=bands,
                switch_scale=switch_scale, depth=depth,
            )  # fmt: skip

        lag_v, sig_v = power_lag.copy(), power_sig.copy()
        lag_v[site, :n] = translate(power_lag[site, :n])
        sig_v[site, :n] = translate(power_sig[site, :n])
        sub = {"x": rows["x"][idx], "prim": rows["prim"][idx], "origin": rows["origin"][idx]}
        blk = e.block_of(
            clean=lag_v, primary=sub["prim"], target_idx=target_idx[idx], origin_idx=sub["origin"]
        )
        values[pick, 0], values[pick, 1] = blk[:, 0], blk[:, 3]
        block[idx] = fs._block(rows=sub, power_lag=lag_v, power_sig=sig_v)
    return tr.row, values


# ---------------------------------------------------------------- the curriculum's injected world


def cached(*, path: Path, build):
    """Return the pickled object at `path`, building and writing it (atomically) when absent."""
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    obj = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")  # unique, so parallel arms never share a file
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.rename(path)
    return obj


@dataclasses.dataclass
class AltWorld:
    """The curriculum's world: the real panel plus persistent steps and switching events."""

    lag_power: np.ndarray
    sig_power: np.ndarray
    era_delta: np.ndarray
    signal: sg.Signal | None
    cut_site: np.ndarray
    n_passes: int
    n_eras: int
    slots: dict | None = None
    hold: np.ndarray | None = None


CURRICULUM_PASSES: Final = 4
CURRICULUM_TRIES: Final = 16
CURRICULUM_ERAS: Final = 8
ONSET_FIRST: Final = 120
ONSET_STEP: Final = 20
ONSET_MARGIN: Final = 14
"""An injected event's rows take the cut `onset - ONSET_MARGIN`, so the onset lies after the cut."""
MONTH_INDEX: Final = (fold.FINAL_START.year - 2025) * 12 + fold.FINAL_START.month - 11
CONTEXT: dict = {}


def stage0_context():
    """Return phase A's context (real panel, features, weather) and the frozen models, built once."""
    import e00709_stage0 as stage0

    if not CONTEXT:
        CONTEXT["ctx"] = stage0.build_context()
        CONTEXT["models"] = classify.load_models(folder=stage0.MODELS_DIR)
        known = pl.read_parquet(features.DATA_DIR / "known_events.parquet")
        CONTEXT["real_events"] = sg.real_oracle_events(
            feats=CONTEXT["ctx"].feats, era_table=CONTEXT["ctx"].era_table, known_events=known
        )
    return CONTEXT["ctx"], CONTEXT["models"], stage0


def world_parts(
    *, passes: list, placebo_seed: int, placebo_window: tuple[int, int], with_real: bool = True
) -> dict:
    """Return the cacheable parts of a world: every arm's signal, the change to the panel, the touched Primaries.

    `with_real` False leaves the real record out of the oracle and the placebo. The bank needs that,
    because it overlays the world only at the touched Primaries, and a placebo that had to place the
    whole real record in the 30-day bank window would fail the separation rule.
    """
    ctx, models, stage0 = stage0_context()
    n_hh = ctx.panel.power.shape[1]
    delta = np.zeros((len(ctx.panel.sites), n_hh), np.float32)
    era_delta = np.zeros_like(delta)
    onset_of: dict[int, int] = {}
    for inj in passes:
        both = np.isfinite(inj.power) & np.isfinite(ctx.panel.power)
        change = np.where(both, inj.power - ctx.panel.power, 0.0).astype(np.float32)
        delta += change
        hosts = [ev.host for ev in inj.eras]
        era_delta[hosts] += change[hosts]
        for ev in inj.eras:
            onset_of[ev.host] = min(onset_of.get(ev.host, 10**9), ev.onset_hh // PER_DAY)
        for sw in inj.switches:
            for site in (sw.donor, sw.recipient):
                if site is not None:
                    onset_of[site] = min(onset_of.get(site, 10**9), sw.onset_hh // PER_DAY)
    results_of = [
        stage0.process_hosts(ctx=ctx, models=models, injected=inj, arms=sg.LEARNED_ARMS)
        for inj in passes
    ]
    pass_feats = [sg.pass_features(ctx_feats=ctx.feats, injected=inj) for inj in passes]
    real_eras, real_switches = CONTEXT["real_events"] if with_real else (None, None)
    world_sigs = sg.world_signals(
        injected=passes, ctx_feats=ctx.feats, panel=ctx.panel, results_of=results_of,
        pass_feats=pass_feats, n_days=ctx.panel.n_days, placebo_seed=placebo_seed,
        placebo_window=placebo_window, real_eras=real_eras, real_switches=real_switches,
    )  # fmt: skip
    return {
        "world_sigs": world_sigs, "delta": delta, "era_delta": era_delta, "onset_of": onset_of,
        "n_passes": len(passes), "n_eras": sum(len(inj.eras) for inj in passes),
    }  # fmt: skip


def build_curriculum_world() -> dict:
    """Build the curriculum's world from phase A's injector (CURRICULUM seeds), once per month and seed.

    Up to `CURRICULUM_PASSES` passes with disjoint touched Primaries are merged. The passes' onsets
    lie 120 to 60 days before the cut. When the accepted passes would hold fewer than
    `CURRICULUM_ERAS` eras, the later passes take more eras (within the host-separation rule).
    """
    ctx, _models, _stage0 = stage0_context()
    base_idx = (MONTH_INDEX * MAX_SEEDS + SEED) * CURRICULUM_TRIES
    passes, taken, n_eras = [], set(), 0
    for t in range(CURRICULUM_TRIES):
        if len(passes) == CURRICULUM_PASSES:
            break
        left = CURRICULUM_PASSES - len(passes)
        want = max(inject.N_ERA_PER_PASS, -(-(CURRICULUM_ERAS - n_eras) // left))
        injected = inject.build_pass(
            panel=ctx.panel, weather_shortwave=ctx.weather.shortwave, ready_days=ctx.ready_days,
            verified_days=ctx.verified_days, world="curriculum", pass_index=base_idx + t,
            onset_day=CUT_DAY - ONSET_FIRST + ONSET_STEP * len(passes), n_era=want,
        )  # fmt: skip
        touched = {ev.host for ev in injected.eras} | {s.donor for s in injected.switches}
        touched |= {s.recipient for s in injected.switches if s.recipient is not None}
        if touched & taken:
            continue
        taken |= touched
        n_eras += len(injected.eras)
        passes.append(injected)
    parts = world_parts(
        passes=passes,
        placebo_seed=sg.PLACEBO_SEED_BASE["curriculum"] + base_idx,
        placebo_window=(int(ctx.ready_days.min()) + 60, CUT_DAY - 40),
    )
    parts["touched"] = sorted(taken)
    return parts


def build_alt_world(*, fs: FoldSignals, clean: np.ndarray) -> AltWorld:
    """Return the curriculum's world for this arm, from the cached parts of the month and seed."""
    parts = cached(
        path=WORLDS_DIR / f"curriculum_{MONTH}_s{SEED}.pkl", build=build_curriculum_world
    )
    n_hh = parts["delta"].shape[1]
    lag_delta = np.zeros_like(clean)
    lag_delta[:, :n_hh] = parts["delta"]
    era_delta = np.zeros_like(clean)
    era_delta[:, :n_hh] = parts["era_delta"]
    signal = None
    if SIGNAL is not None:
        world = parts["world_sigs"][SIGNAL_NAME]
        signal = overlay(real=SIGNAL, world=world, sites=parts["touched"])
    cut_site = np.full(len(clean), CUT_DAY)
    for site, onset in parts["onset_of"].items():
        cut_site[site] = onset - ONSET_MARGIN
    REPORT["alt_passes"], REPORT["alt_eras"] = parts["n_passes"], parts["n_eras"]
    if parts["n_eras"] < CURRICULUM_ERAS:
        print(f"warning: the curriculum world holds {parts['n_eras']} eras", flush=True)
    # real era ids keep their slots; the touched Primaries' ids first seen after the cut share slot 0
    touched = set(parts["touched"])
    slots = None if SLOTS is None else {k: v for k, v in SLOTS.items() if k[0] not in touched}
    hold = None if signal is None else sg.hold_mask(signal=signal, cut_day=CUT_DAY)
    return AltWorld(
        lag_power=fs.fixed + lag_delta, sig_power=fs.power_sig + lag_delta, era_delta=era_delta,
        signal=signal, cut_site=cut_site, n_passes=parts["n_passes"], n_eras=parts["n_eras"],
        slots=slots, hold=hold,
    )  # fmt: skip


def overlay(*, real: sg.Signal, world: sg.Signal, sites: list[int]) -> sg.Signal:
    """Return the real signal with the injected world's columns at `sites` (learned arms), or the world alone (oracle arms).

    The oracle's and the placebo's world already holds the real record's events.
    """
    if not real.causal:
        return world
    cols = {k: v.copy() for k, v in real.columns.items()}
    replaced = [s for s in sites if (world.columns["state"][s] != sg.STATE_NONE).any()]
    for s in replaced:
        for name, values in cols.items():
            values[s] = world.columns[name][s]
    episodes = pl.concat([real.episodes.filter(~pl.col("site").is_in(replaced)), world.episodes])
    return sg.Signal(arm=real.arm, causal=True, columns=cols, episodes=episodes)


ALT: dict = {}
"""The curriculum's tensors of the current fine-tune: `lag` (rows, 2) for columns 37 and 40, `block`,
`era`, and `z`. Under rule `es` the odd epochs read these in place of the plain rows' columns; under rule `b2` each row reads them with probability `MIX_P` in every epoch."""


def train_curriculum(
    *, model: torch.nn.Module, x, z, w, p, is_val, seed: int, lr: float, max_epochs: int,
    fresh_lr: float | None, frozen: bool, anchor: dict | None, lam: float, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Experiment 574's `train` with the curriculum's tensors in `ALT`: read by the odd epochs (rule `es`) or by half of the rows in every epoch (rule `b2`).

    Validation always reads the plain rows. The O-enc layers sit outside `model.weather` and
    `model.datetime`, so freezing the encoders leaves them trainable. The curriculum replaces only
    columns 37 and 40, the signal block, and the era block of a batch, so it never holds a second
    copy of the whole matrix.
    """
    if MINE.pilot:
        max_epochs = min(max_epochs, PILOT_EPOCHS)
    assert ALT or not CURRICULUM, "a curriculum arm reached stage 3 with no injected world"
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > e.c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: e.c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    encoder_ids = {id(q) for part in (model.weather, model.datetime) for q in part.parameters()}  # ty: ignore[unresolved-attribute]
    if frozen:
        for part in (model.weather, model.datetime):
            for q in part.parameters():  # ty: ignore[unresolved-attribute]
                q.requires_grad_(False)
    normal = [q for q in model.parameters() if not (frozen and id(q) in encoder_ids)]
    optimiser = torch.optim.AdamW([{"params": normal, "lr": lr}], weight_decay=e.c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    fraction = e.c.ROW_FRACTION * (PILOT_ROW_FRACTION if MINE.pilot else 1.0)
    era_width = sg.ERA_WIDTH if USE_ERA else 0
    base = x.shape[1] - N_EXTRA - era_width
    for epoch in range(max_epochs):
        model.train()
        if frozen:
            model.weather.eval()  # ty: ignore[unresolved-attribute]
            model.datetime.eval()  # ty: ignore[unresolved-attribute]
        use_alt = bool(ALT) and (B2 or epoch % 2 == 1)
        coin = None
        if B2:
            # each row reads the injected world with probability 0.5; the coin never touches the row order
            coin_gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + epoch)
            coin = (torch.rand(len(x), generator=coin_gen) < MIX_P).to(device)
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(fraction * len(order))]
        for start in range(0, len(order), e.c.BATCH):
            b = order[start : start + e.c.BATCH]
            xb, zb = x[b], z[b]
            wb = w[b]
            if use_alt:
                m = (
                    coin[b]
                    if coin is not None
                    else torch.ones(len(b), dtype=torch.bool, device=device)
                )
                mc = m[:, None]
                xb[:, 37] = torch.where(m, ALT["lag"][b, 0], xb[:, 37])
                xb[:, 40] = torch.where(m, ALT["lag"][b, 1], xb[:, 40])
                xb[:, base : base + N_EXTRA] = torch.where(
                    mc, ALT["block"][b], xb[:, base : base + N_EXTRA]
                )
                if ALT["era"] is not None:
                    xb[:, base + N_EXTRA :] = torch.where(
                        mc, ALT["era"][b], xb[:, base + N_EXTRA :]
                    )
                zb = torch.where(m, ALT["z"][b], zb)
                wb = torch.where(m, w[b] * ALT["w"][b], w[b])
            if float(wb.sum()) <= 0:
                continue
            nll = e.row_nll(out=model(xb, p[b]), z=zb, p=p[b])
            loss = (wb * nll).sum() / wb.sum()
            if hasattr(model, "penalty"):
                loss = loss + model.penalty()  # ty: ignore[call-non-callable]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), e.c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        e.row_nll(out=model(x[i], p[i]), z=z[i], p=p[i])
                        for i in val_idx.split(e.c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        REPORT.setdefault("val_curve", []).append(round(val, 6))
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= e.c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(True)
    ALT.clear()
    return epochs, best


if CURRICULUM or MINE.pilot:
    e.train = train_curriculum

ALT_WORLDS: dict[tuple[str, int], AltWorld] = {}


# ---------------------------------------------------------------- training tensors with the extra columns


def patch_lag(*, lag: torch.Tensor, patch: tuple[np.ndarray, np.ndarray], masked: tuple) -> None:
    """Write the translated lag values into `lag` (rows, 2) for the rows whose lag is not masked."""
    hit, values = patch
    if len(hit) == 0:
        return
    h = torch.from_numpy(hit).to(lag.device)
    v = torch.from_numpy(values).to(lag.device).clamp(-CLIP, CLIP)
    keep = torch.from_numpy(np.stack([~masked[0][hit], ~masked[1][hit]], axis=1)).to(lag.device)
    lag[h] = torch.where(keep, v, lag[h])


class SigGpuMonth(e.LagGpuMonth):
    """The month's rows on the GPU; the fine-tuning tensors get the signal block and the era block appended."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)  # applies block dropout
        fs = FOLD[(self.mr.month, e.CTX["seed"])]
        idx = refit.index
        mr = self.mr
        prim = mr.primary[idx]
        origin = origin_idx_of(origin_us=mr.origin_us[idx])
        day_masked = (t["x"][:, 38] > 0.5).cpu().numpy()
        week_masked = (t["x"][:, 41] > 0.5).cpu().numpy()
        masked = (day_masked, week_masked)
        drop_key = int(e.CTX["seed"]) * 100_003 + int(e.CTX["cut"].timestamp()) % 100_003
        drop_rows = (np.random.default_rng(drop_key).random(len(idx)) < DROP_P) & ~refit.is_val

        def finish(*, b, lag, power_lag, power_sig, signal, cut, slots):
            """Translate, add noise to the rebuilt rows, mask, drop; return the era block (or None)."""
            era = None
            sig.mask_lag_columns(
                block=b, columns=COLUMNS, day_masked=day_masked, week_masked=week_masked
            )
            if signal is not None:
                rows = {"x": mr.x[idx, :45], "prim": prim, "origin": origin}
                hit, values = translate_rows(
                    signal=signal, fs=fs, rows=rows, target_idx=mr.target_idx[idx],
                    power_lag=power_lag, power_sig=power_sig, block=b,
                )  # fmt: skip
                if len(hit):
                    if fs.plan is not None:  # the rebuilt rows get the training noise too
                        rebuilt = b[hit]
                        sig.add_training_noise(
                            block=rebuilt, prim=prim[hit], plan=fs.plan, seed=fs.seed + 17
                        )
                        b[hit] = rebuilt
                    sig.mask_lag_columns(
                        block=b, columns=COLUMNS, day_masked=day_masked, week_masked=week_masked
                    )
                    patch_lag(lag=lag, patch=(hit, values), masked=masked)
                era = era_columns(
                    signal=signal, mr=mr, prim=prim, origin_idx=origin, cut=cut, slots=slots
                )
                REPORT["translated_rows"] = REPORT.get("translated_rows", 0) + len(hit)
            sig.drop_signals(block=b, columns=COLUMNS, rows=drop_rows, only=None)
            return era

        orig_lag = t["x"][:, [LAG_FIRST, LAG_FIRST + 3]].clone()
        lag_plain = orig_lag.clone()
        b_plain = fs.train_block()[idx].copy()
        assert len(b_plain) == len(t["x"]), "extra block and refit tensors differ in rows"
        era_plain = finish(
            b=b_plain, lag=lag_plain, power_lag=fs.fixed, power_sig=fs.power_sig, signal=SIGNAL,
            cut=CUT_DAY, slots=SLOTS,
        )  # fmt: skip
        if CURRICULUM:
            key = (self.mr.month, e.CTX["seed"])
            if key not in ALT_WORLDS:
                ALT_WORLDS[key] = build_alt_world(fs=fs, clean=fs.clean)
            aw = ALT_WORLDS[key]
            n_hh = aw.lag_power.shape[1]
            blk = e.block_of(
                clean=aw.lag_power, primary=prim, target_idx=mr.target_idx[idx], origin_idx=origin
            )
            lag_alt = orig_lag.clone()
            fresh = torch.from_numpy(np.stack([blk[:, 0], blk[:, 3]], axis=1)).to(lag_alt.device)
            keep = torch.from_numpy(np.stack([~day_masked, ~week_masked], axis=1)).to(
                lag_alt.device
            )
            lag_alt = torch.where(keep, fresh.clamp(-CLIP, CLIP), lag_alt)
            sub = {"x": fs.train_rows()["x"][idx], "prim": prim, "origin": origin}
            b_alt = fs._block(rows=sub, power_lag=aw.lag_power, power_sig=aw.sig_power)
            if fs.plan is not None:
                sig.add_training_noise(block=b_alt, prim=prim, plan=fs.plan, seed=fs.seed)
            era_alt = finish(
                b=b_alt, lag=lag_alt, power_lag=aw.lag_power, power_sig=aw.sig_power,
                signal=aw.signal, cut=aw.cut_site[prim], slots=aw.slots,
            )  # fmt: skip
            ti = np.clip(mr.target_idx[idx], 0, n_hh - 1)
            w_alt = np.ones(len(idx), np.float32)
            if aw.signal is not None and aw.hold is not None:
                valid_day = ti // PER_DAY
                w_alt = (~aw.hold[prim, valid_day]).astype(np.float32) * sg.row_weight_factor(
                    signal=aw.signal, cut_day=CUT_DAY, site=prim, valid_day=valid_day
                )
            shift_z_alt = aw.era_delta[prim, ti] / np.asarray(mr.base.target_std)[prim]
            ALT.update(
                lag=lag_alt,
                block=torch.from_numpy(b_alt).to(self.device),
                era=None if era_alt is None else torch.from_numpy(era_alt).to(self.device),
                w=torch.from_numpy(w_alt).to(t["z"].device),
                z=t["z"] + torch.from_numpy(shift_z_alt.astype(np.float32)).to(t["z"].device),
            )
        t["x"][:, LAG_FIRST] = lag_plain[:, 0]
        t["x"][:, LAG_FIRST + 3] = lag_plain[:, 1]
        parts = [t["x"], torch.from_numpy(b_plain).to(self.device)]
        if era_plain is not None:
            parts.append(torch.from_numpy(era_plain).to(self.device))
        t["x"] = torch.cat(parts, dim=1)
        return t


# ---------------------------------------------------------------- preflight and memory


def memory_report() -> dict:
    """Return the peak GPU memory (GiB) and the peak resident memory (GiB) of this process."""
    gpu = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # kB on Linux
    return {"peak_gpu_gib": round(gpu, 2), "peak_rss_gib": round(rss, 2)}


def preflight() -> None:
    """Fail early when a file that experiment 667 or phase A should have written is missing.

    Nothing is created under experiment 667's folders.
    """
    wanted = [
        CHAIN / "step2" / e.SPEC.tag / f"{MONTH}_s{SEED}.pt",
        CHAIN / "stats" / e.SPEC.b1 / f"{fold.FINAL_MONTH}.npz",
        TABLES / f"T_{MONTH}_s{SEED}.npz",
        TABLES / "predict" / f"T_{MONTH}_s{SEED}.npz",
    ]
    if USE_ERA:
        wanted.append(features.DATA_DIR / "known_events.parquet")
        wanted.append(sg.SIGNALS_DIR / f"{SIGNAL_NAME}.parquet")
        if CURRICULUM or CMD == "bank":
            wanted += [
                features.DATA_DIR / "stage0" / "models" / "models.json",
                features.DATA_DIR / "stage0" / "models" / "models.npz",
            ]
    if fold.PHASE != "predict":
        wanted = [p for p in wanted if p.parent.name != "predict"]
    missing = [str(p) for p in wanted if not p.exists()]
    if missing:
        msg = "missing inputs (never created here):\n" + "\n".join(missing)
        raise FileNotFoundError(msg)


# ---------------------------------------------------------------- forecasts


def subset(*, test, sel: np.ndarray):
    return dataclasses.replace(
        test, **{f.name: getattr(test, f.name)[sel] for f in dataclasses.fields(test)}
    )


def forecast_window(
    *,
    model,
    test,
    mr,
    device: str,
    fs: FoldSignals,
    lo: int,
    hi: int,
    power_lag,
    power_sig,
    signal: sg.Signal | None,
    lag_x=None,
    cut=CUT_DAY,
):
    """Return the forecast frame of origins in [lo, hi) with the signal block, the lag translation, and the era block."""
    sel = (test.origin >= lo) & (test.origin < hi)
    if not sel.any():
        return None
    sub = subset(test=test, sel=sel)
    if lag_x is not None:
        sub = dataclasses.replace(sub, x=lag_x[sel])
    block = fs.test_block(test=sub, power_lag=power_lag, power_sig=power_sig)
    x = sub.x.copy()
    parts = []
    era = None
    if signal is not None:
        rows = {"x": sub.x[:, :45], "prim": sub.primary, "origin": sub.origin_idx}
        hit, values = translate_rows(
            signal=signal, fs=fs, rows=rows, target_idx=sub.target_idx, power_lag=power_lag,
            power_sig=power_sig, block=block,
        )  # fmt: skip
        if len(hit):
            x[hit, 37] = np.clip(values[:, 0], -CLIP, CLIP)
            x[hit, 40] = np.clip(values[:, 1], -CLIP, CLIP)
        era = era_columns(
            signal=signal, mr=mr, prim=sub.primary, origin_idx=sub.origin_idx, cut=cut, slots=SLOTS
        )
        REPORT["forecast_translated_rows"] = len(hit)
    parts = [x, block] + ([era] if era is not None else [])
    sub = dataclasses.replace(sub, x=np.concatenate(parts, axis=1))
    model.table = e.CTX["table_ft"]
    return e._window_frame(model=model, test=sub, mr=mr, lo=lo, hi=hi, flag=False, device=device)


def check_fingerprint(*, mr, month: str) -> None:
    """Write the standardiser's hash (train phase) or assert it is unchanged (predict phase)."""
    import hashlib

    std = mr.base.standardiser
    h = hashlib.sha256()
    for a in (std.mean, std.std, std.mask, mr.base.target_mean, mr.base.target_std):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    path = e.WORK / f"standardiser_{month}.sha256"
    if fold.PHASE == "predict":
        assert path.exists() and path.read_text() == h.hexdigest(), (
            "standardiser differs from the train phase"
        )
    elif not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(h.hexdigest())
    else:
        assert path.read_text() == h.hexdigest(), (
            "standardiser changed between two train-phase runs"
        )


def write_contract(*, frame: pl.DataFrame, name: str) -> None:
    """Export a fit's forecasts to contract rows and store them as a prediction set (never overwritten)."""
    import e00667_export as exp  # ty: ignore[unresolved-import]
    from studies import experiment_store as store

    rows = exp.contract_rows(mean=frame)
    assert (rows["nwp_init_time"] == rows["power_fcst_init_time"].dt.truncate("1d")).all()
    # only the finalist rows (valid on or after 2026-07-01) are dropped; a forecast for the last days of
    # the month keeps its half-hours that fall in the next month, so every row set row is complete
    rows = rows.filter(pl.col("valid_time") < pl.lit(HOLDOUT_START))
    try:
        path = store.write_predictions(
            experiment_id=features.EXPERIMENT_ID,
            name=name,
            frame=rows,
            kind="quantiles",
            row_set_id=ROW_SET_ID,
        )
        print(f"wrote prediction set {path}", flush=True)
    except FileExistsError:
        # a rerun after a code fix must not keep the old set without saying so
        msg = f"prediction set {name} already exists: delete it deliberately or rename the arm"
        raise SystemExit(msg) from None


def build_fold(*, device: str):
    """Return the month's rows, test rows, clean stack, GPU rows, and signal builder."""
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, clean.shape
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    mr = e.k.build_month(month=MONTH, clean=clean)
    check_fingerprint(mr=mr, month=MONTH)
    test = e.build_test(month=MONTH, standardiser=mr.base.standardiser)
    require_tables(month=MONTH, seed=SEED)
    g = SigGpuMonth(mr=mr, device=device)
    fs = fold_signals(mr=mr, test=test, clean=clean, month=MONTH, seed=SEED)
    return clean, mr, test, g, fs


def run_fit(*, device: str = "cuda") -> None:
    timings: dict[str, float] = {}
    started = time.monotonic()
    preflight()
    clean, mr, test, g, fs = build_fold(device=device)
    timings["build_fold"] = round(time.monotonic() - started, 1)
    (cut,) = e.k.cutoffs(month=MONTH, cadence=e.CADENCE)
    day = e.f.day_of(month=MONTH, cut=cut)
    out = e.W / "fits" / e.SPEC.tag / f"{MONTH}_d{day}_s{SEED}.parquet"
    mfile = e.model_file(tag=e.SPEC.tag, month=MONTH, day=day, seed=SEED)
    t0 = time.monotonic()
    if mfile.exists():
        model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
        model.load_state_dict(torch.load(mfile, map_location=device))
        model.eval()
        log: dict = {"name": f"{ERA}/{MONTH}_d{day}_s{SEED}", "reused": str(mfile)}
    else:
        assert fold.PHASE != "predict", "the prediction phase must reuse the train phase's model"
        model, log = e.fine_tune(g=g, month=MONTH, cut=cut, seed=SEED, device=device)
        mfile.parent.mkdir(parents=True, exist_ok=True)
        tmp = mfile.with_suffix(".tmp")
        torch.save(model.state_dict(), tmp)
        tmp.rename(mfile)
    timings["fit"] = round(time.monotonic() - t0, 1)
    if fold.TRAIN_ONLY:
        log["signal_report"] = {**fs.report, **REPORT}
        e.c.write_json(
            obj={**log, "timings": timings, "memory": memory_report(), "pilot": MINE.pilot},
            path=e.W / "logs" / f"{MONTH}_s{SEED}_{fold.PHASE}.json",
        )
        print(f"{ERA} {MONTH} s{SEED}: fitted ({timings})", flush=True)
        return
    lo, hi = e.k.window_of(month=MONTH, cadence=e.CADENCE, index=0)
    if MINE.pilot:
        sel_days = test.origin_idx // PER_DAY
        first = int(sel_days.min())
        test = subset(test=test, sel=sel_days < first + PILOT_DAYS)
    t0 = time.monotonic()
    frame = forecast_window(
        model=model, test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi, power_lag=clean,
        power_sig=clean, signal=SIGNAL,
    )  # fmt: skip
    timings["forecast"] = round(time.monotonic() - t0, 1)
    assert frame is not None
    e.atomic_frame(frame=frame, target=out)
    if not MINE.pilot:
        write_contract(frame=frame, name=f"{PREFIX.replace('_', '-')}{ERA}-{MONTH}-s{SEED}")
    e.c.write_json(
        obj={
            **log,
            "timings": timings,
            "signal_report": {**fs.report, **REPORT},
            "memory": memory_report(),
            "pilot": MINE.pilot,
        },
        path=e.W / "logs" / f"{MONTH}_s{SEED}_{fold.PHASE}.json",
    )
    print(
        f"{ERA} {MONTH} s{SEED}: {frame.height:,} forecast rows ({timings}) {memory_report()}",
        flush=True,
    )
    if MINE.pilot:
        print(
            f"pilot alt_passes={REPORT.get('alt_passes')} alt_eras={REPORT.get('alt_eras')}",
            flush=True,
        )


# ---------------------------------------------------------------- the injected bank (inference only)

BANK_PASSES_PER_MONTH: Final = 16


def build_bank_pass(*, pass_index: int) -> dict:
    """Build one bank pass from phase A's injector: its signals of every arm, panel change, and truth."""
    ctx, _models, _stage0 = stage0_context()
    injected = inject.build_pass(
        panel=ctx.panel, weather_shortwave=ctx.weather.shortwave, ready_days=ctx.ready_days,
        verified_days=ctx.verified_days, world="bank", pass_index=pass_index, onset_day=CUT_DAY,
    )  # fmt: skip
    parts = world_parts(
        passes=[injected], placebo_seed=sg.PLACEBO_SEED_BASE["bank"] + pass_index,
        placebo_window=(CUT_DAY - 5, CUT_DAY + 25), with_real=False,
    )  # fmt: skip
    touched = {ev.host for ev in injected.eras} | {s.donor for s in injected.switches}
    touched |= {s.recipient for s in injected.switches if s.recipient is not None}
    parts["touched"] = sorted(touched)
    parts["power"] = injected.power.astype(np.float32)
    parts["truth_table"] = injected.truth
    return parts


def run_bank(*, device: str = "cuda") -> None:
    """Forecast the scored month on the injected bank: 16 passes, each with 2 eras and 2 switching events.

    The passes use the BANK seeds and the phase A bank builder. Each pass is forecast from the
    stored stage 3 model. The lag columns, the signal block, the lag translation, and the era block
    all read the pass's own panel and signals. Only the host, donor, recipient, and neighbour
    Primaries are scored. Truth is NaN for a half-hour on or after 2026-07-01.
    """
    assert fold.PHASE == "predict", "the bank needs the scored month open"
    preflight()
    clean, mr, test, _g, fs = build_fold(device=device)
    (cut,) = e.k.cutoffs(month=MONTH, cadence=e.CADENCE)
    day = e.f.day_of(month=MONTH, cut=cut)
    mfile = e.model_file(tag=e.SPEC.tag, month=MONTH, day=day, seed=SEED)
    assert mfile.exists(), mfile
    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
    model.load_state_dict(torch.load(mfile, map_location=device))
    model.eval()
    lo, hi = e.k.window_of(month=MONTH, cadence=e.CADENCE, index=0)
    ctx, _models, _stage0 = stage0_context()
    n_hh = ctx.panel.power.shape[1]
    sites = k578.sites()
    ts_ids = np.array(sites["time_series_id"].to_list(), dtype=np.int64)
    passes = 1 if MINE.pilot else min(MINE.bank_passes, BANK_PASSES_PER_MONTH)
    out_dir = ROOT / "bank" / DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for j in range(passes):
        started = time.monotonic()
        pass_index = MONTH_INDEX * BANK_PASSES_PER_MONTH + j
        parts = cached(
            path=WORLDS_DIR / f"bank_{MONTH}_p{j}.pkl",
            build=lambda i=pass_index: build_bank_pass(pass_index=i),
        )
        lag_power = clean.copy()
        lag_power[:, :n_hh] += parts["delta"]
        scored = set(parts["touched"])
        for site in parts["touched"]:
            scored |= set(ctx.panel.neighbours.get(site, []))
        signal = None
        if SIGNAL is not None:
            signal = overlay(
                real=SIGNAL, world=parts["world_sigs"][SIGNAL_NAME], sites=parts["touched"]
            )
        keep = np.isin(test.primary, sorted(scored))
        sub_test = subset(test=test, sel=keep)
        # the lag columns of the test rows read the injected panel, as in experiment 693
        lag_blk = e.block_of(
            clean=lag_power, primary=sub_test.primary, target_idx=sub_test.target_idx,
            origin_idx=sub_test.origin_idx,
        )  # fmt: skip
        lag_x = sub_test.x.copy()
        lag_x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = lag_blk
        frame = forecast_window(
            model=model, test=sub_test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
            power_lag=lag_power, power_sig=lag_power, signal=signal, lag_x=lag_x, cut=CUT_DAY,
        )  # fmt: skip
        assert frame is not None
        inside = sub_test.target_idx < n_hh  # on the feature grid, so before 2026-07-01
        t_idx = np.clip(sub_test.target_idx, 0, n_hh - 1)
        truth = np.where(inside, parts["power"][sub_test.primary, t_idx], np.nan)
        clean_share = np.where(inside, clean[sub_test.primary, t_idx], np.nan)
        keys = pl.DataFrame(
            {
                "time_series_id": ts_ids[sub_test.primary],
                "power_fcst_init_time": sub_test.origin,
                "valid_time": sub_test.valid,
                "truth_share": truth,
                "clean_share": clean_share,
                "pass_index": pass_index,
            }
        ).with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        joined = frame.with_columns(pl.col("time_series_id").cast(pl.Int64)).join(
            keys, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="left"
        )
        n_missing = int(joined.filter(pl.col("pass_index").is_null()).height)
        assert n_missing == 0, f"{n_missing} forecast rows have no bank key: the join missed"
        path = out_dir / f"pass{j:02d}.parquet"
        joined.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        parts["truth_table"].write_parquet(out_dir / f"pass{j:02d}_truth.parquet")
        print(
            f"bank {ERA} {MONTH} s{SEED} pass {j}: {joined.height:,} rows, {time.monotonic() - started:.0f} s, {memory_report()}",
            flush=True,
        )


if __name__ == "__main__":
    if CMD == "fit":
        run_fit()
    else:
        run_bank()
    if MINE.pilot:
        print(json.dumps({"pilot_folder": str(e.WORK)}))
