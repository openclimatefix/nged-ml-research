# ruff: noqa: C408, ANN001, ANN003, ANN201, ANN202, D102, D103, D107, D205, E402, E501, F841, I001, PLR0915, PT018
"""Experiment 693 (copy of experiment 677's `e00677_cand.py`): the Candidate with oracle or placebo transfer flags.

Experiment 677's script table-builds, fine-tunes, and scores the Candidate (arm `P7+D50`) one fold
at a time. This copy adds `--flags {none,oracle,placebo}`, which appends four columns (`own_day`,
`nbr_day`, `own_week`, `nbr_week`; see `e00693_flags`) after the signal columns of the lag encoder.
Experiment 677's chains, step-2 weights, and tables are read from its own data folder and never
written. Everything this experiment writes goes under `E00693_ROOT`.

Commands: `fit` (fine-tune and save), `stage0` and `bank` (score the injected banks, see
`e00693_bank`).
"""

import argparse
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

_p = argparse.ArgumentParser()
_p.add_argument("arm")
_p.add_argument("command", choices=("table", "check", "fit", "bank", "stage0", "headbank"))
_p.add_argument("--flags", choices=("none", "oracle", "placebo"), default="none")
_p.add_argument("--bank-passes", type=int, default=48)
_p.add_argument("--bank-kind", choices=("L", "F"), default="L")
_p.add_argument(
    "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
_p.add_argument("--work-suffix", default="")
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--synthetic", type=int, default=1)
_p.add_argument(
    "--models-from",
    default=None,
    help="a folder with the fine-tuned models (the K1-type step 5 keeps them elsewhere)",
)
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
ARM, CMD = MINE.arm, MINE.command
_argv = ["e00574_fit.py", "fit", "--arm", "P5"]
_argv += ["--months", *MINE.months]
_argv += ["--seeds", *map(str, MINE.seeds)]
sys.argv = _argv

import e00574_fit as e  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00591_model as aux  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00613_nbmodel as mymodel  # ty: ignore[unresolved-import]
import e00613_nbsignals as sig  # ty: ignore[unresolved-import]
import e00693_flags as flg
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00677_fold as fold

assert list(MINE.months) == [fold.FINAL_MONTH], "one process per fold month: set E00677_MONTH"

NAME = "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
SRC677 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
)
"""Experiment 677's data folder: chains, step-2 weights, tables, and stored networks, read only."""
ROOT = ML_EXPERIMENTS_DATA_DIR / "00693-oracle-transfer-flag-in-the-candidate-part-1"
"""Everything this experiment writes."""
assert ROOT != SRC677
SHUFFLED = ARM.endswith("-p")
SPEC_NAME = ARM[:-2] if SHUFFLED else ARM
BASE_ARM, *MODS = SPEC_NAME.split("+")
DIR = ARM.replace("-", "").replace("+", "_") + (f"_f{MINE.flags}" if MINE.flags != "none" else "")
TABLES = SRC677 / f"tables_sp{fold.SPAN_MONTHS}"
"""One folder per scored span: the table also covers the scored rows."""
TABLES_610 = (
    ML_EXPERIMENTS_DATA_DIR / "00610-nra-neighbour-summary-lag-encoder-inputs-iteration/tables"
)
TABLES_598 = (
    ML_EXPERIMENTS_DATA_DIR / "00598-nra-gate-against-signals-as-inputs-at-equal-band-width/tables"
)
"""One `T_<month>_s<seed>.npz` per fold month and seed, shared by the arms; experiment 598's is read first."""

P7_SPANS: Final = "gap,ew1,ew3,ew7"


def _stage1(*, spans: str, qs: str | None = None) -> tuple[str, ...]:
    """Return the group tokens of a stage 1 arm: P7's inputs with the weighted averages `spans`."""
    qs = qs if qs is not None else f"gap,{spans}"
    return ("gap", f"own:{spans}", "exp", f"nb:{qs}")


BASES: dict[str, tuple[str, ...]] = {
    "P0": (),  # no signal columns: the two-lag network of ladder step 5, loaded for the bank only
    "P3": ("gap", "exp"),
    "P5": ("gap", "own:ew1,ew3,ew7", "exp"),
    "P7": _stage1(spans="ew1,ew3,ew7"),
    "B1": _stage1(spans="bx1"),
    "B3": _stage1(spans="bx3"),
    "B7": _stage1(spans="bx7"),
    "B14": _stage1(spans="bx14"),
    "B1-14": _stage1(spans="bx1,bx3,bx7,bx14"),
    "B1-28": _stage1(spans="bx1,bx3,bx7,bx14,bx28"),
    "E1": _stage1(spans="ew1"),
    "E7": _stage1(spans="ew7"),
}
"""Per stage 1 arm: the group tokens of its extra columns (see `e00613_nbsignals`)."""
MODIFIERS: Final = ("CF", "D30", "D50", "BN", "X1", "X3", "O1", "O2", "O3", "O4")
assert BASE_ARM in BASES and all(m in MODIFIERS for m in MODS), SPEC_NAME
GROUPS = list(BASES[BASE_ARM])
if "O1" in MODS:  # compact neighbour summaries
    GROUPS = [g.replace("nb:", "nbc:") if g.startswith("nb:") else g for g in GROUPS]
if "O2" in MODS:  # the share of opposite-sign neighbours
    GROUPS += [
        g.replace("nbc:", "nbopp:").replace("nb:", "nbopp:")
        for g in GROUPS
        if g.startswith(("nb:", "nbc:"))
    ]
if "O3" in MODS:  # signals for the week lag only
    GROUPS = [g if g == "exp" else f"{g}@week" for g in GROUPS]
GROUPS = list(dict.fromkeys(GROUPS))
COLUMNS = sig.columns_of(groups=tuple(GROUPS))
N_EXTRA = len(COLUMNS)
N_FLAG = flg.N_FLAGS if MINE.flags != "none" else 0
"""The four flag columns follow the signal columns, so the lag encoder reads them as part of the extra block."""
CROSSFIT = "CF" in MODS
DROP_P = 0.3 if "D30" in MODS else 0.5 if "D50" in MODS else 0.5 if "O4" in MODS else 0.0
DROP_NB_ONLY = "O4" in MODS
AUX_W = 0.1 if "X1" in MODS else 0.3 if "X3" in MODS else 0.0
INPUTS = mymodel.InputSet(n_extra=N_EXTRA + N_FLAG, bottleneck="BN" in MODS, label_head=AUX_W > 0)
CHAIN = Path(os.environ["CHAIN677"])
"""The K1 chain's pre-training and step-2 weights of the final fold (`.../work`), read only."""
W574 = e.W
e.WORK = ROOT / f"{DIR}{MINE.work_suffix}"
e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen
LAG_FIRST, LAG_WIDTH = 37, 8
LEAD_OF = {"day": 43, "week": 44}


def _step2_file(*, month: str, seed: int):
    p = CHAIN / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def _stats_file(*, month: str):
    # the reference month (the month before the fold) reads the fold's own statistics, as experiment 620 did
    p = Path(os.environ.get("STATS677", CHAIN)) / "stats" / e.SPEC.b1 / f"{fold.FINAL_MONTH}.npz"
    assert p.exists(), p
    return p


e.step2_file, e.stats_file = _step2_file, _stats_file
fold.install(e, months=MINE.months, reference_month=fold.PREVIOUS_MONTH)

E3_EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
if MINE.e3 and CMD in {"fit", "inject"}:
    # experiment 599's E3 in the daily fine-tune: no rows from the newest 7 days before the cutoff,
    # and half the weight for the 7 days before those; the step-2 embargo stays at 3 days
    assert e.EMBARGO_DAYS == 3, "the base embargo is experiment 574's 3 days"
    e.EMBARGO_DAYS = E3_EMBARGO_DAYS
    _select_rows = e.c.select_rows

    def _select_rows_e3(**kwargs):
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

    e.c.select_rows = _select_rows_e3


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = mymodel.PermLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, inputs=INPUTS
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


# ---------------------------------------------------------------- neighbours, grid, synthetic shifts


def world() -> s598.Neighbours:
    """Return the Primaries' electrical neighbours and capacities (experiment 582's table)."""
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
SHIFT_PRIMARIES: Final = 5


def shuffled_neighbours(*, nb: s598.Neighbours) -> s598.Neighbours:
    """Return `nb` with each Primary reading the neighbours of the Primary `SHIFT_PRIMARIES` on.

    A Primary never reads itself as a neighbour. The training shifts stay mirrored onto the true
    neighbours (`NB`), so the shuffled summaries carry no information about them.
    """
    n = len(nb.of)
    of = {p: [q for q in nb.of[(p + SHIFT_PRIMARIES) % n] if q != p] for p in range(n)}
    return s598.Neighbours(of=of, capacity=nb.capacity)


NB_READ = shuffled_neighbours(nb=NB) if SHUFFLED else NB
GRID, STEP = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
LABELS: dict[int, np.ndarray] = {}
"""Per month-rows object: (rows, 4) load-shift labels (day active, day depth, week active, week depth; NaN where the lag has none)."""
SYN: dict[int, tuple[np.ndarray, np.ndarray]] = {}
"""Per month-rows object: the synthetic power (16, grid) and the shift added to it (0 where none)."""
_build_month_p5 = e.k.build_month


def origin_idx_of(*, origin_us: np.ndarray) -> np.ndarray:
    return ((origin_us - GRID) // STEP).astype(np.int64)


def build_month_syn(*, month, clean):
    """Build the month's P5 rows with experiment 591's synthetic lag-only load-shifts in the lag
    columns.

    The lag columns are exactly A1's. The synthetic power and its shift are kept for the signal
    block.
    """
    mr = _build_month_p5(month=month, clean=clean)
    if not MINE.synthetic:  # the version without injected events: the lag inputs stay as observed
        zero = np.zeros(clean.shape, np.float32)
        SYN[id(mr)] = (clean, zero)
        LABELS[id(mr)] = np.full((len(mr.primary), mymodel.LABEL_COLUMNS), np.nan, np.float32)
        print(f"{month}: no synthetic events (the version without injected events)", flush=True)
        return mr
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
    lab = np.full((len(mr.primary), mymodel.LABEL_COLUMNS), np.nan, np.float32)
    for j, (miss_col, lead_col) in enumerate(((38, 43), (41, 44))):
        idx = mr.x[:, lead_col].astype(np.int64)
        ok = (idx >= 0) & (mr.x[:, miss_col] < 0.5)
        sh = shift[mr.primary[ok], idx[ok]]
        lab[ok, 2 * j] = (np.abs(sh) > 1e-6).astype(np.float32)
        lab[ok, 2 * j + 1] = sh
    LABELS[id(mr)] = lab
    print(f"{month}: {n_events} synthetic events", flush=True)
    return mr


# ---------------------------------------------------------------- the table T and the row blocks


TABLES_606 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours/tables"
)
CF_BLOCKS: Final = 3
"""The number of contiguous blocks (by valid time) of the cross-fit."""


def table_file(*, month: str, seed: int):
    if fold.PHASE == "predict":  # the prediction phase builds its own tables over the scored window
        return TABLES / "predict" / f"T_{month}_s{seed}.npz"
    if CROSSFIT:
        return TABLES / f"T_cf_{month}_s{seed}.npz"
    return (
        TABLES / f"T_{month}_s{seed}.npz"
    )  # never another experiment's tables: they come from another chain


def cross_fit_quantiles(
    *, mr, month: str, seed: int, q_full: np.ndarray, device: str
) -> np.ndarray:
    """Return `q_full`, with cross-fitted lags-masked quantiles on step 2's training rows.

    Step 2's recipe is experiment 574's `step2` (the pre-trained weights, a fresh mixer, no
    recency weighting, the embargoed rows up to the fold's cutoff). The rows are split into
    `CF_BLOCKS` contiguous blocks by valid time; the network of block `b` is trained on the rows
    outside the block.
    """
    cut = e.c.month_start(month=month)
    refit = e.c.select_rows(rows=mr, cutoff=cut, half_life_days=e.UNIFORM_HALF_LIFE)
    refit = e.embargoed(mr=mr, refit=refit, cut=cut)
    valid = mr.valid_us[refit.index]
    edges = np.quantile(valid, np.linspace(0, 1, CF_BLOCKS + 1))
    block_of_row = np.clip(np.searchsorted(edges, valid, side="right") - 1, 0, CF_BLOCKS - 1)
    out = q_full.copy()
    # Step 2 trained on clean lags, so the cross-fit nets do too: restore the clean lag columns.
    clean = e.c.clean_stack()
    sl = slice(LAG_FIRST, LAG_FIRST + LAG_WIDTH)
    saved = mr.x[:, sl].copy()
    mr.x[:, sl] = e.block_of(
        clean=clean, primary=mr.primary, target_idx=mr.target_idx,
        origin_idx=origin_idx_of(origin_us=mr.origin_us),
    )  # fmt: skip
    g = e.LagGpuMonth(mr=mr, device=device)
    for b in range(CF_BLOCKS):
        started = e.time.monotonic()
        keep = block_of_row != b
        sub = e.c.Refit(
            refit.index[keep], refit.y[keep], refit.weight[keep], refit.is_val[keep], refit.report
        )
        e.CTX.update(on=False, shift=None, sign=0, block=None, seed=seed, cut=cut)
        t = g.refit_tensors(refit=sub)
        model = aux.AuxLagMdn(
            n_primaries=len(mr.base.target_mean), stage="ft", n_knots=e.KNOTS, n_b1=e.N_B1,
            kind="shared",
        ).to(device)  # fmt: skip
        model.table = e.CTX["table_ft"]
        e.load_pretrained_fresh(
            model=model, path=CHAIN / f"pretrained_{e.SPEC.pretrain_from}" / f"{month}_s{seed}.pt"
        )
        model.to(device)
        epochs, best = e.train(
            model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
            lr=e.SPEC.lr, max_epochs=e.SPEC.max_epochs, fresh_lr=e.FRESH_LR, frozen=e.SPEC.frozen,
            anchor=None, lam=0.0, device=device,
        )  # fmt: skip
        rows = refit.index[~keep]
        out[rows] = s598.masked_quantiles(
            model=model, x=mr.x[rows], primary=mr.primary[rows], target_mean=mr.base.target_mean,
            target_std=mr.base.target_std, device=device,
        )  # fmt: skip
        print(
            f"cross-fit {month} s{seed} block {b}: {int(keep.sum()):,} training rows, {epochs} epochs, val NLL {best:.4f}, {e.time.monotonic() - started:.0f} s",
            flush=True,
        )
        del t, model
    del g
    mr.x[:, sl] = saved
    torch.cuda.empty_cache()
    return out


def ensure_tables(*, mr, test, month: str, seeds: list[int], device: str) -> None:
    """Build and cache `T` and the lags-masked q05 and q95 widths of every training and test row."""
    for seed in seeds:
        target = table_file(month=month, seed=seed)
        if target.exists():
            continue
        model = aux.AuxLagMdn(
            n_primaries=len(mr.base.target_mean), stage="ft", n_knots=e.KNOTS, n_b1=e.N_B1,
            kind="shared",
        ).to(device)  # fmt: skip
        model.load_state_dict(torch.load(_step2_file(month=month, seed=seed), map_location=device))
        model.table = e.CTX["table_ft"]
        q_mr = s598.masked_quantiles(
            model=model, x=mr.x, primary=mr.primary, target_mean=mr.base.target_mean,
            target_std=mr.base.target_std, device=device,
        )  # fmt: skip
        q_te = s598.masked_quantiles(
            model=model, x=test.x, primary=test.primary, target_mean=mr.base.target_mean,
            target_std=mr.base.target_std, device=device,
        )  # fmt: skip
        if CROSSFIT:
            q_mr = cross_fit_quantiles(mr=mr, month=month, seed=seed, q_full=q_mr, device=device)
        n_grid = GRID_SHAPE[1]
        # The table also holds the previous month's rows (origins before the cutoff), as the train
        # phase's table does: without them the days before the cutoff, purged from training, have no
        # lag-free forecast.
        ref_prim, ref_origin, ref_target, ref_median = (np.empty(0, np.int64),) * 3 + (
            np.empty(0, np.float32),
        )
        if fold.PHASE == "predict":
            e.CHECK["months"].setdefault(fold.PREVIOUS_MONTH, {})
            ref = e.build_test(month=fold.PREVIOUS_MONTH, standardiser=mr.base.standardiser)
            q_ref = s598.masked_quantiles(
                model=model, x=ref.x, primary=ref.primary, target_mean=mr.base.target_mean,
                target_std=mr.base.target_std, device=device,
            )  # fmt: skip
            ref_prim, ref_origin, ref_target = ref.primary, ref.origin_idx, ref.target_idx
            ref_median = q_ref[:, 1]
        table = s598.build_lag_free_table(
            prim=np.concatenate([mr.primary, test.primary, ref_prim]),
            origin_idx=np.concatenate([origin_idx_of(origin_us=mr.origin_us), test.origin_idx, ref_origin]),
            target_idx=np.concatenate([mr.target_idx, test.target_idx, ref_target]),
            median=np.concatenate([q_mr[:, 1], q_te[:, 1], ref_median]),
            n_primaries=len(mr.base.target_mean), n_grid=n_grid,
        )  # fmt: skip
        filled = table.check_no_lookahead()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npz")
        np.savez(
            tmp, median=table.median, origin_idx=table.origin_idx,
            w_train=np.stack([q_mr[:, 0] - q_mr[:, 1], q_mr[:, 2] - q_mr[:, 1]], axis=1),
            w_test=np.stack([q_te[:, 0] - q_te[:, 1], q_te[:, 2] - q_te[:, 1]], axis=1),
            n_train=len(q_mr), n_test=len(q_te),
        )  # fmt: skip
        tmp.rename(target)
        print(f"{month} s{seed}: T filled {filled:,} of {table.median.size:,} entries", flush=True)
        del model


GRID_SHAPE = (16, e.data.grid_length())
"""16 Primaries by the dense grid: the grid length of the extended inputs."""
assert GRID_SHAPE[0] == 16, GRID_SHAPE


class FoldSignals:
    """The extra-column blocks of one fold month and seed (training rows with synthetic shifts, and test power)."""

    def __init__(self, *, mr, test, clean: np.ndarray, month: str, seed: int) -> None:
        z = np.load(table_file(month=month, seed=seed))
        assert int(z["n_train"]) == len(mr.y) and int(z["n_test"]) == len(test.x), "stale table"
        self.mr, self.test, self.clean, self.month, self.seed = mr, test, clean, month, seed
        self.table = s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
        self.fixed, self.shift = SYN[id(mr)]
        self.choice = s598.mirror_choice(neighbours=NB.of, month=month)
        self.power_sig = s598.mirrored_power(
            base=self.fixed, shift=self.shift, choice=self.choice, capacity=NB.capacity
        )
        self.weather_origin = np.asarray(e.CTX["origin_ft"])
        self._train: np.ndarray | None = None
        self._train_flags: np.ndarray | None = None
        self.report: dict = {}
        self.own_active = flg.active_from_shift(shift=self.shift)
        self.nbr_active = flg.neighbour_active(
            active=self.own_active, nb_of=NB.of, choice=self.choice
        )
        self.window = (int(mr.target_idx.min()), int(mr.target_idx.max()) + 1)
        """The grid range where the fold's training events exist; the placebo rolls inside it."""
        self.placebo_offset = flg.placebo_offset(
            month=month, seed=seed, window_length=self.window[1] - self.window[0]
        )

    def masks(
        self, *, kind: str, own: np.ndarray | None = None, nbr: np.ndarray | None = None,
        window: tuple[int, int] | None = None, offset: int | None = None,
    ):  # fmt: skip
        """Return the (own, nbr) transfer masks the `kind` flags read: the true ones or the rolled placebo."""
        own = self.own_active if own is None else own
        nbr = self.nbr_active if nbr is None else nbr
        if kind == "placebo":
            window = window or self.window
            offset = offset or self.placebo_offset
            return (
                flg.roll_mask(mask=own, offset=offset, window=window),
                flg.roll_mask(mask=nbr, offset=offset, window=window),
            )
        assert kind == "oracle", kind
        return own, nbr

    def train_flags(self) -> np.ndarray:
        """Return the training rows' (rows, 4) flags of this arm, cached; injected shifts only."""
        if self._train_flags is None:
            own, nbr = self.masks(kind=MINE.flags)
            rows = self._train_rows()
            self._train_flags = flg.lag_flags(
                own=own, nbr=nbr, prim=rows["prim"], origin_idx=rows["origin"],
                day_idx=rows["x"][:, LEAD_OF["day"]].astype(np.int64),
                week_idx=rows["x"][:, LEAD_OF["week"]].astype(np.int64),
                day_missing=rows["x"][:, 38] > 0.5, week_missing=rows["x"][:, 41] > 0.5,
            )  # fmt: skip
        return self._train_flags

    def _block(self, *, rows: dict, power_lag, power_sig) -> np.ndarray:
        lag_idx = {k: rows["x"][:, col].astype(np.int64) for k, col in LEAD_OF.items()}
        return sig.signal_columns(
            prim=rows["prim"], origin_idx=rows["origin"], day_idx=lag_idx["day"],
            week_idx=lag_idx["week"], day_missing=rows["x"][:, 38] > 0.5,
            week_missing=rows["x"][:, 41] > 0.5, power_lag=power_lag, power_sig=power_sig,
            table=self.table, nb=NB_READ, weather_origin=self.weather_origin, columns=COLUMNS,
        )  # fmt: skip

    def _train_rows(self) -> dict:
        return {
            "x": self.mr.x[:, :45], "prim": self.mr.primary,
            "origin": origin_idx_of(origin_us=self.mr.origin_us),
        }  # fmt: skip

    def _test_rows(self, *, test=None) -> dict:
        t = test if test is not None else self.test
        return {"x": t.x[:, :45], "prim": t.primary, "origin": t.origin_idx}

    def train_block(self) -> np.ndarray:
        """Return the training rows' block: synthetic shifts in the own and neighbour power,
        training noise where the training-to-test spread ratio is low. Cached.
        """
        if self._train is not None:
            return self._train
        synth = self._block(rows=self._train_rows(), power_lag=self.fixed, power_sig=self.power_sig)
        real_test = self._block(rows=self._test_rows(), power_lag=self.clean, power_sig=self.clean)
        plan = self.noise_plan(synth=synth, real_test=real_test)
        sig.add_training_noise(block=synth, prim=self.mr.primary, plan=plan, seed=self.seed)
        del real_test
        assert synth.shape[0] == len(self.mr.x), "training block and month rows differ"
        self._train = synth
        return synth

    def noise_plan(self, *, synth: np.ndarray, real_test: np.ndarray) -> sig.NoisePlan:
        """Return the noise plan and keep a summary of the per-column training-to-test ratios."""
        names = [sig.ALL_NAMES[c] for c in COLUMNS]
        plan = sig.noise_plan(
            train=synth, train_prim=self.mr.primary, test=real_test, test_prim=self.test.primary,
            n_primaries=16, names=names,
        )  # fmt: skip
        self.report = {
            "ratio_min_by_column": {
                names[c]: float(np.nanmin(plan.ratio[j]))
                if np.isfinite(plan.ratio[j]).any()
                else None
                for j, c in enumerate(plan.columns)
            },
            "noised_columns": [names[c] for j, c in enumerate(plan.columns) if plan.sd[j].any()],
        }
        return plan

    def test_block(self, *, test, power_lag, power_sig) -> np.ndarray:
        """Return a test (sub)set's block from the given power (observed, or the injected bank's)."""
        return self._block(
            rows=self._test_rows(test=test), power_lag=power_lag, power_sig=power_sig
        )


FOLD: dict[tuple[str, int], FoldSignals] = {}


def fold_signals(*, mr, test, clean, month: str, seed: int) -> FoldSignals | None:
    if not N_EXTRA:
        return None
    key = (month, seed)
    if key not in FOLD:
        for old in [k for k in FOLD if k[0] != month]:
            del FOLD[old]
        FOLD[key] = FoldSignals(mr=mr, test=test, clean=clean, month=month, seed=seed)
    return FOLD[key]


# ---------------------------------------------------------------- training tensors with the extra columns


class SigGpuMonth(e.LagGpuMonth):
    """The month's rows on the GPU; the fine-tuning tensors get the extra columns appended."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)  # applies block dropout
        if not N_EXTRA:
            assert N_FLAG == 0, "flag columns need the signal block"
            return t
        fs = FOLD[(self.mr.month, e.CTX["seed"])]
        b = fs.train_block()[refit.index].copy()
        assert len(b) == len(t["x"]), "extra block and refit tensors differ in rows"
        day_masked = (t["x"][:, 38] > 0.5).cpu().numpy()
        week_masked = (t["x"][:, 41] > 0.5).cpu().numpy()
        sig.mask_lag_columns(
            block=b, columns=COLUMNS, day_masked=day_masked, week_masked=week_masked
        )  # fmt: skip
        if DROP_P > 0:  # signal dropout: the "no information" value on a share of the rows
            key = int(e.CTX["seed"]) * 100_003 + int(e.CTX["cut"].timestamp()) % 100_003
            rows = (np.random.default_rng(key).random(len(b)) < DROP_P) & ~refit.is_val
            only = None
            if DROP_NB_ONLY:
                only = [
                    c
                    for c in COLUMNS
                    if sig.ALL_NAMES[c].startswith(("nb_", "own_plus_nb_", "nbopp_"))
                ]
            sig.drop_signals(block=b, columns=COLUMNS, rows=rows, only=only)
        parts = [t["x"], torch.from_numpy(b).to(self.device)]
        if (
            N_FLAG
        ):  # the flags are exempt from signal dropout and are 0 wherever their lag is masked
            fl = fs.train_flags()[refit.index].copy()
            fl[day_masked, :2] = 0.0
            fl[week_masked, 2:] = 0.0
            e.CTX["flag_rate"] = {n: float(fl[:, j].mean()) for j, n in enumerate(flg.FLAG_NAMES)}
            if (
                MINE.flags == "placebo"
            ):  # per-Primary counts and span lengths are kept, so the rates match
                tr = FOLD[(self.mr.month, e.CTX["seed"])]
                true_rows = flg.lag_flags(
                    own=tr.own_active, nbr=tr.nbr_active, prim=self.mr.primary[refit.index],
                    origin_idx=origin_idx_of(origin_us=self.mr.origin_us[refit.index]),
                    day_idx=self.mr.x[refit.index, LEAD_OF["day"]].astype(np.int64),
                    week_idx=self.mr.x[refit.index, LEAD_OF["week"]].astype(np.int64),
                    day_missing=day_masked, week_missing=week_masked,
                )  # fmt: skip
                ratio = fl.mean(axis=0) / np.maximum(true_rows.mean(axis=0), 1e-9)
                assert ((ratio > 0.7) & (ratio < 1.3)).all(), f"placebo flag rate off: {ratio}"
            print(f"{self.mr.month} {MINE.flags} flag rates {e.CTX['flag_rate']}", flush=True)
            parts.append(torch.from_numpy(fl).to(self.device))
        if AUX_W > 0:
            lab = torch.from_numpy(LABELS[id(self.mr)][refit.index]).to(self.device)
            hidden = torch.from_numpy(
                np.stack([day_masked, day_masked, week_masked, week_masked], axis=1)
            )
            lab = torch.where(hidden.to(self.device), torch.full_like(lab, float("nan")), lab)
            parts.append(lab)
        t["x"] = torch.cat(parts, dim=1)
        return t


# ---------------------------------------------------------------- commands


def _out(out_dir, month, r, day, seed):
    return out_dir / f"{month}_r{r:02d}_d{day}_s{seed}.parquet"


def _done(*, out) -> bool:
    return out.exists() or out.with_suffix(".empty").exists()


def subset(*, test, sel: np.ndarray):
    """Return the test rows `sel` (a boolean mask) as a new `TestRows`."""
    return dataclasses.replace(
        test, **{f.name: getattr(test, f.name)[sel] for f in dataclasses.fields(test)}
    )


def forecast_window(
    *,
    model,
    test,
    mr,
    device: str,
    fs: FoldSignals | None,
    lo: int,
    hi: int,
    power_lag,
    power_sig,
    lag_x=None,
    flags=None,
    mask_lags=None,
):
    """Return the forecast frame of origins in [lo, hi) with the extra columns built from the given power.

    `flags` is the (rows of `test`, 4) flag block of an arm with flag columns. `mask_lags` is a
    (rows of `test`, 2) boolean array (day, week) of lags to mask, as the stage-0 arm A0-mask does.
    """
    sel = (test.origin >= lo) & (test.origin < hi)
    if not sel.any():
        return None
    sub = subset(test=test, sel=sel)
    if lag_x is not None:
        sub = dataclasses.replace(sub, x=lag_x[sel])
    if mask_lags is not None:
        sub = mask_lag_inputs(sub=sub, masked=mask_lags[sel])
    if fs is not None:
        block = fs.test_block(test=sub, power_lag=power_lag, power_sig=power_sig)
        if mask_lags is not None:
            sig.mask_lag_columns(
                block=block, columns=COLUMNS, day_masked=sub.x[:, 38] > 0.5,
                week_masked=sub.x[:, 41] > 0.5,
            )  # fmt: skip
        parts = [sub.x, block]
        if N_FLAG:
            assert flags is not None, "an arm with flag columns needs the flag block"
            fl = flags[sel].copy()
            if mask_lags is not None:  # a masked lag carries no flag, as in training
                fl[mask_lags[sel][:, 0], :2] = 0.0
                fl[mask_lags[sel][:, 1], 2:] = 0.0
            parts.append(fl)
        sub = dataclasses.replace(sub, x=np.concatenate(parts, axis=1))
    model.table = e.CTX["table_ft"]
    return e._window_frame(
        model=model, test=sub, mr=mr, lo=lo, hi=hi, flag=False, device=device
    )  # fmt: skip


def no_flags(*, test) -> np.ndarray | None:
    """Return all-zero flags for rows of observed power: real transfers carry flag 0 in every arm's fine-tune."""
    return np.zeros((len(test.x), flg.N_FLAGS), np.float32) if N_FLAG else None


def bank_flags(*, fs, test, shift: np.ndarray, month: str, seed: int, window: tuple[int, int]):
    """Return the flags of experiment 581's injected bank rows (shift in the lag inputs and the mirrored neighbours)."""
    if not N_FLAG:
        return None
    own = flg.active_from_shift(shift=shift)
    nbr = flg.neighbour_active(active=own, nb_of=NB.of, choice=fs.choice)
    offset = flg.placebo_offset(month=month, seed=1000 + seed, window_length=window[1] - window[0])
    own, nbr = fs.masks(kind=MINE.flags, own=own, nbr=nbr, window=window, offset=offset)
    return flg.lag_flags(
        own=own, nbr=nbr, prim=test.primary, origin_idx=test.origin_idx,
        day_idx=test.x[:, LEAD_OF["day"]].astype(np.int64), week_idx=test.x[:, LEAD_OF["week"]].astype(np.int64),
        day_missing=test.x[:, 38] > 0.5, week_missing=test.x[:, 41] > 0.5,
    )  # fmt: skip


def mask_lag_inputs(*, sub, masked: np.ndarray):
    """Return `sub` with the masked day and week lags set to the absent encoding, as block dropout sets it.

    The absent encoding is value 0, missing flag 1, and the largest age of the lag (experiment 574's
    `apply_dropout`).
    """
    x = sub.x.copy()
    amax = e.lg.age_max(cap_steps=e.SPEC.cap_steps)
    for j, c0 in enumerate((37, 40)):
        rows = masked[:, j]
        x[rows, c0], x[rows, c0 + 1], x[rows, c0 + 2] = 0.0, 1.0, float(amax[j])
    return dataclasses.replace(sub, x=x)


def run_prepare(*, device: str, command: str) -> None:
    """The `table` and `check` commands."""
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, clean.shape
    for month in e.ARGS.months:
        mr = e.k.build_month(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        if N_EXTRA:
            ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
        if command == "check":
            for seed in e.ARGS.seeds:
                fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
                if fs is None:
                    print(f"{ARM}: no extra columns, nothing to check", flush=True)
                    continue
                fs.train_block()
                print(
                    f"== {month} s{seed}: columns noised {fs.report['noised_columns']}", flush=True
                )
                print(
                    "  smallest training-to-test sd ratio by column",
                    fs.report["ratio_min_by_column"],
                )
                exp_missing_share(fs=fs, test=test)
                known_answer_real(fs=fs, test=test)
        del mr, test
        FOLD.clear()
        torch.cuda.empty_cache()


def exp_missing_share(*, fs: FoldSignals, test) -> None:
    """Print, per Primary, the share of test rows whose expected-missing flags are set."""
    names = [sig.ALL_NAMES[c] for c in COLUMNS]
    flags = [j for j, n in enumerate(names) if n.startswith("exp_missing")]
    if not flags:
        return
    block = fs._block(rows=fs._test_rows(), power_lag=fs.clean, power_sig=fs.clean)
    print("  test-row share with the expected-missing flag set (day, week), by Primary")
    for p in range(len(fs.mr.base.target_mean)):
        rows = test.primary == p
        if rows.any():
            print(
                f"    Primary {p:2d} ({int(rows.sum()):6d} rows)",
                [round(float(block[rows, j].mean()), 4) for j in flags], flush=True,
            )  # fmt: skip


def known_answer_real(*, fs: FoldSignals, test) -> None:
    """Shift one Primary's real power by 0.2 for 10 days, mirror it onto its neighbours, and check the columns."""
    p, shift = 3, 0.2
    start = int(np.median(test.origin_idx)) - 5 * sig.PER_DAY
    end = start + 10 * sig.PER_DAY
    lag = fs.clean.copy()
    lag[p, start:end] = np.where(np.isfinite(lag[p, start:end]), lag[p, start:end] + shift, np.nan)
    sh = np.zeros_like(lag)
    sh[p, start:end] = shift
    pwr_sig = s598.mirrored_power(base=lag, shift=sh, choice=fs.choice, capacity=NB.capacity)
    sel = (
        (test.primary == p)
        & (test.origin_idx >= start + 8 * sig.PER_DAY // 2)
        & (test.origin_idx < end - sig.PER_DAY)
    )
    sub = subset(test=test, sel=sel)
    base = fs.test_block(test=sub, power_lag=fs.clean, power_sig=fs.clean)
    moved = fs.test_block(test=sub, power_lag=lag, power_sig=pwr_sig)
    diff = moved - base
    live = sub.x[:, 38] < 0.5
    names = [sig.ALL_NAMES[c] for c in COLUMNS]
    for target in ("gap_day", "ew1_day", "own_plus_nb_gap_day", "nb_sum_gap_day"):
        if target in names:
            v = float(np.median(diff[live, names.index(target)])) / sig.SCALE
            print(
                f"  known answer (Primary {p}, shift {shift}): {target} moves by {v:.3f}",
                flush=True,
            )
            if not SHUFFLED and target == "gap_day":
                assert abs(v - shift) < 0.05, v
            if not SHUFFLED and target == "own_plus_nb_gap_day" and NB.of[p]:
                assert abs(v) < 0.05, v  # a pure transfer cancels in the sum


def row_nll(*, out, z, p):
    """The mixture negative log-likelihood, plus the label-head weight times the label loss (training only)."""
    nll = e.base.mixture_nll(out=out, z=z, components=e.m.COMPONENTS)
    lab = mymodel.STASH.pop("shift", None)
    if lab is not None and AUX_W > 0:
        nll = nll + AUX_W * lab
    return nll


def check_fingerprint(*, mr, month: str) -> None:
    """Write the training standardiser and target statistics (train phase) or assert they are
    unchanged (predict phase).

    The predict phase reloads the network fitted in the train phase, so a standardiser built from
    different rows would shift every forecast silently.
    """
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
    elif not path.exists():  # a rerun of the train phase never rewrites it
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(h.hexdigest())
    else:
        assert path.read_text() == h.hexdigest(), (
            "standardiser changed between two train-phase runs"
        )


def run_fit(*, device: str = "cuda") -> None:
    flg.known_answer()
    if AUX_W > 0:
        e.row_nll = row_nll
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, clean.shape
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        mr = e.k.build_month(month=month, clean=clean)
        check_fingerprint(mr=mr, month=month)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        if N_EXTRA:
            ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
        g = SigGpuMonth(mr=mr, device=device)
        for seed in e.ARGS.seeds:
            fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            for i, cut in enumerate(cuts):
                day = e.f.day_of(month=month, cut=cut)
                out = e.W / "fits" / e.SPEC.tag / f"{month}_d{day}_s{seed}.parquet"
                mfile = e.model_file(tag=e.SPEC.tag, month=month, day=day, seed=seed)
                if _done(out=out) and mfile.exists():
                    continue
                started = e.time.monotonic()
                if mfile.exists():
                    model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                    model.load_state_dict(torch.load(mfile, map_location=device))
                    model.eval()
                    log: dict = {"name": f"{ARM}/{month}_d{day}_s{seed}", "reused": str(mfile)}
                else:
                    assert fold.PHASE != "predict", (
                        "the prediction phase must reuse the train phase's model"
                    )
                    model, log = e.fine_tune(g=g, month=month, cut=cut, seed=seed, device=device)
                    mfile.parent.mkdir(parents=True, exist_ok=True)
                    tmp = mfile.with_suffix(".tmp")
                    torch.save(model.state_dict(), tmp)
                    tmp.rename(mfile)
                lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
                fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
                if fold.TRAIN_ONLY and month == fold.FINAL_MONTH:
                    frame = None  # the scored window stays closed while the networks are trained
                else:
                    frame = forecast_window(
                        model=model, test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
                        power_lag=clean, power_sig=clean, flags=no_flags(test=test),
                    )  # fmt: skip
                if frame is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_suffix(".empty").touch()
                else:
                    e.atomic_frame(frame=frame, target=out)
                log["seconds"] = round(e.time.monotonic() - started, 2)
                log["signal_report"] = fs.report if fs is not None else {}
                e.c.write_json(
                    obj=log,
                    path=e.W / "logs" / e.SPEC.tag / f"{month}_d{day}_s{seed}_{fold.PHASE}.json",
                )
                print(
                    f"{ARM} {log['name']}: {log.get('epochs', 'reused')} epochs, {log['seconds']} s",
                    flush=True,
                )
                del model
        del g, mr, test
        FOLD.clear()
        torch.cuda.empty_cache()
    out = e.WORK / f"checks_fit_{DIR}_s{'-'.join(map(str, e.ARGS.seeds))}_{fold.PHASE}.json"
    out.write_text(json.dumps(e.CHECK, indent=1, default=str))
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})


# ---------------------------------------------------------------- the held-out injected bank (follow)


BANK_SEED_BASE: Final = 9_000_000
"""Seeds of the held-out bank: far from every training seed (the training shifts use the month number, 202511 and the like)."""


def run_bank(*, device: str = "cuda") -> None:
    """Forecast the scored window twice per saved model, from the observed power and from the bank's.

    The bank is experiment 581's generator with a seed far from every training seed, so its events are
    new to every network. It shifts the lag inputs only (own lags, and the mirrored neighbour
    summaries), never the targets. For each scored row whose 1-day lag falls inside an injected shift,
    the file holds the shift and both medians (shares of capacity); the follow is the median's change
    divided by the shift.
    """
    import polars as pl

    assert fold.PHASE == "predict", "the bank needs the scored window open"
    flg.known_answer()
    clean = e.c.clean_stack()
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    sites = k578.sites()
    prim_of_ts = dict(
        zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True)
    )
    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        mr = e.k.build_month(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        if N_EXTRA:
            ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
        first, last = int(test.target_idx.min()), int(test.target_idx.max())
        fixed, n_events = bank.synthetic_clean(
            clean=clean, first=first, last=last, seed=BANK_SEED_BASE + int(month.replace("-", ""))
        )
        shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0).astype(
            np.float32
        )
        blk = e.block_of(
            clean=fixed,
            primary=test.primary,
            target_idx=test.target_idx,
            origin_idx=test.origin_idx,
        )
        lag_x = test.x.copy()
        lag_x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
        day_idx = test.x[:, LEAD_OF["day"]].astype(np.int64)
        row_shift = np.where(
            day_idx >= 0, shift[test.primary, np.clip(day_idx, 0, shift.shape[1] - 1)], 0.0
        )
        print(
            f"{month}: bank with {n_events} events; {int((row_shift != 0).sum()):,} scored rows have a shifted 1-day lag",
            flush=True,
        )
        keys = pl.DataFrame(
            {
                "time_series_id": np.array(
                    [sites["time_series_id"][int(i)] for i in test.primary], dtype=np.int64
                ),
                "power_fcst_init_time": test.origin,
                "valid_time": test.valid,
                "lag_shift": row_shift,
            }
        ).with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        for seed in e.ARGS.seeds:
            fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            for i, cut in enumerate(cuts):
                day = e.f.day_of(month=month, cut=cut)
                mfile = (
                    Path(MINE.models_from) / f"{month}_d{day}_s{seed}.pt"
                    if MINE.models_from
                    else e.model_file(tag=e.SPEC.tag, month=month, day=day, seed=seed)
                )
                assert mfile.exists(), mfile
                model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                model.load_state_dict(torch.load(mfile, map_location=device))
                model.eval()
                lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
                base = forecast_window(
                    model=model,
                    test=test,
                    mr=mr,
                    device=device,
                    fs=fs,
                    lo=lo,
                    hi=hi,
                    power_lag=clean,
                    power_sig=clean,
                    flags=no_flags(test=test),
                )
                power_sig = (
                    s598.mirrored_power(
                        base=fixed, shift=shift, choice=fs.choice, capacity=NB.capacity
                    )
                    if fs is not None
                    else None
                )
                inj = forecast_window(
                    model=model,
                    test=test,
                    mr=mr,
                    device=device,
                    fs=fs,
                    lo=lo,
                    hi=hi,
                    power_lag=fixed,
                    power_sig=power_sig,
                    lag_x=lag_x,
                    flags=bank_flags(
                        fs=fs,
                        test=test,
                        shift=shift,
                        month=month,
                        seed=seed,
                        window=(first, last + 1),
                    ),
                )
                k3 = ["time_series_id", "power_fcst_init_time", "valid_time"]
                j = (
                    base.select(
                        *k3,
                        pl.col("q50").alias("median_base"),
                        pl.col("q05").alias("q05_base"),
                        pl.col("q95").alias("q95_base"),
                    )
                    .join(
                        inj.select(
                            *k3,
                            pl.col("q50").alias("median_bank"),
                            pl.col("q05").alias("q05_bank"),
                            pl.col("q95").alias("q95_bank"),
                        ),
                        on=k3,
                    )
                    .join(keys, on=k3)
                )
                assert j.height == base.height == inj.height, (j.height, base.height, inj.height)
                out = e.W / "bank" / f"{month}_d{day}_s{seed}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                j.write_parquet(out.with_suffix(".tmp"))
                out.with_suffix(".tmp").rename(out)
                print(
                    f"{ARM} {month} s{seed}: {j.height:,} rows, {int((j['lag_shift'] != 0).sum()):,} with a shifted lag -> {out}",
                    flush=True,
                )
                del model
        del mr, test
        FOLD.clear()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- the injected banks of experiment 693 (stage 0 and the arms)

import e00660_data as d660  # ty: ignore[unresolved-import]
import e00660_generator as gen660  # ty: ignore[unresolved-import]
import polars as pl
from studies.eval_protocol.slices import MVA_ONLY_LABELS

QUANTILES: Final = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
BANK660_SEED: Final = 693_000
PRE_WINDOW_DAYS: Final = 60
"""The bank's panel starts this long before the fold's first scored month, so events begun earlier reach into it."""
AFTER_DAYS: Final = 14
"""A target within this many days after an event's end is in phase P2 or P3."""
PHASE_NAMES: Final = {0: "other", 1: "P1", 2: "P2", 3: "P3"}


def bank660_config() -> gen660.GeneratorConfig:
    """Return the deciding banks' generator configuration, pre-registered in `result.md`."""
    return dataclasses.replace(
        gen660.GeneratorConfig(),
        hard_negatives=False,
        events_per_primary_year=3.0,
        class_probs=(
            ("edge_transfer", 0.55),
            ("edge_transfer_unseen_donor", 0.10),
            ("edge_transfer_unseen_recipient", 0.15),
            ("permanent_shift", 0.20),
        ),
    )


class World660:
    """One injected pass: the injected power of every Primary and the oracle's transfer masks."""

    def __init__(
        self,
        *,
        clean: np.ndarray,
        fixed: np.ndarray,
        own_active: np.ndarray,
        events: pl.DataFrame,
        stop: int,
        start: int,
    ) -> None:
        self.fixed, self.own_active, self.events, self.stop, self.start = (
            fixed,
            own_active,
            events,
            stop,
            start,
        )
        self.nbr_active = flg.neighbour_active(active=own_active, nb_of=NB.of, choice=None)
        both = np.isfinite(fixed) & np.isfinite(clean)
        self.shift = np.where(both, fixed - clean, 0.0).astype(np.float32)


def world660(*, clean: np.ndarray, month: str, r: int) -> World660:
    """Inject pass `r` of the experiment 660 generator into the cleaned power around the fold's scored months.

    The panel runs from 60 days before the fold's first scored month to the end of the scored
    months (or the generator's cutoff, 2026-07-01). The 16th Primary has no noise unit, so it takes
    no event. The injected power is a share of capacity again, on the full grid.
    """
    sm = k578.sites()
    labels = sm["label"].to_list()
    noise = json.loads((d660.E653 / "noise_units.json").read_text())["sigma_block_mw"]
    cap = NB.capacity
    start = int((fold.FINAL_START.timestamp() * 1e6 - GRID) // STEP) - PRE_WINDOW_DAYS * 48
    stop = min(
        int((fold.FINAL_STOP.timestamp() * 1e6 - GRID) // STEP) + 15 * 48,
        int((gen660.CUTOFF.timestamp() * 1e6 - GRID) // STEP),
    )
    """The panel runs 15 days past the fold's scored months, so the longest horizons of the last origins lie inside it."""
    sites = [lab for lab in labels if lab in noise]
    p_of = {lab: i for i, lab in enumerate(labels)}
    origin = datetime.fromtimestamp(GRID / 1e6, tz=UTC) + timedelta(minutes=30 * start)
    panel = gen660.PowerPanel(
        power={lab: clean[p_of[lab], start:stop].astype(float) * cap[p_of[lab]] for lab in sites},
        is_mva={lab: lab in MVA_ONLY_LABELS for lab in sites},
        noise={lab: float(noise[lab]) for lab in sites},
        neighbours={
            lab: [labels[q] for q in NB.of[p_of[lab]] if labels[q] in sites] for lab in sites
        },
        origin=origin,
    )
    inj = gen660.inject(
        panel=panel,
        config=bank660_config(),
        seed=BANK660_SEED + 100 * (int(fold.FINAL_MONTH[:4]) * 12 + int(fold.FINAL_MONTH[5:])) + r,
    )
    fixed = clean.copy()
    for lab in sites:
        p = p_of[lab]
        seg = inj.power[lab] / cap[p]
        fixed[p, start:stop] = np.where(np.isfinite(clean[p, start:stop]), seg, np.nan)
    own = np.zeros(clean.shape, bool)
    for lab, mask in gen660.event_windows(events=inj.events, length=stop - start).items():
        own[p_of[lab], start:stop] = mask
    assert not own[:, :start].any() and not own[:, stop:].any()
    assert not (
        ((np.abs(fixed - clean) > flg.SHIFT_EPS) & np.isfinite(fixed) & np.isfinite(clean)) & ~own
    ).any(), "a shifted half-hour lies outside every event window"
    return World660(
        clean=clean, fixed=fixed, own_active=own, events=inj.events, stop=stop, start=start
    )


def ended_within(*, active: np.ndarray, days: int) -> np.ndarray:
    """Return, per Primary and grid time, whether a transfer ended within the last `days` days (and none is active)."""
    c = np.concatenate(
        [np.zeros((active.shape[0], 1), np.int64), np.cumsum(active, axis=1)], axis=1
    )
    w = days * 48
    lo = np.clip(np.arange(active.shape[1]) + 1 - w, 0, None)
    seen = c[:, np.arange(active.shape[1]) + 1] - c[:, lo]
    return (seen > 0) & ~active


def phases_of(*, flags: np.ndarray, tin: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return the phase code per row: 1 (P1), 2 (P2), 3 (P3), 0 (any other state)."""
    lag_in = flags.sum(axis=1) > 0
    out = np.zeros(len(flags), np.int8)
    out[lag_in & tin] = 1
    out[lag_in & ~tin & after] = 2
    out[~lag_in & ~tin & after] = 3
    return out


def row_scores(*, q: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row scores, in points of capacity: CRPS (twice the mean pinball loss), absolute error of the median, tail pinball, and the q05 to q95 cover."""
    lv = np.asarray(QUANTILES, np.float32)
    err = y[:, None] - q
    pin = np.maximum(lv * err, (lv - 1.0) * err) * 100.0
    return {
        "crps": 2.0 * pin.mean(axis=1),
        "ae": np.abs(err[:, 5]) * 100.0,
        "pin01": pin[:, 0], "pin05": pin[:, 1], "pin95": pin[:, 9], "pin99": pin[:, 10],
        "cover": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(np.float32),
    }  # fmt: skip


def frame_q(*, frame: pl.DataFrame, keys: pl.DataFrame) -> np.ndarray:
    """Return the frame's quantiles in the row order of `keys`."""
    k3 = ["time_series_id", "power_fcst_init_time", "valid_time"]
    j = (
        keys.select(k3)
        .with_row_index("i")
        .join(frame.select(*k3, *QCOLS), on=k3, how="left", maintain_order="left")
    )
    q = j.select(QCOLS).to_numpy().astype(np.float32)
    assert len(q) == len(keys) and np.isfinite(q).all(), "the frame misses some key rows"
    return np.sort(q, axis=1)


def test_keys(*, test, sites) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_series_id": np.array([sites["time_series_id"][int(i)] for i in test.primary], dtype=np.int64),
            "power_fcst_init_time": test.origin, "valid_time": test.valid,
        }
    ).with_columns(
        pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
        pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
    )  # fmt: skip


def run_stage0(*, device: str = "cuda") -> None:
    """Stage 0: the stored experiment 677 step-8 networks on bank L with no retraining.

    For each pass of the bank, forecast the scored window from the observed lags (A0-clean, the
    ceiling of any lag-time oracle), from the injected lags (A0), and from the injected lags with
    every flagged lag masked (A0-mask). The truth is the unshifted observed power. Rows are kept in
    phases P1, P2, and P3, with the per-seed and the seed-mean scores of each variant.
    """
    assert fold.PHASE == "predict", "stage 0 needs the scored window open"
    flg.known_answer()
    clean = e.c.clean_stack()
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    sites = k578.sites()
    (month,) = e.ARGS.months
    mr = e.k.build_month(month=month, clean=clean)
    test = e.build_test(month=month, standardiser=mr.base.standardiser)
    for seed in e.ARGS.seeds:
        assert table_file(month=month, seed=seed).exists(), (
            "experiment 677's predict table is missing"
        )
    ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
    keys = test_keys(test=test, sites=sites)
    lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=0)
    models = {}
    for seed in e.ARGS.seeds:
        mfile = Path(MINE.models_from) / f"{month}_d0_s{seed}.pt"
        assert mfile.exists(), mfile
        m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
        m.load_state_dict(torch.load(mfile, map_location=device))
        m.eval()
        models[seed] = m
        fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
    cutoff_idx = int((gen660.CUTOFF.timestamp() * 1e6 - GRID) // STEP)
    out_dir = e.W / "stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    day_idx = test.x[:, LEAD_OF["day"]].astype(np.int64)
    week_idx = test.x[:, LEAD_OF["week"]].astype(np.int64)
    y_all = clean[test.primary, test.target_idx].astype(np.float32)
    clean_frames = {
        seed: forecast_window(
            model=model,
            test=test,
            mr=mr,
            device=device,
            fs=fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed),
            lo=lo,
            hi=hi,
            power_lag=clean,
            power_sig=clean,
        )
        for seed, model in models.items()
    }  # the observed-lag forecast (A0-clean) does not depend on the pass
    for r in range(MINE.bank_passes):
        out = out_dir / f"{month}_r{r:02d}.parquet"
        if out.exists():
            continue
        world = world660(clean=clean, month=month, r=r)
        flags = flg.lag_flags(
            own=world.own_active, nbr=world.nbr_active, prim=test.primary, origin_idx=test.origin_idx,
            day_idx=day_idx, week_idx=week_idx, day_missing=test.x[:, 38] > 0.5, week_missing=test.x[:, 41] > 0.5,
        )  # fmt: skip
        masked = np.stack([flags[:, 0] + flags[:, 1] > 0, flags[:, 2] + flags[:, 3] > 0], axis=1)
        masked[:, 0] |= masked[
            :, 1
        ]  # block dropout masks the day lag alone or both lags, never the week lag alone
        blk = e.block_of(
            clean=world.fixed,
            primary=test.primary,
            target_idx=test.target_idx,
            origin_idx=test.origin_idx,
        )
        lag_x = test.x.copy()
        lag_x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
        tin = world.own_active[test.primary, test.target_idx]
        after = ended_within(active=world.own_active, days=AFTER_DAYS)[
            test.primary, test.target_idx
        ]
        phase = phases_of(flags=flags, tin=tin, after=after)
        keep = (
            (phase > 0)
            & np.isfinite(y_all)
            & (test.target_idx < world.stop)
            & (test.target_idx < cutoff_idx)
            & (test.origin >= lo)
            & (test.origin < hi)
        )
        if not keep.any():
            continue
        keys_keep = keys.filter(pl.Series(keep))
        qs: dict[str, list[np.ndarray]] = {"clean": [], "inj": [], "mask": []}
        for seed, model in models.items():
            fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            common = dict(model=model, test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi)
            frames = {
                "clean": clean_frames[seed],
                "inj": forecast_window(**common, power_lag=world.fixed, power_sig=world.fixed, lag_x=lag_x),
                "mask": forecast_window(**common, power_lag=world.fixed, power_sig=world.fixed, lag_x=lag_x, mask_lags=masked),
            }  # fmt: skip
            for name, frame in frames.items():
                qs[name].append(frame_q(frame=frame, keys=keys_keep))
        y = y_all[keep]
        cols: dict[str, np.ndarray] = {
            "prim": test.primary[keep].astype(np.int8), "oi": test.origin_idx[keep].astype(np.int32),
            "vi": test.target_idx[keep].astype(np.int32), "phase": phase[keep],
            "y": y,
            "shift_day": world.shift[test.primary[keep], np.clip(day_idx[keep], 0, None)],
            "shift_week": world.shift[test.primary[keep], np.clip(week_idx[keep], 0, None)],
        }  # fmt: skip
        for j, n in enumerate(flg.FLAG_NAMES):
            cols[n] = flags[keep, j].astype(np.int8)
        for name, per_seed in qs.items():
            mean = np.sort(np.mean(per_seed, axis=0), axis=1)
            for k, v in row_scores(q=mean, y=y).items():
                cols[f"{name}_{k}"] = v.astype(np.float32)
            cols[f"{name}_med"] = mean[:, 5].astype(np.float32)
            for seed, q in zip(models, per_seed, strict=True):
                cols[f"{name}_crps_s{seed}"] = row_scores(q=q, y=y)["crps"].astype(np.float32)
                cols[f"{name}_med_s{seed}"] = q[:, 5].astype(np.float32)
        tmp = out.with_suffix(".tmp")
        pl.DataFrame(cols).write_parquet(tmp, compression="zstd")
        tmp.rename(out)
        in12 = np.isin(phase[keep], (1, 2))
        crps = {n: row_scores(q=np.sort(np.mean(qs[n], axis=0), axis=1), y=y)["crps"] for n in qs}
        gap = (
            float(crps["inj"][in12].mean() - crps["clean"][in12].mean())
            if in12.any()
            else float("nan")
        )
        print(
            f"{month} r{r:02d}: {int(keep.sum()):,} rows ({int(in12.sum()):,} in P1+P2), CRPS(A0) - CRPS(A0-clean) = {gap:+.3f}",
            flush=True,
        )
    print("stage 0 done", month, flush=True)


def head_rows(*, frames: list[pl.DataFrame], prim_of_ts: dict[int, int]) -> dict[str, np.ndarray]:
    """Return (prim, oi, vi, lead, q) of the seed-mean quantiles of the seeds' frames, rows sorted by key.

    `lead` is in hours. Rows keep leads from 3 hours to below 339 hours, the contract's valid times.
    """
    cols = ["time_series_id", "power_fcst_init_time", "valid_time"]
    ordered = [f.sort(cols) for f in frames]
    for f in ordered[1:]:
        assert f.select(cols).equals(ordered[0].select(cols)), (
            "the seeds' frames hold different rows"
        )
    ref = ordered[0]
    prim = np.array([prim_of_ts[int(t)] for t in ref["time_series_id"].to_list()], np.int64)
    oi = ((ref["power_fcst_init_time"].dt.epoch("us") - GRID) // STEP).to_numpy()
    vi = ((ref["valid_time"].dt.epoch("us") - GRID) // STEP).to_numpy()
    lead = ((vi - oi) / 2.0).astype(np.float32)
    q = np.sort(np.mean([f.select(QCOLS).to_numpy() for f in ordered], axis=0), axis=1).astype(
        np.float32
    )
    keep = (lead >= 3.0) & (lead < 339.0)
    return {"prim": prim[keep], "oi": oi[keep], "vi": vi[keep], "lead": lead[keep], "q": q[keep]}


def month_bounds_us(*, month: str) -> tuple[int, int]:
    """Return the first microsecond of `month` and of the next month."""
    y, m = (int(v) for v in month.split("-"))
    nxt = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=UTC)
    return int(datetime(y, m, 1, tzinfo=UTC).timestamp() * 1e6), int(nxt.timestamp() * 1e6)


def run_headbank(*, device: str = "cuda") -> None:
    """The A0 Candidate's forecasts on the prevailing head's injected banks (bank F), no retraining.

    Sources, written to `headbank/<source>.npz` as seed-mean quantile rows:
    `tr_<month>_r<pass>` (experiment 650's training passes, per month of the fold, as experiment 667
    injected them), `held_<month>_r<pass>` (its held-out passes 200 to 203), and `dec_r<pass>` (the
    experiment 660 generator over the fold's whole window, `--bank-passes` passes). The shift enters
    the lag inputs, the signals, and the target (written by the data step).
    """
    import e00650_bank as gen650  # ty: ignore[unresolved-import]
    import e00650_common as c650  # ty: ignore[unresolved-import]

    assert fold.PHASE == "predict", "the bank needs the scored window open"
    flg.known_answer()
    clean = e.c.clean_stack()
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    sites = k578.sites()
    prim_of_ts = dict(
        zip(sites["time_series_id"].to_list(), sites["primary"].to_list(), strict=True)
    )
    (month,) = e.ARGS.months
    mr = e.k.build_month(month=month, clean=clean)
    test = e.build_test(month=month, standardiser=mr.base.standardiser)
    ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
    lo_fold, hi_fold = e.k.window_of(month=month, cadence=e.CADENCE, index=0)
    models = {}
    for seed in e.ARGS.seeds:
        m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
        m.load_state_dict(
            torch.load(Path(MINE.models_from) / f"{month}_d0_s{seed}.pt", map_location=device)
        )
        m.eval()
        models[seed] = m
        fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
    out_dir = e.W / "headbank"
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_of, cap, is_mw = c650.world()
    year_end = int((datetime(2026, 11, 1, tzinfo=UTC).timestamp() * 1e6 - GRID) // STEP)

    def forecast(
        *, fixed: np.ndarray, power_sig: np.ndarray, lo: int, hi: int
    ) -> dict[str, np.ndarray] | None:
        blk = e.block_of(
            clean=fixed,
            primary=test.primary,
            target_idx=test.target_idx,
            origin_idx=test.origin_idx,
        )
        lag_x = test.x.copy()
        lag_x[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = blk
        frames = []
        for seed, model in models.items():
            fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            frame = forecast_window(model=model, test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
                                    power_lag=fixed, power_sig=power_sig, lag_x=lag_x, flags=no_flags(test=test))  # fmt: skip
            if frame is None:
                return None
            frames.append(frame)
        return head_rows(frames=frames, prim_of_ts=prim_of_ts)

    def write(*, name: str, rows: dict[str, np.ndarray]) -> None:
        tmp = out_dir / f"{name}.tmp.npz"
        np.savez(tmp, **rows)
        tmp.rename(out_dir / f"{name}.npz")

    months_of_fold = []
    y, m0 = (int(v) for v in month.split("-"))
    for k in range(fold.SPAN_MONTHS):
        t = m0 - 1 + k
        months_of_fold.append(f"{y + t // 12}-{t % 12 + 1:02d}")
    n = clean.shape[1]
    for sub in months_of_fold:
        ms = c650.month_start_idx(month=sub)
        lo, hi = month_bounds_us(month=sub)
        view = clean.copy()
        # hide every half-hour from the next month's start, as experiment 667's generator view does
        y2, m2 = (int(v) for v in sub.split("-"))
        nxt = c650.month_start_idx(month=f"{y2 + (m2 == 12)}-{m2 % 12 + 1:02d}")
        view[:, nxt:] = np.nan
        for r in (
            *range(gen650.N_TRAIN_PASSES),
            *range(gen650.HELD_OUT_FIRST, gen650.HELD_OUT_FIRST + gen650.N_HELD_OUT_PASSES),
        ):
            kind = "tr" if r < gen650.HELD_OUT_FIRST else "held"
            name = f"{kind}_{sub}_r{r:03d}"
            if (out_dir / f"{name}.npz").exists():
                continue
            own, mirror, _events = gen650.build(clean=view, month_idx=ms, month_start_idx=ms, r=r, nb_of=nb_of,
                                                capacity=cap, is_mw=is_mw, window_end_idx=year_end)  # fmt: skip
            fixed = np.where(np.isfinite(clean), clean + own[:, :n], clean)
            mirrored = np.where(np.isfinite(fixed), fixed + mirror[:, :n], fixed)
            rows = forecast(fixed=fixed, power_sig=mirrored, lo=lo, hi=hi)
            if rows is not None:
                write(name=name, rows=rows)
                print(f"{name}: {len(rows['prim']):,} rows", flush=True)
    for r in range(MINE.bank_passes):
        name = f"dec_r{r:02d}"
        if (out_dir / f"{name}.npz").exists():
            continue
        world = world660(clean=clean, month=month, r=r)
        rows = forecast(fixed=world.fixed, power_sig=world.fixed, lo=lo_fold, hi=hi_fold)
        if rows is not None:
            np.savez(
                out_dir / f"{name}_world.npz",
                fixed=world.fixed.astype(np.float32),
                own_active=world.own_active,
                start=world.start,
                stop=world.stop,
            )
            write(name=name, rows=rows)
            print(f"{name}: {len(rows['prim']):,} rows", flush=True)
    print("headbank done", month, flush=True)


if __name__ == "__main__":
    if CMD in {"table", "check"}:
        e.k.build_month = build_month_syn
        e.k.GpuMonth = SigGpuMonth
        run_prepare(device="cuda", command=CMD)
    elif CMD == "fit":
        run_fit()
    elif CMD == "bank":
        e.k.build_month = build_month_syn
        e.k.GpuMonth = SigGpuMonth
        run_bank()
    elif CMD == "stage0":
        run_stage0()
    elif CMD == "headbank":
        run_headbank()
