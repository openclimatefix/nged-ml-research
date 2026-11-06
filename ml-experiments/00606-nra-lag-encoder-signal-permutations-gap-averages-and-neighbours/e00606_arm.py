# ruff: noqa: D107, ANN001, ANN201, E402, PT018, B023, F821, ANN202, D102, D103, D205, E501, PLR0915
"""Experiment 606: fit and inject the permutation arms P0 to P7c (GPU).

A wrapper around experiment 574's `e00574_fit.py`, as experiment 598's `e00598_arm.py` is, which this
file copies and changes. The rows, recency weights, early stopping, seeds, scoring windows, origin-time
weather assertions, refit days 0 and 14, the embargoed daily fine-tune with synthetic lag-only
load-shifts, and the A1 chain's step-2 weights are experiment 591's A1. What is new is the network
(`e00606_model.PermLagMdn`), whose lag encoder drops some of A1's inputs and reads extra columns of
`e00606_signals`, chosen per arm (see `ARMS`).

The table `T` is experiment 598's (read from its folder when built there, else built here). The extra
columns of the training rows read the synthetic shift in the Primary's own lags and, for the neighbour
columns, the same shift mirrored onto one neighbour. At test time they read observed power (or the
injected bank's contaminated power), by the same code path.

Usage: python e00606_arm.py ARM COMMAND [--months ...] [--seeds ...] [--passes ...] [--work-suffix X]

- ARM: a key of `ARMS`, optionally followed by `-p` (the same arm with its extra columns replaced by
  a
  column-wise resample of the training block, the placebo).
- COMMAND: `table`, `check`, `fit`, `inject`, `assemble`, as in experiment 598.
"""

import argparse
import dataclasses
import json
import sys

_p = argparse.ArgumentParser()
_p.add_argument("arm")
_p.add_argument("command", choices=("table", "check", "step2", "fit", "inject", "assemble"))
_p.add_argument(
    "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
_p.add_argument("--work-suffix", default="")
_p.add_argument("--groups", default=None, help="comma-separated column groups for an ad hoc arm")
MINE = _p.parse_args()
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
import e00606_model as mymodel  # ty: ignore[unresolved-import]
import e00606_signals as sig  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
PLACEBO = ARM.endswith(("-p", "-p-q"))
PAIRED = ARM.endswith("-q")
"""`-p`: the extra columns are replaced by a resample of the training block. `-q`: the paired perturbation
(a shift at one Primary and a smaller opposite shift at a neighbour) in step 2 and step 3."""
BASE_ARM = ARM.removesuffix("-q").removesuffix("-p")
PAIR_SHARE = 0.5
"""A paired shift raises the mirrored neighbour's power by this share of the megawatts the Primary lost."""
DIR = ARM.replace("-", "")
TABLES = ROOT / "tables"
TABLES_598 = (
    ML_EXPERIMENTS_DATA_DIR / "00598-nra-gate-against-signals-as-inputs-at-equal-band-width/tables"
)
"""One `T_<month>_s<seed>.npz` per fold month and seed, shared by the arms; experiment 598's is read first."""

ARMS: dict[str, tuple[mymodel.InputSet, tuple[str, ...]]] = {
    "P0": (mymodel.InputSet(use_expected=False), ()),
    "P1": (mymodel.InputSet(use_age=False), ()),
    "P2": (mymodel.InputSet(use_id=False), ()),
    "P3": (mymodel.InputSet(), ("gap", "exp")),
    "P4": (mymodel.InputSet(), ("gap", "sustain", "exp")),
    "P5": (mymodel.InputSet(), ("gap", "ew", "exp")),
    "P6": (mymodel.InputSet(), ("gap", "ew", "sustain", "exp")),
    "P7": (mymodel.InputSet(), ("gap", "ew", "exp", "nb")),
    "P7b": (mymodel.InputSet(), ("gap", "ew", "exp", "nb_nosum")),
    "P7c": (mymodel.InputSet(), ("gap", "ew", "exp", "bn")),
}
"""Per arm: the lag encoder's kept inputs and the column groups appended. P1 and P2 drop an input and add none."""
if MINE.groups is not None:  # an ad hoc arm, e.g. P8
    ARMS[BASE_ARM] = (mymodel.InputSet(), tuple(MINE.groups.split(",")))
INPUTS, GROUPS = ARMS[BASE_ARM]
COLUMNS = sig.columns_of(groups=GROUPS) if GROUPS else []
N_EXTRA = len(COLUMNS)
INPUTS = mymodel.InputSet(
    use_id=INPUTS.use_id, use_age=INPUTS.use_age, use_expected=INPUTS.use_expected, n_extra=N_EXTRA
)
CHAIN = (
    ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work"
)
"""A1's pre-training and step-2 weights (experiment 591), read only."""
W574 = e.W
e.WORK = ROOT / f"{DIR}{MINE.work_suffix}"
e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen
LAG_FIRST, LAG_WIDTH = 37, 8
LEAD_OF = {"day": 43, "week": 44}


def _step2_file(*, month: str, seed: int):
    """The A1 chain's step-2 weights; a paired arm uses (and trains) its own."""
    p = (e.W if PAIRED else CHAIN) / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    assert (PAIRED and CMD == "step2") or p.exists(), p
    return p


def _a1_step2_file(*, month: str, seed: int):
    """The A1 chain's step-2 weights, from which `T` is built."""
    p = CHAIN / "step2" / e.SPEC.tag / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def _pretrained_file(*, month: str, seed: int):
    p = CHAIN / "pretrained_P2" / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def _stats_file(*, month: str):
    p = W574 / "stats" / e.SPEC.b1 / f"{month}.npz"
    assert p.exists(), p
    return p


e.step2_file, e.stats_file, e.pretrained_file = _step2_file, _stats_file, _pretrained_file


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
GRID, STEP = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
SYN: dict[int, tuple[np.ndarray, np.ndarray]] = {}
"""Per month-rows object: the synthetic power (16, grid) and the shift added to it (0 where none)."""
_build_month_p5 = e.k.build_month


def origin_idx_of(*, origin_us: np.ndarray) -> np.ndarray:
    return ((origin_us - GRID) // STEP).astype(np.int64)


def build_month_syn(*, month, clean):
    """Build the month's P5 rows with experiment 591's synthetic lag-only load-shifts in the lag columns.

    The lag columns are exactly A1's. The synthetic power and its shift are kept for the signal block.
    """
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = origin_idx_of(origin_us=mr.origin_us)
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    fixed, n_events = bank.synthetic_clean(
        clean=clean, first=first, last=last, seed=int(month.replace("-", ""))
    )
    if PAIRED:  # the smaller opposite shift at one neighbour enters that neighbour's own lags too
        own_shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0)
        choice = s598.mirror_choice(neighbours=NB.of, month=month)
        fixed = s598.mirrored_power(
            base=fixed, shift=PAIR_SHARE * own_shift, choice=choice, capacity=NB.capacity
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


# ---------------------------------------------------------------- the table T and the row blocks


def table_file(*, month: str, seed: int):
    theirs = TABLES_598 / f"T_{month}_s{seed}.npz"
    return theirs if theirs.exists() else TABLES / f"T_{month}_s{seed}.npz"


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
        model.load_state_dict(
            torch.load(_a1_step2_file(month=month, seed=seed), map_location=device)
        )
        model.table = e.CTX["table_ft"]
        q_mr = s598.masked_quantiles(
            model=model, x=mr.x, primary=mr.primary, target_mean=mr.base.target_mean,
            target_std=mr.base.target_std, device=device,
        )  # fmt: skip
        q_te = s598.masked_quantiles(
            model=model, x=test.x, primary=test.primary, target_mean=mr.base.target_mean,
            target_std=mr.base.target_std, device=device,
        )  # fmt: skip
        n_grid = GRID_SHAPE[1]
        table = s598.build_lag_free_table(
            prim=np.concatenate([mr.primary, test.primary]),
            origin_idx=np.concatenate([origin_idx_of(origin_us=mr.origin_us), test.origin_idx]),
            target_idx=np.concatenate([mr.target_idx, test.target_idx]),
            median=np.concatenate([q_mr[:, 1], q_te[:, 1]]),
            n_primaries=len(mr.base.target_mean), n_grid=n_grid,
        )  # fmt: skip
        filled = table.check_no_lookahead()
        TABLES.mkdir(parents=True, exist_ok=True)
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


GRID_SHAPE = (16, 121248)


class FoldSignals:
    """The extra-column blocks of one fold month and seed: the training rows (with synthetic shifts)
    and any test power.
    """

    def __init__(self, *, mr, test, clean: np.ndarray, month: str, seed: int) -> None:
        z = np.load(table_file(month=month, seed=seed))
        assert int(z["n_train"]) == len(mr.y) and int(z["n_test"]) == len(test.x), "stale table"
        self.mr, self.test, self.clean, self.month, self.seed = mr, test, clean, month, seed
        self.table = s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
        self.fixed, self.shift = SYN[id(mr)]
        self.choice = s598.mirror_choice(neighbours=NB.of, month=month)
        self.power_sig = (
            self.fixed
            if PAIRED  # the paired shift is already in the lags of the Primary and its neighbour
            else s598.mirrored_power(
                base=self.fixed, shift=self.shift, choice=self.choice, capacity=NB.capacity
            )
        )
        self.weather_origin = np.asarray(e.CTX["origin_ft"])
        self._train: np.ndarray | None = None
        self._source: np.ndarray | None = None
        self.report: dict = {}

    def _block(self, *, rows: dict, power_lag, power_sig) -> np.ndarray:
        lag_idx = {k: rows["x"][:, col].astype(np.int64) for k, col in LEAD_OF.items()}
        return sig.signal_columns(
            prim=rows["prim"], origin_idx=rows["origin"], day_idx=lag_idx["day"],
            week_idx=lag_idx["week"], day_missing=rows["x"][:, 38] > 0.5,
            week_missing=rows["x"][:, 41] > 0.5, power_lag=power_lag, power_sig=power_sig,
            table=self.table, nb=NB, weather_origin=self.weather_origin, columns=COLUMNS,
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
        """Return the training rows' block: synthetic shifts in the own and neighbour power, training
        noise where the training-to-test spread ratio is low, noise for the placebo. Cached.
        """
        if self._train is not None:
            return self._train
        synth = self._block(rows=self._train_rows(), power_lag=self.fixed, power_sig=self.power_sig)
        real_test = self._block(rows=self._test_rows(), power_lag=self.clean, power_sig=self.clean)
        plan = self.noise_plan(synth=synth, real_test=real_test)
        sig.add_training_noise(block=synth, prim=self.mr.primary, plan=plan, seed=self.seed)
        del real_test
        assert synth.shape[0] == len(self.mr.x), "training block and month rows differ"
        self._source = synth
        if PLACEBO:
            synth = s598.placebo_block(source=self._source, n=len(synth), seed=self.seed)
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
        if PLACEBO:
            self.train_block()
        block = self._block(
            rows=self._test_rows(test=test), power_lag=power_lag, power_sig=power_sig
        )
        if PLACEBO:
            assert self._source is not None
            block = s598.placebo_block(
                source=self._source, n=len(block),
                seed=self.seed + 1_000 + int(test.origin_idx.min()) % 997,
            )  # fmt: skip
        return block


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
            return t
        fs = FOLD[(self.mr.month, e.CTX["seed"])]
        b = fs.train_block()[refit.index].copy()
        assert len(b) == len(t["x"]), "extra block and refit tensors differ in rows"
        if not PLACEBO:
            sig.mask_lag_columns(
                block=b, columns=COLUMNS, day_masked=(t["x"][:, 38] > 0.5).cpu().numpy(),
                week_masked=(t["x"][:, 41] > 0.5).cpu().numpy(),
            )  # fmt: skip
        t["x"] = torch.cat([t["x"], torch.from_numpy(b).to(self.device)], dim=1)
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
):
    """Return the forecast frame of origins in [lo, hi) with the extra columns built from the given power."""
    sel = (test.origin >= lo) & (test.origin < hi)
    if not sel.any():
        return None
    sub = subset(test=test, sel=sel)
    if lag_x is not None:
        sub = dataclasses.replace(sub, x=lag_x[sel])
    if fs is not None:
        block = fs.test_block(test=sub, power_lag=power_lag, power_sig=power_sig)
        sub = dataclasses.replace(sub, x=np.concatenate([sub.x, block], axis=1))
    model.table = e.CTX["table_ft"]
    return e._window_frame(
        model=model, test=sub, mr=mr, lo=lo, hi=hi, flag=False, device=device
    )  # fmt: skip


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
    """Shift one Primary's real power by a known amount for 10 days (mirrored onto its neighbours)
    and check the columns.
    """
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
            if not PLACEBO and target == "gap_day":
                assert abs(v - shift) < 0.05, v
            if not PLACEBO and target == "own_plus_nb_gap_day" and NB.of[p]:
                assert abs(v) < 0.05, v  # a pure transfer cancels in the sum


def run_fit(*, device: str = "cuda") -> None:
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, clean.shape
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        mr = e.k.build_month(month=month, clean=clean)
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
                    model, log = e.fine_tune(g=g, month=month, cut=cut, seed=seed, device=device)
                    mfile.parent.mkdir(parents=True, exist_ok=True)
                    tmp = mfile.with_suffix(".tmp")
                    torch.save(model.state_dict(), tmp)
                    tmp.rename(mfile)
                lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
                fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
                frame = forecast_window(
                    model=model, test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
                    power_lag=clean, power_sig=clean,
                )  # fmt: skip
                if frame is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_suffix(".empty").touch()
                else:
                    e.atomic_frame(frame=frame, target=out)
                log["seconds"] = round(e.time.monotonic() - started, 2)
                log["signal_report"] = fs.report if fs is not None else {}
                e.c.write_json(
                    obj=log, path=e.W / "logs" / e.SPEC.tag / f"{month}_d{day}_s{seed}.json"
                )
                print(
                    f"{ARM} {log['name']}: {log.get('epochs', 'reused')} epochs, {log['seconds']} s",
                    flush=True,
                )
                del model
        del g, mr, test
        FOLD.clear()
        torch.cuda.empty_cache()
    out = e.WORK / f"checks_fit_{DIR}_s{'-'.join(map(str, e.ARGS.seeds))}.json"
    out.write_text(json.dumps(e.CHECK, indent=1, default=str))
    print("CHECKS PASSED", {a: v for a, v in e.CHECK.items() if a != "months"})


def run_inject(*, device: str = "cuda") -> None:
    """Forecast every pass of the injected bank with the saved weights; one parquet per cell.

    The lag columns are experiment 584's contaminated lags, exactly as for A1. The signals read the
    contaminated power; the power behind the fresh and neighbour readings also carries each event
    mirrored onto one neighbour, `nbs[(r + p) % len(nbs)]`, as experiment 597's G4 sees it.
    """
    clean = e.c.clean_stack()
    e.k.build_month = build_month_syn
    passes = MINE.passes if MINE.passes is not None else list(range(bank.N_PASSES))
    for month in e.ARGS.months:
        out_dir = e.W / "inject"
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        todo = [
            (r, i, seed)
            for r in passes
            for i in range(len(cuts))
            for seed in e.ARGS.seeds
            if not _done(out=_out(out_dir, month, r, e.f.day_of(month=month, cut=cuts[i]), seed))
        ]
        if not todo:
            continue
        mr = e.k.build_month(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        e.CTX["variants"] = {}
        e.CTX["on"] = False
        if N_EXTRA:
            ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
        start_idx = (e.c.us(t=e.c.month_start(month=month)) - GRID) // STEP
        sl = slice(LAG_FIRST, LAG_FIRST + LAG_WIDTH)
        cache: dict[int, tuple] = {}

        def power_for(r: int):
            if r not in cache:
                cache.clear()
                fixed, _ = bank.contaminate(clean=clean, month_start_idx=start_idx, r=r)
                shift = np.where(np.isfinite(fixed) & np.isfinite(clean), fixed - clean, 0.0)
                choice = {p: qs[(r + p) % len(qs)] for p, qs in NB.of.items() if qs}  # the 597 rule
                mirrored = s598.mirrored_power(
                    base=fixed, shift=shift, choice=choice, capacity=NB.capacity
                )  # fmt: skip
                blk = e.block_of(
                    clean=fixed, primary=test.primary, target_idx=test.target_idx,
                    origin_idx=test.origin_idx,
                )  # fmt: skip
                x = test.x.copy()
                x[:, sl] = blk
                assert np.array_equal(x[:, 43:45], test.x[:, 43:45]), (
                    "contamination moved a lag time"
                )
                cache[r] = (fixed, mirrored, x)
            return cache[r]

        models: dict[tuple[int, int], torch.nn.Module] = {}
        for r, i, seed in todo:
            cut = cuts[i]
            day = e.f.day_of(month=month, cut=cut)
            if (i, seed) not in models:
                m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                m.load_state_dict(
                    torch.load(
                        e.model_file(tag=e.SPEC.tag, month=month, day=day, seed=seed),
                        map_location=device,
                    )
                )
                m.eval()
                models[(i, seed)] = m
            fs = fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            lo, hi = e.k.window_of(month=month, cadence=e.CADENCE, index=i)
            fixed, mirrored, x = power_for(r)
            frame = forecast_window(
                model=models[(i, seed)], test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
                power_lag=fixed, power_sig=mirrored, lag_x=x,
            )  # fmt: skip
            target = _out(out_dir, month, r, day, seed)
            if frame is None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix(".empty").touch()
            else:
                e.atomic_frame(frame=frame, target=target)
            print(f"{ARM} inject {month} r{r:02d} d{day} s{seed}", flush=True)
        del mr, test, models
        FOLD.clear()
        torch.cuda.empty_cache()


def run_assemble() -> None:
    """Concatenate each cutoff's window forecast into one table per month and seed."""
    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        for seed in e.ARGS.seeds:
            parts = [
                e.W
                / "fits"
                / e.SPEC.tag
                / f"{month}_d{e.f.day_of(month=month, cut=cut)}_s{seed}.parquet"
                for cut in cuts
            ]
            assert all(p.exists() or p.with_suffix(".empty").exists() for p in parts), (month, seed)
            frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
            e.atomic_frame(
                frame=frame, target=e.W / "seed_preds" / e.SPEC.tag / f"{month}_s{seed}.parquet"
            )
            print("wrote", month, seed, frame.height, flush=True)


def run_step2(*, device: str = "cuda") -> None:
    """Train this arm's own step 2 (the paired arms): experiment 574's `step2` with the paired lags
    and the extra columns.
    """
    assert PAIRED, "only a paired arm trains its own step 2"
    clean = e.c.clean_stack()
    for month in e.ARGS.months:
        todo = [sd for sd in e.ARGS.seeds if not e.step2_file(month=month, seed=sd).exists()]
        if not todo:
            continue
        mr = e.k.build_month(month=month, clean=clean)
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        ensure_tables(mr=mr, test=test, month=month, seeds=todo, device=device)
        g = SigGpuMonth(mr=mr, device=device)
        cut = e.c.month_start(month=month)
        refit = e.c.select_rows(rows=mr, cutoff=cut, half_life_days=e.UNIFORM_HALF_LIFE)
        assert (
            refit.report["rows_past_valid_cut"] == 0 and refit.report["rows_past_origin_cut"] == 0
        )
        refit = e.embargoed(mr=mr, refit=refit, cut=cut)
        for seed in todo:
            started = e.time.monotonic()
            fold_signals(mr=mr, test=test, clean=clean, month=month, seed=seed)
            e.CTX.update(on=False, shift=None, sign=0, block=None, seed=seed, cut=cut)
            t = g.refit_tensors(refit=refit)
            assert float(t["w"].min()) > 0.999, "step 2 must have no recency weighting"
            model = new_model(n_primaries=len(mr.base.target_mean)).to(device)
            e.load_pretrained_fresh(model=model, path=e.pretrained_file(month=month, seed=seed))
            model.to(device)
            epochs, best = e.train(
                model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                lr=e.SPEC.lr, max_epochs=e.SPEC.max_epochs, fresh_lr=e.FRESH_LR,
                frozen=e.SPEC.frozen, anchor=None, lam=0.0, device=device,
            )  # fmt: skip
            target = e.step2_file(month=month, seed=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
            latest = e.datetime.fromtimestamp(mr.valid_us[refit.index].max() / 1e6, e.UTC)
            assert latest <= cut - e.c.regimes.REFIT_CUTOFF_AGE, (latest, cut)
            print(
                f"step 2 ({ARM}) {month} s{seed}: {len(refit.index):,} rows, {epochs} epochs, "
                f"val NLL {best:.4f}, {e.time.monotonic() - started:.0f} s; latest valid {latest}",
                flush=True,
            )  # fmt: skip
            del t, model
        FOLD.clear()
        del g, mr, test
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if CMD in {"table", "check"}:
        e.k.build_month = build_month_syn
        e.k.GpuMonth = SigGpuMonth
        run_prepare(device="cuda", command=CMD)
    elif CMD == "step2":
        e.k.build_month = build_month_syn
        e.k.GpuMonth = SigGpuMonth
        run_step2()
    elif CMD == "fit":
        run_fit()
    elif CMD == "inject":
        run_inject()
    else:
        run_assemble()
