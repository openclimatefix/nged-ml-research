# ruff: noqa: ANN001, ANN201, ANN202, D102, D103, D107, E402, E501, I001, PLR0911, PT018
"""Experiment 726: run an injected panel of power through the Candidate's saved weights (GPU).

**Source.** This file is a copy of `run_inject` and its helpers in experiment 667's
`prevailing/p667_inject.py` (last changed at commit 2fcc55c49; read at main `d44dec257`), which is
itself a copy of `e00667_cand.py`. Lines kept as they are: the `e00574_fit` import with the
`sys.argv` hack, `new_model`, `build_month_syn` (without the label bookkeeping, which only the
training phase reads), `FoldSignals.__init__`, `_block`, `_test_rows`, `test_block`,
`fold_signals`, `subset`, and `forecast_window`'s body. Lines changed:

1. **Paths.** Every output lies under this experiment's data folder with the world in the file
   name (`candidate_output`), and `write_parquet_atomic` refuses any other path. `_done` reads those
   paths only, so a file of experiment 667 is never mistaken for a finished job.
2. **Read-only inputs.** Tables, saved weights, step-2 weights, the chain, and the statistics are
   read from the root that `--root` names (`e667` for seeds 0 and 1, `seeds713` for seeds 2 to 4),
   and each file is asserted to exist. `ensure_tables` and every other function that writes a table
   are removed. `CHAIN667` is not read from the environment. The module-level stand-in frames of
   `e00667_fold.install` are redirected by `install_stub_redirect`.
3. **Phase.** `E00667_PHASE=predict` is asserted, because `table_file` of the original reads the
   training-phase tables otherwise.
4. **Panels.** `contaminate_year` is kept for the world `e650`. The new worlds are `zero`,
   `e581_all`, `e581_host`, `bank709`, `g2`, `kc6_one`, and `kc6_set`. Each builder returns the two
   panels that `forecast_window` passes on, `power_lag` for the lag columns and `power_sig` for the
   signal block and the neighbour readings. `power_for` caches on `(world, r, set)`, never on `r`.
5. **Cut.** The test rows of the counted worlds are cut to origin days 0 to 20, lead up to 240
   hours, and the members of the set, at the subset that `test_block` receives. The rows that
   `FoldSignals` checks (`n_test == len(test.x)`) are never cut.
6. **Training-only parts.** The E3 patch, the synthetic-training noise plan, the cross-fit, and the
   label head are not copied: they act on the fine-tune and the tables, which this file never runs.

Usage: python e00726_candidate_panel.py WORLD --root e667|seeds713 --month MONTH --seeds S ... [--passes R ...]
[--sets K ...] [--primary P] [--device cuda]

Run it through `run.sh`, which sets `E00667_PHASE=predict`, the thread limits, and the memory cap.
"""

import argparse
import dataclasses
import os
import pickle
import sys
import time
from datetime import UTC, datetime
from typing import Final

import numpy as np

import e00726_common as com

ARGV = sys.argv[1:]
_p = argparse.ArgumentParser()
_p.add_argument("world", choices=com.CANDIDATE_WORLDS)
_p.add_argument("--root", choices=("e667", "seeds713"), required=True)
_p.add_argument("--month", required=True, choices=com.FOLD_MONTHS)
_p.add_argument("--seeds", nargs="+", type=int, required=True)
_p.add_argument("--passes", nargs="+", type=int, default=None)
_p.add_argument("--sets", nargs="+", type=int, default=None)
_p.add_argument("--primary", type=int, default=0, help="the one shifted Primary of world kc6_one")
_p.add_argument("--device", default="cuda")
MINE = _p.parse_args()
WORLD, MONTH = MINE.world, MINE.month
ROOT = com.candidate_root(tag=MINE.root)
for _seed in MINE.seeds:
    ROOT.require_seed(seed=_seed)
assert os.environ.get("E00667_PHASE") == "predict", "set E00667_PHASE=predict (run.sh does)"
os.environ["E00667_MONTH"] = MONTH

e = com.prime_pipeline(month=MONTH)  # imports e00574_fit with the argv of this fold month
import e00578_common as k578  # ty: ignore[unresolved-import]
import e00598_signals as s598  # ty: ignore[unresolved-import]
import e00613_nbmodel as mymodel  # ty: ignore[unresolved-import]
import e00613_nbsignals as sig  # ty: ignore[unresolved-import]
import e00650_bank as gen650  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00667_fold as fold  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

assert fold.PHASE == "predict", "the Candidate module runs in the prediction phase only"
assert fold.FINAL_MONTH == MONTH, (fold.FINAL_MONTH, MONTH)
torch.set_num_threads(1)

ARM: Final = "P7+D50"
DIR: Final = "P7_D50"
GROUPS = ["gap", "own:ew1,ew3,ew7", "exp", "nb:gap,ew1,ew3,ew7"]
"""The group tokens of arm `P7+D50` (`BASES["P7"]` with no modifier but D50, which only sets dropout)."""
COLUMNS = sig.columns_of(groups=tuple(dict.fromkeys(GROUPS)))
N_EXTRA = len(COLUMNS)
INPUTS = mymodel.InputSet(n_extra=N_EXTRA, bottleneck=False, label_head=False)
assert N_EXTRA > 0
CHAIN = ROOT.chain(month=MONTH)
"""The K1 chain's step-2 weights and statistics (`.../work`), read only."""
e.WORK = ROOT.root / f"{DIR}_{MONTH}"
e.W = e.WORK / "work"
assert e.SPEC.tag == "P5" and e.SPEC.sched == "3sE" and e.SPEC.lags == "24h" and e.SPEC.frozen
LAG_FIRST, LAG_WIDTH = 37, 8
LEAD_OF = {"day": 43, "week": 44}


def _stats_file(*, month: str):
    # the reference month (the month before the fold) reads the fold's own statistics, as experiment 620 did
    p = CHAIN / "stats" / e.SPEC.b1 / f"{fold.FINAL_MONTH}.npz"
    assert p.exists(), p
    return p


def _step2_file(*, month: str, seed: int):
    msg = (
        "step-2 weights are read by the fine-tune and the table builder only, never by this module"
    )
    raise AssertionError(msg)


e.step2_file, e.stats_file = _step2_file, _stats_file
fold.install(e, months=[MONTH], reference_month=fold.PREVIOUS_MONTH)


def new_model(*, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    model = mymodel.PermLagMdn(
        n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, inputs=INPUTS
    )
    model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
    return model


e.new_model = new_model


# ---------------------------------------------------------------- neighbours, grid, training rows


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
NB_READ = NB  # arm `P7+D50` is not the shuffled `-p` arm
GRID, STEP = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
_build_month_p5 = e.k.build_month


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
    print(f"{month}: {n_events} synthetic events", flush=True)
    return mr


GRID_SHAPE = (16, e.data.grid_length())
assert GRID_SHAPE[0] == 16, GRID_SHAPE


def table_file(*, month: str, seed: int):
    """Return the prediction-phase table of (`month`, `seed`): read only, asserted to exist."""
    path = ROOT.table_file(month=month, seed=seed)
    assert path.exists(), path
    return path


class FoldSignals:
    """The extra-column blocks of one fold month and seed (test rows only)."""

    def __init__(self, *, mr, test, clean: np.ndarray, month: str, seed: int) -> None:
        z = np.load(table_file(month=month, seed=seed))
        assert int(z["n_train"]) == len(mr.y) and int(z["n_test"]) == len(test.x), "stale table"
        self.mr, self.test, self.clean, self.month, self.seed = mr, test, clean, month, seed
        self.table = s598.LagFreeTable(median=z["median"], origin_idx=z["origin_idx"])
        self.weather_origin = np.asarray(e.CTX["origin_ft"])

    def _block(self, *, rows: dict, power_lag, power_sig) -> np.ndarray:
        lag_idx = {k: rows["x"][:, col].astype(np.int64) for k, col in LEAD_OF.items()}
        return sig.signal_columns(
            prim=rows["prim"], origin_idx=rows["origin"], day_idx=lag_idx["day"],
            week_idx=lag_idx["week"], day_missing=rows["x"][:, 38] > 0.5,
            week_missing=rows["x"][:, 41] > 0.5, power_lag=power_lag, power_sig=power_sig,
            table=self.table, nb=NB_READ, weather_origin=self.weather_origin, columns=COLUMNS,
        )  # fmt: skip

    def _test_rows(self, *, test=None) -> dict:
        t = test if test is not None else self.test
        return {"x": t.x[:, :45], "prim": t.primary, "origin": t.origin_idx}

    def test_block(self, *, test, power_lag, power_sig) -> np.ndarray:
        """Return a test (sub)set's block from the given power (observed, or the injected world's)."""
        return self._block(
            rows=self._test_rows(test=test), power_lag=power_lag, power_sig=power_sig
        )


FOLD: dict[tuple[str, int], FoldSignals] = {}


def fold_signals(*, mr, test, clean, month: str, seed: int) -> FoldSignals:
    key = (month, seed)
    if key not in FOLD:
        for old in [k for k in FOLD if k[0] != month]:
            del FOLD[old]
        FOLD[key] = FoldSignals(mr=mr, test=test, clean=clean, month=month, seed=seed)
    return FOLD[key]


def subset(*, test, sel: np.ndarray):
    """Return the test rows `sel` (a boolean mask) as a new `TestRows`."""
    return dataclasses.replace(
        test, **{f.name: getattr(test, f.name)[sel] for f in dataclasses.fields(test)}
    )


def forecast_window(
    *, model, test, mr, device: str, fs: FoldSignals, lo: int, hi: int, power_lag, power_sig,
    lag_x, keep: np.ndarray,
):  # fmt: skip
    """Return the forecast frame of the kept origins in [lo, hi), with the extra columns built from the given power.

    `keep` is the cut of the scorer's rows. It is applied here, to the subset that `test_block`
    receives, and never to `test`, whose length `FoldSignals` checks against the stored table.
    """
    sel = (test.origin >= lo) & (test.origin < hi) & keep
    if not sel.any():
        return None
    sub = subset(test=test, sel=sel)
    sub = dataclasses.replace(sub, x=lag_x[sel])
    block = fs.test_block(test=sub, power_lag=power_lag, power_sig=power_sig)
    sub = dataclasses.replace(sub, x=np.concatenate([sub.x, block], axis=1))
    model.table = e.CTX["table_ft"]
    return e._window_frame(
        model=model, test=sub, mr=mr, lo=lo, hi=hi, flag=False, device=device
    )  # fmt: skip


# ---------------------------------------------------------------- the worlds' panels

YEAR_END_IDX: Final = int(
    (datetime(2026, 11, 1, tzinfo=UTC) - k578.GRID_START).total_seconds() // 1800
)
"""Experiment 650's events are clipped at the end of the backtest year, as in experiment 667."""


def generator_view(*, clean: np.ndarray, month: str) -> np.ndarray:
    """Return `clean` with every half-hour from the next month's start hidden (as `p667_data.generator_view`)."""
    y, m = (int(v) for v in month.split("-"))
    out = clean.copy()
    out[:, c650.month_start_idx(month=f"{y + (m == 12)}-{m % 12 + 1:02d}") :] = np.nan
    return out


def contaminate_year(*, clean: np.ndarray, month: str, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (`clean` plus pass `r`'s own shifts, that plus the receivers' mirrored shifts); experiment 667's world."""
    nb_of, cap, is_mw = c650.world()
    ms = c650.month_start_idx(month=month)
    own, mirror, _events = gen650.build(
        clean=generator_view(clean=clean, month=month),
        month_idx=ms,
        month_start_idx=ms,
        r=r,
        nb_of=nb_of,
        capacity=cap,
        is_mw=is_mw,
        window_end_idx=YEAR_END_IDX,
    )
    n = clean.shape[1]
    fixed = np.where(np.isfinite(clean), clean + own[:, :n], clean)
    return fixed, np.where(np.isfinite(fixed), fixed + mirror[:, :n], fixed)


def bank709_panel(*, clean: np.ndarray, month: str, index: int) -> np.ndarray:
    """Return experiment 709's bank panel of pass `index`, as `e00709_cand.run_bank` builds it: the cleaned power plus the pass's change."""
    path = com.folder_of(prefix="00709") / "worlds" / f"bank_{month}_p{index}.pkl"
    with path.open("rb") as handle:
        parts = pickle.load(handle)  # our own stored bank
    delta = parts["delta"]
    out = clean.copy()
    out[:, : delta.shape[1]] += delta
    return out


def g2_panel(*, clean: np.ndarray, month: str, index: int) -> tuple[np.ndarray, list[int]]:
    """Return experiment 714's second-generator panel and its touched Primaries.

    A verbatim copy of F4's `g2_panel` (`e00715_era.py` at commit `ac3e768e4`), whose import chain
    loads the MSA network and cannot share a process with the Candidate's `e00574_fit` argv. The
    preflight compares the key lines with the pinned file.
    """
    import e00714_generator2 as g2  # ty: ignore[unresolved-import]

    gen = g2.read_pass(world="g2", month=month, index=index)
    out = clean.astype(np.float64).copy()
    lo, n = gen.power.start_hh, gen.power.power.shape[1]
    assert gen.power.power.shape[0] == clean.shape[0] and lo + n <= clean.shape[1], (
        lo,
        n,
        clean.shape,
    )
    out[:, lo : lo + n] = gen.power.power
    return out, list(gen.touched)


@dataclasses.dataclass
class Panels:
    """What one (world, pass, set) job reads: the two panels, the lag block rows, and the cut."""

    power_lag: np.ndarray
    power_sig: np.ndarray
    keep: np.ndarray
    x: np.ndarray


class PanelBuilder:
    """Builds and caches the panels of one job key `(world, r, set)`; the cache holds one entry."""

    def __init__(self, *, clean: np.ndarray, test, month: str) -> None:
        self.clean, self.test, self.month = clean, test, month
        self.month_idx = c650.month_start_idx(month=month)
        self.sets = com.read_sets() if WORLD in {"e581_host", "kc6_set"} else []
        self._key: tuple | None = None
        self._value: Panels | None = None

    def panels(
        self, *, r: int, k: int | None
    ) -> tuple[np.ndarray, np.ndarray, set[int] | None, bool]:
        """Return (power_lag, power_sig, members the cut keeps or None, whether the rows are cut)."""
        c, ms = self.clean, self.month_idx
        if WORLD == "zero":
            return c, c, None, False
        if WORLD == "e581_all":
            p = com.all_shifted_panel(clean=c, month_start_idx=ms, r=r)
            return p, p, None, True
        if WORLD in {"e581_host", "kc6_set"}:
            assert k is not None
            members = set(self.sets[k])
            p = com.host_panel(clean=c, month_start_idx=ms, r=r, members=members)
            return p, p, members, WORLD == "e581_host"
        if WORLD == "kc6_one":
            p = com.host_panel(clean=c, month_start_idx=ms, r=r, members={MINE.primary})
            return p, p, None, False
        if WORLD == "e650":
            fixed, mirrored = contaminate_year(clean=c, month=self.month, r=r)
            return fixed, mirrored, None, False
        if WORLD == "bank709":
            p = bank709_panel(clean=c, month=self.month, index=r)
            return p, p, None, False
        assert WORLD == "g2", WORLD
        p, touched = g2_panel(clean=c, month=self.month, index=r)
        return p, p, set(touched), False

    def get(self, *, r: int, k: int | None) -> Panels:
        key = (WORLD, r, k)
        if key != self._key:
            self._value = None  # free the old panel before building the next one
            lag, sg, members, cut = self.panels(r=r, k=k)
            t = self.test
            if cut:
                keep = com.cut_mask(
                    origin_idx=t.origin_idx, target_idx=t.target_idx, primary=t.primary,
                    month_start_idx=self.month_idx, members=members,
                )  # fmt: skip
            elif WORLD == "g2":
                keep = np.isin(t.primary, sorted(members or ()))
            else:
                keep = np.ones(len(t.primary), bool)
            rows = np.flatnonzero(keep)
            x = t.x.copy()
            x[rows, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = e.block_of(
                clean=lag, primary=t.primary[rows], target_idx=t.target_idx[rows],
                origin_idx=t.origin_idx[rows],
            )  # fmt: skip
            assert np.array_equal(x[rows][:, 43:45], t.x[rows][:, 43:45]), (
                "the shifted panel moved a lag time"
            )
            self._key, self._value = key, Panels(power_lag=lag, power_sig=sg, keep=keep, x=x)
        assert self._value is not None
        return self._value


def kc5(*, panels: Panels, test, clean: np.ndarray) -> None:
    """Check KC5: the day-old lag column moves where the lag index lies in a shifted window, and nowhere else.

    The lag-time columns (43 and 44) are asserted unchanged in `PanelBuilder.get`.
    """
    rows = np.flatnonzero(panels.keep)
    lag_idx = test.x[rows, LEAD_OF["day"]].astype(np.int64)
    ok = (test.x[rows, 38] < 0.5) & (lag_idx >= 0)
    prim = test.primary[rows]
    in_window = np.zeros(len(rows), bool)
    in_window[ok] = (
        np.nan_to_num(panels.power_lag[prim[ok], lag_idx[ok]] - clean[prim[ok], lag_idx[ok]]) != 0
    )
    base = e.block_of(
        clean=clean, primary=prim, target_idx=test.target_idx[rows], origin_idx=test.origin_idx[rows]
    )  # fmt: skip
    moved = np.abs(panels.x[rows, LAG_FIRST] - base[:, 0]) > 1e-9
    stray = int((moved & ok & ~in_window).sum())
    n_in = int((in_window & ok).sum())
    caught = int((moved & in_window & ok).sum())
    passed = stray == 0 and (n_in == 0 or caught / n_in > 0.9)
    print(
        f"KC5 {'PASS' if passed else 'FAIL'} (rows moved outside a window: {stray}; rows inside: {n_in})",
        flush=True,
    )
    assert passed, "KC5: the Candidate's lag column does not read the shifted panel"


# ---------------------------------------------------------------- the jobs


def default_passes() -> list[int]:
    if MINE.passes is not None:
        return MINE.passes
    if WORLD in {"e581_host", "e581_all"}:
        return list(range(com.N_PASSES))
    if WORLD == "g2":
        import e00714_generator2 as g2  # ty: ignore[unresolved-import]

        return g2.stored_indices(world="g2", month=MONTH)
    msg = f"world {WORLD} needs --passes"
    raise SystemExit(msg)


def job_list() -> list[tuple[int, int | None, int]]:
    """Return the (pass, set, seed) jobs of this process that are not finished."""
    passes = default_passes()
    if WORLD in {"e581_host", "kc6_set"}:
        n_sets = len(com.read_sets())
        sets: list[int | None] = list(MINE.sets) if MINE.sets is not None else list(range(n_sets))
        assert all(0 <= k < n_sets for k in sets if k is not None)
        if WORLD == "kc6_set":
            assert MINE.sets is not None and len(MINE.sets) == 1, "kc6_set needs one --sets value"
    else:
        sets = [None]
    return [
        (r, k, seed)
        for seed in MINE.seeds
        for r in passes
        for k in sets
        if not com.job_done(out=com.candidate_output(world=WORLD, month=MONTH, r=r, seed=seed, k=k))
    ]


def run_jobs(*, device: str) -> None:
    com.cap_gpu(device=device)
    todo = job_list()
    print(f"{WORLD} {MONTH} root {MINE.root}: {len(todo)} jobs to do", flush=True)
    if not todo:
        return
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, (clean.shape, GRID_SHAPE)
    e.k.build_month = build_month_syn
    cuts = e.k.cutoffs(month=MONTH, cadence=e.CADENCE)
    assert len(cuts) == 1, "the Candidate has one cutoff per month, at day 0"
    day = e.f.day_of(month=MONTH, cut=cuts[0])
    assert day == 0, day
    lo, hi = e.k.window_of(month=MONTH, cadence=e.CADENCE, index=0)
    for seed in MINE.seeds:
        mfile = ROOT.model_file(month=MONTH, seed=seed)
        assert mfile.exists(), mfile
        assert e.model_file(tag=e.SPEC.tag, month=MONTH, day=day, seed=seed) == mfile, (
            "e.model_file and the root disagree about the weights"
        )
        table_file(month=MONTH, seed=seed)
    mr = e.k.build_month(month=MONTH, clean=clean)
    test = e.build_test(month=MONTH, standardiser=mr.base.standardiser)
    e.CTX["variants"] = {}
    e.CTX["on"] = False
    builder = PanelBuilder(clean=clean, test=test, month=MONTH)
    models: dict[int, torch.nn.Module] = {}
    checked = False
    for r, k, seed in todo:
        started = time.monotonic()
        if seed not in models:
            m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
            m.load_state_dict(
                torch.load(ROOT.model_file(month=MONTH, seed=seed), map_location=device)
            )
            m.eval()
            models[seed] = m
        fs = fold_signals(mr=mr, test=test, clean=clean, month=MONTH, seed=seed)
        panels = builder.get(r=r, k=k)
        if not checked and WORLD in {"e581_all", "e581_host", "kc6_set", "kc6_one"}:
            kc5(panels=panels, test=test, clean=clean)
            checked = True
        frame = forecast_window(
            model=models[seed], test=test, mr=mr, device=device, fs=fs, lo=lo, hi=hi,
            power_lag=panels.power_lag, power_sig=panels.power_sig, lag_x=panels.x, keep=panels.keep,
        )  # fmt: skip
        target = com.candidate_output(world=WORLD, month=MONTH, r=r, seed=seed, k=k)
        if frame is None:
            com.touch_marker(path=target.with_suffix(".empty"))
        else:
            com.write_parquet_atomic(frame=frame, path=target)
        print(
            f"{WORLD} {MONTH} r{r:02d} k{k} s{seed}: {time.monotonic() - started:.1f} s", flush=True
        )
    com.write_json_atomic(
        obj=com.job_record(
            extra={
                "world": WORLD,
                "root": MINE.root,
                "month": MONTH,
                "seeds": MINE.seeds,
                "jobs": len(todo),
                "argv": ARGV,
            },
        ),
        path=com.data_root() / "logs" / f"candidate_{WORLD}_{MINE.root}_{MONTH}_{os.getpid()}.json",
    )
    del mr, test, models
    FOLD.clear()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_jobs(device=MINE.device)
