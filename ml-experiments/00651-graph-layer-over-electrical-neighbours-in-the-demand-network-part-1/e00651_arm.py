# ruff: noqa: D107, ANN001, ANN201, E402, PT018, B023, F821, ANN202, D102, D103, D205, E501, PLR0915, ANN003
"""Experiment 651 (copy of 613): fit and inject the graph-layer arms (GPU).

New in 651: the arm names `G0+D50+GCN`, `+GAT`, `+GATR` (rewired graph), `+GATF` (all Primaries) and
`+FLOW`; the neighbour bank (`x_msg`, the rows as their neighbours see them, with a mirrored shift in
the lags and signals); the injected banks M, S, and X. Everything else is experiment 613's.

Experiment 613 (copy of 610): fit and inject the neighbour-summary arms (GPU).

A wrapper around experiment 574's `e00574_fit.py`, as experiment 598's `e00598_arm.py` and experiment
606's `e00606_arm.py` are, which this file copies and changes. The rows, recency weights, early
stopping, seeds, scoring windows, origin-time weather assertions, refit days 0 and 14, the embargoed
daily fine-tune with synthetic lag-only load-shifts, and the A1 chain's step-2 weights are experiment
591's A1. What is new is the choice of signal columns (`e00613_nbsignals`) and the changes to the
training of the lag encoder (`MODIFIERS`).

An arm name is a stage 1 base followed by modifiers joined with `+`, for example `B7+CF+D30`, and may
end in `-p` (the placebo: each Primary reads another Primary's neighbours, a fixed derangement).

Usage: python e00613_nbarm.py ARM COMMAND [--months ...] [--seeds ...] [--passes ...]
[--work-suffix X]

- COMMAND: `table`, `check`, `fit`, `inject`, `assemble`.
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Final

_p = argparse.ArgumentParser()
_p.add_argument("arm")
_p.add_argument("command", choices=("table", "check", "fit", "inject", "assemble"))
_p.add_argument(
    "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
)
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
_p.add_argument("--passes", nargs="+", type=int, default=None)
_p.add_argument("--work-suffix", default="")
_p.add_argument("--e3", type=int, default=1)
_p.add_argument("--bank", choices=("M", "S", "X"), default="M")
_p.add_argument(
    "--weights-root",
    default=None,
    help="read the saved networks of this arm folder (a work folder) instead of this experiment's",
)
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
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
import e00651_nbmodel as mymodel  # ty: ignore[unresolved-import]
import e00651_nbsignals as sig  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1"
ROOT613 = ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
SHUFFLED = ARM.endswith("-p")
SPEC_NAME = ARM[:-2] if SHUFFLED else ARM
BASE_ARM, *MODS = SPEC_NAME.split("+")
DIR = ARM.replace("-", "").replace("+", "_")
TABLES = ROOT / "tables"
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
    "P7": _stage1(spans="ew1,ew3,ew7"),
    "B1": _stage1(spans="bx1"),
    "B3": _stage1(spans="bx3"),
    "B7": _stage1(spans="bx7"),
    "B14": _stage1(spans="bx14"),
    "B1-14": _stage1(spans="bx1,bx3,bx7,bx14"),
    "B1-28": _stage1(spans="bx1,bx3,bx7,bx14,bx28"),
    "E1": _stage1(spans="ew1"),
    "E7": _stage1(spans="ew7"),
    "G0": ("gap", "own:ew1,ew3,ew7", "exp"),
}
"""Per stage 1 arm: the group tokens of its extra columns (see `e00613_nbsignals`)."""
GRAPH_MODS: Final = {"GCN": "gcn", "GAT": "gat", "GATR": "gat", "GATF": "gat", "FLOW": "flow"}
MODIFIERS: Final = ("CF", "D30", "D50", "BN", "X1", "X3", "O1", "O2", "O3", "O4", *GRAPH_MODS)
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
CROSSFIT = "CF" in MODS
DROP_P = 0.3 if "D30" in MODS else 0.5 if "D50" in MODS else 0.5 if "O4" in MODS else 0.0
DROP_NB_ONLY = "O4" in MODS
AUX_W = 0.1 if "X1" in MODS else 0.3 if "X3" in MODS else 0.0
INPUTS = mymodel.InputSet(n_extra=N_EXTRA, bottleneck="BN" in MODS, label_head=AUX_W > 0)
GRAPH_MOD = next((m for m in MODS if m in GRAPH_MODS), None)
GRAPH_KIND = GRAPH_MODS[GRAPH_MOD] if GRAPH_MOD else None
assert GRAPH_KIND is None or BASE_ARM == "G0", "a graph layer sits on the own-signal base G0"
FLOW_AUX_W = mymodel.AUX_FLOW_W if GRAPH_KIND == "flow" else 0.0
CHAIN = (
    Path(os.environ["CHAIN613"])
    if os.environ.get("CHAIN613")
    else ML_EXPERIMENTS_DATA_DIR
    / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work"
)
"""A1's pre-training and step-2 weights: experiment 591's, or a folder named by `CHAIN613`, read only."""
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
    p = W574 / "stats" / e.SPEC.b1 / f"{month}.npz"
    assert p.exists(), p
    return p


e.step2_file, e.stats_file = _step2_file, _stats_file

E3_EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
if MINE.e3 and CMD in {"fit", "inject"}:
    # experiment 599's E3 in the daily fine-tune: no rows from the newest 7 days before the cutoff, and
    # half the weight for the 7 days before those; the step-2 embargo stays at 3 days
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
    if GRAPH_KIND:
        names = [sig.ALL_NAMES[c] for c in COLUMNS]
        adj = adjacency(seed=int(e.CTX.get("seed", 0)))
        model = mymodel.GraphLagMdn(
            n_primaries=n_primaries, stage=stage, n_knots=e.KNOTS, n_b1=e.N_B1, inputs=INPUTS,
            kind=GRAPH_KIND, degree=np.array([len(adj.get(p, [])) for p in range(n_primaries)]),
            capacity=NB.capacity, target_std=STD["std"],
            lag_std=np.asarray(e.CTX["stats"])[:, 1],
            gap_cols=(names.index("gap_day"), names.index("gap_week")),
        )  # fmt: skip
        model.table = e.CTX.get(f"table_{stage}")  # ty: ignore[invalid-assignment]
        return model
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
    """Build the month's P5 rows with experiment 591's synthetic lag-only load-shifts in the lag columns.

    The lag columns are exactly A1's. The synthetic power and its shift are kept for the signal block.
    """
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


# ---------------------------------------------------------------- the graph and the neighbour bank

STD: dict[str, np.ndarray] = {}
"""`std`: the per-Primary target standard deviation of the fold month, set before a graph network is built."""
KEY_BASE: Final = 1 << 20


def rewired(*, adj: dict[int, list[int]], seed: int) -> dict[int, list[int]]:
    """Return a degree-preserving rewiring of the undirected graph `adj`: double edge swaps, no self loops or repeats."""
    edges = sorted({tuple(sorted((p, q))) for p, qs in adj.items() for q in qs})
    rng = np.random.default_rng(1_000 + seed)
    present = set(edges)
    for _ in range(20 * len(edges)):
        i, j = rng.choice(len(edges), size=2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if a == d or c == b or new1 in present or new2 in present or new1 == new2:
            continue
        present -= {edges[i], edges[j]}
        present |= {new1, new2}
        edges[i], edges[j] = new1, new2
    out: dict[int, list[int]] = {p: [] for p in adj}
    for a, b in edges:
        out[a].append(b)
        out[b].append(a)
    assert {p: len(v) for p, v in out.items()} == {p: len(v) for p, v in adj.items()}
    return {p: sorted(v) for p, v in out.items()}


_ADJ: dict[int, dict[int, list[int]]] = {}


def adjacency(*, seed: int) -> dict[int, list[int]]:
    """Return the graph the arm's layer reads: electrical edges, a rewired copy (GATR), or every other Primary (GATF)."""
    if seed not in _ADJ:
        if GRAPH_MOD == "GATR":
            _ADJ[seed] = rewired(adj=NB.of, seed=seed)
        elif GRAPH_MOD == "GATF":
            n = len(NB.of)
            _ADJ[seed] = {p: [q for q in range(n) if q != p] for p in range(n)}
        else:
            _ADJ[seed] = {p: list(v) for p, v in NB.of.items()}
    return _ADJ[seed]


def neighbour_table(
    *, prim: np.ndarray, origin_idx: np.ndarray, target_idx: np.ndarray, adj: dict[int, list[int]]
) -> np.ndarray:
    """Return (rows, K) bank rows of each row's neighbours at the same origin and valid time (-1 where absent)."""
    key = origin_idx.astype(np.int64) * KEY_BASE + target_idx.astype(np.int64)
    k_max = max(len(v) for v in adj.values())
    nbr = np.full((len(prim), k_max), -1, np.int64)
    sorted_by: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for q in adj:
        rows = np.flatnonzero(prim == q)
        order = np.argsort(key[rows], kind="stable")
        sorted_by[q] = (key[rows][order], rows[order])
        assert len(np.unique(key[rows])) == len(rows), "two rows of one Primary share a time"
    for p, qs in adj.items():
        rows = np.flatnonzero(prim == p)
        for k, q in enumerate(qs):
            keys_q, pos_q = sorted_by[q]
            if len(keys_q) == 0:
                continue
            j = np.clip(np.searchsorted(keys_q, key[rows]), 0, len(keys_q) - 1)
            hit = keys_q[j] == key[rows]
            nbr[rows[hit], k] = pos_q[j[hit]]
    return nbr


def set_bank(
    *, x: np.ndarray | torch.Tensor, prim: np.ndarray, nbr: np.ndarray, device: str
) -> None:
    """Install the neighbour bank the graph network reads on its next forward pass."""
    assert len(x) < (1 << 24), "row numbers are carried in a float32 column"
    mymodel.BANK.x = (x if torch.is_tensor(x) else torch.from_numpy(x)).to(device)  # ty: ignore[unresolved-attribute]
    mymodel.BANK.p = torch.from_numpy(prim.astype(np.int64)).to(device)
    mymodel.BANK.nbr = torch.from_numpy(nbr).to(device)


def msg_lags(*, x: np.ndarray, power: np.ndarray, prim, target_idx, origin_idx) -> np.ndarray:
    """Return `x` with the 8 lag columns rebuilt from `power` (the power as the neighbours see it)."""
    out = x.copy()
    out[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = e.block_of(
        clean=power, primary=prim, target_idx=target_idx, origin_idx=origin_idx
    )  # fmt: skip
    return out


# ---------------------------------------------------------------- the table T and the row blocks


TABLES_606 = (
    ML_EXPERIMENTS_DATA_DIR
    / "00606-nra-lag-encoder-signal-permutations-gap-averages-and-neighbours/tables"
)
CF_BLOCKS: Final = 3
"""The number of contiguous blocks (by valid time) of the cross-fit."""


def table_file(*, month: str, seed: int):
    if CROSSFIT:
        return TABLES / f"T_cf_{month}_s{seed}.npz"
    for root in (TABLES_598, TABLES_606, TABLES_610):
        if (root / f"T_{month}_s{seed}.npz").exists():
            return root / f"T_{month}_s{seed}.npz"
    return TABLES / f"T_{month}_s{seed}.npz"


def cross_fit_quantiles(
    *, mr, month: str, seed: int, q_full: np.ndarray, device: str
) -> np.ndarray:
    """Return `q_full`, with cross-fitted lags-masked quantiles on step 2's training rows.

    Step 2's recipe is experiment 574's `step2` (the pre-trained weights, a fresh mixer, no recency
    weighting, the embargoed rows up to the fold's cutoff). The rows are split into `CF_BLOCKS`
    contiguous blocks by valid time; the network of block `b` is trained on the rows outside the block.
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
        table = s598.build_lag_free_table(
            prim=np.concatenate([mr.primary, test.primary]),
            origin_idx=np.concatenate([origin_idx_of(origin_us=mr.origin_us), test.origin_idx]),
            target_idx=np.concatenate([mr.target_idx, test.target_idx]),
            median=np.concatenate([q_mr[:, 1], q_te[:, 1]]),
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


GRID_SHAPE = (16, 121248)


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
        self._train_msg: np.ndarray | None = None
        self.report: dict = {}

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
        """Return the training rows' block: synthetic shifts in the own and neighbour power, training
        noise where the training-to-test spread ratio is low. Cached.
        """
        if self._train is not None:
            return self._train
        synth = self._block(rows=self._train_rows(), power_lag=self.fixed, power_sig=self.power_sig)
        real_test = self._block(rows=self._test_rows(), power_lag=self.clean, power_sig=self.clean)
        plan = self.noise_plan(synth=synth, real_test=real_test)
        self.plan = plan
        sig.add_training_noise(block=synth, prim=self.mr.primary, plan=plan, seed=self.seed)
        del real_test
        assert synth.shape[0] == len(self.mr.x), "training block and month rows differ"
        self._train = synth
        return synth

    def train_block_msg(self) -> np.ndarray:
        """Return the training rows' block as their neighbours see it (the mirrored power in every reading), same noise plan."""
        if self._train_msg is None:
            self.train_block()
            synth = self._block(
                rows=self._train_rows(), power_lag=self.power_sig, power_sig=self.power_sig
            )
            sig.add_training_noise(
                block=synth, prim=self.mr.primary, plan=self.plan, seed=self.seed
            )
            self._train_msg = synth
        return self._train_msg

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
        base_x = t["x"]
        parts = [t["x"], torch.from_numpy(b).to(self.device)]
        if AUX_W > 0 or FLOW_AUX_W > 0:
            lab = torch.from_numpy(LABELS[id(self.mr)][refit.index]).to(self.device)
            hidden = torch.from_numpy(
                np.stack([day_masked, day_masked, week_masked, week_masked], axis=1)
            )
            lab = torch.where(hidden.to(self.device), torch.full_like(lab, float("nan")), lab)
            parts.append(lab)
        if GRAPH_KIND:
            self.install_bank(t=t, refit=refit, fs=fs, base_x=base_x, day_masked=day_masked, week_masked=week_masked)  # fmt: skip
            parts.append(
                torch.arange(len(base_x), device=self.device, dtype=torch.float32)[:, None]
            )
        t["x"] = torch.cat(parts, dim=1)
        return t

    def install_bank(self, *, t, refit, fs, base_x, day_masked, week_masked) -> None:
        """Build the training rows' neighbour view (mirrored shifts, same lag masks) and the neighbour table."""
        idx = refit.index
        origin = origin_idx_of(origin_us=self.mr.origin_us[idx])
        xm = base_x.clone()
        blk = e.block_of(
            clean=fs.power_sig, primary=self.mr.primary[idx], target_idx=self.mr.target_idx[idx],
            origin_idx=origin,
        )  # fmt: skip
        xm[:, LAG_FIRST : LAG_FIRST + LAG_WIDTH] = torch.from_numpy(blk).to(self.device)
        e.apply_dropout(x=xm, origin_idx=origin, seed=e.CTX["seed"])
        assert torch.equal(xm[:, 38] > 0.5, torch.from_numpy(day_masked).to(self.device)), (
            "lag masks differ"
        )
        bm = fs.train_block_msg()[idx].copy()
        sig.mask_lag_columns(block=bm, columns=COLUMNS, day_masked=day_masked, week_masked=week_masked)  # fmt: skip
        nbr = neighbour_table(
            prim=self.mr.primary[idx], origin_idx=origin, target_idx=self.mr.target_idx[idx],
            adj=adjacency(seed=int(e.CTX["seed"])),
        )  # fmt: skip
        set_bank(
            x=torch.cat([xm, torch.from_numpy(bm).to(self.device)], dim=1),
            prim=self.mr.primary[idx], nbr=nbr, device=self.device,
        )  # fmt: skip


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
        own_x = sub.x
        sub = dataclasses.replace(sub, x=np.concatenate([sub.x, block], axis=1))
        if GRAPH_KIND:
            # the neighbours see each row with the mirrored power in its lags and signals
            xm = msg_lags(
                x=own_x, power=power_sig, prim=sub.primary, target_idx=sub.target_idx,
                origin_idx=sub.origin_idx,
            )  # fmt: skip
            block_m = fs.test_block(test=sub, power_lag=power_sig, power_sig=power_sig)
            set_bank(
                x=np.concatenate([xm, block_m], axis=1), prim=sub.primary,
                nbr=neighbour_table(
                    prim=sub.primary, origin_idx=sub.origin_idx, target_idx=sub.target_idx,
                    adj=adjacency(seed=int(e.CTX["seed"])),
                ),
                device=device,
            )  # fmt: skip
            ids = np.arange(len(sub.x), dtype=np.float32)[:, None]
            sub = dataclasses.replace(sub, x=np.concatenate([sub.x, ids], axis=1))
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
    if lab is not None and (AUX_W > 0 or FLOW_AUX_W > 0):
        nll = nll + (AUX_W or FLOW_AUX_W) * lab
    return nll


def run_fit(*, device: str = "cuda") -> None:
    if AUX_W > 0 or FLOW_AUX_W > 0:
        e.row_nll = row_nll
    clean = e.c.clean_stack()
    assert clean.shape == GRID_SHAPE, clean.shape
    e.k.build_month = build_month_syn
    e.k.GpuMonth = SigGpuMonth
    for month in e.ARGS.months:
        cuts = e.k.cutoffs(month=month, cadence=e.CADENCE)
        mr = e.k.build_month(month=month, clean=clean)
        STD["std"] = mr.base.target_std
        test = e.build_test(month=month, standardiser=mr.base.standardiser)
        if N_EXTRA:
            ensure_tables(mr=mr, test=test, month=month, seeds=e.ARGS.seeds, device=device)
        g = SigGpuMonth(mr=mr, device=device)
        for seed in e.ARGS.seeds:
            e.CTX["seed"] = seed
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


def weights_file(*, tag: str, month: str, day: int, seed: int):
    """Return a saved network's path: this arm's, or the folder named by `--weights-root`."""
    if MINE.weights_root is None:
        return e.model_file(tag=tag, month=month, day=day, seed=seed)
    saved = e.W
    e.W = Path(MINE.weights_root) / "work"
    try:
        return e.model_file(tag=tag, month=month, day=day, seed=seed)
    finally:
        e.W = saved


def bank_power(*, fixed, shift, choice, r: int):
    """Return the power the neighbours' readings see on injected bank `MINE.bank`.

    M: each shift mirrored onto one neighbour (experiment 597's rule). S: the same neighbour takes 60%
    and the next neighbour 40%. X: no mirror, so the load leaves the graph.
    """
    if MINE.bank == "M":
        return s598.mirrored_power(base=fixed, shift=shift, choice=choice, capacity=NB.capacity)
    if MINE.bank == "X":
        return fixed.copy()
    first = s598.mirrored_power(base=fixed, shift=0.6 * shift, choice=choice, capacity=NB.capacity)
    second = {p: qs[(r + p + 1) % len(qs)] for p, qs in NB.of.items() if len(qs) > 1}
    both = s598.mirrored_power(base=fixed, shift=0.4 * shift, choice=second, capacity=NB.capacity)
    # `first` and `both` each hold `fixed`; add the second neighbour's share to `first`
    return first + (both - fixed)


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
        out_dir = e.W / ("inject" if MINE.bank == "M" else f"inject_{MINE.bank}")
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
        STD["std"] = mr.base.target_std
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
                mirrored = bank_power(fixed=fixed, shift=shift, choice=choice, r=r)
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
            e.CTX["seed"] = seed
            cut = cuts[i]
            day = e.f.day_of(month=month, cut=cut)
            if (i, seed) not in models:
                m = new_model(n_primaries=len(mr.base.target_mean)).to(device)
                m.load_state_dict(
                    torch.load(
                        weights_file(tag=e.SPEC.tag, month=month, day=day, seed=seed),
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


if __name__ == "__main__":
    if CMD in {"table", "check"}:
        e.k.build_month = build_month_syn
        e.k.GpuMonth = SigGpuMonth
        run_prepare(device="cuda", command=CMD)
    elif CMD == "fit":
        run_fit()
    elif CMD == "inject":
        run_inject()
    else:
        run_assemble()
