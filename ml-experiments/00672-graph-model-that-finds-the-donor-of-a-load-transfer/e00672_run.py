"""Tune the decoder on validation worlds, then score a network on injected worlds and real data.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00672-graph-model-that-finds-the-donor-of-a-load-transfer:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00658-edge-state-event-table-neural-network:\
$P/00657-event-table-with-explicit-solar-term-against-plain-arm:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    uv run --no-sync python $P/00672-*/e00672_run.py tune <tag>
    uv run --no-sync python $P/00672-*/e00672_run.py score <tag> <config> <first_seed> <n_worlds>
    uv run --no-sync python $P/00672-*/e00672_run.py real <tag>

`tag` is `<arm>_<mix>_seed<k>`, as written by `e00672_train.py`. Stage `tune` writes the decoder
costs chosen on validation worlds only. Stage `score` writes the scored truth rows of injected
worlds of one generator configuration (the same seeds as experiment 670, so the rows pair with
E1-J's). Stage `real` decodes the real record.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00653_core import events_frame, match
from e00658_data import World
from e00658_decode import CHUNK, STRIDE, DecodeCosts, decode_net
from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]
from e00670_run import (  # ty: ignore[unresolved-import]
    CELLS,
    event_free,
    events_table,
    logged_shifts,
    matched,
    verified_d5_d16,
)
from e00670_run import (
    CONFIGS as CONFIGS_670,
)
from e00672_data import TRAIN_END, build_world, inject_world, rig
from e00672_net import DonorNet
from e00672_train import MIXES, SEED0, WORK, _solar, graph_of

CONFIGS: Final = {**CONFIGS_670, "shaped": MIXES["shaped"]}
"""Experiment 670's configurations plus the shaped one of the cross-injector test."""
DEV: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GRID: Final = (1.0, 2.0)
BOOSTS: Final = tuple(float(x) for x in os.environ.get("E672_BOOSTS", "0.3,1,3,10").split(","))
"""The none-boost grid. The environment variable widens it for the post hoc arm `Gw` only."""
LAMBDAS: Final = (0.0, 1.0)
CUTOFFS: Final = (0.3, 0.5, 0.7)
MAX_VAL_RATE: Final[float] = 6.0
KINDS: Final = ("edge", "solo", "permanent_transfer", "permanent_shift")
N_BLOCKS: Final[int] = GRID_LENGTH // 12


@dataclass(frozen=True)
class Costs:
    """The decoder's switching costs, none-boost, link cut-off, and donor weight."""

    on: float
    boost: float
    cutoff: float
    lam: float


def parse(*, tag: str) -> tuple[str, str, int]:
    """Arm, generator mix, and seed of a tag."""
    arm, mix, seed = tag.split("_")
    return arm, mix, int(seed.removeprefix("seed"))


def load_net(*, tag: str) -> DonorNet:
    """Load a trained network."""
    net = DonorNet().to(DEV)
    net.load_state_dict(torch.load(WORK / f"net_{tag}.pt", map_location=DEV))
    return net.eval()


@torch.no_grad()
def predict(*, net: DonorNet, world: World, nbr: np.ndarray, edges: np.ndarray) -> dict:
    """Run the network over the whole grid in overlapping chunks.

    Args:
        net: The trained network.
        world: The world (features for every block).
        nbr: Neighbour matrix of the arm's graph.
        edges: Link index pairs of the arm's graph.

    Returns:
        Probabilities `cls` (primaries, blocks, 4), `pos` (primaries, blocks), `donor`
        (primaries, blocks), and `edge` (links, blocks, 3).
    """
    seen = np.flatnonzero((world.feats[:, :, 1] > 0).any(axis=0))
    n_b = int(seen[-1]) + 1 if len(seen) else N_BLOCKS
    x = torch.from_numpy(world.feats[:, :n_b]).to(DEV)
    nb, ed = torch.from_numpy(nbr).to(DEV), torch.from_numpy(edges).to(DEV)
    cls = np.zeros((x.shape[0], n_b, 4), np.float32)
    pos = np.zeros((x.shape[0], n_b), np.float32)
    donor = np.zeros((x.shape[0], n_b), np.float32)
    edge = np.zeros((len(edges), n_b, 3), np.float32)
    starts = list(range(0, max(1, n_b - CHUNK) + 1, STRIDE))
    if starts[-1] + CHUNK < n_b:
        starts.append(max(0, n_b - CHUNK))
    for k, s in enumerate(starts):
        out = net(x=x[None, :, s : s + CHUNK], nbr=nb, edges=ed)
        lo = 0 if k == 0 else STRIDE // 2
        hi = CHUNK if k == len(starts) - 1 else CHUNK - STRIDE // 2
        sl = slice(s + lo, s + hi)
        cls[:, sl] = torch.softmax(out.cls[0, :, lo:hi], dim=-1).cpu().numpy()
        pos[:, sl] = torch.sigmoid(out.sign[0, :, lo:hi]).cpu().numpy()
        donor[:, sl] = torch.sigmoid(out.donor[0, :, lo:hi]).cpu().numpy()
        if len(edges):
            edge[:, sl] = torch.softmax(out.edge[0, :, lo:hi], dim=-1).cpu().numpy()
    pad = N_BLOCKS - n_b
    return {
        "cls": np.pad(cls, ((0, 0), (0, pad), (0, 0))),
        "pos": np.pad(pos, ((0, 0), (0, pad))),
        "donor": np.pad(donor, ((0, 0), (0, pad))),
        "edge": np.pad(edge, ((0, 0), (0, pad), (0, 0))),
    }


def with_donor(*, pred: dict, lam: float) -> dict:
    """Move `lam` times the donor probability of the 'none' mass to 'shift, down'.

    With `lam` 0 the decoder uses the class head alone, which is the ablation of the donor head.
    """
    if lam == 0.0:
        return pred
    cls, pos = pred["cls"].copy(), pred["pos"]
    move = np.clip(lam * pred["donor"], 0.0, 1.0) * cls[..., 0]
    up, down = cls[..., 1] * pos, cls[..., 1] * (1 - pos) + move
    cls[..., 0] -= move
    cls[..., 1] = up + down
    new_pos = np.divide(up, cls[..., 1], out=np.full_like(up, 0.5), where=cls[..., 1] > 0)
    return {**pred, "cls": cls, "pos": new_pos}


def decode(*, world: World, pred: dict, costs: Costs, edges: np.ndarray) -> list:
    """Event table of a world at the given costs."""
    r = rig()
    c = DecodeCosts(
        on=costs.on, flip=2 * costs.on, edge_cutoff=costs.cutoff, none_boost=costs.boost
    )
    sigma = {s: (v if np.isfinite(v) else 1.0) for s, v in r.frozen.sigma.items()}
    return decode_net(
        panel=r.p653,
        world=world,
        sigma=sigma,
        pred=with_donor(pred=pred, lam=costs.lam),
        costs=c,
        edges=edges,
    )


def donor_and_found(rows: pl.DataFrame) -> tuple[float, float]:
    """MW-weighted donor found share (edge-truth donors) and all-event found share."""
    big = rows.filter(pl.col("units") >= 4)
    w = big["amp_mw"].abs()
    found = float((big["found"].cast(pl.Float64) * w).sum() / w.sum())  # ty: ignore[unsupported-operator] if big.height else np.nan
    don = big.filter(pl.col("edge_truth") & (pl.col("amp_mw") < 0))
    wd = don["amp_mw"].abs()
    dfound = float((don["found"].cast(pl.Float64) * wd).sum() / wd.sum())  # ty: ignore[unsupported-operator] if don.height else np.nan
    return dfound, found


def tune(*, tag: str) -> None:
    """Choose the decoder costs on validation worlds only (grid fixed in the plan)."""
    arm, mix, seed = parse(tag=tag)
    solar = _solar() if arm == "GS" else None
    net = load_net(tag=tag)
    nbr, edges = graph_of(arm=arm, seed=seed)
    vals = [
        build_world(
            world=inject_world(config=MIXES[mix], seed=SEED0[mix] + 5_000 + k, upto=SCORED_FROM),
            solar=solar,
        )
        for k in range(6)
    ]
    real = build_world(world=None, solar=solar, upto=SCORED_FROM)
    p653 = rig().p653
    n_present = sum(
        int(np.isfinite(real.residual[s][TRAIN_END:SCORED_FROM]).sum() > 1000) for s in p653.sites
    )
    years = n_present * (SCORED_FROM - TRAIN_END) / (365.25 * DAY)
    preds = [predict(net=net, world=w, nbr=nbr, edges=edges) for w in vals]
    rpred = predict(net=net, world=real, nbr=nbr, edges=edges)
    truth = [
        truths(events=w.events, lo=TRAIN_END, hi=SCORED_FROM)  # ty: ignore[invalid-argument-type]
        for w in vals
    ]
    out = []

    def one(c: Costs) -> dict:
        ev = decode(world=real, pred=rpred, costs=c, edges=edges)
        rate = sum(e.start >= TRAIN_END for e in ev) / years
        dfs, fs = [], []
        for w, p, t in zip(vals, preds, truth, strict=True):
            e = [x for x in decode(world=w, pred=p, costs=c, edges=edges) if x.start >= TRAIN_END]
            rows = pl.DataFrame(match(truth=t, events=e, world_end=SCORED_FROM)).filter(
                pl.col("kind").is_in(KINDS)
            )
            d, f = donor_and_found(rows)
            dfs.append(d)
            fs.append(f)
        r = {
            "on": c.on,
            "boost": c.boost,
            "cutoff": c.cutoff,
            "lam": c.lam,
            "rate": rate,
            "found": float(np.nanmean(fs)),
            "donor_found": float(np.nanmean(dfs)),
        }
        print(tag, r, flush=True)
        return r

    for on, boost, lam in itertools.product(ON_GRID, BOOSTS, LAMBDAS):
        out.append(one(Costs(on=on, boost=boost, cutoff=0.5, lam=lam)))
    t0 = pl.DataFrame(out)
    ok = t0.filter(pl.col("rate") <= MAX_VAL_RATE)
    pick = (
        (ok if ok.height else t0)
        .sort("found", descending=True, maintain_order=True)
        .row(0, named=True)
    )
    out.extend(
        one(Costs(on=pick["on"], boost=pick["boost"], cutoff=cut, lam=pick["lam"]))
        for cut in CUTOFFS
        if cut != 0.5
    )
    t = pl.DataFrame(out)
    t.write_parquet(WORK / f"tuning_{tag}.parquet")
    same = t.filter(
        (pl.col("on") == pick["on"])
        & (pl.col("boost") == pick["boost"])
        & (pl.col("lam") == pick["lam"])
    )
    same_ok = same.filter(pl.col("rate") <= MAX_VAL_RATE)
    cut = (
        (same_ok if same_ok.height else same)
        .sort("found", descending=True, maintain_order=True)
        .row(0, named=True)["cutoff"]
    )
    (WORK / f"costs_{tag}.json").write_text(
        json.dumps({"on": pick["on"], "boost": pick["boost"], "cutoff": cut, "lam": pick["lam"]})
    )
    print("picked", tag, pick["on"], pick["boost"], cut, pick["lam"])


def picked(*, tag: str) -> Costs:
    """The costs chosen on validation."""
    return Costs(**json.loads((WORK / f"costs_{tag}.json").read_text()))


def score(*, tag: str, config: str, first_seed: int, n_worlds: int) -> None:
    """Score injected scored-year worlds; the seeds and generator are those of experiment 670."""
    arm, _, seed = parse(tag=tag)
    solar = _solar() if arm == "GS" else None
    net = load_net(tag=tag)
    nbr, edges = graph_of(arm=arm, seed=seed)
    c = picked(tag=tag)
    rows = []
    for s in range(first_seed, first_seed + n_worlds):
        iw = inject_world(config=CONFIGS[config], seed=s, upto=None)
        w = build_world(world=iw, solar=solar)
        pred = predict(net=net, world=w, nbr=nbr, edges=edges)
        ev = decode(world=w, pred=pred, costs=c, edges=edges)
        truth = truths(events=iw.events, lo=SCORED_FROM, hi=GRID_LENGTH)
        scored = match(truth=truth, events=ev, world_end=GRID_LENGTH)
        n_edge = sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in ev)
        rows.extend(
            {
                **r,
                "config": config,
                "world": s,
                "method": tag,
                "world_edge_events": n_edge,
                "world_right_pairs": sum(bool(x["right_class"]) for x in scored),
            }
            for r in scored
        )
        print(tag, config, s, flush=True)
    pl.DataFrame(rows).write_parquet(WORK / f"injected_{tag}_{config}_{first_seed}.parquet")


def real(*, tag: str) -> None:
    """Decode the real record and write its event table and the checks of the plan."""
    arm, _, seed = parse(tag=tag)
    solar = _solar() if arm == "GS" else None
    net = load_net(tag=tag)
    nbr, edges = graph_of(arm=arm, seed=seed)
    c = picked(tag=tag)
    w = build_world(world=None, solar=solar)
    ev = decode(world=w, pred=predict(net=net, world=w, nbr=nbr, edges=edges), costs=c, edges=edges)
    r = rig()
    scored_years = sum(
        int(np.isfinite(r.p653.residual[s][SCORED_FROM:GRID_LENGTH]).sum()) for s in r.p653.sites
    ) / (DAY * 365.25)
    table = events_table(events=ev)
    table.write_parquet(WORK / f"real_events_{tag}.parquet")
    logged = logged_shifts(rig=r, resid=r.real_residual)
    edge = table.filter(pl.col("cls").is_in(["edge_transfer", "permanent_transfer"]))
    res = {
        "scored_primary_years": scored_years,
        "events": table.height,
        "events_per_primary_year": table.height / scored_years,
        "edge_events": edge.height,
        "edge_events_by_cell": {k: edge.filter(pl.col("cell") == k).height for k in CELLS},
        "logged_matched_all": matched(table=table, logged=logged),
        "logged_matched_by_cell": {k: matched(table=table, logged=logged, cell=k) for k in CELLS},
        "d5_d16": verified_d5_d16(events=ev),
        "event_free": event_free(rig=r, tables={tag: table}, scored_years=scored_years),
    }
    (WORK / f"real_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    events_frame(events=ev).write_parquet(WORK / f"real_events_full_{tag}.parquet")
    print(tag, {k: v for k, v in res.items() if k not in ("d5_d16", "event_free")})


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        tune(tag=sys.argv[2])
    elif stage == "score":
        score(
            tag=sys.argv[2],
            config=sys.argv[3],
            first_seed=int(sys.argv[4]),
            n_worlds=int(sys.argv[5]),
        )
    else:
        real(tag=sys.argv[2])
