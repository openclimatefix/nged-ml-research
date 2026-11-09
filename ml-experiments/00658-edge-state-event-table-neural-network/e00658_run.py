"""Tune the decoder on validation worlds, then score the network on experiment 653's worlds.

DATA_PATH_INTERNAL=... PYTHONPATH=<folder of experiment 653>:. \
        uv run --no-sync python e00658_run.py tune <tag>
... e00658_run.py score <tag>          injected scored-year worlds, every decoder cost
... e00658_run.py real <tag>           real record, rewired-graph placebo
... e00658_run.py baseline             experiment 653's default arm on the same worlds
"""

from __future__ import annotations

import copy
import itertools
import json
import sys
from dataclasses import asdict

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00653_core import decode_world, events_frame, freeze, match
from e00653_inject import GenParams
from e00653_run import big, load_params, mw_recall, rewire
from e00658_data import SCORED_FROM as SCORED
from e00658_data import TRAIN_END, build_world
from e00658_decode import DecodeCosts, decode_net, graph_for, predict
from e00658_net import EdgeStateNet
from e00658_train import MIXES, WORK, _make, setup

ON_GRID = (0.5, 1.0, 2.0)
BOOSTS = (0.1, 0.3, 1.0, 3.0, 10.0)
CUTOFFS = (0.3, 0.5, 0.7)
MAX_VAL_RATE = 6.0
KINDS = ["edge", "solo", "permanent_transfer", "permanent_shift"]
SPECS: list[tuple[str, GenParams, int]] = [
    ("default", GenParams(), 6),
    ("constant", GenParams(shape_prop=0.0), 2),
    ("proportional", GenParams(shape_prop=1.0), 2),
    ("short", GenParams(duration_mix="short"), 2),
    ("long", GenParams(duration_mix="long"), 2),
    ("stress", GenParams(share=(0.2, 1.8), start_jitter_hh=24, width_hh=(0, 24)), 2),
]
CROSS_SPECS: list[tuple[str, GenParams, int]] = [
    ("g1_test", MIXES["g1"], 4),
    ("g2_test", MIXES["g2"], 4),
]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def world_rng(*, name: str, k: int) -> np.random.Generator:
    """Experiment 653's seed for its specs; the cross-injector specs get an offset of their own."""
    offset = 500 if name.endswith("_test") else 0
    return np.random.default_rng(1000 + 17 * k + len(name) + offset)


def load_net(*, tag: str) -> EdgeStateNet:
    """Load a trained network."""
    net = (EdgeStateNet(hidden=32, dropout=0.3) if tag.startswith("v2") else EdgeStateNet()).to(DEV)
    net.load_state_dict(torch.load(WORK / f"net_{tag}.pt", map_location=DEV))
    return net.eval()


def costs_for(*, on: float, cutoff: float, boost: float = 1.0) -> DecodeCosts:
    """Decode costs for a grid point (a sign flip costs twice an onset)."""
    return DecodeCosts(on=on, flip=2 * on, edge_cutoff=cutoff, none_boost=boost)


def pairing_recall(rows: pl.DataFrame) -> float:
    """MW-weighted share of edge-transfer ends paired correctly among big positive events."""
    e = big(rows.filter(pl.col("kind").is_in(KINDS))).filter(pl.col("edge_truth"))
    return mw_recall(e, "right_class") if e.height else float("nan")


def tune(*, tag: str) -> None:
    """Pick the onset cost and edge cut-off on validation worlds only."""
    panel, res, sigma = setup()
    net = load_net(tag=tag)
    nbr, edges = graph_for(panel=panel, neighbours=None)
    mix = tag.split("_seed", maxsplit=1)[0].removeprefix("v2")
    seed0 = {"main": 10_000, "g1": 20_000, "g2": 30_000}[mix]
    vals = [_make((mix, seed0 + 5_000 + k, "val")) for k in range(6)]
    real = build_world(
        panel=panel, residual=res, sigma=sigma, rng=None, lo=0, hi=0, gen=None, upto=SCORED
    )
    n_present = sum(
        int(np.isfinite(real_res).sum() > 1000)
        for real_res in (res[s][TRAIN_END:SCORED] for s in panel.sites)
    )
    years = n_present * (SCORED - TRAIN_END) / (365.25 * DAY)
    preds = [predict(net=net, world=w, nbr=nbr, edges=edges, device=DEV) for w in vals]
    rpred = predict(net=net, world=real, nbr=nbr, edges=edges, device=DEV)
    out = []

    def one(on: float, boost: float, cut: float) -> dict[str, float]:
        c = costs_for(on=on, cutoff=cut, boost=boost)
        ev = decode_net(panel=panel, world=real, sigma=sigma, pred=rpred, costs=c, edges=edges)
        rate = sum(e.start >= TRAIN_END for e in ev) / years
        found, pair = [], []
        for w, p in zip(vals, preds, strict=True):
            e = decode_net(panel=panel, world=w, sigma=sigma, pred=p, costs=c, edges=edges)
            e = [x for x in e if x.start >= TRAIN_END]
            rows = pl.DataFrame(match(truth=w.truth, events=e, world_end=SCORED)).filter(
                pl.col("kind").is_in(KINDS)
            )
            found.append(mw_recall(big(rows), "found"))
            pair.append(pairing_recall(rows))
        r = {
            "on": on,
            "boost": boost,
            "cutoff": cut,
            "rate": rate,
            "found": float(np.nanmean(found)),
            "pair": float(np.nanmean(pair)),
        }
        print(r, flush=True)
        return r

    out.extend(one(on, boost, 0.5) for on, boost in itertools.product(ON_GRID, BOOSTS))
    t0 = pl.DataFrame(out)
    ok = t0.filter(pl.col("rate") <= MAX_VAL_RATE)
    pick = (ok if ok.height else t0).sort("found", descending=True).row(0, named=True)
    out.extend(one(pick["on"], pick["boost"], cut) for cut in (0.3, 0.7))
    t = pl.DataFrame(out)
    best_on, best_boost = pick["on"], pick["boost"]
    best_cut = (
        t.filter((pl.col("on") == best_on) & (pl.col("boost") == best_boost))
        .sort("pair", descending=True)
        .row(0, named=True)["cutoff"]
    )
    t.write_parquet(WORK / f"tuning_{tag}.parquet")
    (WORK / f"costs_{tag}.json").write_text(
        json.dumps({"on": best_on, "cutoff": best_cut, "boost": best_boost})
    )
    print("picked", best_on, best_cut)


def picked(*, tag: str) -> DecodeCosts:
    """The costs chosen on validation."""
    d = json.loads((WORK / f"costs_{tag}.json").read_text())
    return costs_for(on=d["on"], cutoff=d["cutoff"], boost=d["boost"])


def score(*, tag: str, specs: list[tuple[str, GenParams, int]], out_name: str) -> None:
    """Score injected scored-year worlds at every grid cost, for the network and its variants."""
    panel, res, sigma = setup()
    net = load_net(tag=tag)
    nbr, edges = graph_for(panel=panel, neighbours=None)
    rows = []
    for name, gen, n in specs:
        for k in range(n):
            w = build_world(
                panel=panel,
                residual=res,
                sigma=sigma,
                rng=world_rng(name=name, k=k),
                lo=SCORED_FROM,
                hi=GRID_LENGTH,
                gen=gen,
                upto=None,
            )
            pred = predict(net=net, world=w, nbr=nbr, edges=edges, device=DEV)
            for on, boost in itertools.product(ON_GRID, BOOSTS):
                cut = picked(tag=tag).edge_cutoff
                c = costs_for(on=on, cutoff=cut, boost=boost)
                for arm in ("net", "e653pair"):
                    ev = decode_net(
                        panel=panel,
                        world=w,
                        sigma=sigma,
                        pred=pred,
                        costs=c,
                        edges=edges,
                        pair="net" if arm == "net" else "e653",
                    )
                    rows.extend(
                        {
                            **r,
                            "tag": tag,
                            "spec": name,
                            "world": k,
                            "arm": arm,
                            "on": on,
                            "boost": boost,
                            "cutoff": cut,
                        }
                        for r in match(truth=w.truth, events=ev)
                    )
            print(tag, name, k, flush=True)
    pl.DataFrame(rows).write_parquet(WORK / f"{out_name}_{tag}.parquet")


def baseline() -> None:
    """Experiment 653's default arm on the same worlds, re-run here."""
    panel, res, sigma = setup()
    fz = freeze(panel=panel, residual=res)
    p = load_params()
    rows = []
    for name, gen, n in SPECS + CROSS_SPECS:
        for k in range(n):
            w = build_world(
                panel=panel,
                residual=res,
                sigma=sigma,
                rng=world_rng(name=name, k=k),
                lo=SCORED_FROM,
                hi=GRID_LENGTH,
                gen=gen,
                upto=None,
            )
            ev, _ = decode_world(
                panel=panel, residual=w.residual, frozen=fz, arm="masked", params=p
            )
            rows.extend(
                {**r, "spec": name, "world": k, "arm": "e653"}
                for r in match(truth=w.truth, events=ev)
            )
    pl.DataFrame(rows).write_parquet(WORK / "baseline_e653.parquet")


def real(*, tag: str) -> None:
    """Decode the real record at the picked costs and run the rewired-graph placebo."""
    panel, res, sigma = setup()
    net = load_net(tag=tag)
    nbr, edges = graph_for(panel=panel, neighbours=None)
    w = build_world(
        panel=panel, residual=res, sigma=sigma, rng=None, lo=0, hi=0, gen=None, upto=None
    )
    pred = predict(net=net, world=w, nbr=nbr, edges=edges, device=DEV)
    c = picked(tag=tag)
    grid = []
    for on, boost in itertools.product(ON_GRID, BOOSTS):
        ev = decode_net(
            panel=panel,
            world=w,
            sigma=sigma,
            pred=pred,
            costs=costs_for(on=on, cutoff=c.edge_cutoff, boost=boost),
            edges=edges,
        )
        events_frame(events=ev).write_parquet(WORK / f"real_events_{tag}_on{on}_b{boost}.parquet")
        grid.append({"on": on, "boost": boost, "n_scored": sum(e.start >= SCORED_FROM for e in ev)})
    ev = decode_net(panel=panel, world=w, sigma=sigma, pred=pred, costs=c, edges=edges)
    events_frame(events=ev).write_parquet(WORK / f"real_events_{tag}.parquet")
    ev653 = decode_net(
        panel=panel, world=w, sigma=sigma, pred=pred, costs=c, edges=edges, pair="e653"
    )
    events_frame(events=ev653).write_parquet(WORK / f"real_events_{tag}_e653pair.parquet")
    rng = np.random.default_rng(7)
    counts, counts653 = [], []
    for _ in range(100):
        nb = rewire(edges=panel.edges, rng=rng)
        nbr_r, edges_r = graph_for(panel=panel, neighbours={s: nb.get(s, []) for s in panel.sites})
        pr = predict(net=net, world=w, nbr=nbr_r, edges=edges_r, device=DEV)
        er = decode_net(panel=panel, world=w, sigma=sigma, pred=pr, costs=c, edges=edges_r)
        counts.append(sum(e.cls == "edge_transfer" and e.start >= SCORED_FROM for e in er))
        e6 = decode_net(
            panel=panel,
            world=w,
            sigma=sigma,
            pred=pr,
            costs=c,
            edges=edges_r,
            pair="e653",
            neighbours={s: nb.get(s, []) for s in panel.sites},
        )
        counts653.append(sum(e.cls == "edge_transfer" and e.start >= SCORED_FROM for e in e6))
    realn = sum(e.cls == "edge_transfer" and e.start >= SCORED_FROM for e in ev)
    (WORK / f"rewired_{tag}.json").write_text(
        json.dumps({"real": realn, "rewired": counts, "rewired_e653pair": counts653, "grid": grid})
    )
    print("edge events real", realn, "rewired median", np.median(counts), grid)
    assert copy.copy is not None
    assert asdict(c)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "tune":
        tune(tag=sys.argv[2])
    elif stage == "score":
        score(tag=sys.argv[2], specs=SPECS, out_name="injected")
    elif stage == "cross":
        score(tag=sys.argv[2], specs=CROSS_SPECS, out_name="cross")
    elif stage == "real":
        real(tag=sys.argv[2])
    else:
        baseline()
