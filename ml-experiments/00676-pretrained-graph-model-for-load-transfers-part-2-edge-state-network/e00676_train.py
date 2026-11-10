"""Train one edge-state neural network of experiment 676 (GPU).

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    (PYTHONPATH as in `e00676_posthoc.py`, plus experiment 675's and this folder)
    uv run --no-sync python e00676_train.py <arm: G|G-R|G-0> <seed> <steps>

Windows of 2,880 hours from the 200 training worlds of experiment 675 (events before 2023-07-01),
half of them centred on an injected edge transfer. Cross-entropy on the link state (weight 1.0;
transfer states weigh 5; off hours inside an E1-decoded real event weigh 0), the lag (0.3), and the
node state (0.5). AdamW 1e-3 with cosine decay. Checkpoint every 1,000 steps; a rerun resumes.
"""

# ruff: noqa: ANN001, ANN201, E501, E702

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00670_run import rewire_low_overlap  # ty: ignore[unresolved-import]
from e00672_data import TRAIN_END
from e00675_pretrain import ROOT as ROOT675
from e00676_data import labels, window_inputs
from e00676_net import EdgeStateNet
from e00676_worlds import ROOT, WORLDS

SPAN: Final[int] = 2880
BATCH: Final[int] = 4
TRAIN_HOURS: Final[int] = TRAIN_END // 2
W_TRANSFER: Final[float] = 5.0
DEV: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS_TRAIN: Final = range(100000, 100200)


def graph_of(*, rig, arm: str):
    """Sites, megawatt-or-megavolt-ampere flags, index edges, and the adjacency of an arm."""
    sites = list(rig.p653.sites)
    index = {s: i for i, s in enumerate(sites)}
    if arm == "G-R":
        nb, kept = rewire_low_overlap(rig=rig, rng=np.random.default_rng(676))
        print("rewired graph keeps", round(kept, 3), "of the real links", flush=True)
    else:
        nb = rig.p653.neighbours
    edges = sorted({tuple(sorted((index[a], index[b]))) for a in nb for b in nb[a] if b in index})
    adj = torch.zeros(len(sites), len(sites))
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1.0
    adj = adj / adj.sum(1, keepdim=True).clamp(min=1)
    mva = torch.tensor([float(rig.panel.is_mva[s]) for s in sites], dtype=torch.float64)
    return sites, mva, edges, adj


def load_worlds(*, sites, edges):
    """Hourly residuals, noise units, labels, and the donor start hours of every training world."""
    worlds = []
    for w in SEEDS_TRAIN:
        z = np.load(ROOT675 / "worlds" / f"train_{w}.npz")
        truth = pl.read_parquet(ROOT675 / "worlds" / f"train_{w}_truth.parquet")
        lk, lg, nd = labels(truth=truth, sites=sites, edges=edges, hours=z["resid"].shape[1])
        centres = [r["start"] // 2 for r in truth.filter((pl.col("kind") == "edge") & (pl.col("amp_mw") < 0)).iter_rows(named=True)]  # fmt: skip
        worlds.append((z["resid"], z["sigma"], lk, lg, nd, centres))
    return worlds


def batch(*, worlds, mask, mva, rng):
    """One batch of inputs and labels."""
    hs, ds, lks, lgs, nds, wts = [], [], [], [], [], []
    for _ in range(BATCH):
        resid, sigma, lk, lg, nd, centres = worlds[rng.integers(len(worlds))]
        if centres and rng.random() < 0.5:
            c = centres[rng.integers(len(centres))]
            s = int(np.clip(c - rng.integers(300, SPAN - 400), 0, TRAIN_HOURS - SPAN)) // 24 * 24
        else:
            s = int(rng.integers(0, (TRAIN_HOURS - SPAN) // 24)) * 24
        h, d = window_inputs(resid=resid, start=s, span=SPAN, sigma=sigma, mva=mva)
        hs.append(h.float()); ds.append(d.float())  # fmt: skip
        lks.append(lk[:, s : s + SPAN]); lgs.append(lg[:, s : s + SPAN]); nds.append(nd[:, s : s + SPAN])  # fmt: skip
        wts.append(mask[:, s : s + SPAN])
    t = lambda a: torch.tensor(np.stack(a)).to(DEV)  # noqa: E731
    return torch.stack(hs).to(DEV), torch.stack(ds).to(DEV), t(lks).long(), t(lgs).long(), t(nds).long(), t(wts), hs  # fmt: skip


def loss_of(*, net, hourly, daily, adj, mva, lk, lg, nd, real_mask, edges):
    """Weighted cross-entropies; `real_mask` is per node, so a link is masked if either end is."""
    link, lag, node = net(hourly, daily, adj, mva)
    ia, ib = net.ia, net.ib
    off_w = 1.0 - (real_mask[:, ia] | real_mask[:, ib]).float()
    w = torch.where(lk > 0, torch.full_like(off_w, W_TRANSFER), off_w)
    ce = torch.nn.functional.cross_entropy(link.permute(0, 3, 1, 2), lk, reduction="none")
    l_link = (ce * w).sum() / w.sum().clamp(min=1)
    on = lk > 0
    if on.any():
        donor_lag = torch.where((lk == 1)[..., None], lag[..., 0, :], lag[..., 1, :])
        l_lag = torch.nn.functional.cross_entropy(donor_lag[on], lg[on])
    else:
        l_lag = link.sum() * 0
    nw = 1.0 - real_mask.float() * (nd == 0).float()
    l_node = (torch.nn.functional.cross_entropy(node.permute(0, 3, 1, 2), nd, reduction="none") * nw).sum() / nw.sum().clamp(min=1)  # fmt: skip
    return l_link + 0.3 * l_lag + 0.5 * l_node


def train(*, arm: str, seed: int, steps: int) -> Path:
    """Train and save `weights/<arm>_seed<k>.pt`."""
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

    rig = Rig()
    sites, mva, edges, adj = graph_of(rig=rig, arm=arm)
    worlds = load_worlds(sites=sites, edges=edges)
    mask = np.load(WORLDS / "real_e1_mask.npy")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EdgeStateNet(edges=edges, no_links=arm == "G-0").to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    out = ROOT / "weights"
    out.mkdir(parents=True, exist_ok=True)
    final, ckpt = out / f"{arm}_seed{seed}.pt", out / f"ckpt_{arm}_seed{seed}.pt"
    if final.exists():
        return final
    step = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=DEV)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]  # fmt: skip
        rng = np.random.default_rng(seed + step)
    adj_d, mva_d = adj.float().to(DEV), mva.float().to(DEV)
    t0 = time.time()
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        h, d, lk, lg, nd, rm, _ = batch(worlds=worlds, mask=mask, mva=mva, rng=rng)
        loss = loss_of(net=net, hourly=h, daily=d, adj=adj_d, mva=mva_d, lk=lk, lg=lg, nd=nd, real_mask=rm, edges=edges)  # fmt: skip
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"{arm} seed {seed} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)  # fmt: skip
        if step % 1000 == 0 or step == steps:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt)
    torch.save(net.state_dict(), final)
    ckpt.unlink(missing_ok=True)
    return final


if __name__ == "__main__":
    print(train(arm=sys.argv[1], seed=int(sys.argv[2]), steps=int(sys.argv[3])))
