"""Build injected-world banks and train the donor-finding networks of experiment 672.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    P=ml-experiments  # then PYTHONPATH as in e00672_run.py
    uv run --no-sync python e00672_train.py bank <mix> [solar]
    uv run --no-sync python e00672_train.py train <mix> <arm> <seed>

`mix` names a generator configuration of experiment 660: `default`, `constant` (constant-MW square
steps), or `shaped` (shaped transfers). `arm` is `G` (real graph), `GR` (rewired graph), `G0` (no
links), or `GS` (real graph, explicit solar term added to the residual; its bank is built with
`solar`). Training worlds start events in 2022-01-01 to 2023-06-30 and show nothing later;
validation worlds start events up to 2023-12-31 and the loss counts only blocks from 2023-07-01.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from functools import cache
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00653_common import SCORED_FROM
from e00653_run import rewire
from e00658_data import UNKNOWN_WEIGHT, graph_arrays
from e00672_data import (
    TRAIN_BLOCKS,
    TRAIN_END,
    VAL_BLOCKS,
    DWorld,
    build_world,
    inject_world,
    rig,
)
from e00672_net import DonorNet, loss_fn
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00672-graph-model-that-finds-the-donor-of-a-load-transfer"
MIXES: Final[dict[str, gen.GeneratorConfig]] = {
    "default": gen.default_config(),
    "constant": gen.constant_mw_square(),
    "shaped": gen.shaped_transfers(),
}
SEED0: Final[dict[str, int]] = {"default": 100_000, "constant": 200_000, "shaped": 300_000}
"""First seed of each mix's banks. Scored worlds use seeds 900 and up, far from these."""
N_TRAIN_WORLDS: Final[int] = 120
N_VAL_WORLDS: Final[int] = 6
CROP: Final[int] = 720
BATCH: Final[int] = 8
STEPS: Final[int] = int(os.environ.get("E672_STEPS", "2400"))
"""Training steps; the environment variable only shortens a smoke test."""
EVAL_EVERY: Final[int] = min(200, STEPS)
ARMS: Final[tuple[str, ...]] = ("G", "GR", "G0", "GS")


def solar_term() -> dict[str, np.ndarray]:
    """Experiment 657's explicit solar term (known CAMS and ERA5 weather, capacity table of 649)."""
    from e00657_solar import build

    return build().delta["solar"]


@cache
def _solar() -> dict[str, np.ndarray]:
    """The explicit solar term, built once per worker process."""
    return solar_term()


def _make(args: tuple[str, int, str, bool]) -> DWorld:
    mix, seed, part, solar = args
    term = _solar() if solar else None
    upto = TRAIN_END if part == "train" else SCORED_FROM
    w = inject_world(config=MIXES[mix], seed=seed, upto=upto)
    out = build_world(world=w, solar=term)
    out.residual = {}
    return out


def make_bank(*, mix: str, solar: bool = False) -> None:
    """Generate the training and validation worlds of one generator mix."""
    WORK.mkdir(parents=True, exist_ok=True)
    seed0 = SEED0[mix]
    jobs = [(mix, seed0 + k, "train", solar) for k in range(N_TRAIN_WORLDS)]
    jobs += [(mix, seed0 + 5_000 + k, "val", solar) for k in range(N_VAL_WORLDS)]
    t = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        worlds = pool.map(_make, jobs, chunksize=4)
    print("bank", mix, len(worlds), "worlds", round(time.time() - t), "s", flush=True)
    for w in worlds[:N_TRAIN_WORLDS]:
        for name in ("feats", "slow", "cls", "weight", "amp", "edge_state", "edge_weight", "donor"):
            a = getattr(w, name)
            setattr(w, name, a[:, :TRAIN_BLOCKS])
    name = f"bank_{mix}{'_solar' if solar else ''}.pkl"
    with (WORK / name).open("wb") as f:
        pickle.dump({"train": worlds[:N_TRAIN_WORLDS], "val": worlds[N_TRAIN_WORLDS:]}, f)


def graph_of(*, arm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour matrix and links an arm trains and runs on."""
    p = rig().p653
    if arm == "G0":
        return np.zeros((len(p.sites), len(p.sites)), np.float32), np.zeros((0, 2), np.int64)
    if arm == "GR":
        r = np.random.default_rng(7_000 + seed)
        real = p.neighbours
        best = None
        for _ in range(300):
            nb = rewire(edges=p.edges, rng=r)
            nb = {s: nb.get(s, []) for s in p.sites}
            kept = sum(b in real[a] for a in nb for b in nb[a]) / max(1, sum(map(len, nb.values())))
            if best is None or kept < best[1]:
                best = (nb, kept)
        assert best is not None
        return graph_arrays(panel=p, neighbours=best[0])
    return graph_arrays(panel=p)


def retarget(*, w: DWorld, edges: np.ndarray) -> None:
    """Rebuild the link targets of a banked world for the links of the arm's own graph.

    The bank stores link state per real link. A rewired graph has other pairs, so each of its
    links takes the real state if the pair is a real link and 'off' otherwise.
    """
    assert w.weight is not None and w.edge_state is not None
    p = rig().p653
    index = {s: i for i, s in enumerate(p.sites)}
    real = {(index[a], index[b]): k for k, (a, b) in enumerate(p.edges)}
    covered = (w.weight > 0.5) & (w.weight > 0)
    present = w.weight > 0
    state = np.zeros((len(edges), w.edge_state.shape[1]), np.int8)
    weight = np.zeros(state.shape, np.float32)
    for k, (a, b) in enumerate(edges):
        i, j = int(a), int(b)
        if (i, j) in real:
            state[k] = w.edge_state[real[(i, j)]]
        elif (j, i) in real:
            flip = w.edge_state[real[(j, i)]]
            state[k] = np.where(flip == 0, 0, 3 - flip)
        known = covered[i] | covered[j]
        weight[k] = np.where(known, 1.0, UNKNOWN_WEIGHT) * present[i] * present[j]
    w.edge_state, w.edge_weight = state, weight


def _tensors(w: DWorld, sl: slice, dev: torch.device) -> list[torch.Tensor]:
    arrs = [w.feats, w.cls, w.weight, w.amp, w.donor, w.edge_state, w.edge_weight]
    return [
        torch.from_numpy(np.ascontiguousarray(a[:, sl])).to(dev)  # ty: ignore[not-subscriptable]
        for a in arrs
    ]


def train(*, mix: str, arm: str, seed: int) -> str:
    """Train one network; keep the checkpoint with the lowest validation loss.

    Returns:
        The tag under which the checkpoint and log were saved.
    """
    tag = f"{arm}_{mix}_seed{seed}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    with (WORK / f"bank_{mix}{'_solar' if arm == 'GS' else ''}.pkl").open("rb") as f:
        bank = pickle.load(f)
    nbr, edges = graph_of(arm=arm, seed=seed)
    if arm in ("GR", "G0"):
        for w in bank["train"] + bank["val"]:
            retarget(w=w, edges=edges)
    nbr_t, edges_t = torch.from_numpy(nbr).to(dev), torch.from_numpy(edges).to(dev)
    val_sl = slice(VAL_BLOCKS - 960, VAL_BLOCKS)
    val = []
    for w in bank["val"]:
        t = _tensors(w, val_sl, dev)
        mask = torch.zeros(t[2].shape[1], device=dev)
        mask[TRAIN_BLOCKS - (VAL_BLOCKS - 960) :] = 1.0
        t[2] = t[2] * mask[None, :]
        t[6] = t[6] * mask[None, :]
        val.append(t)
    net = DonorNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS)
    best, hist = float("inf"), []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        net.train()
        batch = []
        for _ in range(BATCH):
            w = bank["train"][int(rng.integers(len(bank["train"])))]
            o = int(rng.integers(0, TRAIN_BLOCKS - CROP + 1))
            batch.append(_tensors(w, slice(o, o + CROP), dev))
        x, cls, wt, amp, dn, es, ew = (torch.stack([b[k] for b in batch]) for k in range(7))
        out = net(x=x, nbr=nbr_t, edges=edges_t)
        loss, parts = loss_fn(
            out=out, cls=cls, weight=wt, amp=amp, donor=dn, edge_state=es, edge_weight=ew
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                vl = [
                    float(
                        loss_fn(
                            out=net(x=t[0][None], nbr=nbr_t, edges=edges_t),
                            cls=t[1][None],
                            weight=t[2][None],
                            amp=t[3][None],
                            donor=t[4][None],
                            edge_state=t[5][None],
                            edge_weight=t[6][None],
                        )[0]
                    )
                    for t in val
                ]
            v = float(np.mean(vl))
            hist.append({"step": step, "train": float(loss), "val": v, **parts})
            print(tag, hist[-1], round(time.time() - t0), "s", flush=True)
            if v < best or step == EVAL_EVERY:
                best = v
                torch.save(net.state_dict(), WORK / f"net_{tag}.pt")
    (WORK / f"train_log_{tag}.json").write_text(json.dumps(hist, indent=1))
    print("done", tag, "best val", best, "seconds", round(time.time() - t0), flush=True)
    return tag


if __name__ == "__main__":
    if sys.argv[1] == "bank":
        make_bank(mix=sys.argv[2], solar=len(sys.argv) > 3 and sys.argv[3] == "solar")
    else:
        train(mix=sys.argv[2], arm=sys.argv[3], seed=int(sys.argv[4]))
