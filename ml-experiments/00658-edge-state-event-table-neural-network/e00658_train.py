"""Build injected-world banks and train the edge-state network of experiment 658.

    DATA_PATH_INTERNAL=... PYTHONPATH=<folder of experiment 653>:. \
        uv run --no-sync python e00658_train.py bank <mix>
    ... e00658_train.py train <mix> <seed>

`mix` is `main` (experiment 653's default generator), `g1` (constant megawatts, square edges), or
`g2` (shares of the donor's power, wide edges, loose shares, staggered starts).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
import time
from typing import Final

import numpy as np
import torch
from e00653_common import WORK as WORK_653
from e00653_common import Panel
from e00653_core import freeze
from e00653_inject import GenParams
from e00658_data import (
    SCORED_FROM,
    TRAIN_BLOCKS,
    TRAIN_END,
    UNKNOWN_WEIGHT,
    VAL_BLOCKS,
    World,
    build_world,
    graph_arrays,
)
from e00658_net import EdgeStateNet, loss_fn
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final = ML_EXPERIMENTS_DATA_DIR / "00658-edge-state-event-table-neural-network"
MIXES: Final[dict[str, GenParams]] = {
    "main": GenParams(),
    "g1": GenParams(shape_prop=0.0, width_hh=(0, 0)),
    "main2": GenParams(),
    "g2": GenParams(shape_prop=1.0, share=(0.2, 1.8), start_jitter_hh=24, width_hh=(0, 24)),
}
N_TRAIN_WORLDS: Final[int] = 120
N_VAL_WORLDS: Final[int] = 6
CROP: Final[int] = 720
BATCH: Final[int] = 8
STEPS: Final[int] = 2400
EVAL_EVERY: Final[int] = 200


def setup() -> tuple[Panel, dict[str, np.ndarray], dict[str, float]]:
    """Panel, residual with the common mode removed, and noise units (as in experiment 653)."""
    panel = Panel()
    res = panel.mask_common_mode(residual=panel.residual)
    return panel, res, freeze(panel=panel, residual=res).sigma


def _make(args: tuple[str, int, str]) -> World:
    mix, seed, part = args
    mix = "main" if mix == "main2" else mix
    panel, res, sigma = setup()
    lo, hi, upto = (
        (0, TRAIN_END, TRAIN_END) if part == "train" else (TRAIN_END, SCORED_FROM, SCORED_FROM)
    )
    return build_world(
        panel=panel,
        residual=res,
        sigma=sigma,
        rng=np.random.default_rng(seed),
        lo=lo,
        hi=hi,
        gen=MIXES[mix],
        upto=upto,
    )


def _slim(w: World) -> World:
    w.residual = {}
    return w


def make_bank(*, mix: str) -> None:
    """Generate the training and validation worlds of one generator mix."""
    WORK.mkdir(parents=True, exist_ok=True)
    seed0 = {"main": 10_000, "g1": 20_000, "g2": 30_000, "main2": 40_000}[mix]
    n_train = 300 if mix == "main2" else N_TRAIN_WORLDS
    jobs = [(mix, seed0 + k, "train") for k in range(n_train)]
    jobs += [(mix, seed0 + 5_000 + k, "val") for k in range(N_VAL_WORLDS)]
    t = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        worlds = [_slim(w) for w in pool.map(_make, jobs, chunksize=4)]
    print("bank", mix, len(worlds), "worlds", round(time.time() - t), "s", flush=True)
    with (WORK / f"bank_{mix}.pkl").open("wb") as f:
        for w in worlds[:n_train]:
            for name in ("feats", "slow", "cls", "weight", "amp", "edge_state", "edge_weight"):
                a = getattr(w, name)
                setattr(w, name, a[:, :TRAIN_BLOCKS] if a is not None else a)
        pickle.dump({"train": worlds[:n_train], "val": worlds[n_train:]}, f)


def _uninjected(
    *, panel: Panel, res: dict[str, np.ndarray], sigma: dict[str, float], upto: int
) -> World:
    w = build_world(
        panel=panel, residual=res, sigma=sigma, rng=None, lo=0, hi=0, gen=None, upto=upto
    )
    present = w.feats[:, :, 1] > 0
    w.cls = np.zeros(present.shape, np.int8)
    w.weight = (UNKNOWN_WEIGHT * present).astype(np.float32)
    w.amp = np.zeros(present.shape, np.float32)
    ea = np.array([panel.sites.index(a) for a, _ in panel.edges])
    eb = np.array([panel.sites.index(b) for _, b in panel.edges])
    w.edge_state = np.zeros((len(ea), present.shape[1]), np.int8)
    w.edge_weight = (UNKNOWN_WEIGHT * present[ea] * present[eb]).astype(np.float32)
    w.residual = {}
    return w


def _tensors(w: World, sl: slice, dev: torch.device) -> list[torch.Tensor]:
    arrs = [w.feats, w.cls, w.weight, w.amp, w.edge_state, w.edge_weight]
    return [torch.from_numpy(np.ascontiguousarray(a[:, sl])).to(dev) for a in arrs if a is not None]


def train(*, mix: str, seed: int, tag: str | None = None) -> None:  # noqa: PLR0915
    """Train one network; keep the checkpoint with the lowest validation loss.

    A `mix` starting with `v2` is the second look: 300 worlds from the `main` generator, a smaller
    network with more dropout, more weight decay, and Gaussian noise added to the inputs.
    """
    v2 = mix.startswith("v2")
    bank_name = "main2" if v2 else mix
    tag = tag or f"{mix}_seed{seed}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    panel, res, sigma = setup()
    with (WORK / f"bank_{bank_name}.pkl").open("rb") as f:
        bank = pickle.load(f)
    nbr, edges = graph_arrays(panel=panel)
    nbr_t, edges_t = torch.from_numpy(nbr).to(dev), torch.from_numpy(edges).to(dev)
    clean = _uninjected(panel=panel, res=res, sigma=sigma, upto=TRAIN_END)
    train_worlds = bank["train"]
    val_sl = slice(VAL_BLOCKS - 960, VAL_BLOCKS)
    val = []
    for w in bank["val"] + [_uninjected(panel=panel, res=res, sigma=sigma, upto=SCORED_FROM)]:
        t = _tensors(w, val_sl, dev)
        mask = torch.zeros(t[2].shape[1], device=dev)
        mask[TRAIN_BLOCKS - (VAL_BLOCKS - 960) :] = 1.0
        t[2] = t[2] * mask[None, :]
        t[5] = t[5] * mask[None, :]
        val.append(t)
    net = (EdgeStateNet(hidden=32, dropout=0.3) if v2 else EdgeStateNet()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2 if v2 else 1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS)
    best, hist = float("inf"), []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        net.train()
        batch = []
        for _ in range(BATCH):
            w = clean if rng.random() < 0.2 else train_worlds[int(rng.integers(len(train_worlds)))]
            o = int(rng.integers(0, TRAIN_BLOCKS - CROP + 1))
            batch.append(_tensors(w, slice(o, o + CROP), dev))
        x, cls, wt, amp, es, ew = (torch.stack([b[k] for b in batch]) for k in range(6))
        if v2:
            noise = torch.randn_like(x) * 0.3
            noise[..., 1] = 0
            noise[..., 5:] = 0
            x = x + noise
        out = net(x=x, nbr=nbr_t, edges=edges_t)
        loss, parts = loss_fn(out=out, cls=cls, weight=wt, amp=amp, edge_state=es, edge_weight=ew)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                vl = []
                for t in val:
                    o = net(x=t[0][None], nbr=nbr_t, edges=edges_t)
                    vl.append(
                        float(
                            loss_fn(
                                out=o,
                                cls=t[1][None],
                                weight=t[2][None],
                                amp=t[3][None],
                                edge_state=t[4][None],
                                edge_weight=t[5][None],
                            )[0]
                        )
                    )
            v = float(np.mean(vl))
            hist.append(
                {
                    "step": step,
                    "train": float(loss),
                    "val": v,
                    **parts,
                    "s": round(time.time() - t0),
                }
            )
            print(hist[-1], flush=True)
            if v < best:
                best = v
                torch.save(net.state_dict(), WORK / f"net_{tag}.pt")
    (WORK / f"train_log_{tag}.json").write_text(json.dumps(hist, indent=1))
    print("done", tag, "best val", best, "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "bank":
        make_bank(mix=sys.argv[2])
    else:
        train(mix=sys.argv[2], seed=int(sys.argv[3]))
    assert WORK_653 is not None
