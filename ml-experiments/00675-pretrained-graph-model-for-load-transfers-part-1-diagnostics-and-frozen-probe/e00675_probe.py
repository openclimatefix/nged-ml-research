"""The frozen-encoder probe of experiment 675 (X4-1).

    uv run --no-sync python e00675_probe.py <arm> <seed>      # arm: P-M, P-C, R, F

For each arm and seed, a logistic probe is trained on 200 injected worlds and read on the 40
scored worlds. The donor probe asks whether an hour lies in the first 24 hours of a donor step; the
recipient probe asks, for a true donor onset and a linked neighbour, whether the neighbour is the
recipient. Encoder arms are frozen (random initialisation for R). Arm F uses the node input
channels alone. Output: `probe/<arm>_seed<k>_{donor,recipient}.parquet`.
"""

# ruff: noqa: ANN201, B023, E501, E702, PLR0915  # research code run once; behaviour fixed by the results

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from e00653_common import DAY, SCORED_FROM
from e00665_score import cell_of  # ty: ignore[unresolved-import]
from e00675_inputs import causalise, node_inputs
from e00675_net import Pretrainer
from e00675_pretrain import ROOT, SPAN, adjacency

WORLDS: Final[Path] = ROOT / "worlds"
OUT: Final[Path] = ROOT / ("probe_smoke" if os.environ.get("E675_SMOKE") else "probe")
HOURS: Final[int] = 39408
SCORED_H: Final[int] = SCORED_FROM // 2
SMOKE: Final[bool] = bool(os.environ.get("E675_SMOKE"))
DEV: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step6_channel() -> int:
    """Index of the forward 6-hour step channel in `node_inputs`."""
    return 2 + 6 * 3  # gap windows come first; the 6-hour forward step is the next channel


def donor_rows(*, truth: pl.DataFrame) -> pl.DataFrame:
    """Donor steps of edge transfers (negative step, at least one linked recipient)."""
    return truth.filter(
        (pl.col("kind") == "edge") & (pl.col("amp_mw") < 0) & (pl.col("partners") != "")
    )


def embed(
    *,
    model: Pretrainer | None,
    resid: np.ndarray,
    sigma: np.ndarray,
    mva: torch.Tensor,
    adj: torch.Tensor,
):
    """Per-hour features (nodes, hours, d) and the raw channels (nodes, hours, C) of one world.

    The record is cut into windows of `SPAN` hours (the length pretraining saw), each encoded on
    its own; the middle half of every window is kept, and the first and last windows keep their
    outer quarter too.
    """
    x = torch.tensor(resid, dtype=torch.float64)
    n_nodes, total = x.shape
    quarter = SPAN // 4
    starts = list(range(0, total - SPAN + 1, SPAN // 2))
    if starts[-1] + SPAN < total:
        starts.append(total - SPAN)
    raw = torch.zeros(n_nodes, total, 35, dtype=torch.float32)
    feat = None if model is None else torch.zeros(n_nodes, total, 64, dtype=torch.float32)
    for k, s in enumerate(starts):
        lo = 0 if k == 0 else s + quarter
        hi = total if k == len(starts) - 1 else s + SPAN - quarter
        lo_i, hi_i = lo - s, hi - s
        hourly, daily = node_inputs(
            x=x[:, s : s + SPAN], sigma=torch.tensor(sigma), is_mva=mva, hour0=s
        )
        raw[:, lo:hi] = hourly[
            :, lo_i:hi_i
        ].float()  # pair features use the full inputs in every arm
        if model is not None and model.causal:
            hourly, daily = causalise(hourly, daily)
        if model is not None:
            with torch.no_grad():
                h, _ = model.encoder(
                    hourly.float()[None].to(DEV), daily.float()[None].to(DEV), adj.float().to(DEV)
                )
            feat[:, lo:hi] = h[0, :, lo_i:hi_i].cpu()  # ty: ignore[invalid-assignment]
    return feat, raw


def pair_block(*, raw: torch.Tensor, donor: int, nb: int, t0: int) -> torch.Tensor:
    """19 lagged forward steps of the neighbour (sign flipped), the donor's step, and the ratio."""
    c = step6_channel()
    nbs = -raw[nb, t0 : t0 + 19, c] if t0 + 19 <= raw.shape[1] else torch.zeros(19)
    ds = raw[donor, t0, c]
    ratio = (nbs.max() / ds.abs().clamp(min=1e-3)).clamp(-5, 5)
    return torch.cat([nbs, ds[None], ratio[None]])


def features(
    *, feat: torch.Tensor | None, raw: torch.Tensor, adj: torch.Tensor, idx: tuple[int, int]
) -> torch.Tensor:
    """Feature vector of one (node, hour): the embedding, or the raw channels with neighbour mean."""
    i, t = idx
    if feat is not None:
        return feat[i, t]
    nb = (adj[i].float()[:, None] * raw[:, t]).sum(0)
    return torch.cat([raw[i, t], nb])


def fit_logistic(
    *, x: torch.Tensor, y: torch.Tensor, steps: int = 300
) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]:
    """Standardise, then fit an L2-regularised logistic regression with LBFGS."""
    mu, sd = x.mean(0), x.std(0).clamp(min=1e-6)
    xs = (x - mu) / sd
    lin = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xs).squeeze(-1), y)
        loss = loss + 1e-3 * (lin.weight**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin, mu, sd


def donor_scores(*, base: torch.Tensor, dl: tuple) -> torch.Tensor:
    """Donor-probe score of every (node, hour); a function of the features alone."""
    z = (base - dl[1]) / dl[2]
    return (z @ dl[0].weight.T.detach()).squeeze(-1) + dl[0].bias.detach()


def load_world(*, kind: str, seed: int):
    """Residual, noise units, and truth table of a cached world."""
    z = np.load(WORLDS / f"{kind}_{seed}.npz")
    return z["resid"], z["sigma"], pl.read_parquet(WORLDS / f"{kind}_{seed}_truth.parquet")


def run(*, arm: str, seed: int) -> None:
    """Train both probes on the training worlds and score the scored worlds."""
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

    rig = Rig()
    sites = rig.p653.sites
    adj = adjacency(rig=rig)
    mva = torch.tensor([float(rig.panel.is_mva[s]) for s in sites], dtype=torch.float64)
    index = {s: i for i, s in enumerate(sites)}
    nbrs = {
        i: [index[n] for n in rig.p653.neighbours.get(s, []) if n in index]
        for s, i in index.items()
    }
    model = None
    if arm != "F":
        torch.manual_seed(seed)
        model = Pretrainer(causal=arm == "P-C").to(DEV).eval()
        if arm != "R":
            name = {"P-M": "masked", "P-C": "causal"}[arm]
            model.load_state_dict(
                torch.load(ROOT / "weights" / f"encoder_{name}_seed{seed}.pt", map_location=DEV)
            )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xd, yd, xr, yr = [], [], [], []
    for w in range(100000, 100005 if SMOKE else 100200):
        resid, sigma, truth = load_world(kind="train", seed=w)
        feat, raw = embed(model=model, resid=resid, sigma=sigma, mva=mva, adj=adj)
        donors = donor_rows(truth=truth)
        near = np.zeros((len(sites), HOURS), bool)
        for r in donors.iter_rows(named=True):
            i, h0 = index[r["site"]], r["start"] // 2
            near[i, max(0, h0 - 48) : h0 + 72] = True
            for t in range(h0, min(HOURS, h0 + 24), 3):
                xd.append(features(feat=feat, raw=raw, adj=adj, idx=(i, t))); yd.append(1.0)  # fmt: skip
            for nb in nbrs[i]:
                t0 = h0 + 3
                pb = pair_block(raw=raw, donor=i, nb=nb, t0=t0)
                fe = lambda a: features(feat=feat, raw=raw, adj=adj, idx=(a, t0))  # noqa: E731
                xr.append(torch.cat([fe(i), fe(nb), pb])); yr.append(float(sites[nb] in r["partners"].split(",")))  # fmt: skip
        n_neg = 600
        ii = rng.integers(0, len(sites), n_neg * 3)
        tt = rng.integers(24, HOURS - 24, n_neg * 3)
        keep = [(a, b) for a, b in zip(ii, tt, strict=True) if not near[a, b] and raw[a, b, 1] > 0][
            :n_neg
        ]
        for a, b in keep:
            xd.append(features(feat=feat, raw=raw, adj=adj, idx=(int(a), int(b)))); yd.append(0.0)  # fmt: skip
    dl = fit_logistic(x=torch.stack(xd), y=torch.tensor(yd))
    rl = fit_logistic(x=torch.stack(xr), y=torch.tensor(yr))
    OUT.mkdir(parents=True, exist_ok=True)
    scores: list[dict] = []
    pairs: list[dict] = []
    for w in range(900, 903 if SMOKE else 940):
        resid, sigma, truth = load_world(kind="test", seed=w)
        feat, raw = embed(model=model, resid=resid, sigma=sigma, mva=mva, adj=adj)
        base = (
            feat
            if feat is not None
            else torch.cat([raw, torch.einsum("ij,jtc->itc", adj.float(), raw)], -1)
        )
        sc = donor_scores(base=base, dl=dl)
        np.save(OUT / f"score_{arm}_seed{seed}_world{w}.npy", sc.numpy().astype(np.float32))
        for r in donor_rows(truth=truth).iter_rows(named=True):
            days = None if r["end"] < 0 else (r["end"] - r["start"]) / DAY
            cell = cell_of(days=days, units=r["units"]) if r["units"] >= 2.0 else "under 2"
            scores.append(
                {
                    "world": w,
                    "site": r["site"],
                    "start": r["start"],
                    "amp_mw": r["amp_mw"],
                    "cell": cell,
                }
            )
            i, h0 = index[r["site"]], r["start"] // 2
            for nb in nbrs[i]:
                t0 = h0 + 3
                fe = lambda a: features(feat=feat, raw=raw, adj=adj, idx=(a, t0))  # noqa: E731
                x = torch.cat([fe(i), fe(nb), pair_block(raw=raw, donor=i, nb=nb, t0=t0)])
                s = float(
                    (((x - rl[1]) / rl[2]) @ rl[0].weight.T.detach()).squeeze()
                    + rl[0].bias.detach()
                )
                pairs.append({"world": w, "cell": cell, "amp_mw": r["amp_mw"], "label": int(sites[nb] in r["partners"].split(",")), "score": s})  # fmt: skip
    pl.DataFrame(scores).write_parquet(OUT / f"{arm}_seed{seed}_donors.parquet")
    pl.DataFrame(pairs).write_parquet(OUT / f"{arm}_seed{seed}_recipient.parquet")
    print(arm, seed, "done", len(scores), len(pairs), flush=True)


if __name__ == "__main__":
    run(arm=sys.argv[1], seed=int(sys.argv[2]))
