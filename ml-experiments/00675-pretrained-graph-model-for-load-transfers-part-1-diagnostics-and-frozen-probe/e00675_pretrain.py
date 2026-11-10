"""Pretraining of the encoder of experiment 675 on masked windows of the real residual (GPU).

    export DATA_PATH_INTERNAL=<data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation
    uv run --no-sync python $P/00675-*/e00675_pretrain.py {masked|causal} <seed> <steps>

The masked arm (P-M) reconstructs the residual under masked windows (Student-t location and scale
per hour, and the mean of fully masked days); the causal arm (P-C) predicts the mean of the next 6
hours from causal convolutions and past statistics only. Rows from 2024-01-01 on are never read.
Checkpoints every 1,000 steps; a rerun resumes.
"""

# ruff: noqa: ANN201, D205, E501, E702  # research code run once; behaviour fixed by the results

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from e00653_common import SCORED_FROM
from e00675_inputs import causalise, node_inputs
from e00675_net import Pretrainer
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe"
)
SPAN: Final[int] = 2880
"""Hours per training sample (120 days)."""
PRE_HOURS: Final[int] = SCORED_FROM // 2
"""Hours before 2024-01-01: the only hours pretraining may read."""
T_DF: Final[float] = 4.0
BATCH: Final[int] = 4


def hourly_residual(
    *, rig: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real residual in noise units (hourly), noise units, MVA flags, and the neighbour matrix."""
    sites = rig.p653.sites  # ty: ignore[unresolved-attribute]
    sigma = np.array(
        [rig.frozen.sigma[s] if np.isfinite(rig.frozen.sigma[s]) else 1.0 for s in sites]  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    rows = []
    for i, s in enumerate(sites):
        rows.append(hourly_of(x=rig.real_residual[s] / sigma[i]))  # ty: ignore[unresolved-attribute]
    mva = torch.tensor([float(rig.panel.is_mva[s]) for s in sites], dtype=torch.float64)  # ty: ignore[unresolved-attribute]
    return (
        torch.tensor(np.stack(rows), dtype=torch.float64),
        torch.tensor(sigma, dtype=torch.float64),
        mva,
        adjacency(rig=rig),
    )


def hourly_of(*, x: np.ndarray) -> np.ndarray:
    """Mean of each pair of half-hours (NaN only when both are missing)."""
    n = len(x) // 2 * 2
    pair = x[:n].reshape(-1, 2)
    ok = np.isfinite(pair)
    s = np.where(ok, pair, 0.0).sum(1)
    c = ok.sum(1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def adjacency(*, rig: object) -> torch.Tensor:
    """Row-normalised neighbour matrix over `rig.p653.sites`."""
    sites = rig.p653.sites  # ty: ignore[unresolved-attribute]
    a = torch.zeros(len(sites), len(sites), dtype=torch.float64)
    for i, s in enumerate(sites):
        for n in rig.p653.neighbours.get(s, []):  # ty: ignore[unresolved-attribute]
            if n in sites:
                a[i, sites.index(n)] = 1.0
    return a / a.sum(1, keepdim=True).clamp(min=1)


def synthetic_panel(*, gen: torch.Generator, n_nodes: int, hours: int):
    """A random residual panel (with gaps) for the leak tests."""
    x = torch.randn(n_nodes, hours, generator=gen, dtype=torch.float64) * 2
    x[torch.rand(n_nodes, hours, generator=gen, dtype=torch.float64) < 0.02] = float("nan")
    sigma = 0.2 + torch.rand(n_nodes, generator=gen, dtype=torch.float64)
    is_mva = (torch.rand(n_nodes, generator=gen) < 0.5).to(torch.float64)
    return x, sigma, is_mva


def draw_masks(
    *, gen: torch.Generator, n_nodes: int, hours: int, centre: tuple[int, int]
) -> torch.Tensor:
    """Masked hours (True) per node: 40% of nodes get 1-3 windows of 3-24 h and, half of those, a
    window of 1-7 whole days; in 30% of samples each window is also masked at 1-2 other nodes.
    """
    mask = torch.zeros(n_nodes, hours, dtype=torch.bool)
    lo, hi = centre
    co = torch.rand(1, generator=gen).item() < 0.3
    for i in range(n_nodes):
        if torch.rand(1, generator=gen).item() >= 0.4:
            continue
        wins = []
        for _ in range(int(torch.randint(1, 4, (1,), generator=gen))):
            length = int(torch.randint(3, 25, (1,), generator=gen))
            start = int(torch.randint(lo, max(lo + 1, hi - length), (1,), generator=gen))
            wins.append((start, start + length))
        if torch.rand(1, generator=gen).item() < 0.5:
            days = int(torch.randint(1, 8, (1,), generator=gen))
            start = lo // 24 * 24 + 24 * int(
                torch.randint(0, max(1, (hi - lo) // 24 - days), (1,), generator=gen)
            )
            wins.append((start, start + 24 * days))
        for a, b in wins:
            mask[i, a:b] = True
            if co and n_nodes > 2:
                for j in torch.randperm(n_nodes, generator=gen)[
                    : int(torch.randint(1, 3, (1,), generator=gen))
                ]:
                    mask[int(j), a:b] = True
    return mask


def student_t_nll(*, y: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a Student-t with `T_DF` degrees of freedom."""
    z = (y - loc) / log_scale.exp()
    return log_scale + 0.5 * (T_DF + 1) * torch.log1p(z * z / T_DF)


def build_sample(
    *,
    x_all: torch.Tensor,
    start: int,
    mask: torch.Tensor,
    sigma: torch.Tensor,
    mva: torch.Tensor,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs of one masked sample: statistics are computed after the mask is applied."""
    x = x_all[:, start : start + SPAN].clone()
    mask = mask & torch.isfinite(x)
    seen = torch.where(mask, torch.full_like(x, float("nan")), x)
    h, d = node_inputs(x=seen, sigma=sigma, is_mva=mva, hour0=start)
    if causal:
        h, d = causalise(h, d)
    return h, d, x, mask


def make_batch(
    *,
    x_all: torch.Tensor,
    sigma: torch.Tensor,
    mva: torch.Tensor,
    gen: torch.Generator,
    causal: bool,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Inputs, targets, and loss masks of one batch."""
    n = x_all.shape[0]
    hourly, daily, truth, masks = [], [], [], []
    for _ in range(BATCH):
        s = 24 * int(torch.randint(0, (PRE_HOURS - SPAN) // 24, (1,), generator=gen))
        mask = (
            torch.zeros(n, SPAN, dtype=torch.bool)
            if causal
            else draw_masks(gen=gen, n_nodes=n, hours=SPAN, centre=(360, SPAN - 360))
        )
        h, d, x, mask = build_sample(
            x_all=x_all, start=s, mask=mask, sigma=sigma, mva=mva, causal=causal
        )
        hourly.append(h.float()); daily.append(d.float()); truth.append(x); masks.append(mask)  # fmt: skip
    return (
        torch.stack(hourly).to(device), torch.stack(daily).to(device),
        torch.stack(truth).to(device), torch.stack(masks).to(device),
    )  # fmt: skip


def loss_of(
    *,
    model: Pretrainer,
    hourly: torch.Tensor,
    daily: torch.Tensor,
    truth: torch.Tensor,
    masks: torch.Tensor,
    adj: torch.Tensor,
) -> torch.Tensor:
    """The reconstruction loss (masked arm) or the next-6-hours loss (causal arm)."""
    out, day = model(hourly, daily, adj)
    loc, log_scale = out[..., 0], out[..., 1].clamp(-4, 4)
    y = truth.clamp(-30, 30).float()
    t = torch.arange(SPAN, device=y.device)
    inside = (t >= 360) & (t < SPAN - 360)
    if model.causal:
        csum = torch.cat([torch.zeros_like(y[..., :1]), torch.nan_to_num(y).cumsum(-1)], -1)
        ok = torch.isfinite(truth).float()
        ccnt = torch.cat([torch.zeros_like(ok[..., :1]), ok.cumsum(-1)], -1)
        idx = (t + 7).clamp(max=SPAN)
        cnt = ccnt[..., idx] - ccnt[..., t + 1]
        mean = (csum[..., idx] - csum[..., t + 1]) / cnt.clamp(min=1)
        valid = (cnt >= 3) & inside[None, None]
        nll = ((mean - loc) / log_scale.exp()) ** 2 * 0.5 + log_scale
        return nll[valid].mean()
    sel = masks & inside[None, None] & torch.isfinite(truth)
    hour_loss = student_t_nll(y=torch.nan_to_num(y), loc=loc, log_scale=log_scale)[sel].mean()
    b, n, _ = masks.shape
    day_masked = masks.reshape(b, n, SPAN // 24, 24).all(-1) & torch.isfinite(truth).reshape(
        b, n, SPAN // 24, 24
    ).all(-1)
    dtrue = torch.nan_to_num(y).reshape(b, n, SPAN // 24, 24).mean(-1) * 0.2
    if day_masked.any():
        return hour_loss + 0.5 * nn_huber(day[day_masked], dtrue[day_masked])
    return hour_loss


def nn_huber(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Huber loss."""
    return torch.nn.functional.huber_loss(a, b)


def train(*, arm: str, seed: int, steps: int) -> Path:
    """Pretrain and save `encoder_<arm>_seed<k>.pt`; returns its path."""
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rig = Rig()
    x_all, sigma, mva, adj = hourly_residual(rig=rig)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Pretrainer(causal=arm == "causal").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    out = ROOT / "weights"
    out.mkdir(parents=True, exist_ok=True)
    final, ckpt = out / f"encoder_{arm}_seed{seed}.pt", out / f"ckpt_{arm}_seed{seed}.pt"
    step = 0
    if final.exists():
        return final
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); step = state["step"]  # fmt: skip
        gen.manual_seed(seed + step)
    adj_d = adj.float().to(device)
    t0 = time.time()
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        h, d, tr, mk = make_batch(
            x_all=x_all, sigma=sigma, mva=mva, gen=gen, causal=arm == "causal", device=device
        )
        loss = loss_of(model=model, hourly=h, daily=d, truth=tr, masks=mk, adj=adj_d)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(
                f"{arm} seed {seed} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % 1000 == 0 or step == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt)
    torch.save(model.state_dict(), final)
    ckpt.unlink(missing_ok=True)
    return final


if __name__ == "__main__":
    print(train(arm=sys.argv[1], seed=int(sys.argv[2]), steps=int(sys.argv[3])))
