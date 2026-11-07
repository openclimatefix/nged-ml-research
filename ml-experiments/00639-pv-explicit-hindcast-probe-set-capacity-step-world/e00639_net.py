# ruff: noqa: E501, ANN001
"""The explicit-PV training loop and the quantile forecast of experiment 639.

`train_explicit` is experiment 634's `train` (same optimiser, batch, early stopping, row fraction, and
validation sample) with one change: the target is the gross target
`z = (y + C * inc * u - mean) / std`, where `C` is the capacity timeline in megawatts. During training
`C` carries a draw per epoch and a trainable offset per Primary. Validation uses the mean timeline and
the current offset. `forecast` returns net quantiles in points of capacity, with the same
`C * inc * u`
subtracted from every quantile.
"""

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00639_common as k
import numpy as np
import torch  # ty: ignore[unresolved-import]

N_PRIMARIES = 16


def parameter_tensors(*, params: dict, six: dict[str, int], device: str) -> dict[str, torch.Tensor]:
    """Return per-Primary tensors (length 16) of `k`, `d`, and their absolute spreads; zero outside the six."""
    out = {name: np.zeros(N_PRIMARIES) for name in ("k", "d", "sd_k", "sd_d", "six")}
    for site, idx in six.items():
        p = params[site]
        out["k"][idx], out["d"][idx] = p.k, p.d
        out["sd_k"][idx], out["sd_d"][idx] = p.rel_sd_k * p.k, p.rel_sd_d * p.d
        out["six"][idx] = 1.0
    return {name: torch.tensor(v, dtype=torch.float32, device=device) for name, v in out.items()}


def capacity_mw(
    *, t: dict[str, torch.Tensor], delta: torch.Tensor, p, r4, after, ek=None, ed=None
) -> torch.Tensor:
    """Return the capacity timeline (MW) of the rows with Primary index `p`, floored at zero."""
    kk = t["k"] if ek is None else t["k"] + ek
    dd = t["d"] if ed is None else t["d"] + ed
    return torch.relu(kk[p] * r4 + dd[p] * after + delta[p] * t["six"][p])


def train_explicit(
    *, model, x, y, u_inc, r4, after, p, is_val, mean, std, tensors, seed: int, lr: float, device: str,
) -> tuple[int, float, torch.Tensor]:  # fmt: skip
    """Train the network on the gross target; return (epochs, best validation NLL, the offsets).

    `y`, `u_inc`, `r4`, `after`, and `p` are per-row tensors on `device`. `mean` and `std` are the
    per-Primary net-target statistics (length 16), the same as the free network's.
    """
    torch.manual_seed(seed)
    delta = torch.zeros(N_PRIMARIES, device=device, requires_grad=True)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    optimiser = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": c.WEIGHT_DECAY},
            {"params": [delta], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    prior = (delta.detach() * 0).clone()

    def nll_of(idx: torch.Tensor, ek, ed) -> torch.Tensor:
        cap = capacity_mw(
            t=tensors, delta=delta, p=p[idx], r4=r4[idx], after=after[idx], ek=ek, ed=ed
        )
        z = (y[idx] + cap * u_inc[idx] - mean[p[idx]]) / std[p[idx]]
        return base.mixture_nll(out=model(x[idx], p[idx]), z=z, components=m.COMPONENTS)

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        ek = torch.randn(N_PRIMARIES, device=device) * tensors["sd_k"]
        ed = torch.randn(N_PRIMARIES, device=device) * tensors["sd_d"]
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            penalty = (
                ((delta - prior) ** 2).sum() / (2 * k.DELTA_PRIOR_MW**2) / k.DELTA_PENALTY_ROWS
            )
            loss = nll_of(b, ek, ed).mean() + penalty
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat([nll_of(i, None, None) for i in val_idx.split(c.BATCH * 2)]).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = (
                {a: v.detach().clone() for a, v in model.state_dict().items()},
                delta.detach().clone(),
            )
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state[0])
    model.eval()
    return epochs, best, best_state[1]


def forecast(
    *, model, x, p, mean, std, subtract: torch.Tensor | None, device: str
) -> np.ndarray:  # fmt: skip
    """Return the net quantiles (rows, 11) in points: the mixture's quantiles in points, minus `subtract` (points)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 65_536):
            sl = slice(i, i + 65_536)
            xb = torch.from_numpy(x[sl]).to(device)
            pb = torch.from_numpy(p[sl]).to(device)
            log_w, mu, sd = base.mixture_parts(out=model(xb, pb).double(), components=m.COMPONENTS)
            q = base.mixture_quantiles(log_w=log_w, mean=mu, std=sd)
            pts = q * std[pb][:, None] + mean[pb][:, None]
            if subtract is not None:
                pts = pts - subtract[sl].to(device)[:, None]
            out.append(pts.cpu().numpy())
    return np.concatenate(out)
