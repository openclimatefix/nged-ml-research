"""Hard-negative mining for the lagged network trained on injected events (arm HN).

Experiment 404's arm A0-inj trains a mixture density network on one injected world per seed for 6
epochs. Arm HN trains the same network, the same world, and the same loss for the first 3 epochs
(the epoch-3 checkpoint is saved), then scores every training row once, marks the hard rows, and
trains the last 3 epochs with each hard row appearing 3 times per epoch. A row is hard when:

- its median follows the injected step by more than `FOLLOW_LIMIT` of the step (rows whose lags
  differ from the clean readings by at least `STEP_FLOOR` of capacity on average; the follow share
  compares the median on the injected lags with the median on the clean lags of the same row), or
- its pinball loss against the clean target (mean over the 11 quantiles, on the injected lags)
  exceeds the 80th percentile of its lag-status class (a lag inside an injected event, or no lag
  inside one).

The hard set is computed once, never per batch. Everything else is experiment 404's `fit_arm`, as
copied by experiment 486 (whose A0-inj re-fit matched the stored one to 1.4e-4 of capacity).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights:\
ml-experiments/00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs:\
ml-experiments/00495-lagged-network-hard-negative-mining-on-injected-events \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00495-*/e00495_fit.py fit --month 2026-06 \
    --seed 0 --device cuda
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00495-lagged-network-hard-negative-mining-on-injected-events"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
HARD_DIR: Final[Path] = WORK_DIR / "hard"
SCHEDULES: Final[dict[str, str]] = {"HN": "hard-negative mining after epoch 3"}
"""Arm names this experiment fits; the key set is what `e00495_score` treats as network arms."""
PRE_EPOCHS: Final[int] = 3
"""Epochs of plain training before the mining pass (60% of 6, rounded to the checkpoint)."""
POST_EPOCHS: Final[int] = 3
OVERSAMPLE: Final[int] = 3
"""Times a hard row appears in each post-mining epoch."""
FOLLOW_LIMIT: Final[float] = 0.25
"""A row is hard when its median follows more than this share of the injected step."""
STEP_FLOOR: Final[float] = 0.005
"""Rows whose mean lag step is under this fraction of capacity have no defined follow share."""
LOSS_QUANTILE: Final[float] = 0.8
LEVELS: Final[tuple[float, ...]] = models.LEVELS


def world_file(*, seed: int) -> Path:
    """Return experiment 428's stored training world for a training seed (100 + fit seed)."""
    return next(ML_EXPERIMENTS_DATA_DIR.glob("00428-*")) / "work" / "worlds" / f"seed{seed}.npz"


def load_world(*, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the injected world's power and event codes."""
    z = np.load(world_file(seed=seed))
    return z["power"].astype(np.float32), z["code"]


def _quantiles(
    *,
    net,
    pool,
    idx,
    lags,
    stack,
    lag_grid,
    scalers,
    metered,
    mask,
    power,
    mu,
    sd,  # noqa: ANN001
) -> torch.Tensor:
    """Return the 11 quantiles in capacity fractions for the rows `idx`."""
    out = fit.forward(
        net=net, pool=pool, idx=idx, lags=lags, stack=stack, lag_grid=lag_grid,
        scalers=scalers, metered=metered, masked=(), event_mask=mask, power=power,
    )  # fmt: skip
    log_w, mean, std = out[0] if isinstance(out[0], tuple) else out
    return score.mixture_quantiles_capacity(
        log_w=log_w, mean=mean, std=std, mu=mu[pool.primary[idx]], sd=sd[pool.primary[idx]]
    )


@torch.no_grad()
def mine(  # noqa: PLR0913
    *,
    net,
    train,
    injected,
    clean,
    code,
    scalers,
    metered,
    mask,
    device: str,  # noqa: ANN001
) -> pl.DataFrame:
    """Score every training row once and return one row per training row with the hard flags.

    Columns: `corrupted` (a lag inside an injected event), `step` (mean over the 3 lags of the
    injected minus the clean reading, fraction of capacity), `follow` (median on the injected lags
    minus median on the clean lags, over the step; null where the step is under the floor),
    `loss` (pinball loss on the injected lags), `hard_follow`, `hard_loss`, `hard`, plus the row's
    `primary` and `lead_hours`.
    """
    net.eval()
    n = len(train)
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=device)
    power_i, lags_i, grid, stack_i = injected
    power_c, lags_c, _grid_c, stack_c = clean
    size = power_i.shape[-1]
    safe = grid.clamp(0, size - 1)
    code_t = torch.tensor(code, device=device)
    corrupted = (code_t[train.primary[:, None], safe] != 0).any(dim=1)
    diff = power_i[train.primary[:, None], safe] - power_c[train.primary[:, None], safe]
    step = torch.nan_to_num(diff, nan=0.0).mean(dim=1)
    common_kwargs = dict(  # noqa: C408
        net=net,
        pool=train,
        lag_grid=grid,
        scalers=scalers,
        metered=metered,
        mask=mask,
        mu=mu,
        sd=sd,
    )
    loss = torch.empty(n, device=device)
    follow = torch.full((n,), float("nan"), device=device)
    for start in range(0, n, fit.BATCH):
        idx = torch.arange(start, min(start + fit.BATCH, n), device=device)
        q_i = _quantiles(idx=idx, lags=lags_i, stack=stack_i, power=power_i, **common_kwargs)
        err = train.y[idx][:, None] - q_i
        loss[idx] = torch.maximum(levels * err, (levels - 1.0) * err).mean(dim=1)
        usable = corrupted[idx] & (step[idx].abs() >= STEP_FLOOR)
        if bool(usable.any()):
            sub = idx[usable]
            q_c = _quantiles(idx=sub, lags=lags_c, stack=stack_c, power=power_c, **common_kwargs)
            follow[sub] = (q_i[usable][:, 5] - q_c[:, 5]) / step[sub]
    loss_np, corrupted_np = loss.cpu().numpy(), corrupted.cpu().numpy()
    limits = {
        flag: float(np.quantile(loss_np[corrupted_np == flag], LOSS_QUANTILE))
        for flag in (True, False)
    }
    limit_row = np.where(corrupted_np, limits[True], limits[False])
    follow_np = follow.cpu().numpy()
    hard_follow = np.nan_to_num(follow_np, nan=-np.inf) > FOLLOW_LIMIT
    hard_loss = loss_np > limit_row
    return pl.DataFrame(
        {
            "primary": train.primary.cpu().numpy(),
            "lead_hours": train.lead_hours.cpu().numpy(),
            "corrupted": corrupted_np,
            "step": step.cpu().numpy(),
            "follow": follow_np,
            "loss": loss_np,
            "hard_follow": hard_follow,
            "hard_loss": hard_loss,
            "hard": hard_follow | hard_loss,
        }
    )


def fit_hn(*, month: str, seed: int, device: str, stride: int) -> Path:
    """Fit arm HN for a month and seed; save the epoch-3 checkpoint and the final network."""
    started = time.monotonic()
    train, _test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    real = score.real_event_mask()
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    neighbours = rows_mod.neighbour_matrix(
        sites=prep.primaries()["site"].to_list(), path=fit._neighbour_path()
    )
    power_np, code = load_world(seed=100 + seed)
    clean_np, _ = fit.world_power(seed=None)
    pack = {}
    for name, values in (("injected", power_np), ("clean", clean_np.astype(np.float32))):
        power = torch.tensor(values, device=device)
        lags, lag_grid = fit.lag_features(
            pool=train, power=power, lag_stats=lag_stats, device=device,  # ty: ignore[invalid-argument-type]
        )  # fmt: skip
        stack = rows_mod.build_stack(
            power=values, table=table, neighbours=neighbours, device=device
        )
        pack[name] = (power, lags, lag_grid, stack)
    power, lags, lag_grid, stack = pack["injected"]
    mask = fit.event_mask_of(code=code, real=real, device=device)
    metered = torch.tensor(fit.metered_indicator(), device=device)
    torch.manual_seed(seed)
    net = fit.build_network(arm="A0-inj", n_base=train.base.shape[1], n_trust=rows_mod.N_TRUST).to(
        device
    )  # ty: ignore[unresolved-attribute]
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)  # ty: ignore[invalid-argument-type]
    losses: list[float] = []
    hard_idx: torch.Tensor | None = None
    saved = {
        "arm": "A0-inj",
        "month": month,
        "seed": seed,
        "n_base": train.base.shape[1],  # ty: ignore[unresolved-attribute]
        "target_mean": scalers.target_mean,
        "target_std": scalers.target_std,
        "standardiser_mean": scalers.standardiser.mean,
        "standardiser_std": scalers.standardiser.std,
        "standardiser_mask": scalers.standardiser.mask,
        "lag_stats": lag_stats,
        "trust_mean": scalers.trust_mean,
        "trust_std": scalers.trust_std,
        "rows": n,
        "g0": _arrays.g0,
    }
    path = MODEL_DIR / "HN" / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(PRE_EPOCHS + POST_EPOCHS):
        if epoch == PRE_EPOCHS:
            torch.save(
                {"state": net.state_dict(), **saved, "losses": losses},
                path.with_name(f"{month}_s{seed}_epoch{PRE_EPOCHS}.pt"),
            )
            frame = mine(
                net=net, train=train, injected=pack["injected"], clean=pack["clean"], code=code,
                scalers=scalers, metered=metered, mask=mask, device=device,
            )  # fmt: skip
            HARD_DIR.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(HARD_DIR / f"{month}_s{seed}.parquet", compression="zstd")
            hard_idx = torch.tensor(np.flatnonzero(frame["hard"].to_numpy()), device=device)
            print(
                f"HN {month} s{seed} mining: {len(hard_idx):,} hard rows of {n:,} "
                f"({len(hard_idx) / n:.3f}); follow rule {frame['hard_follow'].sum():,}, "
                f"loss rule {frame['hard_loss'].sum():,}",
                flush=True,
            )
        net.train()
        order = torch.randperm(n, device=device)
        if hard_idx is not None:
            extra = hard_idx.repeat(OVERSAMPLE - 1)
            order = torch.cat([order, extra])[torch.randperm(n + len(extra), device=device)]
        total = 0.0
        for start in range(0, len(order), fit.BATCH):
            idx = order[start : start + fit.BATCH]
            out = fit.forward(
                net=net, pool=train, idx=idx, lags=lags, stack=stack, lag_grid=lag_grid,  # ty: ignore[invalid-argument-type]
                scalers=scalers, metered=metered, masked=(), event_mask=mask, power=power,
            )  # fmt: skip
            log_w, mean, std = out[0] if isinstance(out[0], tuple) else out
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]  # ty: ignore[unresolved-attribute]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = train.weight[idx]  # ty: ignore[unresolved-attribute]
            loss = (nll * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / len(order))
        print(f"HN {month} s{seed} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    torch.save(
        {"state": net.state_dict(), **saved, "losses": losses, "seconds": time.monotonic() - started},
        path,
    )  # fmt: skip
    print(f"HN {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


def main() -> None:
    """Fit arm HN for one scored month and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--month", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cuda")
    f.add_argument("--origin-stride", type=int, default=2)
    args = parser.parse_args()
    fit_hn(month=args.month, seed=args.seed, device=args.device, stride=args.origin_stride)


if __name__ == "__main__":
    main()
