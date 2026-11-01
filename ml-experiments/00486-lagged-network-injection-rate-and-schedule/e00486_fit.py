"""Injected worlds at other rates, and the lagged network fitted on a schedule of worlds.

Experiment 404's arm A0-inj trains the lagged network on one injected world per seed, in which
about 20% of training rows have a lag inside an event. This script builds worlds with other
event coverages (experiment 428's injection, only `coverage` changes), measures the share of
training rows with a lag inside an event, and fits A0-inj's network and loss with a different
world in each epoch. A constant rate uses the same world in all 6 epochs; a curriculum switches
worlds every 2 epochs. Everything else is experiment 404's `fit_arm`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights:\
ml-experiments/00428-day-window-convolution-attention-encoder-replacing-lagged-power-inputs:\
ml-experiments/00486-lagged-network-injection-rate-and-schedule \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00486-*/e00486_fit.py fit --arm R05 --month 2026-06 \
    --seed 0 --device cuda
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00402_inject as inj  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score  # ty: ignore[unresolved-import]
import e00428_data as data428  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00486-lagged-network-injection-rate-and-schedule" / "work"
)
WORLD_DIR: Final[Path] = WORK_DIR / "worlds"
MODEL_DIR: Final[Path] = WORK_DIR / "models"
COVERAGE: Final[dict[str, float]] = {"r05": 0.016, "r20": 0.10, "r50": 0.18}
"""Event coverage of the world behind each rate label; r20 is experiment 428's stored world.

The labels name the target share of training rows with at least one of the 3 lags inside an
injected event: 5%, 20% (measured 27.8% at 2026-06), and 50%. The coverages were set by measuring
that share on candidate worlds.
"""
EPOCHS_PER_STAGE: Final[int] = 2
SCHEDULES: Final[dict[str, tuple[str, ...]]] = {
    "A0-repro": ("r20",) * 6,
    "R05": ("r05",) * 6,
    "R50": ("r50",) * 6,
    "Cur-down": ("r50", "r50", "r20", "r20", "r05", "r05"),
    "Cur-up": ("r05", "r05", "r20", "r20", "r50", "r50"),
}
"""The world used in each of the 6 epochs, per arm."""


def world_file(*, rate: str, seed: int) -> Path:
    """Return the file of the training world for a rate and a training seed (100 + fit seed)."""
    if rate == "r20":
        return next(ML_EXPERIMENTS_DATA_DIR.glob("00428-*")) / "work" / "worlds" / f"seed{seed}.npz"
    return WORLD_DIR / f"{rate}_cov{COVERAGE[rate]}_seed{seed}.npz"


def build_world(*, rate: str, seed: int, ctx: worlds402.Context, clean: dict) -> str:
    """Build the training world for a rate and seed with experiment 428's injection."""
    target = world_file(rate=rate, seed=seed)
    if target.exists():
        return f"{rate} seed {seed}: exists"
    labels = data428.sites()
    injected = inj.inject(
        power={s: clean[s].astype(np.float64) for s in labels},
        capacity_mw=ctx.capacity,
        neighbours=ctx.neighbours,
        magnitude_only=ctx.magnitude_only,
        spike_mad_mw=ctx.stage1_mad_mw,
        pool=worlds402.make_pool(ctx=ctx),
        first=int(common.grid_index(times=data428.WORLD_FIRST)),
        last=int(common.grid_index(times=data428.WORLD_LAST)),
        coverage=COVERAGE[rate],
        seed=seed,
        allowed={s: ~ctx.event_mask[s] & np.isfinite(clean[s]) for s in labels},
    )
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        power=np.stack([injected.power[s] for s in labels]).astype(np.float32),
        code=np.stack([injected.mask[s] for s in labels]).astype(np.int8),
    )
    temporary.rename(target)
    return f"{rate} seed {seed}: {injected.events.height:,} events"


def load_world(*, rate: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a training world's power and event codes."""
    z = np.load(world_file(rate=rate, seed=seed))
    return z["power"].astype(np.float32), z["code"]


def lag_event_share(*, pool, code: np.ndarray, device: str) -> float:  # noqa: ANN001
    """Return the share of training rows with at least one of the 3 lags inside an event."""
    lag_grid = rows_mod.lag_grids(
        target=pool.target.cpu().numpy(), origin=pool.origin.cpu().numpy()
    )
    primary = pool.primary.cpu().numpy()
    safe = np.clip(lag_grid, 0, code.shape[1] - 1)
    return float(((code[primary[:, None], safe] != 0).any(axis=1)).mean())


def fit_schedule(*, arm: str, month: str, seed: int, device: str, stride: int) -> Path:
    """Fit the A0-inj network with the arm's world in each epoch; save it; return the path."""
    started = time.monotonic()
    train, _test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    real = e00404_score.real_event_mask()
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    worlds = {}
    for rate in sorted(set(SCHEDULES[arm])):
        power_np, code = load_world(rate=rate, seed=100 + seed)
        power = torch.tensor(power_np, device=device)
        lags, lag_grid = fit.lag_features(
            pool=train,
            power=power,
            lag_stats=lag_stats,
            device=device,  # ty: ignore[invalid-argument-type]
        )
        mask = fit.event_mask_of(code=code, real=real, device=device)
        stack = rows_mod.build_stack(
            power=power_np,
            table=table,
            neighbours=rows_mod.neighbour_matrix(
                sites=prep.primaries()["site"].to_list(), path=fit._neighbour_path()
            ),
            device=device,
        )
        worlds[rate] = (power, lags, lag_grid, mask, stack)
        print(
            f"{arm} {month} s{seed} world {rate}: lag-event share "
            f"{lag_event_share(pool=train, code=code, device=device):.3f}",
            flush=True,
        )
    metered = torch.tensor(fit.metered_indicator(), device=device)
    torch.manual_seed(seed)
    net = fit.build_network(arm="A0-inj", n_base=train.base.shape[1], n_trust=rows_mod.N_TRUST).to(
        device
    )  # ty: ignore[unresolved-attribute]
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)  # ty: ignore[invalid-argument-type]
    losses = []
    for epoch, rate in enumerate(SCHEDULES[arm]):
        power, lags, lag_grid, mask, stack = worlds[rate]
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            idx = order[start : start + fit.BATCH]
            out = fit.forward(
                net=net, pool=train, idx=idx, lags=lags, stack=stack, lag_grid=lag_grid,  # ty: ignore[invalid-argument-type]
                scalers=scalers, metered=metered, masked=(), event_mask=mask, power=power,
            )  # fmt: skip
            log_w, mean, std = out[0] if isinstance(out[0], tuple) else out
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]  # ty: ignore[unresolved-attribute]
            nll = fit.models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = train.weight[idx]  # ty: ignore[unresolved-attribute]
            loss = (nll * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
        print(f"{arm} {month} s{seed} epoch {epoch} ({rate}): loss {losses[-1]:.4f}", flush=True)
    path = MODEL_DIR / arm / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(),
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
            "losses": losses,
            "rows": n,
            "seconds": time.monotonic() - started,
            "g0": _arrays.g0,
        },
        path,
    )
    print(f"{arm} {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


def main() -> None:
    """Build worlds or fit an arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worlds")
    w.add_argument("--rates", nargs="+", default=["r05", "r50"])
    w.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    sh = sub.add_parser("share")
    sh.add_argument("--rates", nargs="+", required=True)
    sh.add_argument("--month", default="2026-06")
    sh.add_argument("--seed", type=int, default=0)
    f = sub.add_parser("fit")
    f.add_argument("--arm", choices=tuple(SCHEDULES), required=True)
    f.add_argument("--month", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cuda")
    f.add_argument("--origin-stride", type=int, default=2)
    args = parser.parse_args()
    if args.command == "worlds":
        import e00422_prep as prep422  # ty: ignore[unresolved-import]

        ctx = worlds402.load_context()
        clean = prep422.cleaned_arrays()
        for rate in args.rates:
            for seed in args.seeds:
                print(build_world(rate=rate, seed=seed, ctx=ctx, clean=clean), flush=True)
    elif args.command == "share":
        train, *_ = fit.build_pools(month=args.month, seed=args.seed, device="cpu", stride=2)
        for rate in args.rates:
            _, code = load_world(rate=rate, seed=100 + args.seed)
            print(rate, f"{lag_event_share(pool=train, code=code, device='cpu'):.3f}", flush=True)
    else:
        fit_schedule(
            arm=args.arm,
            month=args.month,
            seed=args.seed,
            device=args.device,
            stride=args.origin_stride,
        )


if __name__ == "__main__":
    main()
