"""Cross-fitted trust fit (G1) on experiment 404's harness, one scored month at a time.

For a scored month, the training rows are split into the lagless table's 6 contiguous blocks of
target time. A plain lag network (experiment 404's A0-inj recipe) is fitted on the rows outside each
block and forecasts the rows inside it, so every training row has an out-of-fold lagged head and a
cross-fitted lagless head from the same fold. A trust head (experiment 404's B1 trust head: three
per-lag weights, multiplied, smoothed over three half-hours) is then fitted on the training rows
with both heads frozen. The scored month is forecast with experiment 404's stored full-data A0-inj
network, the table's further lagless module, and the new trust head. Nothing from the scored month
or later enters any fit.

```bash
PYTHONPATH=ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=<the internal data folder> OMP_NUM_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00463-*/e00463_crossfit.py fit --month 2026-06
```

`check` refits the lagged head on every training row and compares its final training loss with
experiment 404's stored A0-inj value.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
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

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00463-lag-trust-cross-fitted-trust-fit-seed-zero"
PRED_DIR: Final[Path] = WORK_DIR / "predictions"
OOF_DIR: Final[Path] = WORK_DIR / "oof"
LOG_DIR: Final[Path] = WORK_DIR / "logs"
FOLDS: Final[int] = lagless.FOLDS
ORIGIN_STRIDE: Final[int] = 2
"""Experiment 404 trained every arm on every second origin day."""
EPOCHS: Final[int] = fit.EPOCHS
BATCH: Final[int] = fit.BATCH
LEARNING_RATE: Final[float] = fit.LEARNING_RATE


@dataclass
class Context:
    """Everything a month's fits read: pools, scalers, the injected world, and lag tensors."""

    train: fit.Pool
    scalers: fit.Scalers
    lag_stats: np.ndarray
    power_np: np.ndarray
    stack: rows_mod.Stack
    lags: torch.Tensor
    lag_grid: torch.Tensor
    metered: torch.Tensor
    table: pl.DataFrame
    block: np.ndarray


def _stack(*, power_np: np.ndarray, table: pl.DataFrame, device: str) -> rows_mod.Stack:
    """Return experiment 404's trust-input stack for a world's power."""
    return rows_mod.build_stack(
        power=power_np,
        table=table,
        neighbours=rows_mod.neighbour_matrix(
            sites=prep.primaries()["site"].to_list(), path=fit._neighbour_path()
        ),
        device=device,
    )


def build_context(*, month: str, seed: int, device: str) -> Context:
    """Build the training pool and the lag tensors of experiment 404's A0-inj world."""
    train, _test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=ORIGIN_STRIDE
    )
    assert train is not None
    power_np, _code = fit.world_power(seed=100 + seed)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    lags, lag_grid = fit.lag_features(
        pool=train, power=torch.tensor(power_np, device=device), lag_stats=lag_stats, device=device
    )
    end = int(common.grid_index(times=prep.refit_of(month=month)))
    start = max(lagless.data.FIRST_VALID_INDEX, end - lagless.WINDOW_STEPS)
    grid = train.target.cpu().numpy()
    if not ((grid >= start) & (grid < end)).all():
        msg = f"{month}: training targets fall outside the lagless table's blocks"
        raise ValueError(msg)
    block = lagless.block_of(grid=grid, lo=start, hi=end, folds=FOLDS)
    return Context(
        train=train,
        scalers=scalers,
        lag_stats=lag_stats,
        power_np=power_np,
        stack=_stack(power_np=power_np, table=table, device=device),
        lags=lags,
        lag_grid=lag_grid,
        metered=torch.tensor(fit.metered_indicator(), device=device),
        table=table,
        block=block,
    )


def target_scalers(*, train: fit.Pool, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each Primary's target mean and standard deviation over the rows in `mask`."""
    primary = train.primary.cpu().numpy()
    y = train.y.cpu().numpy()
    mean, std = np.zeros(16), np.ones(16)
    for p in range(16):
        sel = mask & (primary == p)
        if sel.any():
            mean[p], std[p] = y[sel].mean(), max(float(y[sel].std()), 1e-3)
    return mean, std


def fit_plain(
    *, ctx: Context, mask: np.ndarray, mean: np.ndarray, std: np.ndarray, seed: int, device: str
) -> tuple[models.PlainLagMdn, list[float]]:
    """Fit experiment 404's A0-inj network on the rows in `mask`; return it and its epoch losses."""
    train = ctx.train
    torch.manual_seed(seed)
    net = models.PlainLagMdn(n_base=train.base.shape[1], n_primaries=16, lag_dropout=0.0).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    tm = torch.tensor(mean, dtype=torch.float32, device=device)
    ts = torch.tensor(std, dtype=torch.float32, device=device)
    rows = torch.tensor(np.flatnonzero(mask), device=device)
    n = len(rows)
    losses = []
    for _ in range(EPOCHS):
        net.train()
        order = rows[torch.randperm(n, device=device)]
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            log_w, m, s = net(train.base[idx], ctx.lags[idx], train.primary[idx])
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]
            nll = models.mixture_nll_parts(log_w=log_w, mean=m, std=s, z=z)
            w = train.weight[idx]
            loss = (nll * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
    return net, losses


@torch.no_grad()
def predict_plain(
    *,
    net: models.PlainLagMdn,
    ctx: Context,
    rows: np.ndarray,
    from_units: tuple[np.ndarray, np.ndarray],
    to_units: tuple[np.ndarray, np.ndarray],
    device: str,
) -> torch.Tensor:
    """Return the mixture (len(rows), 9) = log weights, means, standard deviations in `to_units`.

    The network's outputs are in `from_units` (per-Primary target mean and standard deviation) and
    are converted through capacity fractions to `to_units`.
    """
    net.eval()
    train = ctx.train
    f_mean, f_std, t_mean, t_std = (
        torch.tensor(a, dtype=torch.float32, device=device) for a in (*from_units, *to_units)
    )
    index = torch.tensor(rows, device=device)
    out = []
    for start in range(0, len(index), 65536):
        idx = index[start : start + 65536]
        p = train.primary[idx]
        log_w, m, s = net(train.base[idx], ctx.lags[idx], p)
        m = ((m * f_std[p][:, None] + f_mean[p][:, None]) - t_mean[p][:, None]) / t_std[p][:, None]
        s = s * (f_std[p] / t_std[p])[:, None]
        out.append(torch.cat([log_w, m, s], dim=1))
    return torch.cat(out)


def out_of_fold_heads(*, ctx: Context, seed: int, device: str) -> tuple[torch.Tensor, list[dict]]:
    """Fit the 6 fold networks; return the out-of-fold mixtures in the final network's units."""
    final_units = (ctx.scalers.target_mean, ctx.scalers.target_std)
    n = len(ctx.train)
    oof = torch.zeros(n, 9, device=device)
    log = []
    for k in range(FOLDS):
        started = time.monotonic()
        fit_mask, held = ctx.block != k, np.flatnonzero(ctx.block == k)
        mean, std = target_scalers(train=ctx.train, mask=fit_mask)
        net, losses = fit_plain(
            ctx=ctx, mask=fit_mask, mean=mean, std=std, seed=seed, device=device
        )
        oof[torch.tensor(held, device=device)] = predict_plain(
            net=net, ctx=ctx, rows=held, from_units=(mean, std), to_units=final_units, device=device
        )
        log.append(
            {
                "fold": k, "fit_rows": int(fit_mask.sum()), "held_rows": len(held),
                "final_loss": losses[-1], "seconds": time.monotonic() - started,
            }
        )  # fmt: skip
        print(f"fold {k}: {log[-1]}", flush=True)
    return oof, log


def mixture_with_trust(
    *, lagged: torch.Tensor, ll: torch.Tensor, w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the 6-component mixture of the lagless and lagged heads (B1's combining rule)."""
    ll_lw, ll_mean, ll_std = models.mixture_parts(out=ll, components=models.COMPONENTS)
    lag_lw, lag_mean, lag_std = lagged[:, :3], lagged[:, 3:6], lagged[:, 6:9]
    w = w[:, None]
    log_w = torch.cat(
        [ll_lw + torch.log1p(-w * (1.0 - 1e-6)), lag_lw + torch.log(w + 1e-6)], dim=-1
    )
    return log_w, torch.cat([ll_mean, lag_mean], -1), torch.cat([ll_std, lag_std], -1)


def trust_batch(
    *,
    net: models.TrustMdn,
    pool: fit.Pool,
    idx: torch.Tensor,
    stack: rows_mod.Stack,
    lag_grid: torch.Tensor,
    scalers: fit.Scalers,
    metered: torch.Tensor,
) -> torch.Tensor:
    """Return the combined lagged weight (rows,) of the trust head for the rows `idx`."""
    primary = pool.primary[idx]
    trust = torch.stack(
        [
            rows_mod.trust_inputs(
                stack=stack, primary=primary, lag_grid=lag_grid[idx], origin=pool.origin[idx],
                step=s,
            )
            for s in range(rows_mod.SMOOTHING_STEPS)
        ],
        dim=1,
    )  # fmt: skip
    trust = ((trust - scalers.trust_mean) / scalers.trust_std).clamp(-fit.CLIP_SD, fit.CLIP_SD)  # ty: ignore[unsupported-operator]
    band = torch.from_numpy(models.lead_band_of(lead_hours=pool.lead_hours[idx].cpu().numpy())).to(
        primary.device
    )
    t = net.trust_weights(trust, pool.lead[idx], primary, metered[primary][:, None], band)
    return t.prod(dim=-1)


def set_trust_statistics(*, ctx: Context, seed: int, device: str) -> None:
    """Set the trust-input mean and standard deviation from 100,000 sampled training rows."""
    train = ctx.train
    torch.manual_seed(seed)
    sample = torch.randperm(len(train), device=device)[: fit.TRUST_SAMPLE]
    stats = rows_mod.trust_inputs(
        stack=ctx.stack, primary=train.primary[sample], lag_grid=ctx.lag_grid[sample],
        origin=train.origin[sample], step=0,
    )  # fmt: skip
    ctx.scalers.trust_mean = stats.mean(0)
    ctx.scalers.trust_std = stats.std(0).clamp(min=1e-3)


def fit_trust(
    *, ctx: Context, oof: torch.Tensor, seed: int, device: str
) -> tuple[models.TrustMdn, list[float]]:
    """Fit the trust head on the training rows with both heads frozen."""
    train, scalers = ctx.train, ctx.scalers
    torch.manual_seed(seed)
    net = models.TrustMdn(
        arm="b1", n_base=train.base.shape[1], n_trust=rows_mod.N_TRUST, n_primaries=16
    ).to(device)
    params = [*net.trust_head.parameters(), *net.embedding.parameters()]
    optimiser = torch.optim.Adam(params, lr=LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)
    losses = []
    for epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            w = trust_batch(
                net=net, pool=train, idx=idx, stack=ctx.stack, lag_grid=ctx.lag_grid,
                scalers=scalers, metered=ctx.metered,
            )  # fmt: skip
            log_w, mean, std = mixture_with_trust(lagged=oof[idx], ll=train.ll[idx].detach(), w=w)
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            weight = train.weight[idx]
            loss = (nll * weight).sum() / weight.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
        print(f"trust epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    return net, losses


@torch.no_grad()
def mean_nll(*, ctx: Context, mixture: torch.Tensor, device: str) -> float:
    """Return the recency-weighted mean negative log-likelihood of a (rows, 9) mixture."""
    train = ctx.train
    tm = torch.tensor(ctx.scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(ctx.scalers.target_std, dtype=torch.float32, device=device)
    z = (train.y - tm[train.primary]) / ts[train.primary]
    nll = models.mixture_nll_parts(
        log_w=mixture[:, :3], mean=mixture[:, 3:6], std=mixture[:, 6:9], z=z
    )
    return float((nll * train.weight).sum() / train.weight.sum())


@torch.no_grad()
def forecast_month(
    *, month: str, seed: int, trust_net: models.TrustMdn, ctx: Context, device: str
) -> pl.DataFrame:
    """Forecast the scored month's real-data rows with the stored A0-inj network and a trust."""
    saved = torch.load(fit.MODEL_DIR / "A0-inj" / f"{month}_s{seed}.pt", weights_only=False)
    for key, value in (
        ("target_mean", ctx.scalers.target_mean),
        ("target_std", ctx.scalers.target_std),
    ):
        if not np.allclose(saved[key], value):
            msg = f"stored A0-inj {key} differs from this month's training scalers"
            raise ValueError(msg)
    lagged_net = fit.build_network(arm="A0-inj", n_base=saved["n_base"], n_trust=rows_mod.N_TRUST)
    lagged_net.load_state_dict(saved["state"])
    lagged_net = lagged_net.to(device).eval()
    trust_net.eval()
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    scalers.trust_mean, scalers.trust_std = ctx.scalers.trust_mean, ctx.scalers.trust_std
    power_np, code = fit.world_power(seed=None)
    power = torch.tensor(power_np, device=device)
    stack = _stack(power_np=power_np, table=ctx.table, device=device)
    lags, lag_grid = fit.lag_features(pool=test, power=power, lag_stats=lag_stats, device=device)
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    quantiles, weights = [], []
    for start in range(0, len(test), 16384):
        idx = torch.arange(start, min(start + 16384, len(test)), device=device)
        lw, m, s = lagged_net(test.base[idx], lags[idx], test.primary[idx])
        w = trust_batch(
            net=trust_net, pool=test, idx=idx, stack=stack, lag_grid=lag_grid, scalers=scalers,
            metered=ctx.metered,
        )  # fmt: skip
        log_w, mean, std = mixture_with_trust(
            lagged=torch.cat([lw, m, s], dim=1), ll=test.ll[idx], w=w
        )
        primary = test.primary[idx]
        quantiles.append(
            score.mixture_quantiles_capacity(
                log_w=log_w, mean=mean, std=std, mu=mu[primary], sd=sd[primary]
            ).cpu()
        )
        weights.append(w.cpu())
    q = torch.cat(quantiles).numpy()
    flags = score.Flags(code=code, real=score.real_event_mask(), power=power_np, device=device).of(
        primary=test.primary, origin=test.origin, target=test.target
    )
    return pl.DataFrame(
        {
            "primary": test.primary.cpu().numpy().astype(np.int16),
            "origin": test.origin.cpu().numpy().astype(np.int32),
            "target": test.target.cpu().numpy().astype(np.int32),
            "lead_hours": test.lead_hours.cpu().numpy(),
            "actual": test.y.cpu().numpy(),
            **{name: q[:, i] for i, name in enumerate(score.LEVEL_NAMES)},
            **{k: v.cpu().numpy() for k, v in flags.items()},
            "trust_w": torch.cat(weights).numpy(),
        }
    )


def check_keys(*, month: str, seed: int, frame: pl.DataFrame) -> None:
    """Raise if the forecast rows differ from experiment 404's stored B-mix rows."""
    ref = pl.read_parquet(
        score.prediction_path(arm="B-mix", month=month, seed=seed, world=-1),
        columns=["primary", "origin", "target"],
    )
    if not frame.select("primary", "origin", "target").equals(ref):
        msg = f"G1-trust rows differ from B-mix in {month} seed {seed}"
        raise ValueError(msg)


def check_test_rows(*, month: str, seed: int, ctx: Context, device: str) -> None:
    """Raise before any fit if the scored month's rows differ from experiment 404's B-mix rows."""
    saved = torch.load(fit.MODEL_DIR / "A0-inj" / f"{month}_s{seed}.pt", weights_only=False)
    _train, test, _s, _a, _l = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    keys = pl.DataFrame(
        {
            "primary": test.primary.cpu().numpy().astype(np.int16),
            "origin": test.origin.cpu().numpy().astype(np.int32),
            "target": test.target.cpu().numpy().astype(np.int32),
        }
    )
    check_keys(month=month, seed=seed, frame=keys)


def run_month(*, month: str, seed: int, device: str) -> None:
    """Fit the 6 fold networks and the trust head for a month, and write the forecast."""
    started = time.monotonic()
    ctx = build_context(month=month, seed=seed, device=device)
    print(
        f"{month}: {len(ctx.train):,} training rows, context in {time.monotonic() - started:.0f} s"
    )
    check_test_rows(month=month, seed=seed, ctx=ctx, device=device)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    cache = OOF_DIR / f"oof_{month}_s{seed}.npy"
    if cache.exists():
        oof = torch.tensor(np.load(cache), device=device)
        fold_log = json.loads((OOF_DIR / f"oof_{month}_s{seed}.json").read_text())
    else:
        oof, fold_log = out_of_fold_heads(ctx=ctx, seed=seed, device=device)
        np.save(cache, oof.cpu().numpy().astype(np.float32))
        (OOF_DIR / f"oof_{month}_s{seed}.json").write_text(json.dumps(fold_log))
    set_trust_statistics(ctx=ctx, seed=seed, device=device)
    trust_started = time.monotonic()
    trust_net, trust_losses = fit_trust(ctx=ctx, oof=oof, seed=seed, device=device)
    trust_seconds = time.monotonic() - trust_started
    saved = torch.load(fit.MODEL_DIR / "A0-inj" / f"{month}_s{seed}.pt", weights_only=False)
    full_net = fit.build_network(arm="A0-inj", n_base=saved["n_base"], n_trust=rows_mod.N_TRUST)
    full_net.load_state_dict(saved["state"])
    full_net = full_net.to(device)
    in_sample = predict_plain(  # ty: ignore[invalid-argument-type]
        net=full_net, ctx=ctx, rows=np.arange(len(ctx.train)),
        from_units=(ctx.scalers.target_mean, ctx.scalers.target_std),
        to_units=(ctx.scalers.target_mean, ctx.scalers.target_std), device=device,
    )  # fmt: skip
    insample_started = time.monotonic()
    insample_net, insample_losses = fit_trust(ctx=ctx, oof=in_sample, seed=seed, device=device)
    insample_seconds = time.monotonic() - insample_started
    lw, lm, ls = models.mixture_parts(out=ctx.train.ll, components=models.COMPONENTS)
    lagless_mixture = torch.cat([lw, lm, ls], dim=1)
    diagnostics = {
        "month": month, "seed": seed, "rows": len(ctx.train), "folds": fold_log,
        "trust_losses": trust_losses, "trust_seconds": trust_seconds,
        "insample_losses": insample_losses, "insample_seconds": insample_seconds,
        "nll_lagged_out_of_fold": mean_nll(ctx=ctx, mixture=oof, device=device),
        "nll_lagged_in_sample": mean_nll(ctx=ctx, mixture=in_sample, device=device),
        "nll_lagless_cross_fitted": mean_nll(ctx=ctx, mixture=lagless_mixture, device=device),
    }  # fmt: skip
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for arm, net in (("G1-trust", trust_net), ("G1-insample", insample_net)):
        frame = forecast_month(month=month, seed=seed, trust_net=net, ctx=ctx, device=device)
        check_keys(month=month, seed=seed, frame=frame)
        (PRED_DIR / arm).mkdir(parents=True, exist_ok=True)
        frame.write_parquet(PRED_DIR / arm / f"{month}_s{seed}_w-1.parquet")
        torch.save(net.state_dict(), OOF_DIR / f"{arm}_{month}_s{seed}.pt")
    diagnostics["seconds"] = time.monotonic() - started
    (LOG_DIR / f"{month}_s{seed}.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({k: v for k, v in diagnostics.items() if k != "folds"}, indent=2))


def check(*, month: str, seed: int, device: str) -> None:
    """Refit the lagged head on every training row and compare it with the stored A0-inj loss."""
    ctx = build_context(month=month, seed=seed, device=device)
    _net, losses = fit_plain(
        ctx=ctx, mask=np.ones(len(ctx.train), dtype=bool), mean=ctx.scalers.target_mean,
        std=ctx.scalers.target_std, seed=seed, device=device,
    )  # fmt: skip
    saved = torch.load(fit.MODEL_DIR / "A0-inj" / f"{month}_s{seed}.pt", weights_only=False)
    print(f"refit loss {losses[-1]:.4f}, stored A0-inj loss {saved['losses'][-1]:.4f}")


def main() -> None:
    """Run one command for one scored month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "check"))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "fit":
        run_month(month=args.month, seed=args.seed, device=args.device)
    else:
        check(month=args.month, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
