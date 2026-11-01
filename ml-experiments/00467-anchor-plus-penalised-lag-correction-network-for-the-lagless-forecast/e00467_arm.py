"""Experiment 467: a frozen lagless anchor plus a small, penalised lag correction network.

The anchor is experiment 404's frozen cross-fitted lagless table (a 3-component mixture per row, a
constant here). The correction network reads the three origin lags, the surprise at each lag (the
lag minus the lagless median), and the two lead features. It outputs a shift of every anchor
component mean and a change of the log of every anchor component standard deviation. Its last layer
starts at zero, and an L2 penalty on its outputs pulls it back to zero, so the forecast equals the
anchor wherever the lags carry no usable signal. Everything else (rows, injected worlds, epochs,
recency weights, scoring worlds) is experiment 404's.

```bash
<PYTHONPATH as e00404_fit.py shows, plus this folder> DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00467-*/e00467_arm.py fit --penalty 1e-2 --month 2026-06
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch import nn  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00467-anchor-plus-penalised-lag-correction-network-for-the-lagless-forecast"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
PENALTIES: Final[tuple[str, ...]] = ("1e-3", "1e-2", "1e-1")
HIDDEN: Final[int] = 32
INPUTS: Final[int] = 9 + 3 + 2
"""Three lags of (value, is-missing flag, age), the surprise at each lag, and two lead features."""
ZERO_SHIFT_CAPACITY: Final[float] = 0.01
"""A shift below this fraction of capacity counts as zero in the stop rule."""


def arm_name(*, penalty: str) -> str:
    """Return the arm name that prediction files carry."""
    return f"C{penalty}"


class AnchorCorrection(nn.Module):
    """A hidden layer of 32 that shifts and rescales the anchor mixture; it starts at zero."""

    def __init__(self) -> None:
        """Build the network; the output layer starts at zero weights and zero bias."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(INPUTS, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self, lags: torch.Tensor, lead: torch.Tensor, ll: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the corrected mixture (log weights, means, stds) and the raw outputs (rows, 2)."""
        out = self.net(torch.cat([lags, lead], dim=-1))
        log_w, mean, std = models.mixture_parts(out=ll, components=models.COMPONENTS)
        return (log_w, mean + out[:, :1], std * torch.exp(out[:, 1:2])), out


def fit_one(*, penalty: str, month: str, seed: int, device: str, stride: int = 2) -> Path:
    """Fit the correction on one scored month and seed; save it; return the path."""
    started = time.monotonic()
    lam = float(penalty)
    train, _test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    power_np, _code = fit.world_power(seed=100 + seed)
    power = torch.tensor(power_np, device=device)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    lags, _grid = fit.lag_features(
        pool=train,  # ty: ignore[invalid-argument-type]
        power=power,
        lag_stats=lag_stats,
        device=device,
        surprise=fit.surprise_array(power=power_np, table=table),
    )
    torch.manual_seed(seed)
    net = AnchorCorrection().to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)  # ty: ignore[invalid-argument-type]
    losses = []
    for epoch in range(fit.EPOCHS):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            idx = order[start : start + fit.BATCH]
            (log_w, mean, std), out = net(lags[idx], train.lead[idx], train.ll[idx])  # ty: ignore[unresolved-attribute]
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]  # ty: ignore[unresolved-attribute]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = train.weight[idx]  # ty: ignore[unresolved-attribute]
            loss = (nll * w).sum() / w.sum() + lam * (out**2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
        print(f"C{penalty} {month} s{seed} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    path = MODEL_DIR / arm_name(penalty=penalty) / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(),
            "penalty": penalty,
            "month": month,
            "seed": seed,
            "n_base": train.base.shape[1],  # ty: ignore[unresolved-attribute]
            "target_mean": scalers.target_mean,
            "target_std": scalers.target_std,
            "standardiser_mean": scalers.standardiser.mean,
            "standardiser_std": scalers.standardiser.std,
            "standardiser_mask": scalers.standardiser.mask,
            "lag_stats": lag_stats,
            "losses": losses,
            "rows": n,
            "seconds": time.monotonic() - started,
        },
        path,
    )
    print(f"C{penalty} {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


@torch.no_grad()
def forecast(
    *, penalty: str, month: str, seed: int, world: int, device: str, real: np.ndarray
) -> pl.DataFrame:
    """Forecast the scored month's rows on a scoring world with a saved correction network."""
    saved = torch.load(
        MODEL_DIR / arm_name(penalty=penalty) / f"{month}_s{seed}.pt", weights_only=False
    )
    net = AnchorCorrection().to(device)
    net.load_state_dict(saved["state"])
    net.eval()
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    power_np, code = fit.world_power(seed=None if world < 0 else world)
    power = torch.tensor(power_np, device=device)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    lags, _grid = fit.lag_features(
        pool=test, power=power, lag_stats=lag_stats, device=device,
        surprise=fit.surprise_array(power=power_np, table=table),
    )  # fmt: skip
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    quantiles, shifts = [], []
    for start in range(0, len(test), 16384):
        idx = torch.arange(start, min(start + 16384, len(test)), device=device)
        (log_w, mean, std), out = net(lags[idx], test.lead[idx], test.ll[idx])
        primary = test.primary[idx]
        quantiles.append(
            score.mixture_quantiles_capacity(
                log_w=log_w, mean=mean, std=std, mu=mu[primary], sd=sd[primary]
            ).cpu()
        )
        shifts.append(torch.stack([out[:, 0] * sd[primary], out[:, 1]], dim=1).cpu())
    q = torch.cat(quantiles).numpy()
    shift = torch.cat(shifts).numpy()
    flags = score.Flags(code=code, real=real, power=power_np, device=device).of(
        primary=test.primary, origin=test.origin, target=test.target
    )
    columns: dict[str, np.ndarray] = {
        "primary": test.primary.cpu().numpy().astype(np.int16),
        "origin": test.origin.cpu().numpy().astype(np.int32),
        "target": test.target.cpu().numpy().astype(np.int32),
        "lead_hours": test.lead_hours.cpu().numpy(),
        "actual": test.y.cpu().numpy(),
        **{name: q[:, i] for i, name in enumerate(score.LEVEL_NAMES)},
        **{k: v.cpu().numpy() for k, v in flags.items()},
        "shift_capacity": shift[:, 0],
        "log_scale_change": shift[:, 1],
    }
    return pl.DataFrame(columns)


def main() -> None:
    """Fit or forecast one penalty on one month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict"))
    parser.add_argument("--penalty", choices=PENALTIES, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, nargs="*", default=[-1, *score.SCORING_SEEDS])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "fit":
        fit_one(penalty=args.penalty, month=args.month, seed=args.seed, device=args.device)
        return
    real = score.real_event_mask()
    for world in args.worlds:
        frame = forecast(
            penalty=args.penalty, month=args.month, seed=args.seed, world=world,
            device=args.device, real=real,
        )  # fmt: skip
        target = score.prediction_path(
            arm=arm_name(penalty=args.penalty), month=args.month, seed=args.seed, world=world
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        print(
            f"C{args.penalty} {args.month} s{args.seed} w{world}: {frame.height:,} rows -> {target}"
        )


if __name__ == "__main__":
    main()
