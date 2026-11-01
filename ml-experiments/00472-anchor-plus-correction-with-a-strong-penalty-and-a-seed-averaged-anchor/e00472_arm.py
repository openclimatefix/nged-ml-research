"""Experiment 472: experiment 467's lag correction with a penalty of 100, or a seed-averaged anchor.

Copied from experiment 467's `e00467_arm.py` (merged, so not edited there). The correction network
reads the three origin lags, the surprise at each lag, and the two lead features, and shifts and
rescales every component of the anchor mixture. Its last layer starts at zero and an L2 penalty
pulls its outputs to zero. Two arms:

- `P100`: the single-seed anchor (a 3-component mixture) with penalty 100.
- `AvgAnchor`: penalty 1e-2 and an anchor that is the equal-weight mixture (9 components) of the
  cross-fitted lagless heads of seeds 0 to 2, built as experiment 454's `LL-ens3` is. The surprise
  at each lag is the lag minus that mixture's median (cached under `work/anchor_median/`).

Network seed `s` trains on injected world `100 + s` and forecasts month rows on worlds -1 (real)
and 200 to 202 (held-out injected load shifts). Forecasts go to `work/predictions/<arm>/`.

```bash
<PYTHONPATH as e00404_fit.py shows, plus the 454 folder and this folder> DATA_PATH_INTERNAL=... \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00472-*/e00472_arm.py run \
    --arm P100 --seeds 0 1 2
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
import e00454_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch import nn  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00472-anchor-plus-correction-with-a-strong-penalty-and-a-seed-averaged-anchor"
    / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
PRED_DIR: Final[Path] = WORK_DIR / "predictions"
MEDIAN_DIR: Final[Path] = WORK_DIR / "anchor_median"
ARMS: Final[dict[str, tuple[float, int]]] = {"P100": (100.0, 1), "AvgAnchor": (1e-2, 3)}
"""Each arm's L2 penalty and its number of anchor seeds (seeds 0 to n - 1, equal weights)."""
ANCHOR_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
HIDDEN: Final[int] = 32
INPUTS: Final[int] = 9 + 3 + 2
"""Three lags of (value, is-missing flag, age), the surprise at each lag, and two lead features."""
MEDIAN_LEVEL_INDEX: Final[int] = 5
"""Position of the 0.5 quantile among the 11 quantile levels."""
DEFAULT_MONTHS: Final[tuple[str, ...]] = build.MONTHS


class AnchorCorrection(nn.Module):
    """A hidden layer of 32 that shifts and rescales the anchor mixture; it starts at zero."""

    def __init__(self, *, components: int) -> None:
        """Build the network; the output layer starts at zero weights and zero bias."""
        super().__init__()
        self.components = components
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
        log_w, mean, std = models.mixture_parts(out=ll, components=self.components)
        return (log_w, mean + out[:, :1], std * torch.exp(out[:, 1:2])), out


def anchor_median(*, month: str, n_seeds: int) -> np.ndarray:
    """Return the equal-weight anchor mixture's median (capacity fraction) at every table row.

    The result has one row per table row and columns (primary, grid, median). All the anchor
    seeds' tables must hold the same (primary, grid) keys.
    """
    path = MEDIAN_DIR / f"{month}_n{n_seeds}.npy"
    if path.exists():
        return np.load(path)
    tables = [
        pl.read_parquet(lagless.table_path(month=month, seed=s)).sort("primary", "grid")
        for s in ANCHOR_SEEDS[:n_seeds]
    ]
    primary = tables[0]["primary"].to_numpy().astype(np.int64)
    grid = tables[0]["grid"].to_numpy().astype(np.int64)
    for table in tables[1:]:
        assert np.array_equal(table["primary"].to_numpy(), primary), "anchor tables differ"
        assert np.array_equal(table["grid"].to_numpy(), grid), "anchor tables differ"
    heads = [
        build.lagless_head(month=month, seed=s, primary=primary, target=grid)
        for s in ANCHOR_SEEDS[:n_seeds]
    ]
    median = build.mix_equal(heads=heads)[:, MEDIAN_LEVEL_INDEX]
    out = np.stack([primary.astype(np.float64), grid.astype(np.float64), median], axis=1)
    MEDIAN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def surprise_of_anchor(*, power: np.ndarray, month: str, n_seeds: int) -> torch.Tensor:
    """Return the world's power minus the anchor mixture's median (NaN where unknown)."""
    rows = anchor_median(month=month, n_seeds=n_seeds)
    median = np.full(power.shape, np.nan, dtype=np.float32)
    median[rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64)] = rows[:, 2]
    return torch.from_numpy(power - median)


def anchor_in_network_units(
    *, pool: fit.Pool, scalers: fit.Scalers, month: str, n_seeds: int, device: str
) -> torch.Tensor:
    """Return the pool's anchor (rows, 3 K) in the network's z units for K = 3 n_seeds components.

    The columns are the log weights (each seed's weights times 1 / n_seeds), the means, and the log
    standard deviations, laid out as `models.mixture_parts` reads them.
    """
    primary = pool.primary.cpu().numpy()
    grid = pool.target.cpu().numpy()
    log_w, means, log_stds = [], [], []
    for s in ANCHOR_SEEDS[:n_seeds]:
        arrays = fit.table_arrays(table=pl.read_parquet(lagless.table_path(month=month, seed=s)))
        raw, present = fit.lagless_in_network_units(
            arrays=arrays, primary=primary, grid=grid, mean=scalers.target_mean,
            std=scalers.target_std,
        )  # fmt: skip
        assert present.all(), f"anchor seed {s} misses pool rows in {month}"
        raw_t = torch.from_numpy(raw)
        log_w.append(torch.log_softmax(raw_t[:, :3], dim=1) - float(np.log(n_seeds)))
        means.append(raw_t[:, 3:6])
        log_stds.append(raw_t[:, 6:9])
    return torch.cat([*log_w, *means, *log_stds], dim=1).to(device)


def _anchor_and_surprise(
    *, month: str, seed: int, n_seeds: int, power_np: np.ndarray
) -> torch.Tensor:
    """Return the surprise array: the single seed's own table for one seed, else the mixture."""
    if n_seeds == 1:
        table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
        return fit.surprise_array(power=power_np, table=table)
    return surprise_of_anchor(power=power_np, month=month, n_seeds=n_seeds)


def fit_one(*, arm: str, month: str, seed: int, device: str, stride: int = 2) -> Path:
    """Fit the correction on one scored month and seed; save it; return the path."""
    started = time.monotonic()
    lam, n_seeds = ARMS[arm]
    train, _test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    assert train is not None
    if n_seeds > 1:
        train.ll = anchor_in_network_units(
            pool=train, scalers=scalers, month=month, n_seeds=n_seeds, device=device
        )
    power_np, _code = fit.world_power(seed=100 + seed)
    power = torch.tensor(power_np, device=device)
    lags, _grid = fit.lag_features(
        pool=train,  # ty: ignore[invalid-argument-type]
        power=power,
        lag_stats=lag_stats,
        device=device,
        surprise=_anchor_and_surprise(month=month, seed=seed, n_seeds=n_seeds, power_np=power_np),
    )
    torch.manual_seed(seed)
    net = AnchorCorrection(components=3 * n_seeds).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)
    losses = []
    for epoch in range(fit.EPOCHS):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            idx = order[start : start + fit.BATCH]
            (log_w, mean, std), out = net(lags[idx], train.lead[idx], train.ll[idx])
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = train.weight[idx]
            loss = (nll * w).sum() / w.sum() + lam * (out**2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
        print(f"{arm} {month} s{seed} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    path = MODEL_DIR / arm / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(), "arm": arm, "penalty": lam, "month": month, "seed": seed,
            "components": 3 * n_seeds, "target_mean": scalers.target_mean,
            "target_std": scalers.target_std, "standardiser_mean": scalers.standardiser.mean,
            "standardiser_std": scalers.standardiser.std,
            "standardiser_mask": scalers.standardiser.mask, "lag_stats": lag_stats,
            "losses": losses, "rows": n, "seconds": time.monotonic() - started,
        },
        path,
    )  # fmt: skip
    print(f"{arm} {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


@torch.no_grad()
def forecast(
    *, arm: str, month: str, seed: int, world: int, device: str, real: np.ndarray
) -> pl.DataFrame:
    """Forecast the scored month's rows on a scoring world with a saved correction network."""
    _lam, n_seeds = ARMS[arm]
    saved = torch.load(MODEL_DIR / arm / f"{month}_s{seed}.pt", weights_only=False)
    net = AnchorCorrection(components=saved["components"]).to(device)
    net.load_state_dict(saved["state"])
    net.eval()
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    if n_seeds > 1:
        test.ll = anchor_in_network_units(
            pool=test, scalers=scalers, month=month, n_seeds=n_seeds, device=device
        )
    power_np, code = fit.world_power(seed=None if world < 0 else world)
    power = torch.tensor(power_np, device=device)
    lags, _grid = fit.lag_features(
        pool=test, power=power, lag_stats=lag_stats, device=device,
        surprise=_anchor_and_surprise(month=month, seed=seed, n_seeds=n_seeds, power_np=power_np),
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


def prediction_path(*, arm: str, month: str, seed: int, world: int) -> Path:
    """Return where a forecast frame of this experiment is stored."""
    return PRED_DIR / arm / f"{month}_s{seed}_w{world}.parquet"


def predict_one(*, arm: str, month: str, seed: int, device: str, worlds: list[int]) -> None:
    """Write the forecast frames of one fitted network on the given worlds, atomically."""
    real = score.real_event_mask()
    for world in worlds:
        frame = forecast(arm=arm, month=month, seed=seed, world=world, device=device, real=real)
        target = prediction_path(arm=arm, month=month, seed=seed, world=world)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".tmp")
        frame.write_parquet(partial)
        partial.replace(target)
        print(f"{arm} {month} s{seed} w{world}: {frame.height:,} rows -> {target}", flush=True)


def main() -> None:
    """Fit and forecast one arm over the given seeds and months, or either step alone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "run"))
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--months", nargs="*", default=list(DEFAULT_MONTHS))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--worlds", type=int, nargs="*", default=[-1, *score.SCORING_SEEDS])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for seed in args.seeds:
        for month in args.months:
            if args.command in ("fit", "run"):
                fit_one(arm=args.arm, month=month, seed=seed, device=args.device)
            if args.command in ("predict", "run"):
                predict_one(
                    arm=args.arm, month=month, seed=seed, device=args.device, worlds=args.worlds
                )


if __name__ == "__main__":
    main()
