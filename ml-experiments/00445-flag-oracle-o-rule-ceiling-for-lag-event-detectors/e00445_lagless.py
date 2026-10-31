"""The lagless module's K-fold cross-fitted forecast table of experiment 404.

For one scored month, the half-hours of the 24 months before it are split into K contiguous blocks.
A pooled lagless module (all 16 Primaries, a Primary embedding, no power lag) is fitted on the other
blocks and forecasts the held-out block, so every half-hour before the scored month gets an
out-of-sample lagless forecast. One further module fitted on every half-hour before the scored month
forecasts the scored month and the days after it. Nothing from the scored month or later enters any
fit. The table is one parquet per scored month and seed.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00445-*/e00445_lagless.py build --month 2026-06 \
    --seed 0 --device cuda
```
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00445_data as data  # ty: ignore[unresolved-import]
import e00445_models as models  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "lagless"
)
FOLDS: Final[int] = 6
WINDOW_STEPS: Final[int] = prep.WINDOW_DAYS * common.STEPS_PER_DAY
FURTHER_STEPS: Final[int] = 62 * common.STEPS_PER_DAY
"""The further module forecasts the scored month and the month after it."""
CLIP_SD: Final[float] = 5.0
EPOCHS: Final[int] = 6
BATCH: Final[int] = 4096
LEARNING_RATE: Final[float] = 2e-3
SHORT_DROP_DATE: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
"""Rows of these Primaries before `SHORT_DROP_DATE` are dropped (maintainer rule, 2026-10-30)."""


@dataclass
class Pool:
    """The dense rows of all Primaries over a window, before any standardising.

    Attributes:
        site: Primary index per row.
        grid: Grid index per row.
        x: Raw lagless inputs.
        y: Observed power over capacity, NaN where unobserved.
    """

    site: np.ndarray
    grid: np.ndarray
    x: np.ndarray
    y: np.ndarray


def power_arrays() -> dict[str, np.ndarray]:
    """Return each Primary's observed power over capacity on the dense grid, short history cut."""
    _, arrays = fd._series_and_arrays()
    cut = int(common.grid_index(times=SHORT_DROP_DATE))
    out = {}
    for site, a in arrays.items():
        power = a.power.astype(np.float32).copy()
        if site in SHORT_HISTORY_SITES:
            power[:cut] = np.nan
        out[site] = power
    return out


def build_pool(*, lo: int, hi: int) -> Pool:
    """Return the dense rows of every Primary with grid index in `[lo, hi)` and a weather input."""
    power = power_arrays()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        dense = data.dense_inputs(site_row=row)
        grid = dense.first + np.arange(len(dense.x))
        keep = (grid >= lo) & (grid < hi) & (dense.valid == 1)
        parts.append(
            Pool(
                site=np.full(int(keep.sum()), row["primary"], dtype=np.int64),
                grid=grid[keep],
                x=dense.x[keep],
                y=power[row["site"]][grid[keep]],
            )
        )
    return Pool(
        site=np.concatenate([p.site for p in parts]),
        grid=np.concatenate([p.grid for p in parts]),
        x=np.concatenate([p.x for p in parts]),
        y=np.concatenate([p.y for p in parts]),
    )


def block_of(*, grid: np.ndarray, lo: int, hi: int, folds: int) -> np.ndarray:
    """Return the contiguous block (0 to folds - 1) of each grid index in `[lo, hi)`."""
    return np.minimum((grid - lo) * folds // (hi - lo), folds - 1)


class Scaler:
    """Mean and standard deviation of the inputs and of each Primary's target."""

    def __init__(self, *, x: np.ndarray, site: np.ndarray, y: np.ndarray, n_primaries: int) -> None:
        """Fit on the rows with an observed target."""
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        self.y_mean = np.zeros(n_primaries)
        self.y_std = np.ones(n_primaries)
        for p in range(n_primaries):
            values = y[(site == p) & np.isfinite(y)]
            if len(values):
                self.y_mean[p], self.y_std[p] = values.mean(), max(float(values.std()), 1e-3)

    def inputs(self, *, x: np.ndarray) -> np.ndarray:
        """Standardise and clip the inputs."""
        return np.clip((x - self.mean) / self.std, -CLIP_SD, CLIP_SD).astype(np.float32)

    def target(self, *, y: np.ndarray, site: np.ndarray) -> np.ndarray:
        """Return the standardised target."""
        return ((y - self.y_mean[site]) / self.y_std[site]).astype(np.float32)


def fit_module(
    *, x: torch.Tensor, z: torch.Tensor, site: torch.Tensor, weight: torch.Tensor, seed: int
) -> models.LaglessModule:
    """Fit a pooled lagless module on its own mixture negative log-likelihood."""
    torch.manual_seed(seed)
    module = models.LaglessModule(n_base=x.shape[1], n_primaries=16).to(x.device)
    optimiser = torch.optim.Adam(module.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(x)
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator).to(x.device)
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            log_w, mean, std = models.mixture_parts(
                out=module(x[idx], site[idx]), components=models.COMPONENTS
            )
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z[idx])
            loss = (nll * weight[idx]).sum() / weight[idx].sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return module


@torch.no_grad()
def predict(
    *, module: models.LaglessModule, x: torch.Tensor, site: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw mixture outputs (rows, 9) and the quantiles q05, q50, q95 in z units."""
    module.eval()
    raw, quant = [], []
    for start in range(0, len(x), 65536):
        out = module(x[start : start + 65536], site[start : start + 65536])
        log_w, mean, std = models.mixture_parts(out=out, components=models.COMPONENTS)
        q = models.mixture_quantiles(log_w=log_w, mean=mean, std=std)
        raw.append(out.cpu().numpy())
        quant.append(q[:, [1, 5, 9]].cpu().numpy())
    return np.concatenate(raw), np.concatenate(quant)


def crossfit_table(*, month: str, seed: int, device: str, folds: int = FOLDS) -> pl.DataFrame:
    """Return the cross-fitted table for a scored month, and the further module's forecasts.

    Each fold fits its own input scaler and target mean and standard deviation on its training
    blocks only, so the held-out block informs nothing about its own forecast.
    """
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = max(data.FIRST_VALID_INDEX, end - WINDOW_STEPS)
    train = build_pool(lo=start, hi=end)
    after = build_pool(lo=end, hi=end + FURTHER_STEPS)
    age_days = (end - train.grid) / common.STEPS_PER_DAY
    weight = (0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)).astype(np.float32)
    observed = np.isfinite(train.y)
    block = block_of(grid=train.grid, lo=start, hi=end, folds=folds)

    def fit_and_predict(
        *, fit_mask: np.ndarray, pool: Pool, rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Scaler]:
        """Fit a module and scaler on the training rows in `fit_mask`; forecast `pool[rows]`."""
        scaler = Scaler(
            x=train.x[fit_mask], site=train.site[fit_mask], y=train.y[fit_mask], n_primaries=16
        )
        z = scaler.target(y=train.y[fit_mask], site=train.site[fit_mask])
        module = fit_module(
            x=torch.tensor(scaler.inputs(x=train.x[fit_mask]), device=device),
            z=torch.tensor(z, device=device),
            site=torch.tensor(train.site[fit_mask], device=device),
            weight=torch.tensor(weight[fit_mask], device=device),
            seed=seed,
        )
        raw_out, quant_out = predict(
            module=module,
            x=torch.tensor(scaler.inputs(x=pool.x[rows]), device=device),
            site=torch.tensor(pool.site[rows], device=device),
        )
        return raw_out, quant_out, scaler

    def frame(
        *, pool: Pool, rows: np.ndarray, raw_out: np.ndarray, quant_out: np.ndarray,
        scaler: Scaler, source: str,
    ) -> pl.DataFrame:  # fmt: skip
        site = pool.site[rows]
        q = quant_out * scaler.y_std[site][:, None] + scaler.y_mean[site][:, None]
        columns = {
            "primary": site.astype(np.int16),
            "grid": pool.grid[rows].astype(np.int32),
            "source": np.full(len(site), source),
            **{f"raw_{i}": raw_out[:, i] for i in range(9)},
            "q05": q[:, 0].astype(np.float32),
            "q50": q[:, 1].astype(np.float32),
            "q95": q[:, 2].astype(np.float32),
            "y_mean": scaler.y_mean[site].astype(np.float32),
            "y_std": scaler.y_std[site].astype(np.float32),
        }
        return pl.DataFrame(columns)

    parts = []
    for k in range(folds):
        held = np.flatnonzero(block == k)
        raw_out, quant_out, scaler = fit_and_predict(
            fit_mask=observed & (block != k), pool=train, rows=held
        )
        parts.append(
            frame(pool=train, rows=held, raw_out=raw_out, quant_out=quant_out, scaler=scaler,
                  source="crossfit")
        )  # fmt: skip
    raw_out, quant_out, scaler = fit_and_predict(
        fit_mask=observed, pool=after, rows=np.arange(len(after.grid))
    )
    parts.append(
        frame(pool=after, rows=np.arange(len(after.grid)), raw_out=raw_out, quant_out=quant_out,
              scaler=scaler, source="further")
    )  # fmt: skip
    return pl.concat(parts)


def table_path(*, month: str, seed: int) -> Path:
    """Return where a scored month's table is stored."""
    return WORK_DIR / f"month={month}" / f"seed={seed}.parquet"


def main() -> None:
    """Build and write a scored month's table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=FOLDS)
    args = parser.parse_args()
    started = time.monotonic()
    table = crossfit_table(month=args.month, seed=args.seed, device=args.device, folds=args.folds)
    target = table_path(month=args.month, seed=args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    print(
        f"{args.month} seed {args.seed}: {table.height:,} rows, {args.folds} folds, "
        f"{time.monotonic() - started:.0f} s -> {target}"
    )


if __name__ == "__main__":
    main()
