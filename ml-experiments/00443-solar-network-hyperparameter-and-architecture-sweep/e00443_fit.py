# ruff: noqa: D102, D107, E501
"""Fit the settings-sweep networks of experiment 443 on the GPU and store their quantiles.

Rows, refits, features, and scoring follow experiment 430's `g0_onepass` (the skeleton): farms A to
F pooled over the six day bands, four quarterly refits from 2025-09, each training on rows whose
valid time is at least one day before the refit. With every setting at its skeleton level the
network, the draw of the weights, the batches, and the best-epoch choice are the same as 430's, so
the gate reproduces the stored forecasts. One job is one (arm, seed, refit); its prediction table
holds every farm and band and is written atomically and skipped when it exists.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00443-*/e00443_fit.py ARM [ARM ...] [--seeds 0,1,2] [--refits 2025-09,...]
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_common as c430  # ty: ignore[unresolved-import]
import e00430_fit as f430  # ty: ignore[unresolved-import]
import e00430_net as n430  # ty: ignore[unresolved-import]
import e00443_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.tail_calibration import level_column
from torch import nn  # ty: ignore[unresolved-import]

DEVICE: Final = f430.DEVICE
INFERENCE_CHUNK: Final[int] = f430.INFERENCE_CHUNK
LOG_SQRT_2PI: Final[float] = n430.LOG_SQRT_2PI
FARM_INDEX: Final[dict[str, int]] = {f: i for i, f in enumerate(c310.FARMS)}


@dataclass
class Data:
    """Standardised inputs and targets of many rows, as GPU tensors."""

    x: torch.Tensor
    farm: torch.Tensor
    y: torch.Tensor
    ceiling: torch.Tensor


class SweepNet(nn.Module):
    """Experiment 430's pooled network with the settings of `common.Config`.

    With the skeleton settings the layers are created in the same order as 430's `UnifiedNet`, so a
    seed draws the same starting weights.
    """

    def __init__(self, *, n_static: int, config: common.Config) -> None:
        super().__init__()
        k = config.components
        self.k = k
        self.embedding = (
            nn.Embedding(len(c310.FARMS), config.embedding) if config.embedding > 0 else None
        )
        layers: list[nn.Module] = []
        width_in = n_static + config.embedding
        for _ in range(config.depth):
            layers += [nn.Linear(width_in, config.width), nn.GELU()]
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            width_in = config.width
        layers.append(nn.Linear(width_in, 3 * k))
        self.trunk = nn.Sequential(*layers)
        last = self.trunk[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        means = [0.0] if k == 1 else [float(v) for v in np.linspace(-0.1, 0.1, k)]
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0] * k + means + [float(np.log(0.08))] * k))

    def parameters_of(
        self, *, x: torch.Tensor, farm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log weights, means, and standard deviations of the Gaussians."""
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(farm)], dim=1)
        out = self.trunk(x)
        k = self.k
        log_w = torch.log_softmax(out[:, :k], dim=1)
        std = torch.exp(out[:, 2 * k : 3 * k].clamp(-6.0, 1.0))
        return log_w, out[:, k : 2 * k], std

    def nll_per_row(self, *, data: Data, rows: torch.Tensor) -> torch.Tensor:
        log_w, mean, std = self.parameters_of(x=data.x[rows], farm=data.farm[rows])
        y = data.y[rows][:, None]
        log_pdf = -0.5 * ((y - mean) / std) ** 2 - torch.log(std) - LOG_SQRT_2PI
        return -torch.logsumexp(log_w + log_pdf, dim=1)


@torch.no_grad()
def quantiles(
    *, model: SweepNet, x: torch.Tensor, farm: torch.Tensor, ceiling: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantiles per row of the mixture, clamped to [0, ceiling]."""
    levels = n430.LEVELS.to(DEVICE)
    log_w, mean, std = model.parameters_of(x=x, farm=farm)
    result = n430.bisect_gaussian_mixture(
        w=log_w.exp().double(), mean=mean.double(), std=std.double(), levels=levels
    )
    c = ceiling.double()
    return torch.minimum(result.clamp(min=0.0), c[:, None])


def training_data(*, cutoff: datetime, stats: dict) -> tuple[Data, np.ndarray, np.ndarray]:
    """Return the training rows before the cutoff, a validation flag, and each row's age in days."""
    xs, farms, ys, ceilings, vals, ages = [], [], [], [], [], []
    for farm in c310.FARMS:
        for band in c310.BAND_ORDER:
            piece = f430.ens_piece(farm=farm, band=band, cutoff=cutoff, need_members=False)
            frame = f430.solar_base(farm, band)
            older = frame.filter(pl.col("time") < cutoff)
            assert older.height == len(piece)
            xs.append(f430.matrix(piece=piece, columns=c430.GATE_COLUMNS, stats=stats))
            farms.append(np.full(len(piece), FARM_INDEX[farm], dtype=np.int64))
            ys.append(piece.y)
            ceilings.append(piece.ceiling)
            vals.append(piece.is_val)
            ages.append(
                older.select((pl.lit(cutoff) - pl.col("time")).dt.total_seconds() / 86400.0)
                .to_series()
                .to_numpy()
            )
    data = Data(
        x=torch.tensor(np.concatenate(xs), device=DEVICE),
        farm=torch.tensor(np.concatenate(farms), device=DEVICE),
        y=torch.tensor(np.concatenate(ys), device=DEVICE),
        ceiling=torch.tensor(np.concatenate(ceilings), device=DEVICE),
    )
    return data, np.concatenate(vals), np.concatenate(ages)


def stats_for(*, cutoff: datetime) -> dict:
    """Return the standardiser of the training rows before the cutoff (as experiment 430 does)."""
    pieces = [
        f430.ens_piece(farm=f, band=b, cutoff=cutoff, need_members=False)
        for f in c310.FARMS
        for b in c310.BAND_ORDER
    ]
    return f430.standardiser(pieces=pieces, columns=c430.GATE_COLUMNS)


def train_network(
    *,
    data: Data,
    is_val: np.ndarray,
    age_days: np.ndarray,
    config: common.Config,
    seed: int,
) -> SweepNet:
    """Fit one network: AdamW, `config.epochs` epochs, the best epoch on the last 61 days."""
    torch.manual_seed(seed)
    model = SweepNet(n_static=data.x.shape[1], config=config).to(DEVICE)
    fit_rows = torch.tensor(np.flatnonzero(~is_val), device=DEVICE)
    val_rows = torch.tensor(np.flatnonzero(is_val), device=DEVICE)
    weights = None
    if config.half_life_years is not None:
        w = 0.5 ** (age_days / (365.25 * config.half_life_years))
        weights = torch.tensor(w / w[~is_val].mean(), dtype=torch.float32, device=DEVICE)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(config.epochs):
        model.train()
        order = fit_rows[torch.randperm(len(fit_rows), generator=generator, device=DEVICE)]
        for start in range(0, len(order), c430.BATCH):
            rows = order[start : start + c430.BATCH]
            per_row = model.nll_per_row(data=data, rows=rows)
            loss = per_row.mean() if weights is None else (per_row * weights[rows]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum(
                float(model.nll_per_row(data=data, rows=val_rows[s : s + INFERENCE_CHUNK]).mean())
                * len(val_rows[s : s + INFERENCE_CHUNK])
                for s in range(0, len(val_rows), INFERENCE_CHUNK)
            ) / len(val_rows)
        if total < best:
            best, best_state = total, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_rows(
    *,
    models: dict[str, SweepNet],
    stats: dict,
    start: datetime,
    end: datetime,
    cutoff: datetime,
    seed: int,
    arm: str,
) -> pl.DataFrame:
    """Return every farm and band's scored rows of the window with the 11 quantiles.

    `models` holds one pooled network under "all", or one network per farm label.
    """
    frames = []
    for farm in c310.FARMS:
        model = models.get(farm, models.get("all"))
        assert model is not None
        for band in c310.BAND_ORDER:
            frame = f430.solar_base(farm, band)
            keep = (
                ((frame["init_time"] >= start) & (frame["init_time"] < end))
                & (frame["time"] < c310.SCORED_END)
            ).to_numpy()
            rows = frame.filter(pl.Series(keep))
            piece = f430.Piece(
                columns=f430._columns(rows, None),
                geo=np.zeros((rows.height, 3), dtype=np.float32),
                members=np.zeros((rows.height, 1, 2), dtype=np.float32),
                y=np.zeros(rows.height, dtype=np.float32),
                ceiling=np.full(
                    rows.height, f430.farm_ceiling(farm, band, cutoff), dtype=np.float32
                ),
            )
            x = torch.tensor(
                f430.matrix(piece=piece, columns=c430.GATE_COLUMNS, stats=stats), device=DEVICE
            )
            ceiling = torch.tensor(piece.ceiling, device=DEVICE)
            farm_id = torch.full((rows.height,), FARM_INDEX[farm], dtype=torch.long, device=DEVICE)
            out = np.empty((rows.height, len(c310.LEVELS)), dtype=np.float32)
            for s in range(0, rows.height, INFERENCE_CHUNK):
                sl = slice(s, s + INFERENCE_CHUNK)
                out[sl] = (
                    quantiles(model=model, x=x[sl], farm=farm_id[sl], ceiling=ceiling[sl])
                    .cpu()
                    .numpy()
                )
            frames.append(
                rows.select("time", "init_time", "lead_hours").with_columns(
                    time_series_id=pl.lit(farm),
                    arm=pl.lit(arm),
                    seed=pl.lit(seed),
                    setting=pl.lit("own_history"),
                    band=pl.lit(band),
                    actual=rows["output"].cast(pl.Float64),
                    **{
                        level_column(level=lv): pl.Series(out[:, i])
                        for i, lv in enumerate(c310.LEVELS)
                    },
                )
            )
    return pl.concat(frames)


def run_job(*, arm: str, config: common.Config, seed: int, start: datetime) -> str:
    """Fit one (arm, seed, refit) and write its prediction table."""
    path = common.PRED_DIR / arm / f"s{seed}_{start:%Y-%m}.parquet"
    if path.exists():
        return f"{arm} s{seed} {start:%Y-%m} skipped"
    began = time.time()
    cutoff = start - timedelta(days=f310.REFIT_GAP_DAYS)
    stats = stats_for(cutoff=cutoff)
    data, is_val, age_days = training_data(cutoff=cutoff, stats=stats)
    if config.per_farm:
        models = {}
        for farm, index in FARM_INDEX.items():
            keep = (data.farm == index).cpu().numpy()
            sub = Data(
                x=data.x[keep], farm=data.farm[keep], y=data.y[keep], ceiling=data.ceiling[keep]
            )
            models[farm] = train_network(
                data=sub, is_val=is_val[keep], age_days=age_days[keep], config=config, seed=seed
            )
    else:
        models = {
            "all": train_network(
                data=data, is_val=is_val, age_days=age_days, config=config, seed=seed
            )
        }
    end = f310.month_add(start=start, months=f310.REFIT_MONTHS)
    table = predict_rows(
        models=models, stats=stats, start=start, end=end, cutoff=cutoff, seed=seed, arm=arm
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.write_parquet(tmp)
    tmp.replace(path)
    del models, data
    torch.cuda.empty_cache()
    return f"{arm} s{seed} {start:%Y-%m} {table.height} rows {time.time() - began:.0f}s"


def main() -> None:
    """Run the named arms, seeds, and refits one after another in this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--seeds", default=",".join(map(str, common.SEEDS_SCREEN)))
    parser.add_argument("--refits", default=",".join(f"{r:%Y-%m}" for r in f430.REFITS))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    refits = [r for r in f430.REFITS if f"{r:%Y-%m}" in args.refits.split(",")]
    arms = common.all_arms()
    for arm in args.arms:
        for seed in seeds:
            for start in refits:
                print(run_job(arm=arm, config=arms[arm], seed=seed, start=start), flush=True)


if __name__ == "__main__":
    main()
