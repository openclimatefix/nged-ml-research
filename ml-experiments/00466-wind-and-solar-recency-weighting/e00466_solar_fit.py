# ruff: noqa: PLR0915
"""Fit experiment 430's `g0_onepass` and `a2_member` solar networks with a recency weight.

The fit loop is experiment 461's (`e00461_solar_fit`: experiment 430's loop with `output` and
`lag_power` divided by the p99.9 capacity ratio), with one change: the training loss of
`train_network` (experiment 430's, copied here) multiplies each row's negative log likelihood by
`0.5 ** (age / half_life)`, where age runs from the row's valid time to the refit's cutoff, and the
weights are rescaled to a mean of 1 over the rows the loss sees. Early stopping is unweighted, as in
experiment 443. Arm labels are `<base>_<half-life label>`, such as `g0_onepass_hl2`. Predictions go
to this experiment's `solar_predictions/<arm>/`, in units of the new capacity, and the capacity
table to `solar_capacity.json`. `--gate` fits the base arm with all weights 1, to compare with
experiment 461's stored forecasts.

    PYTHONPATH=<310>:<430>:<461>:<466 folders> OMP_NUM_THREADS=1 systemd-run --user --scope \
        -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00466-*/e00466_solar_fit.py BASE LABEL [LABEL ...] \
        [--seeds 0,1,2] [--refits 2025-09,...]
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_common as common430  # ty: ignore[unresolved-import]
import e00430_fit as fit430  # ty: ignore[unresolved-import]
import e00430_net as net  # ty: ignore[unresolved-import]
import e00461_common as common461  # ty: ignore[unresolved-import]
import e00461_solar_fit as solar461  # ty: ignore[unresolved-import]
import e00466_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

DEVICE: Final = fit430.DEVICE
PRED_DIR: Final = common.DATA_DIR / "solar_predictions" / "p999"
CAPACITY_FILE: Final[str] = "solar_capacity.json"
CYCLE_49R1: Final[datetime] = datetime(2024, 11, 12, tzinfo=UTC)
"""First day of ECMWF's IFS Cycle 49r1 ensemble (the indicator diagnostic's switch)."""
_COLUMNS: Final = fit430._columns  # noqa: SLF001
_JOB: dict[str, object] = {}
"""The half-life and cutoff of the job being fitted, read by `train_network`."""


@dataclass(frozen=True)
class IndicatorConfig(common430.Config):
    """The base arm with one more 0/1 input: 1 for ensemble runs initialised from Cycle 49r1 on."""

    @property
    def columns(self) -> tuple[str, ...]:
        return (*super().columns, "cycle49")


def columns_with_cycle(frame: pl.DataFrame, rows: np.ndarray | None) -> dict[str, np.ndarray]:
    """Return experiment 430's input columns plus `cycle49`, taken from the run's init time."""
    out = _COLUMNS(frame, rows)
    flag = (frame["init_time"] >= CYCLE_49R1).to_numpy().astype(np.float32)
    out["cycle49"] = flag if rows is None else flag[rows]
    return out


def training_ages(*, cutoff: datetime) -> np.ndarray:
    """Return the age in days at the cutoff of every training row, in `run_job`'s piece order."""
    ages = []
    for farm in c310.FARMS:
        for band in c310.BAND_ORDER:
            times = fit430.solar_base(farm, band)["time"]
            older = times.filter(times < cutoff)
            ages.append((cutoff - older).dt.total_seconds().to_numpy() / 86400.0)
    return np.concatenate(ages)


def train_network(
    *,
    pieces: list[fit430.Piece],
    config: common430.Config,
    seed: int,
    stats: dict,
    learning_rate: float,
    init_state: dict | None = None,
) -> net.UnifiedNet:
    """Experiment 430's `train_network` for a plain fit, with a recency weight on the loss."""
    assert init_state is None
    assert config.pretrain == "none"
    half_life = _JOB["half_life"]
    columns = config.columns
    need_members = config.needs_members
    torch.manual_seed(seed)
    plant = fit430.build_plant(config=config, pieces=pieces)
    model = net.UnifiedNet(n_static=len(columns), config=config, plant=plant).to(DEVICE)
    parts = [
        fit430.to_batch(piece=p, columns=columns, stats=stats, need_members=need_members)
        for p in pieces
    ]
    all_batch = net.Batch(
        x=torch.cat([b.x for b in parts]),
        geo=torch.cat([b.geo for b in parts]),
        members=torch.cat([b.members for b in parts]),
        y=torch.cat([b.y for b in parts]),
        ceiling=torch.cat([b.ceiling for b in parts]),
        clear_sky=torch.cat([b.clear_sky for b in parts if b.clear_sky is not None]),
    )
    del parts
    is_val = np.concatenate([p.is_val for p in pieces])
    ages = training_ages(cutoff=_JOB["cutoff"])  # ty: ignore[invalid-argument-type]
    assert len(ages) == len(is_val), (len(ages), len(is_val))
    raw = common.row_weights(age_days=ages, half_life_years=half_life)  # ty: ignore[invalid-argument-type]
    weights = torch.tensor(raw / raw[~is_val].mean(), dtype=torch.float32, device=DEVICE)
    fit_rows = torch.tensor(np.flatnonzero(~is_val), device=DEVICE)
    val_rows = torch.tensor(np.flatnonzero(is_val), device=DEVICE)
    angle_params = [p for n, p in model.named_parameters() if n in ("plant.tilt", "plant.azimuth")]
    plant_params = [
        p
        for n, p in model.named_parameters()
        if n.startswith("plant.") and n not in ("plant.tilt", "plant.azimuth")
    ]
    other = [p for n, p in model.named_parameters() if not n.startswith("plant.")]
    optimiser = torch.optim.AdamW(
        [
            {"params": other, "weight_decay": common430.WEIGHT_DECAY},
            {"params": plant_params, "weight_decay": 0.0},
            {"params": angle_params, "weight_decay": 0.0, "lr": fit430.ANGLE_LEARNING_RATE},
        ],
        lr=learning_rate,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    best, best_state = float("inf"), None

    def gather(index: torch.Tensor) -> net.Batch:
        return net.Batch(
            x=all_batch.x[index],
            geo=all_batch.geo[index],
            members=all_batch.members[index],
            y=all_batch.y[index],
            ceiling=all_batch.ceiling[index],
            clear_sky=all_batch.clear_sky[index] if all_batch.clear_sky is not None else None,
        )

    for _ in range(common430.EPOCHS):
        model.train()
        order = fit_rows[torch.randperm(len(fit_rows), generator=generator, device=DEVICE)]
        for start in range(0, len(order), common430.BATCH):
            index = order[start : start + common430.BATCH]
            loss = (model.nll_per_row(gather(index)) * weights[index]).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            total = sum(
                float(model.nll(gather(val_rows[s : s + fit430.INFERENCE_CHUNK])))
                * len(val_rows[s : s + fit430.INFERENCE_CHUNK])
                for s in range(0, len(val_rows), fit430.INFERENCE_CHUNK)
            ) / len(val_rows)
        if total < best:
            best, best_state = total, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def set_ratios(*, start: datetime) -> None:
    """Set the per-series p99.9 ratio for the refit starting at `start` and record the capacities."""
    cutoff = start - timedelta(days=f310.REFIT_GAP_DAYS)
    table = {}
    for site in c310.FARMS:
        row = common461.capacity(site=f"Solar {site}", cutoff=cutoff)
        table[site] = row
        solar461._RATIO[site] = float(row["ratio"])  # noqa: SLF001
    fit430.solar_base.cache_clear()  # these two caches hold values in the previous refit's units
    fit430.farm_ceiling.cache_clear()
    path = common.DATA_DIR / CAPACITY_FILE
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored[f"{start:%Y-%m}"] = table
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=1) + "\n")


def main() -> None:
    """Run the base arm with each half-life label, seed, and refit one after another."""
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("labels", nargs="+", help="hl05, hl1, hl2, hl3, none, or ind49")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--refits", default=",".join(f"{r:%Y-%m}" for r in fit430.REFITS))
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    refits = [r for r in fit430.REFITS if f"{r:%Y-%m}" in args.refits.split(",")]
    solar461.feats.load_band_farm = solar461.load_band_farm
    common430.PRED_DIR = PRED_DIR
    fit430.train_network = train_network
    fit430._columns = columns_with_cycle  # noqa: SLF001
    fit430.FLAGS = (*fit430.FLAGS, "cycle49")  # a 0/1 input is not standardised
    base = common430.ARMS[args.base]
    for label in args.labels:
        half_life = common.SOLAR_HALF_LIVES_YEARS.get(label)
        config = IndicatorConfig(**vars(base)) if label == "ind49" else base
        arm = f"{args.base}_{label}"
        for seed in seeds:
            for start in refits:
                set_ratios(start=start)
                _JOB.update(half_life=half_life, cutoff=start - timedelta(days=f310.REFIT_GAP_DAYS))
                began = time.time()
                print(fit430.run_job(arm=arm, config=config, seed=seed, start=start), flush=True)
                print(f"  total {time.time() - began:.0f} s", flush=True)


if __name__ == "__main__":
    main()
