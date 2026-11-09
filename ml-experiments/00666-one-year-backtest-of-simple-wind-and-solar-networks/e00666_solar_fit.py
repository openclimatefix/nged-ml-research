# ruff: noqa: E402
"""Fit solar network PLP quarterly and forecast every daily origin of the following three months.

Copied from experiment 659's `e00659_solar_fit.py` (arm `PLP`), reading this experiment's
`solar_inputs` (`e00666_solar_inputs.py`) and training on leads from 12 h to under 348 h from the
00 UTC run. The 18 inputs are the ensemble mean and standard deviation of shortwave, air
temperature, and 10 m wind speed, the ensemble means of relative humidity, longwave radiation,
precipitation, and surface pressure, the mean and standard deviation of the fixed physics-lite power
and of the Faiman cell temperature, the sun's elevation and the sine and cosine of its azimuth, and
the lead in hours. The mixture's centre is the mean physics-lite power. Two hidden layers of 64
units, a three-component Gaussian mixture, 20 epochs, no pretraining, and no farm column.

The refits are on 2025-11-01, 2026-02-01, 2026-05-01, and 2026-08-01, 3 seeds each; each refit
predicts the origins of the next three months. A refit at `R` trains on daylight half-hours whose
valid time is before `R - 1 day` and whose run is before `R`, and stops early on the last 61 days
of those rows. Capacity is the 99.9th percentile of the farm's cleaned power over the 730 days
before `R`; the target is the observed power over the farm's stored effective capacity divided by
`new / old`. Quantiles are written in MW as the fraction of the new capacity times the new
capacity. Night half-hours (no daylight) are not rows here; `e00666_export.py` sets them to zero.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00666-*/e00666_solar_fit.py [--seeds 0 1 2]
"""

import argparse
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00666_common as common
import e00666_network as network
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.pv_dataset import pv_sites

EPOCHS: Final[int] = 20
VALIDATION_DAYS: Final[int] = 61
REFIT_GAP_DAYS: Final[int] = 1
REFIT_MONTHS: Final[int] = 3
MAX_LEAD_HOURS: Final[float] = 348.0
COLUMNS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_sd",
    "temp_mean",
    "temp_sd",
    "wind_mean",
    "wind_sd",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "pl_mean",
    "pl_sd",
    "cell_mean",
    "cell_sd",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "lead_hours",
)
REFITS: Final[tuple[datetime, ...]] = (
    datetime(2025, 11, 1, tzinfo=UTC),
    datetime(2026, 2, 1, tzinfo=UTC),
    datetime(2026, 5, 1, tzinfo=UTC),
    datetime(2026, 8, 1, tzinfo=UTC),
)


def month_add(*, start: datetime, months: int) -> datetime:
    """Return the first day of the month `months` after `start`."""
    index = start.year * 12 + start.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def load_rows() -> pl.DataFrame:
    """Return every farm's daylight half-hours that have all 18 inputs, with lead below 348 h."""
    frame = pl.concat(
        [pl.read_parquet(common.SOLAR_DIR / f"{s}.parquet") for s in pv_sites()["site"].to_list()]
    ).filter(pl.col("daylight") & (pl.col("lead_hours") < MAX_LEAD_HOURS))
    missing = frame.select(pl.col(list(COLUMNS)).null_count()).row(0)
    if any(missing):
        msg = f"daylight rows with missing inputs: {missing}"
        raise ValueError(msg)
    return frame.sort("series", "init_time", "time")


def fit_and_predict(
    *, train: pl.DataFrame, target: pl.DataFrame, refit: datetime, seed: int
) -> np.ndarray:
    """Train one network, save its state, and return `target`'s quantiles as capacity fractions."""
    cutoff = refit - timedelta(days=REFIT_GAP_DAYS)
    is_stop = (train["time"] >= cutoff - timedelta(days=VALIDATION_DAYS)).to_numpy()
    is_fit = ~is_stop
    x = train.select(list(COLUMNS)).to_numpy().astype(np.float64)
    mean, std = x[is_fit].mean(axis=0), np.maximum(x[is_fit].std(axis=0), 1e-6)
    y = train["scaled"].to_numpy()
    model, state = network.fit_network(
        x=x,
        mu=train["pl_mean"].to_numpy().astype(np.float64),
        y=y,
        is_fit=is_fit,
        is_stop=is_stop,
        mean=mean,
        std=std,
        epochs=EPOCHS,
        seed=seed,
    )
    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = common.MODEL_DIR / f"solar_PLP_{refit:%Y-%m-%d}_seed{seed}.tmp"
    torch.save({"state": state, "mean": mean.tolist(), "std": std.tolist()}, tmp)
    tmp.rename(tmp.with_suffix(".pt"))
    return network.predict_quantiles(
        model=model,
        x=target.select(list(COLUMNS)).to_numpy().astype(np.float64),
        mu=target["pl_mean"].to_numpy().astype(np.float64),
        mean=mean,
        std=std,
    )


def main() -> None:
    """Fit every refit and seed, skipping finished parts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    out_dir = common.FIT_DIR / "solar"
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = pv_sites()
    frame = load_rows()
    started = time.time()
    for refit in REFITS:
        end = month_add(start=refit, months=REFIT_MONTHS)
        cutoff = refit - timedelta(days=REFIT_GAP_DAYS)
        cap = {
            row["site"]: common.capacity_before(
                time_series_id=row["time_series_id"],
                refit=refit,
                old_mw=row["effective_capacity_mw"],
            )
            for row in sites.iter_rows(named=True)
        }
        ratio = pl.col("series").replace_strict(
            {f: v["ratio"] for f, v in cap.items()}, return_dtype=pl.Float64
        )
        train = frame.filter(
            (pl.col("time") < cutoff) & (pl.col("init_time") < refit) & pl.col("actual").is_finite()
        ).with_columns(scaled=pl.col("actual").cast(pl.Float64) / ratio)
        assert train.select(pl.col("time").max()).item() < refit, (
            "a training row is valid after the refit"
        )
        target = frame.filter((pl.col("init_time") >= refit) & (pl.col("init_time") < end))
        for seed in args.seeds:
            path = out_dir / f"refit={refit:%Y-%m-%d}_seed={seed}.parquet"
            if path.exists():
                continue
            fraction = fit_and_predict(train=train, target=target, refit=refit, seed=seed)
            new_mw = target["series"].replace_strict(
                {f: v["new_mw"] for f, v in cap.items()}, return_dtype=pl.Float64
            )
            table = target.select("series", "time_series_id", "init_time", "time").with_columns(
                refit=pl.lit(refit),
                seed=pl.lit(seed),
                new_mw=new_mw,
                **{
                    q: pl.Series(fraction[:, i] * new_mw.to_numpy())
                    for i, q in enumerate(common.QUANTILES)
                },
            )
            table.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(
                f"solar PLP refit {refit:%Y-%m-%d} seed {seed}: {table.height:,} rows, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )
            if args.first_only:
                return


if __name__ == "__main__":
    main()
