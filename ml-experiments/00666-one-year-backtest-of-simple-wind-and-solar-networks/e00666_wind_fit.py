# ruff: noqa: E402
"""Fit wind network S monthly and forecast every daily origin of the following month.

Copied from experiment 659's `e00659_wind_fit.py` (arm `S`). One pooled network per (refit, seed):
two hidden layers of 64 units, a three-component Gaussian mixture, 11 quantiles read from the
mixture and capped at capacity. No pretraining and equal row weights. The inputs are the mean and
standard deviation across the 51 members of the 10 m and 100 m wind speed, the sine and cosine of
the 100 m direction, and the air temperature (10 columns, from `e00666_wind_features.py`), the
sine, cosine, and hundredths of the lead, and a one-hot farm. The mixture's centre is the mean of
the training target.

The refits are on day 0 of each month from 2025-11-01 to 2026-10-01 (12 refits, 3 seeds each). A
refit at `R` trains on cases whose run stamp (the 00 UTC run's availability, 09:00) and valid time
are before `R - 62 days`, stops early on the cases from that date whose valid time is before `R`,
and predicts the runs of the month from `R`. The 62-day window and 40 epochs are those of
experiment 659. No case with a valid time after `R` is read for fitting or stopping.

Capacity is the 99.9th percentile of the farm's cleaned power over the 730 days before `R`. The
target is the observed power over the farm's stored effective capacity (the unit of experiment
659) divided by `new / old`, with the idle-mask rows replaced by zero ("cleaned zeros"). The idle
mask of experiment 210 ends on 2026-09-13, so later rows are unmasked: they are used as observed.
The lead feature is the hours from the run stamp (09:00 UTC) to the valid time, as in experiment
659. Quantiles are written in MW as the fraction of the new capacity times the new capacity.
Each daily origin is 12:00 UTC on the run's day, so the forecast rows are the 672 half-hours from
12:00 on the run's day.

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00666-*/e00666_wind_fit.py [--seeds 0 1 2]
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
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EPOCHS: Final[int] = 40
STOP_WINDOW_DAYS: Final[int] = 62
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
MASK_PATH: Final = ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact" / "mask.parquet"
"""Experiment 210's idle-mask rows (`time_series_id`, `valid_time`), kept private."""
WEATHER: Final[tuple[str, ...]] = (
    "speed10_mean",
    "speed10_sd",
    "speed100_mean",
    "speed100_sd",
    "dir_sin_mean",
    "dir_sin_sd",
    "dir_cos_mean",
    "dir_cos_sd",
    "temp_mean",
    "temp_sd",
)
REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(2025 + (10 + i) // 12, (10 + i) % 12 + 1, 1, tzinfo=UTC) for i in range(12)
)
"""Day 0 of each month from 2025-11-01 to 2026-10-01."""


def next_month(*, day: datetime) -> datetime:
    """Return the first day of the month after `day`."""
    return datetime(day.year + (day.month == 12), day.month % 12 + 1, 1, tzinfo=UTC)


def load_cases(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return every case with its 12 summaries, lead, and the observed target (null if absent)."""
    features = pl.concat(
        [pl.read_parquet(common.FEATURE_DIR / f"{s}.parquet") for s in sites["site"].to_list()]
    ).join(sites.select("time_series_id", "site", "effective_capacity_mw"), on="time_series_id")
    names = [*WEATHER, "pl_mean", "pl_sd"]
    missing = sum(features[n].null_count() + features[n].is_nan().sum() for n in names)
    if missing:
        msg = f"{missing} cases have no member summaries"
        raise ValueError(msg)
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", valid_time="time", power="power")
        .collect()
    )
    mask = pl.read_parquet(MASK_PATH).select("time_series_id", "valid_time")
    mask = mask.with_columns(masked=pl.lit(value=True))
    return (
        features.join(power, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            masked=pl.col("masked").fill_null(value=False),
            actual_norm=pl.col("power") / pl.col("effective_capacity_mw"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
        .with_columns(
            actual_clean=pl.when(pl.col("masked")).then(0.0).otherwise(pl.col("actual_norm"))
        )
        .sort("site", *KEYS)
    )


def design(*, frame: pl.DataFrame, sites: list[str]) -> np.ndarray:
    """Return the network inputs: weather summaries, lead terms, and a one-hot farm."""
    hours = frame["lead_hours"].to_numpy().astype(np.float64)
    angle = 2 * np.pi * hours / 24.0
    return np.concatenate(
        [
            frame.select(list(WEATHER)).to_numpy().astype(np.float64),
            np.stack([np.sin(angle), np.cos(angle), hours / 100.0], axis=1),
            np.stack([(frame["site"] == f).to_numpy().astype(np.float64) for f in sites], axis=1),
        ],
        axis=1,
    )


def fit_and_predict(
    *, train: pl.DataFrame, target: pl.DataFrame, sites: list[str], refit: datetime, seed: int
) -> np.ndarray:
    """Train one network, save its state, and return `target`'s quantiles as capacity fractions."""
    cutoff = refit - timedelta(days=STOP_WINDOW_DAYS)
    is_fit = ((train["power_fcst_init_time"] < cutoff) & (train["valid_time"] < cutoff)).to_numpy()
    is_stop = ((train["power_fcst_init_time"] >= cutoff) & (train["valid_time"] < refit)).to_numpy()
    x = design(frame=train, sites=sites)
    mean, std = x[is_fit].mean(axis=0), np.maximum(x[is_fit].std(axis=0), 1e-6)
    y = train["target"].to_numpy()
    constant = float(y[is_fit].mean())
    model, state = network.fit_network(
        x=x,
        mu=np.full(train.height, constant),
        y=y,
        is_fit=is_fit,
        is_stop=is_stop,
        mean=mean,
        std=std,
        epochs=EPOCHS,
        seed=seed,
    )
    common.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = common.MODEL_DIR / f"wind_S_{refit:%Y-%m-%d}_seed{seed}.tmp"
    torch.save(
        {"state": state, "mean": mean.tolist(), "std": std.tolist(), "constant": constant}, tmp
    )
    tmp.rename(tmp.with_suffix(".pt"))
    return network.predict_quantiles(
        model=model,
        x=design(frame=target, sites=sites),
        mu=np.full(target.height, constant),
        mean=mean,
        std=std,
    )


def main() -> None:
    """Fit every refit and seed, skipping finished parts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--first-only", action="store_true")
    args = parser.parse_args()
    out_dir = common.FIT_DIR / "wind"
    out_dir.mkdir(parents=True, exist_ok=True)
    sites_frame = wind_sites().sort("site")
    sites = sites_frame["site"].to_list()
    frame = load_cases(sites=sites_frame)
    started = time.time()
    for refit in REFITS:
        upper = next_month(day=refit)
        cap = {
            row["site"]: common.capacity_before(
                time_series_id=row["time_series_id"],
                refit=refit,
                old_mw=row["effective_capacity_mw"],
            )
            for row in sites_frame.iter_rows(named=True)
        }
        scale = pl.col("site").replace_strict(
            {f: cap[f]["ratio"] for f in sites}, return_dtype=pl.Float64
        )
        train = frame.filter(
            (pl.col("power_fcst_init_time") < refit)
            & (pl.col("valid_time") < refit)
            & pl.col("actual_clean").is_finite()
        ).with_columns(target=pl.col("actual_clean") / scale)
        assert train.select(pl.col("valid_time").max()).item() < refit, (
            "a training case is valid after the refit"
        )
        # The 672 half-hours from 12:00 on the run's day; the run stamp is 09:00 on that day.
        first_valid = pl.col("power_fcst_init_time") + timedelta(
            hours=common.RUN_TO_ORIGIN_HOURS - common.RUN_TO_PUBLICATION_HOURS
        )
        target = frame.filter(
            (pl.col("power_fcst_init_time") >= refit)
            & (pl.col("power_fcst_init_time") < upper)
            & (pl.col("valid_time") >= first_valid)
            & (pl.col("valid_time") < first_valid + timedelta(days=14))
        )
        for seed in args.seeds:
            path = out_dir / f"refit={refit:%Y-%m-%d}_seed={seed}.parquet"
            if path.exists():
                continue
            fraction = fit_and_predict(
                train=train, target=target, sites=sites, refit=refit, seed=seed
            )
            new_mw = target["site"].replace_strict(
                {f: cap[f]["new_mw"] for f in sites}, return_dtype=pl.Float64
            )
            table = target.select(*KEYS).with_columns(
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
                f"wind S refit {refit:%Y-%m-%d} seed {seed}: {table.height:,} rows, "
                f"{time.time() - started:.0f} s",
                flush=True,
            )
            if args.first_only:
                return


if __name__ == "__main__":
    main()
