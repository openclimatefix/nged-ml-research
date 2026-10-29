"""Stage 1 of experiment 391: a lagless quantile model, refitted monthly, and its surprise table.

For each Primary the script refits experiment 87's `n1h` XGBoost recipe without the three origin
lags at each monthly refit from `FIRST_REFIT`, with no label filter on the training rows, and
`colsample_*` at 1.0. Each refit predicts the 11 quantiles for the half-hours from the refit to the
next refit, which none of its training rows contain (valid times up to the refit minus 3.5 hours).
Two versions are built:

- `ens`: the weather of the control member of the ENS run with the latest forecast origin before
  the half-hour (lead 0.5 to 24 hours), with `lead_hours` as an input, as in the baseline.
- `era5`: ERA5 weather of the half-hour itself (a hindsight ceiling; no `lead_hours` input).

Each version writes `work/stage1/<version>/<site>.parquet` with the dense-grid half-hours that
have a prediction: `valid_time`, `u` (the normal score of the reading's probability integral
transform in the 11 quantiles, clipped to 0.005 to 0.995), `r_mw` (reading minus median, in MW),
`median_mw`, and `refit` (the refit whose model predicted the half-hour).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00391-*/e00391_stage1.py \
    --versions ens era5 --workers 3 --cores 3
```
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import norm
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00391-xgboost-lag-surprise-second-stage" / "work"
STAGE1_DIR: Final[Path] = WORK_DIR / "stage1"
FIRST_REFIT: Final[datetime] = datetime(2024, 10, 1, tzinfo=UTC)
LAST_REFIT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The first half-hour that no stage 1 refit predicts."""
REFIT_CUTOFF_HOURS: Final[float] = 3.5
PIT_CLIP: Final[tuple[float, float]] = (0.005, 0.995)
SMOKE_ROUNDS: Final[int] = 20
VERSIONS: Final[tuple[str, ...]] = ("ens", "era5")
HALF_HOURS_PER_MONTH_FLOOR: Final[int] = 1_000


def monthly_refits() -> list[datetime]:
    """Return the first-of-month refit times from `FIRST_REFIT` to `LAST_REFIT`."""
    out, year, month = [], FIRST_REFIT.year, FIRST_REFIT.month
    while (refit := datetime(year, month, 1, tzinfo=UTC)) <= LAST_REFIT:
        out.append(refit)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def stage1_columns(*, version: str) -> list[str]:
    """Return the input columns of a version: the baseline's without the lags, no lead for ERA5."""
    columns = [c for c in fd.input_columns(a7=False) if c not in common.LAG_COLUMNS]
    return columns if version == "ens" else [c for c in columns if c != "lead_hours"]


def pit_normal_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the normal score of each reading's probability integral transform.

    The transform interpolates linearly between the 11 quantile levels, is clipped to
    `PIT_CLIP` outside the first and last level, and is then passed through the normal quantile
    function. Equal neighbouring quantiles resolve to the lower level.

    Args:
        quantiles: Shape (cases, 11), sorted along the levels.
        actual: Shape (cases,).

    Returns:
        Shape (cases,), NaN where the reading or a quantile is not finite.
    """
    levels = common.LEVELS
    below = (quantiles <= actual[:, None]).sum(axis=1)
    low = np.clip(below - 1, 0, len(levels) - 2)
    rows = np.arange(len(actual))
    q_low, q_high = quantiles[rows, low], quantiles[rows, low + 1]
    span = q_high - q_low
    fraction = np.where(span > 0, (actual - q_low) / np.where(span > 0, span, 1.0), 0.0)
    pit = levels[low] + np.clip(fraction, 0.0, 1.0) * (levels[low + 1] - levels[low])
    pit = np.where(below == 0, 0.0, np.where(below == len(levels), 1.0, pit))
    score = norm.ppf(np.clip(pit, *PIT_CLIP))
    bad = ~np.isfinite(actual) | ~np.isfinite(quantiles).all(axis=1)
    return np.where(bad, np.nan, score)


def ens_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return the control-member rows of the series with `valid_time`, inputs, and `power` (MW)."""
    return fd.training_rows(row=row, arrays=arrays, frozen=False, a7=False)


def era5_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return the dense-grid ERA5 rows of the series with the same columns as `ens_rows`."""
    columns = common.feature_columns()
    frame = pl.DataFrame(arrays.features, schema=columns).with_columns(
        valid_time=arrays.valid_time,
        power=pl.Series(arrays.power * row["effective_capacity_mw"]).cast(pl.Float64),
    )
    frame = frame.with_columns(power_fcst_init_time=pl.col("valid_time"))
    return frame.filter(pl.col("power").is_finite() & (pl.col("valid_time") >= common.GRID_START))


def latest_origin_rows(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Keep, per valid time, the row of the run with the latest forecast origin before it."""
    return rows.sort("valid_time", "power_fcst_init_time").unique(
        subset="valid_time", keep="last", maintain_order=True
    )


def fit_predict(
    *,
    known: pl.DataFrame,
    target: pl.DataFrame,
    row: dict,
    refit: datetime,
    version: str,
    nthread: int,
    rounds: int,
    seed: int,
) -> pl.DataFrame:
    """Fit one refit on `known` and return the quantiles for the `target` rows."""
    columns = stage1_columns(version=version)
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    capacity = row["effective_capacity_mw"]
    train = xgb.DMatrix(
        known.select(columns).to_numpy().astype(np.float32),
        label=(known["power"] / capacity).to_numpy(),
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
    )
    model = xgb.train(
        fd.parameters(config=fd.CONFIGS["n1h"], seed=seed, nthread=nthread),
        train,
        num_boost_round=rounds,
    )
    raw = model.predict(xgb.DMatrix(target.select(columns).to_numpy().astype(np.float32)))
    quantiles = np.sort(raw.reshape(-1, len(common.LEVELS)), axis=1)
    actual = (target["power"] / capacity).to_numpy()
    median = quantiles[:, common.MEDIAN]
    return target.select("valid_time").with_columns(
        u=pl.Series(pit_normal_score(quantiles=quantiles, actual=actual)),
        r_mw=pl.Series((actual - median) * capacity),
        median_mw=pl.Series(median * capacity),
        refit=pl.lit(refit),
    )


def run_job(*, version: str, site: str, nthread: int, rounds: int, seed: int, smoke: bool) -> str:
    """Build one series' stage 1 table for one version, writing it atomically."""
    started = time.monotonic()
    out_dir = (
        STAGE1_DIR / ("smoke_" if smoke else "") / version
        if False
        else (STAGE1_DIR / (f"smoke_{version}" if smoke else version))
    )
    target_path = out_dir / f"{site}.parquet"
    if target_path.exists():
        return f"{version} {site}: exists"
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    rows = (
        ens_rows(row=row, arrays=arrays[site])
        if version == "ens"
        else era5_rows(row=row, arrays=arrays[site])
    )
    refits = monthly_refits()[: 2 if smoke else None]
    pieces = []
    for refit, following in zip(refits, [*refits[1:], END], strict=True):
        cutoff = refit.timestamp() - REFIT_CUTOFF_HOURS * 3600
        valid_seconds = pl.col("valid_time").dt.epoch("s")
        known = rows.filter((valid_seconds <= cutoff) & (pl.col("power_fcst_init_time") <= refit))
        window = rows.filter((pl.col("valid_time") >= refit) & (pl.col("valid_time") < following))
        target = latest_origin_rows(rows=window)
        if known.height < HALF_HOURS_PER_MONTH_FLOOR or target.is_empty():
            msg = f"{site} {version} {refit}: {known.height} known rows, {target.height} targets"
            raise ValueError(msg)
        if (known["valid_time"].max() > refit) or (target["valid_time"].min() < refit):
            msg = "a stage 1 fit would see a row at or after its refit"
            raise ValueError(msg)
        pieces.append(
            fit_predict(
                known=known,
                target=target,
                row=row,
                refit=refit,
                version=version,
                nthread=nthread,
                rounds=rounds,
                seed=seed,
            )
        )
    common.atomic_write(frame=pl.concat(pieces), target=target_path)
    return f"{version} {site}: {sum(p.height for p in pieces):,} half-hours, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Build the stage 1 tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=list(VERSIONS), choices=VERSIONS)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 20 rounds")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    nthread = max(1, args.cores // args.workers)
    rounds = SMOKE_ROUNDS if args.smoke else common.N_ROUNDS
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                version=v,
                site=s,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                smoke=args.smoke,
            )
            for v in args.versions
            for s in sites
        ]
        for done, future in enumerate(futures, start=1):
            print(f"[{done}/{len(futures)}] {future.result()}", flush=True)


if __name__ == "__main__":
    main()
