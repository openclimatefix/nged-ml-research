r"""Refit experiment 87's XGBoost member monthly with median day and week lags (465, arm c).

The recipe is experiment 392's `e00392_fit.py` (as copied by experiment 402's `e00402_refit.py`):
experiment 87's `n1h` configuration, monthly refits, 500 rounds, `eta` 0.05, 11 quantiles, the
control member for training, the 51 members pooled as a mixture at scoring, plain targets (arm
`r0`), `colsample` 1.0. Only the three lag columns differ:

- `lag_origin`: the newest usable reading, as before.
- `lag_day_med7`: the median of the 7 readings at the valid time's time of day on the 7 most recent
  days that end by the origin's newest usable reading.
- `lag_week_med3`: the median of the 3 readings at the valid time's time of day and day of week on
  the 3 most recent weeks that end by the newest usable reading.

A reading that is missing (NaN: unobserved, or a flagged `substation_zero` half-hour) is skipped,
and the median of the remaining readings is used; a median over no readings is NaN. The lags use
only readings no younger than the newest usable one, so the leakage rule of
`e00087_common.origin_lags` carries over and is asserted again here.

Fits write `work/arm_preds/c_robust/<site>_s<seed>_<refit>.parquet` and save each booster under
`work/boosters/c_robust/`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1 DATA_PATH_INTERNAL=... \\
    systemd-run --user --scope -p MemoryMax=10G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00465-*/e00465_fit.py \\
    --workers 6 --cores 6 --device cpu
```
"""

import argparse
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00465-median-day-and-week-lags-for-load-shift-robustness" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
ORIGINAL_LAGS: Final[bool] = os.environ.get("E465_ORIGINAL_LAGS") == "1"
"""Set `E465_ORIGINAL_LAGS=1` to refit experiment 402's arm (a) recipe with this script's CPU
settings, the noise-floor control: the three original lags, written under `a_cpu_control`."""
ARM_NAME: Final[str] = "a_cpu_control" if ORIGINAL_LAGS else "c_robust"
ROBUST_COLUMNS: Final[tuple[str, ...]] = ("lag_origin", "lag_day_med7", "lag_week_med3")
DAY_COUNT: Final[int] = 7
WEEK_COUNT: Final[int] = 3
REFIT_DATES: Final[tuple[str, ...]] = ("2025-10-01", "2026-01-01", "2026-03-01", "2026-06-01")
SMOKE_ROUNDS: Final[int] = 20
WEEK_STEPS: Final[int] = 7 * common.STEPS_PER_DAY


def median_of_recent(
    *, power: np.ndarray, latest: np.ndarray, period: int, count: int
) -> np.ndarray:
    """Return the median of `count` readings spaced `period` steps apart, ending at `latest`.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        latest: Grid index of each case's most recent reading in the series of analogues.
        period: Steps between analogues (a day or a week).
        count: Number of analogues.

    Returns:
        A float32 array of shape (cases,): the median of the analogues that exist, NaN if none.
    """
    stack = np.stack(
        [common.take(values=power, index=latest - k * period) for k in range(count)], axis=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=1).astype(np.float32)


def robust_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return `ROBUST_COLUMNS` for each case: the origin lag and the day and week medians.

    Raises:
        ValueError: If a case's valid time is not after its origin, or an analogue is younger than
            the newest usable observation.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    day_latest = target - common.STEPS_PER_DAY * np.ceil(gap / common.STEPS_PER_DAY).astype(int)
    week_latest = target - WEEK_STEPS * np.ceil(gap / WEEK_STEPS).astype(int)
    if (day_latest > newest).any() or (week_latest > newest).any():
        msg = "a lag index is younger than the newest usable observation"
        raise ValueError(msg)
    return np.stack(
        [
            common.take(values=power, index=newest),
            median_of_recent(
                power=power, latest=day_latest, period=common.STEPS_PER_DAY, count=DAY_COUNT
            ),
            median_of_recent(power=power, latest=week_latest, period=WEEK_STEPS, count=WEEK_COUNT),
        ],
        axis=1,
    )


def with_robust_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Replace the three original lag columns of `frame` with the robust lags."""
    if ORIGINAL_LAGS:
        return frame
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    robust = pl.DataFrame(
        robust_lags(power=power, target=target, origin=origin),
        schema=list(ROBUST_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
    return frame.drop(common.LAG_COLUMNS).with_columns(robust)


def columns() -> list[str]:
    """Return the input columns in matrix order, with the robust lags in place of the originals."""
    if ORIGINAL_LAGS:
        return fd.input_columns(a7=False)
    return [
        *common.feature_columns(),
        *common.N1_EXTRA_COLUMNS,
        *ROBUST_COLUMNS,
    ]


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        known.select(columns()).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
    )


def forecast_run(*, model: xgb.Booster, prepared: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return one run's 11 quantiles per valid time: the mixture of the 51 members."""
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    matrix = prepared.select(columns()).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix))
    return fd.mixture_frame(
        keys=keys, raw=raw, members=prepared["ensemble_member"].n_unique(), seed=seed
    )


_ROBUST_ROWS: dict[str, pl.DataFrame] = {}


def robust_training_rows(*, row: dict, arrays: common.SeriesArrays) -> pl.DataFrame:
    """Return a series' training rows with robust lags, built once per worker process."""
    if row["site"] not in _ROBUST_ROWS:
        rows = fd._training_rows_cached(row=row, arrays=arrays, config=fd.CONFIGS["n1h"])
        _ROBUST_ROWS[row["site"]] = with_robust_lags(frame=rows, power=arrays.power)
    return _ROBUST_ROWS[row["site"]]


def run_job(
    *,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
    device: str,
) -> list[str]:
    """Fit arm (c) at one refit for each site lacking its file, and forecast the refit's runs."""
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not fd.job_path(root=out_dir, config=ARM_NAME, site=s, seed=seed, refit=refit).exists()
    ]
    lines = []
    models: dict[str, xgb.Booster] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        build_started = time.monotonic()
        rows = robust_training_rows(row=row, arrays=arrays[site])
        build_seconds = time.monotonic() - build_started
        known = fd.known_rows(rows=rows, refit=refit, config=config)
        models[site] = xgb.train(
            {**fd.parameters(config=config, seed=seed, nthread=nthread), "device": device},
            train_matrix(known=known, row=row, refit=refit),
            num_boost_round=rounds,
        )
        path = (
            (BOOSTER_DIR if out_dir == ARM_DIR else out_dir.parent / "boosters_smoke")
            / ARM_NAME
            / f"{site}_s{seed}_{refit:%Y%m%d}.ubj"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        models[site].save_model(path)
        lines.append(
            f"{ARM_NAME} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
            f"feature build {build_seconds:.0f} s, fit total {time.monotonic() - started:.0f} s"
        )
    pieces: dict[str, list[pl.DataFrame]] = {s: [] for s in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site in todo:
            row = rows_of[site]
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            )
            prepared = with_robust_lags(frame=prepared, power=arrays[site].power)
            pieces[site].append(forecast_run(model=models[site], prepared=prepared, seed=seed))
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=ARM_NAME, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit arm (c) at the four monthly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--refit-dates", nargs="+", default=list(REFIT_DATES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument(
        "--chunks", type=int, default=1, help="split each refit's sites into this many jobs"
    )
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {r: runs for r, runs in jobs.items() if f"{r:%Y-%m-%d}" in args.refit_dates}
    rounds, out = common.N_ROUNDS, ARM_DIR
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke"
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                refit=r,
                origins=runs,
                sites=chunk,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
                device=args.device,
            )
            for r, runs in jobs.items()
            for chunk in (sites[i :: args.chunks] for i in range(args.chunks))
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
