"""Refit experiment 87's weekly XGBoost demand member with extra lagged-power columns (388).

Each arm refits the `n1h` configuration of experiment 87 (the BG2 weekly member) at the weekly
refit times (or the monthly refit times with `--regime monthly`), for the chosen series, with
experiment 87's training rows, parameters, and leakage checks (its `e00087_fit_demand` module).
It writes the 11 quantiles per daily run to
`work/arm_preds/<regime>/<arm>/<site>_s<seed>_<refit>.parquet`. `colsample_bytree` and
`colsample_bynode` stay at 1.0 in every arm.

A lagged-power column is usable only when its meter reading is at least 3.5 hours older than the
forecast origin (`e00087_common.NEWEST_AGE`); otherwise it is null. Arms:

- `base`: the three origin lags only (the reproduction gate against experiment 379's `only_W`).
- `a`: adds power at the valid time minus 1, 2, 3, 4, 5, and 6 weeks (6 columns).
- `b`: adds the mean and median of the usable weekly analogues (experiment 87's `a7` pair).
- `c`: adds the origin anomaly and the day-ago anomaly (2 columns). Each is a reading minus the
  median reading at the same half-hour and weekday over the 4 weeks before it. The 4-week window is
  a choice, not a tested value.
- `d`: the columns of `b` and `c` together.
- `e`: the baseline without `lag_origin` (the newest usable reading).
- `f`: `e` with the day lag replaced by the power exactly 24 hours before the valid time, null
  where that reading is not yet usable.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00388-*/e00388_fit.py \
    --arms a --workers 3 --cores 3
```
"""

import argparse
import multiprocessing
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
    ML_EXPERIMENTS_DATA_DIR / "00388-xgboost-lagged-power-column-sets-weekly-demand" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
WEEK_STEPS: Final[int] = fd.WEEK_STEPS
DAY_STEPS: Final[int] = common.STEPS_PER_DAY
ANALOGUE_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
ANOMALY_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4)
ANALOGUE_COLUMNS: Final[tuple[str, ...]] = tuple(f"analogue_{w}w" for w in ANALOGUE_WEEKS)
ANOMALY_COLUMNS: Final[tuple[str, ...]] = ("origin_anomaly", "day_ago_anomaly")
EXACT_24H_COLUMN: Final[str] = "lag_exact_24h"
DAY_LAG: Final[str] = "lag_same_time_latest_day"
ORIGIN_LAG: Final[str] = "lag_origin"
SMOKE_ROUNDS: Final[int] = 20
ARMS: Final[dict[str, tuple[str, ...]]] = {
    "base": (),
    "a": ANALOGUE_COLUMNS,
    "b": fd.A7_COLUMNS,
    "c": ANOMALY_COLUMNS,
    "d": (*fd.A7_COLUMNS, *ANOMALY_COLUMNS),
    "e": (),
    "f": (EXACT_24H_COLUMN,),
}
REMOVED: Final[dict[str, tuple[str, ...]]] = {
    "e": (ORIGIN_LAG,),
    "f": (ORIGIN_LAG, DAY_LAG),
}
"""Baseline columns an arm drops: E has no newest reading; F also swaps the day lag for the exact
24-hour lag."""
"""Each arm's added columns, in matrix order."""


def _anomaly(*, power: np.ndarray, reading: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Return `power[reading]` minus the median power 1 to 4 weeks before it, NaN if unusable.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        reading: Grid index of the reading each case subtracts from.
        usable: Whether the reading is no younger than the newest usable observation.

    Returns:
        A float32 array, NaN where the reading is unusable or unobserved or all four earlier
        readings are unobserved.
    """
    earlier = np.stack(
        [common.take(values=power, index=reading - w * WEEK_STEPS) for w in ANOMALY_WEEKS], axis=1
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(earlier, axis=1)
    out = common.take(values=power, index=reading) - median
    return np.where(usable, out, np.nan).astype(np.float32)


def lag_columns(
    *, power: np.ndarray, target: np.ndarray, origin: np.ndarray, names: tuple[str, ...]
) -> np.ndarray:
    """Return the requested added lag columns, using only observations the origin could see.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.
        names: The columns to build, a subset of the arms' columns.

    Returns:
        A float32 array of shape (cases, len(names)).

    Raises:
        ValueError: If a valid time is not after its origin, or a column reads a reading younger
            than the newest usable observation.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - common.NEWEST_AGE
    built: dict[str, np.ndarray] = {}
    if any(n in ANALOGUE_COLUMNS for n in names):
        for weeks, name in zip(ANALOGUE_WEEKS, ANALOGUE_COLUMNS, strict=True):
            index = target - weeks * WEEK_STEPS
            built[name] = np.where(
                index <= newest, common.take(values=power, index=index), np.nan
            ).astype(np.float32)
    if any(n in fd.A7_COLUMNS for n in names):
        summary = fd.a7_lags(power=power, target=target, origin=origin)
        built.update({n: summary[:, i] for i, n in enumerate(fd.A7_COLUMNS)})
    if EXACT_24H_COLUMN in names:
        exact = target - DAY_STEPS
        built[EXACT_24H_COLUMN] = np.where(
            exact <= newest, common.take(values=power, index=exact), np.nan
        ).astype(np.float32)
    if "origin_anomaly" in names:
        built["origin_anomaly"] = _anomaly(power=power, reading=newest, usable=newest >= 0)
    if "day_ago_anomaly" in names:
        day_ago = target - DAY_STEPS
        built["day_ago_anomaly"] = _anomaly(power=power, reading=day_ago, usable=day_ago <= newest)
    return np.stack([built[n] for n in names], axis=1).astype(np.float32)


def add_lag_columns(*, frame: pl.DataFrame, power: np.ndarray, arm: str) -> pl.DataFrame:
    """Add the arm's lag columns to a frame with `valid_time` and `power_fcst_init_time`."""
    names = ARMS[arm]
    if not names:
        return frame
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    added = lag_columns(power=power, target=target, origin=origin, names=names)
    return frame.with_columns(pl.DataFrame(added, schema=list(names), orient="row"))


def columns_of(*, arm: str) -> list[str]:
    """Return all input columns of an arm in matrix order."""
    kept = [c for c in fd.input_columns(a7=False) if c not in REMOVED.get(arm, ())]
    return [*kept, *ARMS[arm]]


def run_job(
    *,
    arm: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
) -> list[str]:
    """Fit one arm at one refit for each site lacking its file, and forecast the refit's runs."""
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not fd.job_path(root=out_dir, config=arm, site=s, seed=seed, refit=refit).exists()
    ]
    lines = []
    models: dict[str, xgb.Booster] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = fd.known_rows(
            rows=add_lag_columns(
                frame=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
                power=arrays[site].power,
                arm=arm,
            ),
            refit=refit,
            config=config,
        )
        age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
        matrix = xgb.DMatrix(
            known.select(columns_of(arm=arm)).to_numpy().astype(np.float32),
            label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=0.5 ** (age_days / config.half_life_days),
        )
        models[site] = xgb.train(
            fd.parameters(config=config, seed=seed, nthread=nthread),
            matrix,
            num_boost_round=rounds,
        )
        lines.append(
            f"{arm} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
            f"fit {time.monotonic() - started:.0f} s"
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
            prepared = add_lag_columns(frame=prepared, power=arrays[site].power, arm=arm)
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            raw = models[site].predict(
                xgb.DMatrix(prepared.select(columns_of(arm=arm)).to_numpy().astype(np.float32))
            )
            pieces[site].append(
                fd.mixture_frame(
                    keys=keys, raw=raw, members=prepared["ensemble_member"].n_unique(), seed=seed
                )
            )
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=arm, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit the requested arms at the weekly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=sorted(ARMS))
    parser.add_argument("--regime", choices=("weekly", "monthly"), default="weekly")
    parser.add_argument("--sites", nargs="+", default=None, help="default: all Primaries")
    parser.add_argument(
        "--refit-stride", type=int, default=1, help="keep every n-th refit (the gate uses 13)"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=(args.regime,), origins=common.all_origins())
    jobs = dict(sorted(jobs.items())[:: args.refit_stride])
    rounds, out = common.N_ROUNDS, ARM_DIR / args.regime
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke" / args.regime
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                arm=a,
                refit=r,
                origins=runs,
                sites=sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
            )
            for r, runs in sorted(jobs.items(), key=lambda kv: -len(kv[1]))
            for a in args.arms
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
