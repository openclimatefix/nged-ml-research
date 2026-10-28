"""Refit experiment 87's monthly XGBoost member with one input group changed (experiment 371).

Each arm refits the member at the 12 monthly refit times for the series D8, D2, and D4, with
experiment 87's training rows, parameters, and leakage checks (its `e00087_fit_demand` module), and
writes the 11 quantiles per daily run to `work/arm_preds/<arm>/<site>_s0_<refit>.parquet`.

Arms (the control changes nothing):

- `control`: the stored recipe, 13 weather and calendar columns, lead hours, three origin lags,
  recency half-life 0.5 years, 51-member mixture.
- `no_extra_weather`: drops longwave, dew point, surface pressure, precipitation, and wind speed.
- `no_lags`: drops the three origin lags.
- `equal_weights`: every training row has weight 1.
- `ens_mean`: predicts from the ensemble-mean input row of each valid time (no mixture).
- `gam_inputs_only`: drops the five extra weather columns and the three origin lags.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00371-*/e00371_fit.py --workers 4 --cores 12
```
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
    ML_EXPERIMENTS_DATA_DIR / "00371-d8-monthly-xgboost-input-group-ablation" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
SITES: Final[tuple[str, ...]] = ("D8", "D2", "D4")
EXTRA_WEATHER: Final[tuple[str, ...]] = (
    "longwave",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
SMOKE_ROUNDS: Final[int] = 20


@dataclass(frozen=True)
class Arm:
    """One arm: the columns dropped, the weighting, and how a run's members are forecast."""

    name: str
    dropped: tuple[str, ...] = ()
    equal_weights: bool = False
    ensemble_mean: bool = False

    def columns(self) -> list[str]:
        """Return the input columns in matrix order."""
        return [c for c in fd.input_columns(a7=False) if c not in self.dropped]


ARMS: Final[dict[str, Arm]] = {
    arm.name: arm
    for arm in (
        Arm("control"),
        Arm("no_extra_weather", dropped=EXTRA_WEATHER),
        Arm("no_lags", dropped=common.LAG_COLUMNS),
        Arm("equal_weights", equal_weights=True),
        Arm("ens_mean", ensemble_mean=True),
        Arm("gam_inputs_only", dropped=(*EXTRA_WEATHER, *common.LAG_COLUMNS)),
    )
}


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
    weight = (
        np.ones(known.height)
        if arm.equal_weights
        else 0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS)
    )
    return xgb.DMatrix(
        known.select(arm.columns()).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=weight,
    )


def forecast_run(
    *, model: xgb.Booster, prepared: pl.DataFrame, arm: Arm, seed: int
) -> pl.DataFrame:
    """Return one run's 11 quantiles per valid time for the arm."""
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    if not arm.ensemble_mean:
        matrix = prepared.select(arm.columns()).to_numpy().astype(np.float32)
        raw = model.predict(xgb.DMatrix(matrix))
        members = prepared["ensemble_member"].n_unique()
        return fd.mixture_frame(keys=keys, raw=raw, members=members, seed=seed)
    averaged = (
        prepared.group_by(*common.KEYS, maintain_order=True)
        .agg(pl.col(c).mean() for c in arm.columns())
        .sort("valid_time")
    )
    matrix = averaged.select(arm.columns()).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix))
    return (
        averaged.select(*common.KEYS)
        .hstack(pl.DataFrame(raw, schema=list(common.LEVEL_COLUMNS), orient="row"))
        .with_columns(seed=pl.lit(seed).cast(pl.Int8))
    )


def run_job(
    *,
    arm_name: str,
    refit: datetime,
    origins: list[datetime],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
) -> list[str]:
    """Fit one arm at one refit for each site lacking its file, and forecast the refit's runs."""
    arm = ARMS[arm_name]
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in SITES
        if not fd.job_path(root=out_dir, config=arm.name, site=s, seed=seed, refit=refit).exists()
    ]
    lines = []
    models: dict[str, xgb.Booster] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = fd.known_rows(
            rows=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
            refit=refit,
            config=config,
        )
        models[site] = xgb.train(
            fd.parameters(config=config, seed=seed, nthread=nthread),
            train_matrix(known=known, row=row, refit=refit, arm=arm),
            num_boost_round=rounds,
        )
        lines.append(
            f"{arm.name} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
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
            pieces[site].append(
                forecast_run(model=models[site], prepared=prepared, arm=arm, seed=seed)
            )
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=arm.name, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit the requested arms at the monthly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cores", type=int, default=12)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    rounds, out = common.N_ROUNDS, ARM_DIR
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke"
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_job,
                arm_name=a,
                refit=r,
                origins=runs,
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
