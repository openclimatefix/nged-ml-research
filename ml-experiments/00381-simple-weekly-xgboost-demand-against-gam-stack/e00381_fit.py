"""Refit experiment 87's XGBoost demand member with small input or weighting changes (381).

Each arm refits the `n1h` configuration of experiment 87 (the BG2 weekly and monthly member) at the
refit times of one regime, for the chosen series, with experiment 87's training rows, parameters,
and leakage checks (its `e00087_fit_demand` module). It writes the 11 quantiles per daily run to
`work/arm_preds/<regime>/<arm>/<site>_s<seed>_<refit>.parquet`. `colsample_bytree` and
`colsample_bynode` stay at 1.0 in every arm.

Arms (the control changes nothing):

- `control`: the stored recipe.
- `e7`: adds a 0/1 flag for the local half-hours 1 to 6 and the half-hours since local 00:30
  (clipped to 0 to 12).
- `solar`: adds the mean forecast shortwave over the local day of the valid time and over the
  previous local day, per ensemble member and run.
- `leadw`: row weights use a half-life of 0.25 years at leads under 48 hours and 1 year beyond.
- `sunny`: adds a sunny flag (shortwave over the Haurwitz clear-sky irradiance above 0.6) and its
  product with shortwave.
- `bundle_<names>`: the arms joined by `+`, for example `bundle_e7+solar`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00381-*/e00381_fit.py \
    --regime monthly --arms control e7 --workers 3 --cores 3
```
"""

import argparse
import multiprocessing
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
    ML_EXPERIMENTS_DATA_DIR / "00381-simple-weekly-xgboost-demand-against-gam-stack" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
E7_COLUMNS: Final[tuple[str, ...]] = ("e7_flag", "half_hours_since_0030")
SOLAR_COLUMNS: Final[tuple[str, ...]] = ("shortwave_day_mean", "shortwave_prev_day_mean")
SUNNY_COLUMNS: Final[tuple[str, ...]] = ("sunny_flag", "sunny_times_shortwave")
E7_FIRST_HALF_HOUR: Final[int] = 1
E7_LAST_HALF_HOUR: Final[int] = 6
E7_RAMP_CLIP: Final[int] = 12
SUNNY_INDEX: Final[float] = 0.6
SHORT_LEAD_HOURS: Final[float] = 48.0
SHORT_LEAD_HALF_LIFE_DAYS: Final[float] = 0.25 * 365.25
LONG_LEAD_HALF_LIFE_DAYS: Final[float] = 1.0 * 365.25
SMOKE_ROUNDS: Final[int] = 20
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")


@dataclass(frozen=True)
class Arm:
    """One arm: the groups of columns it adds and whether it weights rows by lead."""

    name: str
    e7: bool = False
    solar: bool = False
    sunny: bool = False
    lead_weights: bool = False

    def extra_columns(self) -> list[str]:
        """Return the added input columns in matrix order."""
        return [
            *(E7_COLUMNS if self.e7 else ()),
            *(SOLAR_COLUMNS if self.solar else ()),
            *(SUNNY_COLUMNS if self.sunny else ()),
        ]

    def columns(self) -> list[str]:
        """Return all input columns in matrix order."""
        return [*fd.input_columns(a7=False), *self.extra_columns()]


def make_arm(*, name: str) -> Arm:
    """Return the arm called `name`; `bundle_a+b` joins the changes of arms `a` and `b`."""
    parts = name.removeprefix("bundle_").split("+") if name.startswith("bundle_") else [name]
    unknown = set(parts) - {"control", "e7", "solar", "leadw", "sunny"}
    if unknown:
        msg = f"unknown arm parts {sorted(unknown)}"
        raise ValueError(msg)
    return Arm(
        name=name,
        e7="e7" in parts,
        solar="solar" in parts,
        sunny="sunny" in parts,
        lead_weights="leadw" in parts,
    )


def add_extras(*, frame: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Add the arm's columns to a frame of rows with `valid_time`, run, member, and weather.

    Every added column is computed from the row's own valid time, its own run's forecast weather,
    and its own ensemble member, so no column reads a later run or the observed power.
    """
    if not arm.extra_columns():
        return frame
    local = pl.col("valid_time").dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    half_hour = local.dt.hour() * 2 + local.dt.minute() // 30
    out = frame.with_columns(_half_hour=half_hour, _local_date=local.dt.date())
    if arm.e7:
        out = out.with_columns(
            e7_flag=(
                (pl.col("_half_hour") >= E7_FIRST_HALF_HOUR)
                & (pl.col("_half_hour") <= E7_LAST_HALF_HOUR)
            ).cast(pl.Float32),
            half_hours_since_0030=(pl.col("_half_hour") - E7_FIRST_HALF_HOUR)
            .clip(0, E7_RAMP_CLIP)
            .cast(pl.Float32),
        )
    if arm.solar:
        keys = ["power_fcst_init_time", "ensemble_member", "_local_date"]
        daily = out.group_by(keys).agg(pl.col("shortwave").mean().alias("shortwave_day_mean"))
        previous = daily.with_columns(
            _local_date=pl.col("_local_date") + pl.duration(days=1)
        ).rename({"shortwave_day_mean": "shortwave_prev_day_mean"})
        out = out.join(daily, on=keys, how="left", validate="m:1", maintain_order="left").join(
            previous, on=keys, how="left", validate="m:1", maintain_order="left"
        )
        out = out.with_columns(pl.col(*SOLAR_COLUMNS).cast(pl.Float32))
    if arm.sunny:
        cosine = (pl.col("solar_zenith").radians().cos()).clip(lower_bound=0.0)
        clear = 1098.0 * cosine * (-0.057 / cosine.clip(lower_bound=0.01)).exp()
        index = pl.when(clear > 1.0).then(pl.col("shortwave") / clear).otherwise(0.0)
        flag = (index > SUNNY_INDEX).cast(pl.Float32)
        out = out.with_columns(
            sunny_flag=flag, sunny_times_shortwave=flag * pl.col("shortwave")
        ).with_columns(pl.col(*SUNNY_COLUMNS).cast(pl.Float32))
    return out.drop("_half_hour", "_local_date")


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    if arm.lead_weights:
        half_life = np.where(
            known["lead_hours"].to_numpy() < SHORT_LEAD_HOURS,
            SHORT_LEAD_HALF_LIFE_DAYS,
            LONG_LEAD_HALF_LIFE_DAYS,
        )
    else:
        half_life = common.SHORT_HALF_LIFE_DAYS
    return xgb.DMatrix(
        known.select(arm.columns()).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / half_life),
    )


def forecast_run(
    *, model: xgb.Booster, prepared: pl.DataFrame, arm: Arm, seed: int
) -> pl.DataFrame:
    """Return one run's 11 quantiles per valid time: the mixture of the 51 members."""
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    matrix = prepared.select(arm.columns()).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix))
    members = prepared["ensemble_member"].n_unique()
    return fd.mixture_frame(keys=keys, raw=raw, members=members, seed=seed)


def run_job(
    *,
    arm_name: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
) -> list[str]:
    """Fit one arm at one refit for each site lacking its file, and forecast the refit's runs."""
    arm = make_arm(name=arm_name)
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not fd.job_path(root=out_dir, config=arm.name, site=s, seed=seed, refit=refit).exists()
    ]
    lines = []
    models: dict[str, xgb.Booster] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = fd.known_rows(
            rows=add_extras(
                frame=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
                arm=arm,
            ),
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
            prepared = add_extras(frame=prepared, arm=arm)
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
    """Fit the requested arms at the refits of the requested regime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--regime", choices=("weekly", "monthly"), required=True)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all series")
    parser.add_argument("--mva-only", action="store_true", help="the eight MVA-only Primaries")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    for name in args.arms:
        make_arm(name=name)
    series = common.series_table()
    sites = (
        list(MVA_ONLY)
        if args.mva_only
        else args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    )
    jobs = regimes.jobs_for(regime_names=(args.regime,), origins=common.all_origins())
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
                arm_name=a,
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
