"""Refit experiment 87's weekly XGBoost member with GAM-learnt shapes as input columns (385).

Each arm refits the `n1h` configuration of experiment 87 (the BG2 weekly member) at the monthly (default) or weekly
refit times, with experiment 87's training rows, parameters, and leakage checks (its
`e00087_fit_demand` module). `colsample_bytree` and `colsample_bynode` stay at 1.0 in every arm.
The 11 quantiles per daily run go to `work/arm_preds/<arm>/<site>_s<seed>_<refit>.parquet`.

Arms:

- `control`: experiment 87's stored recipe (the reproduction gate and the baseline).
- `fitted`: adds `gam_fitted`, experiment 377's T-all fit net power for the row's forecast weather.
- `shapes`: adds the four partial effects (`gam_calendar`, `gam_season`, `gam_temperature`,
  `gam_lighting`).
- `both`: adds all five.

At the refit time `R` the GAM is fitted at the cut `R - 3.5 h` on its 24-month window. A training
row's GAM columns come from the fit that leaves out the row's valid month (out of fold). A scored
row's columns come from the whole-window fit, driven by that member's forecast weather. See
`e00385_gam`.

Set `E00377_SOURCE` to `control` (D5, D9, D10, D11, D14, D16, D18), `mva` (D2, D3, D4, D8, D12,
D19, D21, D22), or `d15` (D15) before launching:

```bash
E=ml-experiments
E00377_SOURCE=control PYTHONPATH=$E/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
$E/00377-demand-v1-first-run-feasibility:$E/00385-gam-shapes-as-xgboost-inputs-weekly-demand \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00385-*/e00385_fit.py \
    --arms control fitted shapes both --workers 1 --cores 1
```
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00377_common as common377  # ty: ignore[unresolved-import]
import e00385_gam as gam  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00385-gam-shapes-as-xgboost-inputs-weekly-demand" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
SITES_BY_SOURCE: Final[dict[str, tuple[str, ...]]] = {
    "control": ("D5", "D9", "D10", "D11", "D14", "D16", "D18"),
    "mva": ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"),
    "d15": ("D15",),
}
SMOKE_ROUNDS: Final[int] = 20
KEYS: Final[list[str]] = ["power_fcst_init_time", "valid_time"]
MAX_UNEXPLAINED_SHARE: Final[float] = 0.001
"""A refit raises when more than this share of its training rows lacks a GAM column for a reason
other than the two below."""
GAM_HORIZON_HOURS: Final[float] = 336.0
"""The GAM window ends 336 hours after the origin; XGBoost rows beyond it have no GAM column."""
NO_PROXY_DAYS: Final[int] = 7
"""Runs in the first week of the store have no analysis proxy, so no driven weather."""


@dataclass(frozen=True)
class Arm:
    """An arm: which GAM columns it adds to experiment 87's inputs."""

    name: str
    gam_columns: tuple[str, ...]

    def columns(self) -> list[str]:
        """Return all input columns in matrix order."""
        return [*fd.input_columns(a7=False), *self.gam_columns]


ARMS: Final[dict[str, Arm]] = {
    "control": Arm(name="control", gam_columns=()),
    "fitted": Arm(name="fitted", gam_columns=("gam_fitted",)),
    "shapes": Arm(name="shapes", gam_columns=gam.SHAPE_COLUMNS),
    "both": Arm(name="both", gam_columns=gam.COLUMNS),
}


def attach(*, frame: pl.DataFrame, columns: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Left-join GAM columns onto a frame on `keys`, keeping the frame's row order."""
    return frame.join(
        columns.with_columns(
            pl.col("power_fcst_init_time").cast(frame.schema["power_fcst_init_time"]),
            pl.col("valid_time").cast(frame.schema["valid_time"]),
        ),
        on=keys,
        how="left",
        validate="m:1",
        maintain_order="left",
    )


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        known.select(arm.columns()).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
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
    arm_names: list[str],
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
    model_dir: Path,
) -> list[str]:
    """Fit each arm at one refit for each site lacking its file, and forecast the refit's runs."""
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    cut = refit - regimes.REFIT_CUTOFF_AGE
    lines: list[str] = []
    for site in sites:
        todo = [
            n
            for n in arm_names
            if not fd.job_path(root=out_dir, config=n, site=site, seed=seed, refit=refit).exists()
        ]
        if not todo:
            continue
        row = rows_of[site]
        ctx = gam.load_site(site=site)
        vintages = common377.vintages_for(cache=ctx.cache)
        tall_arm = gam.tall_arm()
        started = time.monotonic()
        models = gam.fit_cut(cache=ctx.cache, vintages=vintages, site=site, cut=cut)
        known = fd.known_rows(
            rows=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
            refit=refit,
            config=config,
        )
        run_indices = [i for i, o in enumerate(ctx.setup.origins) if o <= refit]
        train_cols = gam.training_columns(
            ctx=ctx, models=models, arm=tall_arm, run_indices=run_indices
        )
        known = attach(frame=known, columns=train_cols, keys=KEYS)
        first_run = min(ctx.setup.origins)
        unexplained = known.filter(
            pl.col("gam_fitted").is_null()
            & (pl.col("lead_hours") <= GAM_HORIZON_HOURS)
            & (pl.col("power_fcst_init_time") >= first_run + timedelta(days=NO_PROXY_DAYS))
        ).height
        missing = known["gam_fitted"].is_null().mean()
        if unexplained > MAX_UNEXPLAINED_SHARE * known.height:
            msg = f"{site} refit {refit:%Y-%m-%d}: {unexplained} training rows lack GAM columns"
            raise ValueError(msg)
        known = known.with_columns(pl.col(*gam.COLUMNS).fill_nan(None))
        models_by_arm = {
            n: xgb.train(
                fd.parameters(config=config, seed=seed, nthread=nthread),
                train_matrix(known=known, row=row, refit=refit, arm=ARMS[n]),
                num_boost_round=rounds,
            )
            for n in todo
        }
        for n, booster in models_by_arm.items():
            model_path = model_dir / n / f"{site}_{refit:%Y%m%dT%H%M}.ubj"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(model_path)
        pieces: dict[str, list[pl.DataFrame]] = {n: [] for n in todo}
        for origin in origins:
            frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            )
            run_index = ctx.setup.origins.index(origin)
            scored = gam.scored_columns(
                ctx=ctx,
                models=models,
                arm=tall_arm,
                run_index=run_index,
                members=sorted(prepared["ensemble_member"].unique().to_list()),
            )
            prepared = attach(frame=prepared, columns=scored, keys=[*KEYS, "ensemble_member"])
            prepared = prepared.with_columns(pl.col(*gam.COLUMNS).fill_nan(None))
            for n in todo:
                pieces[n].append(
                    forecast_run(model=models_by_arm[n], prepared=prepared, arm=ARMS[n], seed=seed)
                )
        for n, tables in pieces.items():
            common.atomic_write(
                frame=pl.concat(tables),
                target=fd.job_path(root=out_dir, config=n, site=site, seed=seed, refit=refit),
            )
        lines.append(
            f"{site} refit {refit:%Y-%m-%d}: arms {todo}, {known.height:,} rows, "
            f"{missing:.2%} without GAM columns, {time.monotonic() - started:.0f} s"
        )
    return lines


def main() -> None:
    """Fit the requested arms at the refits of the chosen regime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=sorted(ARMS))
    parser.add_argument("--sites", nargs="+", default=None, help="default: the source's sites")
    parser.add_argument("--regime", choices=("monthly", "weekly"), default="monthly")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    parser.add_argument("--refits", type=int, default=None, help="only the first N refits")
    args = parser.parse_args()
    sites = args.sites or list(SITES_BY_SOURCE[common377.SOURCE])
    jobs = regimes.jobs_for(regime_names=(args.regime,), origins=common.all_origins())
    rounds, out = common.N_ROUNDS, ARM_DIR / args.regime
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke" / args.regime
    if args.refits is not None:
        jobs = dict(sorted(jobs.items())[: args.refits])
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                arm_names=args.arms,
                refit=r,
                origins=runs,
                sites=sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
                model_dir=WORK_DIR / ("models_smoke" if args.smoke else "models") / args.regime,
            )
            for r, runs in sorted(jobs.items(), key=lambda kv: -len(kv[1]))
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
