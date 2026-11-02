r"""Refit experiment 87's XGBoost member at the monthly refits with one power lag left out (502).

The recipe is experiment 392's `e00392_fit.py` (as copied by experiment 402's `e00402_refit.py`):
experiment 87's `n1h` configuration, monthly refits at 2025-10-01, 2026-01-01, 2026-03-01, and
2026-06-01, 500 rounds, `eta` 0.05, 11 quantiles, the control member for training, the 51 members
pooled as a mixture at scoring, plain targets, `colsample` 1.0, CUDA. Only the lag columns differ.

Arms (`--arms`), each written to `work/arm_preds/<arm>/<site>_s<seed>_<refit>.parquet`, with every
booster saved under `work/boosters/<arm>/`:

- `a_rep`: the three original lags, refitted on CUDA once more (the noise-floor control).
- `nowk`: no `lag_same_time_latest_week`.
- `noday`: no `lag_same_time_latest_day`.
- `noorigin`: no `lag_origin`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1 DATA_PATH_INTERNAL=... \\
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00502-*/e00502_fit.py \\
    --arms nowk noday a_rep --workers 3 --sites D5 D9 D10 D11 D14 D16
```
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00383_fit as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00502-leave-one-lag-out-power-lag-ablation-xgboost-demand" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six Primaries metered in signed megawatts."""
DROPPED_LAG: Final[dict[str, str | None]] = {
    "a_rep": None,
    "nowk": "lag_same_time_latest_week",
    "noday": "lag_same_time_latest_day",
    "noorigin": "lag_origin",
}
ARM_NAMES: Final[tuple[str, ...]] = tuple(DROPPED_LAG)
DEVICE: Final[str] = "cuda"


def input_columns(*, arm: str) -> list[str]:
    """Return the input columns of `arm` in matrix order: the lagged form minus one lag."""
    dropped = DROPPED_LAG[arm]
    return [c for c in fd.input_columns(a7=False) if c != dropped]


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime, arm: str) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        known.select(input_columns(arm=arm)).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
    )


def forecast_run(
    *, model: xgb.Booster, prepared: pl.DataFrame, arm: str, seed: int
) -> pl.DataFrame:
    """Return one run's 11 quantiles per valid time: the mixture of the 51 members."""
    keys = prepared.select(*common.KEYS).unique(maintain_order=True)
    matrix = prepared.select(input_columns(arm=arm)).to_numpy().astype(np.float32)
    raw = model.predict(xgb.DMatrix(matrix))
    members = prepared["ensemble_member"].n_unique()
    return fd.mixture_frame(keys=keys, raw=raw, members=members, seed=seed)


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
        rows = base.flagged_training_rows(row=row, arrays=arrays[site], with_pseudo=False)
        known = fd.known_rows(rows=rows, refit=refit, config=config)
        models[site] = xgb.train(
            {**fd.parameters(config=config, seed=seed, nthread=nthread), "device": DEVICE},
            train_matrix(known=known, row=row, refit=refit, arm=arm),
            num_boost_round=rounds,
        )
        booster_dir = (
            BOOSTER_DIR.with_name("boosters_smoke") if "smoke" in out_dir.name else BOOSTER_DIR
        )
        path = booster_dir / arm / f"{site}_s{seed}_{refit:%Y%m%d}.ubj"
        path.parent.mkdir(parents=True, exist_ok=True)
        models[site].save_model(path)
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
            pieces[site].append(
                forecast_run(model=models[site], prepared=prepared, arm=arm, seed=seed)
            )
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=arm, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit the requested arms at the four monthly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=ARM_NAMES)
    parser.add_argument("--sites", nargs="+", default=list(SIX))
    parser.add_argument(
        "--refit-dates",
        nargs="+",
        default=["2025-10-01", "2026-01-01", "2026-03-01", "2026-06-01"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = {r: runs for r, runs in jobs.items() if f"{r:%Y-%m-%d}" in args.refit_dates}
    rounds, out = common.N_ROUNDS, ARM_DIR
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = base.SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke"
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                arm=arm,
                refit=r,
                origins=runs,
                sites=[site],
                nthread=1,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
            )
            for r, runs in jobs.items()
            for arm in args.arms
            for site in args.sites
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
