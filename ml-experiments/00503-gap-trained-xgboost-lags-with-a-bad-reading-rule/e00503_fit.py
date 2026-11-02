r"""Fit experiment 503's gap-trained XGBoost demand forecast (arm G) at the six Primaries.

The recipe is experiment 392's `e00392_fit.py` (as copied by experiment 402's `e00402_refit.py` and
experiment 465's `e00465_fit.py`): experiment 87's `n1h` configuration, monthly refits, 500 rounds,
`eta` 0.05, 11 quantiles, the control member for training, the 51 members pooled as a mixture at
scoring, plain targets, `colsample` 1.0, and the three original lags. The only difference is the
training matrix: each of the three lag values of every training row is independently set to
missing (NaN) with probability `GAP_RATE`, so XGBoost learns a default branch for a missing lag.
The mask is drawn once per series and refit from a seeded generator and saved beside the booster
(`work/boosters/g_gap/<site>_s<seed>_<refit>_mask.npz`, bit-packed with its shape). Inputs at
scoring time are not masked here: the serving rules in `e00503_rules.py` act on them.

Fits write `work/arm_preds/g_gap/<site>_s<seed>_<refit>.parquet` and save each booster under
`work/boosters/g_gap/`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1 DATA_PATH_INTERNAL=... \\
    systemd-run --user --scope -p MemoryMax=10G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00503-*/e00503_fit.py \\
    --workers 1 --cores 1 --device cuda
```
"""

import argparse
import multiprocessing
import resource
import time
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
from studies.eval_protocol.slices import SIGNED_MW_LABELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00503-gap-trained-xgboost-lags-with-a-bad-reading-rule" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
ARM_NAME: Final[str] = "g_gap"
GAP_RATE: Final[float] = 0.10
"""Probability that one lag value of one training row is set to missing (a stated default)."""
REFIT_DATES: Final[tuple[str, ...]] = ("2025-10-01", "2026-01-01", "2026-03-01", "2026-06-01")
SMOKE_ROUNDS: Final[int] = 20


def gap_mask(*, rows: int, site: str, refit: datetime, seed: int) -> np.ndarray:
    """Return a boolean (rows, 3) mask, True where a lag value is set to missing.

    The generator is seeded from the experiment seed, the refit date, and the site number, so the
    same fit always draws the same mask, whatever the worker schedule.
    """
    generator = np.random.default_rng([seed, int(f"{refit:%Y%m%d}"), int(site[1:])])
    return generator.random((rows, len(common.LAG_COLUMNS))) < GAP_RATE


def columns() -> list[str]:
    """Return the input columns in matrix order: the originals, with the three original lags."""
    return fd.input_columns(a7=False)


def train_matrix(
    *, known: pl.DataFrame, row: dict, refit: datetime, mask: np.ndarray
) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, with masked lags and weights at `refit`."""
    names = columns()
    values = known.select(names).to_numpy().astype(np.float32)
    lag_positions = [names.index(c) for c in common.LAG_COLUMNS]
    for column, position in enumerate(lag_positions):
        values[mask[:, column], position] = np.nan
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        values,
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


def booster_path(*, site: str, seed: int, refit: datetime, base: Path = BOOSTER_DIR) -> Path:
    """Return the file one booster is saved to."""
    return base / ARM_NAME / f"{site}_s{seed}_{refit:%Y%m%d}.ubj"


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
    """Fit arm G at one refit for each site lacking its file, and forecast the refit's runs."""
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
        rows = fd._training_rows_cached(row=row, arrays=arrays[site], config=config)
        known = fd.known_rows(rows=rows, refit=refit, config=config)
        mask = gap_mask(rows=known.height, site=site, refit=refit, seed=seed)
        models[site] = xgb.train(
            {**fd.parameters(config=config, seed=seed, nthread=nthread), "device": device},
            train_matrix(known=known, row=row, refit=refit, mask=mask),
            num_boost_round=rounds,
        )
        path = booster_path(
            site=site,
            seed=seed,
            refit=refit,
            base=BOOSTER_DIR if out_dir == ARM_DIR else out_dir.parent / "boosters_smoke",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        models[site].save_model(path)
        np.savez_compressed(
            path.with_name(f"{path.stem}_mask.npz"),
            packed=np.packbits(mask, axis=None),
            shape=mask.shape,
        )
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        lines.append(
            f"{ARM_NAME} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
            f"{mask.mean():.3f} of lag values masked, "
            f"fit total {time.monotonic() - started:.0f} s, "
            f"peak resident memory so far {peak_gb:.1f} GB"
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
            pieces[site].append(forecast_run(model=models[site], prepared=prepared, seed=seed))
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=ARM_NAME, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit arm G at the four monthly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=None, help="default: the signed-MW six")
    parser.add_argument("--refit-dates", nargs="+", default=list(REFIT_DATES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    sites = args.sites or sorted(SIGNED_MW_LABELS, key=lambda s: int(s[1:]))
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
                sites=sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
                device=args.device,
            )
            for r, runs in jobs.items()
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
