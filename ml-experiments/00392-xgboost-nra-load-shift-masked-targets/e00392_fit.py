r"""Refit experiment 87's XGBoost member monthly, with load-shift events cut from its targets (392).

Each arm refits the `n1h` configuration of experiment 87 at the monthly refit times, with experiment
87's training rows, parameters, and leakage checks, and writes the 11 quantiles per daily run to
`work/arm_preds/<arm>_<form>/<site>_s<seed>_<refit>.parquet`. `colsample_bytree` and
`colsample_bynode` stay at 1.0. The fit loop follows experiment 383's `run_job`; only the training
mask differs. Every fitted booster is saved for the lag-encroachment sweep.

Arms (`--arms`), each fitted in a `lagged` form (three origin lags) and a `lagless` form:

- `r0`: no rows dropped.
- `r1`: drops training rows whose valid time lies inside a hand-labelled event or within
  `POST_EVENT_STEPS` after it ends. The unpadded event is recovered from the feature cache's
  hand-filtered mask by undoing its `CACHE_PADDING_STEPS` of padding on each side. Only events that
  ended before the refit are used.
- `r2`: the same rule applied to half-hours marked by the detector; reads `--mask-dir/<site>.npy`
  (a boolean array on experiment 87's grid, one file per series, built from data inside each
  fold's training window).
- `rc1`, `rc2`: drop as many training rows as `r1` and `r2` respectively, in random blocks of the
  same lengths (the control for a smaller training set).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1 DATA_PATH_INTERNAL=... \\
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00392-*/e00392_fit.py \\
    --arms r0 r1 --forms lagged lagless --workers 1 --cores 1 --smoke
```
"""

import argparse
import multiprocessing
import time
import zlib
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
    ML_EXPERIMENTS_DATA_DIR / "00392-xgboost-nra-load-shift-masked-targets" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
CACHE_PADDING_STEPS: Final[int] = 48
"""The feature cache pads each hand-labelled event by 24 hours (48 half-hours) on each side."""
POST_EVENT_STEPS: Final[int] = 96
"""Training rows within 48 hours after an event's end are dropped as well."""
ARM_NAMES: Final[tuple[str, ...]] = ("r0", "r1", "r2", "rc1", "rc2")
FORM_NAMES: Final[tuple[str, ...]] = ("lagged", "lagless")


def unpadded_runs(*, padded: np.ndarray) -> list[tuple[int, int]]:
    """Return the `[start, end)` grid steps of each event, with the cache's padding removed.

    A step belongs to an event when the padded mask holds over the `CACHE_PADDING_STEPS` before and
    after it. Two events closer than twice the padding come out joined, which is conservative.
    """
    window = 2 * CACHE_PADDING_STEPS + 1
    csum = np.concatenate([[0], np.cumsum(padded.astype(np.int64))])
    inner = np.zeros(padded.size, dtype=bool)
    centres = np.arange(CACHE_PADDING_STEPS, padded.size - CACHE_PADDING_STEPS)
    inner[centres] = (
        csum[centres + CACHE_PADDING_STEPS + 1] - csum[centres - CACHE_PADDING_STEPS]
    ) == window
    edges = np.diff(np.concatenate([[0], inner.astype(np.int8), [0]]))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def runs_of(*, mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the `[start, end)` grid steps of each run of True in `mask`."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


def drop_mask(*, runs: list[tuple[int, int]], refit_step: int, size: int) -> np.ndarray:
    """Return the steps to drop: each run that ended before the refit, plus 48 hours after it."""
    mask = np.zeros(size, dtype=bool)
    for start, end in runs:
        if end <= refit_step:
            mask[start : min(end + POST_EVENT_STEPS, size)] = True
    return mask


def random_blocks(*, true_mask: np.ndarray, first: int, last: int, seed_text: str) -> np.ndarray:
    """Return a mask with `true_mask`'s block lengths, at random onsets in `[first, last)`."""
    generator = np.random.default_rng(zlib.crc32(seed_text.encode()))
    mask = np.zeros(true_mask.size, dtype=bool)
    for start, end in runs_of(mask=true_mask):
        length = end - start
        onset = int(generator.integers(first, max(first + 1, last - length)))
        mask[onset : onset + length] = True
    return mask


def hand_runs(*, site: str) -> list[tuple[int, int]]:
    """Return the unpadded hand-labelled events of one series."""
    masks, _ = base.worker_state()
    return unpadded_runs(padded=masks[site])


def detector_mask(*, mask_dir: Path, site: str) -> np.ndarray:
    """Return the detector's marked half-hours of one series on experiment 87's grid."""
    return np.load(mask_dir / f"{site}.npy").astype(bool)


def training_drop(
    *, arm: str, site: str, refit: datetime, first: int, mask_dir: Path | None
) -> np.ndarray | None:
    """Return the grid mask of training steps to drop for `arm`, or None for no drop."""
    if arm == "r0":
        return None
    size = common.grid_times().len()
    refit_step = int(common.grid_index(times=refit))
    if arm in ("r1", "rc1"):
        runs = hand_runs(site=site)
    else:
        if mask_dir is None:
            msg = f"arm {arm} needs --mask-dir"
            raise ValueError(msg)
        runs = runs_of(mask=detector_mask(mask_dir=mask_dir, site=site))
    true_mask = drop_mask(runs=runs, refit_step=refit_step, size=size)
    if arm in ("r1", "r2"):
        return true_mask
    return random_blocks(
        true_mask=true_mask,
        first=first,
        last=refit_step,
        seed_text=f"{arm}:{site}:{refit:%Y-%m-%d}",
    )


def run_job(
    *,
    arm_name: str,
    form: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    out_dir: Path,
    mask_dir: Path | None,
) -> list[str]:
    """Fit one arm and form at one refit for each site lacking its file, and forecast its runs."""
    name = f"{arm_name}_{form}"
    arm = base.Arm(
        name, drop_event_targets=False, mask_lags=False, pseudo=False, use_lags=form == "lagged"
    )
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not fd.job_path(root=out_dir, config=name, site=s, seed=seed, refit=refit).exists()
    ]
    lines = []
    models: dict[str, xgb.Booster] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        rows = base.flagged_training_rows(row=row, arrays=arrays[site], with_pseudo=False)
        known = fd.known_rows(rows=rows, refit=refit, config=config)
        target = np.asarray(common.grid_index(times=known["valid_time"]))
        first = int(target.min())
        drop = training_drop(arm=arm_name, site=site, refit=refit, first=first, mask_dir=mask_dir)
        if drop is not None:
            known = known.filter(
                pl.Series(common.take(values=drop.astype(np.float32), index=target) < 0.5)
            )
        models[site] = xgb.train(
            fd.parameters(config=config, seed=seed, nthread=nthread),
            base.train_matrix(known=known, row=row, refit=refit, arm=arm),
            num_boost_round=rounds,
        )
        path = BOOSTER_DIR / name / f"{site}_s{seed}_{refit:%Y%m%d}.ubj"
        path.parent.mkdir(parents=True, exist_ok=True)
        models[site].save_model(path)
        lines.append(
            f"{name} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
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
                base.forecast_run(model=models[site], prepared=prepared, arm=arm, seed=seed)
            )
    for site, tables in pieces.items():
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(root=out_dir, config=name, site=site, seed=seed, refit=refit),
        )
    return lines


def main() -> None:
    """Fit the requested arms and forms at the monthly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=ARM_NAMES)
    parser.add_argument("--forms", nargs="+", default=list(FORM_NAMES), choices=FORM_NAMES)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--refits", type=int, default=None, help="only the first N refits")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--mask-dir", type=Path, default=None, help="detector masks for r2 and rc2")
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    jobs = dict(list(jobs.items())[: args.refits])
    rounds, out = common.N_ROUNDS, ARM_DIR
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = base.SMOKE_ROUNDS
        out = WORK_DIR / "arm_preds_smoke"
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                arm_name=a,
                form=f,
                refit=r,
                origins=runs,
                sites=sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
                mask_dir=args.mask_dir,
            )
            for r, runs in jobs.items()
            for a in args.arms
            for f in args.forms
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
