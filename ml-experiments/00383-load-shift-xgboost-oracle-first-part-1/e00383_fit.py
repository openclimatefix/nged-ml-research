"""Refit experiment 87's weekly XGBoost member with load-shift events told to the model (383).

Each arm refits the `n1h` configuration of experiment 87 (the BG2 weekly member) at the weekly
refit times for the chosen series, with experiment 87's training rows, parameters, and leakage
checks (its `e00087_fit_demand` module). It writes the 11 quantiles per daily run to
`work/arm_preds/<arm>/<site>_s<seed>_<refit>.parquet`. `colsample_bytree` and `colsample_bynode`
stay at 1.0 in every arm. The event labels are the padded hand-filtered load-shift masks of the
feature cache (the oracle).

Arms:

- `d`: experiment 87's stored recipe, used to check that the refit reproduces the stored weekly
  member of experiment 87 (the reproduction gate).
- `c1`: drops the training rows whose valid time lies inside an event, sets to missing every lag
  that lies inside an event (in training and at test time), and, in training only, also sets to
  missing the lags inside pseudo events that cover 30% of the training rows (experiment 100's
  `drop30`, with event durations drawn from the events that start before 2025-07-01).
- `lagless`: drops the three origin lags and drops the training rows inside an event. The scorer
  serves it on the rows with a lag inside an event (arm c2).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00383-*/e00383_fit.py \
    --arms c1 lagless --workers 3 --cores 3
```
"""

import argparse
import multiprocessing
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00383-load-shift-xgboost-oracle-first-part-1" / "work"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SMOKE_ROUNDS: Final[int] = 20
PSEUDO_PERCENT: Final[int] = 30
PSEUDO_SHARE_TOLERANCE: Final[float] = 0.015
MAX_PSEUDO_INTERVALS: Final[int] = 3_000
DURATION_CUT: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""Pseudo-event durations are drawn from events that start before this time."""
EVENT_LAG_COLUMNS: Final[tuple[str, ...]] = tuple(f"event_{name}" for name in common.LAG_COLUMNS)
PSEUDO_LAG_COLUMNS: Final[tuple[str, ...]] = tuple(f"pseudo_{name}" for name in common.LAG_COLUMNS)
ARM_NAMES: Final[tuple[str, ...]] = ("d", "c1", "lagless")


@dataclass(frozen=True)
class Arm:
    """How an arm picks its training rows, masks its lags, and which columns it uses."""

    name: str
    drop_event_targets: bool
    mask_lags: bool
    pseudo: bool
    use_lags: bool

    def columns(self) -> list[str]:
        """Return the input columns in matrix order."""
        columns = fd.input_columns(a7=False)
        return columns if self.use_lags else [c for c in columns if c not in common.LAG_COLUMNS]


ARMS: Final[dict[str, Arm]] = {
    "d": Arm("d", drop_event_targets=False, mask_lags=False, pseudo=False, use_lags=True),
    "c1": Arm("c1", drop_event_targets=True, mask_lags=True, pseudo=True, use_lags=True),
    "lagless": Arm(
        "lagless", drop_event_targets=True, mask_lags=False, pseudo=False, use_lags=False
    ),
}


def event_masks() -> dict[str, np.ndarray]:
    """Return, per Primary label, the padded hand-filtered event mask on experiment 87's grid."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    index = np.asarray(common.grid_index(times=cache.times_series()))
    size = common.grid_times().len()
    out: dict[str, np.ndarray] = {}
    for label, features in cache.sites.items():
        mask = np.zeros(size, dtype=bool)
        inside = (index >= 0) & (index < size)
        mask[index[inside]] = features.switching_mask[inside]
        out[label] = mask
    return out


def event_durations(*, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Return the durations in half-hours of the events that start before `DURATION_CUT`."""
    cut = int(common.grid_index(times=DURATION_CUT))
    durations = []
    for mask in masks.values():
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        durations.extend(int(e - s) for s, e in zip(starts, ends, strict=True) if s < cut)
    if not durations:
        msg = "no event starts before the duration cut"
        raise ValueError(msg)
    return np.asarray(durations, dtype=np.int64)


def lag_flags(*, mask: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return, per row and lag, whether the lag's time lies inside the mask."""
    return common.origin_lags(power=mask.astype(np.float32), target=target, origin=origin) > 0.5


def pseudo_mask(
    *, site: str, rows: pl.DataFrame, durations: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return a pseudo-event mask on the grid and the share of rows with a masked lag.

    Intervals start uniformly over the span of the series' training rows and last as long as a
    bootstrap draw from `durations`. The draw is seeded by the series. An interval is accepted
    unless it would lift the share of rows with at least one pseudo-masked lag above the target
    plus half the tolerance, and drawing stops once the share reaches the target minus half the
    tolerance (experiment 100's recipe, for the 30% target only).

    Raises:
        ValueError: If the draw cannot reach the target share within the tolerance.
    """
    generator = np.random.default_rng(zlib.crc32(f"{common.SEED}:{site}".encode()))
    size = common.grid_times().len()
    target = np.asarray(common.grid_index(times=rows["valid_time"]))
    origin = np.asarray(common.grid_index(times=rows["power_fcst_init_time"]))
    lag_index = common.origin_lags(
        power=np.arange(size, dtype=np.float32), target=target, origin=origin
    )
    first, last = int(target.min()), int(target.max())
    drawn = generator.choice(durations, size=MAX_PSEUDO_INTERVALS)
    onsets = generator.integers(first, last + 1, size=MAX_PSEUDO_INTERVALS)
    masked_rows = np.zeros(rows.height, dtype=bool)
    mask = np.zeros(size, dtype=bool)
    goal, half = PSEUDO_PERCENT / 100.0, PSEUDO_SHARE_TOLERANCE / 2.0
    position = 0
    while position < MAX_PSEUDO_INTERVALS and masked_rows.mean() < goal - half:
        start, stop = int(onsets[position]), int(onsets[position] + drawn[position])
        position += 1
        inside = ((lag_index >= start) & (lag_index < stop)).any(axis=1)
        if (masked_rows | inside).mean() > goal + half:
            continue
        masked_rows |= inside
        mask[start : min(stop, size)] = True
    share = float(masked_rows.mean())
    if abs(share - goal) > PSEUDO_SHARE_TOLERANCE:
        msg = f"{site}: pseudo share {share:.3f} is not within tolerance of {goal:.2f}"
        raise ValueError(msg)
    return mask, share


def with_event_flags(
    *, frame: pl.DataFrame, event: np.ndarray, pseudo: np.ndarray | None
) -> pl.DataFrame:
    """Add `in_event` (target inside an event) and per-lag event and pseudo-event flags."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    out = frame.with_columns(
        in_event=pl.Series(common.take(values=event.astype(np.float32), index=target) > 0.5)
    )
    out = out.hstack(
        pl.DataFrame(
            lag_flags(mask=event, target=target, origin=origin),
            schema=list(EVENT_LAG_COLUMNS),
            orient="row",
        )
    )
    if pseudo is not None:
        out = out.hstack(
            pl.DataFrame(
                lag_flags(mask=pseudo, target=target, origin=origin),
                schema=list(PSEUDO_LAG_COLUMNS),
                orient="row",
            )
        )
    return out


def mask_lags(*, frame: pl.DataFrame, use_pseudo: bool) -> pl.DataFrame:
    """Set to missing each lag whose time lies inside an event (or, in training, a pseudo event)."""
    flags = [pl.col(c) for c in EVENT_LAG_COLUMNS]
    if use_pseudo:
        flags = [f | pl.col(p) for f, p in zip(flags, PSEUDO_LAG_COLUMNS, strict=True)]
    return frame.with_columns(
        pl.when(flag).then(None).otherwise(pl.col(lag)).alias(lag)
        for lag, flag in zip(common.LAG_COLUMNS, flags, strict=True)
    )


def train_matrix(*, known: pl.DataFrame, row: dict, refit: datetime, arm: Arm) -> xgb.DMatrix:
    """Return the training matrix of one series' rows, weights anchored at `refit`."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        known.select(arm.columns()).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
    )


_PREPARED: dict[str, object] = {}


def worker_state() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return the event masks and pseudo-event durations, loaded once per worker process."""
    if "masks" not in _PREPARED:
        masks = event_masks()
        _PREPARED["masks"] = masks
        _PREPARED["durations"] = event_durations(masks=masks)
        _PREPARED["pseudo"] = {}
    return _PREPARED["masks"], _PREPARED["durations"]  # ty: ignore[invalid-return-type]


def cached_pseudo_mask(*, site: str, rows: pl.DataFrame, durations: np.ndarray) -> np.ndarray:
    """Return a series' pseudo-event mask, read from `work/pseudo` or drawn once and saved there."""
    path = WORK_DIR / "pseudo" / f"{site}.npy"
    if path.exists():
        return np.load(path)
    mask, share = pseudo_mask(site=site, rows=rows, durations=durations)
    print(f"{site}: pseudo-event share {share:.3f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{multiprocessing.current_process().pid}.tmp.npy")
    np.save(temporary, mask)
    temporary.replace(path)
    return mask


def flagged_training_rows(
    *, row: dict, arrays: common.SeriesArrays, with_pseudo: bool
) -> pl.DataFrame:
    """Return a series' training rows with event flags, kept for the worker's later jobs."""
    masks, durations = worker_state()
    cache: dict[tuple[str, bool], pl.DataFrame] = _PREPARED.setdefault("rows", {})  # ty: ignore[invalid-assignment]
    key = (row["site"], with_pseudo)
    if key not in cache:
        rows = fd._training_rows_cached(row=row, arrays=arrays, config=fd.CONFIGS["n1h"])
        pseudo = (
            cached_pseudo_mask(site=row["site"], rows=rows, durations=durations)
            if with_pseudo
            else None
        )
        cache[key] = with_event_flags(frame=rows, event=masks[row["site"]], pseudo=pseudo)
    return cache[key]


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
    arm = ARMS[arm_name]
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    masks, _ = worker_state()
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
            rows=flagged_training_rows(row=row, arrays=arrays[site], with_pseudo=arm.pseudo),
            refit=refit,
            config=config,
        )
        if arm.drop_event_targets:
            known = known.filter(~pl.col("in_event"))
        if arm.mask_lags:
            known = mask_lags(frame=known, use_pseudo=arm.pseudo)
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
            if arm.mask_lags:
                prepared = mask_lags(
                    frame=with_event_flags(frame=prepared, event=masks[site], pseudo=None),
                    use_pseudo=False,
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
    """Fit the requested arms at the weekly refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=ARM_NAMES)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--refits", type=int, default=None, help="only the first N refits")
    parser.add_argument("--refit-stride", type=int, default=1, help="every k-th refit")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("weekly",), origins=common.all_origins())
    keep = list(jobs.items())[:: args.refit_stride][: args.refits]
    jobs = dict(keep)
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
                arm_name=a,
                refit=r,
                origins=runs,
                sites=sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
            )
            for r, runs in jobs.items()
            for a in args.arms
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({out})")


if __name__ == "__main__":
    main()
