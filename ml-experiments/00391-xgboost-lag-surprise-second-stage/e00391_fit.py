"""Stage 2 of experiment 391: the monthly XGBoost member with surprise and detector columns.

Each arm refits experiment 87's `n1h` configuration (the BG2 weekly member's recipe) at the
monthly refit times (`--regime monthly`, the default; `--regime weekly` for the reproduction gate),
with its training rows, parameters, and leakage checks (its `e00087_fit_demand` module), and adds
columns built from the stage 1 tables (`e00391_stage1.py`) and the detector (`e00391_features.py`).
`colsample_bytree` and `colsample_bynode` stay at 1.0, the seed is 0, and no arm widens its bands.
It writes the 11 quantiles per daily run to `work/arm_preds/<arm>/<site>_s0_<refit>.parquet`.

Arms (`ARMS`): `b0` (no extra columns, the control), `n0` (own surprise), `n1` (own plus neighbour
surprise), `n2` (n1 plus the detector's open-alarm state, with the neighbour veto), `n2_noveto`,
`n3` (detector state only), `o` (hand-label flags, the oracle ceiling), and `n1_era5` (n1 built
from the ERA5-fed stage 1, a hindsight ceiling).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00391-*/e00391_fit.py \
    --arms b0 n0 n1 --workers 3 --cores 3
```
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00391_features as feats
import e00391_neighbours as neighbours
import e00391_stage1 as stage1
import numpy as np
import polars as pl
import xgboost as xgb
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = stage1.WORK_DIR
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
BOOSTER_DIR: Final[Path] = WORK_DIR / "boosters"
BOOSTER_MONTHS: Final[tuple[int, ...]] = tuple(range(1, 13))
"""Boosters are saved at every monthly refit for the prediction-only sweep."""
K_PATH: Final[Path] = WORK_DIR / "detector" / "k.json"
ROUNDS: Final[int] = 150
ETA: Final[float] = 0.15
"""The screening recipe of the common brief, used by every arm; experiment 381 used 500 and 0.05."""
DEVICE: Final[str] = os.environ.get("E391_DEVICE", "cuda")
SUBSET_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D8")
"""The signed-MW six and D8: the Primaries where the secondary arms run (manager, 2026-10-29)."""
SMOKE_ROUNDS: Final[int] = 20
OWN_COLUMNS: Final[tuple[str, ...]] = ("u_lag_origin", "u_lag_day", "u_lag_week", "z_own")
NEIGHBOUR_COLUMNS: Final[tuple[str, ...]] = ("nb_max_abs_zt", "nb_mean_zt", "nb_opposite_count")
DETECTOR_COLUMNS: Final[tuple[str, ...]] = (
    "det_open", "det_sign", "det_onset_days", "det_zt", "det_r_mw", "det_tag",
    "det_nb_opposite_open",
)  # fmt: skip
ORACLE_COLUMNS: Final[tuple[str, ...]] = (
    "oracle_target", "oracle_lag_origin", "oracle_lag_day", "oracle_lag_week",
)  # fmt: skip
OPPOSITE_ZT: Final[float] = 2.0


@dataclass(frozen=True)
class Arm:
    """An arm's extra columns, its stage 1 version, and whether the detector uses the veto."""

    name: str
    groups: tuple[str, ...]
    version: str = "ens"
    veto: bool = True
    subset: bool = False

    def columns(self) -> list[str]:
        """Return the input columns in matrix order: the baseline's, then the arm's extras."""
        extra = {
            "own": OWN_COLUMNS, "neighbour": NEIGHBOUR_COLUMNS, "detector": DETECTOR_COLUMNS,
            "oracle": ORACLE_COLUMNS,
        }  # fmt: skip
        return [*fd.input_columns(a7=False), *(c for g in self.groups for c in extra[g])]


ARMS: Final[dict[str, Arm]] = {
    "b0": Arm("b0", ()),
    "n0": Arm("n0", ("own",), subset=True),
    "n1": Arm("n1", ("own", "neighbour")),
    "n2": Arm("n2", ("own", "neighbour", "detector")),
    "n2_noveto": Arm("n2_noveto", ("own", "neighbour", "detector"), veto=False, subset=True),
    "n3": Arm("n3", ("detector",), subset=True),
    "o": Arm("o", ("oracle",)),
    "n1_era5": Arm("n1_era5", ("own", "neighbour"), version="era5", subset=True),
}


class Context:
    """Per-process stage 1 tables, detector statistics, alarm states, and neighbour lists."""

    def __init__(self, *, version: str, veto: bool) -> None:
        """Load a version's stage 1 tables and, if `work/detector/k.json` exists, its alarms."""
        series = common.series_table()
        self.sites = series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
        size = common.grid_times().len()
        self.u: dict[str, np.ndarray] = {}
        stats: dict[str, feats.Origins] = {}
        for site in self.sites:
            table = pl.read_parquet(stage1.STAGE1_DIR / version / f"{site}.parquet")
            index = np.asarray(common.grid_index(times=table["valid_time"]))
            u = np.full(size, np.nan, np.float32)
            r = np.full(size, np.nan, np.float32)
            u[index], r[index] = table["u"].to_numpy(), table["r_mw"].to_numpy()
            self.u[site] = u
            stats[site] = feats.origin_statistics(u=u, r=r)
        self.stats = stats
        self.nged = {
            s: tuple(n) for s, n in neighbours.nged_neighbours()[0].items() if s in self.sites
        }
        self.k: float | None = None
        self.alarms: dict[str, dict[str, np.ndarray]] | None = None
        if K_PATH.exists():
            self.k = float(json.loads(K_PATH.read_text())[version])
            self.alarms = feats.alarm_state(stats=stats, k=self.k, veto=veto)


def add_columns(
    *, frame: pl.DataFrame, site: str, arm: Arm, ctx: Context, mask: np.ndarray
) -> pl.DataFrame:
    """Add the arm's extra columns to a frame with `valid_time` and `power_fcst_init_time`.

    Every column reads grid positions at or before the origin minus the telemetry delay
    (`common.NEWEST_AGE` half-hours): the three lags, the window of 48 half-hours ending there, and
    the latest 6-hourly detector origin whose window ends there.
    """
    if not arm.groups:
        return frame
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    newest = origin - common.NEWEST_AGE
    lag_index = common.origin_lags(
        power=np.arange(ctx.u[site].size, dtype=np.float32), target=target, origin=origin
    ).astype(np.int64)
    columns: dict[str, np.ndarray] = {}
    j = (newest + 1) // feats.STEP_PER_ORIGIN
    if "own" in arm.groups:
        for i, name in enumerate(OWN_COLUMNS[:3]):
            columns[name] = ctx.u[site][lag_index[:, i]]
        columns["z_own"] = feats.window_means(values=ctx.u[site], origins=newest + 1)
    if "neighbour" in arm.groups:
        own_zt = ctx.stats[site].z_noise[j]
        zts = (
            np.stack([ctx.stats[n].z_noise[j] for n in ctx.nged.get(site, ())], axis=1)
            if ctx.nged.get(site)
            else np.full((len(j), 0), np.nan)
        )
        with np.errstate(all="ignore"):
            has = np.isfinite(zts).any(axis=1)
            columns["nb_max_abs_zt"] = np.where(
                has, np.nanmax(np.abs(zts), axis=1, initial=0), np.nan
            )
            columns["nb_mean_zt"] = np.where(
                has, np.nanmean(np.where(np.isfinite(zts), zts, np.nan), axis=1), np.nan
            )
        opposite = (np.sign(zts) == -np.sign(own_zt)[:, None]) & (np.abs(zts) > OPPOSITE_ZT)
        columns["nb_opposite_count"] = np.where(
            np.isfinite(own_zt) & has, opposite.sum(axis=1), np.nan
        )
    if "detector" in arm.groups:
        if ctx.alarms is None:
            msg = "detector columns need work/detector/k.json from e00391_detector.py"
            raise FileNotFoundError(msg)
        state = ctx.alarms[site]
        columns["det_open"] = state["open"][j].astype(np.float64)
        columns["det_sign"] = state["sign"][j].astype(np.float64)
        onset = state["onset"][j]
        columns["det_onset_days"] = np.where(state["open"][j] == 1, (newest - onset) / 48.0, np.nan)
        columns["det_zt"] = ctx.stats[site].z_noise[j]
        columns["det_r_mw"] = ctx.stats[site].r[j]
        columns["det_tag"] = state["tag"][j].astype(np.float64)
        columns["det_nb_opposite_open"] = np.array(
            [
                sum(
                    1
                    for n in ctx.nged.get(site, ())
                    if ctx.alarms[n]["open"][jj] == 1
                    and ctx.alarms[n]["sign"][jj] == -ctx.alarms[site]["sign"][jj]
                )
                if state["open"][jj] == 1
                else np.nan
                for jj in j
            ]
        )
    if "oracle" in arm.groups:
        columns["oracle_target"] = mask[target].astype(np.float64)
        flags = lag_flags(mask=mask, target=target, origin=origin)
        for i, name in enumerate(ORACLE_COLUMNS[1:]):
            columns[name] = flags[:, i].astype(np.float64)
    return frame.hstack(
        pl.DataFrame({k: pl.Series(v).cast(pl.Float32) for k, v in columns.items()})
    )


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


def lag_flags(*, mask: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return, per row and lag, whether the lag's time lies inside the mask."""
    return common.origin_lags(power=mask.astype(np.float32), target=target, origin=origin) > 0.5


_STATE: dict[str, object] = {}


def worker_state(*, arm: Arm) -> tuple[Context, dict[str, np.ndarray]]:
    """Return the arm's `Context` and the event masks, built once per worker process."""
    key = (arm.version, arm.veto)
    contexts: dict[tuple[str, bool], Context] = _STATE.setdefault("ctx", {})  # ty: ignore[invalid-assignment]
    if key not in contexts:
        contexts[key] = Context(version=arm.version, veto=arm.veto)
    if "masks" not in _STATE:
        _STATE["masks"] = event_masks()
    return contexts[key], _STATE["masks"]  # ty: ignore[invalid-return-type]


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
    ctx, masks = worker_state(arm=arm)
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not arm.subset or s in SUBSET_SITES
        if not fd.job_path(root=out_dir, config=arm.name, site=s, seed=seed, refit=refit).exists()
    ]
    lines, models = [], {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        known = fd.known_rows(
            rows=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
            refit=refit,
            config=config,
        )
        known = add_columns(frame=known, site=site, arm=arm, ctx=ctx, mask=masks[site])
        models[site] = xgb.train(
            {
                **fd.parameters(config=config, seed=seed, nthread=nthread),
                "device": DEVICE,
                "eta": ETA,
            },
            train_matrix(known=known, row=row, refit=refit, arm=arm),
            num_boost_round=rounds,
        )
        lines.append(
            f"{arm.name} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
            f"fit {time.monotonic() - started:.0f} s"
        )
    if refit.month in BOOSTER_MONTHS:
        for site, model in models.items():
            booster_dir = BOOSTER_DIR / arm.name
            booster_dir.mkdir(parents=True, exist_ok=True)
            temporary = booster_dir / f"{site}_{refit:%Y%m%d}.tmp.ubj"
            model.save_model(temporary)
            temporary.replace(booster_dir / f"{site}_{refit:%Y%m%d}.ubj")
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
            prepared = add_columns(frame=prepared, site=site, arm=arm, ctx=ctx, mask=masks[site])
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
    """Fit the requested arms at the refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=tuple(ARMS))
    parser.add_argument("--regime", default="monthly", choices=("monthly", "weekly"))
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
    jobs = regimes.jobs_for(regime_names=(args.regime,), origins=common.all_origins())
    jobs = dict(list(jobs.items())[:: args.refit_stride][: args.refits])
    rounds, out = ROUNDS, ARM_DIR / args.regime
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds, out = SMOKE_ROUNDS, WORK_DIR / "arm_preds_smoke"
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
