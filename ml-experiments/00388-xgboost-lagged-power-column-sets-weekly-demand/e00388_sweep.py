"""Lag-encroachment sweep for experiment 388: how far does the forecast follow a step in the lags?

Prediction-only. `fit` refits the baseline and arms a, b, and c once, at the latest weekly refit of
the scored year (2026-06-23), for four Primaries (`SWEEP_SITES`: 16 fits), and saves the boosters to
`work/sweep_models/<arm>/<site>.ubj` (3 single-thread GPU worker processes). `run` loads the
boosters and forecasts the runs that refit serves, once with the inputs as they are (k = 0) and once
with a synthetic step of depth D added to the lagged-power inputs:

- k = 1: only the most recent lag reading (the newest usable observation);
- k = 2: the two most recent distinct lag readings;
- k = all: every lag reading.

A column is shifted by D (a fraction of capacity) when the reading it reads lies inside the step,
that is at or after the k-th most recent distinct reading the arm's columns use. The weekly mean
and median are recomputed from the shifted weekly analogues. An anomaly column shifts when the
reading it subtracts from lies inside the step; the 4-week median it subtracts is not shifted.

For each arm, lead band, k, and D it prints: the median's shift as a share of D (0 keeps the
normal running arrangement, 1 follows the step level); the width of the 1-99% band against k = 0;
the share of step-level readings (the k = 0 median plus D) inside the 1-99% and 5-95% bands; the
share of rows whose q95 and q99 (q05 and q01 for a negative step) reach the step level; and the
CRPS against the observed power minus the CRPS at k = 0 (the price of a false alarm).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00388-*/e00388_sweep.py fit
```
"""

import argparse
import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00388_fit as lagfit
import numpy as np
import polars as pl
import xgboost as xgb

SWEEP_REFIT: Final[datetime] = datetime(2026, 6, 23, tzinfo=UTC)
"""The latest weekly refit of the scored year, fixed in the plan before any score was read."""
ARMS: Final[tuple[str, ...]] = ("base", "a", "b", "c")
SWEEP_SITES: Final[tuple[str, ...]] = ("D8", "D4", "D10", "D5")
"""The four Primaries of the screen: D8 (the target), D4 (a placebo), and D10 and D5 (signed-MW)."""
DEPTHS: Final[tuple[float, ...]] = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)
"""Step depths as fractions of capacity."""
KS: Final[tuple[str, ...]] = ("1", "2", "all")
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-24 h": (0.0, 24.0),
    "24-72 h": (24.0, 72.0),
    "72-168 h": (72.0, 168.0),
    "168-240 h": (168.0, 240.0),
}
MAX_LEAD_HOURS: Final[float] = 240.0
"""The sweep covers leads to day 10, the headline range."""
MODEL_DIR: Final[Path] = lagfit.WORK_DIR / "sweep_models"
Q: Final[dict[str, int]] = {name: i for i, name in enumerate(common.LEVEL_COLUMNS)}


def fit_job(*, arm: str, site: str) -> str:
    """Fit one arm for one series at the sweep refit and save the booster."""
    target = MODEL_DIR / arm / f"{site}.ubj"
    if target.exists():
        return f"{arm} {site}: exists"
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    known = fd.known_rows(
        rows=lagfit.add_lag_columns(
            frame=fd._training_rows_cached(row=row, arrays=arrays[site], config=config),
            power=arrays[site].power,
            arm=arm,
        ),
        refit=SWEEP_REFIT,
        config=config,
    )
    age_days = ((SWEEP_REFIT - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    matrix = xgb.DMatrix(
        known.select(lagfit.columns_of(arm=arm)).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / config.half_life_days),
    )
    booster = xgb.train(
        fd.parameters(config=config, seed=common.SEED, nthread=1),
        matrix,
        num_boost_round=common.N_ROUNDS,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(target.with_suffix(".tmp.ubj"))
    target.with_suffix(".tmp.ubj").rename(target)
    return f"{arm} {site}: fitted on {known.height:,} rows"


def reading_indices(*, target: np.ndarray, origin: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    """Return, per lag reading an arm uses, the grid index of that reading (-1 where unusable).

    The readings are the three origin lags, plus the six weekly analogues for arms a and b (arm b
    turns them into a mean and a median), plus the origin and day-ago readings of arm c.
    """
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    day = target - common.STEPS_PER_DAY * np.ceil(gap / common.STEPS_PER_DAY).astype(int)
    week = target - 7 * common.STEPS_PER_DAY * np.ceil(gap / (7 * common.STEPS_PER_DAY)).astype(int)
    out = {
        "lag_origin": newest,
        "lag_same_time_latest_day": day,
        "lag_same_time_latest_week": week,
    }
    if arm in ("a", "b"):
        for w in lagfit.ANALOGUE_WEEKS:
            index = target - w * lagfit.WEEK_STEPS
            out[f"analogue_{w}w"] = np.where(index <= newest, index, -1)
    if arm == "c":
        day_ago = target - common.STEPS_PER_DAY
        out["origin_anomaly"] = newest
        out["day_ago_anomaly"] = np.where(day_ago <= newest, day_ago, -1)
    return out


def step_floor(*, readings: dict[str, np.ndarray], k: str) -> np.ndarray:
    """Return, per case, the oldest grid index inside the step of `k` ('1', '2', or 'all')."""
    stacked = np.stack(list(readings.values()), axis=1).astype(float)
    stacked[stacked < 0] = np.nan
    if k == "all":
        return np.nanmin(stacked, axis=1)
    rank = int(k) - 1
    floor = np.empty(stacked.shape[0])
    for i, row in enumerate(stacked):
        distinct = np.unique(row[~np.isnan(row)])[::-1]
        floor[i] = distinct[min(rank, len(distinct) - 1)]
    return floor


def shifted_matrix(
    *, frame: pl.DataFrame, power: np.ndarray, arm: str, depth: float, k: str
) -> np.ndarray:
    """Return the arm's input matrix with a step of `depth` added to the lags inside the step."""
    columns = lagfit.columns_of(arm=arm)
    matrix = frame.select(columns).to_numpy().astype(np.float32).copy()
    if k == "0":
        return matrix
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    readings = reading_indices(target=target, origin=origin, arm=arm)
    floor = step_floor(readings=readings, k=k)
    inside = {name: (index >= floor) & (index >= 0) for name, index in readings.items()}
    for name, mask in inside.items():
        if name in columns:
            matrix[:, columns.index(name)] += np.where(mask, depth, 0.0).astype(np.float32)
    if arm == "b":
        values = np.stack(
            [
                np.where(
                    readings[f"analogue_{w}w"] >= 0,
                    common.take(values=power, index=readings[f"analogue_{w}w"])
                    + np.where(inside[f"analogue_{w}w"], depth, 0.0),
                    np.nan,
                )
                for w in lagfit.ANALOGUE_WEEKS
            ],
            axis=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            matrix[:, columns.index("weekly_mean")] = np.nanmean(values, axis=1)
            matrix[:, columns.index("weekly_median")] = np.nanmedian(values, axis=1)
    return matrix


def crps(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the quantile-score CRPS of each row (fractions of capacity)."""
    diff = actual[:, None] - quantiles
    levels = np.asarray(common.LEVELS, dtype=np.float64)
    pinball = np.maximum(levels * diff, (levels - 1.0) * diff)
    return 2.0 * pinball.mean(axis=1)


def sweep_one(
    *, arm: str, site: str, row: dict, arrays: common.SeriesArrays, origins: list
) -> list:
    """Return one series' per-row sweep records for one arm (a list of frames)."""
    booster = xgb.Booster()
    booster.load_model(MODEL_DIR / arm / f"{site}.ubj")
    booster.set_param({"nthread": 1, "device": "cuda"})
    records = []
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays,
            a7=False,
        )
        prepared = lagfit.add_lag_columns(frame=prepared, power=arrays.power, arm=arm)
        prepared = prepared.filter(pl.col("lead_hours") <= MAX_LEAD_HOURS)
        keys = prepared.select(*common.KEYS).unique(maintain_order=True)
        members = prepared["ensemble_member"].n_unique()
        actual = common.take(
            values=arrays.power, index=np.asarray(common.grid_index(times=keys["valid_time"]))
        ).astype(np.float64)
        lead = (
            (keys["valid_time"] - keys["power_fcst_init_time"]).dt.total_minutes() / 60.0
        ).to_numpy()
        base_q = None
        for k, depth in [("0", 0.0), *[(k, d) for k in KS for d in DEPTHS]]:
            matrix = shifted_matrix(frame=prepared, power=arrays.power, arm=arm, depth=depth, k=k)
            raw = booster.predict(xgb.DMatrix(matrix))
            quantiles = (
                fd.mixture_frame(keys=keys, raw=raw, members=members, seed=0)
                .select(list(common.LEVEL_COLUMNS))
                .to_numpy()
                .astype(np.float64)
            )
            if k == "0":
                base_q = quantiles
            records.append(
                {
                    "arm": arm,
                    "site": site,
                    "k": k,
                    "depth": depth,
                    "lead": lead,
                    "quantiles": quantiles,
                    "base": base_q,
                    "actual": actual,
                }
            )
    return records


def summarise(*, records: list) -> pl.DataFrame:
    """Return the sweep measures per arm, step size, k, and lead band, pooled over series."""
    out = []
    for rec in records:
        if rec["k"] == "0":
            continue
        depth, q, q0 = rec["depth"], rec["quantiles"], rec["base"]
        median, median0 = q[:, Q["q50"]], q0[:, Q["q50"]]
        step_level = median0 + depth
        reach95 = q[:, Q["q95"]] >= step_level if depth > 0 else q[:, Q["q05"]] <= step_level
        reach99 = q[:, Q["q99"]] >= step_level if depth > 0 else q[:, Q["q01"]] <= step_level
        valid = np.isfinite(rec["actual"])
        price = np.where(
            valid,
            crps(quantiles=q, actual=np.nan_to_num(rec["actual"]))
            - crps(quantiles=q0, actual=np.nan_to_num(rec["actual"])),
            np.nan,
        )
        for band, (low, high) in BANDS.items():
            m = (rec["lead"] > low) & (rec["lead"] <= high)
            if not m.any():
                continue
            out.append(
                {
                    "arm": rec["arm"],
                    "k": rec["k"],
                    "depth_pct": 100 * depth,
                    "band": band,
                    "n": int(m.sum()),
                    "shift_share": float(((median - median0)[m] / depth).mean()),
                    "width_ratio": float(
                        (q[m, Q["q99"]] - q[m, Q["q01"]]).mean()
                        / (q0[m, Q["q99"]] - q0[m, Q["q01"]]).mean()
                    ),
                    "in_1_99": float(
                        (
                            (step_level[m] >= q[m, Q["q01"]]) & (step_level[m] <= q[m, Q["q99"]])
                        ).mean()
                    ),
                    "in_5_95": float(
                        (
                            (step_level[m] >= q[m, Q["q05"]]) & (step_level[m] <= q[m, Q["q95"]])
                        ).mean()
                    ),
                    "q95_reaches": float(reach95[m].mean()),
                    "q99_reaches": float(reach99[m].mean()),
                    "crps_price_pts": float(np.nanmean(price[m]) * 100),
                }
            )
    return pl.DataFrame(out)


def run_sweep(*, series: pl.DataFrame, sites: list[str]) -> None:
    """Forecast with and without the step and print the tables."""
    series_all, arrays = fd._series_and_arrays()
    rows_of = {r["site"]: r for r in series_all.iter_rows(named=True)}
    jobs = regimes.jobs_for(regime_names=("weekly",), origins=common.all_origins())
    origins = jobs[SWEEP_REFIT]
    print(f"refit {SWEEP_REFIT:%Y-%m-%d}: {len(origins)} runs, {len(sites)} Primaries")
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(220)
    for arm in ARMS:
        records = []
        for site in sites:
            records += sweep_one(
                arm=arm, site=site, row=rows_of[site], arrays=arrays[site], origins=origins
            )
        table = summarise(records=records)
        print(f"\n===== arm {arm} =====", flush=True)
        print(
            table.group_by("k", "depth_pct", "band", maintain_order=True).agg(
                pl.col("n").sum(),
                *[
                    (pl.col(c) * pl.col("n")).sum() / pl.col("n").sum()
                    for c in (
                        "shift_share",
                        "width_ratio",
                        "in_1_99",
                        "in_5_95",
                        "q95_reaches",
                        "q99_reaches",
                        "crps_price_pts",
                    )
                ],
            )
        )


def main() -> None:
    """Fit the sweep boosters, or run the sweep and print its tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "run"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    series = common.series_table()
    sites = [s for s in series["site"].to_list() if s in SWEEP_SITES]
    if args.mode == "fit":
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(fit_job, arm=a, site=s) for a in ARMS for s in sites]
            for done, future in enumerate(futures, start=1):
                print(f"[{done}/{len(futures)}] {future.result()}", flush=True)
        return
    run_sweep(series=series, sites=sites)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
