"""Forecast with experiment 389's S1M boosters under the stale-input arms of experiment 410.

S1M is the monthly XGBoost model given the 51 ensemble members' weather plus the ensemble mean and
standard deviation of temperature and shortwave radiation. The script reads the saved boosters
(`.ubj`, one per Primary and monthly refit) and fits nothing. The arms are those of
`e00410_stale_predict.py`: the older run replaces the member weather and the four spread columns,
matched on valid time, while the lead and the derived calendar columns stay those of the fresh
origin; a delayed origin moves the three lags; the XGBoost model reads a missing lag as missing.
The prediction runs on the GPU (a few hundred megabytes).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00389-xgboost-tail-calibration-and-ensemble-spread-columns:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore \
OMP_NUM_THREADS=2 E389_SPREAD=1 systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00410-*/e00410_xgb_predict.py
```
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

os.environ["E389_SPREAD"] = "1"

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00389_fit as f389  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from e00410_stale_predict import (  # ty: ignore[unresolved-import]
    ARMS,
    OUT_DIR,
    SHORT_HISTORY_CUT,
    SHORT_HISTORY_LABELS,
    Arm,
    cleaned_power,
    lag_matrix_with_delay,
)

NETWORK: Final[str] = "xgb_s1m"
BOOSTERS: Final[Path] = f389.BOOSTER_DIR / "s1m"
WEATHER_COLUMNS: Final[list[str]] = [*fd.WEATHER, *f389.SPREAD_COLUMNS]
CONFIG: Final[fd.Config] = fd.CONFIGS["s1m"]


def refit_of_origin() -> dict[datetime, datetime]:
    """Return the refit that forecasts each monthly-regime origin."""
    jobs = fd.selected_jobs(config=CONFIG, regime="monthly", origins=common.all_origins())
    return {origin: refit for refit, runs in jobs.items() for origin in runs}


def load_booster(*, site: str, refit: datetime, device: str) -> xgb.Booster:
    """Return a saved S1M booster, set to predict on `device`."""
    booster = xgb.Booster()
    booster.load_model(BOOSTERS / f"{site}_{refit:%Y%m%dT%H%M}.ubj")
    booster.set_param({"device": device, "nthread": 2})
    return booster


def stale_prepared(
    *,
    fresh: pl.DataFrame,
    run: datetime,
    row: dict,
    arrays: common.SeriesArrays,
    days: int,
) -> pl.DataFrame | None:
    """Return the fresh rows with the member weather and spread columns of an older run."""
    path = common.demand_frame_path(run=run - timedelta(days=days))
    if not path.exists():
        return None
    try:
        older = fd.prepared_run(
            run_rows=pl.read_parquet(path).filter(
                pl.col("time_series_id") == row["time_series_id"]
            ),
            row=row,
            arrays=arrays,
            a7=False,
        )
    except ValueError:
        return None
    older = older.select("valid_time", "ensemble_member", *WEATHER_COLUMNS)
    return fresh.drop(WEATHER_COLUMNS).join(
        older,
        on=["valid_time", "ensemble_member"],
        how="inner",
        validate="1:1",
        maintain_order="left",
    )


def arm_frame(
    *, arm: Arm, fresh: pl.DataFrame, older: dict[int, pl.DataFrame | None], power: np.ndarray
) -> pl.DataFrame | None:
    """Return the prediction rows of an arm: weather of the arm's run, lags of the arm's origin."""
    frame = older[arm.weather_days] if arm.weather_days else fresh
    if frame is None:
        return None
    lags = lag_matrix_with_delay(
        power=power,
        valid=frame["valid_time"],
        origin=frame["power_fcst_init_time"],
        delay_steps=2 * arm.power_hours,
    )
    if arm.drop_lags:
        lags = np.full_like(lags, np.nan)
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def run_month(*, month: str, arms: list[Arm], device: str, smoke: bool) -> None:
    """Forecast one test month under every arm with the saved boosters."""
    series, arrays = fd._series_and_arrays()
    primaries = prep.primaries()
    clean = cleaned_power(
        arrays=arrays, series=series.filter(pl.col("substation_type") == "Primary")
    )
    refits = refit_of_origin()
    pieces: dict[str, list[pl.DataFrame]] = {}
    skipped: dict[str, int] = {}
    origins = prep.test_origins(month=month)
    boosters: dict[tuple[str, datetime], xgb.Booster] = {}
    for origin in origins[:2] if smoke else origins:
        run = origin - common.PUBLICATION_DELAY
        frame = pl.read_parquet(common.demand_frame_path(run=run))
        for row in primaries.iter_rows(named=True):
            site = row["site"]
            key = (site, refits[origin])
            if key not in boosters:
                boosters[key] = load_booster(site=site, refit=refits[origin], device=device)
            fresh = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            ).sort("valid_time", "ensemble_member")
            older = {
                days: stale_prepared(fresh=fresh, run=run, row=row, arrays=arrays[site], days=days)
                for days in sorted({a.weather_days for a in arms if a.weather_days})
            }
            for arm in arms:
                power = arrays[site].power if arm.raw_power else clean[site]
                if site in SHORT_HISTORY_LABELS and not arm.raw_power:
                    power = power.copy()
                    power[: int(common.grid_index(times=SHORT_HISTORY_CUT))] = np.nan
                prepared = arm_frame(arm=arm, fresh=fresh, older=older, power=power)
                if prepared is None:
                    skipped[arm.name] = skipped.get(arm.name, 0) + 1
                    continue
                matrix = fd.predict_inputs(
                    prepared=prepared, row=row, level=None, basis="vintage", config=CONFIG
                )
                table = fd.mixture_frame(
                    keys=prepared.select(*common.KEYS).unique(maintain_order=True),
                    raw=boosters[key].predict(matrix),
                    members=prepared["ensemble_member"].n_unique(),
                    seed=0,
                ).drop("seed")
                pieces.setdefault(arm.name, []).append(table)
    print(f"{month}: skipped (origin, Primary) pairs per arm: {skipped}", flush=True)
    for arm_name, parts in pieces.items():
        suffix = "_smoke" if smoke else ""
        target = OUT_DIR / "preds" / NETWORK / arm_name / f"{month}_s0{suffix}.parquet"
        common.atomic_write(frame=pl.concat(parts), target=target)


def main() -> None:
    """Forecast the requested months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--arms", nargs="+", default=[a.name for a in ARMS])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="two origins, one month")
    args = parser.parse_args()
    arms = [a for a in ARMS if a.name in args.arms]
    for month in args.months[:1] if args.smoke else args.months:
        started = time.monotonic()
        run_month(month=month, arms=arms, device=args.device, smoke=args.smoke)
        print(f"{month}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
