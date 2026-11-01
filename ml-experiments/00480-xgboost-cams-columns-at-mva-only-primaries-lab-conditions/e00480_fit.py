"""Refit experiment 381's monthly control with and without two CAMS columns (experiment 480).

Two arms, both on CPU, both refitted monthly at the 16 Primaries with experiment 87's `n1h`
recipe (as experiment 381's `monthly/control`):

- `a0`: the control, input columns of experiment 87 only.
- `a1`: `a0` plus `cams_unit_pv_shape` and `cams_clear_sky_index`, the half-hour's own realised
  values from the CAMS radiation service's satellite retrieval. The unit PV shape is the output of
  the generic south-facing fleet (`studies.dp_features.FLEET`) under the half-hour's irradiance.
  The clear-sky index is the half-hour's irradiance over the Haurwitz clear-sky irradiance at its
  midpoint, 0 where the clear-sky irradiance is below 1 W/m2, and capped at 1.5. Both columns are
  the same for every ensemble member and read nothing from the observed power.

Differences from experiment 381's fit script, all deliberate: `device` is `cpu`; the observed power
of D15 and D18 is set to missing before 2025-03-01 (training rows and lags both lose it); each
fitted booster is saved as `.ubj`. Everything else (rows, weights, parameters, leakage checks,
mixture of the 51 members) is experiment 381's and experiment 87's code.

```bash
PYTHONPATH=ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00480-*/e00480_fit.py \
    --arms a0 a1 --workers 3 --smoke
```
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00381_fit as fit381  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.baselines import haurwitz_w_m2
from studies.dp_features import CAMS_PATH, FLEET
from studies.pv_physics import half_hour_sun_and_sky, plant_power_mw
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions"
)
ARM_DIR: Final[Path] = WORK_DIR / "arm_preds" / "monthly"
MODEL_DIR: Final[Path] = WORK_DIR / "models"
CAMS_COLUMNS: Final[tuple[str, ...]] = ("cams_unit_pv_shape", "cams_clear_sky_index")
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
SHORT_HISTORY_SITES: Final[tuple[str, ...]] = ("D15", "D18")
MIN_CLEAR_SKY_W_M2: Final[float] = 1.0
MAX_CLEAR_SKY_INDEX: Final[float] = 1.5
"""Cap on the index: near sunrise and sunset the ratio of two small irradiances is unbounded."""
ARM_NAMES: Final[tuple[str, ...]] = ("a0", "a1")


@dataclass(frozen=True)
class Arm:
    """One arm: whether it adds the two CAMS columns."""

    name: str
    cams: bool

    # Attributes experiment 381's `train_matrix` and `forecast_run` read from an arm.
    lead_weights: bool = False
    half_life_years: float | None = None

    def columns(self) -> list[str]:
        """Return all input columns in matrix order."""
        return [*fd.input_columns(a7=False), *(CAMS_COLUMNS if self.cams else ())]


def make_arm(*, name: str) -> Arm:
    """Return the arm called `name`."""
    if name not in ARM_NAMES:
        msg = f"unknown arm {name!r}; expected one of {ARM_NAMES}"
        raise ValueError(msg)
    return Arm(name=name, cams=name == "a1")


def cams_columns(*, site: str, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the two CAMS columns for every half-hour of a site, keyed by `valid_time`.

    `valid_time` is the half-hour end stamp, the convention of the training frames. The hour's
    irradiance is shared between its two half-hours by `half_hour_sun_and_sky`.
    """
    hourly = (
        pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2"])
        .filter(pl.col("site") == site)
        .select("time", pl.col("ghi_w_m2").cast(pl.Float64))
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=latitude, longitude=longitude)
    clear = haurwitz_w_m2(apparent_zenith_deg=sky.zenith_deg)
    index = np.divide(
        sky.ghi_w_m2, clear, out=np.zeros_like(clear), where=clear >= MIN_CLEAR_SKY_W_M2
    )
    return pl.DataFrame(
        {
            "valid_time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "cams_unit_pv_shape": plant_power_mw(sky=sky, parameters=FLEET),
            "cams_clear_sky_index": np.clip(index, 0.0, MAX_CLEAR_SKY_INDEX),
        }
    ).with_columns(pl.col(*CAMS_COLUMNS).cast(pl.Float32))


def add_cams(*, frame: pl.DataFrame, columns: pl.DataFrame) -> pl.DataFrame:
    """Join the CAMS columns onto a frame by valid time; a missing hour gives a null."""
    return frame.join(columns, on="valid_time", how="left", validate="m:1", maintain_order="left")


def cut_short_history(*, site: str, arrays: common.SeriesArrays) -> common.SeriesArrays:
    """Set the observed power of D15 and D18 to missing before 2025-03-01 (arrays are copied)."""
    if site not in SHORT_HISTORY_SITES:
        return arrays
    before = (arrays.valid_time < SHORT_HISTORY_CUT).to_numpy()
    power = arrays.power.copy()
    power[before] = np.nan
    return replace(arrays, power=power)


def training_frame(*, row: dict, arrays: common.SeriesArrays, arm: Arm) -> pl.DataFrame:
    """Return a series' training rows with the cut applied and the arm's columns added."""
    rows = fd._training_rows_cached(row=row, arrays=arrays, config=fd.CONFIGS["n1h"])
    if row["site"] in SHORT_HISTORY_SITES:
        rows = rows.filter(pl.col("valid_time") >= SHORT_HISTORY_CUT)
    if arm.cams:
        rows = add_cams(
            frame=rows,
            columns=cams_columns(
                site=row["site"], latitude=row["latitude"], longitude=row["longitude"]
            ),
        )
    return rows


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
    model_dir: Path,
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
    lines: list[str] = []
    models: dict[str, xgb.Booster] = {}
    site_cams: dict[str, pl.DataFrame] = {}
    cut_arrays = {s: cut_short_history(site=s, arrays=arrays[s]) for s in todo}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        if arm.cams:
            site_cams[site] = cams_columns(
                site=site, latitude=row["latitude"], longitude=row["longitude"]
            )
        known = fd.known_rows(
            rows=training_frame(row=row, arrays=cut_arrays[site], arm=arm),
            refit=refit,
            config=config,
        )
        params = {**fd.parameters(config=config, seed=seed, nthread=nthread), "device": "cpu"}
        models[site] = xgb.train(
            params,
            fit381.train_matrix(known=known, row=row, refit=refit, arm=arm),  # ty: ignore[invalid-argument-type]
            num_boost_round=rounds,
        )
        model_path = model_dir / arm.name / f"{site}_s{seed}_{refit:%Y%m%dT%H%M}.ubj"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        models[site].save_model(model_path)
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
                arrays=cut_arrays[site],
                a7=False,
            )
            if arm.cams:
                prepared = add_cams(frame=prepared, columns=site_cams[site])
            pieces[site].append(
                fit381.forecast_run(model=models[site], prepared=prepared, arm=arm, seed=seed)  # ty: ignore[invalid-argument-type]
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
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="XGBoost threads per worker")
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    args = parser.parse_args()
    for name in args.arms:
        make_arm(name=name)
    series = common.series_table()
    sites = args.sites or series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    rounds, out, models = common.N_ROUNDS, ARM_DIR, MODEL_DIR
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = fd.SMOKE_ROUNDS
        out, models = WORK_DIR / "smoke" / "arm_preds", WORK_DIR / "smoke" / "models"
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
                nthread=args.threads,
                rounds=rounds,
                seed=args.seed,
                out_dir=out,
                model_dir=models,
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
