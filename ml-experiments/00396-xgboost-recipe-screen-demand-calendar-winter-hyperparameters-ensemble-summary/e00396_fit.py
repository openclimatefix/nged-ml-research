"""Fit the arms of experiment 396 on the few-fold screen's test months.

Each fold trains one XGBoost model per Primary and arm on the control member's rows from 2024-04-01
(up to 24 months) before the first day of the test month, with experiment 87's `n1h` recipe
(`e00087_fit_demand`, read only), and forecasts every daily run whose origin falls in the test
month.
The model sees only rows whose valid time is at least 3.5 hours before the fold's refit time and
whose run began at or before it (`fd.known_rows`). Predictions go to
`work/arm_preds/<arm>/<fold>/<site>_s<seed>.parquet` (11 quantiles as fractions of capacity) and
every fitted booster to `work/boosters/<arm>/<fold>/<site>[_band].ubj`. A finished file is never
fitted again, so an interrupted run resumes.

`--dry` builds every training and prediction matrix of the first origin and prints their shapes and
missing-value counts, without fitting or predicting. Run it before asking for a fit slot.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
    ml-experiments/00396-* \
    DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00396-*/e00396_fit.py \
    --arms C0 C1 --folds 2026-01 2026-06 --workers 3
```
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00396_arms as arms  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00396-xgboost-recipe-screen-demand-calendar-winter-hyperparameters-ensemble-summary"
    / ("work_smoke" if os.environ.get("E396_SMOKE") else "work")
)
"""`E396_SMOKE=1` redirects every output to `work_smoke/`, which a smoke run may overwrite."""
PRED_DIR: Final[Path] = WORK / "arm_preds"
BOOSTER_DIR: Final[Path] = WORK / "boosters"
TRAINING_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The control-member training rows start here (experiment 87's ENS archive)."""
WINDOW_MONTHS: Final[int] = 24
FOLDS: Final[tuple[str, ...]] = ("2026-01", "2026-06", "2025-12", "2026-04", "2025-10", "2026-03")
"""Round-1 base folds first, then the event folds, then the round-2 folds."""
MAX_LEAD_HOURS: Final[float] = 336.0
"""Longer leads are not scored, and the trailing-temperature tables stop here."""
SMOKE_ROUNDS: Final[int] = 20
SMOKE_ORIGINS: Final[int] = 3
FIT_ONLY_ONCE_MEMBER: Final[int] = 0
"""The control member, whose values the one-pass arms keep for the variables with no summary."""


def refit_of(*, fold: str) -> datetime:
    """Return the fold's refit time: 00:00 UTC on the first day of its test month."""
    year, month = (int(part) for part in fold.split("-"))
    return datetime(year, month, 1, tzinfo=UTC)


def window_start(*, refit: datetime) -> datetime:
    """Return the oldest valid time the fold trains on: 24 months back, never before 2024-04-01."""
    months = refit.year * 12 + refit.month - 1 - WINDOW_MONTHS
    start = datetime(months // 12, months % 12 + 1, 1, tzinfo=UTC)
    return max(start, TRAINING_START)


def origins_of(*, fold: str) -> list[datetime]:
    """Return the daily runs' origins that fall in the fold's test month."""
    year, month = (int(part) for part in fold.split("-"))
    return [o for o in common.all_origins() if (o.year, o.month) == (year, month)]


def resolve_arms(*, names: list[str]) -> list[arms.Arm]:
    """Return the arms for a list of names or tier letters (`A`, `C`, `D`), keeping the order."""
    chosen: list[arms.Arm] = []
    for name in names:
        matched = [a for a in arms.ARMS.values() if name in (a.name, a.tier)]
        if not matched:
            msg = f"unknown arm or tier {name!r}"
            raise ValueError(msg)
        chosen += [a for a in matched if a not in chosen]
    return chosen


def input_columns(*, arm: arms.Arm) -> list[str]:
    """Return the arm's matrix columns: the recipe's, less any dropped, plus any added."""
    base = [c for c in fd.input_columns(a7=False) if c not in arm.drop]
    if "summary_temperature_shortwave" in arm.add:
        base = [c for c in base if c not in arms.SUMMARY_REPLACES]
    extra = [*arms.calendar_columns(arm=arm)]
    if "trailing_temperature" in arm.add:
        extra += arms.trailing_columns()
    if "summary_temperature_shortwave" in arm.add:
        extra += list(arms.SUMMARY_COLUMNS)
    return [*base, *extra]


def checked_columns(*, arm: arms.Arm) -> list[str]:
    """Return the added columns that must be finite on every row.

    The recipe's own columns keep their missing values (the origin lags are missing by design and
    XGBoost routes a missing value), and so does `holiday_type`, which is missing on a day with no
    holiday.
    """
    recipe = set(fd.input_columns(a7=False))
    return [c for c in input_columns(arm=arm) if c not in recipe and c != "holiday_type"]


def add_arm_columns(*, frame: pl.DataFrame, arm: arms.Arm, site: str) -> pl.DataFrame:
    """Add every column the arm adds to a frame of training or prediction rows."""
    out = arms.add_calendar(frame=frame, arm=arm)
    if "trailing_temperature" in arm.add:
        out = arms.add_trailing_temperature(frame=out, site=site)
    if "summary_temperature_shortwave" in arm.add:
        out = arms.add_summary(frame=out, site=site)
    return out


def monotone_string(*, arm: arms.Arm, columns: list[str]) -> str | None:
    """Return XGBoost's `monotone_constraints` string for the arm, or `None` without constraints.

    A constraint named `ttemp` applies to every trailing-mean column and `heating` to every degrees
    column; any other name is a matrix column.
    """
    if not arm.monotone:
        return None
    signs = dict.fromkeys(columns, 0)
    for name, sign in arm.monotone:
        for column in columns:
            if column == name or (name in ("ttemp", "heating") and column.startswith(f"{name}_")):
                signs[column] = sign
    return "(" + ",".join(str(signs[c]) for c in columns) + ")"


def booster_parameters(*, arm: arms.Arm, columns: list[str], nthread: int) -> dict[str, object]:
    """Return the XGBoost parameters of an arm: the recipe's, the arm's changes, and constraints."""
    params: dict[str, object] = {
        **common.QUANTILE_PARAMETERS,
        "seed": arm.seed,
        "nthread": nthread,
        "device": os.environ.get("E396_DEVICE", "cuda"),
        **arm.params,
    }
    constraint = monotone_string(arm=arm, columns=columns)
    if constraint is not None:
        params["monotone_constraints"] = constraint
    return params


def training_frame(
    *, row: dict, arrays: common.SeriesArrays, arm: arms.Arm, refit: datetime
) -> pl.DataFrame:
    """Return the control-member rows the fold trains on, with the arm's columns, all finite."""
    config = fd.CONFIGS["n1h"]
    rows = fd._training_rows_cached(row=row, arrays=arrays, config=config)
    known = fd.known_rows(rows=rows, refit=refit, config=config)
    known = known.filter(pl.col("valid_time") >= window_start(refit=refit))
    known = add_arm_columns(frame=known, arm=arm, site=row["site"])
    added = checked_columns(arm=arm)
    kept = (
        known.filter(pl.all_horizontal(pl.col(c).is_not_nan() for c in added)) if added else known
    )
    print(
        f"{row['site']} {refit:%Y-%m} {arm.name}: {known.height:,} rows, "
        f"{known.height - kept.height:,} dropped for a missing added column",
        flush=True,
    )
    if kept.is_empty():
        msg = f"{row['site']} {refit:%Y-%m}: no training rows left for arm {arm.name}"
        raise ValueError(msg)
    return kept


def fit_matrix(*, known: pl.DataFrame, row: dict, arm: arms.Arm, refit: datetime) -> xgb.DMatrix:
    """Return the weighted training matrix, weights anchored at the refit time."""
    half_life_days = (
        arm.half_life_years * 365.25
        if arm.half_life_years is not None
        else common.SHORT_HALF_LIFE_DAYS
    )
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    return xgb.DMatrix(
        known.select(input_columns(arm=arm)).to_numpy().astype(np.float32),
        label=(known["power"] / row["effective_capacity_mw"]).to_numpy(),
        weight=0.5 ** (age_days / half_life_days),
    )


def outputs(*, arm: arms.Arm) -> list[tuple[str, int | None]]:
    """Return `(name, rounds)` for each forecast table an arm writes.

    An arm scored at several numbers of rounds writes one table per number, named
    `<arm>_r<rounds>`; any other arm writes one table under its own name, using every tree.
    """
    if arm.scored_rounds:
        return [(f"{arm.name}_r{k}", k) for k in arm.scored_rounds]
    return [(arm.name, None)]


def output_path(*, name: str, fold: str, site: str, seed: int) -> Path:
    """Return the file holding one output arm's forecasts for one Primary and fold."""
    return PRED_DIR / name / fold / f"{site}_s{seed}.parquet"


def fit_arm(
    *,
    known: pl.DataFrame,
    row: dict,
    arm: arms.Arm,
    refit: datetime,
    fold: str,
    nthread: int,
    rounds: int,
) -> list[xgb.Booster]:
    """Fit the arm's boosters (one, or one per lead band) and save each as a .ubj file."""
    columns = input_columns(arm=arm)
    parameters = booster_parameters(arm=arm, columns=columns, nthread=nthread)
    boosters = []
    bands = arm.bands or (None,)
    for index, band in enumerate(bands):
        rows = known
        if band is not None:
            rows = known.filter(pl.col("lead_hours").is_between(band[0], band[1], closed="right"))
        booster = xgb.train(
            parameters,
            fit_matrix(known=rows, row=row, arm=arm, refit=refit),
            num_boost_round=rounds,
        )
        suffix = "" if band is None else f"_band{index}"
        folder = BOOSTER_DIR / arm.name / fold
        folder.mkdir(parents=True, exist_ok=True)
        booster.save_model(folder / f"{row['site']}{suffix}.ubj")
        boosters.append(booster)
    return boosters


def predict_run(
    *, boosters: list[xgb.Booster], prepared: pl.DataFrame, arm: arms.Arm, iterations: int | None
) -> pl.DataFrame:
    """Return one run's 11 quantiles per valid time for one arm and number of rounds.

    The mixture of the 51 members, or the booster's own quantiles for a one-pass arm.
    """
    if arm.once:
        prepared = prepared.filter(pl.col("ensemble_member") == FIT_ONLY_ONCE_MEMBER)
    members = prepared["ensemble_member"].n_unique()
    columns = input_columns(arm=arm)
    pieces = []
    bands = arm.bands or (None,)
    for booster, band in zip(boosters, bands, strict=True):
        rows = prepared
        if band is not None:
            rows = prepared.filter(
                pl.col("lead_hours").is_between(band[0], band[1], closed="right")
            )
        if rows.is_empty():
            continue
        matrix = xgb.DMatrix(rows.select(columns).to_numpy().astype(np.float32))
        raw = booster.predict(
            matrix,
            iteration_range=(0, min(iterations, booster.num_boosted_rounds()))
            if iterations is not None
            else (0, 0),
        )
        row_keys = rows.select(*common.KEYS).unique(maintain_order=True)
        if arm.once:
            levels = np.sort(raw.reshape(-1, len(common.LEVELS)), axis=1).astype(np.float32)
            pieces.append(
                row_keys.hstack(
                    pl.DataFrame(levels, schema=list(common.LEVEL_COLUMNS))
                ).with_columns(seed=pl.lit(arm.seed).cast(pl.Int8))
            )
        else:
            pieces.append(fd.mixture_frame(keys=row_keys, raw=raw, members=members, seed=arm.seed))
    return pl.concat(pieces).sort(*common.KEYS)


def run_job(
    *,
    fold: str,
    site: str,
    arm_names: list[str],
    nthread: int,
    smoke: bool,
    dry: bool,
) -> list[str]:
    """Fit every requested arm for one Primary and fold that lacks its files, then forecast."""
    refit = refit_of(fold=fold)
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    todo = [
        a
        for a in resolve_arms(names=arm_names)
        if dry
        or not all(
            output_path(name=n, fold=fold, site=site, seed=a.seed).exists()
            for n, _ in outputs(arm=a)
        )
    ]
    lines: list[str] = []
    models: dict[str, list[xgb.Booster]] = {}
    for arm in todo:
        started = time.monotonic()
        known = training_frame(row=row, arrays=arrays[site], arm=arm, refit=refit)
        columns = input_columns(arm=arm)
        lines.append(
            f"{arm.name} {site} {fold}: {known.height:,} training rows from "
            f"{known['valid_time'].min():%Y-%m-%d}, {len(columns)} columns"
        )
        if dry:
            continue
        rounds = SMOKE_ROUNDS if smoke else arm.rounds
        models[arm.name] = fit_arm(
            known=known, row=row, arm=arm, refit=refit, fold=fold, nthread=nthread, rounds=rounds
        )
        lines.append(f"{arm.name} {site} {fold}: fit {time.monotonic() - started:.0f} s")
    origins = origins_of(fold=fold)[: SMOKE_ORIGINS if smoke or dry else None]
    pieces: dict[str, list[pl.DataFrame]] = {}
    for origin in origins[: 1 if dry else None]:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        base = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[site],
            a7=False,
        ).filter(pl.col("lead_hours") <= MAX_LEAD_HOURS)
        for arm in todo:
            prepared = add_arm_columns(frame=base, arm=arm, site=site)
            columns = checked_columns(arm=arm)
            missing = (
                prepared.select(columns).null_count().sum_horizontal().item() if columns else 0
            )
            nan = (
                prepared.select(pl.col(columns).is_nan().sum()).sum_horizontal().item()
                if columns
                else 0
            )
            if dry:
                lines.append(
                    f"{arm.name} {site} {fold}: prediction rows {prepared.height:,}, "
                    f"nulls {missing}, NaN {nan}"
                )
                continue
            if nan:
                msg = f"{arm.name} {site} {fold}: a prediction row has a NaN input"
                raise ValueError(msg)
            for name, count in outputs(arm=arm):
                pieces.setdefault(name, []).append(
                    predict_run(
                        boosters=models[arm.name], prepared=prepared, arm=arm, iterations=count
                    )
                )
    seeds = {name: a.seed for a in todo for name, _ in outputs(arm=a)}
    for name, tables in pieces.items():
        target = output_path(name=name, fold=fold, site=site, seed=seeds[name])
        target.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_write(frame=pl.concat(tables), target=target)
    return lines


def main() -> None:
    """Fit the requested arms on the requested folds and Primaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, help="arm names or tiers A, C, D")
    parser.add_argument("--folds", nargs="+", default=list(FOLDS[:2]), choices=FOLDS)
    parser.add_argument("--sites", nargs="+", default=None, help="default: all 16 Primaries")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="CPU feeder threads per process")
    parser.add_argument(
        "--smoke", action="store_true", help="20 rounds, 3 origins; set E396_SMOKE=1 too"
    )
    parser.add_argument("--dry", action="store_true", help="build matrices only; fit nothing")
    args = parser.parse_args()
    resolve_arms(names=args.arms)
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    sites = args.sites or series["site"].to_list()
    jobs = [(fold, site) for fold in args.folds for site in sites]
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                fold=fold,
                site=site,
                arm_names=args.arms,
                nthread=args.threads,
                smoke=args.smoke,
                dry=args.dry,
            )
            for fold, site in jobs
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({PRED_DIR})")


if __name__ == "__main__":
    main()
