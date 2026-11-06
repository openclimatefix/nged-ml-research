# ruff: noqa: D103
"""Row-level assertions on the extended cached inputs, repeating experiment 605's rules.

Rules (experiment 605's `e00605_assertions.py`): the weather of an origin comes from the run issued
at or before the origin minus 9 h, so `power_fcst_init_time` is the run's 00:00 UTC time plus 9 h
and every `valid_time` is after it; a power lag is stamped at least 24 h before the origin. This
script checks the first rule on every row of the three ensemble tables (944 and 345 runs), and the
second on the observed-power cache, then repeats experiment 605's capped-lag presence check on the
extended cache. It also compares the extended cache with the stored cache on every shared row.
Reads only the new tables and the stored cache; reads no forecast and no forecast error. Writes
`assertions.json` and `violations.parquet` (one row per violating row, up to 1,000 per rule).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00558_lags as lags  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

OUT: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*"))
WORK: Final[Path] = OUT / "work"
STORED_OBSERVED: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/observed.parquet"
)
DELAY: Final[timedelta] = timedelta(hours=9)
HORIZON: Final[timedelta] = timedelta(days=15)
FIRST_ORIGIN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LAST_ORIGIN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
TABLES: Final[dict[str, int]] = {"ens_summary": 944, "ens_summary_uv": 944, "ens_summary_tcc": 345}
STEP: Final[timedelta] = timedelta(minutes=30)


def check_ensemble_tables() -> tuple[dict[str, object], list[pl.DataFrame]]:
    report: dict[str, object] = {}
    violations: list[pl.DataFrame] = []
    for name, expected_files in TABLES.items():
        files = sorted((WORK / name).glob("init=*.parquet"))
        run_of = pl.col("run").str.extract(r"init=(\d{4}-\d{2}-\d{2})", 1).str.to_date()
        frame = (
            pl.scan_parquet(files, include_file_paths="run")
            .with_columns(run=run_of.cast(pl.Datetime("us", "UTC")))
            .collect()
        )
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        bad_origin = frame.filter(
            pl.col("power_fcst_init_time") != pl.col("run") + pl.duration(hours=9)
        )
        bad_after = frame.filter(pl.col("valid_time") <= pl.col("power_fcst_init_time"))
        too_far = frame.filter(
            pl.col("valid_time") > pl.col("power_fcst_init_time") + pl.duration(days=16)
        )
        duplicates = frame.height - frame.unique(subset=keys).height
        nulls = int(frame.null_count().sum_horizontal().item())
        per_run = frame.group_by("run").agg(
            n=pl.len(), series=pl.col("time_series_id").n_unique(), last=pl.col("valid_time").max()
        )
        short = per_run.filter(pl.col("series") != 16)
        report[name] = {
            "files": len(files),
            "expected_files": expected_files,
            "rows": frame.height,
            "rows_with_origin_not_run_plus_9h": bad_origin.height,
            "rows_with_valid_time_not_after_origin": bad_after.height,
            "rows_beyond_16_days_after_origin": too_far.height,
            "duplicate_keys": duplicates,
            "null_cells": nulls,
            "runs_without_all_16_primaries": short.height,
        }
        for label, bad in (
            ("origin_not_run_plus_9h", bad_origin),
            ("valid_time_not_after_origin", bad_after),
        ):
            if bad.height:
                violations.append(
                    bad.head(1000)
                    .select("run", *keys)
                    .with_columns(table=pl.lit(name), rule=pl.lit(label))
                )
    return report, violations


def check_observed() -> tuple[dict[str, object], list[pl.DataFrame]]:
    new = pl.read_parquet(WORK / "observed.parquet")
    old = pl.read_parquet(STORED_OBSERVED)
    lag_cols = [c for c in new.columns if c.startswith("power_lag_")]
    # 1. The extended cache equals the stored cache on every row the stored cache holds.
    joined = old.join(new, on=["time_series_id", "valid_time"], how="left", suffix="_new")
    missing_in_new = joined.filter(pl.col("power_new").is_null() & pl.col("power").is_not_null())
    differing = joined.filter(
        pl.col("power_new").is_not_null() & ((pl.col("power") - pl.col("power_new")).abs() > 0)
    )
    report: dict[str, object] = {
        "stored_rows": old.height,
        "extended_rows": new.height,
        "stored_rows_missing_from_extended": missing_in_new.height,
        "stored_rows_whose_power_differs": differing.height,
    }
    # 2. Each lag column equals the power at valid_time minus the lag, row by row.
    power = new.select("time_series_id", "valid_time", "power")
    for col in lag_cols:
        hours = int(col.removeprefix("power_lag_").removesuffix("h"))
        expect = power.select(
            "time_series_id",
            (pl.col("valid_time") + pl.duration(hours=hours)).alias("valid_time"),
            pl.col("power").alias("expected"),
        )
        check = new.select("time_series_id", "valid_time", col).join(
            expect, on=["time_series_id", "valid_time"], how="left"
        )
        wrong = check.filter(
            (pl.col(col).is_null() != pl.col("expected").is_null())
            | ((pl.col(col) - pl.col("expected")).abs() > 0)
        )
        report[f"{col}_rows_not_equal_to_power_{hours}h_earlier"] = wrong.height
        report[f"{col}_age_hours"] = hours
    report["newest_by_series_min"] = str(
        new.group_by("time_series_id").agg(pl.col("valid_time").max())["valid_time"].min()
    )
    return report, []


def check_capped_lags() -> dict[str, object]:
    new = pl.read_parquet(WORK / "observed.parquet")
    ids = sorted(new["time_series_id"].unique().to_list())
    grid_start = datetime(2026, 5, 1, tzinfo=UTC)
    n_steps = int((datetime(2026, 11, 7, tzinfo=UTC) - grid_start) / STEP)
    grid = np.full((len(ids), n_steps), np.nan, dtype=np.float32)
    row_of = {sid: i for i, sid in enumerate(ids)}
    recent = new.filter(pl.col("valid_time") >= grid_start)
    idx = (recent["valid_time"].dt.epoch("us").to_numpy() - int(grid_start.timestamp() * 1e6)) // (
        30 * 60 * 10**6
    )
    keep = (idx >= 0) & (idx < n_steps)
    rows = np.array([row_of[s] for s in recent["time_series_id"].to_list()])
    grid[rows[keep], idx[keep]] = recent["power"].to_numpy()[keep]
    violations = 0
    present = []
    for k in range((LAST_ORIGIN - FIRST_ORIGIN).days + 1):
        origin = FIRST_ORIGIN + timedelta(days=k) + DELAY
        o = int((origin - grid_start) / STEP)
        target = np.arange(o + 1, o + 481)
        day, week = lags.capped_lag_indices(target_idx=target, origin_idx=np.full_like(target, o))
        violations += int(((o - day) < 48).sum() + ((o - week) < 48).sum())
        present.append(float(np.isfinite(grid[:, day]).mean()))
    return {
        "origins": len(present),
        "capped_lag_age_violations": violations,
        "day_lag_share_present_mean": float(np.mean(present)),
        "day_lag_share_present_worst_origin": float(np.min(present)),
        "series_in_cache": len(ids),
    }


def main() -> None:
    ens_report, ens_violations = check_ensemble_tables()
    obs_report, _ = check_observed()
    out = {
        "ensemble_tables": ens_report,
        "observed_cache": obs_report,
        "capped_lags": check_capped_lags(),
    }
    (OUT / "assertions.json").write_text(json.dumps(out, indent=2, default=str))
    if ens_violations:
        pl.concat(ens_violations).write_parquet(OUT / "violations.parquet")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
