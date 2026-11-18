"""Input caches for experiment 707: ENS rows for runs back to 2020-10-15, and the shared row code.

The cached rows combine three steps that earlier experiments kept in separate caches. For each
00Z ENS run from 2020-10-15 to 2024-03-31, the code reads the run from the NWP Delta table with the
production `Nwp.scan_delta`, resamples it to the half-hourly grid with production's
`_upsample_nwp_to_half_hourly` (experiment 48's recipe), takes the mean and the population standard
deviation over the 51 members of each weather variable (experiment 397's summaries), and joins each
Primary's calendar columns and three origin lags (experiment 87's rows). The rows of the runs from
2024-04-01 stay in experiment 397's raw cache, which this experiment reads and never writes.

**Each ENS row takes its weather from one run: the 00Z run issued at least 9 hours before the row's
origin.** `power_fcst_init_time` (the origin) is the run's initialisation time plus 9 hours. The
code asserts `nwp_init_time + 9 h <= origin` on every row it builds, and `check` prints one example
row. `assert_known_at_cut` is the 3.5-hour cut that `e00707_folds` applies to every training row.

**The held-out validation weeks key on `FIRST_RUN = 2024-04-01`.** Origins before that date have
negative origin days, and `//` floors them, so every week keeps its phase.

The copied parts are experiment 397's `e00397_prep.py` (feature matrix, standardiser, held-out
weeks, run inputs), experiment 418's `e00418_prep.py` (`dense_weather`), and experiment 48's
`e00048_build_frames.py` (the archive read). Commands (the environment of `e00707_fit.py` applies):

- `coverage`: list the back-fill runs that the table holds and those it lacks, and write nothing.
- `scan`: print the query plan of one run, to confirm that the read prunes to one partition.
- `summaries`: write the 51-member mean and standard deviation of each run that the table holds
  and that has no summary file yet.
- `raw`: write each Primary's raw input matrix from the summaries and a manifest of the runs used.
- `check`: assert the origin rule on every cached row and print one example row.
- `ens-months`: print each Primary's first origin and its months of ENS rows.
- `step-check`: monthly ENS minus ERA5 and the ENS spread around 2023-06-27 and 2024-04-01.
"""

import argparse
import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from contracts.weather_schemas import Nwp
from e00707_models import GROUPS, Layout  # ty: ignore[unresolved-import]
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR, STUDIES_DATA_DIR

WORK_DIR: Final[Path] = arms.WORK_DIR
BACKFILL_DIR: Final[Path] = WORK_DIR / "work" / "backfill"
NEW_SUMMARY_DIR: Final[Path] = BACKFILL_DIR / "ens_summary"
"""One `init=<date>.parquet` file per back-filled run: the 51-member mean and spread."""
NEW_RAW_DIR: Final[Path] = BACKFILL_DIR / "raw" / "summary"
"""One `<site>.npz` per Primary for the back-filled runs, plus `manifest.json`."""
EXPERIMENT_397_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work"
)
OLD_RAW_DIR: Final[Path] = EXPERIMENT_397_WORK / "raw" / "summary"
"""Experiment 397's raw rows for the runs from 2024-04-01; read only."""
OLD_SUMMARY_DIR: Final[Path] = EXPERIMENT_397_WORK / "ens_summary"
"""Experiment 397's run summaries from 2024-04-01, which the scored runs' inputs read."""
ERA_RAW_DIR: Final[Path] = (
    (ML_EXPERIMENTS_DATA_DIR / "00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens")
    / "work"
    / "raw_pretrain"
)
"""Experiment 418's ERA5-and-CAMS pseudo-run rows for `S_era`; read only."""
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads" / "reanalysis" / "CAMS" / "demand_points"
) / "cams_demand_points.parquet"
NWP_TABLE_PATH: Final[Path] = REPO_DATA_DIR / "NWP"
"""The NWP Delta table, partitioned by `nwp_model_id` and `init_time`."""
NWP_MODEL_ID: Final[str] = "ECMWF_ENS_0_25_degree"

FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The anchor of the held-out validation weeks; do not move it (negative days floor-divide)."""
BACKFILL_FIRST_RUN: Final[datetime] = datetime(2020, 10, 15, tzinfo=UTC)
BACKFILL_LAST_RUN: Final[datetime] = datetime(2024, 3, 31, tzinfo=UTC)
N_MEMBERS: Final[int] = 51
MAX_LEAD_HOURS: Final[float] = 351.0
"""A 00Z run reaches 360 hours; its origin is 9 hours after the run."""
INPUT_CLIP_SD: Final[float] = 5.0
HELD_WEEK_PERIOD: Final[int] = 12
"""One week in every 12 is held out for early stopping."""
LEAD_SCALE_HOURS: Final[float] = 336.0
US_PER_DAY: Final[int] = 86_400 * 1_000_000
US_PER_HOUR: Final[int] = 3_600 * 1_000_000
GROUP_NAMES: Final[tuple[str, ...]] = (*GROUPS, "lead")
ARCHIVE_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
}
"""The network's weather variables that the table serves directly, and their table columns."""
WIND_COLUMNS: Final[tuple[str, str]] = ("wind_u_10m", "wind_v_10m")
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    *ARCHIVE_COLUMNS.values(),
    *WIND_COLUMNS,
)
"""The columns read from the table: the network does not read precipitation type."""
PARTITION_PATTERN: Final[re.Pattern[str]] = re.compile(r"init_time=(\d{4}-\d{2}-\d{2})%20")
STEP_DATES: Final[tuple[datetime, ...]] = arms.FLAG_DATES
STEP_SD_LIMIT: Final[float] = 3.0
STEP_LEADS_HOURS: Final[tuple[float, float]] = (12.0, 36.0)
PRETRAIN_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def primaries() -> pl.DataFrame:
    """Return the 16 Primary series of experiment 87's table, sorted by label number."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    return series.with_row_index("primary")


# --- rows are known at their cut --------------------------------------------------------------


def assert_origin_after_run(*, nwp_init: np.ndarray, origin: np.ndarray) -> None:
    """Assert `nwp_init_time + 9 h <= origin` for every row (both in microseconds since the epoch).

    Raises:
        ValueError: If a row reads a run issued less than 9 hours before its origin.
    """
    late = nwp_init + common.PUBLICATION_DELAY // timedelta(microseconds=1) > origin
    if late.any():
        msg = f"{int(late.sum())} rows read a run issued less than 9 hours before the origin"
        raise ValueError(msg)


def assert_known_at_cut(*, valid: np.ndarray, origin: np.ndarray, refit: datetime) -> None:
    """Assert that every row's valid time is 3.5 hours before the cut and its origin is not after.

    Args:
        valid: Valid times, microseconds since the epoch.
        origin: Origins, microseconds since the epoch.
        refit: The fold's cut.

    Raises:
        ValueError: If a row is not known at the cut.
    """
    too_new = (valid > _us(t=refit - common.REFIT_CUTOFF_AGE)) | (origin > _us(t=refit))
    if too_new.any():
        msg = f"{int(too_new.sum())} training rows are not known 3.5 hours before the cut {refit}"
        raise ValueError(msg)


# --- the archive read ---------------------------------------------------------------------------


def backfill_runs() -> list[datetime]:
    """Return the daily 00:00 UTC runs of the back-fill range, 2020-10-15 to 2024-03-31."""
    days = (BACKFILL_LAST_RUN - BACKFILL_FIRST_RUN).days + 1
    return [BACKFILL_FIRST_RUN + timedelta(days=i) for i in range(days)]


def runs_in_table() -> set[datetime]:
    """Return the runs that have a partition folder in the NWP table (as `run_backfill.sh` does)."""
    folder = NWP_TABLE_PATH / f"nwp_model_id={NWP_MODEL_ID}"
    found = set()
    for path in folder.iterdir():
        match = PARTITION_PATTERN.match(path.name)
        if match is not None:
            found.add(datetime.fromisoformat(match.group(1)).replace(tzinfo=UTC))
    return found


def as_ranges(*, days: list[datetime]) -> list[str]:
    """Return sorted dates as `first to last (n days)` strings, one per unbroken range."""
    out: list[str] = []
    start = previous = None
    for day in sorted(days):
        if start is not None and previous is not None and day - previous == timedelta(days=1):
            previous = day
            continue
        if start is not None and previous is not None:
            out.append(
                f"{start:%Y-%m-%d} to {previous:%Y-%m-%d} ({(previous - start).days + 1} days)"
            )
        start = previous = day
    if start is not None and previous is not None:
        out.append(f"{start:%Y-%m-%d} to {previous:%Y-%m-%d} ({(previous - start).days + 1} days)")
    return out


def coverage() -> tuple[list[datetime], list[datetime]]:
    """Print and return the back-fill runs that the table holds and those it lacks.

    The lockdown runs (2020-03-23 to 2021-07-19) are counted separately.
    """
    wanted = backfill_runs()
    held = runs_in_table()
    present = [r for r in wanted if r in held]
    missing = [r for r in wanted if r not in held]
    lockdown = [r for r in present if common.LOCKDOWN_START <= r <= common.LOCKDOWN_END]
    print(
        f"back-fill range {wanted[0]:%Y-%m-%d} to {wanted[-1]:%Y-%m-%d}: {len(wanted)} runs; "
        f"the table holds {len(present)} ({len(lockdown)} inside the lockdown); "
        f"{len(missing)} are missing"
    )
    for line in as_ranges(days=missing):
        print(f"  MISSING {line}")
    if missing:
        print("PARTIAL: later steps use only the runs that the table holds")
    return present, missing


def run_scan(*, run: datetime) -> pl.LazyFrame:
    """Return a lazy scan of one run's rows from the NWP Delta table, with the needed columns.

    The scan goes through the production `Nwp.scan_delta`. The filter on the partition columns
    comes first so that Polars reads one partition; the `scan` command prints the plan to confirm
    that.
    """
    scan: pl.LazyFrame = Nwp.scan_delta(NWP_TABLE_PATH)
    return scan.filter(pl.col("nwp_model_id") == NWP_MODEL_ID, pl.col("init_time") == run).select(
        "init_time", *SOURCE_COLUMNS
    )


def run_summary(*, run: datetime, targets: pl.DataFrame) -> pl.DataFrame:
    """Return one run's member mean and spread for the target series, from the NWP Delta table.

    Each row is one series and valid time after the run's origin. The weather is resampled to the
    half-hourly grid by production's `_upsample_nwp_to_half_hourly` (short-wave energy-conserving,
    long-wave PCHIP, other variables linear), then summarised over the members. A run is read
    whole, so no row ever reads a later run.

    Args:
        run: The run's 00:00 UTC initialisation time.
        targets: One row per series with `time_series_id`, `h3_res_5`, `latitude`, `longitude`.

    Returns:
        Rows keyed by `time_series_id`, `nwp_init_time`, `power_fcst_init_time`, and `valid_time`,
        with `mean_<w>` and `std_<w>` (float32) for each weather variable of `common.WEATHER`.

    Raises:
        ValueError: If the table holds no rows for the run, a valid time lacks a member, or a row
            reads a run issued less than 9 hours before its origin.
    """
    cells = targets["h3_res_5"].unique().to_list()
    native = (
        run_scan(run=run)
        .filter(pl.col("h3_index").is_in(cells))
        .join(
            targets.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .rename({"init_time": "nwp_init_time"})
    )
    coordinates = targets.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = (
        _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates)
        .collect()
        .rename({source: name for name, source in ARCHIVE_COLUMNS.items()})
    )
    if half_hourly.is_empty():
        msg = f"the NWP table holds no rows of run {run:%Y-%m-%d} for these cells"
        raise ValueError(msg)
    origin = common.origin_of(run=run)
    weather = half_hourly.with_columns(
        wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
        power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
    ).filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
    summary = (
        weather.group_by("time_series_id", "nwp_init_time", "power_fcst_init_time", "valid_time")
        .agg(
            pl.col("ensemble_member").n_unique().alias("n_members"),
            *[pl.col(w).mean().alias(f"mean_{w}") for w in common.WEATHER],
            *[pl.col(w).std(ddof=0).alias(f"std_{w}") for w in common.WEATHER],
        )
        .sort("time_series_id", "valid_time")
    )
    if (summary["n_members"] != N_MEMBERS).any():
        msg = f"run {run:%Y-%m-%d}: a valid time holds fewer than {N_MEMBERS} members"
        raise ValueError(msg)
    lead = (summary["valid_time"] - summary["power_fcst_init_time"]).dt.total_minutes() / 60.0
    if float(lead.max()) > MAX_LEAD_HOURS:  # ty: ignore[invalid-argument-type]
        msg = f"run {run:%Y-%m-%d}: a lead exceeds {MAX_LEAD_HOURS} hours"
        raise ValueError(msg)
    assert_origin_after_run(
        nwp_init=summary["nwp_init_time"].dt.epoch("us").to_numpy(),
        origin=summary["power_fcst_init_time"].dt.epoch("us").to_numpy(),
    )
    return summary.drop("n_members").cast(
        dict.fromkeys([f"{k}_{w}" for k in ("mean", "std") for w in common.WEATHER], pl.Float32)
    )


def write_summary(*, run: datetime) -> str:
    """Write one run's summary file if absent; return a progress line."""
    target = NEW_SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    targets = primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    table = run_summary(run=run, targets=targets)
    common.atomic_write(frame=table, target=target)
    return f"{run:%Y-%m-%d}: {table.height:,} rows, {time.monotonic() - started:.0f} s"


def _write_summary_kw(run: datetime) -> str:
    """Return `write_summary` for a run, as a picklable function of one argument."""
    return write_summary(run=run)


def build_summaries(*, workers: int, limit: int | None) -> None:
    """Write the summary of every run that the table holds and that lacks a file."""
    present, _ = coverage()
    todo = [r for r in present if not (NEW_SUMMARY_DIR / f"init={r:%Y-%m-%d}.parquet").exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} summaries to write")
    if not todo:
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for line in pool.map(_write_summary_kw, todo):
            print(line, flush=True)


# --- feature matrix (experiment 397) --------------------------------------------------------


def feature_names(*, scheme: str = "summary") -> tuple[list[str], np.ndarray, Layout]:
    """Return the input column names, the mask of columns to standardise, and the layout."""
    if scheme != "summary":
        msg = "experiment 707 uses the summary scheme only"
        raise ValueError(msg)
    weather = [f"mean_{w}" for w in common.WEATHER] + [f"std_{w}" for w in common.WEATHER]
    groups = {
        "weather": weather,
        "solar": ["cos_zenith", "sin_azimuth", "cos_azimuth"],
        "calendar": [
            "sin_hour",
            "cos_hour",
            "sin_hour_2",
            "cos_hour_2",
            *[f"dow_{d}" for d in range(1, 8)],
            "covid",
        ],
        "holidays": [
            "christmas_inside",
            "christmas_days",
            "christmas_tent",
            "bank_holiday",
            "nearest_days",
            "nearest_inside",
        ],
        "lags": [
            "lag_origin",
            "lag_day",
            "lag_week",
            "lag_origin_missing",
            "lag_day_missing",
            "lag_week_missing",
            "lag_origin_age",
            "lag_day_age",
            "lag_week_age",
        ],
        "lead": ["lead", "lead_sqrt"],
    }
    names: list[str] = []
    slices: dict[str, slice] = {}
    for g in GROUP_NAMES:
        slices[g] = slice(len(names), len(names) + len(groups[g]))
        names += groups[g]
    standardise = np.array(
        [
            n.startswith(("mean_", "std_", "christmas_days", "nearest_days", "lag_"))
            and not n.endswith(("_missing", "_age"))
            for n in names
        ]
    )
    return names, standardise, Layout(slices=slices, n_inputs=len(names))


def lag_ages_hours(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the age in hours of the three origin lags at each lead, shape (rows, 3)."""
    gap = np.rint(lead_hours * 2.0) + common.NEWEST_AGE
    day = 24.0 * np.ceil(gap / common.STEPS_PER_DAY)
    week = 168.0 * np.ceil(gap / (7 * common.STEPS_PER_DAY))
    newest = gap / 2.0
    return np.stack([newest, day, week], axis=1)


def feature_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the raw (unstandardised) input matrix of a frame, float32, columns of `feature_names`.

    Weather precipitation goes through `log1p`. Lags that are missing are NaN and are flagged here
    and filled after standardisation. Every weather column must be finite.

    Raises:
        ValueError: If a weather input is not finite.
    """
    names, _, _ = feature_names()
    n = frame.height
    cols: dict[str, np.ndarray] = {}
    for k in ("mean", "std"):
        for w in common.WEATHER:
            v = frame[f"{k}_{w}"].to_numpy().astype(np.float64)
            cols[f"{k}_{w}"] = np.log1p(np.clip(v, 0, None)) if w == "precipitation" else v
    weather_columns = [c for c in cols if c.startswith(("mean_", "std_"))]
    if not all(np.isfinite(cols[c]).all() for c in weather_columns):
        msg = "a weather input is not finite; add an is-missing flag for it"
        raise ValueError(msg)
    zenith = np.radians(frame["solar_zenith"].to_numpy().astype(np.float64))
    azimuth = np.radians(frame["solar_azimuth"].to_numpy().astype(np.float64))
    cols["cos_zenith"], cols["sin_azimuth"], cols["cos_azimuth"] = (
        np.cos(zenith),
        np.sin(azimuth),
        np.cos(azimuth),
    )
    hour = frame["hour_local"].to_numpy().astype(np.float64)
    for k, cycles in (("", 1.0), ("_2", 2.0)):
        cols[f"sin_hour{k}"] = np.sin(2 * np.pi * cycles * hour / 24.0)
        cols[f"cos_hour{k}"] = np.cos(2 * np.pi * cycles * hour / 24.0)
    dow = frame["day_of_week_local"].to_numpy().astype(np.int64)
    for d in range(1, 8):
        cols[f"dow_{d}"] = (dow == d).astype(np.float64)
    cols["covid"] = frame["covid_indicator"].to_numpy().astype(np.float64)
    christmas = frame["days_to_christmas"].to_numpy().astype(np.float64)
    inside = np.isfinite(christmas)
    scaled = np.where(inside, christmas / common.CHRISTMAS_CLIP_DAYS, 0.0)
    cols["christmas_inside"] = inside.astype(np.float64)
    cols["christmas_days"] = scaled
    cols["christmas_tent"] = np.where(inside, 1.0 - np.abs(scaled), 0.0)
    nearest = frame["days_to_nearest_holiday"].to_numpy().astype(np.float64)
    cols["bank_holiday"] = (nearest == 0).astype(np.float64)
    cols["nearest_days"] = nearest / common.HOLIDAY_CLIP_DAYS
    cols["nearest_inside"] = (np.abs(nearest) < common.HOLIDAY_CLIP_DAYS).astype(np.float64)
    lag_values = np.stack([frame[c].to_numpy().astype(np.float64) for c in common.LAG_COLUMNS], 1)
    missing = ~np.isfinite(lag_values)
    ages = lag_ages_hours(lead_hours=frame["lead_hours"].to_numpy().astype(np.float64))
    for i, tag in enumerate(("origin", "day", "week")):
        cols[f"lag_{tag}"] = lag_values[:, i]
        cols[f"lag_{tag}_missing"] = missing[:, i].astype(np.float64)
        cols[f"lag_{tag}_age"] = ages[:, i] / LEAD_SCALE_HOURS
    lead = frame["lead_hours"].to_numpy().astype(np.float64) / LEAD_SCALE_HOURS
    cols["lead"], cols["lead_sqrt"] = lead, np.sqrt(np.clip(lead, 0, None))
    out = np.empty((n, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        out[:, j] = cols[name]
    return out


@dataclass(frozen=True)
class Standardiser:
    """Training mean and standard deviation of the standardised columns, applied unchanged later."""

    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    @classmethod
    def fit(cls, *, x: np.ndarray, mask: np.ndarray) -> Standardiser:
        """Fit on the training rows (NaN-aware, for the lag columns)."""
        mean = np.where(mask, np.nanmean(x, axis=0), 0.0)
        std = np.where(mask, np.nanstd(x, axis=0), 1.0)
        return cls(mean=mean, std=np.maximum(std, 1e-6), mask=mask)

    def apply(self, *, x: np.ndarray) -> np.ndarray:
        """Standardise, clip at five standard deviations, and fill NaN with 0."""
        out = (x - self.mean) / self.std
        out = np.clip(out, -INPUT_CLIP_SD, INPUT_CLIP_SD)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


def held_out_weeks(*, origin_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return masks of early-stopping rows and of purged rows, by run origin day.

    One week in every 12 is held out. A training run is purged when it begins up to 14 days before
    a held-out week (its leads reach into the week) or up to 7 days after it (its newest lags lie
    in the week). The origin day counts from `FIRST_RUN`, and is negative before it; `//` floors.
    """
    week = origin_day // 7
    phase = week % HELD_WEEK_PERIOD
    held = phase == HELD_WEEK_PERIOD - 1
    purged = np.isin(phase, [HELD_WEEK_PERIOD - 3, HELD_WEEK_PERIOD - 2, 0])
    return held, purged


def origin_days(*, origin: np.ndarray) -> np.ndarray:
    """Return the origin day of each origin (microseconds), counted from FIRST_RUN."""
    return ((origin - _us(t=FIRST_RUN)) // US_PER_DAY).astype(np.int64)


# --- per-Primary rows and raw matrices -------------------------------------------------------


def load_summary(*, series_id: int) -> pl.DataFrame:
    """Return one series' summaries over all back-filled runs."""
    return (
        pl.scan_parquet(NEW_SUMMARY_DIR / "init=*.parquet")
        .filter(pl.col("time_series_id") == series_id)
        .collect()
    )


def primary_rows(
    *, row: dict, power: np.ndarray, summary: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    """Return a Primary's ENS rows with calendar columns, lags, and target, and the rows dropped.

    A row is dropped if its observed power is missing or a weather input is not finite; the count
    is returned so that the caller can print it.

    Args:
        row: The series' row of `primaries()`.
        power: The series' observed power over capacity on the dense grid.
        summary: `load_summary`.

    Returns:
        The rows, and how many rows the two drops removed.
    """
    rows = summary.filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    rows = common.add_lags(frame=rows, power=power)
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    target = common.take(values=power, index=index)
    rows = rows.with_columns(target=pl.Series(target))
    weather = [f"{k}_{w}" for k in ("mean", "std") for w in common.WEATHER]
    keep = pl.col("target").is_finite() & pl.all_horizontal(pl.col(w).is_finite() for w in weather)
    kept = rows.filter(keep)
    return kept, rows.height - kept.height


def write_raw(*, site: str, run_list: list[str]) -> str:
    """Write a Primary's raw matrix, target, and keys for the back-filled runs; return a line.

    The file is skipped only if the manifest of runs it was built from equals `run_list`.
    """
    target = NEW_RAW_DIR / f"{site}.npz"
    manifest = NEW_RAW_DIR / "manifest.json"
    if (
        target.exists()
        and manifest.exists()
        and json.loads(manifest.read_text())["runs"] == run_list
    ):
        return f"{site}: exists for the same {len(run_list)} runs"
    started = time.monotonic()
    table = primaries()
    row = next(r for r in table.iter_rows(named=True) if r["site"] == site)
    _, arrays = common.series_and_arrays()
    rows, dropped = primary_rows(
        row=row, power=arrays[site].power, summary=load_summary(series_id=row["time_series_id"])
    )
    x = feature_matrix(frame=rows)
    nwp_init = rows["nwp_init_time"].dt.epoch("us").to_numpy()
    origin = rows["power_fcst_init_time"].dt.epoch("us").to_numpy()
    assert_origin_after_run(nwp_init=nwp_init, origin=origin)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        x=x,
        y=rows["target"].to_numpy().astype(np.float32),
        valid=rows["valid_time"].dt.epoch("us").to_numpy(),
        origin=origin,
        lead=rows["lead_hours"].to_numpy().astype(np.float32),
        nwp_init=nwp_init,
    )
    temporary.rename(target)
    return (
        f"{site}: {len(x):,} rows; dropped {dropped:,} for a missing power or weather value; "
        f"{time.monotonic() - started:.0f} s"
    )


def build_raw() -> None:
    """Write every Primary's raw matrix from the summaries on disk, and the manifest."""
    present, missing = coverage()
    files = sorted(NEW_SUMMARY_DIR.glob("init=*.parquet"))
    if not files:
        msg = "no summary files: run `summaries` first"
        raise ValueError(msg)
    run_list = [f.stem.removeprefix("init=") for f in files]
    unsummarised = [r for r in present if f"{r:%Y-%m-%d}" not in set(run_list)]
    print(
        f"{len(run_list)} summaries; {len(unsummarised)} runs in the table have no summary; "
        f"{len(missing)} runs are not in the table"
    )
    for site in primaries()["site"].to_list():
        print(write_raw(site=site, run_list=run_list), flush=True)
    manifest = {
        "runs": run_list,
        "missing_from_table": [f"{r:%Y-%m-%d}" for r in missing],
        "summaries_not_yet_written": [f"{r:%Y-%m-%d}" for r in unsummarised],
        "lockdown_runs": sum(
            common.LOCKDOWN_START
            <= datetime.fromisoformat(r).replace(tzinfo=UTC)
            <= common.LOCKDOWN_END
            for r in run_list
        ),
    }
    temporary = NEW_RAW_DIR / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=1))
    temporary.replace(NEW_RAW_DIR / "manifest.json")


def load_site(*, site: str) -> dict[str, np.ndarray]:
    """Return a Primary's ENS rows in time order: the back-filled rows, then experiment 397's.

    The back-filled file is optional (it may not exist yet); its absence is printed once per call
    by the caller through `backfill_available`. Every array has the keys `x`, `y`, `valid`,
    `origin`, `lead`.
    """
    parts = []
    new = NEW_RAW_DIR / f"{site}.npz"
    if new.exists():
        z = np.load(new)
        parts.append({k: z[k] for k in ("x", "y", "valid", "origin", "lead")})
    z = np.load(OLD_RAW_DIR / f"{site}.npz")
    parts.append({k: z[k] for k in ("x", "y", "valid", "origin", "lead")})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def backfill_available() -> tuple[datetime | None, int]:
    """Return the first back-filled run on disk (or `None`) and the number of runs it holds."""
    manifest = NEW_RAW_DIR / "manifest.json"
    if not manifest.exists():
        return None, 0
    runs = json.loads(manifest.read_text())["runs"]
    return (datetime.fromisoformat(runs[0]).replace(tzinfo=UTC) if runs else None), len(runs)


# --- ERA5 and CAMS weather on the dense grid (experiment 418) --------------------------------


def dense_weather(*, series_id: int, site: str) -> np.ndarray:
    """Return a Primary's pretraining weather on the dense half-hourly grid, shape (grid, 7).

    The ERA5 file is already on the half-hourly grid, so its values sit at their own stamps.
    Shortwave is CAMS global horizontal irradiance, which is hourly: both half-hours of an hour
    take the value stamped at the end of that hour. Columns follow `common.WEATHER`.
    """
    size = common.grid_times().len()
    out = np.full((size, len(common.WEATHER)), np.nan, dtype=np.float32)
    era5 = (
        pl.read_parquet(common.ERA5_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .sort("valid_time")
    )
    index = np.asarray(common.grid_index(times=era5["valid_time"]))
    inside = (index >= 0) & (index < size)
    for name in common.WEATHER:
        if name != "shortwave":
            out[index[inside], common.WEATHER.index(name)] = (
                era5[name].to_numpy()[inside].astype(np.float32)
            )
    cams = pl.read_parquet(CAMS_PATH).filter(pl.col("site") == site).sort("time")
    cams_index = np.asarray(common.grid_index(times=cams["time"]))
    ok = (cams_index > 0) & (cams_index < size)
    ghi = cams["ghi_w_m2"].to_numpy().astype(np.float32)
    column = common.WEATHER.index("shortwave")
    out[cams_index[ok], column] = ghi[ok]
    out[cams_index[ok] - 1, column] = ghi[ok]
    return out


# --- scored runs' inputs (experiment 397) ------------------------------------------------------


def refit_of(*, month: str) -> datetime:
    """Return a fold's refit time: 00:00 UTC on the first day of its test month."""
    return datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)


def test_origins(*, month: str) -> list[datetime]:
    """Return the daily `power_fcst_init_time` of the runs whose origin falls in a month."""
    start = refit_of(month=month)
    stop = (start + timedelta(days=32)).replace(day=1)
    return [o for o in common.all_origins() if start <= o < stop]


def run_inputs(
    *, origin: datetime, standardiser: Standardiser
) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return, for one scored run, each Primary's keys, summary rows, and standardised inputs.

    The inputs of a scored run (2025-07-01 onwards) come from experiment 397's summaries and
    experiment 87's stored 51-member frames, both read only. Each item is (series row, keys
    frame, standardised inputs) with one row per valid time.
    """
    _series, arrays = common.series_and_arrays()
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    out = []
    for row in primaries().iter_rows(named=True):
        prepared = common.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[row["site"]],
        )
        summary = pl.read_parquet(
            OLD_SUMMARY_DIR / f"init={origin - common.PUBLICATION_DELAY:%Y-%m-%d}.parquet"
        ).filter(pl.col("time_series_id") == row["time_series_id"])
        first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
        prepared = first.join(
            summary.drop("time_series_id"),
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
            maintain_order="left",
        )
        keys = prepared.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        out.append((row, keys, standardiser.apply(x=feature_matrix(frame=prepared))))
    return out


# --- checks -------------------------------------------------------------------------------------


def check_raw() -> None:
    """Assert the origin rule on every back-filled row and print one example row."""
    files = sorted(NEW_RAW_DIR.glob("*.npz"))
    if not files:
        print("no back-filled raw files yet: nothing to check")
        return
    names, _, _ = feature_names()
    example_printed = False
    for path in files:
        z = np.load(path)
        assert_origin_after_run(nwp_init=z["nwp_init"], origin=z["origin"])
        lead = (z["valid"] - z["origin"]) / US_PER_HOUR
        if not np.allclose(lead, z["lead"], atol=1e-3):
            msg = f"{path.name}: the stored lead differs from valid minus origin"
            raise ValueError(msg)
        if not example_printed and len(z["lead"]):
            near = int(np.argmin(np.abs(z["lead"] - 128.5)))
            stamp = {
                k: datetime.fromtimestamp(int(z[k][near]) / 1e6, tz=UTC)
                for k in ("nwp_init", "origin", "valid")
            }
            print(
                f"example row of {path.stem}: run {stamp['nwp_init']:%Y-%m-%d %H:%M} UTC, "
                f"origin {stamp['origin']:%Y-%m-%d %H:%M} UTC, "
                f"valid {stamp['valid']:%Y-%m-%d %H:%M} UTC, "
                f"lead {float(z['lead'][near]):.1f} h, y {float(z['y'][near]):.4f}"
            )
            print({n: round(float(v), 3) for n, v in zip(names[:4], z["x"][near][:4], strict=True)})
            example_printed = True
        print(f"{path.stem}: {len(z['lead']):,} rows, nwp_init + 9 h <= origin on every row")


def months_of(*, stamps: np.ndarray) -> np.ndarray:
    """Return the distinct `YYYY-MM` months of microsecond stamps."""
    return np.unique(stamps.astype("datetime64[us]").astype("datetime64[M]").astype(str))


def ens_months() -> None:
    """Print each Primary's first origin and the number of months of ENS rows it holds."""
    print("site: first origin, months with rows (valid time), rows")
    for site in primaries()["site"].to_list():
        z = load_site(site=site)
        first = datetime.fromtimestamp(int(z["origin"].min()) / 1e6, tz=UTC)
        months = months_of(stamps=z["valid"])
        print(f"{site}: {first:%Y-%m-%d}, {len(months)} months, {len(z['y']):,} rows")


def step_check() -> None:
    """Print monthly ENS minus ERA5 (CAMS for shortwave) and the ENS spread around the two dates.

    The leads are 12 to 36 hours. For each variable and each date in `STEP_DATES`, the step is the
    absolute difference of the mean of the 12 monthly values after the date's month and the mean of
    the 12 before it, in units of the mean of the two sides' standard deviations (ddof 1). A step
    above `STEP_SD_LIMIT` fails the check. The check is a screen, not the page's final test.
    """
    names, _, _ = feature_names()
    mean_cols = [names.index(f"mean_{w}") for w in common.WEATHER]
    std_cols = [names.index(f"std_{w}") for w in common.WEATHER]
    pieces = []
    for row in primaries().iter_rows(named=True):
        z = load_site(site=row["site"])
        keep = (z["lead"] >= STEP_LEADS_HOURS[0]) & (z["lead"] <= STEP_LEADS_HOURS[1])
        index = ((z["valid"][keep] - _us(t=common.GRID_START)) // 1_800_000_000).astype(np.int64)
        era = dense_weather(series_id=row["time_series_id"], site=row["site"])[index]
        era[:, common.WEATHER.index("precipitation")] = np.log1p(
            np.clip(era[:, common.WEATHER.index("precipitation")], 0, None)
        )
        x = z["x"][keep]
        month = z["valid"][keep].astype("datetime64[us]").astype("datetime64[M]").astype(str)
        frame = pl.DataFrame({"month": month, "site": np.full(len(month), row["site"])})
        for c, w in enumerate(common.WEATHER):
            frame = frame.with_columns(
                pl.Series(f"diff_{w}", x[:, mean_cols[c]] - era[:, c]),
                pl.Series(f"spread_{w}", x[:, std_cols[c]]),
            )
        pieces.append(frame.group_by("month", "site").mean())
    monthly = pl.concat(pieces).group_by("month").mean().sort("month")
    months = monthly["month"].to_list()
    failed = False
    for date in STEP_DATES:
        label = f"{date:%Y-%m}"
        if label not in months:
            print(f"step at {date:%Y-%m-%d}: no ENS rows in {label}; not checked")
            continue
        at = months.index(label)
        for prefix in ("diff", "spread"):
            for w in common.WEATHER:
                values = monthly[f"{prefix}_{w}"].to_numpy()
                before, after = values[max(0, at - 12) : at], values[at + 1 : at + 13]
                if len(before) < 3 or len(after) < 3:
                    print(f"{date:%Y-%m-%d} {prefix}_{w}: too few monthly values on one side")
                    continue
                scale = 0.5 * (before.std(ddof=1) + after.std(ddof=1))
                step = abs(float(after.mean() - before.mean()))
                flag = "FAIL" if step > STEP_SD_LIMIT * scale else "ok"
                failed |= flag == "FAIL"
                print(
                    f"{date:%Y-%m-%d} {prefix}_{w}: step {step:.4g}, scale {scale:.4g}, "
                    f"{step / max(scale, 1e-12):.1f} sd: {flag}"
                )
    if failed:
        print("STEP CHECK FAILED: stop and report to the manager")


def scan_plan(*, run: datetime) -> None:
    """Print the query plan of one run's scan, to confirm that it prunes to one partition."""
    print(run_scan(run=run).explain())


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("coverage", "scan", "summaries", "raw", "check", "ens-months", "step-check"),
    )
    parser.add_argument("--run", default="2024-03-31", help="the run for `scan`")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1, help="Polars threads per worker")
    parser.add_argument("--limit", type=int, default=None, help="write at most this many summaries")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    if args.command == "coverage":
        _, missing = coverage()
        if args.require_complete and missing:
            raise SystemExit(1)
    elif args.command == "scan":
        scan_plan(run=datetime.fromisoformat(args.run).replace(tzinfo=UTC))
    elif args.command == "summaries":
        build_summaries(workers=args.workers, limit=args.limit)
    elif args.command == "raw":
        build_raw()
    elif args.command == "check":
        check_raw()
    elif args.command == "ens-months":
        ens_months()
    else:
        step_check()


if __name__ == "__main__":
    main()
