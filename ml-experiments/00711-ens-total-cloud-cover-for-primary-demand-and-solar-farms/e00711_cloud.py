"""Build and check the ENS total-cloud-cover summaries of experiment 711 (plan section 9).

For each 00 UTC run that carries cloud, the build reads the run from the NWP Delta table at one
pinned version, resamples the 51 members to the half-hourly grid with production's
`_upsample_nwp_to_half_hourly` (cloud is a continuous variable, so it is interpolated linearly),
and writes one file per run with the four cloud columns of the 16 Primaries:

- `cloud_mean` and `cloud_spread`: the ensemble mean and the population standard deviation of
  total cloud cover, divided by 100 (summary T1).
- `cloud_clear_share` and `cloud_overcast_share`: the share of the 51 members at or below 20% and
  at or above 80% cloud cover (with T1, summary T2).

The files are keyed like the existing summaries of experiments 397 and 707 by `time_series_id`,
`power_fcst_init_time`, and `valid_time`, and also hold `nwp_init_time`. No file is written for a
run in the 19-month gap or for the upstream gap run. The summary T3 (daylight mean) is built later,
from the cached `cloud_mean` and the solar position, in the raw-matrix step.

**Commands** (run `pin` first; every later read uses the recorded Delta version):

- `pin`: record the current Delta version of the table in `lineage.json` with the build settings.
- `table-check`: the plan's conditions on the table at the pinned version. Each run from
  2025-11-21 to 2026-11-18 holds 51 members and no null cloud, and each run from 2025-11-21 to
  2026-06-30 holds 7,243,785 rows. Nulls lie only inside 2024-04-01 to 2025-11-20. Cloud lies
  in 0 to 100.
- `older-columns`: recompute the seven older variables for 20 sampled runs and compare them with
  the stored summaries of experiments 397 and 707.
- `build`: write the cache files of every run that lacks one (`--dry-run-date` builds one run in
  memory, prints its schema, and writes nothing).
- `check-cache`: the data checks on the written files, one example row, and the cross-check
  against the 222 cloud files of experiment 561.
- `era-check`: monthly ENS cloud minus ERA5 cloud and the monthly ENS spread for the 16 Primaries
  and the 6 solar farm cells, and the step tests of plan section 4.

**Expected cost of the full `build`** (measured on one run, 2026-01-14, on one thread): a run takes
about 1 s of reading and resampling plus about 2 s of process start-up, and the process peaks at
0.53 GB of resident memory. The 1,485 files (1,263 runs from 2020-10-15 to 2024-03-31 and 222 runs
from 2025-11-21 to 2026-06-30) therefore take about 25 to 40 minutes on one worker, and under 15
minutes on 4 workers of 1 thread each at under 3 GB in total. A run before 2021-07-20 is a donor
and lag source only, so the cache holds 278 more files than the 1,207 runs that rows read.
"""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00711_common as common
import e00711_origin as origin_rules
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

MIN_PLAUSIBLE_MAX_MEAN: Final[float] = 0.05
"""A real run over 16 cells has a cloud mean above this somewhere; cloud stored as a fraction
(0 to 1) would give a mean at most 0.01."""
MIN_CROSS_CHECK_CORRELATION: Final[float] = 0.95
"""The least correlation of the ensemble means with experiment 561's files that `check-cache`
accepts (pending the manager's agreement)."""
CROSS_CHECK_RUNS: Final[int] = 222
VALID_TIMES_PER_RUN: Final[int] = 702
"""Half-hours from 0.5 to 351 hours after the origin."""
MAX_LEAD_HOURS: Final[float] = 351.0
OLDER_SOURCE_COLUMNS: Final[dict[str, str]] = {
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
    "wind_u_10m": "wind_u_10m",
    "wind_v_10m": "wind_v_10m",
}
"""The table columns behind the seven older variables of `e00707_common.WEATHER`."""
OLDER_RTOL: Final[float] = 1e-4
OLDER_SAMPLE_RUNS: Final[int] = 20
STEP_SD_LIMIT: Final[float] = 3.0
STEP_LEADS_HOURS: Final[tuple[float, float]] = (12.0, 36.0)
RESOLUTION_CHANGE: Final[datetime] = datetime(2023, 6, 27, tzinfo=UTC)
WINTER_MONTHS: Final[tuple[int, ...]] = (12, 1, 2, 3)
OLD_WINTERS: Final[tuple[int, ...]] = (2021, 2022, 2023)
"""The first calendar year of each earlier winter, December to March: 2021-22 to 2023-24."""
RECENT_WINTER: Final[int] = 2025
EXPERIMENT_561_DIR: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00607-bakeoff-extend-cached-inputs-to-2026-10-31"
    / "work"
    / "ens_summary_tcc"
)
"""Experiment 561's cloud summaries (built by 607 from the 561 recipe); read only."""
ERA5_PATH: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00607-bakeoff-extend-cached-inputs-to-2026-10-31"
    / "work"
    / "era5_wind_components.parquet"
)
OLD_SUMMARY_DIRS: Final[tuple[Path, Path]] = (
    common.WORK_DIR.parent
    / "00707-ens-backfill-pretraining-gain-against-ens-history-length"
    / "work"
    / "backfill"
    / "ens_summary",
    common.ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
    / "work"
    / "ens_summary",
)
"""Stored seven-variable summaries: experiment 707 (to 2024-03-31) and 397 (from 2024-04-01)."""


def primaries() -> pl.DataFrame:
    """Return the 16 Primary series of experiment 87's table with their H3 cell and coordinates."""
    series = common.base.series_table().filter(pl.col("substation_type") == "Primary")
    return series.select("time_series_id", "site", "h3_res_5", "latitude", "longitude")


# --- one run ------------------------------------------------------------------------------------


def half_hourly_members(
    *, run: datetime, targets: pl.DataFrame, version: int | None, source_columns: dict[str, str]
) -> pl.DataFrame:
    """Return one run's members on the half-hourly grid for the target series.

    Args:
        run: The run's 00:00 UTC initialisation time.
        targets: One row per series with `time_series_id`, `h3_res_5`, `latitude`, `longitude`.
        version: The pinned Delta version, or `None` for the latest.
        source_columns: Output name to table column, for the variables to read.

    Returns:
        Rows keyed by `time_series_id`, `ensemble_member`, `nwp_init_time`,
        `power_fcst_init_time`, and `valid_time`, with the variables under their output names.
        Rows at or before the origin are dropped.

    Raises:
        ValueError: If the table holds no rows of the run for these cells.
    """
    cells = targets["h3_res_5"].unique().to_list()
    scan = (
        common.scan_nwp(version=version, run=run)
        .filter(pl.col("h3_index").is_in(cells))
        .select("init_time", "h3_index", "ensemble_member", "valid_time", *source_columns.values())
    )
    native = (
        common.typed(scan=scan)
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
        .rename({source: name for name, source in source_columns.items()})
    )
    if half_hourly.is_empty():
        msg = f"the NWP table holds no rows of run {run:%Y-%m-%d} for these cells"
        raise ValueError(msg)
    origin = common.origin_of(run=run)
    return half_hourly.with_columns(
        power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC"))
    ).filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))


def cloud_stats(*, half_hourly: pl.DataFrame, cloud_column: str = "cloud") -> pl.DataFrame:
    """Return the four cloud columns per series and valid time from the members' cloud cover.

    Args:
        half_hourly: Output of `half_hourly_members` with the cloud cover (0 to 100) under
            `cloud_column`.
        cloud_column: The name of the cloud-cover column.

    Returns:
        One row per `time_series_id`, `nwp_init_time`, `power_fcst_init_time`, and `valid_time`,
        with `n_members`, `n_null`, and the four `common.CLOUD_FEATURES`: mean and population
        standard deviation divided by 100, and the member shares at or below 20% and at or above
        80%. Each statistic skips a null member, so a share is over the non-null members, and
        `n_null` must be zero.
    """
    percent = pl.col(cloud_column)
    return (
        half_hourly.group_by(
            "time_series_id", "nwp_init_time", "power_fcst_init_time", "valid_time"
        )
        .agg(
            pl.col("ensemble_member").n_unique().alias("n_members"),
            percent.null_count().alias("n_null"),
            (percent.mean() / 100.0).alias("cloud_mean"),
            (percent.std(ddof=0) / 100.0).alias("cloud_spread"),
            (percent <= common.CLEAR_MAX_PERCENT).mean().alias("cloud_clear_share"),
            (percent >= common.OVERCAST_MIN_PERCENT).mean().alias("cloud_overcast_share"),
        )
        .sort("time_series_id", "valid_time")
    )


def validate_run_table(*, table: pl.DataFrame, run: datetime, n_series: int) -> pl.DataFrame:
    """Check one freshly built run table and return it in the cache layout.

    Args:
        table: Output of `cloud_stats`.
        run: The run that the table was built from.
        n_series: How many series the table should hold.

    Returns:
        The table without `n_members` and `n_null`, from `validate_cache_table`.

    Raises:
        ValueError: If a valid time lacks a member or holds a null cloud value, or the table fails
            `validate_cache_table`.
    """
    label = f"run {run:%Y-%m-%d}"
    if (table["n_members"] != common.N_MEMBERS).any():
        msg = f"{label}: a valid time holds fewer than {common.N_MEMBERS} members"
        raise ValueError(msg)
    if (table["n_null"] != 0).any():
        msg = f"{label}: {int((table['n_null'] > 0).sum())} valid times hold a null cloud value"
        raise ValueError(msg)
    if float(table["cloud_mean"].max()) <= MIN_PLAUSIBLE_MAX_MEAN:  # ty: ignore[invalid-argument-type]
        msg = f"{label}: no cloud mean above {MIN_PLAUSIBLE_MAX_MEAN}: is cloud a fraction?"
        raise ValueError(msg)
    return validate_cache_table(table=table.drop("n_members", "n_null"), run=run, n_series=n_series)


def validate_cache_table(*, table: pl.DataFrame, run: datetime, n_series: int) -> pl.DataFrame:
    """Check the layout and values of one run's cache table and return it with `Float32` columns.

    Args:
        table: Rows keyed by series, run, origin, and valid time, with the four cloud columns.
        run: The run that the table was built from.
        n_series: How many series the table should hold.

    Returns:
        The table with the four cloud columns cast to `Float32`.

    Raises:
        ValueError: If a cloud column is null or not finite, a mean, spread, or share is outside its
            range, the longest lead is not 351 hours, a series lacks 702 valid times, or a row reads
            a run issued less than 9 hours before its origin.
    """
    label = f"run {run:%Y-%m-%d}"
    values = table.select(common.CLOUD_FEATURES)
    if values.null_count().to_numpy().any() or not np.isfinite(values.to_numpy()).all():
        msg = f"{label}: a cloud column is null or not finite"
        raise ValueError(msg)
    in_range = table.select(
        (
            pl.col("cloud_mean").is_between(0.0, 1.0)
            & (pl.col("cloud_spread") >= 0.0)
            & pl.col("cloud_clear_share").is_between(0.0, 1.0)
            & pl.col("cloud_overcast_share").is_between(0.0, 1.0)
            & (pl.col("cloud_clear_share") + pl.col("cloud_overcast_share") <= 1.0 + 1e-6)
        ).all()
    ).item()
    if not in_range:
        msg = f"{label}: a cloud mean, spread, or share is outside its range"
        raise ValueError(msg)
    lead = (table["valid_time"] - table["power_fcst_init_time"]).dt.total_minutes() / 60.0
    if float(lead.max()) != MAX_LEAD_HOURS:  # ty: ignore[invalid-argument-type]
        msg = f"{label}: the longest lead is {lead.max()} hours, not {MAX_LEAD_HOURS}"
        raise ValueError(msg)
    if (
        table.height != n_series * VALID_TIMES_PER_RUN
        or table.select(pl.struct(common.KEYS).is_duplicated().any()).item()
    ):
        msg = f"{label}: duplicate keys or a height other than {n_series} x {VALID_TIMES_PER_RUN}"
        raise ValueError(msg)
    counts = table.group_by("time_series_id").agg(n=pl.col("valid_time").n_unique())
    if counts.height != n_series or (counts["n"] != VALID_TIMES_PER_RUN).any():
        msg = f"{label}: not {n_series} series with {VALID_TIMES_PER_RUN} valid times each"
        raise ValueError(msg)
    origin_rules.assert_primary_rows(frame=table)
    return table.cast(dict.fromkeys(common.CLOUD_FEATURES, pl.Float32))


def run_cloud_table(*, run: datetime, version: int | None) -> pl.DataFrame:
    """Return one run's validated cloud table for the 16 Primaries, read from the Delta table."""
    targets = primaries()
    half_hourly = half_hourly_members(
        run=run,
        targets=targets,
        version=version,
        source_columns={"cloud": common.CLOUD_COLUMN},
    )
    return validate_run_table(
        table=cloud_stats(half_hourly=half_hourly), run=run, n_series=targets.height
    )


def write_cache_file(*, run: datetime, version: int | None) -> tuple[str, bool]:
    """Write one run's cache file if absent; return a progress line and whether the run failed."""
    target = common.cache_path(run=run)
    if target.exists():
        return f"{run:%Y-%m-%d}: exists", False
    started = time.monotonic()
    try:
        table = run_cloud_table(run=run, version=version)
    except ValueError as error:
        return f"{run:%Y-%m-%d}: SKIPPED, {error}", True
    common.require_one_polars_thread()
    table = table.with_columns(delta_version=pl.lit(version, dtype=pl.Int32))
    common.base.atomic_write(frame=table, target=target)
    return f"{run:%Y-%m-%d}: {table.height:,} rows, {time.monotonic() - started:.0f} s", False


def _write_cache_file_pair(job: tuple[datetime, int | None]) -> tuple[str, bool]:
    """Return `write_cache_file` for one (run, version) pair, as a picklable function."""
    run, version = job
    return write_cache_file(run=run, version=version)


# --- pin, build ---------------------------------------------------------------------------------


def _git(*args: str) -> str:
    """Return the output of a git command run in this folder, or `unknown` if git fails."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def _sha256(*, path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_lineage() -> dict[str, object]:
    """Return the code commit, the package versions, and the hashes of the scripts that build."""
    folder = Path(__file__).parent
    scripts = [*sorted(folder.glob("e00711_*.py"))]
    for name in ("e00707_common", "e00707_arms"):
        module = sys.modules.get(name)
        if module is not None and module.__file__:
            scripts.append(Path(module.__file__))
    packages = ("polars", "deltalake", "numpy", "contracts", "ml-core", "h3")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", str(folder))),
        "package_versions": {name: importlib.metadata.version(name) for name in packages},
        "script_sha256": {path.name: _sha256(path=path) for path in scripts},
    }


def stale_cache_files(*, version: int) -> list[Path]:
    """Return the cache files that were built at a Delta version other than `version`."""
    stale = []
    for path in sorted(common.CLOUD_DIR.glob("init=*.parquet")):
        built = pl.read_parquet(path, columns=["delta_version"])["delta_version"].unique()
        if built.len() != 1 or built.item() != version:
            stale.append(path)
    return stale


def pin(*, version: int | None, force: bool, drop_stale: bool) -> None:
    """Write `lineage.json` with the Delta version that every later read uses.

    Re-pinning at a different version needs `force`. It also refuses while the cache holds a file
    built at another version, unless `drop_stale` deletes those files so that `build` rebuilds
    them. A cache never mixes versions.
    """
    common.assert_cloud_in_schema()
    if common.LINEAGE_PATH.exists() and not force:
        recorded = common.read_pinned_version()
        print(f"already pinned at Delta version {recorded}; pass --force to re-pin")
        return
    chosen = common.table_version() if version is None else version
    stale = stale_cache_files(version=chosen) if common.CLOUD_DIR.exists() else []
    if stale and not drop_stale:
        msg = (
            f"{len(stale)} cache files were built at another Delta version than {chosen}: "
            "pass --drop-stale to delete them, then rebuild"
        )
        raise SystemExit(msg)
    for path in stale:
        path.unlink()
    runs = common.cache_runs()
    lineage = {
        "delta_version": chosen,
        "pinned_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "table": "NWP Delta table, nwp_model_id=" + common.NWP_MODEL_ID,
        "column": common.CLOUD_COLUMN,
        "reader": "pl.scan_delta(version=...) cast to the production Nwp dtypes",
        "resampler": "ml_core.features._nwp._upsample_nwp_to_half_hourly",
        "features": list(common.CLOUD_FEATURES),
        "clear_max_percent": common.CLEAR_MAX_PERCENT,
        "overcast_min_percent": common.OVERCAST_MIN_PERCENT,
        "cache_runs": len(runs),
        "cache_first_run": f"{runs[0]:%Y-%m-%d}",
        "cache_last_run": f"{runs[-1]:%Y-%m-%d}",
        "no_file_for": "2024-01-03 and 2024-04-01 to 2025-11-20",
        "deleted_stale_files": len(stale),
        "reproducibility": (
            "delta-rs keeps old versions for 30 days by default and a VACUUM deletes their files, "
            "so the cache can be rebuilt from this version only for a limited time"
        ),
        **code_lineage(),
    }
    common.atomic_write_text(text=json.dumps(lineage, indent=2) + "\n", target=common.LINEAGE_PATH)
    print(f"pinned Delta version {chosen}; lineage in {common.LINEAGE_PATH}")


def build(
    *, workers: int, limit: int | None, last_run: datetime, back_filled_only: bool = False
) -> None:
    """Write the cache file of every run that carries cloud and lacks a file.

    The build refuses unless `table-check`, `older-columns`, and `era-check` have each recorded a
    pass at the pinned Delta version, and unless Polars runs on one thread. A run that fails its
    checks is skipped and listed at the end. The exit status is 1 if any run was skipped.

    Args:
        workers: The number of single-threaded worker processes.
        limit: Build at most this many files, or `None` for all.
        last_run: The last run to build.
        back_filled_only: Skip the three checks, for runs to 2024-03-31 only, when the manager has
            agreed to build them before the re-write ends.

    Raises:
        SystemExit: If a check has no recorded pass, the pin is stale, or `back_filled_only` is
            combined with a later last run.
    """
    common.require_one_polars_thread()
    version = common.read_pinned_version()
    if back_filled_only:
        if last_run > common.BACKFILL_LAST_RUN:
            msg = "--back-filled-only builds runs to 2024-03-31 only"
            raise SystemExit(msg)
    else:
        failing = common.checks_pass_at(version=version)
        if failing:
            msg = "build refused: " + "; ".join(failing)
            raise SystemExit(msg)
    stale = stale_cache_files(version=version) if common.CLOUD_DIR.exists() else []
    if stale:
        msg = f"{len(stale)} cache files were built at another Delta version: re-pin --drop-stale"
        raise SystemExit(msg)
    todo = [r for r in common.cache_runs(last=last_run) if not common.cache_path(run=r).exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"Delta version {version}; {len(todo)} cache files to write")
    skipped: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for line, failed in pool.map(_write_cache_file_pair, [(r, version) for r in todo]):
            print(line, flush=True)
            if failed:
                skipped.append(line)
    if skipped:
        print(f"{len(skipped)} runs skipped:")
        for line in skipped:
            print(f"  {line}")
        raise SystemExit(1)


def dry_run(*, run: datetime, version: int | None) -> None:
    """Build one run in memory, time it, print its schema and example row, and write nothing."""
    started = time.monotonic()
    table = run_cloud_table(run=run, version=version)
    print(
        f"{run:%Y-%m-%d}: {table.height:,} rows, {table['time_series_id'].n_unique()} series, "
        f"{time.monotonic() - started:.0f} s (nothing written)"
    )
    print(table.schema)
    print(table.select(common.CLOUD_FEATURES).describe())
    series = table["time_series_id"].min()
    example_run = origin_rules.PRIMARY_EXAMPLE_RUN
    if run == example_run:
        print(origin_rules.primary_example(frame=table.filter(pl.col("time_series_id") == series)))


# --- the conditions on the table ----------------------------------------------------------------


def table_run_stats(*, version: int | None) -> pl.DataFrame:
    """Return, per run in the table, the rows, members, null cloud values, and cloud range."""
    cloud = pl.col(common.CLOUD_COLUMN)
    return (
        common.scan_nwp(version=version)
        .select("init_time", "ensemble_member", common.CLOUD_COLUMN)
        .group_by("init_time")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("ensemble_member").n_unique().alias("n_members"),
            cloud.null_count().alias("n_null"),
            cloud.is_nan().sum().alias("n_nan"),
            cloud.min().alias("cloud_min"),
            cloud.max().alias("cloud_max"),
        )
        .sort("init_time")
        .collect(engine="streaming")
    )


def presence_problems(*, stats: pl.DataFrame) -> list[str]:
    """Return every way that the per-run statistics break the plan's conditions on the table.

    Args:
        stats: Output of `table_run_stats`.

    Returns:
        A list of problems; empty when every condition holds. The conditions are: every run from
        2020-10-15 to 2026-11-18 except 2024-01-03 is present with 51 members; no run outside
        2024-04-01 to 2025-11-20 holds a null cloud value; cloud lies in 0 to 100; and the runs
        2025-11-21 to 2026-06-30 each hold 7,243,785 rows.
    """
    problems: list[str] = []
    held = dict(zip(stats["init_time"], stats.iter_rows(named=True), strict=True))
    expected = [
        r
        for r in common.runs_between(
            first=common.CACHE_FIRST_RUN, last=common.TABLE_REQUIRED_LAST_RUN
        )
        if r != common.UPSTREAM_GAP_RUN
    ]
    absent = [r for r in expected if r not in held]
    if absent:
        problems.append(f"{len(absent)} runs are absent, first {absent[0]:%Y-%m-%d}")
    if common.UPSTREAM_GAP_RUN in held:
        problems.append("2024-01-03 is present, but the upstream gap should leave it absent")
    for run, row in held.items():
        label = f"{run:%Y-%m-%d}"
        in_gap = common.GAP_FIRST_RUN <= run <= common.GAP_LAST_RUN
        if row["n_members"] != common.N_MEMBERS:
            problems.append(f"{label}: {row['n_members']} members, not {common.N_MEMBERS}")
        if not in_gap and row["n_null"] != 0:
            problems.append(f"{label}: {row['n_null']} null cloud values outside the gap")
        if row["cloud_min"] is not None and row["cloud_min"] < 0.0:
            problems.append(f"{label}: cloud min {row['cloud_min']} is below 0")
        if row["cloud_max"] is not None and row["cloud_max"] > 100.0:
            problems.append(f"{label}: cloud max {row['cloud_max']} is above 100")
        if not in_gap and row["cloud_max"] is not None and row["cloud_max"] <= 1.0:
            problems.append(f"{label}: cloud max {row['cloud_max']} is at most 1: a fraction?")
        if not in_gap and row.get("n_nan", 0):
            problems.append(f"{label}: {row['n_nan']} NaN cloud values outside the gap")
    recent = stats.filter(
        pl.col("init_time").is_between(common.RECENT_FIRST_RUN, common.SCORED_LAST_RUN)
    )
    if recent.height != (common.SCORED_LAST_RUN - common.RECENT_FIRST_RUN).days + 1:
        problems.append(f"{recent.height} recent runs, not 222")
    wrong_rows = recent.filter(pl.col("n_rows") != common.EXPECTED_RECENT_ROWS_PER_RUN)
    if wrong_rows.height:
        first = wrong_rows.row(0, named=True)
        problems.append(
            f"{wrong_rows.height} recent runs do not hold {common.EXPECTED_RECENT_ROWS_PER_RUN:,} "
            f"rows, first {first['init_time']:%Y-%m-%d} with {first['n_rows']:,}"
        )
    return problems


def table_check() -> None:
    """Check the table at the pinned version against the plan's conditions and record the report."""
    version = common.read_pinned_version()
    stats = table_run_stats(version=version)
    problems = presence_problems(stats=stats)
    in_gap = stats.filter(pl.col("init_time").is_between(common.GAP_FIRST_RUN, common.GAP_LAST_RUN))
    print(
        f"Delta version {version}: {stats.height} runs from {stats['init_time'].min():%Y-%m-%d} "
        f"to {stats['init_time'].max():%Y-%m-%d}; {in_gap.height} runs in the gap, "
        f"{int(in_gap['n_null'].sum()):,} null cloud values there"
    )
    common.record_check(
        name="table_check",
        version=version,
        passed=not problems,
        details={"problems": problems, "n_runs": stats.height},
    )
    if problems:
        print(f"TABLE CHECK FAILED ({len(problems)} problems); stop and report to the manager")
        for line in problems[:40]:
            print(f"  {line}")
        raise SystemExit(1)
    print("table check passed")


def older_columns_check() -> None:
    """Recompute the seven older variables for sampled runs and compare with the stored files.

    The sample holds `OLDER_SAMPLE_RUNS` runs, half from the back-filled runs and half from the
    runs from 2025-11-21, drawn with a fixed seed from the runs that have a stored summary.
    The exit status is 1 if any mean or spread differs by more than `OLDER_RTOL` relative to the
    largest absolute stored value of that column over all series (about 10 Pa for surface
    pressure).
    """
    version = common.read_pinned_version()
    rng = np.random.default_rng(711)
    half = OLDER_SAMPLE_RUNS // 2
    pools = (
        [r for r in common.cache_runs(last=common.BACKFILL_LAST_RUN) if _stored(run=r)],
        [r for r in common.cache_runs(first=common.RECENT_FIRST_RUN) if _stored(run=r)],
    )
    sample = [pool[i] for pool in pools for i in rng.choice(len(pool), size=half, replace=False)]
    targets = primaries()
    weather = common.base.WEATHER
    failed = False
    for run in sorted(sample):
        members = half_hourly_members(
            run=run, targets=targets, version=version, source_columns=OLDER_SOURCE_COLUMNS
        ).with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt()
        )
        keys = ["time_series_id", "valid_time"]
        fresh = members.group_by(keys).agg(
            *[pl.col(w).mean().alias(f"mean_{w}") for w in weather],
            *[pl.col(w).std(ddof=0).alias(f"std_{w}") for w in weather],
        )
        stored = pl.read_parquet(_stored_path(run=run))
        joined = fresh.join(stored, on=keys, how="inner", suffix="_stored")
        worst = 0.0
        for column in [f"{k}_{w}" for k in ("mean", "std") for w in weather]:
            scale = max(float(joined[f"{column}_stored"].abs().max()), 1e-6)  # ty: ignore[invalid-argument-type]
            largest = float((joined[column] - joined[f"{column}_stored"]).abs().max())  # ty: ignore[invalid-argument-type]
            worst = max(worst, largest / scale)
        flag = "ok" if joined.height == stored.height and worst <= OLDER_RTOL else "FAIL"
        failed |= flag == "FAIL"
        print(
            f"{run:%Y-%m-%d}: {joined.height:,}/{stored.height:,} rows, "
            f"worst diff {worst:.2e}: {flag}"
        )
    common.record_check(
        name="older_columns", version=version, passed=not failed, details={"n_runs": len(sample)}
    )
    if failed:
        print("OLDER COLUMNS CHANGED: stop and report to the manager")
        raise SystemExit(1)
    print("older columns match the stored summaries for all sampled runs")


def _stored_path(*, run: datetime) -> Path:
    """Return the stored seven-variable summary of a run."""
    directory = OLD_SUMMARY_DIRS[0] if run <= common.BACKFILL_LAST_RUN else OLD_SUMMARY_DIRS[1]
    return directory / f"init={run:%Y-%m-%d}.parquet"


def _stored(*, run: datetime) -> bool:
    """Return whether a run has a stored seven-variable summary."""
    return _stored_path(run=run).exists()


# --- the cache ----------------------------------------------------------------------------------


def stuck_problems(*, table: pl.DataFrame, run: datetime) -> list[str]:
    """Return the series whose `cloud_mean` takes one value over the whole run (a stuck series)."""
    counts = table.group_by("time_series_id").agg(n=pl.col("cloud_mean").n_unique())
    stuck = counts.filter(pl.col("n") == 1)["time_series_id"].to_list()
    return [f"run {run:%Y-%m-%d}: series {s} holds one cloud_mean value" for s in stuck]


def content_hash(*, table: pl.DataFrame) -> str:
    """Return a SHA-256 of a run's cloud columns, sorted by key, so identical runs share a hash."""
    values = table.sort(*common.KEYS).select(common.CLOUD_FEATURES).to_numpy()
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def duplicate_content_problems(*, hashes: dict[str, str]) -> list[str]:
    """Return the groups of files that hold identical cloud values."""
    by_hash: dict[str, list[str]] = {}
    for name, digest in hashes.items():
        by_hash.setdefault(digest, []).append(name)
    return [
        f"identical cloud values in {sorted(names)}" for names in by_hash.values() if len(names) > 1
    ]


def check_cache() -> None:
    """Check the written files, print the example row, and cross-check against experiment 561.

    The checks are: one file per run that carries cloud and none for a gap run; every file built at
    the pinned Delta version; the same keys and row count in every file, with no duplicate key and
    no null; the origin rule; no series stuck at one value; no two runs with identical cloud
    values; and the cross-check of `cross_check_561`. It writes the hour-of-day profile, the monthly
    means and spreads, and the SHA-256 and row count of every file under `checks/`.

    Raises:
        ValueError: If a file is missing or unexpected, or a file or the set of files fails a check.
    """
    version = common.read_pinned_version()
    expected = {common.cache_path(run=r).name for r in common.cache_runs()}
    present = {p.name for p in common.CLOUD_DIR.glob("init=*.parquet")}
    if present - expected:
        msg = f"unexpected files, first {min(present - expected)}"
        raise ValueError(msg)
    missing = sorted(expected - present)
    print(f"{len(present)} cache files; {len(missing)} of {len(expected)} expected runs missing")
    n_series = primaries().height
    problems: list[str] = []
    hashes: dict[str, str] = {}
    hourly: list[pl.DataFrame] = []
    monthly: list[pl.DataFrame] = []
    for name in sorted(present):
        run = datetime.fromisoformat(name.removeprefix("init=").removesuffix(".parquet"))
        table = pl.read_parquet(common.CLOUD_DIR / name)
        if table["delta_version"].unique().to_list() != [version]:
            problems.append(f"{name}: not built at Delta version {version}")
        table = table.drop("delta_version")
        validate_cache_table(table=table, run=run.replace(tzinfo=UTC), n_series=n_series)
        if (table["nwp_init_time"] != run.replace(tzinfo=UTC)).any():
            msg = f"{name} holds rows of another run"
            raise ValueError(msg)
        problems += stuck_problems(table=table, run=run)
        hashes[name] = content_hash(table=table)
        hourly.append(
            table.group_by(hour=pl.col("valid_time").dt.hour()).agg(
                pl.col("cloud_mean").sum().alias("mean_sum"),
                pl.col("cloud_spread").sum().alias("spread_sum"),
                pl.len().alias("n"),
            )
        )
        monthly.append(
            table.select(
                month=pl.col("valid_time").dt.truncate("1mo").dt.date(),
                mean_sum=pl.col("cloud_mean"),
                spread_sum=pl.col("cloud_spread"),
            )
            .group_by("month")
            .agg(pl.col("mean_sum").sum(), pl.col("spread_sum").sum(), pl.len().alias("n"))
        )
    problems += duplicate_content_problems(hashes=hashes)
    if hourly:
        profile = _mean_table(parts=hourly, key="hour")
        if profile.height != 24:
            problems.append(f"the hour-of-day profile has {profile.height} hours, not 24")
        common.CHECK_DIR.mkdir(parents=True, exist_ok=True)
        profile.write_csv(common.CHECK_DIR / "cache_hour_of_day_profile.csv")
        _mean_table(parts=monthly, key="month").write_csv(common.CHECK_DIR / "cache_monthly.csv")
        pl.DataFrame({"file": list(hashes), "sha256": list(hashes.values())}).write_csv(
            common.CHECK_DIR / "cache_file_hashes.csv"
        )
        print(profile)
    example = common.cache_path(run=origin_rules.PRIMARY_EXAMPLE_RUN)
    if example.exists():
        frame = pl.read_parquet(example)
        series = frame["time_series_id"].min()
        print(origin_rules.primary_example(frame=frame.filter(pl.col("time_series_id") == series)))
    cross_check_561(present=present)
    if problems:
        msg = f"{len(problems)} cache problems, first: {problems[0]}"
        raise ValueError(msg)
    print("cache check passed")


def _mean_table(*, parts: list[pl.DataFrame], key: str) -> pl.DataFrame:
    """Return the mean cloud mean and spread per `key` from the per-file sums."""
    return (
        pl.concat(parts)
        .group_by(key)
        .agg(pl.col("mean_sum").sum(), pl.col("spread_sum").sum(), pl.col("n").sum())
        .select(
            key,
            mean_cloud_mean=pl.col("mean_sum") / pl.col("n"),
            mean_cloud_spread=pl.col("spread_sum") / pl.col("n"),
            n=pl.col("n"),
        )
        .sort(key)
    )


def cross_check_561(*, present: set[str]) -> None:
    """Compare the ensemble means and spreads with experiment 561's cloud files, and record it.

    Args:
        present: The names of the cache files that exist.

    Raises:
        ValueError: If recent runs are cached but none overlaps experiment 561's files, or the
            correlation of the ensemble means is below `MIN_CROSS_CHECK_CORRELATION`.
    """
    recent_cached = [
        n
        for n in present
        if datetime.fromisoformat(n.removeprefix("init=").removesuffix(".parquet")).replace(
            tzinfo=UTC
        )
        >= common.RECENT_FIRST_RUN
    ]
    xs: list[pl.DataFrame] = []
    for path in sorted(EXPERIMENT_561_DIR.glob("init=*.parquet")):
        run = datetime.fromisoformat(path.stem.removeprefix("init=")).replace(tzinfo=UTC)
        mine = common.cache_path(run=run)
        if not mine.exists():
            continue
        theirs = pl.read_parquet(path).select(
            *common.KEYS, "mean_total_cloud_cover", "std_total_cloud_cover"
        )
        xs.append(
            pl.read_parquet(mine)
            .join(theirs, on=list(common.KEYS), how="inner")
            .select(
                a=pl.col("cloud_mean") * 100.0,
                b="mean_total_cloud_cover",
                sa=pl.col("cloud_spread") * 100.0,
                sb="std_total_cloud_cover",
            )
        )
    if not xs:
        if recent_cached:
            msg = "recent runs are cached but none overlaps experiment 561's cloud files"
            raise ValueError(msg)
        print("no recent runs cached; cross-check against experiment 561 not applicable yet")
        return
    both = pl.concat(xs)
    correlation = float(both.select(pl.corr("a", "b")).item())
    result = {
        "runs": len(xs),
        "rows": both.height,
        "mean_correlation": correlation,
        "mean_abs_difference_points": float(
            both.select((pl.col("a") - pl.col("b")).abs().mean()).item()
        ),
        "spread_correlation": float(both.select(pl.corr("sa", "sb")).item()),
        "spread_mean_abs_difference_points": float(
            both.select((pl.col("sa") - pl.col("sb")).abs().mean()).item()
        ),
    }
    common.atomic_write_text(
        text=json.dumps(result, indent=2) + "\n", target=common.CHECK_DIR / "cross_check_561.json"
    )
    print(f"cross-check against experiment 561: {result}")
    if not correlation >= MIN_CROSS_CHECK_CORRELATION:
        msg = (
            f"correlation {correlation:.4f} with experiment 561 is below "
            f"{MIN_CROSS_CHECK_CORRELATION}"
        )
        raise ValueError(msg)


# --- the era check ------------------------------------------------------------------------------


def step_failures(*, monthly: pl.DataFrame) -> pl.DataFrame:
    """Return the step tests of plan section 4 for each series and quantity.

    Cloud bias is seasonal, so the first test compares the same calendar months across eras:
    December to March of 2021-22, 2022-23, and 2023-24 against December 2025 to March 2026. The
    second tests the resolution change of 2023-06-27 by pairing each month from 2023-07 to 2024-03
    with the same calendar month a year earlier. Each test reports two scales for its step:

    - `scale_literal`: the standard deviation of the adjacent 12 monthly values, as the plan says.
      It includes the seasonal cycle, so it is lenient.
    - `scale`: the same spread with the seasonal cycle removed. For the winters test it is the
      residual standard deviation of the old months about their calendar-month means. For the
      resolution test it is the standard deviation of the year-over-year differences in the
      12 months before the change. `sd_se` divides the step by the standard error of the step.

    A test fails when either `sd_literal` or `sd` exceeds 3, so the deseasonalised reading decides
    where it is stricter.

    Args:
        monthly: Rows with `series`, `month` (the first day of the month as a date), `diff` (ENS
            cloud minus ERA5 cloud, points) and `spread` (ENS spread, points).

    Returns:
        One row per series, quantity (`diff` or `spread`), and test, with `step`, `scale_literal`,
        `sd_literal`, `scale`, `sd`, `sd_se`, and `failed`.
    """
    rows: list[dict[str, object]] = []
    for (series,), part in monthly.group_by("series", maintain_order=True):
        by_month = {m: i for i, m in enumerate(part["month"].to_list())}
        for quantity in ("diff", "spread"):
            values = part[quantity].to_numpy()
            old_months = [
                m for year in OLD_WINTERS for m in _winter_months(first_year=year) if m in by_month
            ]
            old = np.array([values[by_month[m]] for m in old_months])
            recent = [
                values[by_month[m]]
                for m in _winter_months(first_year=RECENT_WINTER)
                if m in by_month
            ]
            if len(old) >= 6 and recent:
                calendar = np.array([m.month for m in old_months])
                residuals = old - np.array([old[calendar == k].mean() for k in calendar])
                dof = max(len(old) - len(set(calendar.tolist())), 1)
                scale = float(np.sqrt((residuals**2).sum() / dof))
                rows.append(
                    _step_row(
                        series=series,
                        quantity=quantity,
                        test="winters",
                        step=abs(float(np.mean(recent) - old.mean())),
                        scale_literal=float(np.std(old, ddof=1)),
                        scale=scale,
                        standard_error=scale * float(np.sqrt(1 / len(recent) + 1 / len(old))),
                    )
                )
            after = [m for m in by_month if date(2023, 7, 1) <= m <= date(2024, 3, 1)]
            pairs = [
                values[by_month[m]] - values[by_month[m.replace(year=m.year - 1)]]
                for m in after
                if m.replace(year=m.year - 1) in by_month
            ]
            adjacent = [m for m in by_month if date(2022, 6, 1) <= m <= date(2023, 5, 1)]
            before_pairs = [
                values[by_month[m]] - values[by_month[m.replace(year=m.year - 1)]]
                for m in adjacent
                if m.replace(year=m.year - 1) in by_month
            ]
            if len(pairs) >= 3 and len(before_pairs) >= 3:
                scale = float(np.std(before_pairs, ddof=1))
                rows.append(
                    _step_row(
                        series=series,
                        quantity=quantity,
                        test="resolution_change",
                        step=abs(float(np.mean(pairs))),
                        scale_literal=float(
                            np.std([values[by_month[m]] for m in adjacent], ddof=1)
                        ),
                        scale=scale,
                        standard_error=scale / float(np.sqrt(len(pairs))),
                    )
                )
    return pl.DataFrame(
        rows,
        schema={
            "series": pl.String,
            "quantity": pl.String,
            "test": pl.String,
            "step": pl.Float64,
            "scale_literal": pl.Float64,
            "sd_literal": pl.Float64,
            "scale": pl.Float64,
            "sd": pl.Float64,
            "sd_se": pl.Float64,
            "failed": pl.Boolean,
        },
    )


def _winter_months(*, first_year: int) -> list[date]:
    """Return the first days of December to March of the winter that starts in `first_year`."""
    return [
        date(first_year, 12, 1),
        date(first_year + 1, 1, 1),
        date(first_year + 1, 2, 1),
        date(first_year + 1, 3, 1),
    ]


def _step_row(
    *,
    series: object,
    quantity: str,
    test: str,
    step: float,
    scale_literal: float,
    scale: float,
    standard_error: float,
) -> dict[str, object]:
    """Return one row of `step_failures`."""
    sd_literal = step / max(scale_literal, 1e-12)
    sd = step / max(scale, 1e-12)
    return {
        "series": series,
        "quantity": quantity,
        "test": test,
        "step": step,
        "scale_literal": scale_literal,
        "sd_literal": sd_literal,
        "scale": scale,
        "sd": sd,
        "sd_se": step / max(standard_error, 1e-12),
        "failed": sd_literal > STEP_SD_LIMIT or sd > STEP_SD_LIMIT,
    }


def series_cells(*, version: int | None) -> pl.DataFrame:
    """Return the 16 Primaries and 6 solar farms with their store H3 cell and a safe label.

    The solar farms carry their anonymous labels `A` to `F`; their coordinates never leave this
    process.
    """
    import h3
    from studies.pv_dataset import pv_sites

    store_cells = (
        common.scan_nwp(version=version, run=datetime(2025, 12, 1, tzinfo=UTC))
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in store_cells])
    farms = []
    for row in pv_sites().iter_rows(named=True):
        latitude, longitude = row["latitude"], row["longitude"]
        distance = (centres[:, 0] - latitude) ** 2 + (
            (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
        ) ** 2
        farms.append(
            {
                "time_series_id": row["time_series_id"],
                "series": f"farm {row['site']}",
                "h3_res_5": int(store_cells[int(np.argmin(distance))]),
            }
        )
    demand = primaries().select("time_series_id", pl.col("site").alias("series"), "h3_res_5")
    return pl.concat([demand, pl.DataFrame(farms).cast(demand.schema)])


def era_check() -> None:
    """Print the monthly ENS-minus-ERA5 cloud and ENS spread and the step tests.

    The ENS values are the ensemble mean and spread of native stamps at leads 12 to 36 hours,
    read from the pinned table, and the ERA5 cloud is a fraction that the check multiplies by 100.
    The exit status is 1 if any test fails.
    """
    version = common.read_pinned_version()
    cells = series_cells(version=version)
    ens = (
        common.typed(
            scan=common.scan_nwp(version=version)
            .filter(
                pl.col("h3_index").is_in(cells["h3_res_5"].to_list()),
                pl.col(common.CLOUD_COLUMN).is_not_null(),
                (pl.col("valid_time") - pl.col("init_time")).is_between(
                    timedelta(hours=STEP_LEADS_HOURS[0]), timedelta(hours=STEP_LEADS_HOURS[1])
                ),
            )
            .select("init_time", "valid_time", "h3_index", common.CLOUD_COLUMN)
        )
        .group_by("h3_index", "init_time", "valid_time")
        .agg(
            pl.col(common.CLOUD_COLUMN).mean().alias("ens_mean"),
            pl.col(common.CLOUD_COLUMN).std(ddof=0).alias("spread"),
        )
        .collect(engine="streaming")
    )
    era = pl.read_parquet(ERA5_PATH, columns=["time_series_id", "valid_time", "total_cloud_cover"])
    joined = (
        ens.join(cells, left_on="h3_index", right_on="h3_res_5", how="inner")
        .join(era, on=["time_series_id", "valid_time"], how="inner")
        .with_columns(
            diff=pl.col("ens_mean") - pl.col("total_cloud_cover") * 100.0,
            month=pl.col("valid_time").dt.truncate("1mo").dt.date(),
        )
    )
    monthly = joined.group_by("series", "month").agg(pl.col("diff").mean(), pl.col("spread").mean())
    results = step_failures(monthly=monthly.sort("series", "month"))
    common.CHECK_DIR.mkdir(parents=True, exist_ok=True)
    monthly.sort("series", "month").write_csv(common.CHECK_DIR / "era_check_monthly.csv")
    results.write_csv(common.CHECK_DIR / "era_check_steps.csv")
    print(
        f"{monthly['series'].n_unique()} series, {monthly['month'].n_unique()} months; "
        f"{int(results['failed'].sum())} of {results.height} step tests fail"
    )
    print(results.filter(pl.col("failed")))
    common.record_check(
        name="era_check",
        version=version,
        passed=not bool(results["failed"].any()),
        details={"n_tests": results.height, "n_failed": int(results["failed"].sum())},
    )
    if results["failed"].any():
        print("ERA CHECK FAILED: stop and report to the manager (an era indicator needs approval)")
        raise SystemExit(1)


def main() -> None:
    """Run one command of the cloud build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("pin", "table-check", "older-columns", "build", "check-cache", "era-check"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-pin an existing lineage note")
    parser.add_argument("--drop-stale", action="store_true", help="`pin --force`: delete old files")
    parser.add_argument(
        "--back-filled-only", action="store_true", help="`build` runs to 2024-03-31, no checks"
    )
    parser.add_argument("--delta-version", type=int, default=None)
    parser.add_argument(
        "--last-run", default=f"{common.SCORED_LAST_RUN:%Y-%m-%d}", help="last run to build"
    )
    parser.add_argument(
        "--dry-run-date",
        default=None,
        help="build this run in memory, print the schema, write nothing (`build` only)",
    )
    args = parser.parse_args()
    common.assert_cloud_in_schema()
    match args.command:
        case "pin":
            pin(version=args.delta_version, force=args.force, drop_stale=args.drop_stale)
        case "table-check":
            table_check()
        case "older-columns":
            older_columns_check()
        case "build" if args.dry_run_date is not None:
            run = datetime.fromisoformat(args.dry_run_date).replace(tzinfo=UTC)
            if common.is_missing_run(run=run):
                msg = f"{args.dry_run_date} carries no cloud: pick a run outside the gap"
                raise SystemExit(msg)
            pinned = common.read_pinned_version() if common.LINEAGE_PATH.exists() else None
            dry_run(run=run, version=args.delta_version or pinned)
        case "build":
            last = datetime.fromisoformat(args.last_run).replace(tzinfo=UTC)
            build(
                workers=args.workers,
                limit=args.limit,
                last_run=last,
                back_filled_only=args.back_filled_only,
            )
        case "check-cache":
            check_cache()
        case "era-check":
            era_check()


if __name__ == "__main__":
    main()
