"""Constants, run calendars, and the pinned NWP read of experiment 711.

The plan is in
`docs/ml-experiments/00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms/result.md`.
This module holds data and small pure functions only, and importing it reads no file. It imports
experiment 707's `e00707_common` and `e00707_arms` (neither imports `torch`), so run every script
of this folder with that folder on `PYTHONPATH`:

```bash
export DATA_PATH_INTERNAL=<path to the internal data folder>
POLARS_MAX_THREADS=1 \
PYTHONPATH=ml-experiments/00707-ens-backfill-pretraining-gain-against-ens-history-length \
    uv run --no-sync python ml-experiments/00711-*/e00711_cloud.py --help
```

**A run carries cloud unless it lies in the 19-month gap.** The NWP Delta table holds the column
`total_cloud_cover_atmosphere` as null for every run from 2024-04-01 to 2025-11-20. Run 2024-01-03
is absent upstream, so it has no rows at all. Both kinds of run count as missing in every arm, and
the missing pattern of a placebo copies the real pattern exactly.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00707_arms as arms  # ty: ignore[unresolved-import]
import e00707_common as base  # ty: ignore[unresolved-import]
import polars as pl
from contracts.weather_schemas import Nwp
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00711-ens-total-cloud-cover-for-primary-demand-and-solar-farms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder; every other experiment's folder is only read."""
CLOUD_DIR: Final[Path] = WORK_DIR / "work" / "ens_cloud"
"""One `init=<date>.parquet` file per run that carries cloud, plus `lineage.json`."""
LINEAGE_PATH: Final[Path] = CLOUD_DIR / "lineage.json"
PLACEBO_DIR: Final[Path] = WORK_DIR / "work" / "placebo"
CHECK_DIR: Final[Path] = WORK_DIR / "work" / "checks"
"""Where the table check, the era check, and the cross-check write their reports."""

NWP_TABLE_PATH: Final[Path] = REPO_DATA_DIR / "NWP"
NWP_MODEL_ID: Final[str] = "ECMWF_ENS_0_25_degree"
CLOUD_COLUMN: Final[str] = "total_cloud_cover_atmosphere"
N_MEMBERS: Final[int] = 51

CLOUD_FEATURES: Final[tuple[str, ...]] = (
    "cloud_mean",
    "cloud_spread",
    "cloud_clear_share",
    "cloud_overcast_share",
)
"""T1 is the first two columns and T2 is all four. Mean and spread are fractions (percent / 100)."""
T1_COLUMNS: Final[tuple[str, ...]] = CLOUD_FEATURES[:2]
CLEAR_MAX_PERCENT: Final[float] = 20.0
"""A member is clear when its cloud cover is at or below this percentage."""
OVERCAST_MIN_PERCENT: Final[float] = 80.0
"""A member is overcast when its cloud cover is at or above this percentage."""
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")

PUBLICATION_DELAY: Final[timedelta] = base.PUBLICATION_DELAY
"""A run counts as published this long after its initialisation."""

CACHE_FIRST_RUN: Final[datetime] = datetime(2020, 10, 15, tzinfo=UTC)
"""The first back-filled run. Runs before `ROW_FIRST_RUN` only serve as donors and lag sources."""
ROW_FIRST_RUN: Final[datetime] = arms.LONG_WINDOW_FLOOR
"""The first run that Primary demand rows read (2021-07-20)."""
BACKFILL_LAST_RUN: Final[datetime] = datetime(2024, 3, 31, tzinfo=UTC)
UPSTREAM_GAP_RUN: Final[datetime] = datetime(2024, 1, 3, tzinfo=UTC)
"""Absent from the table, so it has no rows and counts as missing."""
GAP_FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
GAP_LAST_RUN: Final[datetime] = datetime(2025, 11, 20, tzinfo=UTC)
RECENT_FIRST_RUN: Final[datetime] = datetime(2025, 11, 21, tzinfo=UTC)
SCORED_LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
"""The last run that either target scores, and the default last run of the cache."""
TABLE_REQUIRED_LAST_RUN: Final[datetime] = datetime(2026, 11, 18, tzinfo=UTC)
"""Every run to here must be in the table with cloud; 2026-11-19 is checked if it is present."""
TABLE_OPTIONAL_RUN: Final[datetime] = datetime(2026, 11, 19, tzinfo=UTC)
EXPECTED_RECENT_ROWS_PER_RUN: Final[int] = 7_243_785
"""Rows that each run from `RECENT_FIRST_RUN` to `SCORED_LAST_RUN` holds in the table.

The plan quotes the figure without saying "per run". The 222 runs hold 1,608,120,270 rows in total,
which is 222 times this figure."""

SOLAR_FIRST_RUN: Final[datetime] = CACHE_FIRST_RUN
SOLAR_ORIGIN_HOUR: Final[int] = 12
"""The solar origin is 12:00 UTC on the run's day."""


def runs_between(*, first: datetime, last: datetime) -> list[datetime]:
    """Return the daily 00:00 UTC runs from `first` to `last` inclusive."""
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def is_missing_run(*, run: datetime) -> bool:
    """Return whether a run has no cloud: it is in the 19-month gap or is the upstream gap run."""
    return GAP_FIRST_RUN <= run <= GAP_LAST_RUN or run == UPSTREAM_GAP_RUN


def cache_runs(
    *, first: datetime = CACHE_FIRST_RUN, last: datetime = SCORED_LAST_RUN
) -> list[datetime]:
    """Return the runs that carry cloud, from `first` to `last`, each of which has a cache file."""
    return [r for r in runs_between(first=first, last=last) if not is_missing_run(run=r)]


def cloud_missing_flags(*, runs: list[datetime]) -> list[bool]:
    """Return the `cloud_missing` flag of each run: 1 in the gap or at the upstream gap run."""
    return [is_missing_run(run=r) for r in runs]


def cache_path(*, run: datetime) -> Path:
    """Return the cache file of one run."""
    return CLOUD_DIR / f"init={run:%Y-%m-%d}.parquet"


def origin_of(*, run: datetime) -> datetime:
    """Return a Primary demand row's origin: the run plus the 9-hour publication delay."""
    return run + PUBLICATION_DELAY


def solar_origin_of(*, run: datetime) -> datetime:
    """Return a solar row's origin: 12:00 UTC on the run's day."""
    return run.replace(hour=SOLAR_ORIGIN_HOUR, minute=0, second=0, microsecond=0)


def assert_cloud_in_schema() -> None:
    """Raise if the pinned production `Nwp` schema lacks the cloud column.

    Raises:
        AssertionError: If `total_cloud_cover_atmosphere` is not a field of `Nwp`.
    """
    assert CLOUD_COLUMN in Nwp.model_fields, (
        f"{CLOUD_COLUMN} is not in Nwp.model_fields: run `uv sync --group gpu` in this worktree"
    )


def table_version() -> int:
    """Return the current version of the NWP Delta table."""
    from deltalake import DeltaTable

    return int(DeltaTable(str(NWP_TABLE_PATH)).version())


def scan_nwp(*, version: int | None, run: datetime | None = None) -> pl.LazyFrame:
    """Return a lazy scan of the NWP table, filtered to the model and optionally one run.

    The scan reads at a pinned Delta version, because `Nwp.scan_delta` takes no version. The
    filters sit directly on the scan so that Polars prunes to one partition, and the dtypes are cast
    afterwards by `typed`.

    Args:
        version: The Delta version to read, or `None` for the latest.
        run: A 00:00 UTC run to keep, or `None` for every run.

    Returns:
        A lazy frame with the table's stored dtypes.
    """
    assert_cloud_in_schema()
    scan = pl.scan_delta(str(NWP_TABLE_PATH), version=version).filter(
        pl.col("nwp_model_id") == NWP_MODEL_ID
    )
    if run is not None:
        scan = scan.filter(pl.col("init_time") == run)
    return scan


def typed(*, scan: pl.LazyFrame) -> pl.LazyFrame:
    """Cast the columns of a scan to the dtypes that the production `Nwp` contract declares."""
    return scan.cast({c: Nwp.dtypes[c] for c in scan.collect_schema().names()})


def read_pinned_version() -> int:
    """Return the Delta version recorded in the lineage note.

    Raises:
        FileNotFoundError: If the lineage note does not exist yet; run the `pin` command first.
    """
    if not LINEAGE_PATH.exists():
        msg = f"{LINEAGE_PATH} does not exist: run `e00711_cloud.py pin` first"
        raise FileNotFoundError(msg)
    return int(json.loads(LINEAGE_PATH.read_text())["delta_version"])


CHECK_NAMES: Final[tuple[str, ...]] = ("table_check", "older_columns", "era_check")
"""The checks that must have passed at the pinned version before `build` runs."""


def check_record_path(*, name: str) -> Path:
    """Return the file in which a check records its result."""
    return CHECK_DIR / f"{name}.json"


def record_check(*, name: str, version: int, passed: bool, details: dict[str, object]) -> None:
    """Record the result of a check at a Delta version."""
    record = {"delta_version": version, "passed": passed, **details}
    atomic_write_text(text=json.dumps(record, indent=2) + "\n", target=check_record_path(name=name))


def checks_pass_at(*, version: int) -> list[str]:
    """Return the checks that have not recorded a pass at `version` (empty when all have)."""
    failing: list[str] = []
    for name in CHECK_NAMES:
        path = check_record_path(name=name)
        if not path.exists():
            failing.append(f"{name}: no record")
            continue
        record = json.loads(path.read_text())
        if record.get("delta_version") != version or record.get("passed") is not True:
            failing.append(f"{name}: not a pass at Delta version {version}")
    return failing


def require_one_polars_thread() -> None:
    """Raise unless Polars runs on one thread, as `POLARS_MAX_THREADS=1` sets.

    Raises:
        RuntimeError: If the Polars thread pool has more than one thread.
    """
    if pl.thread_pool_size() != 1:
        msg = "set POLARS_MAX_THREADS=1 on the command line: each worker must use one thread"
        raise RuntimeError(msg)


def atomic_write_text(*, text: str, target: Path) -> None:
    """Write a text file through a temporary name so an interrupted write leaves no file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.rename(target)
