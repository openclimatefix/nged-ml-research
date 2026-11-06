"""Extend the cached inputs of experiment 605's list that need only CPU or a free fetch.

Builds, for runs 2026-07-01 to 2026-10-31, the ENS summary with wind components (experiment 561),
the ENS total cloud cover summary (experiment 561), and the observed-power cache (stage 1 pilot).
Output goes to `data/ml-experiments/00607-*/work/`; no earlier table is edited. The earlier runs'
files are hard-linked into the new folders so each folder is complete.

Run with the `PYTHONPATH` that `e00561_ens_summary_uv.py` shows plus the folder of experiment 561
and this folder:

    uv run --no-sync python e00607_build.py uv --workers 4
    uv run --no-sync python e00607_build.py tcc --workers 2
    uv run --no-sync python e00607_build.py observed
    uv run --no-sync python e00607_build.py link
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00561_ens_summary_uv as uv_build  # ty: ignore[unresolved-import]
import e00561_ens_tcc as tcc_build  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
FIRST_NEW_RUN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LAST_NEW_RUN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
UV_DIR: Final[Path] = WORK_DIR / "ens_summary_uv"
SUMMARY_DIR: Final[Path] = WORK_DIR / "ens_summary"
TCC_DIR: Final[Path] = WORK_DIR / "ens_summary_tcc"
TCC_NATIVE_DIR: Final[Path] = WORK_DIR / "tcc_native"
OLD_WORK: Final[Path] = prep.WORK_DIR / "work"
ORIGINAL_COLUMNS: Final[list[str]] = [
    f"{k}_{w}" for k in ("mean", "std") for w in prep.WEATHER
]  # fmt: skip


def new_runs() -> list[datetime]:
    """Return the daily 00:00 UTC runs 2026-07-01 to 2026-10-31."""
    return [FIRST_NEW_RUN + timedelta(days=i) for i in range((LAST_NEW_RUN - FIRST_NEW_RUN).days + 1)]


def _init_uv() -> None:
    uv_build.SUMMARY_DIR = UV_DIR


def _init_tcc() -> None:
    tcc_build.NATIVE_DIR = TCC_NATIVE_DIR
    tcc_build.SUMMARY_DIR = TCC_DIR


def _uv_one(run: datetime) -> str:
    line = uv_build.write_summary(run=run)
    # The 14 original columns equal experiment 397's `ens_summary` (experiment 561 checked this).
    target = SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if not target.exists():
        table = pl.read_parquet(UV_DIR / f"init={run:%Y-%m-%d}.parquet")
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        uv_build.common.atomic_write(frame=table.select(*keys, *ORIGINAL_COLUMNS), target=target)
    return line


def _tcc_one(run: datetime) -> str:
    return tcc_build.write_run(run=run)


def build_pool(*, which: str, workers: int, limit: int | None) -> None:
    """Build the new runs of `which` ("uv" or "tcc") in parallel."""
    runs = new_runs()[:limit]
    for folder in (UV_DIR, SUMMARY_DIR, TCC_DIR, TCC_NATIVE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    init, fn = (_init_uv, _uv_one) if which == "uv" else (_init_tcc, _tcc_one)
    with ProcessPoolExecutor(max_workers=workers, initializer=init) as pool:
        for line in pool.map(fn, runs):
            print(line, flush=True)


def link_old_runs() -> None:
    """Hard-link the earlier runs' files into the new folders so each folder is complete."""
    pairs = [
        (OLD_WORK / "ens_summary_uv", UV_DIR),
        (OLD_WORK / "ens_summary", SUMMARY_DIR),
        (OLD_WORK / "ens_summary_tcc", TCC_DIR),
        (OLD_WORK / "tcc_native", TCC_NATIVE_DIR),
    ]
    for old, new in pairs:
        new.mkdir(parents=True, exist_ok=True)
        linked = 0
        for path in old.glob("init=*.parquet"):
            target = new / path.name
            if not target.exists():
                os.link(path, target)
                linked += 1
        print(f"{new.name}: linked {linked} earlier files; folder holds {len(list(new.glob('init=*.parquet')))}")


def build_observed() -> None:
    """Rebuild the observed-power cache to the end of the power table, in a new file."""
    import demand_build_inputs  # ty: ignore[unresolved-import]
    import demand_inputs  # ty: ignore[unresolved-import]

    series = demand_inputs.series_table()
    frame = demand_build_inputs.observed_power(series=series)
    target = WORK_DIR / "observed.parquet"
    frame.write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)
    print(frame.group_by("time_series_id").agg(newest=pl.col("valid_time").max()).sort("newest"))


def main() -> None:
    """Run one build command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["uv", "tcc", "observed", "link"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if args.command in ("uv", "tcc"):
        build_pool(which=args.command, workers=min(args.workers, 6), limit=args.limit)
    elif args.command == "link":
        link_old_runs()
    else:
        build_observed()


if __name__ == "__main__":
    main()
