"""Paths, periods, and the feature cache of experiment 386 (BSP and GSP demand under demand-v1).

Copied from experiment 161's `e00161_common.py` so this folder imports only itself. The cache adds
the two BSPs (D13, D17) and the two GSPs (D1, D6) to the 16 Primaries; experiment 103's panel
supplies the readings.
"""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.dp_features import (
    DpInputPaths,
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 386
SLUG: Final[str] = "bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
INPUT_DIR: Final[Path] = WORK_DIR / "inputs"
CACHE_DIR: Final[Path] = WORK_DIR / "cache"
PANEL_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00103-differentiable-physics-1-data-look" / "panel.parquet"
)
SERIES: Final[tuple[str, ...]] = ("D13", "D17", "D1", "D6")
LOCKDOWN_START: Final[date] = date(2020, 3, 23)
LOCKDOWN_END: Final[date] = date(2021, 7, 19)
SCORED_FIRST: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)


def series_rows() -> pl.DataFrame:
    """Return the cache rows of the BSPs and GSPs, from experiment 103's panel by its rows rule.

    A half-hour is a default row when the reading is present and unflagged, on or after the series'
    first usable instant, outside the fixed lockdown window, and outside the hand-filtered
    switching events.
    """
    paths = default_input_paths()
    first_usable = {
        r["site"]: r["first_usable"]
        for r in pl.read_parquet(paths.sites, columns=["site", "first_usable"]).iter_rows(
            named=True
        )
    }
    panel = pl.read_parquet(PANEL_PATH).filter(pl.col("site").is_in(list(SERIES)))
    in_lockdown = (pl.col("local_date") >= LOCKDOWN_START) & (pl.col("local_date") <= LOCKDOWN_END)
    return (
        panel.with_columns(
            usable=pl.col("time") >= pl.col("site").replace_strict(first_usable),
            lockdown=in_lockdown,
        )
        .with_columns(in_pool=pl.col("present") & ~pl.col("flagged") & pl.col("usable"))
        .with_columns(
            default_rows=pl.col("in_pool") & ~pl.col("lockdown") & ~pl.col("switching_mask")
        )
        .select(
            "time_series_id", "site", "time", "power", "present", "in_pool", "default_rows",
            "lockdown", "switching_mask", "dark",
        )
    )  # fmt: skip


def extended_paths() -> DpInputPaths:
    """Return the input paths with a rows file that adds the BSPs and GSPs to the 16 Primaries."""
    paths = default_input_paths()
    target = INPUT_DIR / "rows_with_bsps_gsps.parquet"
    if not target.exists():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = pl.concat([pl.read_parquet(paths.rows), series_rows()], how="diagonal_relaxed")
        temporary = target.with_name(f"{target.name}.tmp")
        rows.write_parquet(temporary)
        temporary.rename(target)
    return replace(paths, rows=target)


def load_cache() -> FeatureCache:
    """Return the feature cache of the 16 Primaries, the two BSPs, and the two GSPs."""
    return load_or_build_feature_cache(paths=extended_paths(), cache_dir=CACHE_DIR)
