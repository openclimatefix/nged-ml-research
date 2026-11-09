"""Shared constants of experiment 659 (simple shared network, wind and solar)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
E615_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00615-bakeoff-wind-and-solar-development-month-backtests"
)
FEATURE_DIR: Final[Path] = DATA_DIR / "wind_features"
"""One parquet per farm: the per-case mean and spread of raw weather and of the physics-lite power."""
FIT_DIR: Final[Path] = DATA_DIR / "wind_fits"
MODEL_DIR: Final[Path] = DATA_DIR / "models"
TABLE_DIR: Final[Path] = DATA_DIR / "tables"
MEMBERS: Final[int] = 51
CAPACITY_DAYS: Final[int] = 730
"""Capacity is the 99.9th percentile of power over this many days before a refit."""
CAPACITY_QUANTILE: Final[float] = 0.999
DEV_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
VALID_LIMIT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time from this instant is read."""
WIND_CUT_IN: Final[float] = 3.0
WIND_RATED: Final[float] = 12.0
WIND_CUT_OUT: Final[float] = 25.0
RAW_NAMES: Final[tuple[str, ...]] = (
    "speed10_mean",
    "speed10_sd",
    "speed100_mean",
    "speed100_sd",
    "dir_sin_mean",
    "dir_sin_sd",
    "dir_cos_mean",
    "dir_cos_sd",
    "temp_mean",
    "temp_sd",
)
PL_NAMES: Final[tuple[str, ...]] = ("pl_mean", "pl_sd")
