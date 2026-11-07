"""Shared constants of experiment 615 (wind and solar development-month backtests)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00615-bakeoff-wind-and-solar-development-month-backtests"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBERS_DIR: Final[Path] = DATA_DIR / "wind_members"
"""Daily ensemble member rows at the wind farms, one parquet file per run."""
DEV_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The five development months, named by forecast origin."""
FIRST_NEW_RUN: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""The day after experiment 259's last hindcast run."""
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
"""The last forecast origin scored. No origin from 2026-07-01 is used."""
VALID_LIMIT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time from this instant is scored, so July 2026 stays unused."""
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
TABLE_DIR: Final[Path] = DATA_DIR / "tables"
"""One parquet per candidate, in the common layout of `e00615_assemble.py`."""
E1: Final[Path] = ML_EXPERIMENTS_DATA_DIR
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00453-effective-capacity-upper-bound-data-look"
    / "work"
    / "series.parquet"
)
WIND_SITE: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
SOLAR_SERIES: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
