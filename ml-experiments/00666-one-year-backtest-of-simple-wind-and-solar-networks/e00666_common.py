"""Shared constants of experiment 666 (one-year backtest of simple wind and solar networks)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00666-one-year-backtest-of-simple-wind-and-solar-networks"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBERS_DIR: Final[Path] = DATA_DIR / "wind_members"
"""Daily ensemble member rows at the wind farms for the runs rebuilt to the full 15 days."""
FEATURE_DIR: Final[Path] = DATA_DIR / "wind_features"
"""One parquet per farm: per-case mean and spread of raw weather and physics-lite power."""
SOLAR_DIR: Final[Path] = DATA_DIR / "solar_inputs"
"""One parquet per solar farm: the network inputs and the observed output."""
E259_HINDCAST_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work" / "hindcast"
)
"""Experiment 259's member rows, runs 2024-04-01 to 2025-06-16 (read only)."""
E615_MEMBERS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00615-bakeoff-wind-and-solar-development-month-backtests"
    / "wind_members"
)
"""Experiment 615's member rows, runs 2025-06-17 to 2026-06-30, valid times before 2026-07-01."""
E659_FEATURE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar"
    / "wind_features"
)
"""Experiment 659's per-case summaries, the gate for the summaries rebuilt here (read only)."""

FIRST_REBUILT_RUN: Final[datetime] = datetime(2026, 6, 16, tzinfo=UTC)
"""The first run whose 15 days reach experiment 615's valid-time limit of 2026-07-01."""
LAST_RUN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
FIRST_ORIGIN: Final[datetime] = datetime(2025, 11, 1, tzinfo=UTC)
VALID_LIMIT: Final[datetime] = datetime(2026, 11, 20, tzinfo=UTC)
"""No valid time from this instant is built: after the last run's 15 days."""
OLD_VALID_LIMIT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Experiment 659 read no valid time from this instant."""
MEMBERS: Final[int] = 51
HORIZON_HALF_HOURS: Final[int] = 672
"""Half-hours from 12:00 on the origin day to 14 days later."""
WIND_CUT_IN: Final[float] = 3.0
WIND_RATED: Final[float] = 12.0
WIND_CUT_OUT: Final[float] = 25.0

ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The first 00 UTC run of the ensemble table."""
SOLAR_INIT_END: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)
"""Exclusive end of the solar rows' 00 UTC runs (the last run is 2026-10-31)."""
SOLAR_GRID_END: Final[datetime] = datetime(2026, 11, 17, tzinfo=UTC)
"""The sun-position grid runs to here, beyond the last valid time of the last run."""
SOLAR_DAYS: Final[tuple[int, ...]] = tuple(range(1, 15))
"""Day `d` holds half-hours with a lead of `12 + 24(d-1)` to `36 + 24(d-1)` h from the run."""
THREE_HOURLY_UNTIL_HOURS: Final[int] = 144
"""The ensemble table's stamps are 3-hourly up to this lead and 6-hourly after it."""
FLEET_TILT_DEG: Final[float] = 35.0
GAMMA: Final[float] = -0.004
U0: Final[float] = 25.0
U1: Final[float] = 6.84
LAST_LEAD_HOURS: Final[float] = 348.0
"""The last half-hour's period ends at this lead: the archive must hold stamps to here."""

MODEL_DIR: Final[Path] = DATA_DIR / "models"
"""Network state files, one `.pt` per (network, refit, seed)."""
FIT_DIR: Final[Path] = DATA_DIR / "fits"
"""Per-refit, per-seed quantile tables in MW (not the contract's files)."""
OUTPUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
"""The shared folder of the four one-year backtests."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CAPACITY_DAYS: Final[int] = 730
"""Capacity is the 99.9th percentile of power over this many days before a refit."""
CAPACITY_QUANTILE: Final[float] = 0.999
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
N_ORIGINS: Final[int] = 365
"""Daily origins from 2025-11-01 to 2026-10-31."""
RUN_TO_ORIGIN_HOURS: Final[int] = 12
"""The origin is 12:00 UTC, the 00 UTC run's day plus this many hours."""
RUN_TO_PUBLICATION_HOURS: Final[int] = 9
"""The 00 UTC run is treated as available at 09:00 UTC (wind cases carry that stamp)."""


def capacity_before(*, time_series_id: int, refit: datetime, old_mw: float) -> dict[str, float]:
    """Return the old and new capacity (MW) and their ratio for one series at `refit`.

    The new capacity is the 99.9th percentile of the series' cleaned power over the 730 days
    before `refit`; `old_mw` is the series' stored effective capacity, the unit of `actual`.
    """
    start = refit - timedelta(days=CAPACITY_DAYS)
    values = (
        scan_power()
        .filter(
            (pl.col("time_series_id") == time_series_id)
            & (pl.col("time") >= start)
            & (pl.col("time") < refit)
        )
        .select("power")
        .collect()["power"]
        .drop_nulls()
        .to_numpy()
    )
    new = float(np.quantile(values, CAPACITY_QUANTILE))
    return {"old_mw": old_mw, "new_mw": new, "ratio": new / old_mw}
