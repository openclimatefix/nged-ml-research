"""Experiment 461: paths and the p99.9 effective capacity shared by the fit and score scripts.

The new capacity of a fit is the 99.9th percentile of the series' cleaned power (experiment 453's
`series.parquet`, which comes from `studies.power.scan_power`) strictly before the end of that
fit's training. The old capacity is the stored whole-record 99th percentile in the same file.
Targets and lagged power are stored as power over the old capacity, so a fit under the new
capacity divides them by `ratio = new / old`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 461
SLUG: Final[str] = "wind-and-solar-retrain-with-p99-9-effective-capacity"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00453-effective-capacity-upper-bound-data-look"
    / "work"
    / "series.parquet"
)
CAPACITY_QUANTILE: Final[float] = 0.999
WIND_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
"""End of training of experiment 429's wind fits (`e00347_s4_data.MAIN_CUTOFF`)."""
WIND_LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}


def capacity(*, site: str, cutoff: datetime) -> dict[str, float | int]:
    """Return the old and new capacity (MW) of `site` ("Farm A", "Solar A", ...) at `cutoff`.

    The new capacity uses only power before `cutoff`. `n_history` is the number of half-hours
    behind it.
    """
    series = pl.read_parquet(SERIES_PATH).filter(pl.col("site") == site)
    earlier = series.filter(pl.col("time") < pl.lit(cutoff))["power"].to_numpy()
    old = float(series["cap"][0])
    new = float(np.quantile(earlier, CAPACITY_QUANTILE))
    return {"old_mw": old, "new_mw": new, "ratio": new / old, "n_history": len(earlier)}


def write_capacities(*, name: str, table: dict[str, dict]) -> None:
    """Write a capacity table (site and refit to values) as JSON under `DATA_DIR`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(table, indent=1) + "\n")
