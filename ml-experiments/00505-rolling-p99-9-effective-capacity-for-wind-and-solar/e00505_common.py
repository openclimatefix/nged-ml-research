"""Experiment 505: paths and the rolling p99.9 effective capacity for the fit and score scripts.

The capacity of a fit is the 99.9th percentile of the series' cleaned power (experiment 453's
`series.parquet`, which comes from `studies.power.scan_power`) in the window [cutoff - window,
cutoff), where `cutoff` is the end of that fit's training. Experiment 461's rule (arm P-all) is
the same percentile over all power before `cutoff`. If the window holds fewer than `MIN_VALID`
valid half-hours, the capacity falls back to P-all and the row says so. A half-hour is valid if it
passes cleaning (it is present in `series.parquet`); for solar only daylight half-hours (power
above zero) count towards `MIN_VALID`, and for wind every cleaned half-hour counts. The
percentile itself is taken over every cleaned half-hour of the window, night zeros included, as in
experiment 461.
Targets and lagged power are stored as power over the old capacity (the whole-record 99th
percentile in `series.parquet`), so a fit under the new capacity divides them by
`ratio = new / old`.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 505
SLUG: Final[str] = "rolling-p99-9-effective-capacity-for-wind-and-solar"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
EXP461_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00461-wind-and-solar-retrain-with-p99-9-effective-capacity"
)
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00453-effective-capacity-upper-bound-data-look"
    / "work"
    / "series.parquet"
)
CAPACITY_QUANTILE: Final[float] = 0.999
MIN_VALID: Final[int] = 8000
"""Fewer valid half-hours than this in a window sends the capacity back to P-all."""
WINDOWS: Final[dict[str, int]] = {"p730": 730, "p365": 365}
"""Window name to length in days. The names are the arm tags used in file names."""
WIND_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
"""End of training of experiment 429's wind fits (`e00347_s4_data.MAIN_CUTOFF`)."""
WIND_LABEL: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}


def capacity(
    *, site: str, cutoff: datetime, window_days: int | None
) -> dict[str, float | int | bool]:
    """Return the old and new capacity (MW) of `site` ("Farm A", "Solar A", ...) at `cutoff`.

    `window_days` is None for P-all (all power before `cutoff`) and the window length in days for
    a rolling arm. The window is [cutoff - window_days, cutoff). `n_history` is the number of
    half-hours before `cutoff`; `n_window` is the number in the window; `n_valid` is the number
    of valid half-hours in the window (daylight for solar, all for wind); `n_above` is the number
    of half-hours strictly above the new capacity in the half-hours the percentile used;
    `fallback` is True when the window held fewer than `MIN_VALID` valid half-hours and P-all was
    used instead.
    """
    series = pl.read_parquet(SERIES_PATH).filter(pl.col("site") == site)
    earlier = series.filter(pl.col("time") < pl.lit(cutoff))
    old = float(series["cap"][0])
    if window_days is None:
        window = earlier
    else:
        window = earlier.filter(pl.col("time") >= pl.lit(cutoff - timedelta(days=window_days)))
    power = window["power"].to_numpy()
    valid = int((power > 0).sum()) if site.startswith("Solar") else len(power)
    fallback = window_days is not None and valid < MIN_VALID
    used = earlier["power"].to_numpy() if fallback else power
    new = float(np.quantile(used, CAPACITY_QUANTILE))
    return {
        "old_mw": old,
        "new_mw": new,
        "ratio": new / old,
        "n_history": len(earlier),
        "n_window": len(power),
        "n_valid": valid,
        "n_above": int((used > new).sum()),
        "fallback": bool(fallback),
    }


def write_capacities(*, name: str, table: dict[str, dict]) -> None:
    """Write a capacity table (site and refit to values) as JSON under `DATA_DIR`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_text(json.dumps(table, indent=1) + "\n")
