"""Experiment 466: paths, half-lives, and the row weight shared by the wind and solar fits.

The row weight is `0.5 ** (age / half_life)`, where the age of a training row is the time from the
row to the fit's cutoff, and it is rescaled to a mean of 1 over the rows the loss sees. Capacity is
experiment 461's p99.9 of earlier power, so every arm shares one base.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 466
SLUG: Final[str] = "wind-and-solar-recency-weighting"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
DAYS_PER_YEAR: Final[float] = 365.25
WIND_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
WIND_HALF_LIVES_YEARS: Final[dict[str, float | None]] = {
    "none": None,
    "hl9m": 0.75,
    "hl4_5m": 0.375,
}
SOLAR_HALF_LIVES_YEARS: Final[dict[str, float]] = {"hl05": 0.5, "hl1": 1.0, "hl2": 2.0, "hl3": 3.0}
"""Half-life in years of each solar weighting arm. `g0_onepass` runs all four; `a2_member` runs 2 and 3."""


def row_weights(*, age_days: np.ndarray, half_life_years: float | None) -> np.ndarray:
    """Return the weight of each row (all ones when `half_life_years` is None), unnormalised."""
    if half_life_years is None:
        return np.ones(len(age_days))
    return 0.5 ** (age_days / (DAYS_PER_YEAR * half_life_years))
