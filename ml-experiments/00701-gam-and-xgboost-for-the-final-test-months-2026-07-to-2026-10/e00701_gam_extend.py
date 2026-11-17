# ruff: noqa: E501
"""Experiment 701, GAM side: extend the dates of experiments 377 and 103's cache code to 2026-11-14.

Import this module first in every GAM script, before any other `e00377_*` module that reads a date,
as `e00700_horizon.py` does. It changes module constants in memory and edits no file of `src/` or
of an earlier experiment:

- **`studies.dp_features`:** `GRID_END` moves from 2026-06-30 23:30 to 2026-11-14 23:30, and
  `HOLDOUT_START` (the cut applied to the rows table) moves to 2026-11-15, so the cache holds actual
  power up to the last valid time that a 14-day forecast from a 2026-10-31 origin reaches. The
  default input paths point at the extended tables of `e00701_gam_inputs.py` (the folder
  `<ROOT>/gam/inputs`), which equal the stored tables on every shared row.
- **`studies.dp_gam.HOLDOUT_START`:** moved to 2026-11-15, the guard that refuses to score rows at
  or after the held-out period.

`e00701_gam_extend377.py` then changes experiment 377's own dates.

Paths: set `DATA_PATH_INTERNAL` to the real data folder for the input and cache builds, and to
`<ROOT>/gam/sandbox` for every GAM stage. `ROOT` below is the same real folder in both cases.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies import dp_features, dp_gam

NAME: Final[str] = "00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10"
_data: Path = Path(os.path.abspath(os.environ["DATA_PATH_INTERNAL"]))  # noqa: PTH100  (resolve() would follow the data symlink)
REAL_DATA: Final[Path] = _data.parents[3] if _data.name == "sandbox" else _data
"""The real data folder, whether or not `DATA_PATH_INTERNAL` points at the sandbox."""
ROOT: Final[Path] = REAL_DATA / "ml-experiments" / NAME
"""This experiment's private data folder."""
GAM: Final[Path] = ROOT / "gam"
"""Everything the GAM side writes: `inputs`, `cache`, `sandbox`, `timing`."""
INPUTS: Final[Path] = GAM / "inputs"
CACHE_DIR: Final[Path] = GAM / "cache"
SANDBOX: Final[Path] = GAM / "sandbox"
SANDBOX_ML: Final[Path] = SANDBOX / "ml-experiments"
TIMING: Final[Path] = GAM / "timing"
E607_WORK: Final[Path] = (
    REAL_DATA / "ml-experiments" / "00607-bakeoff-extend-cached-inputs-to-2026-10-31" / "work"
)
ERA5_EXTENDED: Final[Path] = E607_WORK / "era5_features.parquet"
CAMS_COMBINED: Final[Path] = INPUTS / "cams_demand_points_combined.parquet"

GRID_END: Final[datetime] = datetime(2026, 11, 14, 23, 30, tzinfo=UTC)
"""The last half-hour end on the extended grid: the last valid time of the 2026-10-31 run is 2026-11-14 12:00."""
ROWS_END_EXCLUSIVE: Final[datetime] = datetime(2026, 11, 15, tzinfo=UTC)
"""The extended rows table holds every half-hour that ends before this instant."""
FIRST_ORIGIN_MONTH: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 10, 31, tzinfo=UTC)
MONTHS: Final[list[datetime]] = [datetime(2026, m, 1, tzinfo=UTC) for m in (7, 8, 9, 10)]
"""The origin months, which are also the cuts of the monthly T-all refits."""
END: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)
"""Exclusive end of the last origin month."""
SMOKE_MONTH: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
"""A stored month (experiment 700's last), used to prove the extended inputs reproduce it."""

# Weather and tables of experiments 103, 607, and the production CAMS download.
dp_features.GRID_END = GRID_END  # ty: ignore[invalid-assignment]
dp_features.HOLDOUT_START = ROWS_END_EXCLUSIVE  # ty: ignore[invalid-assignment]
dp_gam.HOLDOUT_START = ROWS_END_EXCLUSIVE  # ty: ignore[invalid-assignment]
dp_features.PART_1_DIR = INPUTS  # ty: ignore[invalid-assignment]
dp_features.ERA5_PATH = ERA5_EXTENDED  # ty: ignore[invalid-assignment]
dp_features.CAMS_PATH = CAMS_COMBINED  # ty: ignore[invalid-assignment]
