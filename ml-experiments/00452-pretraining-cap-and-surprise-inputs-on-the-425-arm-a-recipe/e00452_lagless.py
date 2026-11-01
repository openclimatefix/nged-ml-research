# ruff: noqa: E501
"""Read experiment 449's cross-fitted lagless tables on ERA5 and CAMS weather (no build).

Experiment 449 built one table per scored month and seed (seeds 0 to 4, five folds) from lagless
modules fitted on ERA5 and CAMS weather, forecasting each contiguous block from the other blocks.
This experiment reuses those 25 tables for the surprise at a lag during pretraining, and builds
nothing.
"""

from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00452-pretraining-cap-and-surprise-inputs-on-the-425-arm-a-recipe"
    / "work"
)
"""This experiment's private folder: new networks, predictions, and scores."""
SOURCE_449: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00449-pretraining-with-lags-and-surprise-inputs-on-the-s1-network"
    / "work"
)
"""Experiment 449's private folder, read and never written."""
TABLE_DIR: Final[Path] = SOURCE_449 / "lagless_era5"


def table_path(*, month: str, seed: int) -> Path:
    """Return where a scored month's ERA5 table is stored."""
    return TABLE_DIR / f"month={month}" / f"seed={seed}.parquet"


def surprise_tables(*, month: str, seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the table as `(median, half band width)` tensors (primaries, grid), NaN where absent."""
    table = pl.read_parquet(table_path(month=month, seed=seed))
    median = np.full((len(data.sites()), data.grid_length()), np.nan, dtype=np.float32)
    half = np.full_like(median, np.nan)
    p, g = table["primary"].to_numpy(), table["grid"].to_numpy()
    median[p, g] = table["q50"].to_numpy()
    half[p, g] = np.maximum(
        (table["q95"] - table["q05"]).to_numpy() / 2.0 * table["half_scale"].to_numpy(), 1e-3
    )
    return torch.from_numpy(median).to(device), torch.from_numpy(half).to(device)
