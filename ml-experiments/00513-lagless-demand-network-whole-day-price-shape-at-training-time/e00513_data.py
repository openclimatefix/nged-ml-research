# ruff: noqa: E501
"""The price, placebo, and national-control shape columns on experiment 428's dense half-hour grid.

Price is the Elexon Market Index (provider `APXMIDP`, deduplicated on `(time, dataProvider)`; rows
with zero volume are missing). The national control is the shape of national demand outturn
(`indo_mw`) minus wind (`FUELHH` fuel `WIND`) for the same local day.
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00513_shape as shape
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PRICE_FILE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00010-elexon-price-lags-stage1a-demand"
    / "prices"
    / "market_index.parquet"
)
CONTROL_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00010-elexon-price-lags-stage1a-demand" / "control"
)
SOURCES: Final[tuple[str, ...]] = ("price", "placebo", "national")


def grid_length() -> int:
    """Return the number of half-hours on the dense grid."""
    return len(common.grid_times())


def _to_grid(*, frame: pl.DataFrame, value: str) -> np.ndarray:
    """Return `value` of `frame` (columns `time`, `value`) on the grid, NaN where absent."""
    out = np.full(grid_length(), np.nan)
    idx = (
        (frame["time"].dt.epoch("us").to_numpy() - prep422.GRID_START_US) // shape.STEP_US
    ).astype(np.int64)
    ok = (idx >= 0) & (idx < len(out))
    out[idx[ok]] = frame[value].to_numpy()[ok]
    return out


def price_values() -> np.ndarray:
    """Return the Market Index price on the grid, NaN where absent or the row has zero volume."""
    frame = (
        pl.read_parquet(PRICE_FILE)
        .filter(pl.col("dataProvider") == "APXMIDP")
        .unique(["time", "dataProvider"], keep="first")
        .filter(pl.col("volume") > 0)
        .select("time", "price")
    )
    return _to_grid(frame=frame, value="price")


def national_values() -> np.ndarray:
    """Return national demand (INDO) minus wind on the grid, NaN where either is absent."""
    demand = (
        pl.read_parquet(CONTROL_DIR / "elexon_demand_outturn" / "elexon_demand_outturn.parquet")
        .unique("time")
        .select("time", demand=pl.col("indo_mw"))
    )
    wind = (
        pl.read_parquet(CONTROL_DIR / "elexon_fuelhh" / "elexon_fuelhh.parquet")
        .filter(pl.col("fuel_type") == "WIND")
        .unique("time")
        .select("time", wind=pl.col("generation_mw"))
    )
    both = demand.join(wind, on="time", how="inner").select(
        "time", residual=pl.col("demand") - pl.col("wind")
    )
    return _to_grid(frame=both, value="residual")


class ShapeTables:
    """The three shape-column tables of one fold, indexed by grid half-hour."""

    def __init__(self, *, fit_until_idx: int) -> None:
        """Build the raw columns once and standardise the level features on rows before `fit_until_idx`.

        Args:
            fit_until_idx: Grid index before which a half-hour may inform the standardiser (the
                fold's refit time), so a test month never shapes its own scale.
        """
        n = grid_length()
        fit_rows = np.arange(n) < fit_until_idx
        raw_price, self.day_id, self.price_missing = shape.grid_features(
            grid_start_us=prep422.GRID_START_US, grid_length=n, values=price_values()
        )
        raw_national, _, self.national_missing = shape.grid_features(
            grid_start_us=prep422.GRID_START_US, grid_length=n, values=national_values()
        )
        n_days = len(self.price_missing)
        first = np.searchsorted(self.day_id, np.arange(n_days), side="left")
        times = prep422.GRID_START_US + first.astype(np.int64) * shape.STEP_US
        self.day_local = shape.local_day_of(times_us=times)
        self.donor, self.unmatched = shape.placebo_draw(
            day_id=self.day_id,
            missing=self.price_missing,
            first_local_day=self.day_local,
            after_refit=first >= fit_until_idx,
        )
        price = shape.standardise_levels(columns=raw_price, fit_rows=fit_rows)
        self.tables: dict[str, np.ndarray] = {
            "price": price,
            "placebo": shape.placebo_columns(columns=price, day_id=self.day_id, donor=self.donor),
            "national": shape.standardise_levels(columns=raw_national, fit_rows=fit_rows),
        }

    def columns(self, *, source: str, target_idx: np.ndarray) -> np.ndarray:
        """Return the shape columns of the target half-hours of some rows (float32, 11 wide)."""
        return self.tables[source][target_idx]
