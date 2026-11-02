"""Data layer for experiment 506: cleaned power, CAMS sky, and injected solar farms on one grid.

Everything is on experiment 87's dense half-hour grid (period-end stamps, UTC). Power is in points
arrays experiment 412 stored. No model is fitted; `torch` is not imported.
arrays experiment 412 stored. No model is fitted and `torch` is not imported.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import default_input_paths
from studies.experiment_store import HOLDOUT_START
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-sign-evidence-dip-depth-and-gated-solar-rule"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00506-{SLUG}" / "work"
CAMS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00412-mdn-signed-mw-from-mva-meters-known-answer" / "work" / "cams"
)
FARM_FRAME: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
FARM_OF_SITE: Final[dict[str, str]] = {
    "D5": "A",
    "D9": "B",
    "D10": "C",
    "D11": "A",
    "D14": "B",
    "D16": "C",
}
"""Experiment 408's rotation of three farms over the six signed-MW Primaries."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI: Final[float] = 100.0
SUNNY_K: Final[float] = 0.7
LEVEL_WINDOW_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)


def u99() -> float:
    """Return experiment 412's stored scaling constant for `u`."""
    return float(json.loads((CAMS_DIR.parent / "u99.json").read_text())["u99"])


def grid() -> pl.Series:
    """Return the dense grid of period-end stamps."""
    return common.grid_times()


def series_table() -> pl.DataFrame:
    """Return the site table (time_series_id, units, effective capacity in MW)."""
    return pl.read_parquet(default_input_paths().series_table)


def cams(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the scaled unit PV shape `u` and the clear-sky index `k` on the grid."""
    z = np.load(CAMS_DIR / f"{site}.npz")
    return (z["u"] / u99()).astype(np.float64), z["k"].astype(np.float64)


def power_points(*, site: str) -> np.ndarray:
    """Return cleaned power in points of capacity on the grid (NaN where absent or cut).

    `scan_power` drops the flagged exact-zero half-hours. D15 and D18 lose rows before 2025-03-01,
    and nothing on or after the July 2026 holdout is read.
    """
    row = series_table().filter(pl.col("site") == site).row(0, named=True)
    frame = (
        scan_power()
        .filter(
            pl.col("time_series_id") == row["time_series_id"],
            pl.col("time") < HOLDOUT_START,
        )
        .select("time", "power")
        .collect()
    )
    if site in SHORT_HISTORY:
        frame = frame.filter(pl.col("time") >= SHORT_HISTORY_CUT)
    joined = pl.DataFrame({"time": grid()}).join(
        frame.with_columns(pl.col("time").dt.cast_time_unit("us")), on="time", how="left"
    )
    return joined["power"].to_numpy().astype(np.float64) / float(row["effective_capacity_mw"]) * 100


def farm_norm(*, farm: str) -> np.ndarray:
    """Return a farm's power over its 99th percentile on the grid (NaN in a dropout or a gap).

    Copied from experiment 405's `farm_norm_series`: night is zero, a reading at or below zero with
    irradiance above 100 W/m2 is a dropout, and other negatives are set to zero.
    """
    everything = pl.read_parquet(FARM_FRAME).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    daylight = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG).filter(
        pl.col("power").is_not_null()
    )
    p99 = float(np.nanpercentile(daylight["power"].to_numpy(), 99))
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI)
    frame = everything.select(
        "time",
        norm=pl.when(pl.col("zenith") >= FARM_MAX_ZENITH_DEG)
        .then(0.0)
        .when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    ).unique(subset="time", keep="first")
    joined = pl.DataFrame({"time": grid()}).join(
        frame.with_columns(pl.col("time").dt.cast_time_unit("us")), on="time", how="left"
    )
    return joined["norm"].to_numpy().astype(np.float64)


def local_hour_start(*, stamps: pl.Series) -> np.ndarray:
    """Return the UK clock hour (fractional) at the start of each period."""
    start = stamps.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    return (start.dt.hour() + start.dt.minute() / 60).to_numpy().astype(np.float64)
