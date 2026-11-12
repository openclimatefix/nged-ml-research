# ruff: noqa: E501
"""Paths, constants, and loaders shared by the scripts of experiment 695.

D8 is a substation and may be named. The engine record is called "Gas A" everywhere that is printed
or drawn, never with dates.
"""

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 695
SLUG: Final[str] = "gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{EXPERIMENT_ID:05d}-{SLUG}"
)
E677_ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
)
STEP9A_PATH: Final[Path] = E677_ROOT / "backtests" / "step5w_with_injected_sp1.parquet"
"""Experiment 677's step 9a (monthly refits of step 5 with injected shifts), 3 seeds, no manifest."""
E677_PROTOCOL_ROWS: Final[Path] = E677_ROOT / "scores" / "protocol_all" / "rows.parquet"
SHARED_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
E694_STATE_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00694-gas-engine-state-tracker-part-1" / "d8.pkl"
)
"""Experiment 694's hindsight M2 and M1 states of D8 for three eras, as a pickle."""
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
"""The first half-hour of the stored power grid."""
D8_SERIES: Final[int] = 26
"""D8's `time_series_id` in the contract's files."""
D8_CAPACITY_MVA: Final[float] = 10.379
"""D8's effective capacity, fixed for every arm."""
ORIGIN_FIRST: Final[datetime] = datetime(2025, 11, 1, tzinfo=UTC)
"""The first scored forecast origin."""
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Valid times and origins from here on are never scored."""
LEAD_MIN_HOURS: Final[float] = 24.0
LEAD_MAX_HOURS: Final[float] = 240.0
"""Leads counted from the 12:00 UTC contract origin, both ends included."""
ENGINE_MONTHS: Final[tuple[str, ...]] = ("2025-11", "2025-12", "2026-01", "2026-02")
FADE_MONTHS: Final[tuple[str, ...]] = ("2026-03", "2026-04", "2026-05", "2026-06")
SCORED_MONTHS: Final[tuple[str, ...]] = (*ENGINE_MONTHS, *FADE_MONTHS)
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
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
"""The 11 forecast quantile columns of the contract, in level order."""


def month_label(*, valid_time: pl.Expr) -> pl.Expr:
    """Return the `YYYY-MM` label of a UTC timestamp expression."""
    return valid_time.dt.strftime("%Y-%m")


def d8_truth() -> pl.DataFrame:
    """Return D8's metered apparent power (MVA) with `valid_time`, `power`; null where dropped."""
    return (
        pl.scan_parquet(SHARED_DIR / "ground_truth_primaries.parquet")
        .filter(pl.col("time_series_id") == D8_SERIES)
        .select("valid_time", "power")
        .collect()
    )


def scored_window(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of the decision window: leads 24 to 240 h, origins and valid times before July."""
    lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    return frame.filter(
        pl.col("power_fcst_init_time") >= ORIGIN_FIRST,
        pl.col("power_fcst_init_time") < HOLDOUT,
        pl.col("valid_time") < HOLDOUT,
        lead >= LEAD_MIN_HOURS,
        lead <= LEAD_MAX_HOURS,
    )


def e694_flags() -> pl.DataFrame:
    """Return experiment 694's hindsight and filtered M2 flags of D8 by half-hour.

    The columns are `valid_time`, `reading` (MVA, null where dropped), `flag` (smoothed, any unit
    running) and `filtered` (filter only). The three eras of the pickle do not overlap.
    """
    eras = pickle.loads(E694_STATE_PATH.read_bytes())
    frames = []
    for era in eras:
        n = len(era["y"])
        start = np.datetime64(GRID_START.replace(tzinfo=None), "s") + int(
            era["start"]
        ) * np.timedelta64(1800, "s")
        stamps = start + np.arange(n) * np.timedelta64(1800, "s")
        frames.append(
            pl.DataFrame(
                {
                    "valid_time": pl.Series(stamps.astype("datetime64[us]")).dt.replace_time_zone(
                        "UTC"
                    ),
                    "reading": np.where(np.isfinite(era["y"]), era["y"], np.nan),
                    "flag": np.asarray(era["M2"]["flag"], dtype=bool),
                    "filtered": np.asarray(era["M2"]["filt"], dtype=bool),
                    "era": era["label"],
                }
            ).with_columns(reading=pl.col("reading").fill_nan(None))
        )
    return pl.concat(frames)
