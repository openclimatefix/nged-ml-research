# ruff: noqa: E501
"""Experiment 647: shared constants and loaders for the departure head on frozen K1.

Time is the half-hour grid index of experiment 578 (`GRID_START`, 48 per UTC day); power is a share of
capacity unless a name says MW. A forecast origin is at 09:00 UTC. Nothing here reads a day on or
after 2026-07-01.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00578_common as k578  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00647-two-output-forecast-departure-head-on-frozen-k1"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
QNAMES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
TAUS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
WINDOW_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
DEV_FIRST_DAY: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
TAU_HOURS: Final[tuple[int, int]] = (72, 168)
"""The two decay constants of R0 (the design answer's pair)."""
R0_FIRST_DAY: Final[int] = 2
R0_LAST_DAY: Final[int] = 8
"""R0 averages the residual over the valid days `origin_day - 8` to `origin_day - 2`."""
R0_LEAD_H: Final[tuple[float, float]] = (24.0, 48.0)
"""R0's residual uses each valid time's forecast with a lead of 24 to below 48 h (the day-ahead forecast)."""
DESIGN_TABLE: Final[dict[str, float]] = {
    "ef_K1": 3.752, "ef_72": 3.637, "ef_168": 3.577,
    "ver_K1": 14.93, "ver_72": 13.42, "ver_168": 12.79,
}  # fmt: skip
"""The design answer's section 0 table (CRPS in points of capacity, leads 24 to 240 h)."""
DESIGN_TOLERANCE: Final[float] = 0.01


def crps_points(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return per-row CRPS in points of capacity: 2 x mean pinball loss over the 11 levels, x 100."""
    e = y[:, None] - q
    return 2.0 * np.maximum(TAUS * e, (TAUS - 1.0) * e).mean(axis=1) * 100.0


def load_export(*, network: str = "K1") -> pl.DataFrame:
    """Return the stored roster export of `network` for the five months, power as a share of capacity.

    Adds `capacity`, `month`, and the grid indices `vi` (valid time), `oi` (origin), and `od` (origin
    day). Rows are the export's own (leads 0 to 239.5 h).
    """
    sub = pl.read_csv(EXPORT / "substations.csv").select("time_series_id", "capacity")
    parts = []
    for m in MONTHS:
        df = pl.read_parquet(EXPORT / f"{network}_{m}.parquet")
        assert df["valid_time"].max() < WINDOW_START  # ty: ignore[unsupported-operator]
        parts.append(df.with_columns(month=pl.lit(m)))
    df = pl.concat(parts).join(sub, on="time_series_id", how="left", validate="m:1")
    assert df["capacity"].null_count() == 0
    t0, step = k578.prep.GRID_START_US, k578.prep.STEP_US
    return df.with_columns(
        *(pl.col(c) / pl.col("capacity") for c in (*QNAMES, "observed_power", "nra_power")),
        vi=((pl.col("valid_time").dt.epoch("us") - t0) // step).cast(pl.Int64),
        oi=((pl.col("power_fcst_init_time").dt.epoch("us") - t0) // step).cast(pl.Int64),
        od=pl.col("power_fcst_init_time").dt.truncate("1d"),
    )


def date_of(*, idx: int, grid_start: datetime) -> str:
    """Return the UTC date of a half-hour grid index."""
    return (grid_start + timedelta(minutes=30 * int(idx))).strftime("%Y-%m-%d")
