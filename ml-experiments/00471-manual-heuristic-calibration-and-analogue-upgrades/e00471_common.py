"""Shared constants and the analogue-member table of experiment 471 (no fit, no GPU).

The stored members of `manual_heuristic` (one row per series, origin, valid time, and member, in MW)
are reshaped to one row per `(time_series_id, power_fcst_init_time, valid_time)` holding the sorted
members, padded with NaN to 13 columns, for the 16 Primaries and leads of 0 to 240 hours. The
actuals are experiment 425's cleaned scoring actuals (flagged exact zeros removed).
"""

from pathlib import Path
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00471-manual-heuristic-calibration-and-analogue-upgrades" / "work"
)
MEMBER_GLOB: Final[str] = str(
    ML_EXPERIMENTS_DATA_DIR.parent
    / "power_forecasts/experiment_name=manual_heuristic/fold_id=mid_2025_to_mid_2026/*.parquet"
)
MEMBER_TABLE: Final[Path] = WORK_DIR / "members.parquet"
N_MAX: Final[int] = 13
MAX_LEAD_HOURS: Final[float] = 240.0
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
LEVELS: Final[dict[str, float]] = {
    "q01": 0.01, "q05": 0.05, "q10": 0.10, "q25": 0.25, "q40": 0.40, "q50": 0.50,
    "q60": 0.60, "q75": 0.75, "q90": 0.90, "q95": 0.95, "q99": 0.99,
}  # fmt: skip
LEVEL_VALUES: Final[np.ndarray] = np.array(list(LEVELS.values()))
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
BAND_EDGES_HOURS: Final[tuple[float, ...]] = (0.0, 24.0, 72.0, 120.0, 168.0, 240.0)
DELAY_HOURS: Final[int] = 24
"""A pair enters a fit at origin `o` only if its valid time is at most `o` minus this delay."""
MEMBER_COLUMNS: Final[list[str]] = [f"m{i:02d}" for i in range(N_MAX)]


def band_of(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead-band index (0 to 4) of each lead in hours."""
    return np.clip(np.searchsorted(BAND_EDGES_HOURS, lead_hours, side="left") - 1, 0, 4)


def build_member_table() -> pl.DataFrame:
    """Write and return the sorted-member table for the 16 Primaries, leads 0 to 240 hours."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ids = series["time_series_id"].to_list()
    frames = []
    for sid in ids:
        raw = (
            pl.scan_parquet(MEMBER_GLOB)
            .filter(pl.col("time_series_id") == sid)
            .select("time_series_id", "power_fcst_init_time", "valid_time", "power_fcst")
            .filter(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds()
                <= MAX_LEAD_HOURS * 3600
            )
            .collect()
        )
        grouped = raw.group_by(KEYS).agg(
            pl.col("power_fcst").cast(pl.Float64).drop_nans().sort().alias("v")
        )
        mat = np.full((grouped.height, N_MAX), np.nan)
        for i, v in enumerate(grouped["v"].to_list()):
            mat[i, : len(v)] = v
        frames.append(
            grouped.select(KEYS).with_columns(
                *[pl.Series(c, mat[:, j]) for j, c in enumerate(MEMBER_COLUMNS)]
            )
        )
        print(f"series {sid}: {grouped.height:,} keys")
    table = pl.concat(frames).with_columns(
        n=pl.sum_horizontal(
            *[pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in MEMBER_COLUMNS]
        ),
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_seconds()
        / 3600.0,
    )
    table.write_parquet(MEMBER_TABLE)
    return table


def load_members() -> pl.DataFrame:
    """Return the member table, building it first if it does not exist."""
    if MEMBER_TABLE.exists():
        return pl.read_parquet(MEMBER_TABLE)
    return build_member_table()


def cleaned_actuals(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 425's cleaned actuals in MW (flagged exact zeros removed)."""
    ctx = prep425.cleaned_scoring_context(series=series)
    return ctx["actuals"].select("time_series_id", "valid_time", "actual")
