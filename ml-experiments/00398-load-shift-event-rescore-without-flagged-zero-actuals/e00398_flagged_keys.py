# ruff: noqa: E501
"""List the flagged exact-zero half-hours and count them in each stored per-row score file.

DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00398-*/e00398_flagged_keys.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.power import CLEANED_POWER_DELTA_URI
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
)
FIRST: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
ROW_FILES: Final[dict[str, str]] = {
    "379_stack": "00379-bg2-components-all-16-primaries-to-14-days/work/scores/rows_stack.parquet",
    "379_components": "00379-bg2-components-all-16-primaries-to-14-days/work/scores/rows_components.parquet",
    "391": "00391-xgboost-lag-surprise-second-stage/work/scores/rows.parquet",
    "395_control": "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts/work/b_rows_m_control.parquet",
    "395_stack": "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts/work/b_rows_stack.parquet",
}


def main() -> None:
    """Write the flagged keys and print the unique flagged keys per Primary in each file."""
    OUT.mkdir(parents=True, exist_ok=True)
    flagged = (
        pl.scan_delta(CLEANED_POWER_DELTA_URI)
        .filter(
            pl.col("drop_reason") == "substation_zero",
            pl.col("time").is_between(FIRST, LAST),
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power", "drop_reason")
        .collect()
    )
    flagged.write_parquet(OUT / "flagged_keys.parquet")
    print(f"flagged substation_zero rows in {FIRST.date()} to {LAST.date()}: {flagged.height}")
    print(f"actual power values among them: {flagged['power'].unique().to_list()}")
    for name, rel in ROW_FILES.items():
        keys = (
            pl.scan_parquet(ML_EXPERIMENTS_DATA_DIR / rel)
            .select("time_series_id", "label", "valid_time")
            .unique()
            .join(
                flagged.lazy().select("time_series_id", "valid_time"),
                on=["time_series_id", "valid_time"],
            )
            .group_by("label")
            .agg(n=pl.len())
            .sort("label")
            .collect(engine="streaming")
        )
        print(name, keys.height and dict(keys.iter_rows()), "total", keys["n"].sum())


if __name__ == "__main__":
    main()
