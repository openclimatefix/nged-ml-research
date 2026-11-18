# ruff: noqa: E501
"""Experiment 700: cut the stored ladder files of experiment 677 down to the origins of 2025-11 to 2026-06.

    uv run --no-sync python ml-experiments/00700-*/e00700_ladder_year.py

Reads `backtests/step{5,8}w_with_injected_sp{3,1}.parquet` of experiment 677 (read only) and
writes `backtests/step{5,8}_year_sp{3,1}.parquet` under this experiment's private folder, so the
scorer of experiment 677 can score the same month as the GAM and XGBoost files. The suffix `_sp3`
(quarterly refits) or `_sp1` (monthly refits, steps 9a and 9b) is kept because the scorer reads it.
"""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SRC = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
    / "backtests"
)
DST: Path = (
    ML_EXPERIMENTS_DATA_DIR / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit" / "backtests"
)

for step in ("5", "8"):
    for span in (3, 1):
        frame = pl.read_parquet(SRC / f"step{step}w_with_injected_sp{span}.parquet").filter(
            (pl.col("power_fcst_init_time") >= datetime(2025, 11, 1, tzinfo=UTC))
            & (pl.col("power_fcst_init_time") < datetime(2026, 7, 1, tzinfo=UTC))
            & (pl.col("seed") < 3)
        )
        DST.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(DST / f"step{step}_year_sp{span}.parquet")
        print(step, span, frame.height, frame["seed"].unique().sort().to_list())
