"""Experiment 701: cut the stored ladder files of experiment 677 to the origins 2026-07 to 2026-10.

    uv run --no-sync python ml-experiments/00701-*/e00701_ladder.py

Reads `backtests/step{5,8}w_with_injected_sp{3,1}.parquet` of experiment 677 (read only) and
writes `backtests/step{5,8}_final_sp{3,1}.parquet` under this experiment's private folder, for the
origins 2026-07-01 to 2026-10-31 (seeds 0 to 2), so that `e00701_protocol.py` scores them beside the
GAM and XGBoost files. The suffix `_sp3` (quarterly refits) or `_sp1` (monthly refits, steps 9a and
9b) is kept as in experiment 700.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SRC: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00677-building-up-the-candidate-demand-network-one-component-at-a-time"
    / "backtests"
)
DST: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00701-gam-and-xgboost-for-the-final-test-months-2026-07-to-2026-10"
    / "backtests"
)
FIRST_ORIGIN: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
END_ORIGIN: Final[datetime] = datetime(2026, 11, 1, tzinfo=UTC)


def main() -> None:
    """Write the four cut files."""
    DST.mkdir(parents=True, exist_ok=True)
    for step in ("5", "8"):
        for span in (3, 1):
            frame = pl.read_parquet(SRC / f"step{step}w_with_injected_sp{span}.parquet").filter(
                (pl.col("power_fcst_init_time") >= FIRST_ORIGIN)
                & (pl.col("power_fcst_init_time") < END_ORIGIN)
                & (pl.col("seed") < 3)
            )
            frame.write_parquet(DST / f"step{step}_final_sp{span}.parquet")
            print(
                step,
                span,
                frame.height,
                frame["seed"].unique().sort().to_list(),
                frame["power_fcst_init_time"].min(),
                frame["power_fcst_init_time"].max(),
                frame["power_fcst_init_time"].n_unique(),
            )


if __name__ == "__main__":
    main()
