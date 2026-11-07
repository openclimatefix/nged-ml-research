"""Experiment 630: join the roster export with ensemble temperature and write one frame per network.

Reads `backtest_export_roster_2026-11-07/` (read-only) and the production NWP Delta table, and
writes `work/<network>.parquet` in the experiment's data folder. Only the five development months
are read.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction"
    / "work"
)
NETWORKS: Final[tuple[str, ...]] = ("K1", "Candidate", "K3")
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
Q: Final[tuple[str, ...]] = (
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
TAU: Final[tuple[float, ...]] = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)


def temperature_by_series(*, valid_min: datetime, valid_max: datetime) -> pl.DataFrame:
    """Ensemble-mean 2 m temperature (deg C) at each Primary's H3 cell, from the latest run.

    The run is the latest one that started at or before each valid time.
    """
    meta = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet").select(
        "time_series_id", pl.col("h3_res_5").alias("h3_index")
    )
    cells = meta["h3_index"].cast(pl.Int64).to_list()
    nwp = (
        pl.scan_delta(str(ML_EXPERIMENTS_DATA_DIR.parent / "NWP"))
        .filter(
            pl.col("h3_index").is_in(cells),
            pl.col("valid_time") >= valid_min,
            pl.col("valid_time") < valid_max,
        )
        .group_by("h3_index", "init_time", "valid_time")
        .agg(pl.col("temperature_2m").mean().alias("t"))
        .filter(pl.col("init_time") <= pl.col("valid_time"))
        .collect()
    )
    latest = (
        nwp.sort("init_time")
        .group_by("h3_index", "valid_time")
        .agg(pl.col("t").last().alias("temperature_c"))
    )
    return (
        meta.with_columns(pl.col("h3_index").cast(pl.Int64))
        .join(latest, on="h3_index")
        .select("time_series_id", "valid_time", "temperature_c")
    )


def main() -> None:
    """Join each network's export with temperature and write one frame per network."""
    WORK.mkdir(parents=True, exist_ok=True)
    temp = temperature_by_series(
        valid_min=datetime(2025, 9, 25, tzinfo=UTC), valid_max=datetime(2026, 7, 11, tzinfo=UTC)
    ).sort("time_series_id", "valid_time")
    sub = pl.read_csv(EXPORT / "substations.csv")
    for network in NETWORKS:
        frames = [
            pl.read_parquet(EXPORT / f"{network}_{m}.parquet").with_columns(
                pl.lit(m).alias("month")
            )
            for m in MONTHS
        ]
        df = pl.concat(frames).join(
            sub.select("time_series_id", "primary_name", "group", "capacity"), on="time_series_id"
        )
        df = df.sort("time_series_id", "valid_time").join_asof(
            temp, on="valid_time", by="time_series_id", strategy="backward", tolerance="6h"
        )
        df = df.filter(pl.col("observed_power").is_not_null() & (pl.col("lead_hours") < 240))
        df.write_parquet(WORK / f"{network}.parquet", compression="zstd")
        print(
            network,
            df.height,
            df["temperature_c"].null_count(),
            df["valid_time"].min(),
            df["valid_time"].max(),
        )


if __name__ == "__main__":
    main()
