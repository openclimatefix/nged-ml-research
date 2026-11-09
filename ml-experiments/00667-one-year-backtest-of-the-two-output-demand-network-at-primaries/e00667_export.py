# ruff: noqa: D103
"""Experiment 667: turn the Candidate's monthly fits into the contract's `primaries.parquet` rows.

The network runs at its native origin of 09:00 UTC (the 00Z ENS run plus 9 hours). The contract's
origin is 12:00 UTC the same day, valid from 12:00 to 11:30 UTC 14 days later (672 half-hours), so
this module keeps the valid times in that window, labels `power_fcst_init_time` 12:00 UTC and
`nwp_init_time` 00:00 UTC of the same day, averages the two seeds' quantiles, and sorts them.

Usage: python e00667_export.py MONTH ...   (writes the NRA rows of each month to exports/NRA_<month>.parquet)
"""

import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
)
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
SEEDS: Final = (0, 1)
CONTRACT_HOUR: Final = 12
NETWORK_ORIGIN_HOUR: Final = 9
HALF_HOURS: Final = 14 * 48


def fits_dir(*, month: str) -> Path:
    return ROOT / f"P7_D50_{month}" / "work" / "fits" / "P5"


def seed_mean(*, frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Return the seed mean of each quantile, sorted within the row (no crossing)."""
    cat = pl.concat(frames)
    mean = cat.group_by(KEYS).agg([pl.col(q).mean() for q in QCOLS]).sort(KEYS)
    q = np.sort(mean.select(QCOLS).to_numpy().astype(np.float32), axis=1)
    return mean.with_columns([pl.Series(name, q[:, i]) for i, name in enumerate(QCOLS)])


def contract_rows(*, mean: pl.DataFrame) -> pl.DataFrame:
    """Relabel the network origin (09:00) as the contract origin (12:00) and keep 12:00 to +14 d."""
    assert (mean["power_fcst_init_time"].dt.hour() == NETWORK_ORIGIN_HOUR).all()
    assert (mean["power_fcst_init_time"].dt.minute() == 0).all()
    origin = pl.col("power_fcst_init_time") + timedelta(hours=CONTRACT_HOUR - NETWORK_ORIGIN_HOUR)
    out = (
        mean.with_columns(contract_origin=origin)
        .filter(
            (pl.col("valid_time") >= pl.col("contract_origin"))
            & (pl.col("valid_time") < pl.col("contract_origin") + timedelta(days=14))
        )
        .select(
            pl.col("time_series_id").cast(pl.Int8),
            pl.col("valid_time"),
            nwp_init_time=pl.col("contract_origin").dt.truncate("1d"),
            power_fcst_init_time=pl.col("contract_origin"),
            *QCOLS,
        )
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    return out


def nra_month(*, month: str) -> pl.DataFrame:
    frames = [
        pl.read_parquet(fits_dir(month=month) / f"{month}_d0_s{s}.parquet", columns=[*KEYS, *QCOLS])
        for s in SEEDS
    ]
    assert all(f.height == frames[0].height for f in frames), "seeds differ in rows"
    for f in frames:
        assert f.height == f.unique(KEYS).height, "duplicate keys in a seed's fit"
    return contract_rows(mean=seed_mean(frames=frames))


def main() -> None:
    out = ROOT / "exports"
    out.mkdir(exist_ok=True)
    for month in sys.argv[1:]:
        frame = nra_month(month=month)
        tmp = out / f"NRA_{month}.parquet.tmp"
        frame.write_parquet(tmp)
        tmp.rename(out / f"NRA_{month}.parquet")
        n_orig = frame["power_fcst_init_time"].n_unique()
        print(month, frame.height, "rows;", n_orig, "origins; per series-origin",
              frame.group_by("time_series_id", "power_fcst_init_time").len()["len"].unique().to_list())


if __name__ == "__main__":
    main()
