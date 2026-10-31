"""Build the raw and widened forecasts of the mean-only and std-only arms.

Reads `work/anchor_preds/<arm>_m/*.parquet` of this experiment, joins the actual as experiment 389's
`e00389_calibrate.py` does, and writes `work/arms/<ARM>_raw.parquet` and `<ARM>_base.parquet` in
fractions of capacity. It also copies experiment 389's stored `S0M` and `S1M` arms beside them so
the scorer reads one folder.

    PYTHONPATH=ml-experiments/00379-*:ml-experiments/00087-*:ml-experiments/00389-* \
        DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00394-*/e00394_arms.py mean
"""

import shutil
import sys
from pathlib import Path
from typing import Final

import e00389_calibrate as cal  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00394-xgboost-ensemble-mean-and-std-columns-separately" / "work"
)
REFITS: Final[int] = 12


def main() -> None:
    """Write the arms of the named ensemble-column variant."""
    arm = sys.argv[1]
    stored = pl.read_parquet(cal.STORED_BG2).select(*cal.KEYS, "actual", "site")
    ids = stored["time_series_id"].unique().to_list()
    files = sorted((WORK / "anchor_preds" / f"{arm}_m").glob("*.parquet"))
    if len(files) != REFITS * len(ids):
        msg = f"{arm} is incomplete: {len(files)} files"
        raise FileNotFoundError(msg)
    raw = (
        pl.concat([pl.read_parquet(f) for f in files])
        .filter(pl.col("power_fcst_init_time") >= cal.FIRST_REFIT)
        .select(*cal.KEYS, *cal.Q)
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .filter(pl.col("lead_hours") <= 336.0)
        .join(stored.select(*cal.KEYS, "actual", "site"), on=cal.KEYS, how="inner", validate="1:1")
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    out = WORK / "arms"
    out.mkdir(parents=True, exist_ok=True)
    quantiles = raw.select(cal.Q).to_numpy().astype(np.float64)
    median = quantiles[:, [cal.MEDIAN]]
    wide = median + cal.BASE_FACTOR * (quantiles - median)
    for suffix, values in (("raw", quantiles), ("base", wide)):
        keys = raw.select(*cal.KEYS, "lead_hours")
        keys.hstack(pl.DataFrame(values.astype(np.float32), schema=list(cal.Q))).write_parquet(
            out / f"{arm.upper()}_{suffix}.parquet"
        )
    for name in ("S0M_raw", "S0M_base", "S1M_raw", "S1M_base"):
        shutil.copy(cal.OUT / "arms" / f"{name}.parquet", out / f"{name}.parquet")
    print(f"{arm}: {raw.height:,} rows written", flush=True)


if __name__ == "__main__":
    main()
