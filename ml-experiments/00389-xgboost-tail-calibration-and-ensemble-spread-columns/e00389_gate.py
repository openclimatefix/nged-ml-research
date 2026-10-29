"""Compare the refitted baseline (S0) with experiment 87's stored weekly refits, file by file.

The gate passes when, over the refits and Primaries refitted, the mean absolute gap of the 11
quantiles is under 0.001 points of capacity and the CRPS gap is under 0.005 points.

    PYTHONPATH=ml-experiments/00087-* DATA_PATH_INTERNAL=... uv run --no-sync python \
        ml-experiments/00389-*/e00389_gate.py
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00389-xgboost-tail-calibration-and-ensemble-spread-columns"
    / "work"
    / "anchor_preds"
    / "s0"
)
STORED: Final[Path] = common.ANCHOR_PREDS_DIR / "n1h"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
Q: Final[list[str]] = list(common.LEVEL_COLUMNS)
MEAN_GAP_POINTS: Final[float] = 0.001
CRPS_GAP_POINTS: Final[float] = 0.005


def crps_points(*, frame: pl.DataFrame, suffix: str) -> float:
    """Return the mean CRPS in points of capacity from 11 quantiles."""
    actual = frame["actual"].to_numpy()
    quantiles = frame.select([f"{q}{suffix}" for q in Q]).to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    levels = common.LEVELS[None, :]
    pinball = np.maximum(levels * error, (levels - 1.0) * error)
    return float(100.0 * 2.0 * pinball.mean())


def main() -> None:
    """Print the gap per file and overall."""
    actuals = pl.read_parquet(common.WORK_DIR / "arms" / "p_s6_bg2.parquet").select(*KEYS, "actual")
    gaps, crps_gaps = [], []
    for path in sorted(OUT.glob("*.parquet")):
        stored_path = STORED / path.name
        mine = pl.read_parquet(path)
        theirs = pl.read_parquet(stored_path)
        joined = (
            mine.join(theirs, on=KEYS, suffix="_87", validate="1:1")
            .join(actuals, on=KEYS, validate="m:1")
            .filter(pl.col("actual").is_finite())
        )
        gap = 100.0 * np.mean([(joined[q] - joined[f"{q}_87"]).abs().mean() for q in Q])
        crps_gap = crps_points(frame=joined, suffix="") - crps_points(frame=joined, suffix="_87")
        gaps.append(gap)
        crps_gaps.append(crps_gap)
        print(
            f"{path.name}: {joined.height:,} rows, mean |gap| {gap:.5f} points, CRPS gap {crps_gap:+.5f}"
        )
    print(
        f"overall: mean |gap| {np.mean(gaps):.5f} points (limit {MEAN_GAP_POINTS}), "
        f"mean CRPS gap {np.mean(crps_gaps):+.5f} (limit {CRPS_GAP_POINTS})"
    )
    passed = np.mean(gaps) < MEAN_GAP_POINTS and abs(np.mean(crps_gaps)) < CRPS_GAP_POINTS
    print("GATE", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    main()
