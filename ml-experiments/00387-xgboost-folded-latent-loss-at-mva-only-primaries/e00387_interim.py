"""Interim scoring of the real-data arms: mean pinball loss over the 11 levels (x2, a CRPS estimate).

Points of effective capacity, days 1 to 10, all refits written so far. Not the `demand-v1` scorer.
"""

import sys
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00387_fit as fit
import numpy as np
import polars as pl

LEVELS = common.LEVELS
COLS = list(common.LEVEL_COLUMNS)


def main() -> None:
    """Print CRPS estimate and sunny bias per site and arm."""
    sites = sys.argv[1:]
    series = common.series_table()
    arrays = common.load_series_arrays(series=series.filter(pl.col("site").is_in(sites)))
    for site in sites:
        a = arrays[site]
        grid = pl.DataFrame({"valid_time": a.valid_time, "actual": a.power * 100,
                             "shortwave": a.features[:, common.feature_columns().index("shortwave")]})
        for arm_dir in sorted((fit.WORK / "preds" / "real").iterdir()):
            files = sorted(arm_dir.glob(f"{site}_*.parquet"))
            if not files:
                continue
            p = pl.concat([pl.read_parquet(f) for f in files]).join(grid, on="valid_time")
            p = p.with_columns(lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours())
            p = p.filter((pl.col("lead") <= 240) & pl.col("actual").is_finite())
            q = p.select(COLS).to_numpy() * 100
            y = p["actual"].to_numpy()[:, None]
            u = y - q
            crps = float((2 * np.maximum(LEVELS * u, (LEVELS - 1) * u)).mean())
            sunny = (p["shortwave"] > 600).to_numpy()
            bias = float((q[:, 5] - p["actual"].to_numpy())[sunny].mean())
            print(f"{site} {arm_dir.name:14s} n={p.height:7d} CRPS~{crps:.3f} sunny median-actual {bias:+.2f}")


if __name__ == "__main__":
    main()
