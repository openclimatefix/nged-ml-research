# ruff: noqa: D103, E501
"""Experiment 667: score the exported rows against the cleaned observed power (outside the file).

The score of a row is its CRPS (twice the mean pinball loss over the 11 stored quantile levels) and
the absolute error of its median, both divided by the Primary's capacity and multiplied by 100
(points of capacity). Rows whose observed power is missing or flagged are not scored. Output:
`score/<name>_by_band_month.csv` and `score/<name>_by_band_primary.csv`.

Usage: python e00667_score.py NAME   (reads exports/<NAME>_<month>.parquet for the 12 months)
"""

import sys
from pathlib import Path
from typing import Final

import e00510_common as c510  # ty: ignore[unresolved-import]
import e00667_export as exp
import e00667_extend as ext  # ty: ignore[unresolved-import]
import e00578_common as k578  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

TAUS: Final = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
BANDS: Final = ((0, 1), (1, 3), (3, 7), (7, 14))


def score_frame(*, frame: pl.DataFrame, clean: np.ndarray, grid_start_us: int, step_us: int) -> pl.DataFrame:
    """Return `frame` with columns `crps_pts`, `abs_err_pts`, `lead_days`, and `observed` (NaN where not scored)."""
    sm = k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    cap_of = dict(zip(sm["time_series_id"].to_list(), sm["capacity_mw"].to_list(), strict=True))
    ts = frame["time_series_id"].to_numpy().astype(int)
    prim = np.array([prim_of[t] for t in ts])
    cap = np.array([cap_of[t] for t in ts])
    idx = ((frame["valid_time"].dt.epoch("us").to_numpy() - grid_start_us) // step_us).astype(int)
    y_pu = np.where((idx >= 0) & (idx < clean.shape[1]), clean[prim, np.clip(idx, 0, clean.shape[1] - 1)], np.nan)
    q_pu = frame.select(exp.QCOLS).to_numpy().astype(np.float64) / cap[:, None]
    d = y_pu[:, None] - q_pu
    pin = np.maximum(TAUS * d, (TAUS - 1.0) * d)
    lead = (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes().to_numpy() / 1440.0
    return frame.select("time_series_id", "power_fcst_init_time").with_columns(
        pl.Series("crps_pts", 200.0 * pin.mean(axis=1)),
        pl.Series("abs_err_pts", 100.0 * np.abs(y_pu - q_pu[:, 5])),
        pl.Series("lead_days", lead),
        pl.Series("observed", y_pu),
    )


def main() -> None:
    name = sys.argv[1]
    ext.redirect()
    clean = c510.clean_stack()
    grid_start, step = c510.prep422.GRID_START_US, c510.prep422.STEP_US
    parts = []
    for month in MONTHS:
        f = pl.read_parquet(exp.ROOT / "exports" / f"{name}_{month}.parquet")
        s = score_frame(frame=f, clean=clean, grid_start_us=grid_start, step_us=step).with_columns(month=pl.lit(month))
        parts.append(s)
    s = pl.concat(parts)
    n_all = s.height
    s = s.filter(pl.col("observed").is_finite())
    print(name, "rows", n_all, "scored", s.height, "(observed power present and unflagged)")
    band = pl.when(pl.col("lead_days") < 1).then(pl.lit("0-1 d")).when(pl.col("lead_days") < 3).then(pl.lit("1-3 d")).when(pl.col("lead_days") < 7).then(pl.lit("3-7 d")).otherwise(pl.lit("7-14 d")).alias("band")
    s = s.with_columns(band)
    out = exp.ROOT / "score"
    out.mkdir(exist_ok=True)
    agg = [pl.col("crps_pts").mean().alias("crps_pts"), pl.col("abs_err_pts").mean().alias("nmae_median_pts"), pl.len().alias("rows")]
    by_month = s.group_by("band", "month").agg(agg).sort("band", "month")
    by_prim = s.group_by("band", "time_series_id").agg(agg).sort("band", "time_series_id")
    allrows = s.group_by("band").agg(agg).sort("band")
    by_month.write_csv(out / f"{name}_by_band_month.csv")
    by_prim.write_csv(out / f"{name}_by_band_primary.csv")
    print(allrows)
    print("all bands:", s.select(*agg))


if __name__ == "__main__":
    main()
