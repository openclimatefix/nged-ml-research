# ruff: noqa: B905, F405, UP031
"""Audit of experiment 696: which Primary is the host."""

import datetime as dt

import numpy as np
import polars as pl
from e00696_audit_common import *  # noqa: F403

caps = pl.read_parquet(SH / "series_capacity.parquet")
cap = dict(zip(caps["time_series_id"].cast(pl.Int64).to_list(), caps["capacity"].to_list()))
gt = pl.read_parquet(SH / "ground_truth_primaries.parquet").with_columns(
    pl.col("time_series_id").cast(pl.Int64)
)
v = np.load(E5 / "stacks" / "W0" / "2026-06.npz")["values"].astype(float)
start = np.datetime64("2019-09-01T00:00", "us")
for i in sorted(cap):
    g = gt.filter(
        pl.col("time_series_id") == i,
        pl.col("valid_time") >= dt.datetime(2025, 11, 1, tzinfo=dt.UTC),
        pl.col("valid_time") < dt.datetime(2026, 7, 1, tzinfo=dt.UTC),
    )
    idx = (
        (g["valid_time"].dt.replace_time_zone(None).to_numpy() - start) // np.timedelta64(30, "m")
    ).astype(int)
    diff = v[idx] * cap[i] - g["power"].to_numpy()
    ok = np.isfinite(diff)
    if not ok.any():
        print(i, "no truth")
        continue
    print(
        i,
        round(cap[i], 3),
        "share |diff|<1e-3: %.3f" % (np.abs(diff[ok]) < 1e-3).mean(),
        "min %.3f" % np.nanmin(diff),
        "mean %.3f" % np.nanmean(diff),
    )
