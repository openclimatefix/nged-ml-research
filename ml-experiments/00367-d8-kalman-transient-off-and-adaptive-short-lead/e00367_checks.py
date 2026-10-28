# ruff: noqa: E501
"""Print the audit checks of experiment 367: median minus mean forecast, and the mean-forecast error contrast."""

import os
from pathlib import Path

import numpy as np
import polars as pl

SRC = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments/00364-d8-stage-blame-and-analogue-recent-power/work"
)
for site, key in {"D8": "D8_weekly", "D2": "D2_monthly", "D4": "D4_monthly"}.items():
    w = pl.read_parquet(SRC / f"wide_{key}.parquet")
    short = w["lead_hours"].to_numpy() <= 24.0
    actual = w["actual"].to_numpy().astype(float)
    t = (w["S2__q50"] - w["S1__q50"]).to_numpy().astype(float)
    mean_s4 = w["S4__mean"].to_numpy().astype(float)
    med_s4 = w["S4__q50"].to_numpy().astype(float)
    mae_ctl = np.abs(mean_s4 - actual)[short].mean()
    mae_k0 = np.abs(mean_s4 - t - actual)[short].mean()
    print(site, "median minus mean (S4, all rows, points):", round(float((med_s4 - mean_s4).mean()), 4),
          "| mean-forecast MAE at 0-24h control", round(float(mae_ctl), 3), "K0", round(float(mae_k0), 3),
          "diff", round(float(mae_k0 - mae_ctl), 3))  # fmt: skip
