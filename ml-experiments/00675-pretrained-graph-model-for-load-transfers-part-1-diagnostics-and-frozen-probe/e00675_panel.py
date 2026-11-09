"""Step 0b of experiment 675: a leave-one-year-out (cross-fitted) lag-free convex residual panel.

For each of the 15 Primaries and each calendar year Y of 2022 to 2026-06, the convex demand model
of experiment 130 (no drift, fitted PV capacity, no lagged power) is fitted on every cleaned
half-hour outside year Y, and forecasts year Y. Year Y's events therefore never enter the copy that
scores it, and the model has no free slow offset. Weather is recorded (ERA5 and CAMS), cut at
2026-07-01. Output: one forecast array per series on experiment 258's half-hour grid, in MW, at
`<data>/forecast_loyo/forecast_<series>.npy`.

    PYTHONPATH=ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual:\
ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline \
        systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00675_panel.py [series ...]
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00130_gam_panel as g  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_gam import TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe"
    / "forecast_loyo"
)
GRID_START64: Final = np.datetime64("2022-01-01T00:00", "us")
YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024, 2025, 2026)
CUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Fits may train on any row that ends before the held-out 2026-07-01."""


def fit_job(series: int) -> str:
    """Fit and forecast every held-out year of one series."""
    torch.set_num_threads(1)
    site, capacity = g.site_of_series()[series]
    cache = g.load_cache()
    features = cache.sites[site]
    end = cache.end_times
    stamps = ((end - GRID_START64) // np.timedelta64(30, "m")).astype(np.int64)
    year_of = end.astype("datetime64[Y]").astype(int) + 1970
    all_rows = features.default_rows | (
        features.switching_mask & features.in_pool & ~features.lockdown
    )
    forecast = np.full(g.GRID_LENGTH, np.nan)
    for year in YEARS:
        rows = all_rows & (year_of != year)
        trial = replace(cache, sites={**cache.sites, site: replace(features, default_rows=rows)})
        fit = fit_gam(
            cache=trial,
            site=site,
            specs=[g.base_spec()],
            cut=CUT,
            window=TrainingWindow(),
            season_years=g.SEASON_YEARS,
        )
        target = np.flatnonzero((year_of == year) & np.isfinite(features.y) & (end < np.datetime64("2026-07-01", "us")))
        if not len(target):
            continue
        points = predict_gam(cache=cache, fit=fit, index=target)
        st = stamps[target]
        keep = (st >= 0) & (st < g.GRID_LENGTH)
        forecast[st[keep]] = points[keep] * capacity / 100.0
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"forecast_{series}.npy.tmp.npy"
    np.save(tmp, forecast)
    tmp.rename(OUT / f"forecast_{series}.npy")
    return f"series {series}: {int(np.isfinite(forecast).sum())} forecast half-hours"


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or list(g.SERIES_IDS)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(4, len(ids))) as pool:
        for line in pool.imap_unordered(fit_job, ids):
            print(line, flush=True)
