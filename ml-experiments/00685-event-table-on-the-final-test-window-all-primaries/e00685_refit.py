"""Refit experiment 327's arm C (the lag-free convex demand model) over the final-test window.

Arm C trains on every row with a reading, refitted monthly on the 1st at 00:00 UTC. This script
reuses experiment 327's recipe unchanged (`fit_job` there is the model; the code below repeats its
loop for the new cuts only) and differs in four ways, all in service of reading the window:

1. The feature cache is built from `inputs/` (experiment 103's rows, ERA5 temperature, and CAMS
   irradiance, each extended to 2026-10-31), on a grid that ends at 2026-10-31 23:30.
2. `HOLDOUT_START` and `GRID_END` are patched in `studies.dp_features`, and `HOLDOUT_START` in
   `studies.dp_gam`, in this process only. The maintainer opened the final-test window for the
   event table; nothing else may use this script.
3. Cuts are 2026-06-01 (a reproduction gate against experiment 327's stored forecast), then
   2026-07-01, 2026-08-01, 2026-09-01, and 2026-10-01. The last cut forecasts to 2026-11-01.
4. The output is the stored forecast array padded to the new grid, with the window filled in from
   the new cuts, at `e327_extended/forecast/C/forecast_<series>.npy`.

Run from the repository root:

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    PYTHONPATH=ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00685-*/e00685_refit.py
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00130_gam_panel as g  # ty: ignore[unresolved-import]
import numpy as np
import studies.dp_features as dpf
import studies.dp_gam as dpg
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import DpInputPaths, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E327: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00685-event-table-on-the-final-test-window-all-primaries"
)
INPUTS: Final[Path] = EXPERIMENT_DIR / "inputs"
E327_EXTENDED: Final[Path] = EXPERIMENT_DIR / "e327_extended"
"""Experiment 327's folder layout (`power/`, `forecast/C/`) on the extended grid."""
OUT: Final[Path] = E327_EXTENDED / "forecast" / "C"
GRID_START64: Final = np.datetime64("2022-01-01T00:00", "us")
OLD_LENGTH: Final[int] = 78816
NEW_LENGTH: Final[int] = 84720
"""Half-hours from 2022-01-01 00:00 to the end of 2026-10-31 (1,765 days)."""
WINDOW_STOP: Final[np.datetime64] = np.datetime64("2026-11-01T00:00", "us")
N_WORKERS: Final[int] = 2
CUT_MONTHS: Final[tuple[tuple[int, int], ...]] = (
    (2026, 6),
    (2026, 7),
    (2026, 8),
    (2026, 9),
    (2026, 10),
)
GATE_MAX_ABS_MW: Final[float] = 1e-6


def patch_window_guards() -> None:
    """Move the three constants that close the window, in this process only."""
    dpf.HOLDOUT_START = datetime(2026, 11, 1, tzinfo=UTC)
    dpf.GRID_END = datetime(2026, 10, 31, 23, 30, tzinfo=UTC)
    dpg.HOLDOUT_START = datetime(2026, 11, 1, tzinfo=UTC)


def extended_paths() -> DpInputPaths:
    """Experiment 103's input paths with rows, ERA5, and CAMS replaced by the extended files."""
    return replace(
        default_input_paths(),
        rows=INPUTS / "rows.parquet",
        era5=INPUTS / "era5_features.parquet",
        cams=INPUTS / "cams.parquet",
    )


def fit_job(series: int) -> str:
    """Fit arm C at each new cut for one series and write its extended forecast array."""
    patch_window_guards()
    torch.set_num_threads(1)
    site, capacity = g.site_of_series()[series]
    cache = load_or_build_feature_cache(paths=extended_paths(), cache_dir=INPUTS / "cache")
    features = cache.sites[site]
    rows = features.default_rows | (features.switching_mask & features.in_pool & ~features.lockdown)
    end = cache.end_times
    row_stamps = ((end - GRID_START64) // np.timedelta64(30, "m")).astype(np.int64)
    spec = g.base_spec()
    stored = np.load(E327 / "forecast" / "C" / f"forecast_{series}.npy")
    forecast = np.full(NEW_LENGTH, np.nan)
    forecast[:OLD_LENGTH] = stored[:OLD_LENGTH]
    gate = 0.0
    cuts = [datetime(y, m, 1, tzinfo=UTC) for y, m in CUT_MONTHS]
    for k, cut in enumerate(cuts):
        naive_cut = np.datetime64(cut.replace(tzinfo=None), "us")
        usable = rows & np.isfinite(features.y)
        if int((usable & (end <= naive_cut)).sum()) < g.MIN_TRAINING_DAYS * 48:
            continue
        trial = replace(cache, sites={**cache.sites, site: replace(features, default_rows=rows)})
        try:
            fit = fit_gam(
                cache=trial,
                site=site,
                specs=[spec],
                cut=cut,
                window=TrainingWindow(months=g.WINDOW_MONTHS),
                season_years=g.SEASON_YEARS,
            )
        except ValueError:
            continue
        stop = (
            np.datetime64(cuts[k + 1].replace(tzinfo=None), "us")
            if k + 1 < len(cuts)
            else WINDOW_STOP
        )
        target = np.flatnonzero(
            (end > naive_cut)
            & (end <= stop)
            & np.isfinite(features.y)
            & np.isfinite(features.temperature_c)
        )  # the last half-hour has no ERA5 hour after it
        if not len(target):
            continue
        points = predict_gam(cache=cache, fit=fit, index=target)
        stamps = row_stamps[target]
        keep = (stamps >= 0) & (stamps < NEW_LENGTH)
        mw = points[keep] * capacity / 100.0
        if k == 0:
            kept = stamps[keep]
            old_part = kept < OLD_LENGTH
            both = old_part & np.isfinite(stored[np.minimum(kept, OLD_LENGTH - 1)])
            gate = float(np.max(np.abs(mw[both] - stored[kept[both]]))) if both.any() else np.inf
            forecast[kept[~old_part]] = mw[~old_part]
            continue
        forecast[stamps[keep]] = mw
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"forecast_{series}.npy.tmp.npy"
    np.save(tmp, forecast)
    tmp.rename(OUT / f"forecast_{series}.npy")
    window = forecast[OLD_LENGTH:]
    if gate > GATE_MAX_ABS_MW:
        msg = f"series {series}: the 2026-06-01 cut differs from experiment 327 by {gate:.2e} MW"
        raise ValueError(msg)
    return (
        f"series {series}: gate max abs diff {gate:.2e} MW; window half-hours with a forecast "
        f"{int(np.isfinite(window).sum())} of {len(window)}"
    )


if __name__ == "__main__":
    patch_window_guards()
    load_or_build_feature_cache(paths=extended_paths(), cache_dir=INPUTS / "cache")
    series_list = [s for s in g.SERIES_IDS if s != 12]
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        for line in pool.imap_unordered(fit_job, series_list):
            print(line, flush=True)
