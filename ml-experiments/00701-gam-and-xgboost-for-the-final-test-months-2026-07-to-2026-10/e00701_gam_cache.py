# ruff: noqa: E501
"""Experiment 701, GAM side, step 2: build the extended feature cache and prove it equals the stored one.

    PYTHONPATH=ml-experiments/00701-* DATA_PATH_INTERNAL=<real data folder> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_cache.py

`e00701_gam_extend` moves `studies.dp_features.GRID_END` to 2026-11-14 23:30 and points the default
input paths at the extended tables of `e00701_gam_inputs.py`, ERA5 file of experiment 607, and the
combined CAMS file. The script builds the cache with `load_or_build_feature_cache` into
`<ROOT>/gam/cache` (experiment 106's cache folder is only read), then loads the stored cache and
prints, for every Primary and every array, the largest absolute difference on the half-hours that
both grids hold (and the number of positions where exactly one of the two values is `NaN`). The
script stops if any array differs by more than 0.
"""

import time
from pathlib import Path

import e00701_gam_extend as ext
import numpy as np

from studies import dp_features as dpf

STORED_CACHE_DIR: Path = (
    ext.REAL_DATA
    / "ml-experiments"
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
STORED_INPUTS = dpf.DpInputPaths(
    rows=ext.REAL_DATA
    / "ml-experiments"
    / "00103-differentiable-physics-1-data-look"
    / dpf.ROWS_NAME,
    series_table=ext.REAL_DATA
    / "ml-experiments"
    / "00103-differentiable-physics-1-data-look"
    / dpf.SERIES_TABLE_NAME,
    sites=ext.REAL_DATA
    / "ml-experiments"
    / "00103-differentiable-physics-1-data-look"
    / dpf.SITES_NAME,
    era5=ext.REAL_DATA / "studies/per_study/stage1_pilot/demand/inputs/era5_features.parquet",
    cams=ext.REAL_DATA
    / "studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet",
)


def compare_arrays(*, name: str, old: np.ndarray, new: np.ndarray) -> tuple[float, int]:
    """Return the largest absolute difference where both are finite and the count of one-sided NaNs."""
    new = new[: len(old)]
    if old.dtype.kind == "f":
        nan_mismatch = int((np.isnan(old) != np.isnan(new)).sum())
        both = np.isfinite(old) & np.isfinite(new)
        gap = float(np.abs(old[both] - new[both]).max()) if both.any() else 0.0
        return gap, nan_mismatch
    if old.dtype.kind in "biuM":
        return float((old != new).sum()), 0
    msg = f"{name}: unhandled dtype {old.dtype}"
    raise TypeError(msg)


def main() -> None:
    """Build the extended cache and compare it with the stored cache."""
    started = time.monotonic()
    paths = dpf.default_input_paths()
    print("extended inputs:", paths)
    extended = dpf.load_or_build_feature_cache(paths=paths, cache_dir=ext.CACHE_DIR)
    print(
        f"extended cache {extended.key}: {len(extended.end_times)} half-hours, {extended.end_times[0]} to {extended.end_times[-1]}, {time.monotonic() - started:.0f} s"
    )
    stored_path = STORED_CACHE_DIR / f"dp_features_{dpf.input_key(paths=STORED_INPUTS)}.npz"
    assert stored_path.exists(), (
        f"the stored cache {stored_path} is missing; this script never builds into experiment 106's folder"
    )
    stored = dpf.load_feature_cache(path=stored_path)
    n = len(stored.end_times)
    print(
        f"stored cache {stored.key}: {n} half-hours, {stored.end_times[0]} to {stored.end_times[-1]}"
    )
    assert (extended.end_times[:n] == stored.end_times).all(), "the grids differ"
    assert set(extended.sites) == set(stored.sites)
    worst = 0.0
    nan_total = 0
    for name in dpf._CALENDAR_ARRAYS:
        old, new = getattr(stored.calendar, name), getattr(extended.calendar, name)
        gap, nans = compare_arrays(name=f"calendar/{name}", old=old, new=new)
        worst, nan_total = max(worst, gap), nan_total + nans
        print(f"calendar/{name:15s} max abs diff {gap:.3g}, one-sided NaN {nans}")
    for site in sorted(stored.sites, key=lambda s: int(s[1:])):
        parts = []
        for name in dpf._SITE_ARRAYS:
            gap, nans = compare_arrays(
                name=f"{site}/{name}",
                old=getattr(stored.sites[site], name),
                new=getattr(extended.sites[site], name),
            )
            worst, nan_total = max(worst, gap), nan_total + nans
            parts.append(f"{name}={gap:.3g}/{nans}")
        print(site, " ".join(parts))
        assert stored.sites[site].table == extended.sites[site].table, f"{site} table differs"
    print(
        f"WORST max abs diff over every array and Primary: {worst}; one-sided NaN positions: {nan_total}"
    )
    assert worst == 0.0, "the extended cache differs from the stored cache"
    assert nan_total == 0, "the NaN positions differ"
    tail = slice(n, None)
    for site in sorted(extended.sites, key=lambda s: int(s[1:])):
        features = extended.sites[site]
        print(
            site, "new half-hours", len(extended.end_times) - n,
            "present", int(features.present[tail].sum()),
            "finite y", int(np.isfinite(features.y[tail]).sum()),
            "finite temp (to 2026-10-31)", int(np.isfinite(features.temperature_c[tail]).sum()),
            "finite pv shape", int(np.isfinite(features.pv_unit_shape[tail]).sum()),
        )  # fmt: skip


if __name__ == "__main__":
    main()
