# ruff: noqa: E501
"""Experiment 725: the CPU step that caches the ENS arrays, and the per-fold standardisation of the cache.

**What the cache holds.** For each Primary and each grid half-hour, the weather that the live rule
reads at ages 2 to 6 and in the youngest cells of age 3 (design sections 2.3 and 9):

- `ens_raw.npy`: (16, G, 19) float32. Columns 0 to 6 are the seven ENS-mean weather values of the
  `ft` table (`e00560_data.raw_table(source="ft")`, the stored run with the smallest lead for the
  half-hour). Columns 7 to 18 are the twelve trailing-window columns of experiment 437's stored table
  (`TableLookup`, columns `B1`), looked up at the key (the `ft` run's forecast origin, the half-hour).
  NaN where a row is missing.
- `ens_origin.npy`: (16, G) int32. The grid index of the `ft` run's forecast origin (the run's issue
  time plus 9 hours); -1 where the half-hour has no `ft` row.
- `ens_found.npy`: (16, G) bool. True where both the `ft` row and the stored B1 row exist.
- `meta.json`: the folders read, their modification times, the grid length, and the missing share per
  Primary.

The cache holds raw values. The fits standardise them with the fold's statistics through
`standardise_cache`, so the file does not depend on the fold. The fits load the cache and never the
parquet tables of experiment 437.

The step runs on the CPU in its own scope with one thread. Nothing is written into another
experiment's folder.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NamedTuple

import e00725_common as cm
import numpy as np

N_WEATHER: Final = 7
N_B1_WINDOW: Final = 12
N_ENS: Final = N_WEATHER + N_B1_WINDOW
CLIP_SD: Final = 5.0
"""Experiment 632's clip of the standardised weather (`e00632_data.CLIP_SD`)."""
GRID_START: Final = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final = datetime(2026, 11, 20, 23, 30, tzinfo=UTC)
"""The last half-hour of the grid that `e00667_extend.redirect` sets."""
STEP_US: Final = 1_800_000_000
N_GRID: Final = int((GRID_END - GRID_START).total_seconds() // 1800) + 1
NAME_560: Final = "00560-modular-lag-encoder-capped-lags-and-block-dropout"


class CacheDirs(NamedTuple):
    """The folders a phase's cache reads.

    Attributes:
        expected: The folder that holds `expected_ft_cols.npy` and `expected_ft_origin.npy`.
        trailing: The folder of experiment 437's per-Primary tables.
    """

    expected: Path
    trailing: Path


def phase_dirs(*, phase: str, data_dir: Path) -> CacheDirs:
    """Return the folders that each phase reads (design section 2.3).

    Training reads experiment 560's `expected` cache and experiment 437's `trailing` folder.
    Prediction reads experiment 667's extended copies, the ones `e00667_fold` redirects to.
    """
    assert phase in {"train", "predict"}, phase
    if phase == "train":
        import e00437_features as f437  # ty: ignore[unresolved-import]

        return CacheDirs(
            expected=data_dir / NAME_560 / "work" / "expected", trailing=Path(f437.TRAILING_DIR)
        )
    import e00667_extend as ext  # ty: ignore[unresolved-import]

    root = Path(ext.OUT_ROOT)
    return CacheDirs(
        expected=root / "inputs" / "w560" / "expected", trailing=root / "inputs" / "trailing437"
    )


def pad_to_grid(*, array: np.ndarray, n_grid: int, fill: float) -> np.ndarray:
    """Return `array` (primaries, G, ...) padded or cut along axis 1 to `n_grid`, new cells set to `fill`."""
    have = array.shape[1]
    if have == n_grid:
        return np.asarray(array)
    out = np.full((array.shape[0], n_grid, *array.shape[2:]), fill, dtype=array.dtype)
    keep = min(have, n_grid)
    out[:, :keep] = np.asarray(array[:, :keep])
    return out


def assemble_raw(
    *, cols: np.ndarray, origin_us: np.ndarray, grid_start_us: int, b1_lookup: object, n_grid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw, origin index, found) of the cache from the `ft` table and a B1 lookup.

    Args:
        cols: (16, G0, >=7) the `ft` table's raw columns; NaN where the half-hour has no row.
        origin_us: (16, G0) int64 forecast origin in microseconds; 0 where the half-hour has no row.
        grid_start_us: The grid start in microseconds.
        b1_lookup: An object with `get(primary=, origin_idx=, valid_idx=)` returning (values, found)
            like `e00562_columns.TableLookup`, whose first `N_B1_WINDOW` columns are the trailing columns.
        n_grid: The grid length to pad to.

    Returns:
        The raw (16, n_grid, 19) float32 array, the (16, n_grid) int32 origin indices (-1 where
        missing), and the (16, n_grid) bool mask of cells with both a `ft` row and a stored B1 row.
    """
    n_primaries = cols.shape[0]
    cols = pad_to_grid(array=cols[..., :N_WEATHER], n_grid=n_grid, fill=np.nan)
    origin_us = pad_to_grid(array=origin_us, n_grid=n_grid, fill=0)
    raw = np.full((n_primaries, n_grid, N_ENS), np.nan, dtype=np.float32)
    origin_idx = np.full((n_primaries, n_grid), -1, dtype=np.int32)
    found = np.zeros((n_primaries, n_grid), dtype=bool)
    raw[..., :N_WEATHER] = cols
    for p in range(n_primaries):
        valid = np.nonzero(origin_us[p] > 0)[0]
        if len(valid) == 0:
            continue
        o_idx = ((origin_us[p, valid] - grid_start_us) // STEP_US).astype(np.int64)
        origin_idx[p, valid] = o_idx
        values, hit = b1_lookup.get(  # ty: ignore[unresolved-attribute]
            primary=np.full(len(valid), p, dtype=np.int64), origin_idx=o_idx, valid_idx=valid
        )
        raw[p, valid[hit], N_WEATHER:] = values[hit][:, :N_B1_WINDOW]
        found[p, valid[hit]] = True
    return raw, origin_idx, found


def build_phase_cache(*, phase: str, data_dir: Path, dirs: CacheDirs | None = None) -> dict:
    """Build and write the cache of one phase on the CPU, and return the metadata.

    The stored tables are read through experiment 560's `raw_table` (its `.npy` cache must exist) and
    experiment 562's `TableLookup` with the 14 columns of arm `B1`. The step needs the extended grid, so
    it calls `e00667_extend.redirect()` first, as `e00667_fold.install` does.
    """
    import e00397_prep as prep  # ty: ignore[unresolved-import]
    import e00422_prep as prep422  # ty: ignore[unresolved-import]
    import e00437_features as f437  # ty: ignore[unresolved-import]
    import e00560_data as d560  # ty: ignore[unresolved-import]
    import e00562_columns as c562  # ty: ignore[unresolved-import]
    import e00667_extend as ext  # ty: ignore[unresolved-import]

    raw_dir_original = Path(prep.RAW_DIR)
    ext.redirect()
    dirs = dirs or phase_dirs(phase=phase, data_dir=data_dir)
    for name in ("expected_ft_cols.npy", "expected_ft_origin.npy"):
        assert (dirs.expected / name).exists(), (
            f"{dirs.expected / name} is missing: `raw_table` would build and save it into the folder of "
            "another experiment, so the cache step refuses to run"
        )
    if phase == "predict":
        prep.RAW_DIR = ext.RAW_DIR
    raw_dir = Path(prep.RAW_DIR) if phase == "predict" else raw_dir_original
    f437.TRAILING_DIR = dirs.trailing
    assert int(prep422.GRID_START_US) == int(GRID_START.timestamp() * 1e6)
    cols, origin = d560.raw_table(source="ft", cache=dirs.expected)
    lookup = c562.TableLookup(columns=c562.TABLE_ARMS["B1"])
    raw, origin_idx, found = assemble_raw(
        cols=cols, origin_us=np.asarray(origin), grid_start_us=int(prep422.GRID_START_US),
        b1_lookup=lookup, n_grid=N_GRID,
    )  # fmt: skip
    out = cm.cache_dir(phase=phase, data_dir=data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, array in (("ens_raw", raw), ("ens_origin", origin_idx), ("ens_found", found)):
        tmp = out / f"{name}.tmp.npy"
        np.save(tmp, array)
        tmp.rename(out / f"{name}.npy")
    has_ft = origin_idx >= 0
    meta = {
        "phase": phase, "n_grid": N_GRID, "expected_dir": str(dirs.expected),
        "trailing_dir": str(dirs.trailing),
        "raw_dir": str(raw_dir),
        "expected_mtimes_ns": {
            p.name: p.stat().st_mtime_ns for p in sorted(dirs.expected.glob("expected_ft_*.npy"))
        },
        "ft_share_per_primary": has_ft.mean(axis=1).round(5).tolist(),
        "found_share_per_primary": found.mean(axis=1).round(5).tolist(),
        "ft_but_no_b1_cells": int((has_ft & ~found).sum()),
        "built_at": datetime.now(UTC).isoformat(),
    }  # fmt: skip
    cm.write_json_atomic(obj=meta, path=out / "meta.json")
    return meta


class EnsCache(NamedTuple):
    """The raw cache of a phase, memory-mapped.

    Attributes:
        raw: (16, G, 19) float32.
        origin_idx: (16, G) int32, -1 where missing.
        found: (16, G) bool.
    """

    raw: np.ndarray
    origin_idx: np.ndarray
    found: np.ndarray


def load_cache(*, phase: str, data_dir: Path) -> EnsCache:
    """Load the cache of a phase (never the parquet tables)."""
    folder = cm.cache_dir(phase=phase, data_dir=data_dir)
    assert (folder / "meta.json").exists(), f"{folder} has no cache: run the cache step first"
    return EnsCache(
        raw=np.load(folder / "ens_raw.npy", mmap_mode="r"),
        origin_idx=np.load(folder / "ens_origin.npy", mmap_mode="r"),
        found=np.load(folder / "ens_found.npy", mmap_mode="r"),
    )


def standardise_cache(
    *, cache: EnsCache, weather_mean: np.ndarray, weather_std: np.ndarray, b1_mean: np.ndarray,
    b1_std: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """Return the cache as float16 (16, G, 19), standardised with the fold's statistics.

    The weather columns follow `e00632_data.standardised_inputs` (clip at 5 standard deviations,
    NaN becomes 0). The B1 columns follow `e00632_data.b1_scale` (no clip, NaN becomes 0). Cells with no
    `ft` row, or no stored B1 row, are 0 in every column.

    Args:
        cache: The raw cache.
        weather_mean: (7,) the fold's mean of the weather columns of the Candidate's standardiser.
        weather_std: (7,) the standard deviation.
        b1_mean: (>=12,) the fold's stored B1 mean (`stats/B1/<month>.npz`).
        b1_std: (>=12,) the stored B1 standard deviation.
    """
    n_primaries, n_grid, _ = cache.raw.shape
    out = np.zeros((n_primaries, n_grid, N_ENS), dtype=np.float16)
    for p in range(n_primaries):
        raw = np.asarray(cache.raw[p])
        z_w = np.clip((raw[:, :N_WEATHER] - weather_mean) / weather_std, -CLIP_SD, CLIP_SD)
        z_b = (raw[:, N_WEATHER:] - b1_mean[:N_B1_WINDOW]) / b1_std[:N_B1_WINDOW]
        z = np.concatenate([z_w, z_b], axis=1)
        z = np.nan_to_num(z, nan=0.0)
        z[~np.asarray(cache.found[p])] = 0.0
        out[p] = z.astype(np.float16)
    return out


def coverage_report(*, cache: EnsCache, first: int, last: int) -> dict:
    """Return the share of half-hours in [first, last) with a usable cell, per Primary and pooled."""
    found = np.asarray(cache.found[:, first:last])
    return {"per_primary": found.mean(axis=1).round(5).tolist(), "pooled": float(found.mean())}


def read_meta(*, phase: str, data_dir: Path) -> dict:
    """Return a phase's cache metadata."""
    return json.loads((cm.cache_dir(phase=phase, data_dir=data_dir) / "meta.json").read_text())


def main() -> int:
    """Build the cache of one phase: `e00725_sources.py PHASE` (run it through `run.sh`, on the CPU)."""
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "predict"))
    args = parser.parse_args()
    meta = build_phase_cache(phase=args.phase, data_dir=ML_EXPERIMENTS_DATA_DIR)
    print(json.dumps(meta, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
