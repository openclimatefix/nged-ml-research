# ruff: noqa: E501
"""Smoke test of every real file that experiment 721's code reads. Read-only, capped, and silent
about values.

It opens each file once (one row, one site, or one fold), and prints one line per input with a
boolean,
a shape, or a column-presence flag. It prints no power value, no site name, no farm name, no
injected
size, and no generator series. It is not a pytest module: run it once by hand, only when the code
review's checklist says so:

    SLOT_CPU=<n> ml-experiments/00721-*/run.sh --go tests/smoke_real_files.py

(`run.sh` takes a script name relative to the experiment folder, so pass
`tests/smoke_real_files.py`.)
Exit status 0 means every input opened.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

import e00721_paths  # noqa: F401

# isort: split
import e00721_core as core
import numpy as np
import polars as pl

DATA = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
QCOLS = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
FARM_COLUMNS = ("site", "kind", "time", "power", "zenith", "ghi_u")


def _first(pattern: str) -> Path | None:
    found = sorted(DATA.glob(pattern))
    return found[0] if found else None


def check_dense_raw() -> str:
    """Return the shape flags of experiment 716's cached dense raw array (an `np.load` memory
    map)."""
    path = _first("00716-*/work/cache/dense_raw.npy")
    assert path is not None, "dense_raw.npy not found"
    raw = np.load(path, mmap_mode="r")
    return f"ndim {raw.ndim}, first axis is 16 Primaries: {raw.shape[0] == 16}, columns at least 21: {raw.shape[2] >= 21}"


def check_grid_and_sites() -> str:
    """Return flags for experiment 87's grid and the series table: 16 Primaries, both pairs
    present."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    grid = common.grid_times()
    table = common.series_table().filter(pl.col("substation_type") == "Primary")
    sites = set(table["site"].to_list())
    return (
        f"grid steps positive: {len(grid) > 0}, Primaries: {len(sites) == 16}, "
        f"hidden pair present: {set(core.HIDDEN_PAIR) <= sites}, visible signed present: {set(core.VISIBLE_SIGNED) <= sites}"
    )


def check_one_site_power_and_cams() -> str:
    """Return flags for one Primary's cleaned power and CAMS arrays (the first hidden Primary)."""
    import e00506_data as data  # ty: ignore[unresolved-import]

    site = core.HIDDEN_PAIR[0]
    power = data.power_points(site=site)
    u, k = data.cams(site=site)
    return (
        f"power, u and k have the same length: {len(power) == len(u) == len(k)}, "
        f"power has finite values: {bool(np.isfinite(power).any())}, u in 0 to 1.5: {bool(np.nanmax(u) <= 1.5)}"
    )


def check_farm_frame() -> str:
    """Return column-presence flags for the farm frame (its first row group only)."""
    import e00506_data as data  # ty: ignore[unresolved-import]

    schema = pl.read_parquet_schema(data.FARM_FRAME)
    return "farm frame has " + ", ".join(f"{c}: {c in schema}" for c in FARM_COLUMNS)


def check_stored_713_forecasts() -> str:
    """Return flags for one stored forecast file of experiment 713's reference network (the G3 power
    check)."""
    path = _first("00713-*/work/fits/A/*_s0.parquet") or _first("00713-*/bank713/*")
    assert path is not None, "no stored 713 forecast found"
    if path.is_dir():
        return f"bank folder holds files: {any(path.iterdir())}"
    frame = pl.scan_parquet(path).head(1).collect()
    return f"quantile columns present: {all(q in frame.columns for q in QCOLS)}, rows read: {frame.height}"


def check_stage0_curves() -> str:
    """Return flags for the stored learning curves that Stage 0 reads."""
    known = _first("00716-*/work/known_answers/WH_s0.json")
    stage1 = _first("00713-*/work/logs/A/*_fit_stage1.json")
    return f"716 known answers present: {known is not None}, 713 stage-1 logs present: {stage1 is not None}"


def check_capacity_probe_files() -> str:
    """Return whether experiment 662's folder holds anything for the capacity provider to read
    (names are not printed)."""
    folder = _first("00662-*")
    return f"662 data folder present: {folder is not None}, holds files: {folder is not None and any(folder.rglob('*'))}"


def check_as_of_capacity_tables() -> str:
    """Return whether experiment 716's as-of capacity function reads its stored tables for the 16 Primaries.

    The check prints booleans and a count, never a capacity.
    """
    import e00716_pv as pv  # ty: ignore[unresolved-import]
    import e00720_real as real  # ty: ignore[unresolved-import]

    caps = pv.capacity_table(fold="2026-05", sites=[*real.SIGNED_SIX, *real.MVA_ONLY])
    return f"capacities read: {len(caps)}, all finite: {all(c.mw >= 0.0 for c in caps)}"


def check_real_modules_import() -> str:
    """Return whether the real loader and the driver import, and the probe provider names a missing file."""
    import e00721_driver  # noqa: F401
    import e00721_real as real

    provider = real.ProbeProvider(
        root=Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / "nonexistent-721"
    )
    try:
        provider.capacity_mw(world="W2", fold="2026-05")
    except real.ProbeFileMissingError as exc:
        return f"modules import: True, a missing probe file raises with its name: {'W2_2026-05.json' in str(exc)}"
    return "modules import: True, a missing probe file raises: False"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("dense raw array", check_dense_raw),
    ("grid and sites", check_grid_and_sites),
    ("one site's power and CAMS", check_one_site_power_and_cams),
    ("farm frame", check_farm_frame),
    ("stored 713 forecasts", check_stored_713_forecasts),
    ("stage 0 curves", check_stage0_curves),
    ("capacity probe folder", check_capacity_probe_files),
    ("as-of capacity tables", check_as_of_capacity_tables),
    ("real modules", check_real_modules_import),
)


def main() -> int:
    """Run every check, print one line each, and return the number of failures."""
    failures = 0
    for name, check in CHECKS:
        try:
            print(f"ok   {name}: {check()}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
