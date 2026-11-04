# ruff: noqa: E501
"""Extend experiment 428's 16-Primary harness to 20 series: the Primaries, 2 BSPs, and 2 GSPs.

`install()` replaces `e00397_prep.primaries()` with a table of 20 series (the 16 Primaries keep
indices 0 to 15, as in experiment 545; D1, D6, D13, and D17 take indices 16 to 19), points the
summary and pretraining raw-matrix folders at this experiment's folders (symlinks to the existing
files, plus the extra series' files), and makes experiment 422's `cleaned_arrays` cover all 20.

The Primaries' rows are exactly M0's. The four extra series' ensemble-summary rows are experiment
415's `raw/<site>.npz` (the same `summary` scheme, weather summarised from the run issued at the
origin), and their ERA5 and CAMS pretraining rows are built by experiment 418's `write_raw` over the
patched table.
"""

import os
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00415_prep as p415  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced" / "work"
)
RAW: Final[Path] = WORK / "raw"
RAW_PRETRAIN: Final[Path] = WORK / "raw_pretrain"
N_PRIMARIES: Final[int] = 16
EXTRA_SITES: Final[tuple[str, ...]] = ("D1", "D6", "D13", "D17")
"""GSP, GSP, BSP, BSP, in label order. Series ids 28, 27, 2, and 1."""

_primaries = prep.primaries
_table: pl.DataFrame | None = None


def pooled_table() -> pl.DataFrame:
    """Return the 20 series: the Primaries (indices 0 to 15), then the 4 extra series (16 to 19)."""
    global _table  # noqa: PLW0603
    if _table is None:
        series = common.series_table()
        primaries = series.filter(pl.col("substation_type") == "Primary")
        extra = series.filter(pl.col("site").is_in(EXTRA_SITES)).sort(
            pl.col("site").str.slice(1).cast(pl.Int32)
        )
        assert primaries.height == N_PRIMARIES, primaries.height
        assert extra["site"].to_list() == list(EXTRA_SITES), extra["site"].to_list()
        assert sorted(extra["substation_type"].unique().to_list()) == ["BSP", "GSP"]
        _table = pl.concat([primaries, extra]).with_row_index("primary")
    return _table


def kinds() -> list[str]:
    """Return each series' type (Primary, BSP, or GSP) in series-index order."""
    return pooled_table()["substation_type"].to_list()


def cleaned_arrays() -> dict[str, np.ndarray]:
    """Return each series' power over capacity, NaN at flagged half-hours and (D15, D18) before the cut."""
    series, arrays = fd._series_and_arrays()
    table = pooled_table()
    clean = prep422.stale.cleaned_power(
        arrays=arrays, series=series.filter(pl.col("site").is_in(table["site"].to_list()))
    )
    cut = int(common.grid_index(times=prep422.stale.SHORT_HISTORY_CUT))
    for label in prep422.stale.SHORT_HISTORY_LABELS:
        clean[label][:cut] = np.nan
    return clean


def _link(*, source: Path, target: Path) -> None:
    """Make `target` a symlink to `source` if it does not exist."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        assert source.exists(), source
        os.symlink(source, target)


def prepare_folders() -> None:
    """Symlink the Primaries' summary and pretraining matrices, and the extra series' summary matrices."""
    for row in pooled_table().iter_rows(named=True):
        site = row["site"]
        if row["primary"] < N_PRIMARIES:
            _link(
                source=_original_raw_dir() / "summary" / f"{site}.npz",
                target=RAW / "summary" / f"{site}.npz",
            )
            _link(
                source=_original_pretrain_dir() / f"{site}.npz",
                target=RAW_PRETRAIN / f"{site}.npz",
            )
        else:
            _link(source=p415.RAW_DIR / f"{site}.npz", target=RAW / "summary" / f"{site}.npz")


_RAW0, _PRE0 = prep.RAW_DIR, prep418.RAW_DIR


def _original_raw_dir() -> Path:
    """Return experiment 397's raw folder (before `install` repoints `prep.RAW_DIR`)."""
    return _RAW0


def _original_pretrain_dir() -> Path:
    """Return experiment 418's pretraining folder (before `install` repoints it)."""
    return _PRE0


def build_extra_pretraining() -> None:
    """Write the extra series' ERA5 and CAMS pretraining matrices (experiment 418's recipe)."""
    prep418.RAW_DIR = RAW_PRETRAIN
    for site in EXTRA_SITES:
        print(prep418.write_raw(site=site), flush=True)


def install() -> None:
    """Switch the harness to the 20 series. Call once, before any row is built."""
    prepare_folders()
    prep.primaries = pooled_table
    prep.RAW_DIR = RAW
    prep418.RAW_DIR = RAW_PRETRAIN
    prep422.cleaned_arrays = cleaned_arrays
