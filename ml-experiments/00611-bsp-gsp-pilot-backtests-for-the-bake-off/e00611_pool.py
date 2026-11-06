"""Experiment 611: point experiment 428's harness at one group of the BSP/GSP series.

`install(group)` replaces `e00397_prep.primaries()` with a table of the group's 2 series (index 0
and 1), points the summary and pretraining raw-matrix folders at this experiment's folders
(symlinks to experiment 564's files, which hold the 4 series), and makes experiment 422's
`cleaned_arrays` cover the group. The flagged half-hours are cleaned as for the Primaries; the
short-history cut of D15 and D18 does not apply.
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00611-bsp-gsp-pilot-backtests-for-the-bake-off"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
W564: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced" / "work"
GROUPS: Final[dict[str, tuple[str, ...]]] = {"bsp": ("D13", "D17"), "gsp": ("D1", "D6")}
"""Each group's sites, in series-index order. Series ids 2 and 1 (BSPs), 28 and 27 (GSPs)."""

_table: pl.DataFrame | None = None
GROUP: str = "bsp"


def group_table() -> pl.DataFrame:
    """Return the group's 2 series with a `primary` index of 0 and 1."""
    global _table  # noqa: PLW0603
    if _table is None:
        series = common.series_table()
        rows = pl.concat([series.filter(pl.col("site") == s) for s in GROUPS[GROUP]])
        assert rows["site"].to_list() == list(GROUPS[GROUP])
        assert set(rows["substation_type"].to_list()) == {GROUP.upper()}
        _table = rows.with_row_index("primary")
    return _table


def cleaned_arrays() -> dict[str, np.ndarray]:
    """Return each series' power over capacity, NaN at flagged half-hours."""
    series, arrays = fd._series_and_arrays()
    table = group_table()
    return prep422.stale.cleaned_power(
        arrays=arrays, series=series.filter(pl.col("site").is_in(table["site"].to_list()))
    )


def _link(*, source: Path, target: Path) -> None:
    """Make `target` a symlink to `source` if it does not exist."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        assert source.exists(), source
        target.symlink_to(source)


def install(*, group: str) -> Path:
    """Switch the harness to the group's 2 series; return the group's work folder."""
    global GROUP, _table
    GROUP, _table = group, None
    work = WORK / "work" / group
    for site in GROUPS[group]:
        _link(
            source=W564 / "raw" / "summary" / f"{site}.npz",
            target=work / "raw" / "summary" / f"{site}.npz",
        )
        _link(
            source=W564 / "raw_pretrain" / f"{site}.npz",
            target=work / "raw_pretrain" / f"{site}.npz",
        )
    prep.primaries = group_table
    prep.RAW_DIR = work / "raw"
    prep418.RAW_DIR = work / "raw_pretrain"
    prep422.cleaned_arrays = cleaned_arrays
    return work
