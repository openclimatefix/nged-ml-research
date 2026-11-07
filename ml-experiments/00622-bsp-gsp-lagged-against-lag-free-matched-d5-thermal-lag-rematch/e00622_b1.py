# ruff: noqa: E402, D103
"""Experiment 622: experiment 437's thermal-lag and solar-gain tables for the 2 BSPs and 2 GSPs.

Experiment 437 stored these tables for the 16 Primaries only. `site_table` is the same function, run
on the 4 series' ensemble-mean summaries (experiment 415's `ens_summary_nodes`) and their CAMS
irradiance, and the tables are written to this experiment's folder. The causality check of
experiment 437 (a column never depends on a later run) is run on D13.

Usage: python e00622_b1.py {build|check}
"""

import sys
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00415_prep as p415  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch"
ROOT: Final = ML_EXPERIMENTS_DATA_DIR / NAME
TRAILING: Final = ROOT / "trailing"
SITES: Final[tuple[str, ...]] = ("D13", "D17", "D1", "D6")


def install() -> None:
    """Point experiment 437's table reader at this experiment's table folder."""
    f437.TRAILING_DIR = TRAILING


def install_for_build() -> None:
    """Also point the builder at the nodes' ensemble summaries and the 4 series."""
    install()
    prep.SUMMARY_DIR = p415.NODE_SUMMARY_DIR
    series = common.series_table().filter(pl.col("site").is_in(SITES))
    prep.primaries = lambda: series.with_row_index("primary")


if __name__ == "__main__":
    install_for_build()
    if sys.argv[1] == "build":
        f437.build_all()
    else:
        f437.check(site="D13")
