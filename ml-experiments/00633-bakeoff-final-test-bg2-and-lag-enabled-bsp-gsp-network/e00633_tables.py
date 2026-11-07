# ruff: noqa: D103, E501
"""Experiment 633, Part 2: extend experiment 622's thermal-lag and solar-gain tables of the 4 nodes to the final-test window (CPU, weather only).

Rebuilds each node's table with experiment 437's own `site_table` (through experiment 622's `e00622_b1.install_for_build`),
from the extended ensemble summaries of the nodes (`e00633_nodes.py summaries`) and the CAMS irradiance of
2019-09-01 to 2026-11-01 that experiment 620 extended (read only). No power is read. `check` proves the extension
changes nothing for the past: on every key that both the stored table and the rebuilt table hold, every column is bit-equal.

Usage: python e00633_tables.py {build|check|coverage}
"""

import sys
from datetime import UTC, datetime

import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00622_b1 as b1  # ty: ignore[unresolved-import]
import e00633_common as k
import numpy as np
import polars as pl

OUT = k.ROOT / "inputs" / "trailing437_nodes"
STORED = b1.TRAILING
CAMS = next(k.ML_EXPERIMENTS_DATA_DIR.glob("00620-*")) / "inputs" / "cams_demand_points_extended.parquet"


def setup() -> None:
    k.redirect()
    b1.install_for_build()
    b1.p415.NODE_SUMMARY_DIR = k.ROOT / "inputs" / "ens_summary_nodes"
    b1.prep.SUMMARY_DIR = b1.p415.NODE_SUMMARY_DIR
    f437.common.GRID_END = datetime(2026, 11, 30, 23, 30, tzinfo=UTC)  # the summaries' leads reach 351 h
    prep418.CAMS_PATH = CAMS
    f437.TRAILING_DIR = OUT


def build() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    local_date = f437.local_date_codes()
    daily = {"school": f437.school_columns(), "national": f437.national_daily()}
    for row in f437.prep.primaries().iter_rows(named=True):
        target = f437.table_path(site=row["site"])
        if target.exists():
            continue
        table = f437.site_table(row=row, local_date=local_date, daily=daily)
        bad = {c: int(table[c].null_count() + table[c].is_nan().sum()) for c in f437.BUNDLES["T"] + f437.BUNDLES["S"]}
        assert not any(bad.values()), (row["site"], bad)
        table.write_parquet(target)
        print(row["site"], table.height, "rows, last origin", table["origin_us"].max(), flush=True)


def check() -> None:
    setup()
    cols = [*f437.INPUT_COLUMNS, *f437.DIAGNOSTIC_COLUMNS]
    worst_all = 0.0
    for row in f437.prep.primaries().iter_rows(named=True):
        old = pl.read_parquet(STORED / f"{row['site']}.parquet").sort("origin_us", "valid_us")
        new = pl.read_parquet(OUT / f"{row['site']}.parquet")
        j = old.join(new, on=["origin_us", "valid_us"], suffix="_n")
        assert j.height == old.height, f"{row['site']}: {old.height - j.height} stored keys missing from the rebuilt table"
        worst = {}
        for c in cols:
            a, b = j[c].to_numpy().astype(np.float64), j[c + "_n"].to_numpy().astype(np.float64)
            worst[c] = 0.0 if ((a == b) | (np.isnan(a) & np.isnan(b))).all() else float("inf")
        w = max(worst.values())
        print(row["site"], "stored", old.height, "rebuilt", new.height, "new", new.height - old.height, "max diff", w, "last origin", new["origin_us"].max(), flush=True)
        worst_all = max(worst_all, w)
    print("MAX DIFF on stored keys:", worst_all)
    assert worst_all == 0.0, "a stored value or NaN mask differs"


def coverage() -> None:
    """Every (origin, valid) key the window's test rows need is in each rebuilt table (keys only)."""
    setup()
    first = int(datetime(2026, 7, 1, 9, tzinfo=UTC).timestamp() * 1_000_000)
    day, step = 86_400 * 1_000_000, 1_800_000_000
    for row in f437.prep.primaries().iter_rows(named=True):
        t = pl.read_parquet(OUT / f"{row['site']}.parquet", columns=["origin_us", "valid_us"])
        have = set(zip(t["origin_us"].to_list(), t["valid_us"].to_list(), strict=True))
        missing = sum((first + d * day, first + d * day + j * step) not in have for d in range(123) for j in range(1, 481))
        print(row["site"], "missing keys:", missing, flush=True)
        assert missing == 0


if __name__ == "__main__":
    {"build": build, "check": check, "coverage": coverage}[sys.argv[1]]()
