# ruff: noqa: E501, ANN201
"""Experiment 620: extend experiment 437's per-Primary weather-column tables to the final-test window (CPU, weather only).

Experiment 437's tables hold, for every (forecast run, valid time), the trailing temperature and
shortwave means, the clear-sky index and the CAMS previous-day irradiance (the B1 columns, bundles
T and S), plus calendar and national columns. They stop before the window. This script rebuilds each
table with experiment 437's own `site_table` from the extended ensemble summaries of experiment 607
(runs to 2026-10-31) and the CAMS irradiance of 2019-09-01 to 2026-11-01. No power is read.

`check` proves the extension changes nothing for the past: on every key that both the stored table
and the rebuilt table hold, every column must be bit-equal. The national columns (prices to
2026-10-12, demand to 2026-10-01, no data after those dates) are forward-filled by 437's code
and are not read by K1, K3 or the candidate; the B1 columns do not use them.

Usage: python e00620_tables.py build | check
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00437_features as f437  # ty: ignore[unresolved-import]
import e00620_extend as ext  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

R = ML_EXPERIMENTS_DATA_DIR / "00620-bakeoff-final-test-demand-networks-at-primaries"
OUT = R / "inputs" / "trailing437"
CAMS = ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "weather" / "CAMS" / "demand_points"
CAMS_EXTENDED = R / "inputs" / "cams_demand_points_extended.parquet"
STORED = f437.TRAILING_DIR


def setup() -> None:
    """Point experiment 437's code at the extended summaries, CAMS file, and output folder."""
    ext.redirect()
    f437.common.GRID_END = datetime(2026, 11, 30, 23, 30, tzinfo=UTC)  # the summaries' leads reach 351 h
    CAMS_EXTENDED.parent.mkdir(parents=True, exist_ok=True)
    if not CAMS_EXTENDED.exists():
        pl.concat([pl.read_parquet(p) for p in sorted(CAMS.glob("cams_demand_points*.parquet"))]).unique(["site", "time"]).sort("site", "time").write_parquet(CAMS_EXTENDED)
    prep418.CAMS_PATH = CAMS_EXTENDED
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
    """Compare every stored table with the rebuilt one on their common keys; report the new rows."""
    setup()
    cols = [*f437.INPUT_COLUMNS, *f437.DIAGNOSTIC_COLUMNS]
    worst_all = 0.0
    for row in f437.prep.primaries().iter_rows(named=True):
        old = pl.read_parquet(STORED / f"{row['site']}.parquet").sort("origin_us", "valid_us")
        new = pl.read_parquet(OUT / f"{row['site']}.parquet")
        j = old.join(new, on=["origin_us", "valid_us"], suffix="_n")
        assert j.height == old.height, f"{row['site']}: {old.height - j.height} stored keys missing from the rebuilt table"
        national = set(f437.BUNDLES["C"])
        worst = {}
        for c in cols:
            a, b = j[c].to_numpy().astype(np.float64), j[c + "_n"].to_numpy().astype(np.float64)
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            worst[c] = 0.0 if same.all() else float("inf")  # any value or NaN-mask difference
        t_s = max(v for k, v in worst.items() if k not in national)
        c_max = max((v for k, v in worst.items() if k in national), default=0.0)
        print(row["site"], "stored", old.height, "rebuilt", new.height, "new", new.height - old.height, "max diff T/S/diag", t_s, "C columns", c_max, "last origin", new["origin_us"].max(), flush=True)
        worst_all = max(worst_all, t_s, c_max)
    print("MAX DIFF on stored keys (non-calendar columns):", worst_all)
    assert worst_all == 0.0 and c_max == 0.0, "a stored value or NaN mask differs (C columns included)"


def coverage() -> None:
    """Check that every (origin, valid) key the window's test rows need is in each rebuilt table (keys only)."""
    from datetime import datetime  # noqa: PLC0415

    first = int(datetime(2026, 7, 1, 9, tzinfo=UTC).timestamp() * 1_000_000)
    day, step = 86_400 * 1_000_000, 1_800_000_000
    for row in f437.prep.primaries().iter_rows(named=True):
        t = pl.read_parquet(OUT / f"{row['site']}.parquet", columns=["origin_us", "valid_us"])
        have = set(zip(t["origin_us"].to_list(), t["valid_us"].to_list(), strict=True))
        missing = 0
        for d in range(123):
            origin = first + d * day
            missing += sum((origin, origin + k * step) not in have for k in range(1, 481))
        print(row["site"], "missing keys:", missing, flush=True)
        assert missing == 0


def causality() -> None:
    """Rebuild a Primary's table from a summary with every later run deleted; the rows of the cut run must not change."""
    from datetime import datetime  # noqa: PLC0415

    setup()
    row = next(r for r in f437.prep.primaries().iter_rows(named=True) if r["site"] == "D2")
    new = pl.read_parquet(OUT / "D2.parquet")
    full = f437.prep.load_summary(series_id=int(row["time_series_id"]))
    local_date = f437.local_date_codes()
    daily = {"school": f437.school_columns(), "national": f437.national_daily()}
    original = f437.prep.load_summary
    for cut in (datetime(2026, 7, 1, 9, tzinfo=UTC), datetime(2026, 8, 15, 9, tzinfo=UTC), datetime(2026, 10, 31, 9, tzinfo=UTC)):
        truncated = full.filter(pl.col("power_fcst_init_time") <= cut)
        f437.prep.load_summary = lambda *, series_id, t=truncated: t
        try:
            rebuilt = f437.site_table(row=row, local_date=local_date, daily=daily)
        finally:
            f437.prep.load_summary = original
        key = int(cut.timestamp() * 1_000_000)
        a = new.filter(pl.col("origin_us") == key).sort("valid_us")
        b = rebuilt.filter(pl.col("origin_us") == key).sort("valid_us")
        assert a.height == b.height > 0
        worst = max(float(np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy()))) for c in f437.INPUT_COLUMNS)
        print("run", cut, "largest difference with all later runs deleted:", worst, flush=True)
        assert worst <= 1e-5, "a column depends on a later run"


if __name__ == "__main__":
    {"build": build, "check": check, "causality": causality, "coverage": coverage}[sys.argv[1]]()
