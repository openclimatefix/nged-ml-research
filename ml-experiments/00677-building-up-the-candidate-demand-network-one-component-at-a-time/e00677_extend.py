# ruff: noqa: E501, D103
"""Experiment 667 (copy of 620's): rebuild the raw ensemble-summary matrices to 2026-10-31.

Experiment 397's `write_raw` builds `work/raw/summary/<site>.npz` (feature matrix `x`, power over
capacity `y`, `valid`, `origin`, and `lead`) from the control member's rows joined with the
ensemble summary. The control member's frames stop at the run of 2026-06-29, so this script takes
the keys and weather from the summary tables of experiment 607 (944 runs, 2024-04-01 to
2026-10-31) and everything else (solar angles, calendar, and capped lags) from the extended power
grid. It writes to this experiment's data folder; no earlier file is edited. `check` compares the
rebuilt rows with experiment 397's stored rows wherever both exist.

Usage: python e00677_extend.py {build|check} [SITE ...]
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_ROOT: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00667-*"), None) or (
    ML_EXPERIMENTS_DATA_DIR
    / "00667-one-year-backtest-of-the-two-output-demand-network-at-primaries"
)
RAW_DIR: Final[Path] = OUT_ROOT / "inputs" / "raw"
INPUTS_620: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00620-*")) / "inputs"
"""Experiment 620's extended weather-column tables (trailing437, w560, frames), read only."""
E607: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
GRID_END: Final[datetime] = datetime(2026, 11, 20, 23, 30, tzinfo=UTC)
"""Last target of the backtest: the origin of 2026-10-31 09:00 plus 351 hours is 2026-11-14 12:00; rounded up."""
REPLACE_FROM: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""Stored rows from the first daily run on are rebuilt: they lack every half-hour with no cleaned power."""


def redirect() -> None:
    """Point experiment 87's loaders at the extended inputs of experiment 607 and a longer grid."""
    skip = os.environ.get("E00677_SKIP", "").split(
        ","
    )  # bisection switches for the dev-month check
    if "grid" not in skip:
        common.GRID_END = GRID_END
    if "era5" not in skip:
        common.ERA5_PATH = E607 / "era5_features.parquet"
    if "obs" not in skip:
        common.OBSERVED_PATH = E607 / "observed.parquet"
    if "summary" not in skip:
        prep.SUMMARY_DIR = E607 / "ens_summary"


def build_site(*, site: str) -> dict[str, np.ndarray]:
    import e00087_fit_demand as fd  # ty: ignore[unresolved-import]

    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    summary = (
        pl.scan_parquet(prep.SUMMARY_DIR / "init=*.parquet")
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .collect()
    )
    idx = np.asarray(common.grid_index(times=summary["valid_time"]))
    power = common.take(values=arrays[site].power, index=idx) * row["effective_capacity_mw"]
    rows = summary.with_columns(power=pl.Series(power).cast(pl.Float32))
    # keep rows with no observed power (y is NaN): the backtest files need every valid time
    rows = rows.filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    rows = fd.add_lags(frame=rows, power=arrays[site].power, a7=False)
    rows = rows.with_columns(capacity=pl.lit(float(row["effective_capacity_mw"])))
    x = prep.feature_matrix(frame=rows, scheme="summary")
    new = {
        "x": x,
        "y": (rows["power"] / rows["capacity"]).to_numpy().astype(np.float32),
        "valid": rows["valid_time"].dt.epoch("us").to_numpy(),
        "origin": rows["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        "lead": rows["lead_hours"].to_numpy().astype(np.float32),
    }
    # Experiment 397's stored rows before 2025-07-01 are kept as they are, because the fold's
    # training rows (and so its standardiser) must not change: before 2025-07-01 they hold every
    # fifth day's run only, while the summary tables hold every day. Only the runs after the stored
    # rows are added.
    old = np.load(prep.RAW_DIR / "summary" / f"{site}.npz")
    cut = int(REPLACE_FROM.timestamp() * 1_000_000) - 1
    before_old, after = old["origin"] <= cut, new["origin"] > cut
    return {k: np.concatenate([old[k][before_old], new[k][after]]) for k in new}


def main() -> None:
    command, sites = sys.argv[1], sys.argv[2:]
    redirect()
    for site in sites or prep.primaries()["site"].to_list():
        mine = RAW_DIR / "summary" / f"{site}.npz"
        if command == "build":
            mine.parent.mkdir(parents=True, exist_ok=True)
            np.savez(mine.with_name(mine.name + ".tmp.npz"), **build_site(site=site))  # ty: ignore[invalid-argument-type]
            mine.with_name(mine.name + ".tmp.npz").rename(mine)
            print("built", site, flush=True)
        else:
            old = np.load(prep.RAW_DIR / "summary" / f"{site}.npz")
            new = np.load(mine)
            key_old = (
                old["origin"].astype(np.int64) * (1 << 22)
                + (old["valid"] - old["origin"]) // 1_800_000_000
            )
            key_new = (
                new["origin"].astype(np.int64) * (1 << 22)
                + (new["valid"] - new["origin"]) // 1_800_000_000
            )
            order = np.argsort(key_old)
            pos = np.clip(np.searchsorted(key_old[order], key_new), 0, len(key_old) - 1)
            hit = key_old[order][pos] == key_new
            sel, j = np.nonzero(hit)[0], order[pos[hit]]
            worst = {
                k: float(
                    np.nanmax(np.abs(new[k][sel].astype(np.float64) - old[k][j].astype(np.float64)))
                )
                for k in ("x", "y", "lead")
            }
            old_keys_after = key_old[old["origin"] >= int(REPLACE_FROM.timestamp() * 1e6)]
            missing = int((~np.isin(old_keys_after, key_new)).sum())
            print(site, "stored rows after the cut missing from the rebuild:", missing, flush=True)
            print(site, "rows with no observed power (y NaN):", int(np.isnan(new["y"]).sum()))
            print(
                site,
                "old rows",
                len(old["origin"]),
                "matched",
                len(sel),
                "of new",
                len(new["origin"]),
                "max abs diff",
                worst,
                flush=True,
            )


if __name__ == "__main__":
    main()
