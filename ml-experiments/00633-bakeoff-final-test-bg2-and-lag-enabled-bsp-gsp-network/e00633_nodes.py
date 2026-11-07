# ruff: noqa: E501, D103, PT018, D205
"""Experiment 633, Part 2: extend the bulk and grid supply points' weather inputs to the final-test
window (CPU, weather only).

- `summaries`: the ensemble mean and standard deviation of each node, for the 123 runs from 2026-07-01
  to 2026-10-31, from the weather frames of `e00633_frames.py`. The folder `inputs/ens_summary_nodes`
  also links experiment 415's files for the earlier runs, so one folder holds every run.
- `check`: rebuilds the run of 2026-06-30 and compares it with experiment 415's stored file.
- `raw`: the raw input matrix of each node, with the rows of runs after 2026-06-29 appended to experiment
  564's stored matrix (a copy of `build_site` in experiment 620's `e00620_extend.py`).

No power from the window is read by `summaries` or `check`. `raw` reads the extended power grid to
build the lag
columns and the target of the window rows, as experiment 620's version did; the window rows' target
(the held-out
power) is never read by training, because `e00633_fold.py` hides the cleaned power from the cutoff
on in the
training phase and returns no test rows.

Usage: python e00633_nodes.py {summaries|check|raw}
"""

import sys
from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00415_prep as p415  # ty: ignore[unresolved-import]
import e00633_common as k
import e00633_frames as fr
import numpy as np
import polars as pl

NODES = p415.NODES
OUT = k.ROOT / "inputs" / "ens_summary_nodes"
RAW = k.ROOT / "inputs" / "raw" / "summary"
OLD_RAW = next(k.ML_EXPERIMENTS_DATA_DIR.glob("00564-*")) / "work" / "raw" / "summary"


def node_ids() -> list[int]:
    series = common.series_table()
    return series.filter(pl.col("site").is_in(NODES))["time_series_id"].to_list()


def summarise_run(*, frame: pl.DataFrame) -> pl.DataFrame:
    table = prep.summarise(frame=frame.filter(pl.col("time_series_id").is_in(node_ids()))).sort(
        "time_series_id", "valid_time"
    )
    return table.cast(
        dict.fromkeys([f"{a}_{w}" for a in ("mean", "std") for w in prep.WEATHER], pl.Float32)
    )


def summaries() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in sorted(p415.NODE_SUMMARY_DIR.glob("init=*.parquet")):
        link = OUT / p.name
        if not link.exists():
            link.symlink_to(p)
    for run in k.runs():
        target = OUT / f"init={run:%Y-%m-%d}.parquet"
        if target.exists():
            continue
        frame = pl.read_parquet(k.FRAMES / f"init={run:%Y-%m-%d}.parquet")
        common.atomic_write(frame=summarise_run(frame=frame), target=target)
    print("node summaries:", len(list(OUT.glob("init=*.parquet"))), "files")


def check() -> None:
    run = datetime(2026, 6, 30, tzinfo=UTC)
    mine = summarise_run(frame=fr.build(run=run))
    stored = pl.read_parquet(p415.NODE_SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet")
    j = mine.join(stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_s")
    assert j.height == stored.height == mine.height, (j.height, stored.height, mine.height)
    worst = max(
        float((j[c].cast(pl.Float64) - j[f"{c}_s"].cast(pl.Float64)).abs().max())  # ty: ignore[invalid-argument-type]
        for c in mine.columns
        if c.startswith(("mean_", "std_"))
    )  # ty: ignore[invalid-argument-type]
    print("max abs difference from experiment 415's file of 2026-06-30:", worst)
    assert worst == 0.0


CHECK_OVERLAP = False


def build_site(*, site: str) -> dict[str, np.ndarray]:
    """Return the node's raw matrix: experiment 564's stored rows plus the rows of the runs after them.

    A copy of `build_site` in experiment 620's `e00620_extend.py`, which builds a Primary's rows from the
    ensemble summaries, the solar angles, the calendar, and the lags of the extended power grid.
    """
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
    rows = rows.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
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
    old = np.load(OLD_RAW / f"{site}.npz")
    if CHECK_OVERLAP:
        return {"new": new, "old": {key: old[key] for key in new}}  # ty: ignore[invalid-return-type]
    after = new["origin"] > old["origin"].max()
    return {key: np.concatenate([old[key], new[key][after]]) for key in new}


def raw() -> None:
    k.redirect()
    prep.SUMMARY_DIR = OUT
    RAW.mkdir(parents=True, exist_ok=True)
    for site in NODES:
        target = RAW / f"{site}.npz"
        if target.exists():
            continue
        data = build_site(site=site)
        np.savez(target.with_name(target.name + ".tmp.npz"), **data)  # ty: ignore[invalid-argument-type]
        target.with_name(target.name + ".tmp.npz").rename(target)
        print("built", site, len(data["y"]), flush=True)


def check_raw() -> None:
    """The extension must change nothing for the past: every stored row is unchanged, and the rows
    both files hold agree.
    """
    for site in NODES:
        old, new = np.load(OLD_RAW / f"{site}.npz"), np.load(RAW / f"{site}.npz")
        n = len(old["y"])
        assert np.array_equal(new["x"][:n], old["x"], equal_nan=True) and np.array_equal(
            new["y"][:n], old["y"], equal_nan=True
        ), site
        print(
            site,
            "stored rows",
            n,
            "total",
            len(new["y"]),
            "last origin",
            np.datetime64(int(new["origin"].max()), "us"),
        )


def row_key(d: dict) -> np.ndarray:
    return d["origin"].astype(np.int64) * (1 << 22) + (d["valid"] - d["origin"]) // 1_800_000_000


def check_overlap() -> None:
    """Rebuild the rows the stored matrix already holds, with the extension's code, and compare them
    key by key.
    """
    global CHECK_OVERLAP  # noqa: PLW0603
    CHECK_OVERLAP = True
    k.redirect()
    prep.SUMMARY_DIR = OUT
    for site in NODES:
        both = build_site(site=site)
        new, old = both["new"], both["old"]  # ty: ignore[invalid-argument-type]
        ko, kn = row_key(old), row_key(new)  # ty: ignore[invalid-argument-type]
        order = np.argsort(ko)
        pos = np.clip(np.searchsorted(ko[order], kn), 0, len(ko) - 1)
        hit = ko[order][pos] == kn
        sel, j = np.nonzero(hit)[0], order[pos[hit]]
        worst = {
            c: float(
                np.nanmax(np.abs(new[c][sel].astype(np.float64) - old[c][j].astype(np.float64)))
            )
            for c in ("x", "y", "lead")
        }
        print(
            site,
            "stored rows",
            len(old["y"]),
            "rebuilt rows",
            len(new["y"]),
            "matched",
            len(sel),
            "max abs diff",
            worst,
            flush=True,
        )


if __name__ == "__main__":
    {
        "summaries": summaries,
        "check": check,
        "raw": raw,
        "check_raw": check_raw,
        "check_overlap": check_overlap,
    }[sys.argv[1]]()
