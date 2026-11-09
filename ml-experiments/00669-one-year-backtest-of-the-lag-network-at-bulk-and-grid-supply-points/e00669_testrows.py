# ruff: noqa: D103
"""Experiment 669: test matrices that keep every half-hour, observed or not.

Experiment 633's `raw/summary/*.npz` keep only half-hours with an observed, unflagged reading,
because they are training matrices. A forecast file needs a row for every half-hour, so this
script rebuilds the rows of the 365 origins with the code of experiment 633's `build_site`
(`e00633_nodes.py`), without the finite-power filter. The target column `y` is NaN where no reading
exists and is never read at forecast time.

Usage: python e00669_testrows.py
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00633_common as k633  # ty: ignore[unresolved-import]
import e00669_common as k
import numpy as np
import polars as pl


def build_site(*, site: str) -> dict[str, np.ndarray]:
    k633.redirect()
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    summary = (
        pl.scan_parquet(k633.ROOT / "inputs" / "ens_summary_nodes" / "init=*.parquet")
        .filter(pl.col("time_series_id") == row["time_series_id"])
        .filter(
            (pl.col("power_fcst_init_time") >= k.FIRST_ORIGIN_DAY.replace(hour=9))
            & (pl.col("power_fcst_init_time") <= k.origin_days()[-1].replace(hour=9))
        )
        .collect()
    )
    assert summary["power_fcst_init_time"].n_unique() == k.N_ORIGINS, summary[
        "power_fcst_init_time"
    ].n_unique()
    idx = np.asarray(common.grid_index(times=summary["valid_time"]))
    power = common.take(values=arrays[site].power, index=idx) * row["effective_capacity_mw"]
    rows = summary.with_columns(power=pl.Series(power).cast(pl.Float32))
    rows = rows.filter(pl.col("valid_time") > pl.col("power_fcst_init_time"))
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    rows = fd.add_lags(frame=rows, power=arrays[site].power, a7=False)
    rows = rows.with_columns(capacity=pl.lit(float(row["effective_capacity_mw"])))
    x = prep.feature_matrix(frame=rows, scheme="summary")
    return {
        "x": x,
        "y": (rows["power"] / rows["capacity"]).to_numpy().astype(np.float32),
        "valid": rows["valid_time"].dt.epoch("us").to_numpy(),
        "origin": rows["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        "lead": rows["lead_hours"].to_numpy().astype(np.float32),
    }


def main() -> None:
    out = k.ROOT / "inputs" / "raw_test" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    for site in k.NODE_SITES:
        target = out / f"{site}.npz"
        if target.exists():
            continue
        d = build_site(site=site)
        lead = (d["valid"] - d["origin"]) / 3.6e9
        print(
            site,
            len(d["origin"]),
            "rows; max lead",
            float(lead.max()),
            "h; rows without target",
            int(np.isnan(d["y"]).sum()),
            flush=True,
        )
        np.savez(target.with_name(target.name + ".tmp.npz"), **d)  # ty: ignore[invalid-argument-type]
        target.with_name(target.name + ".tmp.npz").rename(target)


if __name__ == "__main__":
    main()
