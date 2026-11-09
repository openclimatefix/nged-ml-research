# ruff: noqa: E501, D103
"""Experiment 669: compare 633's stored 24-hour-lag forecasts with the 3.5-hour-lag forecasts of this experiment.

Both are scored on the same rows (July to October 2026 origins, valid times 12:00 to 72 hours later), CPU.


Experiment 633's forecasts are shares of capacity at its 09:00 origin (leads to 240 h); this
experiment's file is in MW at the relabelled 12:00 origin. Rows are matched on series, origin day, and valid time.

Usage: python e00669_compare633.py OUT_CSV
"""

import sys
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00611_pool as pool  # ty: ignore[unresolved-import]
import e00633_common as k633
import e00669_common as k
import e00669_score as sc
import numpy as np
import polars as pl


def main(out: Path) -> None:
    table, clean = sc.observed(sites=k.NODE_SITES)
    cap = {
        int(r["time_series_id"]): (r["site"], float(r["effective_capacity_mw"]))
        for r in table.iter_rows(named=True)
    }
    rows = []
    for group, fname in sc.FILES.items():
        mine = pl.read_parquet(k.SHARED / fname)
        seeds = [
            pl.read_parquet(
                k633.ROOT
                / "work"
                / group.lower()
                / "seed_preds"
                / "J_LG_B1"
                / f"2026-07_s{s}.parquet"
            )
            for s in (0, 1)
        ]
        old = (
            pl.concat(seeds)
            .group_by("time_series_id", "power_fcst_init_time", "valid_time")
            .agg(*[pl.col(c).mean() for c in sc.QCOLS])
        )
        old = old.with_columns(day=pl.col("power_fcst_init_time").dt.truncate("1d")).with_columns(
            time_series_id=pl.col("time_series_id").cast(pl.Int8)
        )
        mine = mine.with_columns(day=pl.col("power_fcst_init_time").dt.truncate("1d"))
        j = mine.join(
            old, on=["time_series_id", "day", "valid_time"], suffix="_old", validate="1:1"
        )
        j = j.filter(pl.col("day") >= pl.datetime(2026, 7, 1, time_zone="UTC"))
        for tid in j["time_series_id"].unique().to_list():
            site, capacity = cap[int(tid)]
            s = j.filter(pl.col("time_series_id") == tid)
            idx = np.asarray(common.grid_index(times=s["valid_time"]))
            y = common.take(values=clean[site], index=idx).astype(np.float64)
            lead = (
                s["valid_time"] - s["power_fcst_init_time"]
            ).dt.total_minutes().to_numpy() / 60.0
            q_new = s.select(sc.QCOLS).to_numpy() / capacity
            q_old = np.sort(s.select([f"{c}_old" for c in sc.QCOLS]).to_numpy(), axis=1)
            for name, q in (
                ("3.5-hour lags (this experiment)", q_new),
                ("24-hour lags (experiment 633)", q_old),
            ):
                d = y[:, None] - q
                pin = np.maximum(sc.TAUS * d, (sc.TAUS - 1.0) * d)
                for band, (lo, hi) in {
                    "hours 0 to 72": (0, 72),
                    "hours 72 to 240": (72, 240),
                }.items():
                    m = np.isfinite(y) & (lead >= lo) & (lead < hi)
                    rows.append(
                        {
                            "group": group,
                            "site": site,
                            "network": name,
                            "band": band,
                            "rows": int(m.sum()),
                            "crps_sum": float(200 * pin[m].mean(axis=1).sum()),
                            "abs_sum": float(100 * np.abs(y - q[:, 5])[m].sum()),
                        }
                    )
    df = pl.DataFrame(rows)
    pooled = (
        df.group_by("network", "band")
        .agg(pl.col("rows").sum(), pl.col("crps_sum").sum(), pl.col("abs_sum").sum())
        .with_columns(
            crps=pl.col("crps_sum") / pl.col("rows"), mae=pl.col("abs_sum") / pl.col("rows")
        )
        .sort("band", "network")
    )
    bysite = df.with_columns(crps=pl.col("crps_sum") / pl.col("rows")).sort(
        "band", "site", "network"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled.write_csv(out)
    bysite.write_csv(out.with_name(out.stem + "_by_site.csv"))
    print(pooled.select("network", "band", "rows", "crps", "mae"))
    _ = pool


if __name__ == "__main__":
    main(Path(sys.argv[1]))
