# ruff: noqa: D103
"""Experiment 669: score the two Parquet files against observed power, outside the files (CPU).

Scores are in points of effective capacity (percent). CRPS is twice the mean pinball loss over the
11 quantile levels. Observed power is the cleaned power (flagged half-hours are NaN and not scored).
Every mean has a day-block bootstrap interval over origin days.

Usage: python e00669_score.py OUT_DIR
"""

import sys
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00611_pool as pool  # ty: ignore[unresolved-import]
import e00633_common as k633
import e00669_common as k
import numpy as np
import polars as pl

QCOLS = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
TAUS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
FILES = {"BSP": "bsps.parquet", "GSP": "gsps.parquet"}
N_BOOT = 1000


def observed(*, sites: tuple[str, ...]) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    k633.redirect()
    series, arrays = fd._series_and_arrays()
    table = series.filter(pl.col("site").is_in(list(sites)))
    clean = pool.prep422.stale.cleaned_power(arrays=arrays, series=table)
    return table, clean


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table, clean = observed(sites=k.NODE_SITES)
    cap = {
        int(r["time_series_id"]): (r["site"], float(r["effective_capacity_mw"]))
        for r in table.iter_rows(named=True)
    }
    rows = []
    for group, name in FILES.items():
        f = pl.read_parquet(k.SHARED / name)
        for tid in f["time_series_id"].unique().to_list():
            site, capacity = cap[int(tid)]
            s = f.filter(pl.col("time_series_id") == tid)
            rows.append((group, site, capacity, s))
    parts = []
    for group, site, capacity, s in rows:
        idx = np.asarray(common.grid_index(times=s["valid_time"]))
        y = common.take(values=clean[site], index=idx).astype(np.float64)
        q = s.select(QCOLS).to_numpy() / capacity
        lead_h = (s["valid_time"] - s["power_fcst_init_time"]).dt.total_minutes().to_numpy() / 60.0
        d = y[:, None] - q
        pin = np.maximum(TAUS * d, (TAUS - 1.0) * d)
        parts.append(
            pl.DataFrame(
                {
                    "group": group,
                    "site": site,
                    "origin": s["power_fcst_init_time"],
                    "valid": s["valid_time"],
                    "lead_h": lead_h,
                    "y": y,
                    "crps": 200.0 * pin.mean(axis=1),
                    "abs_err_median": 100.0 * np.abs(y - q[:, 5]),
                    "in90": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(float),
                    "in98": ((y >= q[:, 0]) & (y <= q[:, 10])).astype(float),
                    "pin01": 100.0 * pin[:, 0],
                    "pin05": 100.0 * pin[:, 1],
                    "pin95": 100.0 * pin[:, 9],
                    "pin99": 100.0 * pin[:, 10],
                    "below_q05": (y < q[:, 1]).astype(float),
                    "above_q95": (y > q[:, 9]).astype(float),
                    "below_q01": (y < q[:, 0]).astype(float),
                    "above_q99": (y > q[:, 10]).astype(float),
                }
            )
        )
    allrows = pl.concat(parts)
    n_total = allrows.height
    allrows = allrows.filter(pl.col("y").is_finite())
    pl.DataFrame(
        {
            "what": ["rows in files", "scored rows (observed, unflagged)"],
            "rows": [n_total, allrows.height],
        }
    ).write_csv(out / "row_counts.csv")
    allrows = allrows.with_columns(
        month=pl.col("origin").dt.strftime("%Y-%m"),
        day=pl.col("origin").dt.truncate("1d"),
        band=pl.when(pl.col("lead_h") < 24)
        .then(pl.lit("days 0 to 1"))
        .when(pl.col("lead_h") < 72)
        .then(pl.lit("days 1 to 3"))
        .when(pl.col("lead_h") < 168)
        .then(pl.lit("days 3 to 7"))
        .otherwise(pl.lit("days 7 to 14")),
        lead_day=(pl.col("lead_h") // 24).cast(pl.Int32),
    )
    metrics = [
        "crps",
        "abs_err_median",
        "in90",
        "in98",
        "pin01",
        "pin05",
        "pin95",
        "pin99",
        "below_q01",
        "below_q05",
        "above_q95",
        "above_q99",
    ]
    rng = np.random.default_rng(0)

    def summarise(by: list[str], name: str) -> None:
        agg = (
            allrows.group_by([*by, "day"])
            .agg(*[pl.col(m).sum().alias(f"{m}_sum") for m in metrics], pl.len().alias("n"))
            .sort([*by, "day"])
        )
        out_rows = []
        for key, g in agg.group_by(by, maintain_order=True):
            n = g["n"].to_numpy().astype(float)
            rec = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
            rec["rows"] = int(n.sum())
            draw = rng.integers(0, len(n), size=(N_BOOT, len(n)))
            for m in metrics:
                s = g[f"{m}_sum"].to_numpy()
                rec[m] = float(s.sum() / n.sum())
                if m in ("crps", "abs_err_median"):
                    boot = s[draw].sum(axis=1) / n[draw].sum(axis=1)
                    rec[f"{m}_lo"], rec[f"{m}_hi"] = (
                        float(v) for v in np.quantile(boot, [0.025, 0.975])
                    )
            out_rows.append(rec)
        pl.DataFrame(out_rows).write_csv(out / f"{name}.csv")

    summarise(["band"], "by_band")
    summarise(["group", "band"], "by_group_band")
    summarise(["site", "band"], "by_site_band")
    summarise(["month"], "by_month")
    summarise(["month", "band"], "by_month_band")
    summarise(["lead_day"], "by_lead_day")
    summarise(["group", "lead_day"], "by_group_lead_day")
    summarise(["group", "month"], "by_group_month")
    summarise(["group"], "by_group")
    allrows.group_by("month").agg(pl.len().alias("scored_rows")).sort("month").write_csv(
        out / "scored_rows_by_month.csv"
    )
    print("scored", allrows.height, "of", n_total)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
