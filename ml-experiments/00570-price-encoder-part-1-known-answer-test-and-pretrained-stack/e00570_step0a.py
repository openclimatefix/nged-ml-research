# ruff: noqa: ANN001, ANN201, D103, E501
"""Step 0(a) of experiment 570: how much of M0's residual variance can a price rule explain?

For each Primary, take the stored experiment 554 M0 (5-seed) median forecast on the five fold months,
residual = actual - q50 (metered basis). Remove the half-hour-of-day x weekend mean (calendar), then
regress on price features of the target half-hour from the market index and system price tables
(2025-09 onward): level, within-day rank, price minus day mean, system minus market index, net
imbalance volume. Report the share of calendar-residual variance explained (in-sample and by
leave-one-month-out), per Primary. Prices at the target are oracle values; this is a ceiling.
"""

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

M = ML_EXPERIMENTS_DATA_DIR.parent / "studies/downloads/market"
W554 = next(ML_EXPERIMENTS_DATA_DIR.glob("00554-*")) / "work/scores"
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")


def price_features() -> pl.DataFrame:
    mid = pl.read_parquet(M / "elexon_mid_apx/elexon_mid_apx.parquet").select(
        "time", mid=pl.col("price_gbp_per_mwh"), vol=pl.col("volume_mwh")
    )
    sp = pl.read_parquet(M / "elexon_system_prices/elexon_system_prices.parquet").select(
        "time", sys=pl.col("system_sell_price_gbp_per_mwh"), niv=pl.col("net_imbalance_volume_mwh")
    )
    df = mid.join(sp, on="time", how="full", coalesce=True).sort("time")
    df = df.with_columns(mid=pl.when(pl.col("vol") > 0).then(pl.col("mid")))  # mask zero-volume
    day = pl.col("time").dt.date()
    return df.with_columns(
        day=day,
        rank=pl.col("mid").rank().over(day) / pl.col("mid").count().over(day),
        dev=pl.col("mid") - pl.col("mid").mean().over(day),
        prem=pl.col("sys") - pl.col("mid"),
    ).drop("vol")


def main():
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    grid = common.grid_times()
    feats = price_features()
    out = []
    for month in MONTHS:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        q = pl.read_parquet(W554 / f"rows_{month}.parquet").select("primary", "origin", "target", "M0__q50")
        d = base.select("primary", "origin", "target", "actual", "valid_time", "lead_hours").join(
            q, on=["primary", "origin", "target"], how="inner"
        )
        d = d.join(feats, left_on="valid_time", right_on="time", how="left").with_columns(month=pl.lit(month))
        out.append(d)
    d = pl.concat(out).filter(pl.col("M0__q50").is_not_null() & pl.col("mid").is_not_null() & pl.col("actual").is_finite() & pl.col("M0__q50").is_finite())
    d = d.with_columns(
        res=pl.col("actual") - pl.col("M0__q50"),
        hh=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
        we=pl.col("valid_time").dt.weekday() >= 6,
    )
    d = d.with_columns(
        res_c=pl.col("res") - pl.col("res").mean().over("primary", "hh", "we")
    )
    cols = ["mid", "rank", "dev", "prem", "niv"]
    print("n rows", d.height, "primaries", d["primary"].n_unique())
    rows = []
    for p in sorted(d["primary"].unique().to_list()):
        s = d.filter(pl.col("primary") == p).drop_nulls(cols + ["res_c"]).drop_nans(cols + ["res_c"])
        y = s["res_c"].to_numpy()
        X = s.select(cols).to_numpy()
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X = np.c_[np.ones(len(X)), X, X[:, :1] ** 2]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        r2_in = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        mo = s["month"].to_numpy()
        pred = np.zeros_like(y)
        for m in MONTHS:
            tr, te = mo != m, mo == m
            if te.sum() and tr.sum():
                b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                pred[te] = X[te] @ b
        r2_cv = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append((p, len(y), round(100 * r2_in, 2), round(100 * r2_cv, 2)))
        print(f"primary {p:2d} n={len(y):6d} R2 in-sample {100*r2_in:6.2f}%  leave-one-month-out {100*r2_cv:6.2f}%")
    pl.DataFrame(rows, schema=["primary", "n", "r2_in_pct", "r2_cv_pct"], orient="row").write_csv(
        ML_EXPERIMENTS_DATA_DIR / "00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/step0a.csv"
    )


if __name__ == "__main__":
    main()
