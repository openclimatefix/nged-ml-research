# ruff: noqa: C408, E741
"""Coarse-band contrasts for the S2-M add-on (origin-day block bootstrap, as e00524_bands.py)."""

import math

import e00524_addon_s2m as a
import e00524_bands as b
import numpy as np
import polars as pl

PAIRS = [
    ("S2-M-0", "L-Ww"),
    ("S2-M-B", "L-Ww"),
    ("S2-M-0", "S2-J-0.1x"),
    ("S2-M-B", "S2-J-0.1x"),
    ("S2-M-0", "S2-J-0.4x"),
    ("S2-M-B", "S2-J-0.4x"),
    ("S2-M-B", "S2-M-0"),
    ("S2-M-B", "S2-F-B-nll"),
    ("S2-M-0", "S2-0F"),
]
TESTS = (
    3 * 2 * 3 * 1
)  # counted: S2-M-0 and S2-M-B vs L-Ww and S2-M-B vs S2-J-0.1x = 3 contrasts x 2 groups x 3 bands
COUNTED = {("S2-M-0", "L-Ww"), ("S2-M-B", "L-Ww"), ("S2-M-B", "S2-J-0.1x")}
a.configure_scoring()
rows = pl.concat(
    [
        pl.read_parquet(a.s510.score470.SCORE_DIR / f"rows_{m}.parquet").with_columns(
            fold=pl.lit(m)
        )
        for m in b.FOLDS
    ]
).filter(pl.col("main16"))
rng = np.random.default_rng(524)
recs = []
for first, second in PAIRS:
    for group, flt in (
        ("eight", pl.col("group") == "mva_eight"),
        ("six", (pl.col("group") == "signed_six") & (pl.col("cls") == "clean")),
    ):
        for band, (lo, hi) in b.BANDS.items():
            part = rows.filter(flt & (pl.col("lead_hours") >= lo) & (pl.col("lead_hours") < hi))
            diff = part.select(
                day=pl.col("origin").cast(pl.Int64) // 48,
                fold="fold",
                d=pl.col(f"{first}__crps") - pl.col(f"{second}__crps"),
            ).drop_nulls()
            by = diff.group_by("fold", "day").agg(total=pl.col("d").sum(), count=pl.col("d").len())
            t, n = by["total"].to_numpy(), by["count"].to_numpy().astype(float)
            m, l, h = b.intervals(total=t, count=n, level=0.95, rng=rng)
            cnt = (first, second) in COUNTED
            _, lb, hb = (
                b.intervals(total=t, count=n, level=1 - 0.05 / 18, rng=rng)
                if cnt
                else (math.nan,) * 3
            )
            folds = diff.group_by("fold").agg(m=pl.col("d").mean()).sort("fold")["m"].to_numpy()
            recs.append(
                dict(
                    first=first,
                    second=second,
                    group=group,
                    band=band,
                    counted=cnt,
                    difference=m,
                    low95=l,
                    high95=h,
                    low_bonf18=lb,
                    high_bonf18=hb,
                    fold_signs="".join("-" if x < 0 else "+" for x in folds),
                    fold_min=float(folds.min()),
                    fold_max=float(folds.max()),
                )
            )
t = pl.DataFrame(recs)
t.write_csv(a.OUT / "band_contrasts.csv")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
print(t.with_columns(pl.col(pl.Float64).round(3)).drop("counted"))
