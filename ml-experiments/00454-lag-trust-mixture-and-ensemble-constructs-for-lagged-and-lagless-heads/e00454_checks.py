"""Checks for experiment 454: median minus mean forecast per site, and the mean-forecast error.

For fit-seed cell 0 of five constructs, the mean of each row's mixture is computed from the cached
heads (`e00454_build.py`). The script prints, at the signed-MW six on rows with no labelled event:
the mean over rows of (median minus mean forecast) per Primary, and the mean absolute error of the
mean forecast and of the median forecast, in points of capacity, with the contrast against B-mix.

```bash
<PYTHONPATH as e00454_build.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00454-*/e00454_checks.py
```
"""

from typing import Final

import e00454_build as build  # ty: ignore[unresolved-import]
import e00454_score as scoring  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

POINTS: Final[float] = 100.0
ARMS: Final[tuple[str, ...]] = ("B-mix", "S1-mix", "K2-S1", "K3-S1", "A-ens2", "A0-inj")


def mixture_mean(*, members: list[build.Head]) -> np.ndarray:
    """Return the mean (capacity fraction) of the equal-weight mixture of the heads."""
    k = len(members)
    return sum((h[0].exp() * h[1]).sum(1).numpy() / k for h in members)  # ty: ignore[invalid-return-type]


def main() -> None:
    """Print the median-minus-mean and mean-error tables."""
    rows = []
    for month in build.MONTHS:
        heads, _keys = build.load_heads(month=month)
        cells = build.constructs(heads=heads)
        base = scoring.month_base(month=month)
        full = pl.read_parquet(build.prediction_path(arm="B-mix", month=month, cell=0))
        for arm in ARMS:
            mean = mixture_mean(members=cells[arm][0])
            frame = full.select("primary", "origin", "target", "q50").with_columns(
                mean=pl.Series(mean)
            )
            joined = (
                base.join(frame, on=["primary", "origin", "target"]) if arm == "B-mix" else None
            )
            q = pl.read_parquet(build.prediction_path(arm=arm, month=month, cell=0)).select(
                "primary", "origin", "target", "q50"
            )
            joined = base.join(q, on=["primary", "origin", "target"]).join(
                frame.select("primary", "origin", "target", "mean"),
                on=["primary", "origin", "target"],
            )
            part = joined.filter((pl.col("group") == "signed_six") & (pl.col("cls") == "clean"))
            rows.append(
                part.group_by("primary")
                .agg(
                    ((pl.col("q50") - pl.col("mean")) * POINTS).mean().alias("median_minus_mean"),
                    ((pl.col("actual") - pl.col("mean")).abs() * POINTS).sum().alias("abs_mean"),
                    ((pl.col("actual") - pl.col("q50")).abs() * POINTS).sum().alias("abs_median"),
                    pl.len().alias("n"),
                )
                .with_columns(arm=pl.lit(arm), month=pl.lit(month))
            )
    out = pl.concat(rows)
    pl.Config.set_tbl_rows(100)
    per_site = out.group_by("arm", "primary").agg(
        (pl.col("median_minus_mean") * pl.col("n")).sum() / pl.col("n").sum()
    )
    print("median minus mean forecast, points of capacity, by Primary index:")
    print(per_site.pivot(on="arm", index="primary", values="median_minus_mean").sort("primary"))
    per_month = out.group_by("arm", "month").agg(
        pl.col("abs_mean").sum() / pl.col("n").sum(),
        (pl.col("abs_median").sum() / pl.col("n").sum()).alias("mae_median"),
    )
    pooled = per_month.group_by("arm").agg(
        pl.col("abs_mean").mean().alias("mae_mean"), pl.col("mae_median").mean()
    )
    ref = pooled.filter(pl.col("arm") == "B-mix").row(0, named=True)
    print(
        pooled.with_columns(
            mean_minus_bmix=pl.col("mae_mean") - ref["mae_mean"],
            median_minus_bmix=pl.col("mae_median") - ref["mae_median"],
        ).sort("mae_mean")
    )


if __name__ == "__main__":
    main()
