# ruff: noqa: D103, E501
"""Experiment 652, step 3: contrasts with month-block intervals, levels by lead day, tails, and reliability (CPU).

Reads `score/agg.parquet` and `score/reliability.parquet`; writes `score/{levels,contrasts,tails,reliability}.csv`.
A cell's value in a month is the mean over lead days 1 to 10 of the event-weighted mean (seeds
averaged first). Intervals are percentile bootstraps over the 5 development months.

Usage: python e00652_report.py
"""

from typing import Final

import e00652_common as c652  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SC: Final = c652.WORK / "score"
PAIRS: Final = (
    ("Rhaz", "Bhaz"), ("Rhaz", "R0-168"), ("Rhaz", "Rnew"), ("Rhaz", "Placebo"), ("Rhaz", "K1"),
    ("Bhaz", "R0-168"), ("Rnew", "R0-168"), ("Rnew", "Bhaz"), ("Placebo", "K1"),
)  # fmt: skip
N_BOOT: Final[int] = 10_000
RNG: Final = np.random.default_rng(652)


def cell_values(*, agg: pl.DataFrame, col: str) -> pl.DataFrame:
    """Return the per-month, per-lead-day value of `col` (seeds averaged)."""
    per_seed = agg.group_by("month", "set", "mixture", "arm", "seed", "stratum", "lead_day").agg(
        pl.col(col).sum().alias("num"), pl.col("sw").sum().alias("den")
    )
    per_seed = per_seed.with_columns(v=pl.col("num") / pl.col("den"))
    return per_seed.group_by("month", "set", "mixture", "arm", "stratum", "lead_day").agg(
        pl.col("v").mean()
    )


def boot(*, x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, 2.5%, 97.5%) of the month-block bootstrap of the mean of `x`."""
    draws = x[RNG.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    agg = pl.read_parquet(SC / "agg.parquet")
    levels, contrasts = [], []
    for col in ("crps_s", "crps"):
        vals = cell_values(agg=agg, col=col)
        monthly = vals.group_by("month", "set", "mixture", "arm", "stratum").agg(pl.col("v").mean())
        for (st, mix, sset), grp in monthly.group_by("stratum", "mixture", "set"):
            wide = grp.pivot(on="arm", index="month", values="v").sort("month")
            for arm in [c for c in wide.columns if c != "month"]:
                x = wide[arm].drop_nulls().to_numpy()
                if len(x) == 5:
                    mean, lo, hi = boot(x=x)
                    levels.append(
                        {
                            "metric": col,
                            "set": sset,
                            "mixture": mix,
                            "stratum": st,
                            "arm": arm,
                            "mean": mean,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
            for a, b in PAIRS:
                if a in wide.columns and b in wide.columns:
                    d = (wide[a] - wide[b]).drop_nulls().to_numpy()
                    if len(d) == 5:
                        mean, lo, hi = boot(x=d)
                        contrasts.append(
                            {
                                "metric": col,
                                "set": sset,
                                "mixture": mix,
                                "stratum": st,
                                "a": a,
                                "b": b,
                                "diff": mean,
                                "lo": lo,
                                "hi": hi,
                                "per_month": ";".join(f"{v:.3f}" for v in d),
                            }
                        )
        # by lead day, for the headline figure
        by_day = vals.group_by("set", "mixture", "arm", "stratum", "lead_day").agg(
            pl.col("v").mean().alias("mean"),
            pl.col("v").min().alias("min"),
            pl.col("v").max().alias("max"),
        )
        by_day.with_columns(metric=pl.lit(col)).write_csv(SC / f"by_lead_day_{col}.csv")
    pl.DataFrame(levels).write_csv(SC / "levels.csv")
    pl.DataFrame(contrasts).write_csv(SC / "contrasts.csv")
    tails = (
        agg.group_by("set", "mixture", "arm", "stratum")
        .agg(
            *[
                pl.col(c).sum() / pl.col("sw").sum()
                for c in ("cov90", "cov98", "pin01", "pin05", "pin95", "pin99")
            ]
        )
        .sort("set", "mixture", "stratum", "arm")
    )
    tails.write_csv(SC / "tails.csv")
    rel = pl.read_parquet(SC / "reliability.parquet")
    rel.group_by("set", "mixture", "arm", "band", "bin").agg(
        pl.col("sw").sum(), pl.col("sp").sum(), pl.col("sl").sum(), pl.col("n").sum()
    ).with_columns(pred=pl.col("sp") / pl.col("sw"), obs=pl.col("sl") / pl.col("sw")).sort(
        "set", "mixture", "arm", "band", "bin"
    ).write_csv(SC / "reliability.csv")
    print(
        pl.DataFrame(contrasts)
        .filter(
            (pl.col("metric") == "crps_s")
            & (pl.col("set") == "fam")
            & (pl.col("mixture") == "m50")
            & (pl.col("stratum") == "inforce")
        )
        .drop("per_month")
    )


if __name__ == "__main__":
    main()
