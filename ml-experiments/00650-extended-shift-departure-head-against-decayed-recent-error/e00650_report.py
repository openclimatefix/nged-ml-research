# ruff: noqa: E501, D103
"""Experiment 650, step 6: contrasts with month-block intervals, cell tables, and tail tables (CPU).

Reads `score*/per_row_summary.parquet` and `score*/scorecard.csv` of step 5. Seeds are averaged within an
arm and month; intervals are percentile bootstraps over the months of the scored set (10,000 resamples).
The decision contrast is the inside phase of the out-of-support held-out events, after equal width scaling.

Usage: python e00650_report.py [score|score5]
"""

import sys

import e00650_common as c650
import numpy as np
import polars as pl

SC = c650.WORK / (sys.argv[1] if len(sys.argv) > 1 else "score5")
RNG = np.random.default_rng(650)


def month_means(*, d: pl.DataFrame, col: str) -> pl.DataFrame:
    return (
        d.group_by("month", "arm", "seed")
        .agg(pl.col(col).mean())
        .group_by("month", "arm")
        .agg(pl.col(col).mean())
    )


def contrast(*, d: pl.DataFrame, a: str, b: str, col: str) -> dict:
    mm = month_means(d=d, col=col)
    pa = dict(mm.filter(pl.col("arm") == a).select("month", col).iter_rows())
    pb = dict(mm.filter(pl.col("arm") == b).select("month", col).iter_rows())
    months = sorted(set(pa) & set(pb))
    diff = np.array([pa[m] - pb[m] for m in months])
    boot = np.array([RNG.choice(diff, size=len(diff)).mean() for _ in range(10_000)])
    return {"a": a, "b": b, "metric": col, "n_months": len(months), "diff": float(diff.mean()),
            "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "per_month": "; ".join(f"{m}: {x:+.2f}" for m, x in zip(months, diff, strict=True))}  # fmt: skip


def main() -> None:
    pr = pl.read_parquet(SC / "per_row_summary.parquet")
    held = pr.filter(pl.col("set") == "held")
    oos = held.with_columns(oos=pl.col("cell").str.split("|").list.first() != "")
    rows = []
    sets = {
        "inside, out-of-support events (decision)": oos.filter(
            (pl.col("phase") == "inside") & pl.col("oos")
        ),
        "inside, in-support events": oos.filter((pl.col("phase") == "inside") & ~pl.col("oos")),
        "inside, all held-out events": held.filter(pl.col("phase") == "inside"),
        "after the shift ends (veto 1)": held.filter(pl.col("phase") == "after"),
        "event-free real rows (veto 2)": pr.filter(pl.col("set") == "real"),
        "inside, training-family events": pr.filter(
            (pl.col("set") == "fam") & (pl.col("phase") == "inside")
        ),
        "after, training-family events": pr.filter(
            (pl.col("set") == "fam") & (pl.col("phase") == "after")
        ),
    }
    for name, d in sets.items():
        for col in ("crps_scaled", "crps"):
            for a, b in (
                ("Rnew", "R0-168"),
                ("Rnew", "Rold"),
                ("Rnew", "K1"),
                ("Rnew", "Placebo"),
                ("Rold", "R0-168"),
                ("R0-168", "K1"),
            ):
                rows.append({"set": name, **contrast(d=d, a=a, b=b, col=col)})
    pl.DataFrame(rows).write_csv(SC / "contrasts.csv")
    levels = []
    for name, d in sets.items():
        mm = month_means(d=d, col="crps_scaled").group_by("arm").agg(pl.col("crps_scaled").mean())
        raw = month_means(d=d, col="crps").group_by("arm").agg(pl.col("crps").mean())
        levels.append(
            mm.join(raw, on="arm").with_columns(
                set=pl.lit(name),
                n_rows=pl.lit(d.height // max(d["seed"].n_unique() * d["arm"].n_unique(), 1)),
            )
        )
    pl.concat(levels).write_csv(SC / "levels.csv")
    # cells of the inside phase: by depth and by out-of-support label
    cells = oos.filter(pl.col("phase") == "inside").with_columns(
        label=pl.col("cell").str.split("|").list.first(),
        days=pl.col("cell").str.split("|").list.last(),
    )
    by_depth = (
        cells.group_by("depth", "arm", "month", "seed")
        .agg(pl.col("crps_scaled").mean())
        .group_by("depth", "arm", "month")
        .agg(pl.col("crps_scaled").mean())
        .group_by("depth", "arm")
        .agg(pl.col("crps_scaled").mean())
    )
    by_label = (
        cells.group_by("label", "arm", "month", "seed")
        .agg(pl.col("crps_scaled").mean())
        .group_by("label", "arm", "month")
        .agg(pl.col("crps_scaled").mean())
        .group_by("label", "arm")
        .agg(pl.col("crps_scaled").mean())
    )
    by_depth.write_csv(SC / "by_depth.csv")
    by_label.write_csv(SC / "by_label.csv")
    # tail table from the scorecard (seed- and month-averaged, pinball in points of capacity)
    sc = pl.read_csv(SC / "scorecard.csv")
    sc.group_by("set", "phase", "arm").agg(
        pl.col(c).mean()
        for c in (
            "crps",
            "cover90",
            "cover98",
            "pin01",
            "pin05",
            "pin95",
            "pin99",
            "crps_scaled",
            "factor",
        )
    ).sort("set", "phase", "arm").write_csv(SC / "tails.csv")
    c = pl.read_csv(SC / "contrasts.csv")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    print(
        c.filter((pl.col("metric") == "crps_scaled") & pl.col("a").is_in(["Rnew"])).select(
            "set", "b", "diff", "lo", "hi", "per_month"
        )
    )


if __name__ == "__main__":
    main()
