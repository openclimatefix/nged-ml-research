"""Print the diagnostic tables of experiment 436 that the write-up quotes.

Reads `work/diag_*.parquet` and `work/scores_all.parquet`, and prints the marginal check, the
spread-skill ratio, the rank-histogram statistics, the dependence check, and the month-block
bootstrap contrasts of the downstream proxies (Brier scores, group-sum scores).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
uv run --no-sync python ml-experiments/00436-*/e00436_summary.py
```
"""

import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_report as report  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

W = build.WORK_DIR
ARMS = ["J0", "J1", "J2", "J3", "J4", "J5", "J6", "J6-median"]


def marginal() -> None:
    """Print the 51-value CRPS, tail exceedance, and spread-skill ratio of J2, J6, and J6-median."""
    m = pl.read_parquet(W / "diag_marginal.parquet")
    print("\n=== marginal check, days 1 to 10 (mean over runs; exceedance shares) ===")
    for window, (lo, hi) in {"days_1_10": (1, 10), "day_1_2": (1, 2)}.items():
        sub = m.filter(pl.col("lead_day").is_between(lo, hi))
        t = (
            sub.group_by("arm", "family")
            .agg(
                (pl.col("crps") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("above_q95") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("above_q99") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("below_q05") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("below_q01") * pl.col("n")).sum() / pl.col("n").sum(),
                ((pl.col("var") * pl.col("n")).sum() / (pl.col("mse") * pl.col("n")).sum())
                .sqrt()
                .alias("spread_ratio_raw"),
            )
            .with_columns((pl.col("spread_ratio_raw") * np.sqrt(1 + 1 / 51)).alias("spread_skill"))
            .sort("family", "arm")
        )
        print(window)
        print(t)


def ranks() -> None:
    """Print the share of observations in the end bins and middle bins of each rank histogram."""
    r = pl.read_parquet(W / "diag_rank.parquet")
    print(
        "\n=== rank histograms: share in the lowest+highest 10% of bins (flat 0.20), middle 20% (flat 0.20) ==="
    )
    rows = []
    for (arm, block, kind), g in r.group_by("arm", "block", "kind"):
        c = g.sort("bin")["count"].to_numpy()
        n = c.sum()
        if n == 0:
            continue
        ends = (c[:5].sum() + c[-5:].sum()) / n
        mid = c[21:31].sum() / n
        rows.append(
            {
                "arm": arm,
                "block": block,
                "kind": kind,
                "n": int(n),
                "ends": round(ends, 3),
                "middle": round(mid, 3),
            }
        )
    print(pl.DataFrame(rows).filter(pl.col("kind") == "average_rank").sort("block", "arm"))


def dependence() -> None:
    """Print the root-mean-square difference of the dependence matrices."""
    d = pl.read_parquet(W / "diag_dependence.parquet")
    print(
        "\n=== dependence check: rmsd of scenario against realised score correlations (scenario / observed means) ==="
    )
    print(
        d.with_columns(
            pl.col("rmsd").round(3), pl.col("scenario").round(3), pl.col("observed").round(3)
        )
        .sort("item", "arm")
        .pivot(on="arm", index="item", values="rmsd")
        .sort("item")
    )
    print(
        d.filter(pl.col("item").is_in(["lag0_demand_solar", "lag0_demand_wind"]))
        .select("item", "arm", "scenario", "observed")
        .sort("item", "arm")
    )


def proxies() -> None:
    """Print the proxy scores and their paired contrasts against J0 (month-block bootstrap)."""
    p = pl.read_parquet(W / "diag_proxy.parquet")
    print(
        "\n=== downstream proxies, mean over runs and groups, relative difference against J0 (month-block interval) ==="
    )
    for metric in (
        "brier_two_over",
        "brier_export",
        "brier_ramp",
        "sum_crps",
        "excess_volume_abs_error",
    ):
        monthly = (
            p.group_by("month", "arm")
            .agg(pl.col(metric).mean())
            .pivot(on="arm", index="month", values=metric)
            .drop_nulls()
            .sort("month")
        )
        base = monthly["J0"].to_numpy()
        parts = [f"{metric:24s} J0 {base.mean():.4f}"]
        for arm in ARMS[1:]:
            if arm in monthly.columns:
                point, low, high = report.bootstrap_relative(
                    base=base, other=monthly[arm].to_numpy()
                )
                parts.append(f"{arm} {point:+.1%} ({low:+.1%} to {high:+.1%})")
        print("  ".join(parts))
    cov = (
        p.group_by("arm")
        .agg(pl.col("sum_above_q95").mean(), pl.col("sum_above_q99").mean())
        .sort("arm")
    )
    print(cov)
    brier = (
        p.filter(pl.col("arm") == "J0")
        .group_by("month")
        .agg(
            pl.col("brier_two_over").mean().alias("two_over_J0"),
            pl.col("brier_export").mean().alias("export_J0"),
        )
        .sort("month")
    )
    print(brier)


if __name__ == "__main__":
    marginal()
    ranks()
    dependence()
    proxies()
