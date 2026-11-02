"""Print every number the page of experiment 494 quotes, from the scorer's stored tables.

For each comparison: the two-seed minimum detectable effect and the limit, the counted contrasts
(CRPS and q99 pinball loss at the eight and the six) with the verdict of the decision rule, the
leave-one-fold-out difference, the absolute CRPS, the tail coverage and pinball loss, the cold-night
and 12 to 36 hour slices, the avoidable-curtailment cost, the median-minus-mean forecast per
Primary, and the ENS window of each network's cut.

```bash
source <paths>
uv run --no-sync python ml-experiments/00494-*/e00494_report.py
```
"""

import json
from typing import Final

import e00494_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

CAPS: Final[dict[str, float]] = {"Stale3": 0.05, "PO": 0.05, "Stale6": 0.10, "Stale6_stored": 0.10}
Q99_LIMIT: Final[float] = 0.02
SLICES: Final[tuple[str, ...]] = ("eight", "six_clean")


def fold_level(*, values: list[float]) -> str:
    """Return the t interval of the mean across folds (an interval that ignores no fold effect)."""
    from scipy import stats

    n = len(values)
    if n < 3:
        return "n/a (fewer than 3 folds)"
    mean, se = float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return f"[{mean - half:+.3f}, {mean + half:+.3f}]"


def verdict(*, upper: float, limit: float) -> str:
    """Return the decision-rule verdict for an upper Bonferroni bound."""
    return "equivalent" if upper <= limit else "not shown equivalent"


def comparison(*, name: str) -> None:
    """Print one comparison's tables."""
    folder = score.WORK_DIR / "scores" / name
    arm = score.COMPARISONS[name][0]
    mde = pl.read_csv(folder / "mde_k2.csv")
    contrasts = pl.read_csv(folder / "contrasts.csv")
    absolute = pl.read_csv(folder / "absolute.csv")
    print(f"\n##### {name} ({arm} minus K2), months {score.COMPARISONS[name][1]}")
    print(mde)
    mdes = {"eight": mde["mde"][0], "six_clean": mde["mde"][1]}
    for sname in SLICES:
        for metric in ("crps", "pin_q99"):
            row = contrasts.filter((pl.col("slice") == sname) & (pl.col("metric") == metric)).row(
                0, named=True
            )
            limit = max(CAPS.get(name, 0.10), mdes[sname]) if metric == "crps" else Q99_LIMIT
            by_month = json.loads(row["by_month"])
            loo = {m: float(np.mean([v for k, v in by_month.items() if k != m])) for m in by_month}
            print(
                f"{sname:9s} {metric:8s} diff {row['difference']:+.3f} "
                f"95% [{row['low']:+.3f}, {row['high']:+.3f}] "
                f"Bonferroni-12 [{row['low_bonferroni']:+.3f}, {row['high_bonferroni']:+.3f}] "
                f"limit {limit:.3f} -> {verdict(upper=row['high_bonferroni'], limit=limit)}; "
                f"folds {by_month}; leave-one-fold-out "
                f"{ {k: round(v, 3) for k, v in loo.items()} }; "
                f"fold-level 95% t interval {fold_level(values=list(by_month.values()))}"
            )
    wide = absolute.filter(pl.col("rowset") == "main16").select(
        "slice", "arm", "crps", "pin_q01", "pin_q05", "pin_q95", "pin_q99",
        "cov_q01", "cov_q05", "cov_q95", "cov_q99",
    )  # fmt: skip
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200):
        print(
            wide.filter(
                pl.col("slice").is_in(
                    [
                        "eight",
                        "six_clean",
                        "cold_eight",
                        "cold_six",
                        "band_eight",
                        "band_six",
                        "report",
                    ]
                )
            ).with_columns(pl.col(pl.Float64).round(3))
        )
        curtailment = pl.read_csv(folder / "curtailment_bootstrap.csv")
        print(curtailment.filter(pl.col("item").str.contains("minus K2|^K2$|^" + arm + "$")))


def checks() -> None:
    """Print the audit checks: median minus mean forecast per Primary, and the median's error."""
    import e00087_common as common  # ty: ignore[unresolved-import]
    import e00377_adapter as adapter  # ty: ignore[unresolved-import]
    import e00425_prep as prep425  # ty: ignore[unresolved-import]
    import e00470_build as build470  # ty: ignore[unresolved-import]
    import e00470_score as score470  # ty: ignore[unresolved-import]

    score.configure(comparison="Stale3")
    sites = build470.site_map()
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    masses = np.asarray(common.ATOM_MASSES, dtype=np.float64)
    masses = masses / masses.sum()
    arms = ("K2", "Stale3", "Stale6", "PO")
    rows = []
    for month in score.FIVE:
        base = score470.month_context(month=month, sites=sites, ctx=ctx)
        for arm in arms:
            frame = pl.read_parquet(build470.prediction_path(arm=arm, month=month, cell=0))
            joined = base.join(frame, on=build470.KEYS, how="inner")
            q = joined.select(list(common.LEVEL_COLUMNS)).to_numpy().astype(np.float64)
            rows.append(
                joined.select("primary", "group", actual=pl.col("actual")).with_columns(
                    arm=pl.lit(arm),
                    median=pl.Series(q[:, 5]),
                    mean=pl.Series(q @ masses),
                )
            )
    table = pl.concat(rows).with_columns(
        median_error=(pl.col("median") - pl.col("actual")).abs() * score470.POINTS,
        mean_error=(pl.col("mean") - pl.col("actual")).abs() * score470.POINTS,
        median_minus_mean=(pl.col("median") - pl.col("mean")) * score470.POINTS,
    )
    with pl.Config(tbl_rows=80, tbl_width_chars=160):
        print("\nCHECKS: error of the median and mean forecast (points of capacity), by group")
        print(
            table.group_by("group", "arm", maintain_order=True)
            .agg(pl.col("median_error").mean(), pl.col("mean_error").mean())
            .sort("group", "arm")
        )
        print("CHECKS: median minus mean forecast, per Primary (points of capacity), K2 and PO")
        print(
            table.filter(pl.col("arm").is_in(["K2", "PO"]))
            .group_by("primary", "arm", maintain_order=True)
            .agg(pl.col("median_minus_mean").mean())
            .sort("primary", "arm")
        )


def main() -> None:
    """Print every comparison and the audit checks."""
    for name in score.COMPARISONS:
        comparison(name=name)
    checks()


if __name__ == "__main__":
    main()
