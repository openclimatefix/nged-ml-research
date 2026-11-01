"""Print the four registered contrasts of experiment 436 with month-block intervals.

C1: J4 against J3. C2: J5 against J2. C3: J6 against the best of J2 to J5 on the joint block.
C4: the best construction against J0 on the two-Primaries-over-limit Brier score.
Each is a relative difference of the variogram score (lower is better), and each interval is
Bonferroni-corrected over the four contrasts.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
uv run --no-sync python ml-experiments/00436-*/e00436_contrasts.py
```
"""

import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_report as report  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

BLOCKS = [
    ("joint", "all", "all"),
    ("spatial", "all", "demand"),
    ("spatial", "all", "wind"),
    ("spatial", "all", "solar"),
    ("cross", "daylight", "all"),
]


def month_means(
    *, scores: pl.DataFrame, block: str, slice_name: str, label: str, window: str
) -> pl.DataFrame:
    """Return a wide table of monthly mean variogram score per arm."""
    sub = scores.filter(
        (pl.col("block") == block)
        & (pl.col("window") == window)
        & (pl.col("slice") == slice_name)
        & (pl.col("label") == label)
    )
    return (
        report.monthly(scores=sub, metric="variogram")
        .pivot(on="arm", index="month", values="variogram")
        .drop_nulls()
        .sort("month")
    )


def main() -> None:
    """Print C1 to C3 per block and window, then C4."""
    scores = pl.read_parquet(build.WORK_DIR / "scores_all.parquet")
    for window in ("days_1_10", "h12_36"):
        print(f"\n##### window {window}")
        for block, _, label in BLOCKS:
            slice_name = "daylight" if block == "cross" else "all"
            scope = label if block == "spatial" else "all"
            wide = month_means(
                scores=scores, window=window, block=block, slice_name=slice_name, label=scope
            )
            if wide.height < 3:
                continue
            margin = report.margin(
                ind=float(wide["J0"].to_numpy().mean()), co=float(wide["J1"].to_numpy().mean())
            )
            row = [f"{block}/{label:6s} margin {margin:.1%}"]
            for name, other, base in (("C1", "J4", "J3"), ("C2", "J5", "J2")):
                p, lo, hi = report.bootstrap_relative(
                    base=wide[base].to_numpy(), other=wide[other].to_numpy()
                )
                row.append(f"{name} {other}/{base} {p:+.1%} ({lo:+.1%} to {hi:+.1%})")
            means = {a: float(wide[a].to_numpy().mean()) for a in ("J2", "J3", "J4", "J5")}
            best = min(means, key=lambda a: means[a])
            p, lo, hi = report.bootstrap_relative(
                base=wide[best].to_numpy(), other=wide["J6"].to_numpy()
            )
            row.append(f"C3 J6/{best} {p:+.1%} ({lo:+.1%} to {hi:+.1%})")
            print("  ".join(row))
    prox = pl.read_parquet(build.WORK_DIR / "diag_proxy.parquet")
    monthly = (
        prox.group_by("month", "arm")
        .agg(pl.col("brier_two_over").mean())
        .pivot(on="arm", index="month", values="brier_two_over")
        .drop_nulls()
        .sort("month")
    )
    means = {a: float(monthly[a].to_numpy().mean()) for a in ("J2", "J3", "J4", "J5", "J6")}
    best = min(means, key=lambda a: means[a])
    p, lo, hi = report.bootstrap_relative(
        base=monthly["J0"].to_numpy(), other=monthly[best].to_numpy()
    )
    print(f"\nC4 {best}/J0 two-over Brier {p:+.1%} ({lo:+.1%} to {hi:+.1%})")
    print(np.round(list(means.values()), 4), "J0", round(float(monthly["J0"].to_numpy().mean()), 4))


if __name__ == "__main__":
    main()
