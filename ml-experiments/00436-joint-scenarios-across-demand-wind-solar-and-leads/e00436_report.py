"""Report the scores of experiment 436: coherence margins and month-block bootstrap contrasts.

Reads `work/<name>.parquet` (from `e00436_score.py`), averages each score over the runs of a
calendar month, and prints, for every block, window, and slice, the mean variogram and energy score
of each arm, the coherence margin (a quarter of the relative gap between J1 and J0, bounded to 2%
to 5%), and the relative difference of each arm against J0 with a month-block bootstrap interval.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
uv run --no-sync python ml-experiments/00436-*/e00436_report.py scores_tier0
```
"""

import sys
from typing import Final

import e00436_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
CONFIDENCE: Final[float] = 0.9875
"""Two-sided level after Bonferroni correction over the four registered contrasts."""
MARGIN_LOW: Final[float] = 0.02
MARGIN_HIGH: Final[float] = 0.05


def monthly(*, scores: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Return the mean `metric` per calendar month, arm, block, window, slice, and label."""
    return (
        scores.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .group_by("month", "arm", "block", "window", "slice", "label")
        .agg(pl.col(metric).mean())
    )


def margin(*, ind: float, co: float) -> float:
    """Return a quarter of the relative gap between J1 and J0, bounded to 2% to 5%."""
    return float(np.clip((co - ind) / ind / 4.0, MARGIN_LOW, MARGIN_HIGH))


def bootstrap_relative(
    *, base: np.ndarray, other: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Return the relative difference of mean(other) against mean(base), with a month-block interval.

    `base` and `other` hold one value per month, in the same month order.
    """
    rng = np.random.default_rng(seed)
    n = len(base)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    rel = other[idx].mean(1) / base[idx].mean(1) - 1.0
    alpha = (1 - CONFIDENCE) / 2
    return (
        float(other.mean() / base.mean() - 1.0),
        float(np.quantile(rel, alpha)),
        float(np.quantile(rel, 1 - alpha)),
    )


def main() -> None:
    """Print the report for the named score file."""
    name = sys.argv[1] if len(sys.argv) > 1 else "scores_tier0"
    scores = pl.read_parquet(build.WORK_DIR / f"{name}.parquet")
    print(f"runs: {scores['power_fcst_init_time'].n_unique()}")
    print(
        scores.with_columns(month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(pl.col("power_fcst_init_time").n_unique().alias("runs"))
        .sort("month")
    )
    for metric in ("variogram", "energy"):
        table = monthly(scores=scores, metric=metric)
        print(f"\n=== {metric} score, relative to J0 (month-block {CONFIDENCE:.2%} interval) ===")
        keys = (
            table.select("block", "window", "slice", "label")
            .unique()
            .sort("block", "window", "slice", "label")
        )
        for block, window, slice_name, label in keys.iter_rows():
            sub = table.filter(
                (pl.col("block") == block) & (pl.col("window") == window)
                & (pl.col("slice") == slice_name) & (pl.col("label") == label)
            )  # fmt: skip
            wide = sub.pivot(on="arm", index="month", values=metric).drop_nulls().sort("month")
            if "J0" not in wide.columns or wide.height < 3:
                continue
            base = wide["J0"].to_numpy()
            gap = (
                margin(ind=base.mean(), co=wide["J1"].mean())
                if "J1" in wide.columns
                else float("nan")
            )
            parts = [
                f"{block:8s} {window:9s} {slice_name:8s} {label:7s} J0 {base.mean():.4f} "
                f"margin {gap:.1%}"
            ]
            for arm in [c for c in wide.columns if c not in ("month", "J0")]:
                point, low, high = bootstrap_relative(base=base, other=wide[arm].to_numpy())
                parts.append(f"{arm} {point:+.1%} ({low:+.1%} to {high:+.1%})")
            print("  ".join(parts))


if __name__ == "__main__":
    main()
