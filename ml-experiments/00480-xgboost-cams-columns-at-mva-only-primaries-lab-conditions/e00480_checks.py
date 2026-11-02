"""Print experiment 480's audit-trigger checks: median minus mean forecast, and leave-one-out.

The mean forecast is estimated from the 11 stored quantiles as the trapezoid integral of the
quantile function over the levels, with the first and last quantile held flat out to levels 0 and 1.
Leave-one-out drops one Primary from the pooled eight (and from the pooled six) and prints the
A1 minus A0 contrast at the 99.67% level.

```bash
PYTHONPATH=ml-experiments/00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00480-*/e00480_checks.py
```
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00480_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics

LEVELS: np.ndarray = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
QUANTILE_COLUMNS: list[str] = [f"q{round(100 * x):02d}" for x in LEVELS]


def mean_from_quantiles(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the mean of each row's forecast distribution, by trapezoid over the levels."""
    values = frame.select(QUANTILE_COLUMNS).to_numpy()
    grid = np.concatenate([[0.0], LEVELS, [1.0]])
    padded = np.concatenate([values[:, :1], values, values[:, -1:]], axis=1)
    return np.sum((padded[:, 1:] + padded[:, :-1]) / 2 * np.diff(grid), axis=1)


def main() -> None:
    """Print the checks."""
    pl.Config.set_tbl_rows(50)
    series = adapter.series_table()
    arms = {n: score.own_arm(name=n, series=series) for n in ("a0", "a1")}
    labels = series.select("time_series_id", "label")
    print("== median minus mean forecast, per Primary, MW (mean of the half-hours scored) ==")
    parts = []
    for name, frame in arms.items():
        mean = mean_from_quantiles(frame=frame)
        parts.append(
            frame.with_columns(gap=pl.col("q50") - pl.Series(mean))
            .join(labels, on="time_series_id")
            .group_by("label")
            .agg(pl.col("gap").mean().alias(f"median_minus_mean_{name}"))
        )
    print(parts[0].join(parts[1], on="label").sort("label"))
    core = {n: arms[n] for n in ("a0", "a1")}
    ctx = score.s381.build_context(series=series)
    table, _, rows = score.s381.score_arms(arms=core, series=series, ctx=ctx)
    print("\n== leave-one-out, A1 minus A0, CRPS days 1 to 10, 99.67% ==")
    for gname, group in (("MVA-only eight", score.MVA_ONLY), ("signed-MW six", score.SIGNED)):
        for left_out in group:
            keep = pl.col("label").is_in([x for x in group if x != left_out])
            r = metrics.month_bootstrap_contrast(
                treatment=rows["a1"],
                reference=rows["a0"],
                metric="crps",
                where=keep & pl.col(score.HEADLINE),
                level_percent=score.LEVEL,
                n_resamples=score.RESAMPLES,
                slice_table=table,
            )
            print(
                f"{gname:16s} without {left_out:4s} {r['difference']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f})"
            )


if __name__ == "__main__":
    main()
