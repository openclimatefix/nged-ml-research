"""Avoidable-curtailment cost of experiment 491's arms (descriptive, uncounted).

Reuses experiment 470's costing (`e00470_curtailment`) with this experiment's stored cells:
forecasts are the mean over an arm's cells in quantile space, widened by 1.15 on the four tail
quantiles, converted to MW, on the rows with no labelled event that `manual_heuristic` also holds,
at f = 0.9, 0.75, and 0.5 of the scaled limit, for the signed-MW six.

```bash
source /tmp/e491env.sh
uv run --no-sync python ml-experiments/00491-*/e00491_curtailment.py
```
"""

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as c470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00491_score as score491  # ty: ignore[unresolved-import]

ARMS = ("S0", "S0f", "EMA", "SWA", "Pool2", "Pool3", "Pool5", "PoolEMA3")
REFERENCES = ("manual_heuristic",)
COMPARISONS = (
    ("EMA", "S0f"), ("SWA", "S0f"), ("EMA", "Pool2"), ("SWA", "Pool2"), ("EMA", "Pool5"),
    ("SWA", "Pool5"), ("S0f", "S0"), ("Pool5", "S0f"), ("PoolEMA3", "Pool3"),
    *[(a, "manual_heuristic") for a in ("S0", "S0f", "EMA", "SWA", "Pool5")],
)  # fmt: skip


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    build470.prediction_path = score491.pred_path  # ty: ignore[invalid-assignment]
    score470.arm_cells = score491.cells_of  # ty: ignore[invalid-assignment]
    c470.ARMS = ARMS
    c470.REFERENCES = REFERENCES
    c449.ARMS = ARMS  # ty: ignore[invalid-assignment]
    c449.REFERENCES = REFERENCES  # ty: ignore[invalid-assignment]
    c449.COMPARISONS = COMPARISONS  # ty: ignore[invalid-assignment]
    c449.RESAMPLES = c470.RESAMPLES  # ty: ignore[invalid-assignment]
    cost = c470.build()
    score491.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    cost.write_parquet(score491.SCORE_DIR / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(score491.SCORE_DIR / "curtailment_bootstrap.csv")
    import polars as pl

    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()
