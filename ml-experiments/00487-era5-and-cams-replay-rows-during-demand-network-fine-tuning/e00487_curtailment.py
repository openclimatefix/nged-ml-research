"""Avoidable-curtailment cost of experiment 487's arms at the signed-MW six (descriptive).

Runs experiment 470's `e00470_curtailment` with this experiment's arms: the module's arm list,
comparisons, row loader, and score folder are replaced by `R0`, `R25`, `R50`, `Rcold`, and
`A-pool`, read from `work/predictions`. Rows are the `gam14` rows with no labelled event that BG2
and `manual_heuristic` also hold, fractions f = 0.9, 0.75, and 0.5 of the scaled limit, 1.15
widening on the tail quantiles.

```bash
source .e_replay_env.sh   # sets PYTHONPATH; run from the repo root
uv run --no-sync python ml-experiments/00487-*/e00487_curtailment.py
```
"""

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as curt470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00487_build as build  # ty: ignore[unresolved-import]
import e00487_score as score  # ty: ignore[unresolved-import]
import polars as pl

ARMS = ("R0", "R25", "R50", "Rcold", "A-pool")
COMPARISONS = (
    ("R25", "R0"), ("R50", "R0"), ("Rcold", "R0"),
    *[(a, ref) for a in ("R0", "R25", "R50", "Rcold") for ref in curt470.REFERENCES],
)  # fmt: skip


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    build470.prediction_path = build.prediction_path  # ty: ignore[invalid-assignment]
    score470.arm_cells = lambda *, arm: score.CELLS.get(arm, 1)  # ty: ignore[invalid-assignment]
    score470.SCORE_DIR = score.SCORE_DIR  # ty: ignore[invalid-assignment]
    curt470.ARMS = ARMS
    c449.ARMS = ARMS  # ty: ignore[invalid-assignment]
    c449.REFERENCES = curt470.REFERENCES  # ty: ignore[invalid-assignment]
    c449.COMPARISONS = COMPARISONS  # ty: ignore[invalid-assignment]
    c449.RESAMPLES = curt470.RESAMPLES  # ty: ignore[invalid-assignment]
    curt470.MONTHS = build.fit.MONTHS
    cost = curt470.build()
    cost.write_parquet(score.SCORE_DIR / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(score.SCORE_DIR / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()
