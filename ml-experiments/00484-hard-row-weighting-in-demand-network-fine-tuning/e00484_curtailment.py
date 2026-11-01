r"""Avoidable-curtailment cost of experiment 484's arms beside the stored fit and the references.

Reuses experiment 470's `e00470_curtailment.py` with this experiment's arms: the mean over an
arm's three seed cells in quantile space, widened by 1.15 on the four tail quantiles, in MW, on
the signed-MW six rows with no labelled event that BG2 and manual_heuristic also hold, at the
fractions f = 0.9, 0.75, and 0.5 of the scaled limit. Intervals resample (month, series) cells.

```bash
source /tmp/e484env.sh
uv run --no-sync python ml-experiments/00484-*/e00484_curtailment.py
```
"""

from typing import Final

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_curtailment as c470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00484_score as score484  # ty: ignore[unresolved-import]
import polars as pl

ARMS: Final[tuple[str, ...]] = score484.ARMS
REFERENCES: Final[tuple[str, ...]] = ("BG2", "manual_heuristic")
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    *[(arm, score484.REFERENCE) for arm in score484.NEW_ARMS],
    *[(arm, ref) for arm in ARMS for ref in REFERENCES],
)


def main() -> None:
    """Write and print the per-cell costs and the bootstrap table."""
    build470.prediction_path = lambda *, arm, month, cell: score484.pred_path(  # ty: ignore[invalid-assignment]
        arm=arm, month=month, cell=cell
    )
    score470.CELLS = {**score470.CELLS, **dict.fromkeys(ARMS, len(score484.SEEDS))}  # ty: ignore[invalid-assignment]
    score470.SCORE_DIR = score484.SCORE_DIR
    # Rows come from experiment 470's own table (its gam14 flags), read from its folder.
    c470.score470.load_rows = lambda: pl.concat(  # ty: ignore[invalid-assignment]
        [
            pl.read_parquet(score484.build470.heads470.WORK_DIR / "scores" / f"rows_{m}.parquet")
            for m in score484.MONTHS
        ]
    )
    c470.MONTHS = score484.MONTHS
    c470.ARMS = ARMS
    c449.ARMS = ARMS
    c449.REFERENCES = REFERENCES
    c449.COMPARISONS = COMPARISONS
    c449.RESAMPLES = c470.RESAMPLES
    cost = c470.build()
    out = score484.SCORE_DIR
    cost.write_parquet(out / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(out / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=140):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()
