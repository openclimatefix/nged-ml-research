r"""Score S1 and its five-seed pool at several widening factors (experiment 448, audit item).

The 1.15 widening was chosen for a single S1 fit, and pooling already widens the mixture, so the
script prints CRPS, lower-tail coverage (q01, q05), and upper-tail coverage (q95, q99) of S1 and of
the pool at factors 1.00, 1.05, 1.10, and 1.15 on the clean rows of the four screen folds.

```bash
source <paths>; PYTHONPATH=$PP DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00448-*/e00448_widen.py
```
"""

from typing import Final

import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00448_run as run  # ty: ignore[unresolved-import]
import polars as pl

FACTORS: Final[tuple[float, ...]] = (1.0, 1.05, 1.10, 1.15)
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


def main() -> None:
    """Print the sweep."""
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    score.ARMS = ("S1-pool5", "S1")  # ty: ignore[invalid-assignment]
    score.CONTRASTS = {"pool minus S1": ("S1-pool5", "S1")}  # ty: ignore[invalid-assignment]
    frames = []
    for factor in FACTORS:
        table = score.collect(months=list(fit.SCREEN_MONTHS), seeds=SEEDS, factor=factor)
        absolute = score.summarise(table=table, seeds=SEEDS).with_columns(factor=pl.lit(factor))
        frames.append(absolute.filter(pl.col("row_set") == "clean"))
    out = pl.concat(frames).sort("group", "arm", "factor")
    out.write_csv(run.WORK_DIR / "figure_tables" / "widening_sweep.csv")
    with pl.Config(tbl_rows=60, tbl_cols=14, tbl_width_chars=200):
        print(
            out.select(
                "group", "arm", "factor", "crps", "coverage_q01", "coverage_q05",
                "coverage_q95", "coverage_q99",
            )
        )  # fmt: skip


if __name__ == "__main__":
    main()
