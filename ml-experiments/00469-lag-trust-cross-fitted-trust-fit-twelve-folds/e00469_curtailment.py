"""Avoidable-curtailment cost of G1-trust-12 beside G1-trust, B-mix, K2-S1, BG2, and the heuristic.

Reuses experiment 454's costing (`e00454_curtailment`) with this experiment's arms: the module's
`build_mod.WORK_DIR` is pointed at a folder of this experiment that holds one link per arm, so the
reader of stored forecasts finds them. Signed-MW six only, widening 1.15, f = 0.9, 0.75, 0.5, as
in experiment 454. K2-S1 is the mean over its stored cells.

```bash
PYTHONPATH=<every folder under ml-experiments/> DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00469-*/e00469_curtailment.py
```
"""

from pathlib import Path
from typing import Final

import e00449_curtailment as c449  # ty: ignore[unresolved-import]
import e00454_build as build_mod  # ty: ignore[unresolved-import]
import e00454_curtailment as cur  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00469-lag-trust-cross-fitted-trust-fit-twelve-folds"
LINKS: Final[Path] = OUT / "curtailment_inputs"
SOURCES: Final[dict[str, Path]] = {
    "B-mix": ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights/work/predictions/B-mix",
    "K2-S1": OUT / "predictions" / "K2-S1-mean",
    "G1-trust": ML_EXPERIMENTS_DATA_DIR
    / "00463-lag-trust-cross-fitted-trust-fit-seed-zero/predictions/G1-trust",
    "G1-trust-12": OUT / "predictions" / "G1-trust-12",
}
ARMS: Final[tuple[str, ...]] = tuple(SOURCES)
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("G1-trust-12", "B-mix"), ("G1-trust-12", "G1-trust"), ("G1-trust-12", "K2-S1"),
    ("G1-trust", "B-mix"), ("K2-S1", "B-mix"),
    ("G1-trust-12", "BG2"), ("G1-trust-12", "manual_heuristic"),
)  # fmt: skip


def main() -> None:
    """Link the arms, cost them, and write the per-cell costs and the bootstrap table."""
    for arm, source in SOURCES.items():
        link = LINKS / "predictions" / arm
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source)
    (LINKS / "scores").mkdir(parents=True, exist_ok=True)
    build_mod.WORK_DIR = LINKS  # ty: ignore[invalid-assignment]
    cur.ARMS = ARMS  # ty: ignore[invalid-assignment]
    cur.COMPARISONS = COMPARISONS  # ty: ignore[invalid-assignment]
    c449.ARMS = ARMS
    c449.REFERENCES = cur.REFERENCES
    c449.COMPARISONS = COMPARISONS
    c449.RESAMPLES = cur.RESAMPLES
    cost = cur.build()
    cost.write_parquet(LINKS / "scores" / "curtailment_cells.parquet")
    table = c449.bootstrap(cost=cost)
    table.write_csv(LINKS / "scores" / "curtailment_bootstrap.csv")
    with pl.Config(tbl_rows=100, tbl_width_chars=160):
        print(table.with_columns(pl.col("estimate", "low", "high").round(3)))


if __name__ == "__main__":
    main()
