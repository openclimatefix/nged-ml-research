"""Experiment 692: give every figure a random three-digit id so no file name reveals its source.

Run once after `e00692_run.py render`. Renames the PNG files in place and rewrites the answer key.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00692-event-table-reviewer-calibration-gate"


def main() -> None:
    """Rename each set's figures to `<set>_<NNN>` in a random order."""
    path = OUT / "answer_key_private" / "key.parquet"
    key = pl.read_parquet(path)
    rng = np.random.default_rng(6920)
    parts = []
    for fset in ("calibration", "realness"):
        sub = key.filter(pl.col("set") == fset)
        perm = rng.permutation(sub.height)
        new = [f"{fset}_{i:03d}" for i in perm]
        folder = OUT / "figures" / fset
        for old, n in zip(sub["figure"], new, strict=True):
            (folder / f"{old}.png").rename(folder / f"tmp_{n}.png")
        for n in new:
            (folder / f"tmp_{n}.png").rename(folder / f"{n}.png")
        parts.append(sub.with_columns(pl.Series("figure", new)))
    pl.concat(parts).write_parquet(path)


if __name__ == "__main__":
    main()
