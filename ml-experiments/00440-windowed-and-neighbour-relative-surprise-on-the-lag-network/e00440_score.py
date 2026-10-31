"""Score S2, S3, and S3-rw beside S1 and R-inj on the same rows (no fit).

Experiment 428's stored forecasts of S1 and R-inj (seeds 0 to 2) are linked into this experiment's
predictions folder by `link`; S2, S3, and S3-rw are this experiment's own. Rows, row sets, groups,
the 1.15 widening, the block bootstrap, and the minimum detectable effect are experiment 428's
(`e00428_score`). The tail columns (pinball loss at q95 and q99) are in the absolute table.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00440-* DATA_PATH_INTERNAL=<private data folder> \
  uv run --no-sync python ml-experiments/00440-*/e00440_score.py link
  uv run --no-sync python ml-experiments/00440-*/e00440_score.py score
```
"""

import argparse
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00440_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

REFERENCES: Final[tuple[str, ...]] = ("S1", "R-inj")
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "W1 S2 - S1": ("S2", "S1"),
    "W2 S3 - S2": ("S3", "S2"),
    "W3 S3 - S3-rw": ("S3", "S3-rw"),
    "W4 S3 - S1": ("S3", "S1"),
    "W5 S3-rw - S2": ("S3-rw", "S2"),
}


def link() -> None:
    """Link experiment 428's stored forecasts of S1 and R-inj into this predictions folder."""
    target = run.WORK_DIR / "predictions"
    target.mkdir(parents=True, exist_ok=True)
    for arm in REFERENCES:
        if not (target / arm).is_symlink():
            (target / arm).symlink_to(data.WORK_DIR / "predictions" / arm, target_is_directory=True)


def main() -> None:
    """Link the reference forecasts, or score the contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--raw", action="store_true", help="no 1.15 widening")
    args = parser.parse_args()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        return
    arms = (*REFERENCES, *run.ARM_NAMES)
    score.ARMS = arms  # ty: ignore[invalid-assignment]
    score.CONTRASTS = CONTRASTS  # ty: ignore[invalid-assignment]
    out = run.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    factor = 1.0 if args.raw else score.WIDENING
    seeds = tuple(args.seeds)
    table = score.collect(months=args.months, seeds=seeds, factor=factor)
    counts = table.group_by("arm", "month", "world", "row_set", "group").agg(pl.col("n").sum())
    assert (
        counts.group_by("month", "world", "row_set", "group").agg(pl.col("n").n_unique())["n"].max()
        == 1
    ), "arms are scored on different rows"
    table.write_parquet(out / "blocks.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60):
        absolute = score.summarise(table=table, seeds=seeds)
        absolute.write_parquet(out / "absolute.parquet")
        print(absolute.sort("row_set", "group", "arm"))
        mde = score.minimum_detectable_effect(table=table, seeds=seeds)
        mde.write_parquet(out / "mde.parquet")
        print(mde.sort("row_set", "group"))
        contrasts = score.contrast_table(table=table, rng=np.random.default_rng(0))
        contrasts.write_parquet(out / "contrasts.parquet")
        print(
            contrasts.join(mde, on=["row_set", "group"]).drop("fold_differences")
            .sort("row_set", "group")
        )  # fmt: skip


if __name__ == "__main__":
    main()
