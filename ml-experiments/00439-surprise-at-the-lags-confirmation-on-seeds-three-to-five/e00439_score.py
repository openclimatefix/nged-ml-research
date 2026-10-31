"""Score S1 against R-inj on seeds 3 to 5, on seeds 0 to 2, and pooled over seeds 0 to 5 (no fit).

Experiment 428's stored forecasts for seeds 0 to 2 are linked into this experiment's predictions
folder by `link`; seeds 3 to 5 are this experiment's own. Rows, row sets, groups, the 1.15
widening, the block bootstrap, and the minimum detectable effect are experiment 428's
(`e00428_score`). Besides the pooled contrast, the output has each fold's contrast and each seed's
contrast, so a fold or seed that carries the result is visible.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00439-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00439-*/e00439_score.py link
  uv run --no-sync python ml-experiments/00439-*/e00439_score.py score
```
"""

import argparse
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00439_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SEED_SETS: Final[dict[str, tuple[int, ...]]] = {
    "seeds 3-5": (3, 4, 5),
    "seeds 0-2": (0, 1, 2),
    "seeds 0-5": (0, 1, 2, 3, 4, 5),
}
CONTRASTS: Final[dict[str, tuple[str, str]]] = {"C12 S1 - R-inj": ("S1", "R-inj")}


def link() -> None:
    """Link experiment 428's seed 0 to 2 forecasts of S1 and R-inj into this predictions folder."""
    for arm in run.ARMS:
        target = run.WORK_DIR / "predictions" / arm
        target.mkdir(parents=True, exist_ok=True)
        for source in (data.WORK_DIR / "predictions" / arm).glob("*.parquet"):
            if int(source.stem.split("_s")[1].split("_")[0]) < run.FROZEN_SEEDS:
                link_path = target / source.name
                if not link_path.is_symlink():
                    link_path.symlink_to(source)


def main() -> None:
    """Link the seed 0 to 2 forecasts, or score each seed set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--raw", action="store_true", help="no 1.15 widening")
    args = parser.parse_args()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        return
    score.ARMS = run.ARMS[::-1]  # ty: ignore[invalid-assignment]
    score.CONTRASTS = CONTRASTS  # ty: ignore[invalid-assignment]
    out = run.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    factor = 1.0 if args.raw else score.WIDENING
    for name, seeds in SEED_SETS.items():
        table = score.collect(months=args.months, seeds=seeds, factor=factor)
        counts = table.group_by("arm", "month", "world", "row_set", "group").agg(pl.col("n").sum())
        assert (
            counts.group_by("month", "world", "row_set", "group")
            .agg(pl.col("n").n_unique())["n"]
            .max()
            == 1
        ), "S1 and R-inj are scored on different rows"
        tag = name.replace(" ", "_")
        table.write_parquet(out / f"blocks_{tag}.parquet")
        with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60):
            print(f"===== {name} =====")
            absolute = score.summarise(table=table, seeds=seeds)
            absolute.write_parquet(out / f"absolute_{tag}.parquet")
            print(absolute.sort("row_set", "group", "arm"))
            mde = score.minimum_detectable_effect(table=table, seeds=seeds)
            mde.write_parquet(out / f"mde_{tag}.parquet")
            print(mde.sort("row_set", "group"))
            contrasts = score.contrast_table(table=table, rng=np.random.default_rng(0))
            contrasts.write_parquet(out / f"contrasts_{tag}.parquet")
            folds = (
                contrasts.join(mde, on=["row_set", "group"])
                .with_columns(
                    folds_negative=pl.col("fold_differences").map_elements(
                        lambda d: sum(v is not None and v < 0 for v in d.values()),
                        return_dtype=pl.Int64,
                    )
                )
                .sort("row_set", "group")
            )
            print(folds)
            for seed in seeds:
                single = score.collect(months=args.months, seeds=(seed,), factor=factor)
                one = score.contrast_table(table=single, rng=np.random.default_rng(0))
                print(f"-- seed {seed} alone")
                print(one.drop("fold_differences").sort("row_set", "group"))


if __name__ == "__main__":
    main()
