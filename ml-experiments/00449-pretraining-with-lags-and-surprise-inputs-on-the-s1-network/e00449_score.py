"""Score the arms of experiment 449 (no fit).

Stored forecasts are linked into this experiment's predictions folder: S1 seeds 0 to 2 from
experiment 428 and `PT0`, which is experiment 448's `S1PT` (seeds 0 to 2). Rows, row sets, groups,
the 1.15 widening, the block bootstrap, and the minimum detectable effect are experiment 428's
(`e00428_score`); the minimum-detectable-effect function is copied from experiment 448.

```bash
source /tmp/e449_env.sh
uv run --no-sync python ml-experiments/00449-*/e00449_score.py link
uv run --no-sync python ml-experiments/00449-*/e00449_score.py score
```
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00449_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS: Final[tuple[str, ...]] = ("S1", "PT0", "PT1", "PT1-full", "PT1b", "PT2")
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 PT1 - S1": ("PT1", "S1"),
    "C2 PT1-full - S1": ("PT1-full", "S1"),
    "C3 PT2 - S1": ("PT2", "S1"),
    "C4 PT1 - PT0": ("PT1", "PT0"),
    "C5 PT2 - PT1": ("PT2", "PT1"),
    "C6 PT0 - S1": ("PT0", "S1"),
    "C7 PT1b - S1": ("PT1b", "S1"),
    "C8 PT1 - PT1b": ("PT1", "PT1b"),
    "C9 PT1-full - PT1": ("PT1-full", "PT1"),
}
SEED_SETS: Final[dict[str, tuple[int, ...]]] = {"seeds 0-2": (0, 1, 2)}
PT0_SOURCE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool/work/predictions/S1PT"
)


def minimum_detectable_effect(*, table: pl.DataFrame, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the minimum detectable effect from S1's seed spread (experiment 428's formula)."""
    records = []
    for (row_set, group), part in table.filter(pl.col("arm") == "S1").group_by("row_set", "group"):
        per_fold = part.group_by("month").agg(
            pl.col("n").sum(), *[pl.col(f"score_s{s}").sum() for s in seeds]
        )
        values = np.array(
            [
                [r[f"score_s{s}"] / r["n"] * score.POINTS for s in seeds]
                for r in per_fold.iter_rows(named=True)
            ]
        )
        folds, n_seeds = values.shape
        residual = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((residual**2).sum() / (folds * (n_seeds - 1)))
        records.append(
            {"row_set": row_set, "group": group, "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)}
        )  # fmt: skip
    return pl.DataFrame(records)


def link() -> None:
    """Link stored S1 (experiment 428) and PT0 (experiment 448's S1PT) forecasts, seeds 0 to 2."""
    sources = {"S1": data.WORK_DIR / "predictions" / "S1", "PT0": PT0_SOURCE}
    for arm, folder in sources.items():
        target = run.WORK_DIR / "predictions" / arm
        target.mkdir(parents=True, exist_ok=True)
        for source in Path(folder).glob("*.parquet"):
            if int(source.stem.split("_s")[1].split("_")[0]) in (0, 1, 2):
                link_path = target / source.name
                if not link_path.is_symlink():
                    link_path.symlink_to(source)


def main() -> None:
    """Link the stored forecasts, or score each seed set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--raw", action="store_true", help="no 1.15 widening")
    args = parser.parse_args()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        return
    score.ARMS = ARMS[::-1]  # ty: ignore[invalid-assignment]
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
        ), "arms are scored on different rows"
        tag = name.replace(" ", "_") + ("_raw" if args.raw else "")
        table.write_parquet(out / f"blocks_{tag}.parquet")
        with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60):
            print(f"===== {name} =====")
            absolute = score.summarise(table=table, seeds=seeds)
            absolute.write_parquet(out / f"absolute_{tag}.parquet")
            print(absolute.sort("row_set", "group", "arm"))
            mde = minimum_detectable_effect(table=table, seeds=seeds)
            mde.write_parquet(out / f"mde_{tag}.parquet")
            print(mde.sort("row_set", "group"))
            contrasts = score.contrast_table(table=table, rng=np.random.default_rng(0))
            contrasts.write_parquet(out / f"contrasts_{tag}.parquet")
            print(contrasts.drop("fold_differences").sort("row_set", "group"))


if __name__ == "__main__":
    main()
