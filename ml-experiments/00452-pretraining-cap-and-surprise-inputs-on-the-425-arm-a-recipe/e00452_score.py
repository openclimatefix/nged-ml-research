# ruff: noqa: E501
"""Score the arms of experiment 452 (no fit).

Stored forecasts are linked into this experiment's predictions folder: S1 seeds 0 to 2 from
experiment 428 and `PT0`, which is experiment 448's `S1PT` (seeds 0 to 2). Rows, row sets, groups,
the 1.15 widening, the block bootstrap, and the minimum detectable effect are experiment 428's
(`e00428_score`); the minimum-detectable-effect function is copied from experiment 448.

```bash
source /tmp/e452_env.sh
uv run --no-sync python ml-experiments/00452-*/e00452_score.py link
uv run --no-sync python ml-experiments/00452-*/e00452_score.py score
```
"""

import argparse
import math
from pathlib import Path
from typing import Final

import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00452_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS: Final[tuple[str, ...]] = ("S1", "PT1", "AS1L", "AL", "PT1-avg5", "AS1L-avg5", "AL-avg5")
TESTS: Final[int] = 26
"""Bonferroni count of the 449 audit, kept so the intervals are comparable."""
RESAMPLES: Final[int] = 20_000
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 AS1L - PT1 (cap only)": ("AS1L", "PT1"),
    "C2 AL - PT1 (surprise inputs and cap)": ("AL", "PT1"),
    "C3 AS1L - AL (surprise inputs)": ("AS1L", "AL"),
    "C4 PT1 - S1": ("PT1", "S1"),
    "C5 AS1L-avg5 - PT1-avg5 (cap only)": ("AS1L-avg5", "PT1-avg5"),
    "C6 AL-avg5 - PT1-avg5": ("AL-avg5", "PT1-avg5"),
    "C7 AS1L-avg5 - AL-avg5": ("AS1L-avg5", "AL-avg5"),
}
"""Contrasts scored here; contrasts against the stored references are scored by `e00452_refs`."""
SEED_SETS: Final[dict[str, tuple[int, ...]]] = {"seeds 0-4": (0, 1, 2, 3, 4)}
S1_439: Final[str] = "00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five"


def contrast_table(*, table: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return each contrast with the 95% and the Bonferroni block-bootstrap intervals (points).

    Experiment 428's blocks (one fold and one origin day), with `RESAMPLES` draws, and a second
    interval at level `1 - 0.05 / TESTS`.
    """
    keys = ["row_set", "group"]
    alpha = 100.0 * 0.05 / TESTS / 2.0
    records = []
    for name, (a, b) in score.CONTRASTS.items():
        left = table.filter(pl.col("arm") == a).select(*keys, "month", "world", "day", "n", "score")
        right = table.filter(pl.col("arm") == b).select(
            *keys, "month", "world", "day", pl.col("score").alias("score_b")
        )
        joined = left.join(right, on=[*keys, "month", "world", "day"], how="inner")
        for (row_set, group), part in joined.group_by(keys):
            blocks = part.group_by("month", "day").agg(
                pl.col("n").sum(), (pl.col("score") - pl.col("score_b")).sum().alias("diff")
            )
            n, diff = blocks["n"].to_numpy().astype(float), blocks["diff"].to_numpy()
            mean = diff.sum() / n.sum() * score.POINTS
            draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
            boot = diff[draws].sum(1) / n[draws].sum(1) * score.POINTS
            by_month = part.group_by("month").agg(
                pl.col("n").sum(), (pl.col("score") - pl.col("score_b")).sum().alias("diff")
            )
            records.append(
                {
                    "contrast": name, "row_set": row_set, "group": group, "n": int(n.sum()),
                    "difference": mean, "low": float(np.percentile(boot, 2.5)),
                    "high": float(np.percentile(boot, 97.5)),
                    "low_bonferroni": float(np.percentile(boot, alpha)),
                    "high_bonferroni": float(np.percentile(boot, 100.0 - alpha)),
                    "fold_differences": {
                        r["month"]: r["diff"] / r["n"] * score.POINTS
                        for r in by_month.iter_rows(named=True)
                    },
                }
            )  # fmt: skip
    return pl.DataFrame(records)


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
    """Link stored forecasts into this experiment's folder: `PT1` (449) and `S1` (439), seeds 0 to 4."""
    sources = {
        "PT1": [run.lagless.SOURCE_449 / "predictions" / "PT1"],
        "S1": [ML_EXPERIMENTS_DATA_DIR / S1_439 / "work" / "predictions" / "S1"],
    }
    for arm, folders in sources.items():
        target = run.WORK_DIR / "predictions" / arm
        target.mkdir(parents=True, exist_ok=True)
        for source in (f for folder in folders for f in Path(folder).glob("*.parquet")):
            if int(source.stem.split("_s")[1].split("_")[0]) in (0, 1, 2, 3, 4):
                link_path = target / source.name
                if not link_path.is_symlink():
                    link_path.symlink_to(source)


def main() -> None:
    """Link the stored forecasts, or score each seed set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--raw", action="store_true", help="no 1.15 widening")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--tag", default="", help="suffix of the output files")
    parser.add_argument("--factor", type=float, default=None, help="widening factor")
    args = parser.parse_args()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        return
    arms = tuple(args.arms)
    score.ARMS = arms[::-1]  # ty: ignore[invalid-assignment]
    score.CONTRASTS = {  # ty: ignore[invalid-assignment]
        k: v for k, v in CONTRASTS.items() if v[0] in arms and v[1] in arms
    }
    out = run.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    factor = args.factor or (1.0 if args.raw else score.WIDENING)
    for name, seeds in SEED_SETS.items():
        table = score.collect(months=args.months, seeds=seeds, factor=factor)
        counts = table.group_by("arm", "month", "world", "row_set", "group").agg(pl.col("n").sum())
        assert (
            counts.group_by("month", "world", "row_set", "group")
            .agg(pl.col("n").n_unique())["n"]
            .max()
            == 1
        ), "arms are scored on different rows"
        tag = name.replace(" ", "_") + ("_raw" if args.raw else "") + args.tag
        table.write_parquet(out / f"blocks_{tag}.parquet")
        with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=60):
            print(f"===== {name} =====")
            absolute = score.summarise(table=table, seeds=seeds)
            absolute.write_parquet(out / f"absolute_{tag}.parquet")
            print(absolute.sort("row_set", "group", "arm"))
            mde = minimum_detectable_effect(table=table, seeds=seeds)
            mde.write_parquet(out / f"mde_{tag}.parquet")
            print(mde.sort("row_set", "group"))
            contrasts = contrast_table(table=table, rng=np.random.default_rng(0))
            contrasts.write_parquet(out / f"contrasts_{tag}.parquet")
            print(contrasts.drop("fold_differences").sort("row_set", "group"))


if __name__ == "__main__":
    main()
