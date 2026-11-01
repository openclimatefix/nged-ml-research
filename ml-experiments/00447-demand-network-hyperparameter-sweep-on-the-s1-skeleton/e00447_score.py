# ruff: noqa: E501
"""Score the sweep arms against S1 (no fit): the screen, the random search, and the confirmation.

Rows, row sets, groups, the 1.15 widening, and the quantile score are experiment 428's
(`e00428_score`). The reference is S1, whose stored forecasts experiments 428 (seeds 0 to 2) and 439
(seeds 3 to 5) wrote; `link` links them into this experiment's predictions folder. A fourth group,
`pooled 14`, adds the signed-MW six and the MVA-only eight block by block. The decision columns are
`gain` (S1 minus the arm, in CRPS points, so positive is better), `minimum_detectable_effect`
(1.96 x sqrt(2) x seed sd / sqrt(seeds), the seed sd from S1's six stored seeds on the same folds),
the q95 and q99 pinball differences, and the folds improved.

```bash
PYTHONPATH=<the experiment 397, 404, and 428 folders>:ml-experiments/00447-* DATA_PATH_INTERNAL=... \
  uv run --no-sync python ml-experiments/00447-*/e00447_score.py link
  uv run --no-sync python ml-experiments/00447-*/e00447_score.py score --stage screen
```
"""

import argparse
import math
from typing import Final

import e00428_score as score  # ty: ignore[unresolved-import]
import e00447_common as common
import e00447_run as run
import numpy as np
import polars as pl

POOLED: Final[str] = "pooled 14"
POOLED_FROM: Final[tuple[str, str]] = ("signed-MW six", "MVA-only eight")
GUARD_ROW_SETS: Final[tuple[str, ...]] = ("clean", "real_harm", "inj_harm")
"""Row sets on which the q95 and q99 pinball loss must not rise by more than the tail limit."""
STORED_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)


def link() -> None:
    """Link experiments 428's and 439's stored S1 forecasts into this predictions folder."""
    target = common.WORK_DIR / "predictions" / common.REFERENCE
    target.mkdir(parents=True, exist_ok=True)
    for source_dir in (common.E428_WORK, common.E439_WORK):  # 428 holds seeds 0-2, 439 holds 3-5
        for source in (source_dir / "predictions" / common.REFERENCE).glob("*.parquet"):
            link_path = target / source.name
            if not link_path.is_symlink():
                link_path.symlink_to(source)


def link_refit_reference() -> None:
    """Link the refitted reference (`X-ref`) forecasts of months without a stored S1 as S1's."""
    target = common.WORK_DIR / "predictions" / common.REFERENCE
    for source in (common.WORK_DIR / "predictions" / "X-ref").glob("*.parquet"):
        if source.name.split("_s")[0] not in common.STORED_MONTHS:
            link_path = target / source.name
            if not link_path.is_symlink():
                link_path.symlink_to(source)


def collect(*, months: tuple[str, ...], seeds: tuple[int, ...], arms: list[str]) -> pl.DataFrame:
    """Return the block table of S1 and `arms`, with the pooled group added."""
    score.ARMS = tuple(arms)  # ty: ignore[invalid-assignment]
    table = score.collect(months=list(months), seeds=seeds, factor=score.WIDENING)
    keys = ["arm", "month", "world", "row_set", "day"]
    pooled = (
        table.filter(pl.col("group").is_in(POOLED_FROM))
        .group_by(keys)
        .agg(pl.all().exclude(*keys, "group").sum())
        .with_columns(group=pl.lit(POOLED))
    )
    return pl.concat([table, pooled.select(table.columns)])


def minimum_detectable_effect(*, months: tuple[str, ...], n_seeds: int) -> pl.DataFrame:
    """Return the seed spread and minimum detectable effect from S1's six stored seeds."""
    months = tuple(m for m in months if m in common.STORED_MONTHS)  # six stored seeds exist there
    table = collect(months=months, seeds=STORED_SEEDS, arms=[common.REFERENCE])
    records = []
    for (row_set, group), part in table.group_by("row_set", "group"):
        per_fold = part.group_by("month").agg(
            pl.col("n").sum(), *[pl.col(f"score_s{s}").sum() for s in STORED_SEEDS]
        )
        values = np.array(
            [
                [r[f"score_s{s}"] / r["n"] * score.POINTS for s in STORED_SEEDS]
                for r in per_fold.iter_rows(named=True)
            ]
        )
        folds, seeds = values.shape
        residual = values - values.mean(axis=1, keepdims=True)
        sd = math.sqrt((residual**2).sum() / (folds * (seeds - 1)))
        records.append(
            {"row_set": row_set, "group": group, "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)}
        )  # fmt: skip
    return pl.DataFrame(records)


def decisions(
    *, table: pl.DataFrame, arms: list[str], mde: pl.DataFrame, rng: np.random.Generator
) -> pl.DataFrame:
    """Return one row per arm, row set, and group: gain, interval, folds improved, tail differences."""
    score.CONTRASTS = {a: (a, common.REFERENCE) for a in arms}  # ty: ignore[invalid-assignment]
    contrasts = score.contrast_table(table=table, rng=rng)
    absolute = score.summarise(table=table, seeds=())
    reference = absolute.filter(pl.col("arm") == common.REFERENCE).select(
        "row_set", "group", *[f"pinball_{t}" for t in ("q95", "q99")]
    )
    arm_tails = absolute.select(
        pl.col("arm").alias("contrast"), "row_set", "group", "crps",
        pl.col("pinball_q95").alias("arm_q95"), pl.col("pinball_q99").alias("arm_q99"),
    )  # fmt: skip
    return (
        contrasts.join(mde, on=["row_set", "group"])
        .join(arm_tails, on=["contrast", "row_set", "group"])
        .join(reference, on=["row_set", "group"])
        .with_columns(
            gain=-pl.col("difference"),
            gain_low=-pl.col("high"),
            gain_high=-pl.col("low"),
            q95_change=pl.col("arm_q95") - pl.col("pinball_q95"),
            q99_change=pl.col("arm_q99") - pl.col("pinball_q99"),
            folds_improved=pl.col("fold_differences").map_elements(
                lambda d: sum(v is not None and v < 0 for v in d.values()), return_dtype=pl.Int64
            ),
            margin=pl.max_horizontal(pl.lit(common.CRPS_MARGIN), pl.col("mde")),
        )
        .rename({"contrast": "arm"})
        .drop("difference", "low", "high", "pinball_q95", "pinball_q99")
    )


def verdicts(*, frame: pl.DataFrame, min_folds: int) -> pl.DataFrame:
    """Return per arm: does the clean pooled gain pass the margin, and do the tail guards hold."""
    clean = frame.filter((pl.col("row_set") == "clean") & (pl.col("group") == POOLED)).select(
        "arm", "gain", "gain_low", "gain_high", "margin", "mde", "folds_improved"
    )
    guard = (
        frame.filter((pl.col("group") == POOLED) & pl.col("row_set").is_in(GUARD_ROW_SETS))
        .group_by("arm")
        .agg(
            worst_q95=pl.col("q95_change").max(), worst_q99=pl.col("q99_change").max(),
        )
    )  # fmt: skip
    return (
        clean.join(guard, on="arm")
        .with_columns(
            tails_ok=(pl.col("worst_q95") <= common.TAIL_HARM_LIMIT)
            & (pl.col("worst_q99") <= common.TAIL_HARM_LIMIT),
            moved=(pl.col("gain") >= pl.col("margin")) & (pl.col("folds_improved") >= min_folds),
            inside_mde=pl.col("gain").abs() < pl.col("mde"),
        )
        .with_columns(passes=pl.col("moved") & pl.col("tails_ok"))
        .sort("gain", descending=True)
    )


def main() -> None:
    """Link the stored forecasts, or score the arms of a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("link", "score"))
    parser.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--arms", nargs="+", default=None, help="short names; default all on disk")
    args = parser.parse_args()
    score.pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "link":
        link()
        link_refit_reference()
        return
    months = common.SCREEN_FOLDS if args.stage == "screen" else common.CONFIRM_FOLDS
    seeds = common.SCREEN_SEEDS if args.stage == "screen" else common.CONFIRM_SEEDS
    names = args.arms or [
        p.name.removeprefix(common.ARM_PREFIX)
        for p in sorted((run.WORK_DIR / "predictions").glob(f"{common.ARM_PREFIX}*"))
        if p.name != common.ARM_PREFIX + "ref"
    ]
    arms = [common.ARM_PREFIX + n for n in names]
    out = run.WORK_DIR / "scores" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    table = collect(months=months, seeds=seeds, arms=[common.REFERENCE, *arms])
    counts = table.group_by("arm", "month", "world", "row_set", "group").agg(pl.col("n").sum())
    assert (
        counts.group_by("month", "world", "row_set", "group").agg(pl.col("n").n_unique())["n"].max()
        == 1
    ), "an arm is scored on different rows from S1"
    days = table.group_by("arm", "month", "world", "row_set", "group").agg(pl.col("day").n_unique())
    assert (
        days.group_by("month", "world", "row_set", "group")
        .agg(pl.col("day").n_unique())["day"]
        .max()
        == 1
    ), "an arm is scored on different days from S1"
    mde = minimum_detectable_effect(months=months, n_seeds=len(seeds))
    frame = decisions(table=table, arms=arms, mde=mde, rng=np.random.default_rng(0))
    table.write_parquet(out / "blocks.parquet")
    mde.write_parquet(out / "mde.parquet")
    frame.write_parquet(out / "decisions.parquet")
    absolute = score.summarise(table=table, seeds=seeds)
    absolute.write_parquet(out / "absolute.parquet")
    verdict = verdicts(frame=frame, min_folds=2 if args.stage == "screen" else 3)
    verdict.write_parquet(out / "verdicts.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=20, fmt_str_lengths=40):
        print(mde.sort("row_set", "group"))
        print(verdict)
        print(frame.drop("fold_differences").sort("row_set", "group", "arm"))


if __name__ == "__main__":
    main()
