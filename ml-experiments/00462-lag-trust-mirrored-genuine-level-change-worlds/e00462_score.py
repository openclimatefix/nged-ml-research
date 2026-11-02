# ruff: noqa: E501
"""Score S3-amb, S3-rw-amb, and (if fitted) S3-attn-amb beside U1n-2, S1, and R-inj (no fit).

Rows, row sets, groups, lead bands, the 1.15 widening, the month-block bootstrap, and the NRA
corrected real-event rows are experiment 446's (`e00446_score`), reused unchanged. This module
redirects its forecast folder to this experiment's, adds the row sets that split the held-out level
changes by what a neighbour shows, and adds the contrasts and the minimum detectable effect that the
plan registers.

**Added row sets** (worlds 7 to 9, `plain` rows as in experiment 446):

- `gen_nomirror`: the newest lag with a level change lies inside an own event with no mirror.
- `gen_mirror_donor`: inside an own event whose load-conserving mirror is at 1 or 2 neighbours. A
  mirror looks like a load shift at the neighbours, so the plan scores these apart.
- `gen_recipient`: inside a mirror received from a neighbour's event (the recipient's own step is a
  level change whose target moves, with an opposite-signed step at the donor).

**Contrasts** (arm minus arm, CRPS and tail pinball in points of effective capacity, block-bootstrap
95% intervals; the Bonferroni count is stated in the plan):

- `S3-amb - S3-rw-amb` on `gen_nomirror`, `gen_mirror_donor`, `gen_jump`, and `inj_harm`.
- `S3-amb - U1n-2` and `S3-rw-amb - U1n-2` on `clean`, `real_harm`, `real_event_nra100`, and the
  held-out sets. Also against `S1` and `R-inj`, as the stated production cost.

**Minimum detectable effect** of the pair, from the seed-0 against seed-1 difference of each arm per
fold: `1.96 * sqrt(2) * sd / sqrt(n_seeds)`, as experiment 428 does for R-inj.

```bash
source /tmp/pp462.sh; export DATA_PATH_INTERNAL=<private data folder>
PYTHONPATH=$PP uv run --no-sync python ml-experiments/00462-*/e00462_score.py score --variants raw w1.15
```
"""

import argparse
import math
from typing import Final

import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00446_score as sc  # ty: ignore[unresolved-import]
import e00462_run as run
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("R-inj", "S1", "U1n-2", "S3", "S3-rw", *run.ARM_NAMES)
NEW: Final[tuple[str, ...]] = (*run.ARM_NAMES, "S3-attn-amb")
MISSING_OK: Final[tuple[str, ...]] = ("Bmix", "R", "S3", "S3-rw")
"""Arms with stored forecasts on worlds 0, 4, 5, and 6 only (no forecasts on the new held-out worlds)."""
SCORE_DIR = run.WORK_DIR / "scores"
GEN_SETS: Final[tuple[str, ...]] = ("gen_nomirror", "gen_mirror_donor", "gen_recipient", "gen_jump")
PAIR_SETS: Final[tuple[str, ...]] = (
    "gen_nomirror", "gen_mirror_donor", "gen_recipient", "gen_jump", "inj_harm", "inj_clean",
    "gen_clean", "clean", "real_harm", "real_event_nra100", "real_event_nra100_wo_largest",
)  # fmt: skip


def row_sets() -> list[sc.RowSet]:
    """Return experiment 446's row sets plus the three that split the held-out level changes."""
    plain = ~sc._f("win_real") & ~sc._f("tgt_post")
    kind = pl.col("level_kind").fill_null(0)
    extra = [
        sc.RowSet("gen_nomirror", (7, 8, 9), lambda: sc._f("shift_lag") & plain & (kind == 1)),
        sc.RowSet("gen_mirror_donor", (7, 8, 9), lambda: sc._f("shift_lag") & plain & (kind == 2)),
        sc.RowSet("gen_recipient", (7, 8, 9), lambda: sc._f("shift_lag") & plain & (kind == 3)),
    ]  # fmt: skip
    return [*ORIGINAL_ROW_SETS(), *extra]


ORIGINAL_ROW_SETS = sc.row_sets


ORIGINAL_GEN_COLUMNS = sc.with_gen_columns


def with_gen_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add a null `level_kind` to arms with no forecasts on worlds 7 to 9, then experiment 446's columns."""
    if "level_kind" not in frame.columns:
        frame = frame.with_columns(level_kind=pl.lit(None, dtype=pl.Int8))
    return ORIGINAL_GEN_COLUMNS(frame=frame)


def install() -> None:
    """Point experiment 446's scorer at this experiment's forecasts, row sets, and tolerant arms."""
    run.install()
    pr.PRED_DIR = run.PRED_DIR
    sc.row_sets = row_sets
    sc.with_gen_columns = with_gen_columns
    sc.MISSING_OK_ON_GEN = MISSING_OK  # ty: ignore[invalid-assignment]
    sc.run.GEN_WORLDS = run.GEN_WORLDS
    sc.NEW_ARMS = NEW  # ty: ignore[invalid-assignment]
    sc.SCORE_DIR = SCORE_DIR


def pair_mde(*, table: pl.DataFrame, arms: tuple[str, ...], seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return the minimum detectable effect from the seed spread of the given arms, per row set and group."""
    records = []
    for (row_set, group), part in table.filter(pl.col("arm").is_in(list(arms))).group_by(
        "row_set", "group"
    ):
        values = []
        for arm in arms:
            per_fold = (
                part.filter(pl.col("arm") == arm)
                .group_by("month")
                .agg(pl.col("n").sum(), *[pl.col(f"score_s{s}").sum() for s in seeds])
                .sort("month")
            )
            values.append(
                np.array(
                    [[r[f"score_s{s}"] / r["n"] * sc.POINTS for s in seeds]
                     for r in per_fold.iter_rows(named=True)]
                )
            )  # fmt: skip
        stacked = np.concatenate(values, axis=0)
        residual = stacked - stacked.mean(axis=1, keepdims=True)
        folds, n_seeds = stacked.shape
        sd = math.sqrt((residual**2).sum() / (folds * (n_seeds - 1)))
        records.append(
            {"row_set": row_set, "group": group, "seed_sd": sd,
             "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(n_seeds)}
        )  # fmt: skip
    return pl.DataFrame(records)


def report(*, table: pl.DataFrame, variant: str, seeds: tuple[int, ...]) -> None:
    """Print and store the absolute scores, the pair's minimum detectable effect, and the contrasts."""
    out = SCORE_DIR / variant.replace(".", "_")
    out.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out / "blocks.parquet")
    arms_present = set(table["arm"].unique().to_list())
    with pl.Config(tbl_rows=400, tbl_cols=30, fmt_str_lengths=50, fmt_float="mixed"):
        print(f"===== variant {variant} =====")
        absolute = sc.summarise(table=table, by=["arm", "row_set", "group"])
        absolute.write_parquet(out / "absolute.parquet")
        print(absolute.sort("row_set", "group", "arm"))
        if len(seeds) >= 2 and set(run.ARM_NAMES) <= arms_present:
            mde = pair_mde(table=table, arms=run.ARM_NAMES, seeds=seeds)
            mde.write_parquet(out / "mde_pair.parquet")
            print(mde.sort("row_set", "group"))
        pairs = [("S3-amb", "S3-rw-amb")]
        for arm in (a for a in NEW if a in arms_present):
            pairs += [(arm, ref) for ref in ("U1n-2", "S1", "R-inj") if ref in arms_present]
        sets = [s for s in PAIR_SETS if not table.filter(pl.col("row_set") == s).is_empty()]
        diffs = sc.contrasts(table=table, pairs=pairs, row_sets_=sets, rng=np.random.default_rng(0))
        if not diffs.is_empty():
            diffs.write_parquet(out / "contrasts.parquet")
            print(diffs.drop("fold_differences").sort("row_set", "group", "metric", "contrast"))
        ratios = pl.concat(
            [
                sc.width_ratios(table=table, numerator=n, denominator=d)
                for n, d in (("gen_nomirror", "gen_clean"), ("gen_mirror_donor", "gen_clean"),
                             ("gen_recipient", "gen_clean"), ("inj_harm", "inj_clean"),
                             ("real_harm", "clean"), ("gen_clean", "clean"))
                if not table.filter(pl.col("row_set") == n).is_empty()
            ]
        )  # fmt: skip
        ratios.write_parquet(out / "width_ratios.parquet")
        print(ratios.sort("row_set", "group", "arm"))


def main() -> None:
    """Score the requested variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score",))
    parser.add_argument("--variants", nargs="+", default=["raw", "w1.15"], choices=("raw", "w1.15"))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    install()
    context = sc.Context(need_cold=False, need_nra=True)
    seeds = tuple(args.seeds)
    arms = [a for a in (*ARMS, "S3-attn-amb") if (run.PRED_DIR / a).exists()]
    print(f"arms scored: {', '.join(arms)}")
    for variant in args.variants:
        table = sc.collect(
            arms=arms, months=sorted(args.months), seeds=seeds, variant=variant, context=context
        )
        report(table=table, variant=variant, seeds=seeds)


if __name__ == "__main__":
    main()
