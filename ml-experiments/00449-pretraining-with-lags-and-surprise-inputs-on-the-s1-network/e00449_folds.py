"""Fold-by-fold contrasts, leave-one-fold-out, and the Christmas slice for experiment 449 (no fit).

Reads the block table written by `e00449_score.py` (five folds) for the per-fold and leave-one-fold-
out differences on the injected-world clean rows, and scores 22 to 31 December of the 2025-12 fold
(valid times 2025-12-22 00:00 to 2026-01-01 00:00 UTC, days 1 to 10) for the arms and references
through experiment 448's reference machinery.

```bash
source /tmp/e449_env.sh
uv run --no-sync python ml-experiments/00449-*/e00449_folds.py
```
"""

from datetime import UTC, datetime
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00449_refs as refs449  # ty: ignore[unresolved-import]
import e00449_run as run  # ty: ignore[unresolved-import]
import polars as pl

CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "PT1 - S1": ("PT1", "S1"),
    "PT1-pool5 - S1-pool5": ("PT1-pool5", "S1-pool5"),
    "PT1-pool5 - PT1": ("PT1-pool5", "PT1"),
}
FOUR: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
ALL: Final[tuple[str, ...]] = (*FOUR, "2025-12")
CHRISTMAS: Final[tuple[datetime, datetime]] = (
    datetime(2025, 12, 22, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
)
ARMS: Final[tuple[str, ...]] = (
    "S1",
    "PT0",
    "PT1",
    "PT1-full",
    "PT1b",
    "PT2",
    "S1-pool5",
    "PT1-pool5",
)


def per_fold(*, blocks: pl.DataFrame) -> None:
    """Print each contrast by fold and with each fold left out (points, injected clean rows)."""
    clean = blocks.filter(pl.col("row_set") == "inj_clean")
    for name, (a, b) in CONTRASTS.items():
        left = clean.filter(pl.col("arm") == a).select(
            "group", "month", "world", "day", "n", "score"
        )
        right = clean.filter(pl.col("arm") == b).select(
            "group", "month", "world", "day", pl.col("score").alias("score_b")
        )
        joined = left.join(right, on=["group", "month", "world", "day"]).with_columns(
            diff=pl.col("score") - pl.col("score_b")
        )
        rows = []
        for group in joined["group"].unique().sort().to_list():
            part = joined.filter(pl.col("group") == group)
            by_month = part.group_by("month").agg(pl.col("n").sum(), pl.col("diff").sum())
            record = {"contrast": name, "group": group}
            for r in by_month.iter_rows(named=True):
                record[r["month"]] = 100.0 * r["diff"] / r["n"]
            four = by_month.filter(pl.col("month").is_in(FOUR))
            record["four folds"] = 100.0 * four["diff"].sum() / four["n"].sum()  # ty: ignore[unsupported-operator]
            record["five folds"] = 100.0 * by_month["diff"].sum() / by_month["n"].sum()  # ty: ignore[unsupported-operator]
            for month in FOUR:
                rest = four.filter(pl.col("month") != month)
                record[f"four without {month}"] = 100.0 * rest["diff"].sum() / rest["n"].sum()  # ty: ignore[unsupported-operator]
            rows.append(record)
        with pl.Config(tbl_cols=20, tbl_width_chars=250):
            print(pl.DataFrame(rows))


def christmas() -> None:
    """Print days 1 to 10 CRPS of 22 to 31 December 2025 for the arms and references, by group."""
    base.run = run  # ty: ignore[invalid-assignment]
    series = adapter.series_table()
    refs = refs449.reference_frames(series=series)
    tables, _, _ = base.build_month(month="2025-12", arms=ARMS, refs=refs, seeds=(0, 1, 2))
    out = []
    for name, table in tables.items():
        for label, lo, hi in (
            ("22 to 31 December", CHRISTMAS[0], CHRISTMAS[1]),
            ("1 to 21 December", datetime(2025, 12, 1, tzinfo=UTC), CHRISTMAS[0]),
        ):
            part = table.filter(
                (pl.col("valid_time") >= lo)
                & (pl.col("valid_time") < hi)
                & (pl.col("lead_hours") <= 240.0)
            )
            for group in ("signed-MW six", "MVA-only eight", "D15 and D18"):
                sub = part.filter(pl.col("group") == group)
                out.append(
                    {"arm": name, "slice": label, "group": group, "n": sub.height,
                     "crps": sub["crps"].mean()}
                )  # fmt: skip
            out.append(
                {"arm": name, "slice": label, "group": "all 16", "n": part.height,
                 "crps": part["crps"].mean()}
            )  # fmt: skip
    table = pl.DataFrame(out)
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        for label in ("22 to 31 December", "1 to 21 December"):
            print(label)
            print(
                table.filter(pl.col("slice") == label).pivot(on="group", index="arm", values="crps")
            )


def main() -> None:
    """Print the per-fold table and the Christmas slice."""
    blocks = pl.read_parquet(run.WORK_DIR / "scores" / "blocks_seeds_0-2_5m.parquet")
    per_fold(blocks=blocks)
    christmas()


if __name__ == "__main__":
    main()
