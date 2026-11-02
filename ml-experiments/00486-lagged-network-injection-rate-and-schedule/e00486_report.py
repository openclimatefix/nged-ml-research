"""Print the numbers experiment 486's write-up quotes from stored scores (no fit).

Tables: the minimum detectable effect from the stored seeds at 2026-06; the pooled contrasts at the
signed-MW six on rows with no labelled event and injected harm rows (from `e00404_pool.py` run on
this experiment's folders); the encroachment sweep (median shift as a share of a 5%-of-capacity
step, mean of the months, seeds within a month averaged first); the bad-data injections; and the
tail table. Everything is also written to `work/scores/report.txt`.

```bash
PYTHONPATH=<every folder under ml-experiments/> \
    uv run --no-sync python ml-experiments/00486-*/e00486_report.py
```
"""

from pathlib import Path

import e00486_fit as fit486  # ty: ignore[unresolved-import]
import polars as pl

WORK = fit486.WORK_DIR
ARMS = ["A0-inj", "R05", "R50", "Cur-down", "Cur-up"]


def _month(*, path: str) -> str:
    return Path(path).name[:7]


def mde() -> pl.DataFrame:
    """Return 2.8 times the seed standard deviation of the 2026-06 absolute score at the six."""
    absolute = pl.read_parquet(WORK / "pooled_absolute.parquet").filter(
        (pl.col("month") == "2026-06")
        & (pl.col("group") == "signed_six")
        & pl.col("cls").is_in(["clean_real", "harm_inj"])
        & pl.col("arm").is_in([*ARMS, "B-mix"])
    )
    return (
        absolute.group_by("arm", "cls")
        .agg(pl.col("score").mean().alias("mean"), (2.8 * pl.col("score").std()).alias("mde"))
        .sort("cls", "arm")
    )


def contrasts() -> pl.DataFrame:
    """Return the pooled contrasts of the four new arms against both references."""
    parts = []
    for reference in ("A0-inj", "B-mix"):
        frame = pl.read_csv(WORK / f"pooled_summary_vs_{reference}.csv").filter(
            (pl.col("group") == "signed_six")
            & pl.col("cls").is_in(["clean_real", "harm_inj", "harm_real"])
            & ~pl.col("arm").str.contains(r"\+")
            & (pl.col("arm") != reference)
        )
        parts.append(frame.with_columns(reference=pl.lit(reference)))
    return (
        pl.concat(parts)
        .select("reference", "cls", "arm", "absolute", "minus_ref", "low", "high", "cells_better")
        .sort("reference", "cls", "arm")
    )


def sweep() -> pl.DataFrame:
    """Return the sweep share by arm, number of lags, and lead band."""
    frames = [
        pl.read_parquet(p).with_columns(month=pl.lit(_month(path=str(p))))
        for p in sorted((WORK / "sweep").glob("*.parquet"))
    ]
    data = pl.concat(frames).filter(~pl.col("arm").str.ends_with("masked"))
    return (
        data.group_by("arm", "k", "lead_band", "month")
        .agg(pl.col("shift").mean())
        .group_by("arm", "k", "lead_band")
        .agg(pl.col("shift").mean().alias("share"), pl.col("shift").min().alias("lowest_month"),
             pl.col("shift").max().alias("highest_month"))
        .sort("lead_band", "k", "arm")
    )  # fmt: skip


def bad() -> pl.DataFrame:
    """Return the median shift as a share of the corruption, by arm and variant."""
    frames = [
        pl.read_parquet(p).with_columns(month=pl.lit(_month(path=str(p))))
        for p in sorted((WORK / "bad").glob("*.parquet"))
    ]
    data = pl.concat(frames).filter(
        pl.col("shift").is_finite()
        & pl.col("corruption").is_finite()
        & (pl.col("corruption").abs() > 1e-4)
    )
    per_month = data.group_by("arm", "variant", "month").agg(
        (
            (pl.col("shift") * pl.col("corruption").sign()).sum() / pl.col("corruption").abs().sum()
        ).alias("share")
    )
    return (
        per_month.group_by("arm", "variant")
        .agg(pl.col("share").mean(), pl.col("share").min().alias("lowest_month"),
             pl.col("share").max().alias("highest_month"))
        .sort("variant", "arm")
    )  # fmt: skip


def main() -> None:
    """Print and store every table."""
    tables = {
        "Minimum detectable effect (2.8 x seed SD, 2026-06, signed-MW six)": mde(),
        "Pooled contrasts at the signed-MW six (points of capacity)": contrasts(),
        "Sweep: median shift as a share of a 5% step": sweep(),
        "Bad-data injections: median shift as a share of the corruption": bad(),
    }
    out = []
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=160, float_precision=3):
        for name, table in tables.items():
            out.append(f"## {name}\n{table}\n")
    text = "\n".join(out)
    print(text)
    (WORK / "scores").mkdir(exist_ok=True)
    (WORK / "scores" / "report.txt").write_text(text)


if __name__ == "__main__":
    main()
