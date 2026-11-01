"""Experiment 435, step 4 (S3): does pooling three seeds widen the band on harm rows?

For each experiment 428 arm that has three seeds (`R-inj`, `E3`, `E3-surp`) and each metering group,
the script reports the mean 90% band width (q95 minus q05, share of capacity) on `inj_harm` rows
and on `clean` rows, for each single seed and for the seed-pooled quantiles (the quantiles averaged
over the three seeds and sorted), with the ratio of the two widths and the q95 coverage on both row
sets. Rows are classified as in `e00435_table_p`. Output: `work/s3_428.parquet`.

```bash
# PYTHONPATH as for e00435_table_p.py
uv run --no-sync python ml-experiments/00435-*/e00435_s3.py
```
"""

import e00435_table_p as tp  # ty: ignore[unresolved-import]
import polars as pl

ARMS = ("R-inj", "E3", "E3-surp")


def stats(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the band width and q95 coverage per group and class (clean, inj_harm)."""
    return (
        frame.filter(pl.col("cls").is_in(["clean", "inj_harm"]))
        .with_columns(width=pl.col("q95") - pl.col("q05"), hit=(pl.col("actual") <= pl.col("q95")))
        .group_by("group", "cls")
        .agg(pl.col("width").mean(), pl.col("hit").mean().alias("cov_q95"), pl.len().alias("n"))
    )


def main() -> None:
    """Compare one-seed and pooled bands on harm and clean rows."""
    powers = tp.world_powers()
    out: list[pl.DataFrame] = []
    for arm in ARMS:
        for month in tp.MONTHS:
            tables = {"pooled": stats(frame=tp.load_arm(arm=arm, month=month, powers=powers))}
            for seed in (0, 1, 2):
                tp.ARM_SEEDS[arm] = (seed,)
                tables[f"seed{seed}"] = stats(
                    frame=tp.load_arm(arm=arm, month=month, powers=powers)
                )
            tp.ARM_SEEDS[arm] = (0, 1, 2)
            for basis, table in tables.items():
                out.append(
                    table.with_columns(arm=pl.lit(arm), month=pl.lit(month), basis=pl.lit(basis))
                )
        print("done", arm, flush=True)
    result = pl.concat(out)
    result.write_parquet(tp.WORK_DIR / "s3_428.parquet")


if __name__ == "__main__":
    main()
