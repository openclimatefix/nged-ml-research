"""Turn the sums into old and new means, per Primary and pooled per group and overall."""

from typing import Final

import polars as pl
from e00398_flagged_keys import OUT

NAMES: Final[tuple[str, ...]] = ("379_stack", "379_components", "391", "395")


def long(*, name: str) -> pl.DataFrame:
    """Return per-Primary old and new means, long over metric, with the dropped-row count."""
    s = pl.read_parquet(OUT / f"sums_{name}.parquet")
    metrics = sorted({c.split("__", 1)[1] for c in s.columns if c.startswith("old__")})
    return pl.concat(
        [
            s.select(
                "arm",
                "time_series_id",
                "label",
                "group",
                "capacity_mw",
                "slice",
                "n_old",
                "n_new",
                metric=pl.lit(m),
                sum_old=pl.col(f"old__{m}"),
                sum_new=pl.col(f"new__{m}"),
            )
            for m in metrics
        ]
    ).with_columns(file=pl.lit(name))


def pooled(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Pool rows over Primaries (sums over rows) and add old, new, and the change."""
    return (
        frame.group_by("file", "arm", "slice", "metric", *by)
        .agg(
            pl.col("n_old").sum(),
            pl.col("n_new").sum(),
            pl.col("sum_old").sum(),
            pl.col("sum_new").sum(),
        )
        .with_columns(
            old=pl.col("sum_old") / pl.col("n_old"), new=pl.col("sum_new") / pl.col("n_new")
        )
        .with_columns(change=pl.col("new") - pl.col("old"))
    )


def main() -> None:
    """Write the tables and print the largest changes."""
    frame = pl.concat([long(name=n) for n in NAMES])
    per_primary = pooled(frame=frame, by=["label", "group"])
    per_group = pooled(frame=frame, by=["group"])
    overall = pooled(frame=frame.with_columns(group=pl.lit("all16")), by=["group"])
    for name, t in (("per_primary", per_primary), ("per_group", per_group), ("overall", overall)):
        t.write_parquet(OUT / f"compare_{name}.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    for name, t in (("per_group", per_group), ("per_primary", per_primary)):
        print(f"\n== largest |change| in CRPS, {name} ==")
        print(
            t.filter(pl.col("metric") == "crps")
            .with_columns(abs_change=pl.col("change").abs())
            .sort("abs_change", descending=True)
            .select(
                "file",
                "arm",
                "slice",
                *(["label"] if name == "per_primary" else []),
                "group",
                "n_old",
                "n_new",
                "old",
                "new",
                "change",
            )
            .head(15)
        )
    print("\n== headline slice lead_days_1_10, CRPS, per group and overall, max over arms ==")
    for t in (per_group, overall):
        print(
            t.filter(pl.col("metric") == "crps", pl.col("slice") == "lead_days_1_10")
            .group_by("file", "group")
            .agg(
                max_abs=pl.col("change").abs().max(),
                rows_dropped=(pl.col("n_old") - pl.col("n_new")).max(),
            )
            .sort("file", "group")
        )
    print("\n== all metrics and slices: largest |change| as share of old, per group ==")
    print(
        per_group.filter(pl.col("old").abs() > 1e-9)
        .with_columns(rel=(pl.col("change") / pl.col("old")).abs())
        .sort("rel", descending=True)
        .select("file", "arm", "slice", "metric", "group", "n_old", "n_new", "old", "new", "rel")
        .head(10)
    )


if __name__ == "__main__":
    main()
