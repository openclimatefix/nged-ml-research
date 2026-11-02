"""Summarise experiment 503's sweep sums: shift shares, rule recall, false alarms, genuine steps.

Reads `work/sweep/sweep_all.parquet` (every Primary of the six, four refits). Prints, for each arm
and rule set, the shift of the median as a share of the corruption (`abs_shift_sum` over
`corruption_sum`, the summed absolute change of the three lag values), by lead band and
injection; the share of corrupted rows in which each rule newly fires; how often each rule fires
on real rows; and, for a genuine 5% step, the shift as a share of the step, the band-width ratio,
and the share of lags that rule far sets to missing.
"""

import e00503_fit as fit  # ty: ignore[unresolved-import]
import polars as pl

SUMS = fit.WORK_DIR / "sweep" / "sweep_all.parquet"
KEYS = ["booster", "rule_set", "kind", "variant", "band"]
ARM_LABEL = {
    ("a", "none"): "a",
    ("a", "iii"): "a-rule(i,iii)",
    ("a", "iii2"): "a-rule(i,ii,iii)",
    ("g", "none"): "G",
    ("g", "iii"): "G-A",
    ("g", "iii2"): "G-B",
}


def pooled(*, widening: float) -> pl.DataFrame:
    """Return the sums pooled over the six Primaries and four refits, with the derived shares."""
    sums = pl.read_parquet(SUMS).filter(pl.col("widening") == widening)
    return (
        sums.group_by(*KEYS, "sign")
        .agg(pl.all().exclude("site", "refit", "widening").sum())
        .with_columns(
            share=pl.col("abs_shift_sum") / pl.col("corruption_sum"),
            signed_share=pl.col("shift_sum") / pl.col("corruption_sum"),
            fire_zero_new=pl.col("new_fire_zero_rows") / pl.col("corrupted_rows"),
            fire_far_new=pl.col("new_fire_far_rows") / pl.col("corrupted_rows"),
            fire_stuck_new=pl.col("new_fire_stuck_rows") / pl.col("corrupted_rows"),
            width_ratio=pl.col("width_sum") / pl.col("width_base_sum"),
            cells=pl.col("lag_cells"),
            share_zero=pl.col("fire_zero") / pl.col("lag_cells"),
            share_far=pl.col("fire_far") / pl.col("lag_cells"),
            share_stuck=pl.col("fire_stuck") / pl.col("lag_cells"),
            share_gap=pl.col("table_gap") / pl.col("lag_cells"),
        )
        .with_columns(
            arm=pl.struct("booster", "rule_set").map_elements(
                lambda s: ARM_LABEL[(s["booster"], s["rule_set"])], return_dtype=pl.String
            )
        )
    )


def main() -> None:
    """Print the summaries."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    table = pooled(widening=1.0)
    for band in ("0-24 h", "24-72 h", "72-168 h"):
        print(f"\n== shift of the median as a share of the corruption, lead {band} ==")
        print(
            table.filter((pl.col("band") == band) & pl.col("kind").is_in(["stuck", "dropout", "spike"]))
            .with_columns(case=pl.col("kind") + " " + pl.col("variant"))
            .pivot(on="arm", index="case", values="share")
            .sort("case")
            .with_columns(pl.selectors.float().round(3))
        )
    print("\n== share of corrupted rows in which each rule newly fires (any lead, G-B) ==")
    print(
        table.filter(
            (pl.col("band") == "0-240 h")
            & (pl.col("arm") == "G-B")
            & pl.col("kind").is_in(["stuck", "dropout", "spike"])
        ).select("kind", "variant", "fire_zero_new", "fire_far_new", "fire_stuck_new", "corrupted_rows")
    )
    print("\n== real rows (clean variant), share of lag values each rule sets to missing ==")
    print(
        table.filter((pl.col("kind") == "clean") & (pl.col("band") == "0-240 h") & (pl.col("arm") == "G-B"))
        .select("n", "cells", "share_zero", "share_far", "share_stuck", "share_gap")
    )
    steps = table.filter((pl.col("kind") == "step") & (pl.col("band") == "24-72 h"))
    print("\n== genuine 5% step, lead 24-72 h: |shift| / step (5% of capacity) ==")
    print(
        steps.with_columns(share_of_step=pl.col("abs_shift_sum") / pl.col("n") / 0.05)
        .pivot(on="arm", index=["variant", "sign"], values="share_of_step")
        .sort("variant", "sign")
        .with_columns(pl.selectors.float().round(3))
    )
    print("\n== genuine 5% step: share of lag values that rule far sets to missing ==")
    print(
        steps.filter(pl.col("arm") == "G-B")
        .select("variant", "sign", "share_far", "width_ratio")
        .sort("variant", "sign")
    )
    print("\n== band-width ratio (width with the step over width without), lead 24-72 h ==")
    print(
        steps.pivot(on="arm", index=["variant", "sign"], values="width_ratio")
        .sort("variant", "sign")
        .with_columns(pl.selectors.float().round(3))
    )


if __name__ == "__main__":
    main()
