# ruff: noqa: E501
"""Checks for experiment 471: error of the median forecast (trigger 3) and member share of cold nights.

Prints, per variant and group, the mean absolute error of the q50 forecast in points of capacity on
the `main` pass rows, beside the CRPS gap, and the median minus the mean of the analogue members per
Primary. Writes `work/scores/checks.csv`.
"""

import e00448_refs as base  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import e00471_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

ARMS = ("manual_heuristic", "E56", "Q56", "Tau95_56", "MeanLS56", "Lvl14", "Lvl28")


def main() -> None:
    """Write and print the checks."""
    rows = pl.read_parquet(common.WORK_DIR / "scores/main_rows_E56.parquet").select(
        *base.KEYS, "actual", "capacity_mw", "group", "label", "lead_hours"
    )
    members = common.load_members()
    faithful = score.faithful_frame(members=members)
    records = []
    for arm in ARMS:
        frame = faithful if arm == "manual_heuristic" else score.variant_frame(name=arm)
        joined = rows.join(frame.select(*base.KEYS, "q50"), on=base.KEYS)
        err = (joined["q50"] / joined["capacity_mw"] - joined["actual"]).abs() * 100.0
        for group in ("MVA-only eight", "signed-MW six"):
            sel = (joined["group"] == group).to_numpy()
            records.append({"arm": arm, "group": group, "median_mae_points": float(err.to_numpy()[sel].mean()),
                            "median_bias_points": float(((joined["q50"] / joined["capacity_mw"] - joined["actual"]) * 100.0).to_numpy()[sel].mean())})  # fmt: skip
    out = pl.DataFrame(records)
    out.write_csv(common.WORK_DIR / "scores/checks.csv")
    m = members.filter(pl.col("n") >= 12)
    mm = m.select(common.MEMBER_COLUMNS).to_numpy()
    diff = np.nanmedian(mm, axis=1) - np.nanmean(mm, axis=1)
    per = (
        m.select("time_series_id")
        .with_columns(median_minus_mean_mw=pl.Series(diff))
        .group_by("time_series_id")
        .agg(pl.col("median_minus_mean_mw").mean())
    )
    print(out.with_columns(pl.col(pl.Float64).round(3)))
    print(per.sort("time_series_id").with_columns(pl.col("median_minus_mean_mw").round(4)))


if __name__ == "__main__":
    main()
