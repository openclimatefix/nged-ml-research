# ruff: noqa: E501
"""Experiment 656, post hoc: recoveries of the fold models when the fit also starts at 6 and 8 times the midday reading.

Written after the audit found that the pre-registered start grid (0.25, 1, and 3 times) cannot reach
the true capacity of the 6 times worlds. These rows are labelled post hoc; no bar is re-scored.
"""

import e00656_analyse as a
import e00656_common as k
import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
res = pl.concat([pl.read_parquet(p) for p in sorted((k.WORK / "probes_wide" / "2026-06").glob("*.parquet"))], how="diagonal_relaxed")
rec = a.recoveries(res=res)
six = rec.filter(pl.col("site").is_in(k.SIX) & pl.col("method").is_in(["F4", "F1", "FA"]))
print("== Post hoc, wide starts: median recovery over the six ==")
print(six.group_by("condition", "method").agg(rec=pl.col("rec").median()).pivot(on="method", index="condition", values="rec").sort("condition").with_columns(pl.col(pl.Float64).round(2)))
print("== Post hoc FA recovery per Primary at 6 times, 2 times, 0.5 times (r0) ==")
pp = six.filter((pl.col("method") == "FA") & pl.col("condition").is_in(["pf_fleet_l05_r0", "pf_fleet_l2_r0", "pf_fleet_l6_r0", "pf_phys_l6", "pf_sunload"]))
print(pp.group_by("site", "condition").agg(rec=pl.col("rec").median()).pivot(on="condition", index="site", values="rec").with_columns(pl.col(pl.Float64).round(2)))
nat = rec.filter(pl.col("site").is_in(k.NATIVE) & (pl.col("method") == "FA") & pl.col("condition").is_in(["pf_fleet_l05_r0", "pf_fleet_l1_r0", "pf_fleet_l2_r0", "pf_fleet_l6_r0"]))
print("== Post hoc FA recovery at the five native Primaries ==")
print(nat.group_by("condition").agg(rec=pl.col("rec").median()).sort("condition").with_columns(pl.col(pl.Float64).round(2)))
