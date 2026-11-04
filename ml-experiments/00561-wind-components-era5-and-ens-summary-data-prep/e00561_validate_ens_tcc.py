"""Validate the ENS cloud-cover summaries and the checkpointed cell frames."""

import json

import e00561_ens_tcc as m  # ty: ignore[unresolved-import]
import polars as pl

runs = m.run_times()
summary = pl.scan_parquet(m.SUMMARY_DIR / "init=*.parquet")
native = pl.scan_parquet(m.NATIVE_DIR / "init=*.parquet")
keys = ["time_series_id", "power_fcst_init_time", "valid_time"]

n_files = len(list(m.SUMMARY_DIR.glob("init=*.parquet")))
n_native_files = len(list(m.NATIVE_DIR.glob("init=*.parquet")))
n_rows = summary.select(pl.len()).collect().item()
duplicates = n_rows - summary.select(keys).unique().select(pl.len()).collect().item()
stats = summary.select(
    pl.col("mean_total_cloud_cover").min().alias("mean_min"),
    pl.col("mean_total_cloud_cover").max().alias("mean_max"),
    pl.col("mean_total_cloud_cover").mean().alias("mean_mean"),
    pl.col("std_total_cloud_cover").min().alias("std_min"),
    pl.col("std_total_cloud_cover").max().alias("std_max"),
    pl.col("std_total_cloud_cover").mean().alias("std_mean"),
    pl.col("mean_total_cloud_cover").is_null().sum().alias("null_mean"),
    pl.col("mean_total_cloud_cover").is_nan().sum().alias("nan_mean"),
    pl.col("std_total_cloud_cover").is_null().sum().alias("null_std"),
    pl.col("std_total_cloud_cover").is_nan().sum().alias("nan_std"),
    (pl.col("valid_time") - pl.col("power_fcst_init_time"))
    .dt.total_minutes()
    .min()
    .alias("min_lead_min"),
    (pl.col("power_fcst_init_time").dt.hour().n_unique()).alias("distinct_origin_hours"),
    pl.col("power_fcst_init_time").dt.date().n_unique().alias("origin_days"),
).collect()
cells = native.select(
    pl.len().alias("rows"),
    pl.col("tcc").min().alias("min"),
    pl.col("tcc").max().alias("max"),
    pl.col("tcc").is_null().sum().alias("nulls"),
    pl.col("ensemble_member").n_unique().alias("members"),
    pl.col("h3_index").n_unique().alias("cells"),
).collect()
spread_by_lead = (
    summary.with_columns(
        lead_day=((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() // 24)
    )
    .group_by("lead_day")
    .agg(pl.col("std_total_cloud_cover").mean())
    .sort("lead_day")
    .collect()
)
report = {
    "runs_expected": len(runs),
    "summary_files": n_files,
    "native_files": n_native_files,
    "summary_rows": n_rows,
    "duplicate_keys": duplicates,
    "summary_stats": stats.to_dicts()[0],
    "native_cell_stats": cells.to_dicts()[0],
    "mean_std_percent_by_lead_day": {
        int(d): round(float(s), 2) for d, s in spread_by_lead.iter_rows()
    },
}
print(json.dumps(report, indent=1, default=str))
(m.SUMMARY_DIR.parent / "ens_summary_tcc_validation.json").write_text(
    json.dumps(report, indent=1, default=str)
)
