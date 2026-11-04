"""Validate the rebuilt ensemble summaries against the originals and against physical limits."""

import json

import e00561_ens_summary_uv as m  # ty: ignore[unresolved-import]
import polars as pl

new = pl.scan_parquet(m.SUMMARY_DIR / "init=*.parquet")
old = pl.scan_parquet(m.prep.SUMMARY_DIR / "init=*.parquet")
keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
n_new = new.select(pl.len()).collect().item()
n_old = old.select(pl.len()).collect().item()
duplicates = n_new - new.select(keys).unique().select(pl.len()).collect().item()
old_cols = [c for c in old.collect_schema().names() if c not in keys]
joined = new.join(old, on=keys, suffix="_o", how="inner")
diff = joined.select(
    pl.len().alias("rows"),
    *[(pl.col(c) - pl.col(f"{c}_o")).abs().max().alias(c) for c in old_cols],
).collect()
new_cols = [c for c in new.collect_schema().names() if c not in keys]
wind_cols = [c for c in new_cols if "wind_u" in c or "wind_v" in c]
stats = new.select(
    *[pl.col(c).min().alias(f"{c}__min") for c in wind_cols],
    *[pl.col(c).max().alias(f"{c}__max") for c in wind_cols],
    *[(pl.col(c).is_null() | pl.col(c).is_nan()).sum().alias(f"{c}__bad") for c in new_cols],
).collect()
origin = new.select(
    ((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60)
    .min()
    .alias("min_lead_hours"),
    pl.col("power_fcst_init_time").dt.hour().n_unique().alias("distinct_origin_hours"),
    pl.col("power_fcst_init_time").dt.hour().first().alias("origin_hour_utc"),
    pl.col("power_fcst_init_time").dt.date().n_unique().alias("origin_days"),
).collect()
bad = sum(stats[c].item() for c in stats.columns if c.endswith("__bad"))
report = {
    "rows_new": n_new,
    "rows_old": n_old,
    "duplicate_keys": duplicates,
    "columns": new_cols,
    "joined_rows": diff["rows"].item(),
    "max_abs_difference_old_columns": max(diff.drop("rows").row(0)),
    "null_or_nan_cells": bad,
    "wind_ranges": {c: stats[c].item() for c in stats.columns if c.endswith(("__min", "__max"))},
    "origin_check": origin.to_dicts()[0],
}
print(json.dumps(report, indent=1, default=str))
(m.SUMMARY_DIR.parent / "ens_summary_uv_validation.json").write_text(
    json.dumps(report, indent=1, default=str)
)
