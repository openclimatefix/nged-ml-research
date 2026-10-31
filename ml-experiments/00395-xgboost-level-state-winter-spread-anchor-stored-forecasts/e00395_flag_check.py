"""Count how many scored half-hours of experiment 395 are flagged in the cleaned power table.

The actuals of experiments 377, 379, 381, 391, and 395 come from experiment 103's `rows.parquet`
through the feature cache, which carries no `drop_reason`. This script joins the scored
half-hours (the shared slice table of Part A) to the production `cleaned_power_time_series` Delta
table and counts the rows whose `drop_reason` is set, and the scored readings that are exactly zero.

    DATA_PATH_INTERNAL=... uv run --no-sync python .../e00395_flag_check.py
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA = ML_EXPERIMENTS_DATA_DIR
WORK = DATA / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts" / "work"
CLEANED = Path(DATA).parent / "NGED" / "cleaned_power_time_series.delta"


def main() -> None:
    """Print the counts."""
    table = (
        pl.read_parquet(WORK / "slice_table.parquet")
        .select("time_series_id", "valid_time")
        .unique()
    )
    rows = (
        pl.read_parquet(WORK / "rows_K0.parquet")
        .select("time_series_id", "valid_time", "label", "actual_points")
        .unique(["time_series_id", "valid_time"])
    )
    keys = table.join(rows, on=["time_series_id", "valid_time"])
    flagged = (
        pl.scan_delta(str(CLEANED))
        .filter(pl.col("drop_reason").is_not_null())
        .select("time_series_id", "time", "drop_reason", "power")
        .collect()
    )
    print("scored unique (series, valid_time) half-hours:", keys.height)
    print("exactly zero actual:", keys.filter(pl.col("actual_points") == 0).height)
    for name, shift in (
        ("time == valid_time", timedelta(0)),
        ("time == valid_time - 30 min", timedelta(minutes=30)),
    ):
        j = keys.with_columns(time=pl.col("valid_time") - shift).join(
            flagged.with_columns(pl.col("time_series_id").cast(keys.schema["time_series_id"])),
            on=["time_series_id", "time"],
        )
        print(f"flagged, {name}: {j.height}")
        print(
            j.group_by("drop_reason").agg(rows=pl.len(), zero=(pl.col("actual_points") == 0).sum())
        )
        print(j.group_by("label").len().sort("len", descending=True).head(8))
        if shift == timedelta(0):
            events_check(flagged_keys=j)


def events_check(*, flagged_keys: pl.DataFrame) -> None:
    """Count the flagged zeros that fall inside a hand-labelled event, unpadded and padded 24 h."""
    labels = pl.read_csv(DATA / "briefs" / "sw_block_detector_label_table.csv").with_columns(
        pl.col("onset_utc").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", strict=False),
        pl.col("end_utc").str.to_datetime("%Y-%m-%d %H:%M", time_zone="UTC", strict=False),
    )
    out = []
    for r in flagged_keys.iter_rows(named=True):
        named = labels.filter(
            pl.col("pair").str.contains(r["label"] + r"\b"), pl.col("end_utc").is_not_null()
        )
        inside = named.filter(
            pl.col("onset_utc") <= r["valid_time"], pl.col("end_utc") >= r["valid_time"]
        ).height
        pad = timedelta(hours=24)
        padded = named.filter(
            pl.col("onset_utc") - pad <= r["valid_time"], pl.col("end_utc") + pad >= r["valid_time"]
        ).height
        out.append({"label": r["label"], "inside": inside > 0, "padded": padded > 0})
    print(
        pl.DataFrame(out)
        .group_by("label")
        .agg(
            flagged=pl.len(),
            inside_unpadded=pl.col("inside").sum(),
            inside_padded=pl.col("padded").sum(),
        )
        .sort("label")
    )


if __name__ == "__main__":
    main()
