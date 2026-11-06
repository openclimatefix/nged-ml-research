# ruff: noqa: D103
"""Power coverage of the production NGED Delta tables for the final bake-off test months.

Reads only observed power and the cleaning flags. It reads no forecast and no forecast error.
Writes one parquet per table under the experiment's folder in the private data store.
"""

from pathlib import Path
from typing import Final

import polars as pl
from studies.anonymise import SITE_LABELS  # noqa: F401  (labels come from pv_dataset)
from studies.experiment_store import ML_EXPERIMENTS_DATA_DIR
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import REPO_DATA_DIR

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00605-bakeoff-final-test-data-check"
MONTHS: Final[tuple[str, ...]] = (
    "2025-10",
    "2025-12",
    "2026-01",
    "2026-03",
    "2026-05",
    "2026-06",
    "2026-07",
    "2026-08",
    "2026-09",
    "2026-10",
)
DEVELOPMENT_MONTHS: Final[frozenset[str]] = frozenset(
    {"2025-10", "2025-12", "2026-01", "2026-03", "2026-06"}
)
TEST_END: Final[str] = "2026-11-01"  # period-ending stamps up to and including 2026-11-01 00:00


def series_labels() -> pl.DataFrame:
    """Return one row per series of interest with a publishable label and group."""
    meta = pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet")
    primaries = meta.filter(pl.col("substation_type") == "Primary").select(
        "time_series_id",
        label=pl.col("time_series_name"),
        group=pl.when(pl.col("units") == "MW")
        .then(pl.lit("Primary, MW"))
        .otherwise(pl.lit("Primary, MVA")),
    )
    nodes = meta.filter(pl.col("substation_type").is_in(["BSP", "GSP"])).select(
        "time_series_id", label=pl.col("time_series_name"), group=pl.lit("BSP or GSP")
    )
    wind = wind_sites().select("time_series_id", label=pl.col("site"), group=pl.lit("Wind"))
    solar = pv_sites().select("time_series_id", label=pl.col("site"), group=pl.lit("Solar"))
    return pl.concat([primaries, nodes, wind, solar]).sort("group", "label")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = series_labels()
    ids = labels["time_series_id"].to_list()
    cleaned = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(ids))
        .filter(pl.col("time") > pl.datetime(2025, 9, 1, time_zone="UTC"))
        .collect()
    )
    raw = (
        pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(ids))
        .filter(pl.col("time") > pl.datetime(2025, 9, 1, time_zone="UTC"))
        .collect()
    )
    print("labels", labels.height, "cleaned rows", cleaned.height, "raw rows", raw.height)
    # Period-ending stamps: month m holds stamps in (m-01 00:00, (m+1)-01 00:00].
    cleaned = cleaned.with_columns(
        month=(pl.col("time") - pl.duration(minutes=30)).dt.strftime("%Y-%m")
    )
    raw = raw.with_columns(month=(pl.col("time") - pl.duration(minutes=30)).dt.strftime("%Y-%m"))
    expected = pl.DataFrame({"month": list(MONTHS)}).with_columns(
        expected=pl.col("month").map_elements(
            lambda m: (
                pl.date_range(
                    pl.date(int(m[:4]), int(m[5:]), 1),
                    pl.date(int(m[:4]), int(m[5:]), 1).dt.month_end(),
                    "1d",
                    eager=True,
                ).len()
                * 48
            ),
            return_dtype=pl.Int64,
        )
    )
    per = (
        cleaned.group_by("time_series_id", "month")
        .agg(
            present=pl.len(),
            unique=pl.col("time").n_unique(),
            flagged=pl.col("drop_reason").is_not_null().sum(),
            null_power=pl.col("power").is_null().sum(),
            last=pl.col("time").max(),
        )
        .join(expected, on="month", how="right")
        .join(pl.DataFrame({"time_series_id": ids}), on="time_series_id", how="right")
    )
    grid = labels.join(expected, how="cross")
    table = (
        grid.join(per.drop("expected"), on=["time_series_id", "month"], how="left")
        .with_columns(pl.col("present", "unique", "flagged", "null_power").fill_null(0))
        .with_columns(
            missing=pl.col("expected") - pl.col("unique"),
            development=pl.col("month").is_in(list(DEVELOPMENT_MONTHS)),
        )
        .sort("group", "label", "month")
    )
    table.write_parquet(OUT_DIR / "power_by_series_month.parquet")
    reasons = (
        cleaned.filter(pl.col("drop_reason").is_not_null())
        .group_by("time_series_id", "month", "drop_reason")
        .len()
        .join(labels, on="time_series_id")
        .filter(pl.col("month").is_in(list(MONTHS)))
        .sort("group", "label", "month", "drop_reason")
    )
    reasons.write_parquet(OUT_DIR / "drop_reasons.parquet")
    ends = (
        cleaned.group_by("time_series_id")
        .agg(first=pl.col("time").min(), last=pl.col("time").max())
        .join(labels, on="time_series_id")
        .sort("group", "label")
    )
    ends.write_parquet(OUT_DIR / "series_end.parquet")
    # Raw and cleaned tables hold the same stamps when their row counts match.
    print("raw vs cleaned", raw.height, cleaned.height)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    print(ends)
    print(
        table.filter(pl.col("month") >= "2026-05")
        .group_by("group", "month")
        .agg(
            missing=pl.col("missing").sum(),
            expected=pl.col("expected").sum(),
            flagged=pl.col("flagged").sum(),
        )
        .sort("group", "month")
    )
    print(reasons.group_by("drop_reason").agg(pl.col("len").sum()))


if __name__ == "__main__":
    main()
