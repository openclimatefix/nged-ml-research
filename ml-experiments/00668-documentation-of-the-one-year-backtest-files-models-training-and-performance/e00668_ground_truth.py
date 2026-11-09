"""Write the five ground-truth observed-power Parquet files of the one-year backtest.

uv run --no-sync python ml-experiments/00668-*/e00668_ground_truth.py

Each file holds one row per series per half-hour from 2025-11-01 00:00 UTC to the last half-hour in
the cleaned power table, with explicit null rows for missing readings, so it joins to the forecast
files on `time_series_id` and `valid_time`. The files are private and carry the original ids.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
CLEANED_URI: Final[str] = str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta")
RAW_URI: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")
METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
MASK_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact" / "mask.parquet"
)
START: Final[datetime] = datetime(2025, 11, 1, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
EXPECTED: Final[dict[str, int]] = {"primaries": 16, "bsps": 2, "gsps": 2, "wind": 3, "solar": 6}
SUBSTATION_TYPE: Final[dict[str, str]] = {"primaries": "Primary", "bsps": "BSP", "gsps": "GSP"}


def series_ids(*, name: str, metadata: pl.DataFrame) -> list[int]:
    """Return the original `time_series_id`s of one file's series, sorted."""
    if name in SUBSTATION_TYPE:
        ids = metadata.filter(pl.col("substation_type") == SUBSTATION_TYPE[name])
        usable = ids.filter(pl.col("time_series_id") != 33)  # id 33 holds under 400 readings
        ids = usable["time_series_id"].to_list()
    elif name == "wind":
        ids = wind_sites()["time_series_id"].to_list()
    else:
        ids = pv_sites()["time_series_id"].to_list()
    return sorted(ids)


def build(
    *, name: str, ids: list[int], raw: pl.DataFrame, cleaned: pl.DataFrame, metadata: pl.DataFrame
) -> pl.DataFrame:
    """Return one file's frame on the full half-hourly grid."""
    last = raw["time"].max()
    grid = pl.datetime_range(START, last, interval="30m", time_zone="UTC", eager=True)
    base = pl.DataFrame({"time_series_id": ids}, schema={"time_series_id": pl.Int32}).join(
        pl.DataFrame({"valid_time": grid}), how="cross"
    )
    units = metadata.select("time_series_id", pl.col("units").cast(pl.String).alias("power_unit"))
    frame = (
        base.join(
            raw.rename({"time": "valid_time", "power": "power_raw"}),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="1:1",
        )
        .join(
            cleaned.rename({"time": "valid_time"}).select(
                "time_series_id", "valid_time", "drop_reason"
            ),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="1:1",
        )
        .join(units, on="time_series_id", how="left", validate="m:1")
    )
    if name == "wind":
        mask = pl.read_parquet(MASK_PATH).rename({"drop_reason": "research_flag"})
        frame = frame.join(mask, on=["time_series_id", "valid_time"], how="left", validate="1:1")
    if "research_flag" not in frame.columns:
        frame = frame.with_columns(research_flag=pl.lit(None, dtype=pl.String))
    return (
        frame.with_columns(
            power=pl.when(pl.col("drop_reason").is_null()).then(pl.col("power_raw")),
        )
        .select(
            pl.col("time_series_id").cast(pl.Int8),
            "valid_time",
            pl.col("power").cast(pl.Float32),
            pl.col("power_raw").cast(pl.Float32),
            "drop_reason",
            "research_flag",
            "power_unit",
        )
        .sort("time_series_id", "valid_time")
    )


def check(*, name: str, frame: pl.DataFrame, ids: list[int]) -> None:
    """Assert the file's checks and print their results."""
    n_grid = frame["valid_time"].n_unique()
    assert frame.height == len(ids) * n_grid, (name, frame.height, len(ids), n_grid)
    assert (
        frame.select(pl.struct("time_series_id", "valid_time").is_duplicated().any()).item()
        is False
    )
    assert frame["valid_time"].dtype == pl.Datetime("us", "UTC")
    assert frame["time_series_id"].n_unique() == EXPECTED[name], name
    assert frame["power_unit"].is_not_null().all()
    assert (frame.filter(pl.col("drop_reason").is_not_null())["power"].is_null()).all()
    print(
        name,
        "rows",
        frame.height,
        "series",
        frame["time_series_id"].n_unique(),
        "first",
        frame["valid_time"].min(),
        "last",
        frame["valid_time"].max(),
        "null raw",
        frame["power_raw"].null_count(),
        "flagged",
        frame["drop_reason"].is_not_null().sum(),
        "research flagged",
        frame["research_flag"].is_not_null().sum(),
        flush=True,
    )


def main() -> None:
    """Write the five files."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    metadata = pl.read_parquet(METADATA_PATH)
    from contracts.power_schemas import DROP_REASONS

    print("cleaning rules in the code when the data was read:", DROP_REASONS, flush=True)
    raw = pl.scan_delta(RAW_URI).filter(pl.col("time") >= pl.lit(START)).collect()
    cleaned = pl.scan_delta(CLEANED_URI).filter(pl.col("time") >= pl.lit(START)).collect()
    for frame in (raw, cleaned):
        assert not frame.select(pl.struct("time_series_id", "time").is_duplicated().any()).item()
    assert set(cleaned["drop_reason"].drop_nulls().unique()) <= set(DROP_REASONS)
    joined = raw.join(cleaned, on=["time_series_id", "time"], how="full", suffix="_cleaned")
    assert joined.height == raw.height == cleaned.height, "raw and cleaned rows differ"
    assert joined.filter(pl.col("power").ne_missing(pl.col("power_cleaned"))).is_empty()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED:
        ids = series_ids(name=name, metadata=metadata)
        frame = build(name=name, ids=ids, raw=raw, cleaned=cleaned, metadata=metadata)
        check(name=name, frame=frame, ids=ids)
        tmp = OUT_DIR / f".ground_truth_{name}.parquet.tmp"
        frame.write_parquet(tmp)
        tmp.replace(OUT_DIR / f"ground_truth_{name}.parquet")


if __name__ == "__main__":
    main()
