# ruff: noqa: D103, E501
"""Rebuild the two stage-1 ERA5 tables to 2026-10-31 from the stored box plus the new months.

Merges `stage1_box/<variable>.parquet` (to 2026-07-31) with the new rows from
`e00607_fetch_era5_box_cds.py` into a combined box folder under the experiment's data folder, then
runs the stage 1 pilot's `era5_features` and experiment 561's wind-component builder on that
folder. Output: `era5_features.parquet` and `era5_wind_components.parquet` there. The stored tables
are not edited. Checks that every shared row equals the stored table's row.

    PYTHONPATH=studies/stage1_pilot:ml-experiments/00561-wind-components-era5-and-ens-summary-data-prep \
        uv run --no-sync python e00607_derive_era5.py
"""

import json
from pathlib import Path
from typing import Final

import demand_inputs as demand  # ty: ignore[unresolved-import]
import e00561_era5_uv as uv_build  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00607-*")) / "work"
NEW: Final[Path] = WORK / "era5_stage1_box_cds"
COMBINED: Final[Path] = WORK / "era5_stage1_box_extended"


def merge_box() -> None:
    COMBINED.mkdir(exist_ok=True)
    for stored in sorted(demand.BOX_DIR.glob("*.parquet")):
        new = NEW / stored.name
        old = pl.scan_parquet(stored)
        frame = pl.concat([old, pl.scan_parquet(new).select(old.collect_schema().names())]).sort(
            "time", "latitude", "longitude"
        )
        frame.sink_parquet(COMBINED / stored.name)
        print(
            stored.name,
            pl.scan_parquet(COMBINED / stored.name)
            .select(pl.len(), pl.col("time").max())
            .collect()
            .rows(),
        )


def compare(*, name: str, new: pl.DataFrame, old: pl.DataFrame) -> dict[str, object]:
    keys = ["time_series_id", "valid_time"]
    both = old.join(new, on=keys, how="left", suffix="_new")
    cols = [c for c in old.columns if c not in keys]
    worst = {c: float((both[c] - both[f"{c}_new"]).abs().max()) for c in cols}  # ty: ignore[invalid-argument-type]
    return {
        "table": name,
        "rows": new.height,
        "stored_rows": old.height,
        "stored_rows_missing_from_new": int(both[f"{cols[0]}_new"].null_count())
        - int(both[cols[0]].null_count()),
        "largest_difference_on_shared_rows": max(worst.values()),
        "last_valid_time": str(new["valid_time"].max()),
    }


def main() -> None:
    merge_box()
    demand.BOX_DIR = COMBINED
    series = demand.series_table()
    features = demand.era5_features(series=series)
    features.write_parquet(WORK / "era5_features.parquet")
    components = uv_build.build()
    components.write_parquet(WORK / "era5_wind_components.parquet")
    report = [
        compare(
            name="era5_features",
            new=features,
            old=pl.read_parquet(demand.INPUTS_DIR / "era5_features.parquet"),
        ),
        compare(
            name="era5_wind_components",
            new=components,
            old=pl.read_parquet(uv_build.EXISTING.with_name("era5_wind_components.parquet")),
        ),
    ]
    (WORK / "era5_derived_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
