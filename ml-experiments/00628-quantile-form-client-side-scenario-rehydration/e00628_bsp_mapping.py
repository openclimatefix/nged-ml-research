"""Experiment 628: write `work/bsp_mapping.csv`, the Primary-to-BSP mapping of the 16 Primaries.

Joins experiment 436's node order and capacities to NGED's hierarchy file
(`data/NGED/substation_hierarchy_2026-10-15/Substations.csv`) through the metadata table's
substation number. Columns: node, time_series_id, bsp, capacity_mw. Private data: never commit.
"""

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00628_score as s628  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import REPO_DATA_DIR


def main() -> None:
    """Write the mapping."""
    series = adapter.series_table().with_row_index("node")
    meta = pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet").select(
        "time_series_id", "substation_number"
    )
    hierarchy = pl.read_csv(
        REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
    ).select("substation_number", "bsp_name")
    out = (
        series.select("node", "time_series_id", "capacity_mw")
        .join(meta, on="time_series_id")
        .join(hierarchy, on="substation_number", how="left")
        .select("node", "time_series_id", pl.col("bsp_name").alias("bsp"), "capacity_mw")
        .sort("node")
    )
    if out.height != 16 or out["bsp"].null_count():
        msg = f"mapping incomplete: {out}"
        raise ValueError(msg)
    out.write_csv(s628.OUT / "bsp_mapping.csv")
    print(out)


if __name__ == "__main__":
    main()
