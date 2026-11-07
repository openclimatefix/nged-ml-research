# ruff: noqa: E501
"""Shared constants and loaders of experiment 631: Primary scenarios plus metered farms, summed to a BSP.

Extends experiment 629 (`e00629_common`): the 16 Primaries, the stored K1 bake-off backtests, and the
stored direct BSP forecasts are read exactly as there. The nine metered wind and solar farms with a
stored forecast (experiment 615) are added as nodes. Farms are labelled W1 to W3 and Solar A to F;
which farm sits under which BSP is a private mapping and is never printed in a published artefact.
Rows with a valid time on or after 2026-07-01 are never read into a score.
"""

from pathlib import Path
from typing import Final

import e00629_common as c629  # ty: ignore[unresolved-import]
import polars as pl
from e00629_common import (  # ty: ignore[unresolved-import]
    ARMS,
    END_OF_DEV,
    GROUP_IDS,
    MONTHS,
    N_MEMBERS,
    N_STEPS,
    STEP,
)
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR

__all__ = ["ARMS", "END_OF_DEV", "GROUP_IDS", "MONTHS", "N_MEMBERS", "N_STEPS", "STEP"]

SLUG: Final[str] = "00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
FARM_EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_wind_solar_dev_2026-11-07"
WIND_CANDIDATE: Final[str] = "wind_default2"
SOLAR_CANDIDATE: Final[str] = "solar_a2_member_quarterly"
BSP_GROUPS: Final[tuple[int, int]] = (1, 2)
"""Group ids of the two BSPs (Boston, Skegness); the GSPs are not scored here."""
BSP_BY_FRAGMENT: Final[dict[int, str]] = {1: "BOSTON", 2: "SKEGNESS"}
FARMS: Final[tuple[str, ...]] = (
    "W1",
    "W2",
    "W3",
    "Solar A",
    "Solar B",
    "Solar C",
    "Solar D",
    "Solar E",
    "Solar F",
)
POWER_SIGN: Final[float] = -1.0
"""Flow into the BSP is positive, so a farm's output (positive when generating) enters the flow with this sign."""
FIRST_HISTORY: Final = c629.SCHAAKE_FIRST
NIGHT_PRESENCE: Final[float] = 0.3
"""A solar (month, half-hour) slot present on fewer than this share of days is night: absent means zero output."""
OFFSET_BUFFER_DAYS: Final[int] = 16


def farm_table() -> pl.DataFrame:
    """Return the nine farms with `time_series_id`, label, and BSP group id (private mapping)."""
    wind = wind_sites().select("time_series_id", "site")
    solar = pv_sites().select("time_series_id", pl.format("Solar {}", pl.col("site")).alias("site"))
    farms = pl.concat([wind, solar]).rename({"site": "label"})
    md = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet").select(
        "time_series_id", "substation_number"
    )
    hier = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR.parent
        / "NGED"
        / "substation_hierarchy_2026-10-15"
        / "Substations.csv"
    ).select("substation_number", "bsp_name")
    out = farms.join(md, on="time_series_id").join(hier, on="substation_number", how="left")
    group = [
        next((g for g, frag in BSP_BY_FRAGMENT.items() if frag in str(n)), 0)
        for n in out["bsp_name"]
    ]
    out = out.with_columns(group=pl.Series(group, dtype=pl.Int64))
    order = {lab: i for i, lab in enumerate(FARMS)}
    return out.sort(pl.col("label").replace_strict(order, return_dtype=pl.Int64)).select(
        "time_series_id", "label", "group"
    )


def read_farm(*, candidate: str, month: str) -> pl.DataFrame:
    """Return one stored farm backtest file (fractions of the series' scoring capacity)."""
    return pl.read_parquet(FARM_EXPORT / f"{candidate}_{month}.parquet")
