# ruff: noqa: E501
"""Shared constants and loaders of experiment 629: Primary scenarios summed to a BSP or GSP.

Reads the stored bake-off backtests (16 Primaries, 11 quantiles, seed-mean) and the stored direct
forecasts of the bulk supply points (BSPs) and grid supply points (GSPs). Nothing is fitted.
Rows with a valid time on or after 2026-07-01 are never read into a score.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"

EXPORT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
NETWORKS: Final[tuple[str, ...]] = ("K1", "Candidate", "K3")
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
N_STEPS: Final[int] = 480
N_MEMBERS: Final[int] = 51
STEP: Final[timedelta] = timedelta(minutes=30)
END_OF_DEV: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Valid times on or after this instant belong to the unopened final-test window."""
LIMIT_BEFORE: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
"""The scaled limits use the BSP meter's history before this instant."""
SCHAAKE_FIRST: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
SCHAAKE_BUFFER: Final[timedelta] = timedelta(days=16)

# The direct forecasts: (label, folder under ML_EXPERIMENTS_DATA_DIR, file prefix).
D611: Final = "00611-bsp-gsp-pilot-backtests-for-the-bake-off/export"
D619: Final = "00619-bsp-gsp-lag-enabled-network-for-the-bake-off/export"
D622: Final = "00622-bsp-gsp-lagged-against-lag-free-matched-d5-thermal-lag-rematch/export"
DIRECT: Final[dict[str, tuple[str, str]]] = {
    "d622_lagged": (D622, "lagged_b1"),
    "d622_lagfree": (D622, "lagfree_b1"),
    "d611_design1": (D611, "design1"),
    "d611_design2": (D611, "design2"),
    "d619_lag": (D619, "lag_enabled"),
}
HEADLINE_DIRECT: Final[dict[str, str]] = {
    "K1": "d622_lagged",
    "Candidate": "d622_lagged",
    "K3": "d622_lagfree",
}
GROUP_IDS: Final[tuple[int, ...]] = (1, 2, 27, 28)
GROUP_NAMES: Final[dict[int, str]] = {
    1: "BSP D17 (Boston)",
    2: "BSP D13 (Skegness)",
    27: "GSP D6 (Walpole)",
    28: "GSP D1 (Bicker Fen)",
}
ARMS: Final[tuple[str, ...]] = ("J0", "J2", "J3", "J5")


def primaries() -> pl.DataFrame:
    """Return the 16 Primaries with name, unit, and the BSP and GSP they sit under (hierarchy file)."""
    sub = pl.read_csv(EXPORT / "substations.csv")
    md = pl.read_parquet(ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet").select(
        "time_series_id", "substation_number"
    )
    hier = pl.read_csv(
        ML_EXPERIMENTS_DATA_DIR.parent
        / "NGED"
        / "substation_hierarchy_2026-10-15"
        / "Substations.csv"
    ).select("substation_number", "bsp_name", "gsp_name")
    out = sub.join(md, on="time_series_id", how="left").join(
        hier, on="substation_number", how="left"
    )
    if out["bsp_name"].is_null().any():
        msg = "a Primary has no BSP in the hierarchy file"
        raise ValueError(msg)
    return out.sort("time_series_id")


def group_nodes(*, prim: pl.DataFrame) -> dict[int, list[int]]:
    """Return, for each group id, the row positions (in `prim`) of the Primaries beneath it."""
    boston = [i for i, n in enumerate(prim["bsp_name"]) if "BOSTON" in n]
    skeg = [i for i, n in enumerate(prim["bsp_name"]) if "SKEGNESS" in n]
    if len(boston) + len(skeg) != prim.height:
        msg = "a Primary is under neither BSP"
        raise ValueError(msg)
    return {1: boston, 2: skeg, 27: boston, 28: skeg}


def read_network(*, network: str, month: str) -> pl.DataFrame:
    """Return one network-month file of the bake-off export."""
    return pl.read_parquet(EXPORT / f"{network}_{month}.parquet")


def read_direct(*, key: str, month: str) -> pl.DataFrame:
    """Return one stored direct BSP/GSP forecast file."""
    folder, prefix = DIRECT[key]
    return pl.read_parquet(ML_EXPERIMENTS_DATA_DIR / folder / f"{prefix}_{month}.parquet")


def steps_of(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the step index (0 = lead 0.5 h) of each lead in hours."""
    return np.rint(lead_hours * 2).astype(int) - 1


def run_arrays(
    *, frame: pl.DataFrame, ids: list[int], columns: list[str]
) -> dict[datetime, np.ndarray]:
    """Return, for each init time, an array (len(ids), 480, len(columns)) with NaN where a row is absent."""
    out: dict[datetime, np.ndarray] = {}
    pos = {sid: i for i, sid in enumerate(ids)}
    for init, sub in frame.group_by("power_fcst_init_time", maintain_order=True):
        arr = np.full((len(ids), N_STEPS, len(columns)), np.nan)
        p = np.array([pos[s] for s in sub["time_series_id"].to_list()])
        s = steps_of(lead_hours=sub["lead_hours"].to_numpy())
        ok = (s >= 0) & (s < N_STEPS)
        arr[p[ok], s[ok]] = sub.select(columns).to_numpy()[ok]
        out[init[0]] = arr
    return out


def valid_mask(*, init: datetime) -> np.ndarray:
    """Return which of the 480 steps have a valid time before the final-test window."""
    t = np.array([init + (i + 1) * STEP for i in range(N_STEPS)])
    return t < END_OF_DEV
