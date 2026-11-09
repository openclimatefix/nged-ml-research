"""Experiment 669: shared constants of the one-year backtest at the bulk and grid supply points.

Reuses the inputs of experiment 633 (read only): the ensemble summaries of every 00Z run from
2024-04-01 to 2026-10-31, the thermal-lag tables, the extended ERA5 and power grid.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points"
ROOT: Final[Path] = Path(os.environ.get("E00669_ROOT") or ML_EXPERIMENTS_DATA_DIR / NAME)
"""The work folder; `E00669_ROOT` redirects it (the audit re-runs one fold elsewhere)."""
SHARED: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
MONTHS: Final[tuple[str, ...]] = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
FIRST_ORIGIN_DAY: Final[datetime] = datetime(2025, 11, 1, tzinfo=UTC)
N_ORIGINS: Final[int] = 365
ORIGIN_HOUR_CONTRACT: Final[int] = 12
"""The contract's origin hour; the network runs at its own 09:00 origin and is relabelled."""
HALF_HOURS: Final[int] = 672
"""Valid times per origin: 12:00 on the origin day to 11:30 on day 14."""
TAG: Final[str] = "K_LG35_B1"
SPREAD_NOTE: Final[str] = "no spread factor is applied"
GROUP_DIR: Final[dict[str, str]] = {"bsp": "bsp", "gsp": "gsp"}
GROUP_SITES: Final[dict[str, tuple[str, ...]]] = {"bsp": ("D13", "D17"), "gsp": ("D1", "D6")}
NODE_SITES: Final[tuple[str, ...]] = ("D13", "D17", "D1", "D6")
MAX_LEAD_HOURS: Final[float] = 351.0


def origin_days() -> list[datetime]:
    """Return the 365 origin days (00:00 UTC) of the year."""
    return [FIRST_ORIGIN_DAY + timedelta(days=i) for i in range(N_ORIGINS)]
