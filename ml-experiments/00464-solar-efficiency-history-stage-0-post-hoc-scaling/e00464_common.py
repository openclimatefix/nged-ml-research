# ruff: noqa: E501
"""Shared constants for experiment 464 (stage 0: post-hoc efficiency scaling of stored solar forecasts).

Farms appear only as labels A to F (Solar A to Solar F). Every table written here lives under
`data/ml-experiments/`.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00464-solar-efficiency-history-stage-0-post-hoc-scaling"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
RATIO_DIR: Final[Path] = DATA_DIR / "ratios"
SCORE_DIR: Final[Path] = DATA_DIR / "scores"
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
E430_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
)

FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
GROUP_DRIFT: Final[tuple[str, ...]] = ("A", "B", "D")
"""Farms whose output level changed between years (experiment 453)."""
GROUP_OTHER: Final[tuple[str, ...]] = ("C", "E", "F")
NETWORKS: Final[tuple[str, ...]] = ("a2_member", "g0_onepass")
PRIMARY_NETWORK: Final[str] = "a2_member"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
REFITS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, 1, tzinfo=UTC) for y, m in ((2025, 9), (2025, 12), (2026, 3), (2026, 6))
)
"""Quarterly refit dates of experiment 430's stored forecasts; a row belongs to the latest one at or before its origin."""

# Validity rules for a half-hour.
STUCK_RUN: Final[int] = 4
STUCK_TOLERANCE: Final[float] = 1e-4
STUCK_MIN_POWER: Final[float] = 0.02
ZERO_DAYLIGHT_CLEAR_SKY: Final[float] = 200.0
SPIKE_POWER: Final[float] = 1.5
# Daily ratio.
MIN_EXPECTED: Final[float] = 0.10
MIN_HALF_HOURS_PER_DAY: Final[int] = 4
NEAR_CLEAR_INDEX: Final[float] = 0.9
NEAR_CLEAR_MIN_CLEAR_SKY: Final[float] = 300.0
# Methods.
WINDOWS: Final[tuple[int, ...]] = (7, 14, 28, 56, 112)
MIN_VALID_SHARE: Final[float] = 0.40
HALF_LIVES: Final[tuple[int, ...]] = (14, 28, 56)
KALMAN_LEVEL_SHARE: Final[float] = 0.01
KALMAN_TREND_SHARE: Final[float] = 0.0001
DELIVERY_DAYS: Final[int] = 2
"""The newest day used is this many calendar days before the origin's date."""
MULTIPLIER_RANGE: Final[tuple[float, float]] = (0.5, 1.5)
METHODS: Final[tuple[str, ...]] = (
    *[f"med{w}" for w in WINDOWS],
    *[f"ew{h}" for h in HALF_LIVES],
    "kalman",
)
SOURCES: Final[tuple[str, ...]] = ("a", "b", "c")
PRIMARY_SOURCES: Final[tuple[str, ...]] = ("b", "c")

DRAWS: Final[int] = 200_000
N_PRIMARY_CONTRASTS: Final[int] = 108
BONFERRONI_ALPHA: Final[float] = 0.05 / N_PRIMARY_CONTRASTS
