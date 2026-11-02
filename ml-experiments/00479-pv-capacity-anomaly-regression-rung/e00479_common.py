"""Constants shared by the scripts of experiment 479.

The plan is in `docs/ml-experiments/00479-pv-capacity-anomaly-regression-rung/result.md`.
"""

from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 479
SLUG: Final[str] = "pv-capacity-anomaly-regression-rung"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files; private and never committed."""
E474_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
)
SIGNED_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_ONLY_SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
OTHER_SITES: Final[tuple[str, ...]] = ("D18",)
"""D18 has history from 2025-03-01 only and is report-only. D15 is a generator case with no
estimate and is not run."""
ALL_SITES: Final[tuple[str, ...]] = SIGNED_SITES + MVA_ONLY_SITES + OTHER_SITES
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""E is left out: its history is too short for the read-out season-year (experiment 269)."""
INJECTION_MW: Final[tuple[float, ...]] = (2.0, 4.0)
N_SHUFFLES: Final[int] = 20
LATEST_SEASON_YEAR: Final[int] = 2025
"""July 2025 to June 2026, the season-year of the NGED table."""
DC_AC_CENTRE: Final[float] = 1.25
DC_AC_RATIOS: Final[tuple[float, ...]] = (1.1, 1.25, 1.5)
GATE_PLACEBO: Final[float] = 0.20
GATE_RECOVERY: Final[float] = 0.85
GATE_COVERAGE: Final[float] = 0.70
ALPHA: Final[float] = 0.05
POWER: Final[float] = 0.80
N_TESTS: Final[int] = 36
"""Bonferroni count: 6 signed-MW Primaries x 2 estimators (without and with the cloud terms) x 3
gates (placebo, recovery, coverage)."""
