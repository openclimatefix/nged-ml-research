"""Experiment 663: shared paths."""

from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00663-end-detector-for-the-prevailing-forecast-followup-to-654"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
E654: Final = ML_EXPERIMENTS_DATA_DIR / "00654-two-output-network-end-to-end-with-end-detector"
E652: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00652-two-output-forecast-with-explicit-rules-and-survival-hazard"
)
E647: Final = ML_EXPERIMENTS_DATA_DIR / "00647-two-output-forecast-departure-head-on-frozen-k1"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
PER_DAY: Final[int] = 48
N_REWIRINGS: Final[int] = 3
HARD_NEGATIVE_SHARE: Final[float] = 0.3
