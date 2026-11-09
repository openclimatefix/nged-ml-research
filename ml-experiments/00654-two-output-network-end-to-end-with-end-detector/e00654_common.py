# ruff: noqa: D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654: shared paths."""

from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00654-two-output-network-end-to-end-with-end-detector"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
