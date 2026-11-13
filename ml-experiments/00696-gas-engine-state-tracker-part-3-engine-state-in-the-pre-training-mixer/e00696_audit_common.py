# ruff: noqa: E501, I001, PTH207
"""Paths and constants shared by the audit scripts of experiment 696 (`e00696_audit_*.py`).

The audit re-derives every number of the page from the raw per-fold fits with its own loss, pairing,
t-interval, and month-bootstrap code. It reuses only the truth. D8's truth is `ground_truth_primaries`.
The host's truth is experiment 695's `stacks/W0` row times the host's capacity. Outputs go to the
`audit/` folder of this experiment's data folder.
"""

import glob
from pathlib import Path

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR


E5 = Path(glob.glob(str(ML_EXPERIMENTS_DATA_DIR / "00695-*"))[0])
E6 = Path(glob.glob(str(ML_EXPERIMENTS_DATA_DIR / "00696-*"))[0])
OUT = E6 / "audit"
OUT.mkdir(exist_ok=True)
SH = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
Q = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
LV = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
HOST = 11
"""The host's series id: the series whose real reading equals the W0 stack row off the engine (see `e00696_audit_host.py`)."""
D8 = 26
D8_CAPACITY_MVA = 10.379
HOST_CAPACITY_MVA = 12.138


def fits_path(arm: str, month: str, seed: int) -> Path:
    """Return a stored fit file: A0 and W0 seeds 0 to 2 are experiment 695's, all others this experiment's."""
    root = E5 if arm in ("A0", "W0") and seed <= 2 else E6
    return (
        root
        / "runs"
        / arm
        / "arms"
        / "k1_5w_sp1"
        / "work"
        / "fits"
        / "P5"
        / f"{month}_d0_s{seed}.parquet"
    )
