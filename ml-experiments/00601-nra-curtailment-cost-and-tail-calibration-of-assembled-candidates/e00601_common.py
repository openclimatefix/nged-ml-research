# ruff: noqa: E501, D103
"""Experiment 601: shared constants and the per-row curtailment cost.

The cost is experiment 277's and 478's metric (`metrics.avoidable_curtailment_cost` over
`studies.node_curtailment`, node-only mode, breach ratio 20, price £2,000/MWh, leads 12 to 36 h), written
here as a per-row formula so that the injected bank (many events share one valid time) can be scored by
phase. `known_answer` checks the formula against the metric itself on the real forecasts.

For one forecast row of a node with limit `L` (MW), forecast q95 `u`, and metered flow `y`:
planned = 0.5 * max(0, u - L) MWh; breach = 0.5 * max(0, y - planned / 0.5 - L) MWh;
cost = price * (planned + 20 * breach); the perfect forecast costs price * 0.5 * max(0, y - L);
the avoidable cost is the difference.
"""

from typing import Final

import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00601-nra-curtailment-cost-and-tail-calibration-of-assembled-candidates"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
EXPORT: Final = ML_EXPERIMENTS_DATA_DIR / "backtest_export_candidates_2026-11-06"
EXPORT_PLAIN: Final = ML_EXPERIMENTS_DATA_DIR / "backtest_export_2026-11-06"
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
Q: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
SYSTEMS: Final = {  # name -> (folder, file prefix)
    "C0": (EXPORT_PLAIN, "C0"),
    "E2": (EXPORT_PLAIN, "E2"),
    "C1": (EXPORT, "C1"),
    "C1 + G4": (EXPORT, "C1_G4"),
    "A1": (EXPORT, "A1"),
    "A1 + G4": (EXPORT, "A1_G4"),
}
FRACTIONS: Final = (0.9, 0.75, 0.5)
HEADLINE_FRACTION: Final = 0.9
PRICE: Final = 2000.0
RATIO: Final = 20.0
LEAD_COST: Final = (12.0, 36.0)
SIX: Final = "six MW Primaries"
EIGHT: Final = "eight MVA Primaries"
REPORT_ONLY: Final = "report-only (D15 and D18)"
N_COUNTED: Final = 4
"""Counted tests: (C1 + G4 minus C1, A1 + G4 minus A1) x (event-free rows, 7 days after an injected end), cost at f = 0.9."""
BONFERRONI_ALPHA: Final = 0.05 / N_COUNTED
N_BOOT: Final = 20000


def avoidable_cost(*, u: np.ndarray, y: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Return the avoidable cost in pounds of each row: the forecast's cost minus the perfect forecast's."""
    planned = 0.5 * np.clip(u - limit, 0.0, None)
    needed = 0.5 * np.clip(y - limit, 0.0, None)
    breach = 0.5 * np.clip(y - planned / 0.5 - limit, 0.0, None)
    return PRICE * (planned + RATIO * breach) - PRICE * needed


def boot_interval(*, vals: np.ndarray, alpha: float, seed: int = 601) -> tuple[float, float, float]:
    """Return the mean and the two-sided percentile bootstrap interval (level 1 - alpha) over entries."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return (float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
