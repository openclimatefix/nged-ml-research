"""Experiment 648: the screen margin, from stored forecasts (CPU, no fits).

Reads experiment 613's event-free cells for the seed sets {0, 1} and {2, 3, 4} on the three screen
months, 24-120 h. The seed standard deviation sigma is estimated from the disagreement of the two
seed-set means (variance of their difference = sigma^2 (1/2 + 1/3)), pooled over months. A one-seed
arm minus a one-seed reference has standard deviation sigma sqrt(2), so the minimum detectable effect
is 1.96 sqrt(2) sigma. The other candidate margin is experiment 447's 0.064 points scaled by the
network's CRPS over S1's pooled 3.011 points (experiment 447, 3 folds).
"""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORE = ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds" / "score"
MONTHS = ["2025-10", "2026-01", "2026-06"]
S1_POINTS = 3.011
MARGIN_447 = 0.064


def pooled(*, arm: str, system: str) -> dict[str, float]:
    """Return the row-weighted CRPS (points of capacity) per month, 24-120 h, all Primaries."""
    df = pl.read_csv(SCORE / arm / "event_free_cells.csv").filter(
        (pl.col("band") == "24-120 h") & (pl.col("system") == system) & pl.col("month").is_in(MONTHS)
    )
    g = df.group_by("month").agg((pl.col("crps") * pl.col("n_rows")).sum() / pl.col("n_rows").sum())
    return {m: 100 * c for m, c in g.iter_rows()}


for name, arm, system in (("K1", "k1_d5", "K2"), ("candidate", "c2_e3", "K2")):
    try:
        a, b = pooled(arm=f"{arm}_s01", system=system), pooled(arm=f"{arm}_s234", system=system)
    except Exception as err:  # noqa: BLE001
        print(name, "unavailable:", err)
        continue
    diff = np.array([a[m] - b[m] for m in MONTHS])
    sigma = float(np.sqrt(np.mean(diff**2) / (1 / 2 + 1 / 3)))
    mde = 1.96 * np.sqrt(2) * sigma
    level = float(np.mean([*a.values(), *b.values()]))
    scaled = MARGIN_447 * level / S1_POINTS
    print(
        f"{name}: CRPS {level:.3f} points, sigma {sigma:.4f}, MDE {mde:.4f}, "
        f"scaled 447 margin {scaled:.4f}, margin {max(mde, scaled):.4f}; set diffs {diff.round(4)}"
    )
