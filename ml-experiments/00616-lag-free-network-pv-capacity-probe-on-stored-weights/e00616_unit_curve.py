# ruff: noqa: E501
"""Experiment 616: the pooled unit PV curve `u(G, elevation, T)` of experiment 269, saved as a small table.

Experiment 269 fits one learned response table per metered farm (Solar A to F): output over capacity
as a function of global horizontal irradiance `G` and sun elevation, with the heat loss divided out.
The pooled curve is the mean of the six tables. The temperature derating is applied afterwards, as the
fit of experiment 129 does: `u = u0 * max(1 - 0.004 * (T + 25 u0 - 25), 0)`. Only the pooled table (a
shape, not a series) is written, to the private data folder.

Usage: python e00616_unit_curve.py
"""

import e00269_common as c269  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00616-lag-free-network-pv-capacity-probe-on-stored-weights"
    / "unit_curve.npz"
)
DERATING_PER_KELVIN = 0.004
CELL_RISE_K = 25.0
REFERENCE_C = 25.0


def main() -> None:
    """Fit the six farm tables at Primary D5's temperature and save their mean."""
    cache = c269.load_cache()
    tables = [c269.learned_response(cache=cache, site="D5", farm=f) for f in c269.ALL_FARMS]
    values = np.mean([t.values for t in tables], axis=0)
    grid = c269.RESPONSE_GRID
    np.savez(OUT, irradiance=grid.irradiance_w_m2, elevation=grid.elevation_deg, values=values)
    print("saved", OUT, values.shape, float(values.min()), float(values.max()))


if __name__ == "__main__":
    main()
