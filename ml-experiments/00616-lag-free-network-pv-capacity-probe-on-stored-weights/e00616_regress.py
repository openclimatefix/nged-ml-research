# ruff: noqa: E501
"""Experiment 616: the donor regression `y = a_slot + C * u(G, T) + b * (T - mean T) [+ cloud terms]`.

`y` is the probed response in MW. `u` is the pooled unit PV curve of experiment 269 (MW of AC
output per MW of AC capacity), read at the donor's rescaled irradiance, the probe date's sun
elevation, and the donor's temperature. The slot intercepts absorb everything that depends on the
fixed datetime inputs.
`C` is the negative of the slope on `u`, because a larger `u` lowers net demand.
"""

from typing import Final

import numpy as np
import polars as pl
from studies.pv_plant_fit import ResponseGrid, ResponseTable
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00616-lag-free-network-pv-capacity-probe-on-stored-weights"
DERATING_PER_KELVIN: Final[float] = 0.004
CELL_RISE_K: Final[float] = 25.0
REFERENCE_C: Final[float] = 25.0


def unit_table() -> ResponseTable:
    """Load the pooled unit curve of experiment 269 (saved by `e00616_unit_curve.py`)."""
    z = np.load(OUT / "unit_curve.npz")
    return ResponseTable(
        grid=ResponseGrid(irradiance_w_m2=z["irradiance"], elevation_deg=z["elevation"]),
        values=z["values"],
    )


def unit_curve(
    *, table: ResponseTable, g: np.ndarray, elevation: np.ndarray, temp: np.ndarray
) -> np.ndarray:
    """Return `u`: the learned response with the temperature derating applied."""
    u0 = table.predict(
        irradiance_w_m2=np.clip(g, 0.0, 1100.0), elevation_deg=np.clip(elevation, 5.0, 65.0)
    )
    cell = temp + CELL_RISE_K * u0
    return u0 * np.clip(1.0 - DERATING_PER_KELVIN * (cell - REFERENCE_C), 0.0, None)


def add_regressors(
    *, rows: pl.DataFrame, table: ResponseTable, g_col: str = "g_in"
) -> pl.DataFrame:
    """Add `u`, the half-hour cloud index `cs`, the donor day's mean cloud index `cd`, and the donor day."""
    g = rows[g_col].to_numpy()
    u = unit_curve(
        table=table, g=g, elevation=rows["elev_probe"].to_numpy(), temp=rows["temp"].to_numpy()
    )
    cs = np.clip(1.0 - g / np.maximum(rows["clear_probe"].to_numpy(), 20.0), 0.0, 1.0)
    out = rows.with_columns(
        u=pl.Series(u), cs=pl.Series(cs), day=pl.col("donor_us") // 86_400_000_000
    )
    return out.with_columns(cd=pl.col("cs").mean().over(["primary", "date", "day"]))
