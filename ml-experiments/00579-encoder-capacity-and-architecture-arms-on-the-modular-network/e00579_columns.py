# ruff: noqa: E501, I001, PT018
"""Extra datetime columns of experiment 579, built from the grid index of the valid time alone.

Every column is a deterministic function of the half-hour grid time (local civil time in
Europe/London), so no label, target, or weather enters it. `grid_features` returns one row per grid
index, and rows, pre-training rows, and the expected-input table index into it.

- F1 (14 columns): time-of-day harmonics 3 and 4 (sin and cos of the fractional local hour; the base
  holds harmonics 1 and 2 of the whole local hour), day-of-week harmonics 1 and 2 (sin and cos of 2
  pi k (weekday - 1) / 7), and day-of-year harmonics 1 to 3 (sin and cos of 2 pi k (day - 1) /
  365.25).
- F2 (15 columns): F1 plus an Economy 7 proxy: 1 inside the 00:30-07:30 local-time window, with a
  logistic ramp of 10 minutes at each edge. The real tariff window varies by supplier.
- P2 (1 column): the local half-hour-of-day index (0-47), consumed by the learnt per-Primary
  embedding and not fed to the network as a number.
"""

from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

KINDS: Final[tuple[str, ...]] = ("none", "F1", "F2", "P2")
EDGE_MINUTES: Final[float] = 10.0
E7_START_HOUR: Final[float] = 0.5
E7_END_HOUR: Final[float] = 7.5


def n_columns(*, kind: str) -> int:
    """Return how many columns the kind appends."""
    return {"none": 0, "F1": 14, "F2": 15, "P2": 1}[kind]


def column_names(*, kind: str) -> list[str]:
    """Return the names of the appended columns."""
    if kind == "none":
        return []
    if kind == "P2":
        return ["half_hour_local_index"]
    names = [f"{f}_hour_{k}" for k in (3, 4) for f in ("sin", "cos")]
    names += [f"{f}_dow_{k}" for k in (1, 2) for f in ("sin", "cos")]
    names += [f"{f}_doy_{k}" for k in (1, 2, 3) for f in ("sin", "cos")]
    return names + (["economy7_proxy"] if kind == "F2" else [])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def grid_features(*, kind: str) -> np.ndarray:
    """Return a (grid length, n columns) float32 array of the kind's columns at each grid time."""
    n = data.grid_length()
    if kind == "none":
        return np.zeros((n, 0), dtype=np.float32)
    t_us = prep422.GRID_START_US + np.arange(n, dtype=np.int64) * prep422.STEP_US
    local = (
        pl.Series("t", t_us)
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )
    hour = (local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60.0).to_numpy()
    if kind == "P2":
        return np.rint(hour * 2.0).astype(np.float32)[:, None]
    weekday = local.dt.weekday().cast(pl.Float64).to_numpy() - 1.0
    doy = local.dt.ordinal_day().cast(pl.Float64).to_numpy() - 1.0
    cols = []
    for k in (3, 4):
        cols += [np.sin(2 * np.pi * k * hour / 24.0), np.cos(2 * np.pi * k * hour / 24.0)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * weekday / 7.0), np.cos(2 * np.pi * k * weekday / 7.0)]
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    if kind == "F2":
        s = EDGE_MINUTES / 60.0 / 4.0
        cols.append(_sigmoid((hour - E7_START_HOUR) / s) - _sigmoid((hour - E7_END_HOUR) / s))
    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == n_columns(kind=kind)
    return out


def check(*, kind: str) -> str:
    """Assert the columns are finite and bounded; return a one-line example for the printed report."""
    g = grid_features(kind=kind)
    assert np.isfinite(g).all()
    if kind in {"F1", "F2"}:
        assert g.min() >= -1.0 - 1e-6 and g.max() <= 1.0 + 1e-6
    i = 12345
    t = np.datetime64(prep422.GRID_START_US + i * prep422.STEP_US, "us")
    return f"{kind}: grid index {i} (UTC {t}) -> {np.round(g[i], 3).tolist()}"
