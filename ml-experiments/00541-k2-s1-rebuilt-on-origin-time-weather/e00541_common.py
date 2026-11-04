# ruff: noqa: E501
"""Shared pieces of experiment 541.

The pieces build experiment 511's test rows with a choice of weather feed (origin-time or the
freshest run of experiment 404).
"""

from datetime import timedelta
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_data as d404  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final = ML_EXPERIMENTS_DATA_DIR / "00541-k2-s1-rebuilt-on-origin-time-weather"
WORK: Final = DATA / "work"
SCORES: Final = DATA / "scores"
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")


def weather_names() -> list[str]:
    """Return the 14 ensemble mean and spread column names."""
    return [f"mean_{w}" for w in prep.WEATHER] + [f"std_{w}" for w in prep.WEATHER]


def dense_by_primary() -> dict[int, d404.DenseInputs]:
    """Return experiment 404's freshest-run dense inputs for every Primary."""
    return {
        row["primary"]: d404.dense_inputs(site_row=row)
        for row in prep.primaries().iter_rows(named=True)
    }


def raw_test(*, month: str, mode: str, dense: dict[int, d404.DenseInputs]) -> dict[str, np.ndarray]:
    """Return experiment 511's test rows of a month with raw (unstandardised) non-lag inputs.

    `mode` is "origin" (the run published at the forecast origin, as 511 uses) or "fresh" (the 14 weather
    columns replaced by 404's freshest published run for the target's valid time). Rows with no freshest
    run are dropped in "fresh" mode. Same filters as `e00428_data.build_test`.
    """
    keep_cols, names, _ = data.non_lag_layout()
    assert names[:-2] == d404.lagless_columns(), "column order differs from 404's lag-free inputs"
    wnames = weather_names()
    start = data._us(t=prep.refit_of(month=month))
    stop = data._us(t=(prep.refit_of(month=month) + timedelta(days=32)).replace(day=1))
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["origin"] >= start)
            & (z["origin"] < stop)
            & (z["lead"] <= data.HEADLINE_MAX_HOURS)
            & (z["valid"] < data._us(t=data.HOLDOUT))
        )
        if not keep.any():
            continue
        x = z["x"][keep][:, keep_cols].astype(np.float32)
        target_idx = (z["valid"][keep] - prep422.GRID_START_US) // prep422.STEP_US
        origin_idx = (z["origin"][keep] - prep422.GRID_START_US) // prep422.STEP_US
        ok = np.ones(len(x), dtype=bool)
        if mode == "fresh":
            dn = dense[row["primary"]]
            pos = target_idx - dn.first
            ok = (pos >= 0) & (pos < len(dn.x))
            ok &= dn.valid[np.clip(pos, 0, len(dn.x) - 1)] == 1
            for name in wnames:
                x[ok, names.index(name)] = dn.x[pos[ok], dn.columns.index(name)]
        parts.append(
            {
                "x": x[ok],
                "origin": z["origin"][keep][ok],
                "valid": z["valid"][keep][ok],
                "origin_idx": origin_idx[ok].astype(np.int64),
                "target_idx": target_idx[ok].astype(np.int64),
                "lead": z["lead"][keep][ok],
                "primary": np.full(int(ok.sum()), row["primary"], dtype=np.int64),
                "dropped": np.array([int((~ok).sum())]),
            }
        )
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "dropped"}
    out["dropped"] = np.array([int(sum(p["dropped"][0] for p in parts))])
    return out
