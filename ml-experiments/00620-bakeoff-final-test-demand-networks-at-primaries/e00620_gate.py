# ruff: noqa: E501
"""Experiment 620: the G4 gate state over any window of origin days, from a lag-free forecast and the observed power.

Experiment 592 defined G4 and stored its dev-month state. The gate reads only observed power, the
lag-free network's median (K3, experiment 565's "E2"), and the neighbours' readings, so its state
does not depend on K1. This module repeats the same arithmetic as `e00592_common.FreshGap`,
`gate_fresh` and `arm_masks` on a frame of lag-free forecasts that covers a different window, and
takes the per-Primary threshold from experiment 592's stored dev months (`fresh_thr`), so no
threshold is fitted on the test window.

`check_dev()` rebuilds the dev-month state from experiment 565's stored forecasts and compares it
with experiment 592's stored `gate_state.npz`.
"""

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MIN_FRESH = g.MIN_FRESH_HALF_HOURS
FRESH_LEAD_H = g.FRESH_LEAD_H
STEP_US = c.k578.prep.STEP_US
START_US = c.k578.prep.GRID_START_US


def frozen_threshold() -> np.ndarray:
    """Return experiment 592's stored per-Primary G4 threshold (share of capacity), fitted on dev months."""
    return np.load(g.WORK / "gate_state.npz")["fresh_thr"]


def neighbours() -> dict[int, list[int]]:
    """Return each Primary's metered neighbours, as experiment 582's table does."""
    sm = c.k578.sites()
    label = np.array(sm["label"].to_list())
    lab2p = dict(zip(label.tolist(), range(len(label)), strict=True))
    return {
        lab2p[x]: [lab2p[y] for y in v if y in lab2p]
        for x, v in a.b400.nged_neighbours().items()
        if x in lab2p
    }


def fresh_gap(*, lag_free: pl.DataFrame, clean: np.ndarray, n_days: int) -> np.ndarray:
    """Return (16, n_days) readings: the daily mean of observed minus the lag-free median over the fresh window.

    `lag_free` holds the seed-mean forecast rows (`time_series_id`, `power_fcst_init_time`,
    `valid_time`, `q50`). Reading day `d` uses the forecast issued on origin day `d - 1`.
    """
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    f = lag_free.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=((pl.col("valid_time").dt.epoch("us") - START_US) // STEP_US).cast(pl.Int64),
        oidx=((pl.col("power_fcst_init_time").dt.epoch("us") - START_US) // STEP_US).cast(pl.Int64),
    ).with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    f = f.filter(pl.col("lead") < FRESH_LEAD_H)
    p, i = f["prim"].to_numpy(), f["idx"].to_numpy()
    obs = clean[p, i]
    e2 = f["q50"].to_numpy().astype(np.float32)
    ok = np.isfinite(obs) & np.isfinite(e2)
    day = (f["oidx"].to_numpy() // 48 + 1)[ok]
    keep = day < n_days
    p, day = p[ok][keep], day[keep]
    res = (obs[ok] - e2[ok]).astype(np.float64)[keep]
    total, cnt = np.zeros((16, n_days)), np.zeros((16, n_days))
    np.add.at(total, (p, day), res)
    np.add.at(cnt, (p, day), 1)
    return np.where(cnt >= MIN_FRESH, total / np.maximum(cnt, 1), np.nan)


def g4_mode(
    *, gap: np.ndarray, origin_days: list[int], thr: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mode, on, snap), each (16, days): `mode` is True while the band is wide (gate on, or in the 7 days after it turned off)."""
    thr = frozen_threshold() if thr is None else thr
    on, snap = g.gate_fresh(gap=gap, thr=thr, origin_days=origin_days, neighbours=neighbours())
    return c.hold_from(snap=snap) | on, on, snap


def check_dev() -> dict[str, bool]:
    """Rebuild the dev-month G4 state from experiment 565's stored forecasts; compare with 592's."""
    z = np.load(g.WORK / "gate_state.npz")
    n_days = z["fresh_gap"].shape[1]
    e = a._load(root=a.K565_W, tag="D5", months=c.MONTHS)
    gap = fresh_gap(lag_free=e, clean=c.k578.clean(), n_days=n_days)
    origin_days = sorted(
        {
            int(x)
            for x in (e["power_fcst_init_time"].dt.epoch("us").to_numpy() - START_US)
            // STEP_US
            // 48
        }
    )
    _, on, snap = g4_mode(gap=gap, origin_days=origin_days, thr=z["fresh_thr"])
    return {
        "fresh_gap": bool(np.allclose(z["fresh_gap"], gap, equal_nan=True)),
        "G4_on": bool((z["G4_on"] == on).all()),
        "G4_snap": bool((z["G4_snap"] == snap).all()),
    }


if __name__ == "__main__":
    print(check_dev())
