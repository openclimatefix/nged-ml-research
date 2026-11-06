# ruff: noqa: E501, D205, D107
"""Experiment 592: the six gates, the fresh-power gap they share, and the C2 band built from each gate.

Everything reads stored forecasts (experiment 584's C1 and its lags-masked twin, and experiment
565's lag-free D5 network "E2"); nothing is refitted. Time is the half-hour grid index of
experiment 578; power is a share of effective capacity.

Gates (all return `(on, snap)`, booleans of shape (16 Primaries, days) by forecast origin day):

- `G0`: experiment 583's gate, a sustained gap of 3 days on lags at least 24 h old.
- `G1`, `G2`: the same gate with a sustain of 2 days and of 1 day.
- `G3`: the fast gate. One reading per origin day: the mean of (observed power minus E2's median)
  over the half-hours from the previous origin to the current origin minus 4.0 h. A sustain of 1
  reading.
- `G4`: `G3` that turns on only when an electrical neighbour shows a gap of the opposite sign above
  its own threshold.
- `G5`: `G3` that releases (a snap-back) as soon as the gap changes sign or falls by more than 60% in
  one day.
"""

from typing import Final

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00592-nra-snap-back-gate-faster-onset-and-snap-back-trigger"
WORK: Final = ML_EXPERIMENTS_DATA_DIR / NAME
GATES: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3", "G4", "G5")
GATE_LABEL: Final[dict[str, str]] = {
    "G0": "G0: 3-day sustain on day-old lags (experiment 583)",
    "G1": "G1: 2-day sustain on day-old lags",
    "G2": "G2: 1-day sustain on day-old lags",
    "G3": "G3: fresh power against lag-free expectation",
    "G4": "G4: G3 and a mirrored neighbour",
    "G5": "G5: G3 released at the snap-back",
}
FRESH_LEAD_H: Final[float] = 20.5
"""Rows with a lead under 20.5 h: forecasts are issued at 09:00 UTC with a first lead of 0.5 h, so the last valid time is the next origin minus 4.0 h (40 half-hours)."""
MIN_FRESH_HALF_HOURS: Final[int] = 36
"""Of the 40 half-hours in the fresh window, how many a reading needs."""
COLLAPSE: Final[float] = 0.4
"""G5: the new gap is below this share of the previous day's gap (a fall of more than 60%)."""


# ---------------------------------------------------------------- the fresh-power gap


def lag_free_median(*, t: c.X4Table) -> np.ndarray:
    """Return, per table row, experiment 565's lag-free D5 median (NaN where it is absent)."""
    e = a._load(root=a.K565_W, tag="D5", months=c.MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=(
            (pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US
        ).cast(pl.Int64),
        oidx=(
            (pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US)
            // c.k578.prep.STEP_US
        ).cast(pl.Int64),
    )
    kk = s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    kk_sorted = kk[order]
    mine = s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk_sorted, mine), 0, len(kk_sorted) - 1)
    ok = kk_sorted[pos] == mine
    med = e["q50"].to_numpy().astype(np.float32)[order[pos]]
    return np.where(ok, med, np.nan).astype(np.float32)


class FreshGap:
    """The daily mean of (observed power minus E2's median) over the fresh window, with the rows
    behind it, so that an injected shift can be added to exactly the half-hours it covers.

    Reading `d` (an origin day) uses the forecast issued at origin day `d - 1`, valid from that
    origin up to origin `d` minus 4.0 h, so every reading is available at origin `d`.
    """

    def __init__(self, *, t: c.X4Table, e2: np.ndarray, n_days: int) -> None:
        sel = np.flatnonzero((t.lead < FRESH_LEAD_H) & np.isfinite(e2))
        self.n_days = n_days
        self.p = t.prim[sel]
        self.idx = t.idx[sel]
        self.day = t.origin_day[sel] + 1
        self.res = (t.obs[sel] - e2[sel]).astype(np.float64)
        keep = self.day < n_days
        self.p, self.idx, self.day, self.res = (
            self.p[keep],
            self.idx[keep],
            self.day[keep],
            self.res[keep],
        )
        self.sum = np.zeros((16, n_days))
        self.cnt = np.zeros((16, n_days))
        np.add.at(self.sum, (self.p, self.day), self.res)
        np.add.at(self.cnt, (self.p, self.day), 1)
        self.by_p = {p: np.flatnonzero(self.p == p) for p in range(16)}

    def gap(self, *, shifts: list[tuple[int, int, int, float]] | None = None) -> np.ndarray:
        """Return (16, days) readings; `shifts` lists (Primary, onset, end, share of capacity) added
        to the observed power on the half-hours in [onset, end).
        """
        total = self.sum.copy()
        for p, onset, end, shift in shifts or []:
            rows = self.by_p[p]
            m = rows[(self.idx[rows] >= onset) & (self.idx[rows] < end)]
            np.add.at(total, (p, self.day[m]), shift)
        return np.where(self.cnt >= MIN_FRESH_HALF_HOURS, total / np.maximum(self.cnt, 1), np.nan)


def threshold(*, gap: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Return the per-Primary threshold `max(3 sd, 5% of capacity)` from days away from any event."""
    sd = np.array([np.nanstd(gap[p][~near[p]]) for p in range(gap.shape[0])])
    sd = np.where(np.isfinite(sd), sd, 0.0)
    return np.maximum(a.GATE_K_SD * sd, a.GATE_THETA)


# ---------------------------------------------------------------- the fresh-power gates


def gate_fresh(
    *,
    gap: np.ndarray,
    thr: np.ndarray,
    origin_days: list[int],
    neighbours: dict[int, list[int]] | None = None,
    snap_rule: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (on, snap) for gates G3 (default), G4 (`neighbours` given), and G5 (`snap_rule`).

    The gate turns on when one reading is above the Primary's threshold (G4 also needs a neighbour
    above its own threshold with the opposite sign on the same day). It turns off, which is the
    snap-back, when the reading falls below half the threshold; G5 also turns it off when the sign
    changes or the reading falls below `COLLAPSE` of the previous day's reading. A missing reading
    leaves the state as it was; a gap between fold months resets it.
    """
    n_days = gap.shape[1]
    big = np.isfinite(gap) & (np.abs(gap) > thr[:, None])
    sgn = np.where(big, np.sign(gap), 0.0)
    on = np.zeros((16, n_days), bool)
    snap = np.zeros((16, n_days), bool)
    for p in range(16):
        state, prev = False, None
        for d in origin_days:
            if prev is not None and d - prev > 1:
                state = False
            g = gap[p, d]
            if not state and sgn[p, d] != 0:
                if neighbours is None or any(
                    sgn[q, d] == -sgn[p, d] for q in neighbours.get(p, [])
                ):
                    state = True
            elif state and np.isfinite(g):
                released = abs(g) < 0.5 * thr[p]
                if snap_rule and prev == d - 1 and np.isfinite(gap[p, d - 1]):
                    before = gap[p, d - 1]
                    released |= (np.sign(g) != np.sign(before) and abs(g) > 0.5 * thr[p]) or (
                        abs(g) < COLLAPSE * abs(before)
                    )
                if released:
                    state = False
                    snap[p, d] = True
            prev = d
            on[p, d] = state
    return on, snap


def all_gates(
    *,
    old_gap: np.ndarray,
    old_thr: np.ndarray,
    fresh_gap: np.ndarray,
    fresh_thr: np.ndarray,
    origin_days: list[int],
    neighbours: dict[int, list[int]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return `(on, snap)` of every gate for one pair of gap arrays."""
    out = {
        f"G{i}": c.gate_state(
            gap=old_gap, thr=old_thr, origin_days=origin_days, release_days=1, m=m
        )
        for i, m in ((0, 3), (1, 2), (2, 1))
    }
    out["G3"] = gate_fresh(gap=fresh_gap, thr=fresh_thr, origin_days=origin_days)
    out["G4"] = gate_fresh(
        gap=fresh_gap, thr=fresh_thr, origin_days=origin_days, neighbours=neighbours
    )
    out["G5"] = gate_fresh(gap=fresh_gap, thr=fresh_thr, origin_days=origin_days, snap_rule=True)
    return out


def arm_masks(
    *, states: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return, per gate, (event mode, on) by origin day: the C2 band is wide while the gate is on and
    for 7 days after it releases (experiment 584's C2 = experiment 583's A2).
    """
    return {g: (c.hold_from(snap=snap) | on, on) for g, (on, snap) in states.items()}


def sign_changes(*, gap: np.ndarray) -> np.ndarray:
    """Return the (16, days) day-over-day product of sign, for plots of the fresh gap."""
    return np.sign(gap[:, 1:]) * np.sign(gap[:, :-1])


def years_of(*, pairs: int) -> float:
    """Return the Primary-years that `pairs` (Primary, origin day) pairs make."""
    return pairs / 365.25
