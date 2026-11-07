# ruff: noqa: D103, E501, E402
"""Experiment 608: tail constants on K1, and the lag-free K3 beside K1 and K2, on event-free rows (CPU).

Two-constant tail widening (experiment 492): per metering group and lead band, a lower factor scales
the distance of q01 and q05 from the median and an upper factor scales the distance of q95 and q99,
each the grid value that minimises the summed pinball loss of its two quantiles. The constants for a
month are fitted on event-free rows of the earlier months only, so the first month (2025-10) is never
scored with constants. They widen the network's band and its lags-masked twin's band before the G4
union, so G4 is unchanged. Nothing reads a row from 2026-07-01 onwards.

Also scores the lag-free network K3 (D5 with E3), K1 (G4) and K2 (no G4) on the same event-free rows.

Usage: python e00608_tails.py [--arm k1_d5] [--twin k3_d5]
"""

import argparse

_p = argparse.ArgumentParser()
_p.add_argument("--arm", default="k1_d5")
_p.add_argument("--twin", default="k3_d5")
_p.add_argument("--twin-tag", default="D5")
_p.add_argument("--out", default="tails_k1_d5")
MINE = _p.parse_args()

import e00582_arms as a  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00608-nra-combined-networks-per-class-and-leave-one-out"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
GROUPS = ("MVA-only eight", "signed-MW six")
GRID = np.round(np.arange(0.50, 1.801, 0.01), 2)
c.MONTHS = MONTHS
work = ROOT / "arms" / MINE.arm / "work"
k584.ARMS[MINE.arm] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
twin_root = ROOT / "arms" / MINE.twin / "work"
_orig = g.a._load
g.a._load = lambda *, root, tag, months: (
    _orig(root=twin_root, tag=MINE.twin_tag, months=months) if tag == "D5" else _orig(root=root, tag=tag, months=months)
)  # fmt: skip
k597.WORK = ROOT / "score" / MINE.out
inp = k597.Inputs(arm=MINE.arm)
t = inp.t
wide = inp.mode[t.prim, t.origin_day]
Q = a.QCOLS
IQ = a.IQ


def widen(*, q: np.ndarray, lower: float, upper: float) -> np.ndarray:
    out = q.copy()
    med = q[:, IQ[0.5]]
    for tau in (0.01, 0.05):
        out[:, IQ[tau]] = med - lower * (med - q[:, IQ[tau]])
    for tau in (0.95, 0.99):
        out[:, IQ[tau]] = med + upper * (q[:, IQ[tau]] - med)
    return np.sort(out, axis=1)


def fit(*, y: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    med = q[:, IQ[0.5]]
    best = []
    for names, sign in ((((0.01, 0.05)), -1.0), (((0.95, 0.99)), 1.0)):
        loss = np.zeros(len(GRID))
        for tau in names:
            dist = sign * (q[:, IQ[tau]] - med)
            for i, f in enumerate(GRID):
                loss[i] += a.pinball(y=y, q=med + sign * f * dist, tau=tau).mean()
        best.append(float(GRID[int(loss.argmin())]))
    return best[0], best[1]


def load_lag_free() -> np.ndarray:
    e = _orig(root=twin_root, tag=MINE.twin_tag, months=MONTHS)
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
    kk = g.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    mine = g.s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk[order], mine), 0, len(kk) - 1)
    ok = kk[order][pos] == mine
    q = e.select(list(Q)).to_numpy().astype(np.float32)[order[pos]]
    q[~ok] = np.nan
    return np.sort(q, axis=1)


def score(*, y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    return {
        "cover90": float(((q[:, IQ[0.05]] <= y) & (y <= q[:, IQ[0.95]])).mean()),
        "below_q05": float((y < q[:, IQ[0.05]]).mean()),
        "above_q95": float((y > q[:, IQ[0.95]]).mean()),
        "pin05": float(a.pinball(y=y, q=q[:, IQ[0.05]], tau=0.05).mean()),
        "pin95": float(a.pinball(y=y, q=q[:, IQ[0.95]], tau=0.95).mean()),
        "crps": float(a.crps_from_quantiles(y=y, q=q).mean()),
    }


V, M, y = t.arms["V"], t.arms["M"], t.obs
free = t.free
k3 = load_lag_free()
out = []
consts = []
for gname in GROUPS:
    for band in (0, 1):
        sel_all = free & (t.group_of_row == gname) & (t.band == band)
        for mi in range(1, len(MONTHS)):
            earlier = sel_all & np.isin(t.month, MONTHS[:mi])
            now = sel_all & (t.month == MONTHS[mi])
            if earlier.sum() < 500 or now.sum() < 100:
                continue
            lower, upper = fit(y=y[earlier], q=V[earlier])
            consts.append(
                {
                    "group": gname,
                    "band": band,
                    "month": MONTHS[mi],
                    "lower": lower,
                    "upper": upper,
                    "fit_rows": int(earlier.sum()),
                }
            )
            idx = np.flatnonzero(now)
            v, m = V[idx], M[idx]
            base = np.where(wide[idx, None], a.union_band(lag=v, masked=m), v)
            wv, wm = widen(q=v, lower=lower, upper=upper), widen(q=m, lower=lower, upper=upper)
            tail = np.where(wide[idx, None], a.union_band(lag=wv, masked=wm), wv)
            ok = np.isfinite(k3[idx]).all(axis=1)
            for name, q in (
                ("K1 (G4)", base),
                ("K1 + tail constants", tail),
                ("K2 (no G4)", v),
                ("K2 + tail constants", wv),
                ("K3 (lag-free)", k3[idx]),
            ):
                sel = ok if name.startswith("K3") else np.ones(len(idx), bool)
                out.append(
                    {
                        "group": gname,
                        "band": band,
                        "month": MONTHS[mi],
                        "arm": name,
                        "n_rows": int(sel.sum()),
                        **score(y=y[idx][sel], q=q[sel]),
                    }
                )
tab = pl.DataFrame(out)
(ROOT / "score" / MINE.out).mkdir(parents=True, exist_ok=True)
tab.write_csv(ROOT / "score" / MINE.out / "tails_event_free.csv")
pl.DataFrame(consts).write_csv(ROOT / "score" / MINE.out / "tail_constants.csv")
w = (
    tab.group_by("group", "band", "arm")
    .agg(
        [
            (pl.col(k) * pl.col("n_rows")).sum() / pl.col("n_rows").sum()
            for k in ("cover90", "below_q05", "above_q95", "pin05", "pin95", "crps")
        ]
    )
    .sort("group", "band", "arm")
)
with pl.Config(tbl_rows=40, tbl_cols=12):
    print(w)
