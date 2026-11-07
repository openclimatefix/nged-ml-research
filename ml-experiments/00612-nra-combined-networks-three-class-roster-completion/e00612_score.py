# ruff: noqa: ANN001, E402, D103, E501, PLR0915, PLR0912, ANN202
"""Experiment 612: score one network with and without tail constants, at three lead bands (CPU).

For one fitted arm (`--arm NAME`, the folder `arms/NAME/work`), this scores four systems on the same
rows:

- `K2`: the network's own quantiles (no G4 gate);
- `K1`: the assembled forecast (the G4 union band while the gate is on or held);
- `K2+t` and `K1+t`: the same with the two-constant tail widening applied to the network's band and its
  lags-masked twin's band before the union (experiment 492's constants; experiment 608's method).

The tail constants of a month are fitted on the event-free rows of the *earlier* months only, per
metering group and lead band (24-120 h and 120-240 h), on the network's own plain quantiles. The first
month therefore has no `+t` system. Nothing reads a row from 2026-07-01 onwards. The G4 state is
recomputed from the arm's own lag-free twin (`--twin`, `--twin-tag`), never read from experiment 592.

Writes under `data/ml-experiments/00612-*/score/OUT/`:
`event_free_cells.csv` (one row per Primary, month, band, system: row count, coverage, tail shares,
pinball losses, CRPS), `cost_cells.csv` (curtailment cost per six-MW Primary and month at limits 0.9,
0.75, 0.5, leads 12-36 h), `tail_constants.csv`, `false_triggers.csv`, `bank_event_phase.csv`,
`bank_summary.csv`, `real_event_phase.csv`, and `event_free_rows.npz` (per-row CRPS).

Usage: python e00612_score.py OUT --arm NAME [--twin TWIN] [--twin-tag TAG] [--months ...] [--no-bank]
"""

import argparse
import json

_p = argparse.ArgumentParser()
_p.add_argument("out")
_p.add_argument("--arm", required=True)
_p.add_argument("--twin", default="k3_d5")
_p.add_argument("--twin-tag", default="D5")
_p.add_argument(
    "--months", nargs="+", default=["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
)
_p.add_argument("--no-bank", action="store_true")
_p.add_argument(
    "--width-factors",
    default=None,
    help="CSV (month, factor): add system K2w, the network rescaled about its median by the month's factor",
)
_p.add_argument(
    "--lagfree",
    default=None,
    help="score this lag-free arm's fits (D5/O2 tag) on the rows of --arm",
)
_p.add_argument("--lagfree-tag", default="D5")
MINE = _p.parse_args()

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00592_score as k592  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00600_common as k600  # ty: ignore[unresolved-import]
import e00601_bank as b601  # ty: ignore[unresolved-import]
import e00601_common as k601  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
OUT = ROOT / "score" / MINE.out
OUT.mkdir(parents=True, exist_ok=True)
MONTHS = tuple(MINE.months)
assert max(MONTHS) <= "2026-06", "July 2026 onwards is never opened"
assert MINE.lagfree or MINE.twin != "stored", "the G4 gate must read the arm's own lag-free twin"
c.MONTHS = MONTHS
BANDS = k600.BANDS
BAND_IDS = (0, 1, 2)
TAIL_BANDS = (1, 2)
GROUPS = k597.GROUPS_SCORED
PHASES = k597.PHASES
WFAC: dict[str, float] = (
    {}
    if MINE.width_factors is None
    else dict(
        __import__("polars").read_csv(MINE.width_factors).select("month", "factor").iter_rows()
    )
)
SYSTEMS = ("K2", "K1", "K2+t", "K1+t", *(("K2w",) if WFAC else ()))
GRID = np.round(np.arange(0.50, 1.801, 0.01), 2)
IQ = a.IQ
STAT_NAMES = (*c.METRICS, "crps")
SIDE = {1: c.IQ[0.05], -1: c.IQ[0.95]}

work = ROOT / "arms" / MINE.arm / "work"
k584.ARMS[MINE.arm] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
if MINE.twin and MINE.twin != "stored":
    twin_root = ROOT / "arms" / MINE.twin / "work"
    _orig = g.a._load

    def _load(*, root, tag, months):
        return (
            _orig(root=twin_root, tag=MINE.twin_tag, months=months)
            if tag == "D5"
            else _orig(root=root, tag=tag, months=months)
        )

    g.a._load = _load
k597.WORK = OUT


# ---------------------------------------------------------------- tail constants


def widen(*, q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return `q` with the lower tail (q01, q05) and upper tail (q95, q99) scaled about the median.

    `lower` and `upper` are one factor per row; a NaN factor leaves the row unchanged.
    """
    out = q.copy()
    med = q[:, IQ[0.5]]
    lo = np.where(np.isfinite(lower), lower, 1.0)
    hi = np.where(np.isfinite(upper), upper, 1.0)
    for tau in (0.01, 0.05):
        out[:, IQ[tau]] = med - lo * (med - q[:, IQ[tau]])
    for tau in (0.95, 0.99):
        out[:, IQ[tau]] = med + hi * (q[:, IQ[tau]] - med)
    return np.sort(out, axis=1)


def fit_constants(*, y: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Return the lower and upper factor that minimise the summed pinball loss of their two quantiles."""
    med = q[:, IQ[0.5]]
    best = []
    for names, sign in (((0.01, 0.05), -1.0), ((0.95, 0.99), 1.0)):
        loss = np.zeros(len(GRID))
        for tau in names:
            dist = sign * (q[:, IQ[tau]] - med)
            for i, f in enumerate(GRID):
                loss[i] += a.pinball(y=y, q=med + sign * f * dist, tau=tau).mean()
        best.append(float(GRID[int(loss.argmin())]))
    return best[0], best[1]


def row_factors(*, t) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Return per-row (lower, upper) factors (NaN where no constants apply) and the constants table."""
    lower = np.full(t.n, np.nan, np.float32)
    upper = np.full(t.n, np.nan, np.float32)
    consts = []
    v, y = t.arms["V"], t.obs
    for gname in GROUPS:
        for band in TAIL_BANDS:
            base = t.free & (t.group_of_row == gname) & (t.band == band)
            for mi in range(1, len(MONTHS)):
                now = (t.group_of_row == gname) & (t.band == band) & (t.month == MONTHS[mi])
                if not now.any():
                    continue
                # purge: a fitting row's target must be observed before the month's first origin
                earlier = base & np.isin(t.month, MONTHS[:mi]) & (t.idx < t.origin_idx[now].min())
                if earlier.sum() < 500 or now.sum() < 100:
                    continue
                lo, hi = fit_constants(y=y[earlier], q=v[earlier])
                lower[now], upper[now] = lo, hi
                consts.append(
                    {"group": gname, "band": BANDS[band], "month": MONTHS[mi], "lower": lo,
                     "upper": hi, "fit_rows": int(earlier.sum())}
                )  # fmt: skip
    return lower, upper, pl.DataFrame(consts)


# ---------------------------------------------------------------- systems and row statistics


def systems(*, v, m, wide, lower, upper, wf=None) -> dict[str, np.ndarray]:
    """Return the quantiles of the systems; `+t` systems equal the plain ones where no factor applies.

    `K2w` (only with `--width-factors`) is the network scaled about its median by the per-row factor `wf`.
    """
    wv, wm = widen(q=v, lower=lower, upper=upper), widen(q=m, lower=lower, upper=upper)
    out = {
        "K2": v,
        "K1": k597.assemble(v=v, m=m, wide=wide),
        "K2+t": wv,
        "K1+t": k597.assemble(v=wv, m=wm, wide=wide),
    }
    if wf is not None:
        med = v[:, IQ[0.5] : IQ[0.5] + 1]
        out["K2w"] = (med + wf[:, None] * (v - med)).astype(np.float32)
    return out


def month_factor(*, month_col: np.ndarray) -> np.ndarray | None:
    """Return one width factor per row from `--width-factors` (1.0 for a month without one), or None."""
    if not WFAC:
        return None
    return (
        np.array([WFAC.get(str(x), 1.0) for x in month_col], np.float32)
        if len(month_col)
        else np.zeros(0, np.float32)
    )


def row_stats(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [c.row_metrics(y=y, q=q), a.crps_from_quantiles(y=y, q=q)[:, None].astype(np.float32)],
        axis=1,
    )


def means(*, rows_metrics: np.ndarray, sgn: float | None) -> dict[str, float]:
    m = dict(zip(STAT_NAMES, rows_metrics.mean(axis=0).tolist(), strict=True))
    if sgn is not None:
        m["nra_side95"] = m["ge05"] if sgn > 0 else m["le95"]
        m["nra_side99"] = m["ge01"] if sgn > 0 else m["le99"]
    return m


def follow(*, q, twin, sgn: float, depth: float) -> dict[str, float]:
    j = SIDE[int(sgn)]
    return {
        "follow_q50": float(((q[:, c.IQ[0.5]] - twin[:, c.IQ[0.5]]) * sgn).mean() / depth),
        "follow_safety": float(((q[:, j] - twin[:, j]) * sgn).mean() / depth),
    }


def has_tail(*, name: str, month: str) -> bool:
    return not name.endswith("+t") or month != MONTHS[0]


# ---------------------------------------------------------------- event-free cells


def event_free(*, inp, lower, upper) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    t = inp.t
    wide = inp.mode[t.prim, t.origin_day]
    sy = systems(
        v=t.arms["V"],
        m=t.arms["M"],
        wide=wide,
        lower=lower,
        upper=upper,
        wf=month_factor(month_col=t.month),
    )
    st = {n: row_stats(y=t.obs, q=q) for n, q in sy.items()}
    j = {
        n: STAT_NAMES.index(n)
        for n in ("cover90", "pin05", "pin95", "width90", "crps", "ge05", "le95")
    }
    cells = []
    for prim in range(16):
        for mo in MONTHS:
            for band in BAND_IDS:
                sel = t.free & (t.prim == prim) & (t.month == mo) & (t.band == band)
                if sel.sum() < 100:
                    continue
                for n in SYSTEMS:
                    if not has_tail(name=n, month=mo):
                        continue
                    r = st[n][sel]
                    cells.append(
                        {"prim": prim, "label": str(t.label[prim]), "group": str(t.group[prim]),
                         "month": mo, "band": BANDS[band], "system": n, "n_rows": int(sel.sum()),
                         "cover90": float(r[:, j["cover90"]].mean()),
                         "below_q05": float(1.0 - r[:, j["ge05"]].mean()),
                         "above_q95": float(1.0 - r[:, j["le95"]].mean()),
                         "pin05": float(r[:, j["pin05"]].mean()), "pin95": float(r[:, j["pin95"]].mean()),
                         "width90": float(r[:, j["width90"]].mean()), "crps": float(r[:, j["crps"]].mean()),
                         "share_wide": float(wide[sel].mean()) if n.startswith("K1") else 0.0}
                    )  # fmt: skip
    lim = b601.limits_mw(t=t)
    cost = []
    lead_ok = (t.lead >= k601.LEAD_COST[0]) & (t.lead < k601.LEAD_COST[1])
    for prim in range(16):
        if t.group[prim] != "signed-MW six":
            continue
        cap = float(t.capacity[prim])
        for mo in MONTHS:
            sel = t.free & lead_ok & (t.prim == prim) & (t.month == mo)
            if sel.sum() < 100:
                continue
            for n in SYSTEMS:
                if not has_tail(name=n, month=mo):
                    continue
                q = sy[n][sel]
                rec = {
                    "prim": prim,
                    "label": str(t.label[prim]),
                    "month": mo,
                    "system": n,
                    "n_rows": int(sel.sum()),
                }
                for f in k601.FRACTIONS:
                    rec[f"cost_{f}"] = float(
                        k601.avoidable_cost(u=q[:, c.IQ[0.95]] * cap, y=t.obs[sel] * cap,
                                            limit=np.full(int(sel.sum()), float(lim[f][prim]))).mean() * 48
                    )  # fmt: skip
                cost.append(rec)
    rows = {
        "prim": t.prim, "origin_idx": t.origin_idx, "idx": t.idx, "band": t.band, "free": t.free,
        "group": t.group_of_row.astype(str), "month": t.month.astype(str),
        **{f"crps_{n}": st[n][:, j["crps"]] for n in SYSTEMS},
    }  # fmt: skip
    return pl.DataFrame(cells), pl.DataFrame(cost), rows


# ---------------------------------------------------------------- the verified pair


def real_events(*, inp, lower, upper) -> pl.DataFrame:
    t = inp.t
    wide = inp.mode[t.prim, t.origin_day]
    sy = systems(
        v=t.arms["V"],
        m=t.arms["M"],
        wide=wide,
        lower=lower,
        upper=upper,
        wf=month_factor(month_col=t.month),
    )
    st = {n: row_stats(y=t.nra, q=q) for n, q in sy.items()}
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    out = []
    for e in ev.iter_rows(named=True):
        sgn = 1.0 if e["depth_share"] > 0 else -1.0
        for phase in PHASES:
            lo, hi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
            for band in BAND_IDS:
                sel = np.flatnonzero(
                    (t.prim == e["primary"]) & (t.idx >= lo) & (t.idx < hi) & (t.band == band)
                )
                if len(sel) < 100:
                    continue
                for n in SYSTEMS:
                    if n.endswith("+t") and not np.isfinite(lower[sel]).all():
                        continue
                    out.append(
                        {"event": e["name"], "group": e["group"], "length_days": e["length_days"],
                         "phase": phase, "band": BANDS[band], "system": n, "n_rows": len(sel),
                         "share_origin_days_wide": float(wide[sel].mean()),
                         **means(rows_metrics=st[n][sel], sgn=sgn)}
                    )  # fmt: skip
    return pl.DataFrame(out)


# ---------------------------------------------------------------- the injected bank


def score_bank(*, inp, lower, upper) -> tuple[pl.DataFrame, list[dict]]:
    t, gi = inp.t, inp.gi
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in MONTHS if m in set(ev_all["month"].unique().to_list())]
    lim_all = b601.limits_mw(t=t)
    rows_out: list[dict] = []
    trig_rows: list[dict] = []
    real_wide_all = inp.mode[t.prim, t.origin_day]
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        sl = slice(days_m.min(), days_m.max() + 1)
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in GROUPS])
            )
            if ev.height == 0:
                continue
            lo_p = {e["primary"]: e["onset_idx"] - 7 * 48 for e in ev.iter_rows(named=True)}
            hi_p = {e["primary"]: e["end_idx"] + 14 * 48 for e in ev.iter_rows(named=True)}
            lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
            hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
            rows = np.flatnonzero(in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi))
            vin = s.load_injected(
                t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys
            )
            ok = np.isfinite(vin).all(axis=1)
            rows, vin = rows[ok], vin[ok]
            if len(rows) == 0:
                continue
            old = np.full_like(gi["gap"], np.nan)
            old[:, sl] = gi["gap"][:, sl]
            shifts: list[tuple[int, int, int, float]] = []
            for e in ev.iter_rows(named=True):
                p, shift = e["primary"], e["sign"] * e["depth"]
                old[p, e["onset_idx"] // 48 : e["end_idx"] // 48] += shift
                shifts.append((p, e["onset_idx"], e["end_idx"], float(shift)))
                nbs = inp.nb.get(p, [])
                if nbs:
                    n = nbs[(r + p) % len(nbs)]
                    shifts.append((n, e["onset_idx"], e["end_idx"], float(-shift * t.capacity[p] / t.capacity[n])))  # fmt: skip
            fresh = inp.fresh.gap(shifts=shifts)
            fresh[:, : origin_days_m[0] + 1] = np.nan
            fresh[:, origin_days_m[-1] + 1 :] = np.nan
            states = k592.g.all_gates(
                old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr,
                origin_days=origin_days_m, neighbours=inp.nb,
            )  # fmt: skip
            wide_inj = k592.g.arm_masks(states=states)["G4"][0]
            y = t.obs[rows]
            v_clean, m_q = t.arms["V"][rows], t.arms["M"][rows]
            lw, up = lower[rows], upper[rows]
            wide_rows = wide_inj[t.prim[rows], t.origin_day[rows]]
            q_inj = systems(
                v=vin,
                m=m_q,
                wide=wide_rows,
                lower=lw,
                upper=up,
                wf=month_factor(month_col=t.month[rows]),
            )
            q_twin = systems(
                v=v_clean,
                m=m_q,
                wide=real_wide_all[rows],
                lower=lw,
                upper=up,
                wf=month_factor(month_col=t.month[rows]),
            )
            st = {n: row_stats(y=y, q=q) for n, q in q_inj.items()}
            lead = t.lead[rows]
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                cap = float(t.capacity[p])
                is_six = t.group[p] == "signed-MW six"
                evf = pl.DataFrame(
                    [{**e, "length_days": float(e["length_days"]), "first_day": int(days_m.min())}]
                )
                trig_rows.extend(
                    k592.trigger_rows(
                        ev=evf, states={"G4": states["G4"]}, source=f"bank {m} r{r}",
                        extra={"month": m, "pass": r, "has_neighbour": bool(inp.nb.get(p)),
                               "depth": depth, "sign": int(sgn)},
                    )
                )  # fmt: skip
                for phase in PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    in_win = (t.prim[rows] == p) & (t.idx[rows] >= wlo) & (t.idx[rows] < whi)
                    base = {
                        "month": m, "pass": r, "primary": p, "group": str(t.group[p]), "sign": int(sgn),
                        "depth": depth, "length_days": int(e["length_days"]), "phase": phase,
                    }  # fmt: skip
                    for band in BAND_IDS:
                        sel = np.flatnonzero(in_win & (t.band[rows] == band))
                        if len(sel) < 60:
                            continue
                        for n in SYSTEMS:
                            if not has_tail(name=n, month=m):
                                continue
                            wide_sel = (
                                wide_rows[sel] if n.startswith("K1") else np.zeros(len(sel), bool)
                            )
                            rows_out.append(
                                {**base, "band": BANDS[band], "system": n, "n_rows": len(sel),
                                 "share_origin_days_wide": float(wide_sel.mean()),
                                 **means(rows_metrics=st[n][sel], sgn=sgn),
                                 **follow(q=q_inj[n][sel], twin=q_twin[n][sel], sgn=sgn, depth=depth)}
                            )  # fmt: skip
                    if is_six:
                        sel = np.flatnonzero(
                            in_win & (lead >= k601.LEAD_COST[0]) & (lead < k601.LEAD_COST[1])
                        )
                        if len(sel) >= 30:
                            for n in SYSTEMS:
                                if not has_tail(name=n, month=m):
                                    continue
                                rec = {
                                    **base,
                                    "band": "cost 12-36 h",
                                    "system": n,
                                    "n_rows": len(sel),
                                }
                                for f in k601.FRACTIONS:
                                    rec[f"cost_{f}"] = float(
                                        k601.avoidable_cost(u=q_inj[n][sel][:, c.IQ[0.95]] * cap, y=y[sel] * cap,
                                                            limit=np.full(len(sel), float(lim_all[f][p]))).mean() * 48
                                    )  # fmt: skip
                                rows_out.append(rec)
            print("bank", m, r, flush=True)
    return pl.DataFrame(rows_out, infer_schema_length=None), trig_rows


def summarise_bank(*, tab: pl.DataFrame) -> pl.DataFrame:
    metrics = [*STAT_NAMES, "nra_side95", "nra_side99", "follow_q50", "follow_safety",
               *[f"cost_{f}" for f in k601.FRACTIONS]]  # fmt: skip
    classes = {
        "all lengths": pl.lit(True),
        "15 and 30 days": pl.col("length_days") >= 15,
        "7 days": pl.col("length_days") == 7,
    }
    out = []
    for gname, gexpr in (("MVA-only eight", pl.col("group") == "MVA-only eight"),
                         ("signed-MW six", pl.col("group") == "signed-MW six"),
                         ("both groups", pl.lit(True))):  # fmt: skip
        for cname, cexpr in classes.items():
            sub0 = tab.filter(gexpr & cexpr)
            for mset, sub in (
                ("all", sub0),
                ("tail months", sub0.filter(pl.col("month") != MONTHS[0])),
            ):
                for (phase, band, system), grp in sub.group_by("phase", "band", "system"):
                    if mset == "all" and system.endswith("+t"):
                        continue
                    rec = {"month_set": mset, "group": gname, "length_class": cname, "phase": phase, "band": band,
                       "system": system, "n_events": grp.height}  # fmt: skip
                    for mname in metrics:
                        if mname not in grp.columns or grp[mname].null_count() == grp.height:
                            continue
                        mean, lo, hi = sc._boot_mean(vals=grp[mname].drop_nulls().to_numpy())
                        rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                    out.append(rec)
    return pl.DataFrame(out, infer_schema_length=None)


def lag_free_quantiles(*, t) -> np.ndarray:
    """Return the lag-free arm's quantiles aligned to the table rows (NaN where absent), sorted."""
    e = g.a._load(root=ROOT / "arms" / MINE.lagfree / "work", tag=MINE.lagfree_tag, months=MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=((pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
        oidx=((pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
    )  # fmt: skip
    kk = s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    mine = s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk[order], mine), 0, len(kk) - 1)
    ok = kk[order][pos] == mine
    q = e.select(list(a.QCOLS)).to_numpy().astype(np.float32)[order[pos]]
    q[~ok] = np.nan
    return np.sort(q, axis=1)


def main() -> None:
    k600.WORK = OUT
    inp = k600.Inputs(arm=MINE.arm)
    t = inp.t
    if MINE.lagfree:
        q = lag_free_quantiles(t=t)
        ok = np.isfinite(q).all(axis=1)
        print("lag-free rows present:", float(ok.mean()), flush=True)
        t.arms["V"] = np.where(ok[:, None], q, 0.0).astype(np.float32)
        t.arms["M"] = t.arms["V"]
        t.free = t.free & ok
        MINE.no_bank = True
    chk = inp.check_gate_state()
    (OUT / "gate_check.json").write_text(json.dumps({"equals_592_stored_state": chk}))
    print("rows", t.n, "months", MONTHS, flush=True)
    lower, upper, consts = row_factors(t=t)
    consts.write_csv(OUT / "tail_constants.csv")
    print(consts, flush=True)
    ft = k592.false_triggers(inp=inp).filter(pl.col("gate").is_in(["G0", "G4"]))
    ft.write_csv(OUT / "false_triggers.csv")
    cells, cost, rows = event_free(inp=inp, lower=lower, upper=upper)
    cells.write_csv(OUT / "event_free_cells.csv")
    cost.write_csv(OUT / "cost_cells.csv")
    np.savez(OUT / "event_free_rows.npz", **rows)
    real_events(inp=inp, lower=lower, upper=upper).write_csv(OUT / "real_event_phase.csv")
    np.save(OUT / "mode_g4.npy", inp.mode)
    if MINE.no_bank:
        print("done (no bank)", flush=True)
        return
    tab, trig = score_bank(inp=inp, lower=lower, upper=upper)
    tab.write_csv(OUT / "bank_event_phase.csv")
    summarise_bank(tab=tab).write_csv(OUT / "bank_summary.csv")
    pl.DataFrame(trig, infer_schema_length=None).write_csv(OUT / "trigger_times.csv")
    print("done", tab.height, "bank rows", flush=True)


if __name__ == "__main__":
    main()
