# ruff: noqa: E501, D103, PLR0915, PLR0912, C901
"""Experiment 601, injected bank: curtailment cost and tail calibration by phase, for one network.

No fit. Follows experiment 597's `score_bank` (experiment 581's injected load-shifts, the forecasts made
from the contaminated lags, the G4 gate rebuilt on the contaminated gap), and for each event, phase, and
system (`plain` = the network, `G4` = the assembled forecast, `twin` = the lags-masked twin, which reads no
lags and so is unaffected by the injection) writes tail calibration and the avoidable curtailment cost.

Units. The stored arrays are shares of capacity; the cost is scored in megawatts (share times the Primary's
capacity). The limit of a six-MW Primary at fraction f is f times its largest metered power before
2025-10-01. The cost columns are in pounds per Primary-day (mean over the 12 to 36 h rows times 48).
The bank scores every second half-hour (`idx % 2 == 0`) as experiment 597 does.

Usage: python e00601_bank.py ARM   (ARM: c1 or a1)
"""

import sys
from datetime import UTC, datetime

import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_score as s  # ty: ignore[unresolved-import]
import e00592_score as k592  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_score as s597  # ty: ignore[unresolved-import]
import e00601_common as k
import numpy as np
import polars as pl

BANDS = {0: "0-24 h", 1: "24-120 h"}
SYSTEMS = ("plain", "G4", "twin")
CUTOFF_IDX = int((datetime(2025, 10, 1, tzinfo=UTC).timestamp() * 1e6 - 1567296000000000) // 1800000000)
"""The half-hour grid index of 2025-10-01 00:00 UTC."""


def limits_mw(*, t) -> dict[float, np.ndarray]:
    """Return per fraction the limit in MW of every Primary (NaN outside the six-MW group)."""
    peak = np.nanmax(t.clean[:, :CUTOFF_IDX], axis=1) * t.capacity
    six = t.group == "signed-MW six"
    return {f: np.where(six, f * peak, np.nan) for f in k.FRACTIONS}


def tail_cols(*, y: np.ndarray, q: np.ndarray) -> dict[str, float]:
    m = dict(zip(c.METRICS, c.row_metrics(y=y, q=q).mean(axis=0).tolist(), strict=True))
    return {
        "ex95": 1.0 - m["le95"], "ex99": 1.0 - m["le99"], "lo05": 1.0 - m["ge05"], "lo01": 1.0 - m["ge01"],
        "cover90": m["cover90"], "pin05": m["pin05"], "pin95": m["pin95"], "pin99": m["pin99"],
        "over": float((q[:, c.IQ[0.5]] > y).mean()), "under": float((q[:, c.IQ[0.5]] < y).mean()),
    }  # fmt: skip


def cost_cols(*, q: np.ndarray, y: np.ndarray, cap: float, lim: dict[float, float]) -> dict[str, float]:
    return {
        f"cost_{f}": float(k.avoidable_cost(u=q[:, c.IQ[0.95]] * cap, y=y * cap, limit=np.full(len(y), lim[f])).mean() * 48)
        for f in k.FRACTIONS
    }  # fmt: skip


def score(*, arm: str) -> pl.DataFrame:
    inp = k597.Inputs(arm=arm)
    t, gi = inp.t, inp.gi
    lim_all = limits_mw(t=t)
    ev_all = pl.read_parquet(s.BANK_PARQUET)
    kk_all = s.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    months = [m for m in c.MONTHS if m in set(ev_all["month"].unique().to_list())]
    real_wide_all = inp.mode[t.prim, t.origin_day]
    out: list[dict] = []
    for m in months:
        in_m = t.month == m
        days_m = np.unique(t.idx[in_m] // 48)
        origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
        sl = slice(days_m.min(), days_m.max() + 1)
        for r in range(bank.N_PASSES):
            ev = ev_all.filter((pl.col("month") == m) & (pl.col("pass") == r)).filter(
                pl.col("primary").is_in([p for p in range(16) if t.group[p] in k597.GROUPS_SCORED])
            )
            if ev.height == 0:
                continue
            lo_p = {e["primary"]: e["onset_idx"] - 7 * 48 for e in ev.iter_rows(named=True)}
            hi_p = {e["primary"]: e["end_idx"] + 14 * 48 for e in ev.iter_rows(named=True)}
            lo = np.array([lo_p.get(p, 1 << 40) for p in range(16)])[t.prim]
            hi = np.array([hi_p.get(p, -1) for p in range(16)])[t.prim]
            rows = np.flatnonzero(in_m & t.free & (t.idx % 2 == 0) & (t.idx >= lo) & (t.idx < hi))
            vin = s.load_injected(t=t, month=m, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
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
                    shifts.append((n, e["onset_idx"], e["end_idx"], float(-shift * t.capacity[p] / t.capacity[n])))
            fresh = inp.fresh.gap(shifts=shifts)
            fresh[:, : origin_days_m[0] + 1] = np.nan
            fresh[:, origin_days_m[-1] + 1 :] = np.nan
            states = k592.g.all_gates(
                old_gap=old, old_thr=gi["thr"], fresh_gap=fresh, fresh_thr=inp.fresh_thr,
                origin_days=origin_days_m, neighbours=inp.nb,
            )  # fmt: skip
            wide_inj = k592.g.arm_masks(states=states)["G4"][0]
            y = t.obs[rows]
            m_q = t.arms["M"][rows]
            q_inj = s597.systems(v=vin, m=m_q, wide=wide_inj[t.prim[rows], t.origin_day[rows]])
            q_inj["twin"] = m_q
            lead = t.lead[rows]
            for e in ev.iter_rows(named=True):
                p, sgn, depth = e["primary"], float(e["sign"]), float(e["depth"])
                cap = float(t.capacity[p])
                lim = {f: float(lim_all[f][p]) for f in k.FRACTIONS}
                is_six = bool(np.isfinite(lim[k.FRACTIONS[0]]))
                for phase in k597.PHASES:
                    wlo, whi = sc.window(onset=e["onset_idx"], end=e["end_idx"], phase=phase)
                    in_win = (t.prim[rows] == p) & (t.idx[rows] >= wlo) & (t.idx[rows] < whi)
                    base = {
                        "month": m, "pass": r, "primary": p, "label": str(t.label[p]), "group": str(t.group[p]),
                        "sign": int(sgn), "depth": depth, "length_days": int(e["length_days"]), "phase": phase,
                    }  # fmt: skip
                    for band_i, band in BANDS.items():
                        sel = np.flatnonzero(in_win & (t.band[rows] == band_i))
                        if len(sel) < 60:
                            continue
                        for name in SYSTEMS:
                            wide_sel = wide_inj[t.prim[rows][sel], t.origin_day[rows][sel]] if name == "G4" else np.zeros(len(sel), bool)
                            out.append({**base, "system": name, "band": band, "n_rows": len(sel), "share_wide": float(wide_sel.mean()), **tail_cols(y=y[sel], q=q_inj[name][sel])})  # fmt: skip
                    if is_six:
                        sel = np.flatnonzero(in_win & (lead >= k.LEAD_COST[0]) & (lead < k.LEAD_COST[1]))
                        if len(sel) >= 30:
                            for name in SYSTEMS:
                                out.append({**base, "system": name, "band": "cost 12-36 h", "n_rows": len(sel), **cost_cols(q=q_inj[name][sel], y=y[sel], cap=cap, lim=lim),
                                            "n_exceed": int((y[sel] * cap > lim[0.9]).sum())})  # fmt: skip
            print("bank", m, r, flush=True)
    return pl.DataFrame(out, infer_schema_length=None)


def event_free_cost(*, arm: str) -> None:
    """Not needed: the event-free rows are scored from the stored exports by `e00601_real.py`."""


def summarise(*, tab: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Means over events with 95% event-bootstrap intervals, and the G4-minus-plain contrasts per event."""
    classes = {"all lengths": pl.lit(True), "15 and 30 days": pl.col("length_days") >= 15, "7 days": pl.col("length_days") == 7}
    groups = {k.EIGHT: pl.col("group") == "MVA-only eight", k.SIX: pl.col("group") == "signed-MW six"}
    metrics = ["ex95", "ex99", "lo05", "pin05", "pin95", "pin99", "over", "under", "cover90", *[f"cost_{f}" for f in k.FRACTIONS]]
    means, contr = [], []
    for gname, gexpr in groups.items():
        for cname, cexpr in classes.items():
            sub = tab.filter(gexpr & cexpr)
            for (phase, band, system), g in sub.group_by("phase", "band", "system"):
                rec = {"group": gname, "length_class": cname, "phase": phase, "band": band, "system": system, "n_events": g.height}
                for mname in metrics:
                    if mname not in g.columns:
                        continue
                    mean, lo, hi = k.boot_interval(vals=g[mname].to_numpy(), alpha=0.05)
                    rec[mname], rec[f"{mname}_lo"], rec[f"{mname}_hi"] = mean, lo, hi
                means.append(rec)
            key = ["month", "pass", "primary", "phase", "band"]
            plain = sub.filter(pl.col("system") == "plain")
            for other in ("G4", "twin"):
                oth = sub.filter(pl.col("system") == other)
                j = plain.join(oth, on=key, suffix="_o")
                for (phase, band), g in j.group_by("phase", "band"):
                    for mname in metrics:
                        if mname not in g.columns:
                            continue
                        d = (g[f"{mname}_o"] - g[mname]).to_numpy()
                        for level, alpha in (("95%", 0.05), (f"Bonferroni ({k.N_COUNTED})", k.BONFERRONI_ALPHA)):
                            mean, lo, hi = k.boot_interval(vals=d, alpha=alpha)
                            contr.append({"group": gname, "length_class": cname, "phase": phase, "band": band, "contrast": f"{other} minus plain", "measure": mname, "level": level, "n_events": int(np.isfinite(d).sum()), "diff": mean, "lo": lo, "hi": hi, "mde": (hi - lo) / 2})  # fmt: skip
    return pl.DataFrame(means, infer_schema_length=None), pl.DataFrame(contr, infer_schema_length=None)


def main(arm: str, *, summary_only: bool = False) -> None:
    out = k.WORK / arm
    out.mkdir(parents=True, exist_ok=True)
    if summary_only:
        tab = pl.read_csv(out / "bank_event_phase.csv", infer_schema_length=None)
    else:
        tab = score(arm=arm)
        tab.write_csv(out / "bank_event_phase.csv")
    means, contr = summarise(tab=tab)
    means.write_csv(out / "bank_summary.csv")
    contr.write_csv(out / "bank_contrasts.csv")
    print("done", tab.height, "rows", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], summary_only=len(sys.argv) > 2)
