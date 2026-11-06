# ruff: noqa: E501, D103, ANN001, ANN201, PERF401
"""Experiment 588: event gallery for experiment 584's candidate forecast. No fit.

Draws day-ahead forecasts of C0 (v3+D5, experiment 572) and C2 (P5 recipe with synthetic
load-shifts plus the gated band, experiment 584) on verified events, labelled events, injected
events, and event-free windows, from the forecasts experiment 584 stored. Demand series only: every
plotted series is a Primary substation. Run experiment 584's `e00584_score.py c0` and `c1` first.

Usage: DATA_PATH_INTERNAL=... PYTHONPATH=(all ml-experiments folders) python e00588_gallery.py
"""

import shutil
from pathlib import Path

import altair as alt
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00584_figures as fig4  # ty: ignore[unresolved-import]
import e00584_score as sc4  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00588-nra-event-gallery-for-the-candidate-forecast"
DATA = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
V580 = ML_EXPERIMENTS_DATA_DIR / "00580-verified-load-shift-event-set-and-label-audit/work"
IQ = c.IQ
W = 620
GREY, ORANGE, BLUE, SKY = "#6b6b6b", theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_SKY
INJECT_MONTH = "2026-03"
DESIGNS = [  # (length in days, depth share of capacity, sign): spans depth, length, and sign
    (length, depth, sign)
    for length in (7, 15, 30)
    for depth, sign in ((0.15, 1), (0.30, -1), (0.50, 1), (0.50, -1))
]
STATUS_TEXT = {
    "verified transfer": "verified by the power readings",
    "wrong dates or sign": "label has wrong dates or sign; the power readings show a different event",
    "rejected": "not seen by the power rule (label not verified by the power readings)",
    "too short to check": "too short to check",
}


def to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


def panel(*, x, q0, q2, obs, nra, label_nra=None, strips=(), span=None, xdom, title, height=150,
          xtitle=None, ydom=None):  # fmt: skip
    """One panel: C0 and C2 q05-q95 bands and medians, observed power, the NRA, and the label's NRA."""
    d = {"x": x, "c0lo": q0[:, IQ[0.05]] * 100, "c0hi": q0[:, IQ[0.95]] * 100, "c0m": q0[:, IQ[0.5]] * 100,
         "c2lo": q2[:, IQ[0.05]] * 100, "c2hi": q2[:, IQ[0.95]] * 100, "c2m": q2[:, IQ[0.5]] * 100,
         "obs": obs * 100, "nra": nra * 100}  # fmt: skip
    if label_nra is not None:
        d["lab"] = label_nra * 100
    df = pl.DataFrame({k: to_none(np.asarray(v, float)) for k, v in d.items()})
    lo = np.nanmin([np.nanmin(d["c0lo"]), np.nanmin(d["c2lo"]), np.nanmin(d["obs"])]) - 2
    hi = np.nanmax([np.nanmax(d["c0hi"]), np.nanmax(d["c2hi"]), np.nanmax(d["obs"])]) + 2
    sc_y = alt.Scale(domain=ydom or [float(lo), float(hi)])
    xax = alt.X(
        "x:Q",
        title=xtitle,
        scale=alt.Scale(domain=list(xdom), zero=False, nice=False),
        axis=alt.Axis(tickMinStep=1),
    )
    base = alt.Chart(df)
    layers = []
    dom = sc_y.domain
    if len(strips):
        bar = pl.DataFrame(list(strips)).with_columns(
            y0=pl.lit(dom[0]), y1=pl.lit(dom[0] + 0.05 * (dom[1] - dom[0]))
        )
        layers.append(
            alt.Chart(bar)  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.9, color=SKY, clip=True)
            .encode(x="x0:Q", x2="x1:Q", y=alt.Y("y0:Q", scale=sc_y), y2="y1:Q")
        )
    if span is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"a": [span[0]], "b": [span[1]]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color="#999999", opacity=0.12, clip=True)
            .encode(x="a:Q", x2="b:Q")
        )
    layers += [
        base.mark_area(color=GREY, opacity=0.30, aria=False, clip=True).encode(x=xax, y=alt.Y("c0lo:Q", scale=sc_y, title="% of capacity"), y2="c0hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_area(color=ORANGE, opacity=0.35, aria=False, clip=True).encode(x=xax, y=alt.Y("c2lo:Q", scale=sc_y), y2="c2hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_line(color=GREY, strokeWidth=1, aria=False, clip=True).encode(x=xax, y=alt.Y("c0m:Q", scale=sc_y)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="#b33000", strokeWidth=1, aria=False, clip=True).encode(x=xax, y=alt.Y("c2m:Q", scale=sc_y)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="black", strokeWidth=0.8, aria=False, clip=True).encode(x=xax, y=alt.Y("obs:Q", scale=sc_y)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="black", strokeDash=[3, 2], strokeWidth=0.8, aria=False, clip=True).encode(x=xax, y=alt.Y("nra:Q", scale=sc_y)),  # ty: ignore[unresolved-attribute]
    ]  # fmt: skip
    if label_nra is not None:
        layers.append(
            base.mark_line(  # ty: ignore[unresolved-attribute]
                color=BLUE, strokeDash=[5, 3], strokeWidth=1.2, aria=False, clip=True
            ).encode(x=xax, y=alt.Y("lab:Q", scale=sc_y))
        )
    return alt.layer(*layers).properties(
        width=W, height=height, title=alt.TitleParams(title, fontSize=11, anchor="start")
    )


def save(*, charts, title, out, shared=True):
    alt.vconcat(*charts).properties(
        title=alt.TitleParams(title, fontSize=11, anchor="start")
    ).resolve_scale(x="shared" if shared else "independent").save(str(DOCS / out))


def strips_for(*, t, rows, on, hold, prim, onset):
    out = []
    for d in sorted({int(d) for d in t.origin_day[rows]}):
        if on[prim, d] or hold[prim, d]:
            out.append({"x0": (d + 1) - onset / 48.0, "x1": (d + 2) - onset / 48.0})
    return out


def c2_of(*, t, rows, v, on, hold):
    blocks = c.building_blocks(v=v, m=t.arms["M"][rows])
    return c.combine(
        base=blocks,
        hold=hold[t.prim[rows], t.origin_day[rows]],
        on=on[t.prim[rows], t.origin_day[rows]],
    )["A2"]


def covers(*, q, nra):
    return (q[:, IQ[0.05]] <= nra) & (nra <= q[:, IQ[0.95]])


def phase_cover(*, t, rows, q0, q2, nra, onset, end) -> list[dict]:
    out = []
    idx = t.idx[rows]
    for ph, (lo, hi) in {"before": (onset - 7 * 48, onset), "days0-3": (onset, min(onset + 144, end)),
                          "day4-end": (onset + 144, end), "after0-7": (end, end + 7 * 48)}.items():  # fmt: skip
        s = (idx >= lo) & (idx < hi) & np.isfinite(q0).all(axis=1) & np.isfinite(nra)
        if s.sum() < 10:
            continue
        out.append({"phase": ph, "n_rows": int(s.sum()), "C0": float(covers(q=q0[s], nra=nra[s]).mean()),
                    "C2": float(covers(q=q2[s], nra=nra[s]).mean()),
                    "width_C0": float((q0[s, IQ[0.95]] - q0[s, IQ[0.05]]).mean() * 100),
                    "width_C2": float((q2[s, IQ[0.95]] - q2[s, IQ[0.05]]).mean() * 100)})  # fmt: skip
    return out


def get_rows(*, t, look, prim, lo, hi, mask=None):
    rows = f3.day_ahead_rows(t=t, prim=prim, lo=lo, hi=hi, mask=mask)
    r0 = look.rows0(t1=t, rows=rows)
    return rows, r0


def c0_q(*, look, r0):
    return np.where((r0 >= 0)[:, None], look.t0.arms["V"][np.maximum(r0, 0)], np.nan)


def verified(*, t, look, st, tables):
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    for e in ev.iter_rows(named=True):
        onset, end, p = e["onset_idx"], e["end_idx"], e["primary"]
        rows, r0 = get_rows(t=t, look=look, prim=p, lo=onset - 14 * 48, hi=end + 14 * 48)
        if len(rows) < 200:
            continue
        q0, q2 = (
            c0_q(look=look, r0=r0),
            c2_of(t=t, rows=rows, v=t.arms["V"][rows], on=st["on"], hold=st["hold"]),
        )
        x = (t.idx[rows] - onset) / 48.0
        length = (end - onset) / 48.0
        strips = strips_for(t=t, rows=rows, on=st["on"], hold=st["hold"], prim=p, onset=onset)
        date = sc.k.date_of(onset)
        ttl = [f"{e['label']}: verified load-shift of {e['depth_mw']:+.1f} MW for {length:.0f} days, starting {date}",
               "Black solid: observed power. Black dashed: the NRA (observed minus the fitted load-shift). Grey: C0 median and q05-q95 band. Orange: C2 median and band.",
               "Grey shade: the load-shift. Sky-blue bar along the bottom: origin days with the gate on or held."]  # fmt: skip
        kw = {
            "x": x,
            "q0": q0,
            "q2": q2,
            "obs": t.obs[rows],
            "nra": t.nra[rows],
            "strips": strips,
            "span": (0, length),
        }
        save(charts=[panel(**kw, xdom=(-14, length + 14), title="Whole window", xtitle="Days since the load-shift began"),
                     panel(**kw, xdom=(length, length + 7), title="Zoom: the 7 days after the load-shift ends", xtitle="Days since the load-shift began")],
             title=ttl, out=f"verified_{e['label']}_{date}.svg", shared=False)  # fmt: skip
        for r in phase_cover(t=t, rows=rows, q0=q0, q2=q2, nra=t.nra[rows], onset=onset, end=end):
            tables["verified"].append({"event": f"{e['label']} {date}", **r})


def labelled(*, t, look, st, tables):
    ev = pl.read_parquet(V580 / "events_labelled_status.parquet")
    lo, hi = int(t.idx.min()), int(t.idx.max())
    sel = ev.filter(
        (pl.col("label_end_idx") >= lo)
        & (pl.col("label_onset_idx") <= hi)
        & (pl.col("label_length_days") >= 3)
    )
    lab_to_p = {str(lab): i for i, lab in enumerate(t.label)}
    sel = sel.filter(pl.col("label").is_in(list(lab_to_p))).with_columns(
        adepth=pl.col("label_depth_mw").abs()
    )
    sel = sel.sort("label_length_days", "adepth", descending=True)
    for e in sel.iter_rows(named=True):
        p = lab_to_p[e["label"]]
        onset, end = e["label_onset_idx"], e["label_end_idx"]
        rows, r0 = get_rows(t=t, look=look, prim=p, lo=onset - 7 * 48, hi=end + 7 * 48)
        if len(rows) < 100 or len(tables["labelled"]) >= 12:
            continue
        q0, q2 = (
            c0_q(look=look, r0=r0),
            c2_of(t=t, rows=rows, v=t.arms["V"][rows], on=st["on"], hold=st["hold"]),
        )
        assert abs(float(t.capacity[p]) - e["capacity_mw"]) < 0.05 * e["capacity_mw"], (
            e["label"],
            t.capacity[p],
            e["capacity_mw"],
        )
        depth = e["label_depth_mw"] / e["capacity_mw"]
        inside = (t.idx[rows] >= onset) & (t.idx[rows] < end)
        lab_nra = t.obs[rows] - np.where(inside, depth, 0.0)
        length = (end - onset) / 48.0
        status = STATUS_TEXT.get(e["final_class"], e["final_class"])
        if e["final_class"].startswith("unpaired"):
            status = "one-sided step only, no mirrored step at a second Primary"
        partner = ev.filter(
            (pl.col("label_onset_idx") == onset)
            & (pl.col("label_end_idx") == end)
            & (pl.col("event_id") != e["event_id"])
        )["label"].to_list()
        date = sc.k.date_of(onset)
        ttl = [f"{e['label']}, labelled event {e['event_id']}: {e['label_depth_mw']:+.2f} MW for {length:.1f} days from {date}. 580 status: {status}." + (f" Same dates as the label at {', '.join(partner)}." if partner else ""),
               "NGED's labels are known to be imperfect, so a forecast that does not follow the label is not an error. Black solid: observed power. Blue dashed: observed power minus the labelled depth.",
               "Grey: C0 median and q05-q95 band. Orange: C2 median and band. Grey shade: the labelled event. Sky-blue bar along the bottom: gate on or held."]  # fmt: skip
        x = (t.idx[rows] - onset) / 48.0
        strips = strips_for(t=t, rows=rows, on=st["on"], hold=st["hold"], prim=p, onset=onset)
        save(charts=[panel(x=x, q0=q0, q2=q2, obs=t.obs[rows], nra=np.full(len(rows), np.nan), label_nra=lab_nra, strips=strips, span=(0, length),
                           xdom=(-7, length + 7), title="Day-ahead forecast", xtitle="Days since the labelled start")],
             title=ttl, out=f"labelled_{e['event_id']:03d}_{e['label']}_{length:.0f}d.svg")  # fmt: skip
        tables["labelled"].append({"event_id": e["event_id"], "primary": e["label"], "onset": date, "length_days": round(length, 1),
                                   "label_depth_mw": e["label_depth_mw"], "status_580": e["final_class"], "partner": ",".join(partner),
                                   "gate_on_origin_days": int(sum(st["on"][p, d] for d in {int(d) for d in t.origin_day[rows]}))})  # fmt: skip


def injected(*, t, look, gi, tables):
    ev_all = pl.read_parquet(s3.BANK_PARQUET).filter(pl.col("month") == INJECT_MONTH)
    kk = s3.key(t.prim, t.origin_idx, t.idx)
    order, sorted_keys = np.argsort(kk), np.sort(kk)
    in_m = t.month == INJECT_MONTH
    days_m = np.unique(t.idx[in_m] // 48)
    origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
    gap_m = np.full_like(gi["gap"], np.nan)
    gap_m[:, days_m.min() : days_m.max() + 1] = gi["gap"][:, days_m.min() : days_m.max() + 1]
    groups = ("MVA-only eight", "signed-MW six")
    for length, depth, sign in DESIGNS:
        charts = []
        for g in groups:
            cand = ev_all.filter((pl.col("length_days") == length) & (pl.col("depth") == depth) & (pl.col("sign") == sign)
                                 & pl.col("primary").is_in([p for p in range(16) if t.group[p] == g])).sort("pass", "primary")  # fmt: skip
            e = cand.row(0, named=True)
            r, p = e["pass"], e["primary"]
            onset, end = e["onset_idx"], e["end_idx"]
            rows = f3.day_ahead_rows(t=t, prim=p, lo=onset - 7 * 48, hi=end + 7 * 48, mask=in_m)
            vin = s3.load_injected(
                t=t, month=INJECT_MONTH, r=r, rows=rows, order=order, sorted_keys=sorted_keys
            )
            ok = np.isfinite(vin).all(axis=1)
            rows, vin = rows[ok], vin[ok]
            gap_i = gap_m.copy()
            for ee in ev_all.filter(pl.col("pass") == r).iter_rows(named=True):
                gap_i[ee["primary"], ee["onset_idx"] // 48 : ee["end_idx"] // 48] += (
                    ee["sign"] * ee["depth"]
                )
            on, snap = c.gate_state(
                gap=gap_i, thr=gi["thr"], origin_days=origin_days_m, release_days=1
            )
            hold = c.hold_from(snap=snap)
            r0 = look.rows0(t1=t, rows=rows)
            q0 = fig4.c0_injected(month=INJECT_MONTH, r=r, rows0=r0)
            q2 = c2_of(t=t, rows=rows, v=vin, on=on, hold=hold)
            x = (t.idx[rows] - onset) / 48.0
            nra = t.obs[rows]
            ph = phase_cover(t=t, rows=rows, q0=q0, q2=q2, nra=nra, onset=onset, end=end)
            for row in ph:
                tables["injected"].append(
                    {
                        "length_days": length,
                        "depth": depth,
                        "sign": sign,
                        "group": g,
                        "primary": str(t.label[p]),
                        "pass": r,
                        **row,
                    }
                )
            started = x.min() <= 0
            charts.append(panel(x=x, q0=q0, q2=q2, obs=nra, nra=nra, strips=strips_for(t=t, rows=rows, on=on, hold=hold, prim=p, onset=onset),
                                span=(0.0 if started else float(x.min()), (end - onset) / 48.0), xdom=(float(x.min()), (end - onset) / 48.0 + 7),
                                title=f"{t.label[p]} ({g})", height=120, xtitle="Days since the injected shift began" if g == groups[-1] else None))  # fmt: skip
        sg = "up" if sign > 0 else "down"
        save(charts=charts, shared=False, out=f"injected_{length}d_{sg}_{int(depth * 100)}pct.svg",
             title=[f"Injected load-shift in the lag inputs only: {sign * depth * 100:+.0f}% of capacity for {length} days, March 2026",
                    "The target stays clean, so the observed power (black) is also the NRA. Grey: C0 median and q05-q95 band. Orange: C2 median and band.",
                    "Grey shade: the injected shift (it began before the plotted window if the x axis starts above 0). Sky-blue bar along the bottom: gate on or held."])  # fmt: skip


def event_free(*, t, look, st, tables):
    free_prim = {
        g: [p for p in range(16) if t.group[p] == g] for g in ("MVA-only eight", "signed-MW six")
    }
    wanted = [
        ("MVA-only eight", "2025-10"),
        ("signed-MW six", "2026-03"),
        ("MVA-only eight", "2026-06"),
        ("signed-MW six", "2026-06"),
    ]
    charts = []
    for g, month in wanted:
        for p in free_prim[g]:
            m = (t.month == month) & (t.prim == p)
            start = int(t.idx[m].min()) + 7 * 48
            rows, _r0 = get_rows(
                t=t, look=look, prim=p, lo=start, hi=start + 14 * 48, mask=(t.month == month)
            )
            if len(rows) < 600 or not t.free[rows].all() or not np.isfinite(t.obs[rows]).all():
                continue
            v = t.arms["V"][rows]
            q2 = c2_of(t=t, rows=rows, v=v, on=st["on"], hold=st["hold"])
            diff = np.abs(q2 - v).max(axis=1)
            date = sc.k.date_of(start)
            x = (t.idx[rows] - start) / 48.0
            tables["event_free"].append({"primary": str(t.label[p]), "group": g, "start": date, "n_rows": len(rows), "share_rows_c2_equals_c1": float((diff == 0).mean()),
                                         "max_abs_diff_any_quantile_pct": float(diff.max() * 100), "gate_on_origin_days": int(sum(st["on"][p, d] for d in {int(d) for d in t.origin_day[rows]}))})  # fmt: skip
            charts.append(panel(x=x, q0=v, q2=q2, obs=t.obs[rows], nra=t.obs[rows], height=110, xdom=(0, 14),
                                title=f"{t.label[p]} ({g}), 14 days from {date}", xtitle="Days since the window began" if len(charts) == len(wanted) - 1 else None))  # fmt: skip
            break
    save(charts=charts, out="event_free_windows.svg",
         title=["Event-free windows: C1 (grey) and C2 (orange) on 14 days with no labelled or verified load-shift within 7 days",
                "The two bands lie on top of each other while the gate is off. Black: observed power. Window rule: the lowest-numbered Primary of the group whose 14 days starting on day 8 of the month hold no event."])  # fmt: skip


def summaries(*, tables):
    DOCS.mkdir(parents=True, exist_ok=True)
    src = ML_EXPERIMENTS_DATA_DIR / "00584-nra-candidate-pack-x6/score_pack"
    b = pl.read_csv(src / "bank_summary.csv").filter(
        (pl.col("band") == "24-120 h")
        & (pl.col("length_class") == "15 and 30 days")
        & pl.col("arm").is_in(["C0", "C1", "C2"])
        & (pl.col("group") == "both groups")
    )
    phases = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
    cols = {"C0": GREY, "C1": theme.DATA_PURPLE, "C2": ORANGE}
    for metric, ttl, fn in (("cover90", "Share of the NRA inside q05-q95 (nominal 0.90)", "summary_coverage_by_phase.svg"),
                            ("width90", "Mean q05-q95 band width (fraction of capacity)", "summary_width_by_phase.svg")):  # fmt: skip
        alt.Chart(b).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("phase:N", sort=phases, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("arm:N"),
            y=alt.Y(f"{metric}:Q", title=ttl),
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(cols), range=list(cols.values())), legend=alt.Legend(title="Arm", orient="top", columns=3)),
        ).properties(width=W, height=180, title=alt.TitleParams([f"Injected events of 15 and 30 days, both groups, 24-120 h: {ttl}", "Phases are by valid time relative to the event; 'after0-7' is the 7 days after the end."], fontSize=11, anchor="start")).save(str(DOCS / fn))  # fmt: skip
    pd = pl.DataFrame(tables["gate_timeline"])
    if pd.height:
        alt.Chart(pd).mark_rect(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Days since the event began"), x2="day2:Q",
            y=alt.Y("event:N", title=None, sort=None),
            color=alt.Color("state:N", scale=alt.Scale(domain=["gate on", "held 7 days"], range=[ORANGE, SKY]), legend=alt.Legend(title="Origin day", orient="top")),
        ).properties(width=W, height=20 * pd["event"].n_unique() + 30, title=alt.TitleParams(["Gate state on each origin day around every plotted real event", "Blank: the gate is off. Orange: gate on. Sky blue: held for 7 days after the gate released. C2's band is the wider union band on orange and blue days."], fontSize=11, anchor="start")).save(str(DOCS / "summary_gate_timeline.svg"))  # fmt: skip
    wd = pl.DataFrame(tables["width_over_time"])
    if wd.height:
        alt.Chart(wd).mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Days since the load-shift began"),
            y=alt.Y("width:Q", title="Width (% of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["C1", "C2"], range=[theme.DATA_PURPLE, ORANGE]),
                legend=alt.Legend(title="Arm", orient="top"),
            ),
        ).properties(width=W, height=130).facet(row=alt.Row("event:N", title=None)).save(
            str(DOCS / "summary_width_over_time.svg")
        )
    for f in (
        "coverage90_verified_events.svg",
        "coverage90_15_and_30_days_both_groups.svg",
        "tail_pinball.svg",
    ):
        shutil.copy(DOCS.parent / "00584-nra-candidate-pack-x6" / f, DOCS / f"from584_{f}")


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    sc4.setup("c0")
    t0 = c.X4Table()
    look = fig4.C0Lookup(t0=t0)
    fig4.STATE["look"] = look
    sc4.setup("c1")
    t = c.X4Table()
    gi = c.gate_inputs(t=t)
    z = np.load(sc4.ROOT / "score_c1/gate_state.npz")
    st = {"on": z["on"], "hold": z["hold"]}
    tables: dict[str, list] = {
        k: []
        for k in (
            "verified",
            "labelled",
            "injected",
            "event_free",
            "gate_timeline",
            "width_over_time",
        )
    }
    verified(t=t, look=look, st=st, tables=tables)
    labelled(t=t, look=look, st=st, tables=tables)
    injected(t=t, look=look, gi=gi, tables=tables)
    event_free(t=t, look=look, st=st, tables=tables)
    # gate timeline and width over time for the real events
    for e in sc.attach_depth(ev=sc.merged_events(), t=t).iter_rows(named=True):
        onset, p = e["onset_idx"], e["primary"]
        rows = f3.day_ahead_rows(t=t, prim=p, lo=onset - 14 * 48, hi=e["end_idx"] + 14 * 48)
        if len(rows) < 200:
            continue
        name = f"{e['label']} {sc.k.date_of(onset)}"
        for d in sorted({int(d) for d in t.origin_day[rows]}):
            for flag, state in (
                (st["on"][p, d], "gate on"),
                (st["hold"][p, d] and not st["on"][p, d], "held 7 days"),
            ):
                if flag:
                    tables["gate_timeline"].append(
                        {
                            "event": name,
                            "day": d + 1 - onset / 48.0,
                            "day2": d + 2 - onset / 48.0,
                            "state": state,
                        }
                    )
        q2 = c2_of(t=t, rows=rows, v=t.arms["V"][rows], on=st["on"], hold=st["hold"])
        day = np.floor((t.idx[rows] - onset) / 48.0)
        for dd in np.unique(day):
            s = day == dd
            for arm, q in (("C1", t.arms["V"][rows]), ("C2", q2)):
                tables["width_over_time"].append(
                    {
                        "event": name,
                        "arm": arm,
                        "day": float(dd),
                        "width": float((q[s, IQ[0.95]] - q[s, IQ[0.05]]).mean() * 100),
                    }
                )
    summaries(tables=tables)
    for k, v in tables.items():
        if v:
            pl.DataFrame(v).write_csv(DATA / f"{k}.csv")
            print(k, len(v))


if __name__ == "__main__":
    main()
