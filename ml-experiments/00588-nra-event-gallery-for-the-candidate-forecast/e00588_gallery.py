# ruff: noqa: E501, D103, ANN001, ANN201, PERF401, DTZ001
"""Experiment 588: event gallery for experiment 584's candidate forecast. No fit.

Draws day-ahead forecasts of C0 (v3+D5, experiment 572) and C2 (P5 recipe with synthetic
load-shifts plus the gated band, experiment 584) on verified events, labelled events, injected
events, and event-free windows, from the forecasts experiment 584 stored. Demand series only: every
plotted series is a Primary substation. Run experiment 584's `e00584_score.py c0` and `c1` first.

Usage: DATA_PATH_INTERNAL=... PYTHONPATH=(all ml-experiments folders) python e00588_gallery.py
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00584_figures as fig4  # ty: ignore[unresolved-import]
import e00584_score as sc4  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
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
# The maintainer's switch times for the December 2025 example event (UTC).
SWITCH_TIMES = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
C0_MEDIAN, C2_MEDIAN = "#444444", "#b33000"
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


def when(idx) -> datetime:
    """The UTC time (naive) of a half-hour grid index."""
    return (sc.k.k.GRID_START + timedelta(minutes=30 * float(idx))).replace(tzinfo=None)


def two_arm_figure(*, x, q0, q2, obs, nra, strips, span, xdom, title, out, xtitle, xtype="Q",
                   rules=(), label_nra=None, label=""):  # fmt: skip
    """One figure: C0 in the top panel, C2 (with its gate strip) in the bottom panel, then the key."""
    lo_obs = obs if label_nra is None else np.fmin(obs, label_nra)
    ydom = cp.common_ydom([(q0, lo_obs), (q2, lo_obs)], iq=IQ)
    common = {"x": x, "iq": IQ, "obs": obs, "nra": nra, "ydom": ydom, "xdom": xdom, "xtype": xtype,
              "span": span, "rules": rules, "label_nra": label_nra}  # fmt: skip
    units = [
        cp.arm_panel(q=q0, colour=GREY, median_colour=C0_MEDIAN, title=f"{label}C0", **common),
        cp.arm_panel(q=q2, colour=ORANGE, median_colour=C2_MEDIAN, title=f"{label}C2", strip=strips, xtitle=xtitle, **common),
    ]  # fmt: skip
    key = [
        *cp.band_entries("C0", GREY, C0_MEDIAN),
        *cp.band_entries("C2", ORANGE, C2_MEDIAN),
        cp.OBS,
    ]
    if label_nra is None:
        key.append(cp.NRA)
    else:
        key.append(cp.LABEL_NRA)
    if span is not None:
        key.append(cp.SHADE)
    if strips:
        key.append(cp.GATE)
    if len(rules):
        key.append(cp.RULES)
    cp.figure(units=units, title=title, key=key).save(str(DOCS / out))


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
        length = (end - onset) / 48.0
        date = sc.k.date_of(onset)
        strips = [{"x0": when(r["x0"] * 48 + onset), "x1": when(r["x1"] * 48 + onset)}
                  for r in strips_for(t=t, rows=rows, on=st["on"], hold=st["hold"], prim=p, onset=onset)]  # fmt: skip
        x = [when(i) for i in t.idx[rows]]
        kw = {"x": x, "q0": q0, "q2": q2, "obs": t.obs[rows], "nra": t.nra[rows], "strips": strips,
              "span": (when(onset), when(end)), "rules": SWITCH_TIMES, "xtype": "T",
              "xtitle": "Date (UTC)", "label": f"{e['label']}: "}  # fmt: skip
        ttl = f"{e['label']}: day-ahead forecasts through the verified load-shift of {e['depth_mw']:+.1f} MW for {length:.0f} days, starting {date}"
        two_arm_figure(
            **kw,
            xdom=(when(onset - 14 * 48), when(end + 14 * 48)),
            title=ttl,
            out=f"verified_{e['label']}_{date}.svg",
        )
        two_arm_figure(**{**kw, "span": None}, xdom=(when(end), when(end + 7 * 48)), title=f"{ttl}. Zoom on the 7 days after the end", out=f"verified_{e['label']}_{date}_zoom.svg")  # fmt: skip
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
        ttl = [
            f"{e['label']}, labelled event {e['event_id']}: {e['label_depth_mw']:+.2f} MW for {length:.1f} days from {date}",
            f"580 status: {status}."
            + (f" Same dates as the label at {', '.join(partner)}." if partner else ""),
        ]
        x = (t.idx[rows] - onset) / 48.0
        strips = strips_for(t=t, rows=rows, on=st["on"], hold=st["hold"], prim=p, onset=onset)
        out = f"labelled_{e['event_id']:03d}_{e['label']}_{length:.0f}d.svg"
        two_arm_figure(x=x, q0=q0, q2=q2, obs=t.obs[rows], nra=np.full(len(rows), np.nan), label_nra=lab_nra, strips=strips, span=(0, length),
                       xdom=(-7, length + 7), xtitle="Days since the labelled start", title=ttl, out=out)  # fmt: skip
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
    snippets = []
    for length, depth, sign in DESIGNS:
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
            sg = "up" if sign > 0 else "down"
            tag = "mva" if g == groups[0] else "mw"
            out = f"injected_{length}d_{sg}_{int(depth * 100)}pct_{tag}.svg"
            two_arm_figure(x=x, q0=q0, q2=q2, obs=nra, nra=nra, strips=strips_for(t=t, rows=rows, on=on, hold=hold, prim=p, onset=onset),
                           span=(0.0 if started else float(x.min()), (end - onset) / 48.0), xdom=(float(x.min()), (end - onset) / 48.0 + 7),
                           xtitle="Days since the injected shift began", out=out,
                           title=f"{t.label[p]} ({g}): injected {sign * depth * 100:+.0f}% of capacity for {length} days, March 2026, day-ahead")  # fmt: skip
            snippets.append(f"![{t.label[p]}, injected {length} days, {sign * depth * 100:+.0f}%]({out})\n\n{t.label[p]} ({g}): {sign * depth * 100:+.0f}% of capacity for {length} days.\n")  # fmt: skip
    (DATA / "injected_snippets.md").write_text("\n".join(snippets))


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
    pending: list = []
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
            pending.append(
                {
                    "x": x,
                    "q": q2,
                    "obs": t.obs[rows],
                    "title": f"{t.label[p]} ({g}), 14 days from {date}",
                }
            )
            if len(pending) == 2:
                emit_event_free(
                    pending=pending, out=f"event_free_windows_{len(tables['event_free']) // 2}.svg"
                )
                pending = []
            break


def emit_event_free(*, pending, out):
    """Two event-free windows in one figure, each drawn with C2 only (C1 is identical on every row)."""
    ydom = cp.common_ydom([(w["q"], w["obs"]) for w in pending], iq=IQ)
    units = [
        cp.arm_panel(x=w["x"], q=w["q"], iq=IQ, obs=w["obs"], nra=w["obs"], colour=ORANGE, median_colour=C2_MEDIAN, xdom=(0, 14), ydom=ydom,
                     title=w["title"], strip=[], xtitle="Days since the window began" if i == len(pending) - 1 else None)
        for i, w in enumerate(pending)
    ]  # fmt: skip
    key = [*cp.band_entries("C2", ORANGE, C2_MEDIAN), cp.OBS, entry_nra_same()]
    cp.figure(
        units=units,
        key=key,
        title="Event-free windows: C2 on 14 days with no labelled or verified load-shift within 7 days",
    ).save(str(DOCS / out))


def entry_nra_same():
    return cp.entry("NRA (equal to observed power here)", "dashed", "black")


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
        ).properties(width=W, height=280).facet(row=alt.Row("event:N", title=None)).save(
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
