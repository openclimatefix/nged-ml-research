# ruff: noqa: E501, D103, ANN001, DTZ001
"""Experiment 612 figures (CPU). Run the scoring, `e00612_summary.py` and `e00612_tails.py` first.

`python e00612_figures.py summary` draws the leave-one-out chart and the tail-constant chart.
`python e00612_figures.py real` draws the December 2025 example event for the two Primaries D5 and D16:
K1 over K2 (day-ahead forecasts, one arm per panel), and K3 alone. Demand series only (Primaries).
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00584_score as k584  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k597  # ty: ignore[unresolved-import]
import e00597_figures as f597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00612-nra-combined-networks-three-class-roster-completion"
ROOT = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
MONTHS = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SWITCH_TIMES = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
IQ = c.IQ


def save(chart, name: str) -> None:
    chart.save(str(DOCS / name))


def interval_chart(*, d: pl.DataFrame, title: str, width: int = 240, stack: bool = False) -> alt.Chart:
    """One dot-and-interval panel per measure (columns `row`, `measure`, `diff`, `lo`, `hi`), zero line dashed."""
    panels = []
    for j, m in enumerate(d["measure"].unique(maintain_order=True).to_list()):
        dm = d.filter(pl.col("measure") == m)
        order = dm["row"].to_list()
        base = alt.Chart(dm).encode(y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        bars = base.mark_rule(color=theme.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        lo_x, hi_x = min(0.0, float(dm["lo"].min())), max(0.0, float(dm["hi"].max()))
        pad = 0.08 * (hi_x - lo_x)
        pts = base.mark_point(filled=True, size=60, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("diff:Q", title=m, scale=alt.Scale(domain=[lo_x - pad, hi_x + pad], nice=False))
        )
        panels.append(alt.layer(zero, bars, pts).properties(width=width, height=max(120, 36 * len(order)), title=m))
    return (alt.vconcat(*panels) if stack else alt.hconcat(*panels)).resolve_scale(x="independent").properties(title=title)


def summary() -> None:
    out = ROOT / "tables_out"
    t1 = pl.read_csv(out / "t1_levels.csv", infer_schema_length=None).filter(
        (pl.col("band") == "24-120 h") & pl.col("free_cover90").is_not_null()
    )
    nets = ["K1 (D5)", "K2 (D5)", "K3 (D5)", "K1 (O2)", "K3 (O2)", "Class 2 (gate band width)", "Class 2 (own band width)"]
    t1 = t1.filter(pl.col("network").is_in(nets)).with_columns(
        group=pl.col("group").replace({"signed-MW six": "Six MW Primaries", "MVA-only eight": "Eight MVA Primaries"}),
        tails=pl.when(pl.col("tail_constants")).then(pl.lit("with tail constants")).otherwise(pl.lit("without")),
    )
    target = alt.Chart(pl.DataFrame({"x": [0.9]})).mark_rule(strokeDash=[4, 3], color="#6b6b6b").encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.78, 0.94]))
    )
    panels = []
    for k, group in enumerate(["Eight MVA Primaries", "Six MW Primaries"]):
        dots = (
            alt.Chart(t1.filter(pl.col("group") == group))  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=90)
            .encode(
                y=alt.Y("network:N", sort=nets, title=None, axis=alt.Axis(labelLimit=260) if k == 0 else None),
                x=alt.X("free_cover90:Q", title="Share of event-free rows inside q05-q95", scale=alt.Scale(domain=[0.78, 0.94])),
                color=alt.Color("tails:N", scale=alt.Scale(domain=["without", "with tail constants"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
                                legend=alt.Legend(title="Tail constants (dashed: 0.90 target)", orient="bottom") if k == 0 else None),
            )
        )
        panels.append(alt.layer(target, dots).properties(width=260, height=240, title=group))
    save(alt.hconcat(*panels), "event_free_coverage.svg")

    t2 = pl.read_csv(out / "t2_o2_vs_d5.csv", infer_schema_length=None).filter(
        pl.col("contrast").str.contains("cover90") & pl.col("contrast").str.contains("24-120 h")
    )
    t2 = t2.with_columns(
        row=pl.col("contrast").str.extract(r"^(K\d)", 1) + pl.when(pl.col("contrast").str.contains("injected")).then(pl.lit(", injected, ") + pl.col("contrast").str.extract(r"(days4-end|after0-7)", 1).replace({"days4-end": "day 4 to end", "after0-7": "0-7 days after"})).otherwise(pl.lit(", event-free ") + pl.when(pl.col("contrast").str.contains("eight")).then(pl.lit("eight MVA")).otherwise(pl.lit("six MW"))),
        measure=pl.lit("Coverage of q05-q95, O2 minus D5, 24-120 h"),
    )
    save(interval_chart(d=t2, title="O2 minus D5, tail constants on both"), "o2_vs_d5.svg")

    t4 = pl.read_csv(out / "t4_class2.csv", infer_schema_length=None).filter(
        (pl.col("measure").str.contains("24-120 h")) & ~pl.col("measure").str.contains("after0-7|follow")
    )
    grp = pl.when(pl.col("measure").str.contains("eight")).then(pl.lit("eight MVA")).when(pl.col("measure").str.contains("six")).then(pl.lit("six MW")).otherwise(pl.lit("injected, days 4 to end"))
    t4 = t4.with_columns(
        row=pl.lit("minus ") + pl.col("candidate_minus") + pl.lit(", ") + grp,
        measure=pl.when(pl.col("measure").str.contains("injected")).then(pl.lit("Injected coverage, class 2 minus network"))
        .when(pl.col("measure").str.contains("cover90")).then(pl.lit("Event-free coverage, class 2 minus network"))
        .otherwise(pl.lit("Event-free CRPS, class 2 minus network")),
    )
    save(interval_chart(d=t4, title="Class 2 candidate at the gate's band width, 24-120 h, five months", width=420, stack=True), "class2_vs_k1_k2.svg")


def aligned(*, root_arm: str, tag: str, t) -> np.ndarray:
    """Return the quantiles of the fits `arms/<root_arm>/work/fits/<tag>` aligned to the table rows (NaN where absent)."""
    e = g.a._load(root=ROOT / "arms" / root_arm / "work", tag=tag, months=MONTHS)
    sm = c.k578.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    e = e.with_columns(
        prim=pl.col("time_series_id").replace_strict(prim_of, return_dtype=pl.Int64),
        idx=((pl.col("valid_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
        oidx=((pl.col("power_fcst_init_time").dt.epoch("us") - c.k578.prep.GRID_START_US) // c.k578.prep.STEP_US).cast(pl.Int64),
    )  # fmt: skip
    kk = g.s.key(e["prim"].to_numpy(), e["oidx"].to_numpy(), e["idx"].to_numpy())
    order = np.argsort(kk)
    mine = g.s.key(t.prim, t.origin_idx, t.idx)
    pos = np.clip(np.searchsorted(kk[order], mine), 0, len(kk) - 1)
    ok = kk[order][pos] == mine
    q = np.sort(e.select(list(a.QCOLS)).to_numpy().astype(np.float32)[order[pos]], axis=1)
    q[~ok] = np.nan
    return q


def day_rows(*, t, p: int, lo: int, hi: int) -> np.ndarray:
    """Return the table rows of the day-ahead forecast (issued 09:00 the day before) of Primary `p`, valid in [lo, hi)."""
    _, rows = f597.day_ahead_keys(t=t, prim=p, lo=lo, hi=hi)
    return rows


def situations() -> None:
    """Draw K1 and K2, then K3 and the class 2 candidate, for five situations: the two December Primaries, a quiet week, a cold night, a sunny day."""
    c.MONTHS = MONTHS
    cp.RULES = cp.entry("Load-shift start and end given by the maintainer", "rule", cp.RULE_COLOUR)
    work = ROOT / "arms" / "k1_d5" / "work"
    k584.ARMS["k1_d5"] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
    twin_root = ROOT / "arms" / "k3_d5" / "work"
    orig = g.a._load
    g.a._load = lambda *, root, tag, months: orig(root=twin_root if tag == "D5" else root, tag=tag, months=months)
    k597.WORK = ROOT / "score" / "figures_tmp"
    inp = k597.Inputs(arm="k1_d5")
    t = inp.t
    g.a._load = orig
    q3_all = aligned(root_arm="k3_d5", tag="D5", t=t)
    q4_all = aligned(root_arm="c2_e3", tag="P5", t=t)
    when = lambda i: (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)  # noqa: E731
    eight = [p for p in range(16) if t.group[p] == "MVA-only eight"]
    six = [p for p in range(16) if t.group[p] == "signed-MW six"]

    def draw(*, name: str, p: int, lo: int, hi: int, title: str, event=None, rules=()) -> None:
        rows = day_rows(t=t, p=p, lo=lo, hi=hi)
        v, m, o = t.arms["V"][rows], t.arms["M"][rows], t.obs[rows]
        q1 = k597.assemble(v=v, m=m, wide=inp.mode[t.prim[rows], t.origin_day[rows]])
        q3, q4 = q3_all[rows], q4_all[rows]
        x = [when(i) for i in t.idx[rows]]
        label = t.label[p]
        ydom = cp.common_ydom([(q1, o), (v, o), (np.nan_to_num(q3), o), (np.nan_to_num(q4), o)], iq=IQ)
        common = {"x": x, "iq": IQ, "obs": o, "ydom": ydom, "xtype": "T", "xdom": (when(lo), when(hi)),
                  "span": None if event is None else (when(event[0]), when(event[1])), "rules": list(rules)}  # fmt: skip
        strip = None
        if event is not None:
            strip = [{"x0": when(rr["x0"] * 48 + event[0]), "x1": when(rr["x1"] * 48 + event[0])}
                     for rr in f597.strips_for(t=t, rows=rows, wide=inp.mode, prim=p, onset=event[0])]  # fmt: skip
        else:
            on = inp.mode[t.prim[rows], t.origin_day[rows]]
            strip = [{"x0": x[i], "x1": x[min(i + 1, len(x) - 1)]} for i in np.flatnonzero(on)] if on.any() else None
        key_a = [*cp.FORECAST_ENTRIES, cp.OBS, *([cp.SHADE] if event else []), *([cp.GATE] if strip else []), *([cp.RULES] if rules else [])]
        key_b = [*cp.FORECAST_ENTRIES, cp.OBS, *([cp.SHADE] if event else []), *([cp.RULES] if rules else [])]
        save(cp.figure(units=[cp.arm_panel(q=q1, title=f"{label}: K1 (lag network with the G4 gated band)", strip=strip, **common),
                              cp.arm_panel(q=v, title=f"{label}: K2 (same fits, no gate)", xtitle="Date (UTC)", strip=None, **common)],
                       title=f"{label}: {title}", key=key_a), f"{name}_{label}_k1_k2.svg")  # fmt: skip
        save(cp.figure(units=[cp.arm_panel(q=q3, title=f"{label}: K3 (lag-free)", strip=None, **common),
                              cp.arm_panel(q=q4, title=f"{label}: class 2 candidate (neighbour summaries, signal dropout)", xtitle="Date (UTC)", strip=None, **common)],
                       title=f"{label}: {title}", key=key_b), f"{name}_{label}_k3_c2.svg")  # fmt: skip

    ev = sc.attach_depth(ev=sc.merged_events(), t=t).filter(pl.col("name").str.contains("2025-12-29"))
    for r in ev.sort("name").iter_rows(named=True):
        draw(name="event", p=r["primary"], lo=r["onset_idx"] - 14 * 48, hi=r["end_idx"] + 14 * 48,
             title="day-ahead forecasts through the December 2025 load-shift", event=(r["onset_idx"], r["end_idx"]), rules=SWITCH_TIMES)  # fmt: skip

    def pick(*, p: int, month: str, days: int, score) -> int:
        """Return the first grid index of the best `days`-day window of Primary `p` in `month` by `score` (lower is better)."""
        ok = (t.prim == p) & (t.month == month) & t.free & (t.band == 1)
        days_present = np.unique(t.idx[ok] // 48)
        best, best_s = None, np.inf
        for d0 in days_present[:-days]:
            w = ok & (t.idx // 48 >= d0) & (t.idx // 48 < d0 + days)
            if w.sum() < days * 40:
                continue
            sc_ = score(w)
            if sc_ < best_s:
                best, best_s = int(d0), sc_
        assert best is not None
        return best * 48

    p = eight[0]
    lo = pick(p=p, month="2026-03", days=7, score=lambda w: float(np.nanmean(np.abs(t.arms["V"][w][:, IQ[0.5]] - t.obs[w]))))
    draw(name="quiet_week", p=p, lo=lo, hi=lo + 7 * 48, title="a quiet week (event-free, 7 days with the smallest mean median-forecast error in March 2026)")
    hour = lambda w: ((t.idx[w] % 48) / 2.0)  # noqa: E731
    lo = pick(p=p, month="2026-01", days=3, score=lambda w: -float(np.nanmean(t.obs[w][hour(w) < 7])))
    draw(name="cold_night", p=p, lo=lo, hi=lo + 3 * 48, title="three days around the highest overnight demand in January 2026 (a proxy for a cold winter night)")
    p6 = six[0]
    lo = pick(p=p6, month="2026-06", days=3, score=lambda w: float(np.nanmean(t.obs[w][(hour(w) >= 11) & (hour(w) < 14)]) - np.nanmean(t.obs[w][hour(w) < 5])))
    draw(name="sunny_day", p=p6, lo=lo, hi=lo + 3 * 48, title="three days around the deepest midday dip in June 2026 (a proxy for a very sunny day)")


if __name__ == "__main__":
    {"summary": summary, "situations": situations}[sys.argv[1]]()
