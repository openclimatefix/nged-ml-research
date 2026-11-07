# ruff: noqa: E501, D103, ANN001, DTZ001, B007, PLR0915
"""Experiment 613 figures (CPU). Run the scoring, `e00613_summary.py` and `e00613_tails.py` first.

`python e00613_figures.py intervals` draws the three contrast charts.
`python e00613_figures.py real` draws the December 2025 example event for the two Primaries D5 and D16:
K1 and the class 2 candidate (day-ahead forecasts, one arm per panel). Demand series only (Primaries).
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

NAME = "00613-nra-roster-contrasts-with-intervals-five-seeds"
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


def interval_chart(
    *, d: pl.DataFrame, title: str, width: int = 240, stack: bool = False
) -> alt.Chart:
    """One dot-and-interval panel per measure (columns `row`, `measure`, `diff`, `lo`, `hi`), zero line dashed."""
    panels = []
    for j, m in enumerate(d["measure"].unique(maintain_order=True).to_list()):
        dm = d.filter(pl.col("measure") == m)
        order = dm["row"].to_list()
        base = alt.Chart(dm).encode(
            y=alt.Y(
                "row:N", sort=order, title=None, axis=alt.Axis(labelLimit=420, labelFontSize=11)
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE)
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        bars = base.mark_rule(color=theme.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        lo_x, hi_x = (
            min(0.0, dm.select(pl.col("lo").min()).item()),
            max(0.0, dm.select(pl.col("hi").max()).item()),
        )
        pad = 0.08 * (hi_x - lo_x)
        pts = base.mark_point(filled=True, size=60, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "diff:Q",
                title=m,
                axis=alt.Axis(tickCount=4),
                scale=alt.Scale(domain=[lo_x - pad, hi_x + pad], nice=False),
            )
        )
        panels.append(
            alt.layer(zero, bars, pts).properties(
                width=width, height=max(320, 40 * len(order)), title=m
            )
        )
    return (
        (alt.vconcat(*panels) if stack else alt.hconcat(*panels))
        .resolve_scale(x="independent")
        .properties(title=title)
    )


def intervals() -> None:
    """Draw one interval chart per family (rows are contrasts; the pale band is the tolerance)."""
    out = ROOT / "tables_out"
    meas = {"cover90": "Coverage of the 90% band", "follow": "Share of the shift followed",
            "safe": "Safe-side share", "crps": "Event-free CRPS (share of capacity)",
            "absorption": "Share of the shift absorbed"}  # fmt: skip

    def kind(s: str) -> str:
        if "absorption" in s:
            return "absorption"
        if "follow" in s:
            return "follow"
        if "safe-side" in s:
            return "safe"
        if "CRPS" in s:
            return "crps"
        return "cover90"

    for fam, title, keep in (
        ("class2", "Class 2 candidate at K1's band width minus K1 (CRPS: minus K2), five months, seeds 0 to 4", "in-sample"),
        ("o2", "O2 offset minus D5 offset, tail constants on both, four months", None),
        ("e3", "3-day embargo minus E3 refit rule, five months, seeds 0 to 4", None),
    ):  # fmt: skip
        d = pl.read_csv(out / f"roster_{fam}.csv", infer_schema_length=None)
        if keep:
            d = d.filter(pl.col("family").str.contains(keep))
        d = d.with_columns(
            kind=pl.col("contrast").map_elements(kind, return_dtype=pl.String),
            row=pl.col("contrast")
            .str.replace(r"^(3-day minus E3, |.*O2 minus D5, )", "")
            .str.replace_all("days4-end", "day 4 to end")
            .str.replace_all("after0-7", "0-7 days after")
            .str.replace_all("follow_q50", "follow")
            .str.replace_all("day_4_to_end", "day 4 to end")
            .str.replace_all("after_7_days", "7 days after the end")
            .str.replace(r" \(NRA below q95/above q05\)", ""),
        )
        d = d.with_columns(row=pl.when(pl.col("contrast").str.starts_with("K")).then(pl.col("contrast").str.slice(0, 2) + pl.lit(" ") + pl.col("row")).otherwise(pl.col("row")))
        panels = []
        for k in [x for x in meas if x in d["kind"].unique().to_list()]:
            dm = d.filter(pl.col("kind") == k)
            tol = float(dm["tolerance"][0])
            order = dm["row"].to_list()
            lo_x = min(-tol, float(dm["lo"].min()))
            hi_x = max(tol, float(dm["hi"].max()))
            pad = 0.1 * (hi_x - lo_x)
            dom = alt.Scale(domain=[lo_x - pad, hi_x + pad], nice=False)
            band = alt.Chart(pl.DataFrame({"a": [-tol], "b": [tol], "label": ["Tolerance"]})).mark_rect(opacity=0.5).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("a:Q", scale=dom), x2="b:Q",
                color=alt.Color("label:N", scale=alt.Scale(domain=["Tolerance"], range=[theme.DATA_SKY]), legend=alt.Legend(title=None, orient="bottom")),
            )  # fmt: skip
            base = alt.Chart(dm).encode(y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=380, labelFontSize=11)))
            bars = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=2.5).encode(x=alt.X("lo:Q", scale=dom, title=meas[k] + " (difference)"), x2="hi:Q")  # ty: ignore[unresolved-attribute]
            pts = base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(x=alt.X("diff:Q", scale=dom))  # ty: ignore[unresolved-attribute]
            panels.append(alt.layer(band, bars, pts).properties(width=420, height=max(280, 30 * len(order)), title=meas[k]))
        chart = alt.vconcat(*panels).resolve_scale(x="independent", color="shared").properties(title=title)
        save(chart, f"intervals_{fam}.svg")


def aligned(*, root_arm: str, tag: str, t) -> np.ndarray:
    """Return fit quantiles of `arms/<root_arm>/work/fits/<tag>` aligned to the table rows."""
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
    """Return the table rows of the day-ahead forecast of Primary `p`, valid in [lo, hi)."""
    _, rows = f597.day_ahead_keys(t=t, prim=p, lo=lo, hi=hi)
    return rows


def situations() -> None:
    """Draw K1, K2, K3, and the class 2 candidate for five situations (see the page)."""
    c.MONTHS = MONTHS
    a.SEEDS = (0, 1, 2, 3, 4)
    cp.RULES = cp.entry("Load-shift start and end dates", "rule", cp.RULE_COLOUR)
    work = ROOT / "arms" / "k1_d5" / "work"
    k584.ARMS["k1_d5"] = {"fits": work / "fits", "fit_tag": "P5", "inject": work / "inject"}
    twin_root = ROOT / "arms" / "k3_d5" / "work"
    orig = g.a._load
    g.a._load = lambda *, root, tag, months: orig(
        root=twin_root if tag == "D5" else root, tag=tag, months=months
    )
    k597.WORK = ROOT / "score" / "figures_tmp"
    inp = k597.Inputs(arm="k1_d5")
    t = inp.t
    g.a._load = orig
    q4_all = aligned(root_arm="c2_e3", tag="P5", t=t)
    wf = dict(pl.read_csv(ROOT / "score" / "c2_width_factors.csv").iter_rows())
    med = q4_all[:, IQ[0.5] : IQ[0.5] + 1]
    scale = np.array([wf.get(str(m), 1.0) for m in t.month], np.float32)[:, None]
    q4_all = (med + scale * (q4_all - med)).astype(np.float32)  # the candidate at K1's band width
    when = lambda i: (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)  # noqa: E731
    eight = [p for p in range(16) if t.group[p] == "MVA-only eight"]
    six = [p for p in range(16) if t.group[p] == "signed-MW six"]

    def draw(*, name: str, p: int, lo: int, hi: int, title: str, event=None, rules=()) -> None:
        rows = day_rows(t=t, p=p, lo=lo, hi=hi)
        v, m, o = t.arms["V"][rows], t.arms["M"][rows], t.obs[rows]
        q1 = k597.assemble(v=v, m=m, wide=inp.mode[t.prim[rows], t.origin_day[rows]])
        q4 = q4_all[rows]
        x = [when(i) for i in t.idx[rows]]
        label = t.label[p]
        ydom = cp.common_ydom([(q1, o), (np.nan_to_num(q4), o)], iq=IQ)
        common = {"x": x, "iq": IQ, "obs": o, "ydom": ydom, "xtype": "T", "xdom": (when(lo), when(hi)),
                  "span": None if event is None else (when(event[0]), when(event[1])), "rules": list(rules)}  # fmt: skip
        strip = None
        if event is not None:
            strip = [{"x0": when(rr["x0"] * 48 + event[0]), "x1": when(rr["x1"] * 48 + event[0])}
                     for rr in f597.strips_for(t=t, rows=rows, wide=inp.mode, prim=p, onset=event[0])]  # fmt: skip
        else:
            on = inp.mode[t.prim[rows], t.origin_day[rows]]
            strip = (
                [{"x0": x[i], "x1": x[min(i + 1, len(x) - 1)]} for i in np.flatnonzero(on)]
                if on.any()
                else None
            )
        key_a = [
            *cp.FORECAST_ENTRIES,
            cp.OBS,
            *([cp.SHADE] if event else []),
            *([cp.GATE] if strip else []),
            *([cp.RULES] if rules else []),
        ]
        save(cp.figure(units=[cp.arm_panel(q=q1, title=f"{label}: K1 (lag network with the G4 gated band)", strip=strip, **common),
                              cp.arm_panel(q=q4, title=f"{label}: class 2 candidate at K1's band width (no gate)", xtitle="Date (UTC)", strip=None, **common)],
                       title=f"Primary {label}: {title}", key=key_a), f"{name}_{label}_k1_c2.svg")  # fmt: skip

    ev = sc.attach_depth(ev=sc.merged_events(), t=t).filter(
        pl.col("name").str.contains("2025-12-29")
    )
    for r in ev.sort("name").iter_rows(named=True):
        draw(name="event", p=r["primary"], lo=r["onset_idx"] - 14 * 48, hi=r["end_idx"] + 14 * 48,
             title="day-ahead forecasts through the December 2025 load-shift", event=(r["onset_idx"], r["end_idx"]), rules=SWITCH_TIMES)  # fmt: skip

    def pick(*, p: int, month: str, days: int, score) -> int:
        """Return the first grid index of the best `days`-day window of `p` in `month` by `score`."""
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
    lo = pick(
        p=p,
        month="2026-03",
        days=7,
        score=lambda w: float(np.nanmean(np.abs(t.arms["V"][w][:, IQ[0.5]] - t.obs[w]))),
    )
    draw(
        name="quiet_week",
        p=p,
        lo=lo,
        hi=lo + 7 * 48,
        title="a quiet week (event-free, 7 days with the smallest mean median-forecast error in March 2026)",
    )
    hour = lambda w: (t.idx[w] % 48) / 2.0  # noqa: E731
    lo = pick(
        p=p, month="2026-01", days=3, score=lambda w: -float(np.nanmean(t.obs[w][hour(w) < 7]))
    )
    draw(
        name="cold_night",
        p=p,
        lo=lo,
        hi=lo + 3 * 48,
        title="three days around the highest overnight demand in January 2026 (a proxy for a cold winter night)",
    )
    p6 = six[0]
    lo = pick(
        p=p6,
        month="2026-06",
        days=3,
        score=lambda w: float(
            np.nanmean(t.obs[w][(hour(w) >= 11) & (hour(w) < 14)])
            - np.nanmean(t.obs[w][hour(w) < 5])
        ),
    )
    draw(
        name="sunny_day",
        p=p6,
        lo=lo,
        hi=lo + 3 * 48,
        title="three days around the deepest midday dip in June 2026 (a proxy for a very sunny day)",
    )


if __name__ == "__main__":
    {"intervals": intervals, "situations": situations}[sys.argv[1]]()
