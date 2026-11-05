# ruff: noqa: E501, D103, D205, D107, D102, ANN001, ANN003
"""Experiment 584 figures: day-ahead bands of every arm around the verified December pair and four
injected events, coverage by phase, the width-matched controls, and the false-alarm cost.

Run `e00584_score.py c0`, `c1`, and `c1 1.081` first. The event panels reuse experiment 583's drawing
code (`event_panels`, `verified_figures`, `coverage_figures`) with the arms renamed: C0 is the
v3+D5 run, C1 the P5 run with synthetic lags, C2 its gated band, C3 its always-on union band, C4 and
C4u the width-matched controls. Demand series only: every plotted series is a Primary substation.
"""

from pathlib import Path

import altair as alt
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00584_score as sc4
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

ROOT = sc4.ROOT
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W = 620
OVERRIDE = 1.081
"""Factor of the post hoc control C4w (fitted to the gated band's width on after-end rows of C1's injected bank)."""
RENAMES = [("C0", "c0", "V"), ("C1", "c1", "V"), ("C2", "c1", "A2"), ("C3", "c1", "Union"),
           ("C4", "c1", "C4"), ("C4u", "c1", f"C4u|{OVERRIDE}")]  # fmt: skip
COLOUR = {"C0": "#6b6b6b", "C1": theme.DATA_PURPLE, "C2": theme.BRAND_ORANGE, "C3": theme.DATA_BLUE,
          "C4": theme.DATA_SKY, "C4u": theme.DATA_GREEN, "C4w": "#7a2300"}  # fmt: skip
LABEL = {
    "C0": "C0: v3+D5 (experiment 572)",
    "C1": "C1: P5 recipe with synthetic-load-shift lags",
    "C2": "C2: C1 with the gated wider band",
    "C3": "C3: C1 with the always-on union band",
    "C4": "C4: C1 widened to C2's event-free width",
    "C4u": "C4u: C1 widened to C3's event-free width",
    "C4w": "C4w (post hoc): C1 widened to C2's width on after-end rows",
}
f3.COLOUR, f3.LABEL, f3.PANELS, f3.DOCS = COLOUR, LABEL, ("C0", "C1", "C2", "C3", "C4u"), DOCS
f3.PHASE_ORDER = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
ARMS_SHOWN = ["C0", "C1", "C2", "C3", "C4", "C4u", "C4w"]


def merged(*, name: str) -> pl.DataFrame:
    """Return experiment 583's table `name` for the arms, renamed C0 to C4u (and C4w from the post hoc run)."""
    parts = []
    for new, run, arm in [*RENAMES[:5], ("C4u", "c1", "C4u")]:
        d = pl.read_csv(ROOT / f"score_{run}/{name}").filter(pl.col("arm") == arm)
        parts.append(d.with_columns(arm=pl.lit(new)))
    w = pl.read_csv(ROOT / f"score_c1_w{OVERRIDE}/{name}").filter(pl.col("arm") == "C4u")
    parts.append(w.with_columns(arm=pl.lit("C4w")))
    return pl.concat(parts, how="diagonal")


class C0Lookup:
    """Find the v3+D5 run's forecasts for rows of the P5 run's table, by (Primary, origin, valid time)."""

    def __init__(self, *, t0: c.X4Table) -> None:
        self.t0 = t0
        self.keys0 = s3.key(t0.prim, t0.origin_idx, t0.idx)
        self.order0 = np.argsort(self.keys0)
        self.sorted0 = self.keys0[self.order0]

    def rows0(self, *, t1: c.X4Table, rows: np.ndarray) -> np.ndarray:
        k = s3.key(t1.prim[rows], t1.origin_idx[rows], t1.idx[rows])
        pos = np.clip(np.searchsorted(self.sorted0, k), 0, len(k0 := self.sorted0) - 1)
        ok = k0[pos] == k
        out = np.full(len(rows), -1, np.int64)
        out[ok] = self.order0[pos[ok]]
        return out


STATE: dict = {}
_orig_panels = f3.event_panels


def event_panels(*, arms, rows, t, **kw) -> None:
    """Rename the arms and add C0, then draw experiment 583's stacked panels."""
    look: C0Lookup = STATE["look"]
    r0 = look.rows0(t1=t, rows=rows)
    if STATE.get("pass") is None:
        v0 = np.where((r0 >= 0)[:, None], look.t0.arms["V"][np.maximum(r0, 0)], np.nan)
    else:
        v0 = STATE["c0_injected"](rows0=r0)
    new = {"C0": v0, "C1": arms["V"], "C2": arms["A2"], "C3": arms["Union"], "C4u": arms["C4u"]}
    _orig_panels(t=t, rows=rows, arms=new, **kw)


f3.event_panels = event_panels


def c0_injected(*, month: str, r: int, rows0: np.ndarray) -> np.ndarray:
    """Return the v3+D5 run's injected-bank quantiles of pass `r` for rows (indices into its table)."""
    t0, look = STATE["look"].t0, STATE["look"]
    keep = rows0 >= 0
    out = np.full((len(rows0), 11), np.nan, np.float32)
    saved = c.INJECT
    c.INJECT = sc4.ROOT / "view_c0/inject"
    try:
        out[keep] = s3.load_injected(
            t=t0, month=month, r=r, rows=rows0[keep], order=look.order0, sorted_keys=look.sorted0
        )
    finally:
        c.INJECT = saved
    return out


def bank_examples(*, t: c.X4Table, gi: dict) -> None:
    """Draw four injected events chosen by a fixed rule: in month 2025-10 the lowest-numbered Primary of
    each metering group for two deep 15-day (length, depth, sign) combinations. A 30-day event starts
    before the fold month, so its onset is not scored., taking the lowest pass.
    """
    ev_all = pl.read_parquet(s3.BANK_PARQUET).filter(pl.col("month") == "2025-10")
    kk_all = s3.key(t.prim, t.origin_idx, t.idx)
    order, sorted_keys = np.argsort(kk_all), np.sort(kk_all)
    in_m = t.month == "2025-10"
    days_m = np.unique(t.idx[in_m] // 48)
    origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
    gap_m = np.full_like(gi["gap"], np.nan)
    gap_m[:, days_m.min() : days_m.max() + 1] = gi["gap"][:, days_m.min() : days_m.max() + 1]
    for gname in s3.GROUPS_SCORED:
        for length, depth, sign in ((15, 0.5, 1), (15, 0.5, -1)):
            cand = ev_all.filter(
                (pl.col("length_days") == length) & (pl.col("depth") == depth) & (pl.col("sign") == sign)
                & pl.col("primary").is_in([p for p in range(16) if t.group[p] == gname])
            ).sort("pass", "primary")  # fmt: skip
            if cand.height == 0:
                continue
            e = cand.row(0, named=True)
            r, p = e["pass"], e["primary"]
            evr = ev_all.filter(pl.col("pass") == r)
            lo, hi = e["onset_idx"] - 3 * 48, e["end_idx"] + 7 * 48
            rows = f3.day_ahead_rows(t=t, prim=p, lo=lo, hi=hi, mask=in_m)
            vin = s3.load_injected(
                t=t, month="2025-10", r=r, rows=rows, order=order, sorted_keys=sorted_keys
            )
            ok = np.isfinite(vin).all(axis=1)
            rows, vin = rows[ok], vin[ok]
            gap_i = gap_m.copy()
            for ee in evr.iter_rows(named=True):
                gap_i[ee["primary"], ee["onset_idx"] // 48 : ee["end_idx"] // 48] += (
                    ee["sign"] * ee["depth"]
                )
            on, snap = c.gate_state(
                gap=gap_i, thr=gi["thr"], origin_days=origin_days_m, release_days=1
            )
            hold = c.hold_from(snap=snap)
            hold_r, on_r = (
                hold[t.prim[rows], t.origin_day[rows]],
                on[t.prim[rows], t.origin_day[rows]],
            )
            arms = c.combine(
                base=c.building_blocks(v=vin, m=t.arms["M"][rows]), hold=hold_r, on=on_r
            )
            STATE["pass"] = r
            STATE["c0_injected"] = lambda rows0, r=r: c0_injected(month="2025-10", r=r, rows0=rows0)
            event_panels(
                t=t, rows=rows, arms=arms, obs=t.obs[rows], nra=t.obs[rows], on=on, hold=hold,
                onset=e["onset_idx"], end=e["end_idx"], prim=p,
                title=[
                    f"{t.label[p]}, {gname}: injected lag-only event, {sign * depth * 100:+.0f}% of capacity for {length} days (pass {r})",
                    "Only the lag inputs carry the shift; the observed power (orange) is the normal running arrangement (black dashed, same line).",
                    "Dark band: q05-q95. Light band: q01-q99. Grey shade: the injected shift in the lags. Coloured strips: origin days in event mode (C2's gate).",
                ],
                out=f"injected_{gname.split()[0].lower()}_{length}d_{'up' if sign > 0 else 'down'}_{int(depth * 100)}pct.svg",
            )  # fmt: skip
    STATE["pass"] = None


def width_figure() -> None:
    """Coverage after the end against mean width: does the gated or union band gain more than width alone?"""
    b = merged(name="bank_summary.csv").filter(
        (pl.col("band") == "24-120 h")
        & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("phase") == "after0-7")
    )
    panels = []
    for grp in ("MVA-only eight", "signed-MW six"):
        d = b.filter(pl.col("group") == grp).with_columns(width=pl.col("width90") * 100)
        arms = [a for a in ARMS_SHOWN if a in set(d["arm"].to_list())]
        sc_c = alt.Scale(domain=arms, range=[COLOUR[a] for a in arms])
        base = alt.Chart(d).encode(
            color=alt.Color(
                "arm:N", scale=sc_c, legend=alt.Legend(title="Arm", orient="top", columns=4)
            )
        )
        pts = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "width:Q", title="Mean q05-q95 width (% of capacity)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y(
                "cover90:Q",
                title="Share of the NRA inside q05-q95",
                scale=alt.Scale(domain=[0.7, 1.0]),
            ),
        )  # ty: ignore[unresolved-attribute]
        err = base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="width:Q",
            y=alt.Y("cover90_lo:Q", scale=alt.Scale(domain=[0.7, 1.0])),
            y2="cover90_hi:Q",
        )  # ty: ignore[unresolved-attribute]
        nom = (
            alt.Chart(pl.DataFrame({"y": [0.90]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(nom, err, pts).properties(width=290, height=240, title=grp))
    alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            ["Is the coverage gain just width? Injected events of 15 and 30 days, 0-7 days after the end, 24-120 h",
             "A point on the same height as C2 or C3 at a smaller width would mean width alone is not the explanation. Line: 95% bootstrap over events."],
            fontSize=11, anchor="start")
    ).save(str(DOCS / "width_matched_control.svg"))  # fmt: skip


def false_alarm_figure() -> None:
    """Pinball loss on event-free rows against C1, for C2, C3, C4, and C4u."""
    fr = merged(name="event_free.csv").filter(
        (pl.col("band") == "24-120 h")
        & pl.col("group").is_in(list(s3.GROUPS_SCORED))
        & pl.col("arm").is_in(["C2", "C3", "C4", "C4u"])
    )
    rows = []
    for r in fr.iter_rows(named=True):
        for k, lab in (
            ("loss90", "q05-q95 band (q05 and q95 pinball)"),
            ("loss98", "q01-q99 band (q01 and q99 pinball)"),
        ):
            rows.append({"arm": r["arm"], "group": r["group"], "band": lab, "diff": r[f"{k}_minus_V"], "lo": r[f"{k}_minus_V_lo"], "hi": r[f"{k}_minus_V_hi"]})  # fmt: skip
    df = pl.DataFrame(rows)
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    cols = []
    for g in s3.GROUPS_SCORED:
        base = alt.Chart(df.filter(pl.col("group") == g)).encode(
            y=alt.Y("arm:N", sort=None, title=None),
            color=alt.Color("band:N", scale=alt.Scale(range=[theme.DATA_BLUE, theme.BRAND_ORANGE]), legend=alt.Legend(title="Pinball loss of", orient="top", columns=1)),
            yOffset="band:N",
        )  # fmt: skip
        pts = base.mark_point(filled=True, size=40, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "diff:Q", title="Pinball loss minus C1's (points of capacity; negative is better)"
            )
        )  # ty: ignore[unresolved-attribute]
        err = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        cols.append(alt.layer(zero, err, pts).properties(width=240, height=160, title=g))
    alt.hconcat(*cols).properties(
        title=alt.TitleParams(
            ["False-alarm cost: pinball loss minus C1's, rows with no load-shift within 7 days, 24-120 h",
             "Point: mean over fold months. Line: month-block bootstrap (5 blocks) at the Bonferroni level of 16 counted contrasts."],
            fontSize=11, anchor="start")
    ).save(str(DOCS / "false_alarm_cost.svg"))  # fmt: skip


def tails_figure() -> None:
    """Pinball loss at q01, q05, q95, and q99 on the injected bank, 0-7 days after the end."""
    b = merged(name="bank_summary.csv").filter(
        (pl.col("band") == "24-120 h")
        & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("phase") == "after0-7")
        & (pl.col("group") == "both groups")
    )
    long_ = pl.concat(
        [
            b.select(
                "arm", pl.lit(q).alias("quantile"), (pl.col(f"pin{q[1:]}") * 100).alias("pinball")
            )
            for q in ("q01", "q05", "q95", "q99")
        ]
    )
    arms = [a for a in ARMS_SHOWN if a in set(b["arm"].to_list())]
    alt.Chart(long_).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("arm:N", sort=arms, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pinball:Q", title="Mean pinball loss (% of capacity; lower is better)"),
        color=alt.Color("arm:N", scale=alt.Scale(domain=arms, range=[COLOUR[a] for a in arms]), legend=None),
        column=alt.Column("quantile:N", sort=["q01", "q05", "q95", "q99"], title=None),
    ).properties(width=130, height=200, title=alt.TitleParams(["Tail pinball loss, injected events of 15 and 30 days, 0-7 days after the end, both groups, 24-120 h"], fontSize=11, anchor="start")).save(str(DOCS / "tail_pinball.svg"))  # fmt: skip


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    sc4.setup("c0")
    t0 = c.X4Table()
    STATE["look"] = C0Lookup(t0=t0)
    pack = ROOT / "score_pack"
    pack.mkdir(exist_ok=True)
    for name in ("bank_summary.csv", "real_event_phase.csv"):
        merged(name=name).write_csv(pack / name)
    sc4.setup("c1")
    t = c.X4Table()
    gi = c.gate_inputs(t=t)
    z = np.load(ROOT / "score_c1/gate_state.npz")
    st = {"on": z["on"], "hold": z["hold"]}
    STATE["pass"] = None
    f3.verified_figures(t=t, st=st)
    bank_examples(t=t, gi=gi)
    c.WORK = pack
    f3.coverage_figures()
    width_figure()
    false_alarm_figure()
    tails_figure()
    _ = (bank, sc)


if __name__ == "__main__":
    main()
