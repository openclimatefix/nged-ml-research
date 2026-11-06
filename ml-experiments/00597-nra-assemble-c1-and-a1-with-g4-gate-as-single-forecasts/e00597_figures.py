# ruff: noqa: E501, D103, ANN001, ANN201, PERF401, ANN003, D107, D102, RUF059, B007, DTZ001, PLR0915
"""Experiment 597 figures. Run `e00597_score.py c1` and `e00597_score.py a1` first.

`python e00597_figures.py summary` draws the figures read from the score tables.
`python e00597_figures.py panels` draws the day-ahead panels of the assembled forecasts on the
verified December pair and on twelve injected designs (it rebuilds each arm's table, so it is slow).
Demand series only: every plotted series is a Primary substation.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c  # ty: ignore[unresolved-import]
import e00583_figures as f3  # ty: ignore[unresolved-import]
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00588_clean_panels as cp  # ty: ignore[unresolved-import]
import e00592_common as g  # ty: ignore[unresolved-import]
import e00597_common as k  # ty: ignore[unresolved-import]
import e00597_score as sc597  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W = 620
IQ = c.IQ
SYSTEM_ORDER = ["C1", "C1 + G4", "A1", "A1 + G4"]
SYSTEM_COLOUR = {
    "C1": "#9a9a9a",
    "C1 + G4": theme.BRAND_ORANGE,
    "A1": theme.DATA_SKY,
    "A1 + G4": theme.DATA_BLUE,
}
SYSTEM_SCALE = alt.Scale(domain=SYSTEM_ORDER, range=[SYSTEM_COLOUR[s] for s in SYSTEM_ORDER])
PHASE_LABEL = {
    "before": "Before (7 days)",
    "days0-3": "Days 0-3",
    "days4-end": "Day 4 to end",
    "after0-7": "0-7 days after",
    "after7-14": "7-14 days after",
}
PHASE_ORDER = list(PHASE_LABEL.values())
DESIGNS = [  # (length in days, depth share of capacity, sign): as in experiment 588
    (length, depth, sign)
    for length in (7, 15, 30)
    for depth, sign in ((0.15, 1), (0.30, -1), (0.50, 1), (0.50, -1))
]
INJECT_MONTH = "2026-03"
# The maintainer's switch times for the December 2025 example event (UTC).
SWITCH_TIMES = [datetime(2025, 12, 30, 1, 0), datetime(2026, 1, 14, 11, 30)]
GREY, BLUE, ORANGE = "#6b6b6b", theme.DATA_BLUE, theme.BRAND_ORANGE


def save(chart, name: str) -> None:
    chart.save(str(DOCS / name))


def read_all(name: str, **kw) -> pl.DataFrame:
    out = []
    for arm in ("c1", "a1"):
        d = pl.read_csv(k.WORK / arm / name, infer_schema_length=None, **kw)
        if "system" in d.columns:
            d = d.with_columns(
                system=pl.when(pl.col("system") == "plain")
                .then(pl.lit(k.ARM_LABEL[arm]))
                .otherwise(pl.lit(f"{k.ARM_LABEL[arm]} + G4"))
            )
        out.append(d.with_columns(arm=pl.lit(k.ARM_LABEL[arm])))
    return pl.concat(out)


def dedupe_plain(df: pl.DataFrame) -> pl.DataFrame:
    return df


# ---------------------------------------------------------------- figures read from the tables


def phase_chart(
    *,
    metric: str,
    title: list[str],
    domain: list[float],
    target: float | None,
    name: str,
    cls: str = "15 and 30 days",
) -> None:
    b = (
        read_all("bank_summary.csv")
        .filter(
            (pl.col("group") == "both groups")
            & (pl.col("length_class") == cls)
            & (pl.col("band") == "24-120 h")
        )
        .with_columns(phase_label=pl.col("phase").replace(PHASE_LABEL))
    )
    ys = alt.Scale(domain=domain)
    base = alt.Chart(b).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "phase_label:N",
            sort=PHASE_ORDER,
            title=None,
            axis=alt.Axis(labelAngle=0, labelLimit=140),
        ),
        xOffset=alt.XOffset("system:N", sort=SYSTEM_ORDER),
        color=alt.Color(
            "system:N",
            scale=SYSTEM_SCALE,
            sort=SYSTEM_ORDER,
            legend=alt.Legend(title=None, orient="top", columns=4),
        ),
    )
    layers = [
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(f"{metric}_lo:Q", scale=ys, title=title[0]), y2=f"{metric}_hi:Q"
        ),
        base.mark_point(filled=True, size=55).encode(y=alt.Y(f"{metric}:Q", scale=ys)),  # ty: ignore[unresolved-attribute]
    ]
    if target is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"y": [target]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#444")
            .encode(y=alt.Y("y:Q", scale=ys)),
        )  # ty: ignore[unresolved-attribute]
    save(
        alt.layer(*layers).properties(
            width=W, height=230, title=alt.TitleParams(title[1:], fontSize=11, anchor="start")
        ),
        name,
    )


def follow_hist() -> None:
    t = read_all("bank_event_phase.csv").filter(
        (pl.col("phase") == "days4-end")
        & (pl.col("band") == "24-120 h")
        & (pl.col("length_days") >= 15)
        & (pl.col("system").is_in(["C1", "A1"]))
    )
    t = t.with_columns(
        move=pl.col("follow_q50") * pl.col("depth") * 100,
        depth_label=(pl.col("depth") * 100).round(0).cast(pl.Int64).cast(pl.String) + "% depth",
        length_label=pl.col("length_days").cast(pl.String) + " day events",
    )
    print(
        t.group_by("system", "depth_label", "length_label")
        .agg(
            pl.len().alias("n"),
            pl.col("move").mean().alias("mean"),
            pl.col("move").quantile(0.95).alias("p95"),
            pl.col("move").max().alias("max"),
            pl.col("follow_q50").mean().alias("follow"),
        )
        .sort("system", "depth_label", "length_label")
    )
    ch = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_bar(opacity=0.6, binSpacing=0)
        .encode(
            x=alt.X(
                "move:Q",
                bin=alt.Bin(step=1),
                title="Median forecast moved towards the shift (points of capacity)",
            ),
            y=alt.Y("count():Q", stack=None, title="Events"),
            color=alt.Color(
                "system:N",
                scale=alt.Scale(
                    domain=["C1", "A1"], range=[SYSTEM_COLOUR["C1 + G4"], SYSTEM_COLOUR["A1 + G4"]]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=170, height=110)
        .facet(
            row=alt.Row("length_label:N", title=None),
            column=alt.Column(
                "depth_label:N", title=None, sort=["15% depth", "30% depth", "50% depth"]
            ),
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Median forecast movement on injected 15 and 30 day events, day 4 to the end, 24-120 h ahead",
                    "Bars: events. Positive: the median moved towards the shift; 0: no following. Shift depth: 15, 30, or 50 points.",
                ],
                fontSize=11,
                anchor="start",
            )
        )
    )
    save(ch, "follow_histogram_by_depth_and_length.svg")


def event_free_cost() -> None:
    e = read_all("event_free.csv").filter(
        (pl.col("system").str.contains("G4"))
        & pl.col("pin05_minus_plain").is_not_null()
        & (pl.col("band") == "24-120 h")
        & (pl.col("group") == "both groups")
    )
    rows = []
    for r in e.iter_rows(named=True):
        for q in ("pin05", "pin95"):
            rows.append(
                {
                    "system": r["system"],
                    "quantile": q.replace("pin", "q"),
                    "diff": r[f"{q}_minus_plain"] * 100,
                    "lo": r[f"{q}_lo"] * 100,
                    "hi": r[f"{q}_hi"] * 100,
                }
            )
    d = pl.DataFrame(rows)
    ys = alt.Scale(domain=[-0.15, 0.2])
    base = alt.Chart(d).encode(
        x=alt.X("quantile:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset="system:N",
        color=alt.Color(
            "system:N", scale=SYSTEM_SCALE, legend=alt.Legend(title=None, orient="top")
        ),
    )  # ty: ignore[unresolved-attribute]
    ch = alt.layer(
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#444")
        .encode(y=alt.Y("y:Q", scale=ys)),  # ty: ignore[unresolved-attribute]
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "lo:Q", scale=ys, title="Assembled minus plain network (points of capacity ×0.01)"
            ),
            y2="hi:Q",
        ),
        base.mark_point(filled=True, size=55).encode(y=alt.Y("diff:Q", scale=ys)),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W,
        height=180,
        title=alt.TitleParams(
            [
                "Extra pinball loss from the G4 band on event-free rows, 24-120 h ahead, both groups",
                "Dot: assembled minus plain network. Line: month-block interval, Bonferroni level (6 tests). 0: no cost.",
            ],
            fontSize=11,
            anchor="start",
        ),
    )
    save(ch, "false_alarm_cost.svg")


# ---------------------------------------------------------------- day-ahead panels


def to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in a]


class ArmView:
    """One arm's table, with a lookup from (Primary, origin index, valid index) to its row."""

    def __init__(self, arm: str) -> None:
        self.inp = k.Inputs(arm=arm)
        t = self.inp.t
        self.kk = s3.key(t.prim, t.origin_idx, t.idx)
        self.order = np.argsort(self.kk)
        self.sorted_keys = self.kk[self.order]

    def rows_for(self, keys: np.ndarray) -> np.ndarray:
        pos = np.clip(np.searchsorted(self.sorted_keys, keys), 0, len(self.sorted_keys) - 1)
        ok = self.sorted_keys[pos] == keys
        return np.where(ok, self.order[pos], -1)


def day_ahead_keys(*, t, prim: int, lo: int, hi: int, mask: np.ndarray | None = None) -> np.ndarray:
    rows = f3.day_ahead_rows(t=t, prim=prim, lo=lo, hi=hi, mask=mask)
    return s3.key(t.prim[rows], t.origin_idx[rows], t.idx[rows]), rows  # ty: ignore[invalid-return-type]


def panel(
    *, x, q_a, q_b, obs, nra, strips, span, xdom, title, height=140, xtitle=None, show_nra=True
):
    d = {"x": x, "alo": q_a[:, IQ[0.05]] * 100, "ahi": q_a[:, IQ[0.95]] * 100, "am": q_a[:, IQ[0.5]] * 100,
         "blo": q_b[:, IQ[0.05]] * 100, "bhi": q_b[:, IQ[0.95]] * 100, "bm": q_b[:, IQ[0.5]] * 100, "obs": obs * 100, "nra": nra * 100}  # fmt: skip
    df = pl.DataFrame({kk: to_none(np.asarray(v, float)) for kk, v in d.items()})
    lo = np.nanmin([np.nanmin(d["alo"]), np.nanmin(d["blo"]), np.nanmin(d["obs"])]) - 2
    hi = np.nanmax([np.nanmax(d["ahi"]), np.nanmax(d["bhi"]), np.nanmax(d["obs"])]) + 2
    sy = alt.Scale(domain=[float(lo), float(hi)])
    xax = alt.X(
        "x:Q",
        title=xtitle,
        scale=alt.Scale(domain=list(xdom), zero=False, nice=False),
        axis=alt.Axis(tickMinStep=1),
    )
    base = alt.Chart(df)
    layers = []
    if len(strips):
        bar = pl.DataFrame(list(strips)).with_columns(
            y0=pl.lit(float(lo)), y1=pl.lit(float(lo) + 0.05 * float(hi - lo))
        )
        layers.append(
            alt.Chart(bar)  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.9, color="#7fb7d9", clip=True)
            .encode(x="x0:Q", x2="x1:Q", y=alt.Y("y0:Q", scale=sy), y2="y1:Q")
        )  # ty: ignore[unresolved-attribute]
    if span is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"a": [span[0]], "b": [span[1]]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color="#999999", opacity=0.12, clip=True)
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]
    layers += [
        base.mark_area(color=SYSTEM_COLOUR["C1 + G4"], opacity=0.28, aria=False, clip=True).encode(x=xax, y=alt.Y("alo:Q", scale=sy, title="% of capacity"), y2="ahi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_area(color=SYSTEM_COLOUR["A1 + G4"], opacity=0.28, aria=False, clip=True).encode(x=xax, y=alt.Y("blo:Q", scale=sy), y2="bhi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="#b33000", strokeWidth=1, aria=False, clip=True).encode(x=xax, y=alt.Y("am:Q", scale=sy)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="#1d4f91", strokeWidth=1, aria=False, clip=True).encode(x=xax, y=alt.Y("bm:Q", scale=sy)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="black", strokeWidth=0.8, aria=False, clip=True).encode(x=xax, y=alt.Y("obs:Q", scale=sy)),  # ty: ignore[unresolved-attribute]
    ]  # fmt: skip
    if show_nra:
        layers.append(
            base.mark_line(  # ty: ignore[unresolved-attribute]
                color="black", strokeDash=[3, 2], strokeWidth=0.8, aria=False, clip=True
            ).encode(x=xax, y=alt.Y("nra:Q", scale=sy))
        )  # ty: ignore[unresolved-attribute]
    return alt.layer(*layers).properties(
        width=W, height=height, title=alt.TitleParams(title, fontSize=11, anchor="start")
    )


def strips_for(*, t, rows, wide, prim, onset):
    out = []
    for d in sorted({int(d) for d in t.origin_day[rows]}):
        if wide[prim, d]:
            out.append({"x0": (d + 1) - onset / 48.0, "x1": (d + 2) - onset / 48.0})
    return out


def real_pair(views: dict[str, ArmView]) -> None:
    """One figure per Primary on the verified December pair: C1 + G4 above A1 + G4."""
    v1, v2 = views["c1"], views["a1"]
    t1 = v1.inp.t
    ev = sc.attach_depth(ev=sc.merged_events(), t=t1).filter(
        pl.col("name").str.contains("2025-12-29")
    )
    for e in ev.sort("name").iter_rows(named=True):
        p, onset, end = e["primary"], e["onset_idx"], e["end_idx"]
        keys, rows = day_ahead_keys(t=t1, prim=p, lo=onset - 14 * 48, hi=end + 15 * 48)
        r2 = v2.rows_for(keys)
        ok = r2 >= 0
        rows, r2 = rows[ok], r2[ok]
        q_a = k.assemble(
            v=t1.arms["V"][rows],
            m=t1.arms["M"][rows],
            wide=v1.inp.mode[t1.prim[rows], t1.origin_day[rows]],
        )
        t2 = v2.inp.t
        q_b = k.assemble(
            v=t2.arms["V"][r2], m=t2.arms["M"][r2], wide=v2.inp.mode[t2.prim[r2], t2.origin_day[r2]]
        )
        when = lambda i: (sc.k.k.GRID_START + timedelta(minutes=30 * float(i))).replace(tzinfo=None)  # noqa: E731
        x = [when(i) for i in t1.idx[rows]]
        ydom = cp.common_ydom([(q_a, t1.obs[rows]), (q_b, t1.obs[rows])], iq=IQ)
        strip = [
            {"x0": when(r["x0"] * 48 + onset), "x1": when(r["x1"] * 48 + onset)}
            for r in strips_for(t=t1, rows=rows, wide=v1.inp.mode, prim=p, onset=onset)
        ]
        common = {
            "x": x, "iq": IQ, "obs": t1.obs[rows], "ydom": ydom, "xtype": "T",
            "xdom": (when(onset - 14 * 48), when(end + 14 * 48)), "strip": strip,
            "span": (when(onset), when(end)), "rules": SWITCH_TIMES,
        }  # fmt: skip
        units = [
            cp.arm_panel(q=q_a, title=f"{t1.label[p]}: C1 + G4", **common),
            cp.arm_panel(q=q_b, title=f"{t1.label[p]}: A1 + G4", xtitle="Date (UTC)", **common),
        ]  # fmt: skip
        key = [
            *cp.FORECAST_ENTRIES,
            cp.OBS, cp.SHADE, cp.GATE, cp.RULES,
        ]  # fmt: skip
        fig = cp.figure(
            units=units,
            title=f"{t1.label[p]}: day-ahead forecasts through the December 2025 load-shift (issued 09:00 the day before)",
            key=key,
        )
        save(fig, f"real_pair_{t1.label[p]}_day_ahead.svg")


def injected(views: dict[str, ArmView]) -> None:
    v1, v2 = views["c1"], views["a1"]
    ev_all = pl.read_parquet(s3.BANK_PARQUET).filter(pl.col("month") == INJECT_MONTH)
    inputs = {"c1": v1, "a1": v2}
    snippets = []
    for length, depth, sign in DESIGNS:
        for grp in k.GROUPS_SCORED:
            t1 = v1.inp.t
            cand = ev_all.filter((pl.col("length_days") == length) & (pl.col("depth") == depth) & (pl.col("sign") == sign) & pl.col("primary").is_in([p for p in range(16) if t1.group[p] == grp])).sort("pass", "primary")  # fmt: skip
            e = cand.row(0, named=True)
            r, p = e["pass"], e["primary"]
            onset, end = e["onset_idx"], e["end_idx"]
            qs = {}
            for arm, v in inputs.items():
                t, inp = v.inp.t, v.inp
                in_m = t.month == INJECT_MONTH
                days_m = np.unique(t.idx[in_m] // 48)
                origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
                sl = slice(days_m.min(), days_m.max() + 1)
                keys, rows = day_ahead_keys(
                    t=t, prim=p, lo=onset - 14 * 48, hi=end + 14 * 48, mask=in_m
                )
                kk_all = s3.key(t.prim, t.origin_idx, t.idx)
                order = np.argsort(kk_all)
                vin = s3.load_injected(
                    t=t, month=INJECT_MONTH, r=r, rows=rows, order=order, sorted_keys=kk_all[order]
                )
                old = np.full_like(inp.gi["gap"], np.nan)
                old[:, sl] = inp.gi["gap"][:, sl]
                shifts = []
                for ee in ev_all.filter(pl.col("pass") == r).iter_rows(named=True):
                    if t.group[ee["primary"]] not in k.GROUPS_SCORED:
                        continue
                    pp, sh = ee["primary"], ee["sign"] * ee["depth"]
                    old[pp, ee["onset_idx"] // 48 : ee["end_idx"] // 48] += sh
                    shifts.append((pp, ee["onset_idx"], ee["end_idx"], float(sh)))
                    nbs = inp.nb.get(pp, [])
                    if nbs:
                        n = nbs[(r + pp) % len(nbs)]
                        shifts.append(
                            (
                                n,
                                ee["onset_idx"],
                                ee["end_idx"],
                                float(-sh * t.capacity[pp] / t.capacity[n]),
                            )
                        )
                fresh = inp.fresh.gap(shifts=shifts)
                fresh[:, : origin_days_m[0] + 1] = np.nan
                fresh[:, origin_days_m[-1] + 1 :] = np.nan
                states = g.all_gates(
                    old_gap=old,
                    old_thr=inp.gi["thr"],
                    fresh_gap=fresh,
                    fresh_thr=inp.fresh_thr,
                    origin_days=origin_days_m,
                    neighbours=inp.nb,
                )
                wide = g.arm_masks(states=states)["G4"][0]
                ok = np.isfinite(vin).all(axis=1)
                rows, vin, keys = rows[ok], vin[ok], keys[ok]
                qs[arm] = (
                    keys,
                    rows,
                    k.assemble(
                        v=vin, m=t.arms["M"][rows], wide=wide[t.prim[rows], t.origin_day[rows]]
                    ),
                    wide,
                )
            k1, rows1, qa, wide1 = qs["c1"]
            k2, rows2, qb, _ = qs["a1"]
            common, i1, i2 = np.intersect1d(k1, k2, return_indices=True)
            rows1, qa, qb = rows1[i1], qa[i1], qb[i2]
            t1 = v1.inp.t
            x = (t1.idx[rows1] - onset) / 48.0
            nra = t1.obs[rows1]
            started = x.min() <= 0
            ydom = cp.common_ydom([(qa, nra), (qb, nra)], iq=IQ)
            strip = strips_for(t=t1, rows=rows1, wide=wide1, prim=p, onset=onset)
            common = {"x": x, "iq": IQ, "obs": nra, "ydom": ydom, "strip": strip,
                      "span": (0.0 if started else float(x.min()), (end - onset) / 48.0),
                      "xdom": (float(x.min()), (end - onset) / 48.0 + 14)}  # fmt: skip
            units = [
                cp.arm_panel(q=qa, title="C1 + G4", **common),
                cp.arm_panel(q=qb, title="A1 + G4", xtitle="Days since the injected shift began", **common),
            ]  # fmt: skip
            key = [*cp.FORECAST_ENTRIES, cp.OBS, cp.SHADE, cp.GATE]  # fmt: skip
            sg = "up" if sign > 0 else "down"
            tag = "mva" if k.GROUP_NAME[grp] == k.GROUP_NAME[k.GROUPS_SCORED[0]] else "mw"
            out = f"injected_{length}d_{sg}_{int(depth * 100)}pct_{tag}.svg"
            title = f"{t1.label[p]} ({k.GROUP_NAME[grp]}): injected {sign * depth * 100:+.0f}% of capacity for {length} days, March 2026, day-ahead"
            save(cp.figure(units=units, title=title, key=key), out)
            snippets.append(f"![{t1.label[p]}, injected {length} days, {sign * depth * 100:+.0f}%]({out})\n\n{t1.label[p]} ({k.GROUP_NAME[grp]}): {sign * depth * 100:+.0f}% of capacity for {length} days.\n")  # fmt: skip
    (k.WORK / "injected_snippets.md").write_text("\n".join(snippets))


def panels() -> None:
    views = {arm: ArmView(arm) for arm in ("c1", "a1")}
    real_pair(views)
    injected(views)


def summary() -> None:
    for cls, tag, lead, n in (
        ("15 and 30 days", "", "15 and 30 day events", "756 (672 after the end)"),
        ("all lengths", "_all_lengths", "3, 7, 15, and 30 day events", "events"),
    ):
        phase_chart(
            metric="cover90", cls=cls, name=f"coverage_by_phase{tag}.svg", domain=[0.8, 0.97], target=0.9,
            title=["Share of rows with the NRA inside q05-q95", f"Coverage by phase of an injected load-shift: {lead}, both metering groups, 24-120 h ahead", "Dot: mean over Primary-events. Line: 95% interval from resampling events. Dashed: the 0.90 target. Higher is better."],
        )  # fmt: skip
        phase_chart(
            metric="nra_side95", cls=cls, name=f"safe_side_by_phase{tag}.svg", domain=[0.8, 1.0], target=0.95,
            title=["Share of rows with the NRA on the safe side of q05 or q95", f"Safe-side share by phase of an injected load-shift: {lead}, both groups, 24-120 h ahead", "Safe side: below q95 if the shift lowered the load, above q05 if it raised it. Dot: mean. Line: 95% interval from resampling events. Dashed: 0.95 target."],
        )  # fmt: skip
    follow_hist()
    event_free_cost()


def real_only() -> None:
    real_pair({arm: ArmView(arm) for arm in ("c1", "a1")})


if __name__ == "__main__":
    {"summary": summary, "panels": panels, "real": real_only}[sys.argv[1]]()
    _ = (bank, sc597)
