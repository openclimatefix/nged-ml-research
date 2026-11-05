# ruff: noqa: E501, D103, D205
"""Experiment 583 figures: day-ahead forecast bands around verified and injected events, coverage by
phase, the false-alarm cost, and a 5-day horizon example.

Run `e00583_score.py` first (it writes the CSVs and `gate_state.npz` this script reads). The
forecast figures rebuild the arms from the stored forecasts. Demand series only: every plotted
series is a Primary substation, so Primaries carry their D-numbers and dates.
"""

from pathlib import Path

import altair as alt
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import e00583_common as c
import e00583_score as s3
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
IQ = c.IQ
W = 620
PANELS = ("V", "Union", "A1", "A2", "A3")
COLOUR = {
    "V": theme.DATA_PURPLE,
    "M": "#6b6b6b",
    "Union": "#6b6b6b",
    "A1": theme.DATA_BLUE,
    "A2": theme.BRAND_ORANGE,
    "A3": theme.DATA_GREEN,
    "A4": "#b0b0b0",
}
LABEL = {
    "V": "X3-syn with lags",
    "M": "Lags masked",
    "Union": "Union band always on (experiment 582's control)",
    "A1": "A1: union band for 7 days after the snap-back",
    "A2": "A2: union band while the gate is on and 7 days after",
    "A3": "A3: as A1, with the lags-masked median",
    "A4": "A4: placebo, A1 at random days",
}
PHASE_ORDER = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]


def _band_layers(*, df: pl.DataFrame, col: str, xax: alt.X, sc_y: alt.Scale) -> list:
    base = alt.Chart(df)
    return [
        base.mark_area(color=col, opacity=0.18, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("q01:Q", scale=sc_y, title="% of capacity"), y2="q99:Q"
        ),
        base.mark_area(color=col, opacity=0.35, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("q05:Q", scale=sc_y), y2="q95:Q"
        ),
        base.mark_line(color=col, strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("q50:Q", scale=sc_y)
        ),  # ty: ignore[unresolved-attribute]
        base.mark_line(
            color=theme.BRAND_ORANGE if col != theme.BRAND_ORANGE else "#7a2300",
            strokeWidth=1,
            aria=False,
        ).encode(x=xax, y=alt.Y("obs:Q", scale=sc_y)),  # ty: ignore[unresolved-attribute]
        base.mark_line(color="black", strokeDash=[3, 2], strokeWidth=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("nra:Q", scale=sc_y)
        ),  # ty: ignore[unresolved-attribute]
    ]


def event_panels(
    *,
    t: c.X4Table,
    rows: np.ndarray,
    arms: dict[str, np.ndarray],
    obs: np.ndarray,
    nra: np.ndarray,
    on: np.ndarray,
    hold: np.ndarray,
    onset: int,
    end: int,
    prim: int,
    title: list[str],
    out: str,
) -> None:
    """Draw stacked panels of day-ahead forecasts (issued 09:00 the day before the valid day) from
    3 days before onset to 7 days after the end, with the gate and hold state of each origin day
    shaded behind them. `rows` are the day-ahead table rows sorted by valid time; `obs` and `nra`
    and `arms` are aligned with `rows`.
    """
    x = (t.idx[rows] - onset) / 48.0
    length = (end - onset) / 48.0
    strips = []
    for d in sorted({int(d) for d in t.origin_day[rows]}):
        state = "gate on" if on[prim, d] else ("held 7 days" if hold[prim, d] else None)
        if state:
            strips.append(
                {"x0": (d + 1) - onset / 48.0, "x1": (d + 2) - onset / 48.0, "state": state}
            )
    ymin = float(np.nanmin(np.minimum(obs, nra))) * 100 - 3
    ymax = float(np.nanmax(np.maximum(obs, nra))) * 100 + 3
    sc_y = alt.Scale(domain=[ymin, ymax])
    panels = []
    for name in PANELS:
        q = arms[name] * 100
        df = pl.DataFrame(
            {
                "x": x, "q01": q[:, IQ[0.01]], "q05": q[:, IQ[0.05]], "q50": q[:, IQ[0.5]],
                "q95": q[:, IQ[0.95]], "q99": q[:, IQ[0.99]], "obs": obs * 100, "nra": nra * 100,
            }
        ).filter(pl.col("q50").is_finite())  # fmt: skip
        xax = alt.X(
            "x:Q",
            title="Days since the load-shift began" if name == PANELS[-1] else None,
            scale=alt.Scale(domain=[-3, length + 7]),
            axis=alt.Axis(values=list(range(-3, int(length) + 8, 3 if length < 20 else 6))),
        )
        span = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [length]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color="#999999", opacity=0.10)
            .encode(x="a:Q", x2="b:Q")
        )
        layers = [span, *_band_layers(df=df, col=COLOUR[name], xax=xax, sc_y=sc_y)]
        if strips:
            layers.insert(
                0,
                alt.Chart(pl.DataFrame(strips))  # ty: ignore[unresolved-attribute]
                .mark_rect(opacity=0.22)
                .encode(
                    x="x0:Q", x2="x1:Q",
                    color=alt.Color(
                        "state:N",
                        scale=alt.Scale(domain=["gate on", "held 7 days"], range=[theme.BRAND_ORANGE, theme.DATA_SKY]),
                        legend=alt.Legend(title="Origin day", orient="top") if name == PANELS[0] else None,
                    ),
                ),
            )  # fmt: skip
        panels.append(
            alt.layer(*layers).properties(
                width=W, height=90, title=alt.TitleParams(LABEL[name], fontSize=11, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(title, fontSize=11, anchor="start")
    ).resolve_scale(x="shared").save(str(DOCS / out))


def _wrap(*lines: str) -> list[str]:
    return list(lines)


def day_ahead_rows(
    *, t: c.X4Table, prim: int, lo: int, hi: int, mask: np.ndarray | None = None
) -> np.ndarray:
    sel = (t.prim == prim) & (t.idx >= lo) & (t.idx < hi) & (t.idx // 48 == t.origin_idx // 48 + 1)
    if mask is not None:
        sel &= mask
    rows = np.flatnonzero(sel)
    return rows[np.argsort(t.idx[rows])]


def verified_figures(*, t: c.X4Table, st: dict) -> None:
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    for e in ev.iter_rows(named=True):
        lo, hi = e["onset_idx"] - 3 * 48, e["end_idx"] + 7 * 48
        rows = day_ahead_rows(t=t, prim=e["primary"], lo=lo, hi=hi)
        if len(rows) < 200:
            continue
        hold_r, on_r = (
            st["hold"][t.prim[rows], t.origin_day[rows]],
            st["on"][t.prim[rows], t.origin_day[rows]],
        )
        blocks = c.building_blocks(v=t.arms["V"][rows], m=t.arms["M"][rows])
        arms = c.combine(base=blocks, hold=hold_r, on=on_r)
        event_panels(
            t=t,
            rows=rows,
            arms=arms,
            obs=t.obs[rows],
            nra=t.nra[rows],
            on=st["on"],
            hold=st["hold"],
            onset=e["onset_idx"],
            end=e["end_idx"],
            prim=e["primary"],
            title=_wrap(
                f"{e['label']}: day-ahead forecast, verified load-shift of {e['depth_mw']:+.1f} MW for {e['length_days']:.0f} days",
                "Orange line: observed power. Black dashed: the normal running arrangement (observed minus the fitted load-shift).",
                "Dark band: q05-q95. Light band: q01-q99. Grey shade: the load-shift. Coloured strips: origin days in event mode.",
            ),
            out=f"verified_{e['label']}_{c_date(e['onset_idx'])}.svg",
        )


def c_date(idx: int) -> str:
    return sc.k.date_of(idx)


def bank_examples(*, t: c.X4Table, gi: dict) -> None:
    """Draw six injected events, chosen by a fixed rule: in month 2025-10 the lowest-numbered
    Primary of each group meeting each of three (length, depth, sign) combinations, taking the
    lowest pass.
    """
    ev_all = pl.read_parquet(s3.BANK_PARQUET).filter(pl.col("month") == "2025-10")
    kk_all = s3.key(t.prim, t.origin_idx, t.idx)
    order = np.argsort(kk_all)
    sorted_keys = kk_all[order]
    in_m = t.month == "2025-10"
    days_m = np.unique(t.idx[in_m] // 48)
    origin_days_m = sorted({int(d) for d in np.unique(t.origin_day[in_m])})
    gap_m = np.full_like(gi["gap"], np.nan)
    gap_m[:, days_m.min() : days_m.max() + 1] = gi["gap"][:, days_m.min() : days_m.max() + 1]
    combos = ((15, 0.5, 1), (30, 0.3, -1), (7, 0.5, -1))
    for gname in s3.GROUPS_SCORED:
        for length, depth, sign in combos:
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
            rows = day_ahead_rows(t=t, prim=p, lo=lo, hi=hi, mask=in_m)
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
            event_panels(
                t=t,
                rows=rows,
                arms=arms,
                obs=t.obs[rows],
                nra=t.obs[rows],
                on=on,
                hold=hold,
                onset=e["onset_idx"],
                end=e["end_idx"],
                prim=p,
                title=_wrap(
                    f"{t.label[p]}, {gname}: injected lag-only event, {sign * depth * 100:+.0f}% of capacity for {length} days (pass {r})",
                    "Only the lag inputs carry the shift; the observed power (orange) is the normal running arrangement (black dashed, same line).",
                    "Dark band: q05-q95. Light band: q01-q99. Grey shade: the injected shift in the lags. Coloured strips: origin days in event mode.",
                ),
                out=f"injected_{gname.split()[0].lower()}_{length}d_{'up' if sign > 0 else 'down'}_{int(depth * 100)}pct.svg",
            )


# ---------------------------------------------------------------- summaries


def coverage_figures() -> None:
    bs = pl.read_csv(c.WORK / "bank_summary.csv").filter(
        (pl.col("band") == "24-120 h") & (pl.col("arm") != "M")
    )
    for cls in ("15 and 30 days", "all lengths"):
        for grp in ("both groups", "MVA-only eight", "signed-MW six"):
            d = bs.filter((pl.col("length_class") == cls) & (pl.col("group") == grp))
            arms = [a for a in COLOUR if a in set(d["arm"].to_list())]
            colours = alt.Scale(domain=arms, range=[COLOUR[a] for a in arms])
            for metric, ytitle, fname in (
                ("cover90", "Share of the NRA inside q05-q95", "coverage90"),
                ("nra_side95", "Share of the NRA on the safe side of q05 (or q95)", "nra_side95"),
            ):
                base = alt.Chart(d).encode(
                    x=alt.X("phase:N", sort=PHASE_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
                    color=alt.Color(
                        "arm:N",
                        scale=colours,
                        legend=alt.Legend(title="Arm", orient="top", columns=4),
                    ),
                    xOffset=alt.XOffset("arm:N", sort=arms),
                )
                pts = base.mark_point(filled=True, size=45, aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y(f"{metric}:Q", title=ytitle, scale=alt.Scale(domain=[0.5, 1.0]))
                )  # ty: ignore[unresolved-attribute]
                err = base.mark_rule(aria=False).encode(y=f"{metric}_lo:Q", y2=f"{metric}_hi:Q")  # ty: ignore[unresolved-attribute]
                nom = (
                    alt.Chart(pl.DataFrame({"y": [0.90 if metric == "cover90" else 0.95]}))  # ty: ignore[unresolved-attribute]
                    .mark_rule(color="black", strokeDash=[4, 3])
                    .encode(y="y:Q")
                )  # ty: ignore[unresolved-attribute]
                alt.layer(nom, err, pts).properties(
                    width=W,
                    height=240,
                    title=alt.TitleParams(
                        [
                            f"{ytitle} (larger is better), injected events of {cls}, {grp}, 24-120 h",
                            "Dashed line: nominal level. Point: mean over events. Line: 95% bootstrap over events (descriptive).",
                        ],
                        fontSize=11,
                        anchor="start",
                    ),
                ).save(str(DOCS / f"{fname}_{cls.replace(' ', '_')}_{grp.replace(' ', '_')}.svg"))
    # verified events: coverage per event and phase (one panel per event)
    rp = pl.read_csv(c.WORK / "real_event_phase.csv").filter(pl.col("band") == "24-120 h")
    rp = rp.filter(pl.col("arm").is_in(list(COLOUR)) & (pl.col("arm") != "M"))
    arms = [a for a in COLOUR if a in set(rp["arm"].to_list())]
    colours = alt.Scale(domain=arms, range=[COLOUR[a] for a in arms])
    nom = (
        alt.Chart(pl.DataFrame({"y": [0.90]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    panels = []
    for ev_name in sorted(rp["event"].unique().to_list()):
        base = alt.Chart(rp.filter(pl.col("event") == ev_name)).encode(
            x=alt.X("phase:N", sort=PHASE_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            color=alt.Color(
                "arm:N", scale=colours, legend=alt.Legend(title="Arm", orient="top", columns=4)
            ),
            xOffset=alt.XOffset("arm:N", sort=arms),
        )
        pts = base.mark_point(filled=True, size=40, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("cover90:Q", title="Share inside q05-q95", scale=alt.Scale(domain=[0, 1]))
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(nom, pts).properties(width=W, height=130, title=ev_name))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Verified load-shifts: share of the NRA inside q05-q95 by phase, one panel per event, 24-120 h",
                "Dashed line: nominal 0.90. One point per arm; each event is one Primary (descriptive, no interval).",
            ],
            fontSize=11,
            anchor="start",
        )
    ).save(str(DOCS / "coverage90_verified_events.svg"))


def false_alarm_figures() -> None:
    fr = pl.read_csv(c.WORK / "event_free.csv").filter(
        (pl.col("band") == "24-120 h")
        & pl.col("group").is_in(list(s3.GROUPS_SCORED))
        & pl.col("arm").is_in(["Union", "A1", "A2", "A3", "A4"])
    )
    rows = []
    for r in fr.iter_rows(named=True):
        for key, lab in (
            ("loss90", "q05-q95 band (q05 and q95 pinball)"),
            ("loss98", "q01-q99 band (q01 and q99 pinball)"),
        ):
            rows.append(
                {
                    "arm": r["arm"],
                    "group": r["group"],
                    "band": lab,
                    "diff": r[f"{key}_minus_V"],
                    "lo": r[f"{key}_minus_V_lo"],
                    "hi": r[f"{key}_minus_V_hi"],
                }
            )
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
            color=alt.Color(
                "band:N",
                scale=alt.Scale(range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
                legend=alt.Legend(title="Pinball loss of", orient="top", columns=1),
            ),
            yOffset="band:N",
        )
        pts = base.mark_point(filled=True, size=40, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "diff:Q",
                title="Pinball loss minus X3-syn's (points of capacity; negative is better)",
            )
        )  # ty: ignore[unresolved-attribute]
        err = base.mark_rule(aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        cols.append(alt.layer(zero, err, pts).properties(width=240, height=160, title=g))
    alt.hconcat(*cols).properties(
        title=alt.TitleParams(
            [
                "False-alarm cost: pinball loss minus X3-syn's, rows with no load-shift within 7 days, 24-120 h",
                "Point: mean over fold months. Line: month-block bootstrap (5 blocks) at the Bonferroni level of 16 counted contrasts.",
            ],
            fontSize=11,
            anchor="start",
        )
    ).save(str(DOCS / "false_alarm_cost.svg"))
    g = pl.read_csv(c.WORK / "gates.csv").filter(
        pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
    )
    rows = []
    for r in g.iter_rows(named=True):
        for key, lab in (
            ("share_days_event_mode_A1", "A1 and A3"),
            ("share_days_event_mode_A2", "A2"),
            ("share_days_event_mode_placebo", "A4 (placebo)"),
        ):
            rows.append({"group": r["group"], "arm": lab, "share": r[key] * 100})
    alt.Chart(pl.DataFrame(rows)).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", sort=None, title=None),
        x=alt.X("share:Q", title="% of origin days in event mode"),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
            legend=alt.Legend(title="Group", orient="top"),
        ),
        yOffset="group:N",
    ).properties(
        width=W,
        height=130,
        title=alt.TitleParams(
            [
                "Share of origin days with no load-shift within 3 days on which the rule widens the band (smaller is cheaper)"
            ],
            fontSize=11,
            anchor="start",
        ),
    ).save(str(DOCS / "share_days_in_event_mode.svg"))


def horizon_example(*, t: c.X4Table, st: dict) -> None:
    """Draw the full 5-day horizon of one stored run (X3-syn stores 0 to 120 h only) for a winter
    and a summer example at a Primary, X3-syn against the union band.
    """
    cases = (("2025-12-30", 2), ("2026-06-10", 2))
    for date, prim in cases:
        o = [i for i in np.unique(t.origin_idx) if sc.k.date_of(int(i)) == date]
        if not o:
            continue
        rows = np.flatnonzero((t.prim == prim) & (t.origin_idx == o[0]))
        rows = rows[np.argsort(t.idx[rows])]
        blocks = c.building_blocks(v=t.arms["V"][rows], m=t.arms["M"][rows])
        panels = []
        for name in ("V", "Union"):
            q = blocks[name] * 100
            df = pl.DataFrame({"x": t.lead[rows], "q01": q[:, IQ[0.01]], "q05": q[:, IQ[0.05]], "q50": q[:, IQ[0.5]], "q95": q[:, IQ[0.95]], "q99": q[:, IQ[0.99]], "obs": t.obs[rows] * 100, "nra": t.nra[rows] * 100})  # fmt: skip
            xax = alt.X(
                "x:Q",
                title="Hours ahead of the forecast origin" if name == "Union" else None,
                axis=alt.Axis(values=[0, 24, 48, 72, 96, 120]),
            )
            sc_y = alt.Scale(
                domain=[float(np.nanmin(df["obs"])) - 3, float(np.nanmax(df["obs"])) + 3]
            )
            panels.append(
                alt.layer(*_band_layers(df=df, col=COLOUR[name], xax=xax, sc_y=sc_y)).properties(
                    width=W,
                    height=110,
                    title=alt.TitleParams(LABEL[name], fontSize=11, anchor="start"),
                )
            )
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                [
                    f"{t.label[prim]}: one run issued {date} 09:00 UTC, 0 to 120 h ahead (the stored horizon)",
                    "Orange: observed power. Dark band: q05-q95. Light band: q01-q99.",
                ],
                fontSize=11,
                anchor="start",
            )
        ).save(str(DOCS / f"horizon_{date}.svg"))


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    coverage_figures()
    false_alarm_figures()
    t = c.X4Table()
    gi = c.gate_inputs(t=t)
    z = np.load(c.WORK / "gate_state.npz")
    st = {"on": z["on"], "hold": z["hold"]}
    verified_figures(t=t, st=st)
    bank_examples(t=t, gi=gi)
    horizon_example(t=t, st=st)
    _ = bank


if __name__ == "__main__":
    main()
