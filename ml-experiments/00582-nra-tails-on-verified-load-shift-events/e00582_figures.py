# ruff: noqa: E501, RUF005, D103, D205
"""Experiment 582 figures: day-ahead forecasts with bands around each verified event, and phase
summaries.

Run `e00582_score.py` first (it writes the CSVs this script reads). The event figures rebuild the
arms from the stored forecasts, so this script takes a few minutes. Writes SVGs to the docs
folder.
"""

from pathlib import Path

import altair as alt
import e00582_arms as a  # ty: ignore[unresolved-import]
import e00582_common as k  # ty: ignore[unresolved-import]
import e00582_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
IQ = a.IQ
PANELS = ("v3+D5", "Masked", "Lag-free D5", "R", "Mix50-R", "MixPL-R", "Union", "G2+SB")
ARM_COLOURS = {
    "v3+D5": theme.DATA_PURPLE,
    "Masked": "#6b6b6b",
    "Lag-free D5": theme.DATA_GREEN,
    "R": theme.DATA_SKY,
    "Mix50-R": theme.DATA_BLUE,
    "MixPL-R": theme.DATA_BLUE,
    "Union": "#6b6b6b",
    "G2+SB": theme.DATA_PURPLE,
}
ARM_LABEL = {
    "v3+D5": "v3+D5, observed lags",
    "Masked": "Lags masked",
    "Lag-free D5": "Lag-free D5 network",
    "R": "R: v3+D5 recalibrated on earlier months",
    "Mix50-R": "50/50 mixture of v3+D5 and lag-free D5, recalibrated",
    "MixPL-R": "Mixture with per-lead weights, recalibrated",
    "Union": "Union band of v3+D5 and lags masked (median: v3+D5)",
    "G2+SB": "Gated (G2): masked while on, week lag masked for 7 days after",
}
WEEKDAYS = 48


def day_ahead_rows(
    *, t: a.Table, arms: dict[str, np.ndarray], prim: int, lo: int, hi: int
) -> np.ndarray:
    """Return row indices of the day-ahead forecasts (issued 09:00 the day before the valid day)
    of a Primary for valid times in [lo, hi).
    """
    sel = (
        (t.prim == prim)
        & (t.idx >= lo)
        & (t.idx < hi)
        & (t.idx // WEEKDAYS == t.origin_idx // WEEKDAYS + 1)
    )
    return np.flatnonzero(sel)


def event_figure(*, t: a.Table, arms: dict[str, np.ndarray], gates: dict, e: dict) -> None:
    """Draw one stacked figure: observed power, the NRA, and an arm's bands, 3 days before onset
    to 7 days after the end.
    """
    lo, hi = e["onset_idx"] - 3 * 48, e["end_idx"] + 7 * 48
    rows = day_ahead_rows(t=t, arms=arms, prim=e["primary"], lo=lo, hi=hi)
    rows = rows[np.argsort(t.idx[rows])]
    if len(rows) < 200:
        return
    x = (t.idx[rows] - e["onset_idx"]) / 48.0
    length = (e["end_idx"] - e["onset_idx"]) / 48.0
    obs, nra = t.obs[rows] * 100, t.nra[rows] * 100
    G = gates["state"]["G2"][e["primary"]]
    strips = []
    for d in range(lo // 48, hi // 48 + 1):
        if d in set(gates["origin_days"]):
            state = (
                "gate on"
                if G[d]
                else (
                    "held 7 days"
                    if a.hold_after_release(
                        on=gates["state"]["G2"][e["primary"]][None, :],
                        valid_days=a._origin_day_mask(t=t, n=G.shape[0]),
                    )[0, d]
                    else "off"
                )
            )
            if state != "off":
                strips.append(
                    {
                        "x0": (d + 1) * 48 / 48.0 - e["onset_idx"] / 48.0,
                        "x1": (d + 2) * 48 / 48.0 - e["onset_idx"] / 48.0,
                        "state": state,
                    }
                )
    panels = []
    ymin, ymax = (
        float(np.nanmin(np.minimum(obs, nra))) - 3,
        float(np.nanmax(np.maximum(obs, nra))) + 3,
    )
    for name in PANELS:
        q = arms[name][rows] * 100
        df = (
            pl.DataFrame(
                {
                    "x": x,
                    "q01": q[:, IQ[0.01]],
                    "q05": q[:, IQ[0.05]],
                    "q50": q[:, IQ[0.5]],
                    "q95": q[:, IQ[0.95]],
                    "q99": q[:, IQ[0.99]],
                    "obs": obs,
                    "nra": nra,
                }
            )
            .drop_nulls()
            .filter(pl.col("q50").is_finite())
        )
        if df.height < 100:
            continue
        col = ARM_COLOURS[name]
        sc_y = alt.Scale(domain=[ymin, ymax])
        xax = alt.X(
            "x:Q",
            title="Days since the load-shift began" if name == PANELS[-1] else None,
            scale=alt.Scale(domain=[-3, length + 7]),
            axis=alt.Axis(values=list(range(-3, int(length) + 8, 3 if length < 30 else 6))),
        )
        base = alt.Chart(df)
        b98 = base.mark_area(color=col, opacity=0.18).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("q01:Q", scale=sc_y, title="% of capacity"), y2="q99:Q"
        )
        b90 = base.mark_area(color=col, opacity=0.35).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("q05:Q", scale=sc_y), y2="q95:Q"
        )
        med = base.mark_line(color=col, strokeWidth=1).encode(x=xax, y=alt.Y("q50:Q", scale=sc_y))  # ty: ignore[unresolved-attribute]
        o = base.mark_line(color=theme.BRAND_ORANGE, strokeWidth=1).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("obs:Q", scale=sc_y)
        )
        n = base.mark_line(color="black", strokeDash=[3, 2], strokeWidth=1).encode(  # ty: ignore[unresolved-attribute]
            x=xax, y=alt.Y("nra:Q", scale=sc_y)
        )
        span = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [length]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(color="#999999", opacity=0.10)
            .encode(x="a:Q", x2="b:Q")
        )
        layers = [span, b98, b90, med, o, n]
        if name == "G2+SB" and strips:
            s = pl.DataFrame(strips)
            layers.insert(
                0,
                alt.Chart(s)  # ty: ignore[unresolved-attribute]
                .mark_rect(opacity=0.25)
                .encode(
                    x="x0:Q",
                    x2="x1:Q",
                    color=alt.Color(
                        "state:N",
                        scale=alt.Scale(
                            domain=["gate on", "held 7 days"],
                            range=[theme.BRAND_ORANGE, theme.DATA_SKY],
                        ),
                        legend=alt.Legend(title="Gate", orient="top"),
                    ),
                ),
            )
        panels.append(
            alt.layer(*layers).properties(
                width=620,
                height=95,
                title=alt.TitleParams(ARM_LABEL[name], fontSize=11, anchor="start"),
            )
        )
    lab = e["name"]
    title = [
        f"{lab}: day-ahead forecast, load-shift of {e['depth_mw']:+.1f} MW for {e['length_days']:.0f} days",
        "Orange: observed power. Black dashed: the NRA (observed minus the fitted load-shift).",
        "Dark band: q05-q95. Light band: q01-q99. Grey shade: the load-shift.",
    ]
    chart = (
        alt.vconcat(*panels)
        .properties(title=alt.TitleParams(title, fontSize=11, anchor="start", subtitleFontSize=10))
        .resolve_scale(x="shared")
    )
    chart.save(str(DOCS / f"event_{lab.replace(' ', '_')}.svg"))


def summary_figures() -> None:
    """Phase summaries: coverage, follow ratio, and the false-alarm cost."""
    ps = pl.read_csv(k.WORK / "phase_summary.csv").filter(
        (pl.col("band") == "24-120 h") & (pl.col("group") == "both groups")
    )
    keep = ["v3+D5", "Masked", "Lag-free D5", "R", "Mix50-R", "MixPL-R", "Union", "G2+SB"]
    order = ["before", "days0-3", "days4-end", "after0-7", "after7-14"]
    d = ps.filter(pl.col("arm").is_in(keep))
    colours = alt.Scale(
        domain=keep,
        range=[
            theme.DATA_PURPLE,
            "#6b6b6b",
            theme.DATA_GREEN,
            theme.DATA_SKY,
            theme.DATA_BLUE,
            "#1a3a99",
            "#b0b0b0",
            theme.BRAND_ORANGE,
        ],
    )
    for metric, title, fname in (
        (
            "cover90",
            "Share of the NRA inside q05-q95 (nominal 0.90; larger is better)",
            "coverage90_by_phase.svg",
        ),
        (
            "nra_side_cover95",
            "Share of the NRA on the safe side of q95 or q05 (nominal 0.95)",
            "coverage_nra_side_by_phase.svg",
        ),
        (
            "follow_safety",
            "Tail follow ratio on the NRA side (0: tail did not move with the event; smaller is better)",
            "tail_follow_by_phase.svg",
        ),
    ):
        base = alt.Chart(d).encode(
            x=alt.X("phase:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            color=alt.Color(
                "arm:N", scale=colours, legend=alt.Legend(title="Arm", orient="top", columns=4)
            ),
            xOffset=alt.XOffset("arm:N", sort=keep),
        )
        pts = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(f"{metric}:Q", title=title.split(" (")[0])
        )
        err = base.mark_rule().encode(y=f"{metric}_lo:Q", y2=f"{metric}_hi:Q")  # ty: ignore[unresolved-attribute]
        alt.layer(err, pts).properties(
            width=620,
            height=240,
            title=alt.TitleParams(
                [
                    title,
                    "Point: mean of 6 Primary-events, 24-120 h. Line: 95% event-block bootstrap (descriptive).",
                ],
                fontSize=11,
                anchor="start",
            ),
        ).save(str(DOCS / fname))
    fr = pl.read_csv(k.WORK / "event_free.csv").filter(
        (pl.col("band") == "24-120 h")
        & pl.col("group").is_in(["MVA-only eight", "signed-MW six"])
        & pl.col("arm").is_in(keep + ["Mix50", "Mix70", "MixPL", "G1", "G1+SB", "G2"])
    )
    rows = []
    for r in fr.iter_rows(named=True):
        for key, lab in (("pinball_q05", "q05"), ("pinball_q95", "q95")):
            rows.append(
                {
                    "arm": r["arm"],
                    "group": r["group"],
                    "level": lab,
                    "diff": r[f"{key}_minus_R"],
                    "lo": r[f"{key}_minus_R_lo"],
                    "hi": r[f"{key}_minus_R_hi"],
                }
            )
    df = pl.DataFrame(rows).filter(pl.col("arm") != "R")
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    cols = []
    for g in ("MVA-only eight", "signed-MW six"):
        base = alt.Chart(df.filter(pl.col("group") == g)).encode(
            y=alt.Y("arm:N", sort=None, title=None),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=["q05", "q95"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
                legend=alt.Legend(title="Pinball level", orient="top"),
            ),
            yOffset="level:N",
        )
        pts = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("diff:Q", title="Pinball loss minus R (points of capacity; negative is better)")
        )
        err = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
        cols.append(alt.layer(zero, err, pts).properties(width=240, height=260, title=g))
    alt.hconcat(*cols).properties(
        title=alt.TitleParams(
            [
                "False-alarm cost: pinball loss minus R, rows with no labelled event, 24-120 h",
                "Point: mean over fold months. Line: month-block bootstrap (5 blocks, near min to max).",
            ],
            fontSize=11,
            anchor="start",
        )
    ).save(str(DOCS / "false_alarm_cost.svg"))
    cal = pl.read_csv(k.WORK / "event_free.csv").filter(
        (pl.col("band") == "24-120 h")
        & (pl.col("group") != "D15 and D18")
        & pl.col("arm").is_in(keep)
    )
    rows = [
        {"arm": r["arm"], "group": r["group"], "level": f"q{lv}", "share": r[f"below_q{lv}"]}
        for r in cal.iter_rows(named=True)
        for lv in ("05", "95", "99")
    ]
    nominal = {"q05": 0.05, "q95": 0.95, "q99": 0.99}
    cdf = pl.DataFrame(rows).with_columns(nominal=pl.col("level").replace_strict(nominal))
    cols = []
    for lv in ("q05", "q95", "q99"):
        sub = cdf.filter(pl.col("level") == lv)
        base = alt.Chart(sub).encode(y=alt.Y("arm:N", sort=None, title=None))
        pts = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share:Q", title="Share at or below"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["MVA-only eight", "signed-MW six"],
                    range=[theme.DATA_BLUE, theme.BRAND_ORANGE],
                ),
                legend=alt.Legend(title="Group", orient="top"),
            ),
        )
        nom = base.mark_tick(color="black", thickness=2, size=18).encode(x="nominal:Q")  # ty: ignore[unresolved-attribute]
        cols.append(alt.layer(nom, pts).properties(width=170, height=180, title=lv))
    alt.hconcat(*cols).properties(
        title=alt.TitleParams(
            [
                "Calibration on rows with no labelled event, 24-120 h",
                "Black tick: nominal level. Point: share of readings at or below the quantile.",
            ],
            fontSize=11,
            anchor="start",
        )
    ).save(str(DOCS / "calibration_event_free.svg"))


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    summary_figures()
    t = a.Table()
    gates = a.run_gates(t=t)
    arms, _, _ = sc.build_arms(t=t, gates=gates)
    ev = sc.attach_depth(ev=sc.merged_events(), t=t)
    for e in ev.iter_rows(named=True):
        event_figure(t=t, arms=arms, gates=gates, e=e)


if __name__ == "__main__":
    main()
