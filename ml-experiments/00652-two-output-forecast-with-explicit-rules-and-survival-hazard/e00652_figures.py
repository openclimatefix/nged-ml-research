# ruff: noqa: D103, E501
"""Experiment 652, step 4: the figures (CPU). SVG and PNG go to `figures/` in the data folder.

1. Scaled and unscaled CRPS where a shift is in force at the origin, with month-block intervals.
2. CRPS by lead day far from and near a shift's end.
3. CRPS by lead day in the middle of a shift (scaled) and at event-free origins (unscaled).
4. and 5. Survival reliability on training-family and held-out shifts.
6. Two example 10-day forecasts from one origin, far from and near a shift's end.

Usage: python e00652_figures.py
"""

import altair as alt
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00652_fit as f652  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf  # ty: ignore[unresolved-import]
import polars as pl

SC = c652.WORK / "score"
OUT = c652.WORK / "figures"
OUT.mkdir(exist_ok=True)
W = 600
LABEL = {
    "K1": "K1",
    "R0-168": "R0 (168 h)",
    "Bhaz": "B-haz",
    "Rnew": "R-new",
    "Rhaz": "R-haz",
    "Placebo": "Placebo",
}
COLOUR = {
    "K1": ocf.DATA_BLUE,
    "R0 (168 h)": ocf.DATA_SKY,
    "B-haz": ocf.DATA_PURPLE,
    "R-new": ocf.BRAND_ORANGE_LIGHT,
    "R-haz": ocf.BRAND_ORANGE,
    "Placebo": ocf.BLACK_1,
}
ORDER = list(COLOUR)


def save(chart: alt.TopLevelMixin, name: str) -> None:
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / f"{name}.png", scale_factor=2.0)
    print("saved", name, flush=True)


def colour() -> alt.Color:
    return alt.Color(
        "arm:N",
        sort=ORDER,
        scale=alt.Scale(domain=ORDER, range=[COLOUR[k] for k in ORDER]),
        legend=alt.Legend(title=None),
    )


def levels_panel(*, metric: str, stratum: str, title: str) -> alt.Chart:
    d = pl.read_csv(SC / "levels.csv").filter(
        (pl.col("metric") == metric)
        & (pl.col("set") == "fam")
        & (pl.col("mixture") == "m50")
        & (pl.col("stratum") == stratum)
        & pl.col("arm").is_in(list(LABEL))
    )
    d = d.with_columns(arm=pl.col("arm").replace(LABEL))
    base = alt.Chart(d).encode(y=alt.Y("arm:N", sort=ORDER, title=None), color=colour())
    dots = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "mean:Q",
            title="CRPS, points of capacity (smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )  # ty: ignore[unresolved-attribute]
    err = base.mark_rule(strokeWidth=2, aria=False).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    return (err + dots).properties(
        width=W - 150, height=170, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def lead_panel(*, metric: str, stratum: str, title: str) -> alt.Chart:
    d = pl.read_csv(SC / f"by_lead_day_{metric}.csv").filter(
        (pl.col("set") == "fam")
        & (pl.col("mixture") == "m50")
        & (pl.col("stratum") == stratum)
        & pl.col("arm").is_in(list(LABEL))
    )
    d = d.with_columns(arm=pl.col("arm").replace(LABEL))
    base = alt.Chart(d).encode(
        x=alt.X("lead_day:O", title="Lead day (1 is 24 to 47 h ahead)"), color=colour()
    )
    line = base.mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("mean:Q", title="CRPS, points of capacity", scale=alt.Scale(zero=False))
    )  # ty: ignore[unresolved-attribute]
    band = base.mark_errorband(opacity=0.12).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("min:Q", title="CRPS, points of capacity"), y2="max:Q"
    )  # ty: ignore[unresolved-attribute]
    return (band + line).properties(
        width=W - 80, height=230, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def reliability(*, sset: str, name: str, title: list[str]) -> None:
    r = pl.read_csv(SC / "reliability.csv").filter(
        (pl.col("set") == sset)
        & (pl.col("mixture") == "m50")
        & pl.col("arm").is_in(["Rhaz", "Bhaz"])
        & (pl.col("n") >= 200)
    )
    r = r.with_columns(arm=pl.col("arm").replace({"Rhaz": "R-haz", "Bhaz": "B-haz"}))
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999999", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q", y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    panels = []
    for arm in ("R-haz", "B-haz"):
        pts = alt.Chart(r.filter(pl.col("arm") == arm)).mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("pred:Q", title="Predicted survival probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("obs:Q", title="Observed share still on", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("band:N", title="Lead days", scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE])),
        )  # fmt: skip
        panels.append(
            (diag + pts).properties(
                width=(W - 120) // 2,
                height=240,
                title=alt.TitleParams(arm, anchor="start", fontSize=12),
            )
        )
    save(
        alt.hconcat(*panels).properties(
            title=alt.TitleParams(title[0], subtitle=title[1:], anchor="start")
        ),
        name,
    )


def examples() -> None:
    m = "2025-12"
    src = f650.sources(month=m, bank="new")
    z = src["bank"]
    f652.with_state(z=z, month=m)
    fit = {a: np.load(f652.OUT / f"{a}_{m}_s0.npz") for a in ("Rhaz_m50",)}
    r5 = np.load(c652.c650.WORK / "fits5" / f"Rnew_{m}_s0.npz")
    idx = fit["Rhaz_m50"]["fam_idx"]
    qh = np.sort(fit["Rhaz_m50"]["fam_q"], axis=1)
    qn = np.sort(r5["fam_q"], axis=1)
    sel_all = {
        k: z[k][idx]
        for k in (
            "prim",
            "oi",
            "vi",
            "lead",
            "q",
            "y",
            "shift",
            "pass_",
            "onset",
            "end",
            "stratum",
            "res",
            "org",
        )
    }
    ok = np.isfinite(sel_all["res"]) & (sel_all["pass_"] < 200)
    panels = []
    for want, label in (
        ("far", "Origin far from the shift's end"),
        ("near", "Origin within 24 h of the shift's end"),
    ):
        cand = np.flatnonzero(
            ok
            & (sel_all["stratum"] == want)
            & (sel_all["end"] - sel_all["onset"] > 21 * 48)
            & (sel_all["end"] - sel_all["onset"] < 60 * 48)
        )
        if want == "near":
            cand = np.flatnonzero(
                ok & (sel_all["stratum"] == want) & (sel_all["end"] - sel_all["onset"] > 7 * 48)
            )
        key = (
            sel_all["pass_"][cand].astype(np.int64) * 100 + sel_all["prim"][cand]
        ) * 10_000_000 + sel_all["oi"][cand].astype(np.int64)
        keys, counts = np.unique(key, return_counts=True)
        pick = keys[np.argmax(counts)]
        rows = cand[key == pick]
        rows = rows[np.argsort(sel_all["lead"][rows])]
        d = pl.DataFrame({
            "hours": sel_all["lead"][rows], "Truth": (sel_all["y"][rows] + sel_all["shift"][rows]) * 100,
            "K1": sel_all["q"][rows, 5] * 100, "R-haz": qh[rows, 5] * 100, "R-new": qn[rows, 5] * 100,
            "lo": qh[rows, 1] * 100, "hi": qh[rows, 9] * 100,
        })  # fmt: skip
        end_h = float((sel_all["end"][rows[0]] - sel_all["oi"][rows[0]]) / 2.0)
        band = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.2, color=ocf.BRAND_ORANGE, aria=False)
            .encode(
                x=alt.X("hours:Q", title="Hours after the forecast origin"),
                y=alt.Y("lo:Q", title="Power, points of capacity", scale=alt.Scale(zero=False)),
                y2="hi:Q",
            )
        )  # ty: ignore[unresolved-attribute]
        long = d.unpivot(
            index="hours",
            on=["Truth", "K1", "R-haz", "R-new"],
            variable_name="series",
            value_name="v",
        )
        lines = alt.Chart(long).mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x="hours:Q", y="v:Q",
            color=alt.Color("series:N", scale=alt.Scale(domain=["Truth", "K1", "R-new", "R-haz"], range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE_LIGHT, ocf.BRAND_ORANGE]), legend=alt.Legend(title=None)),
        )  # fmt: skip
        rule = (
            alt.Chart(pl.DataFrame({"h": [end_h if end_h <= 240 else -10.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#999999", aria=False)
            .encode(x="h:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            (band + lines + rule).properties(
                width=W - 80,
                height=200,
                title=alt.TitleParams(
                    label
                    + (
                        f" (shift ends at {end_h:.0f} h)"
                        if end_h < 241
                        else " (shift ends after the horizon)"
                    ),
                    anchor="start",
                    fontSize=12,
                ),
            )
        )
    save(
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                "Two injected shifts: the head carries the shift past its end",
                subtitle=[
                    "Medians of K1, R-new, and R-haz against power with the shift; band: R-haz q05-q95; dashed line: end of the shift."
                ],
                anchor="start",
            )
        ),
        "fig6_examples",
    )


def main() -> None:
    save(alt.vconcat(
        levels_panel(metric="crps_s", stratum="inforce", title="Scaled CRPS, shift in force at the origin"),
        levels_panel(metric="crps", stratum="inforce", title="Unscaled CRPS, shift in force at the origin"),
    ).properties(title=alt.TitleParams("R-haz and a fitted survival curve score alike where a shift is in force", subtitle=["Dot: mean over 5 development months; line: 95% interval from resampling the months. Unseen training-family shifts."], anchor="start")), "fig1_inforce")  # fmt: skip
    save(alt.vconcat(
        lead_panel(metric="crps_s", stratum="far", title="Origin far from the shift's end (more than 10 days remain)"),
        lead_panel(metric="crps_s", stratum="near", title="Origin within 24 h of the shift's end"),
    ).properties(title=alt.TitleParams("Carrying a shift forward wins far from its end and loses near it", subtitle=["Scaled CRPS by lead day; band: range over the 5 development months."], anchor="start")), "fig2_far_near")  # fmt: skip
    save(alt.vconcat(
        lead_panel(metric="crps_s", stratum="middle", title="Origin 1 to 10 days before the shift's end (scaled CRPS)"),
        lead_panel(metric="crps", stratum="free", title="Event-free origin (unscaled CRPS)"),
    ).properties(title=alt.TitleParams("In the middle of a shift the arms are close; event-free origins show no loss", subtitle=["Band: range over the 5 development months."], anchor="start")), "fig3_middle_free")  # fmt: skip
    reliability(
        sset="fam",
        name="fig4_reliability_training_family",
        title=[
            "Survival reliability on unseen training-family shifts",
            "In-force origins, event-weighted; bins with at least 200 rows; dashed line is perfect calibration.",
        ],
    )
    reliability(
        sset="held",
        name="fig5_reliability_held_out",
        title=[
            "Survival reliability on held-out durations (5, 10, 21, and 45 days)",
            "In-force origins, event-weighted; bins with at least 200 rows; dashed line is perfect calibration.",
        ],
    )
    examples()


if __name__ == "__main__":
    main()
