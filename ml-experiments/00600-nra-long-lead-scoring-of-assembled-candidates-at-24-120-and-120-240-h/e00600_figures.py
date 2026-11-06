# ruff: noqa: E501, D103, ANN001, ANN201
"""Experiment 600: figures, written as SVG beside `result.md`.

Usage: python e00600_figures.py [bars|series]. `bars` reads the scored tables only; `series` rebuilds
the row tables of C1 and A1 (about a minute each) to draw the December 2025 pair.
"""

import sys
from pathlib import Path

import altair as alt
import e00583_score as s3  # ty: ignore[unresolved-import]
import e00600_common as k  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
W = 300
LONG, MID = "120-240 h", "24-120 h"
BANDS = [k.BANDS[0], MID, LONG]
PHASE_LABEL = {
    "before": "Before",
    "days0-3": "Days 0-3",
    "days4-end": "Day 4 to end",
    "after0-7": "0-7 after",
    "after7-14": "7-14 after",
}
PHASES = list(PHASE_LABEL.values())
SYSTEMS = ["C1", "C1 + G4", "A1", "A1 + G4"]
COLOUR = {
    "C1": theme.BRAND_ORANGE_LIGHT,
    "C1 + G4": theme.BRAND_ORANGE,
    "A1": theme.DATA_BLUE_LIGHT,
    "A1 + G4": theme.DATA_BLUE,
}
SCALE = alt.Scale(domain=SYSTEMS, range=[COLOUR[s] for s in SYSTEMS])
RNG = np.random.default_rng(600)
KEYS = ["month", "pass", "primary", "sign", "depth", "length_days"]


def save(chart, name: str) -> None:
    chart.save(str(DOCS / name))


def boot(vals: np.ndarray) -> tuple[float, float, float]:
    v = vals[np.isfinite(vals)]
    m = v[RNG.integers(0, len(v), size=(10000, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bank() -> pl.DataFrame:
    out = []
    for arm, lab in (("c1", "C1"), ("a1", "A1")):
        d = pl.read_csv(k.WORK / arm / "bank_event_phase.csv", infer_schema_length=None).filter(
            pl.col("length_days") >= 15
        )
        out.append(
            d.with_columns(
                system=pl.when(pl.col("system") == "plain")
                .then(pl.lit(lab))
                .otherwise(pl.lit(f"{lab} + G4")),
                arm=pl.lit(lab),
            )
        )
    return pl.concat(out)


def coverage_chart(
    *,
    metric: str,
    ytitle: str,
    target: float,
    domain: list[float],
    name: str,
    caption: str,
    group: str,
) -> None:
    b = bank()
    if group != "both groups":
        b = b.filter(pl.col("group") == group)
    rows = []
    for (system, band, phase), g in b.group_by("system", "band", "phase"):
        if g.height < 20:
            continue
        m, lo, hi = boot(g[metric].to_numpy())
        rows.append(
            {
                "system": system,
                "band": band,
                "phase": PHASE_LABEL[phase],
                "v": m,
                "lo": lo,
                "hi": hi,
            }
        )
    df = pl.DataFrame(rows)
    ys = alt.Scale(domain=domain)
    base = alt.Chart().encode(
        x=alt.X("phase:N", sort=PHASES, title=None, axis=alt.Axis(labelAngle=-40)),
        xOffset=alt.XOffset("system:N", sort=SYSTEMS),
        color=alt.Color(
            "system:N",
            scale=SCALE,
            sort=SYSTEMS,
            legend=alt.Legend(title="Forecast", orient="bottom", columns=4),
        ),
    )
    layer = alt.layer(
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=theme.BLACK_1)
        .encode(y=alt.datum(target, scale=ys)),
        base.mark_rule().encode(y=alt.Y("lo:Q", scale=ys, title=ytitle), y2="hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=45).encode(y=alt.Y("v:Q", scale=ys)),  # ty: ignore[unresolved-attribute]
        data=df,
    ).properties(width=W, height=290)
    chart = layer.facet(column=alt.Column("band:N", sort=BANDS, title="Hours ahead")).properties(
        title=alt.TitleParams([caption], fontSize=11, anchor="start")
    )
    save(chart, name)


def gate_value() -> None:
    b = bank()
    rows = []
    for lab in ("C1", "A1"):
        for phase in ("days4-end", "after0-7"):
            for band in BANDS:
                g4 = b.filter(
                    (pl.col("system") == f"{lab} + G4")
                    & (pl.col("phase") == phase)
                    & (pl.col("band") == band)
                ).select(*KEYS, g4="cover90")
                pl_ = b.filter(
                    (pl.col("system") == lab)
                    & (pl.col("phase") == phase)
                    & (pl.col("band") == band)
                ).select(*KEYS, plain="cover90")
                j = g4.join(pl_, on=KEYS)
                if j.height < 20:
                    continue
                m, lo, hi = boot((j["g4"] - j["plain"]).to_numpy())
                rows.append(
                    {
                        "network": lab,
                        "phase": PHASE_LABEL[phase],
                        "band": band,
                        "gain": m,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    df = pl.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("band:N", sort=BANDS, title="Hours ahead", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("network:N"),
        color=alt.Color(
            "network:N",
            scale=alt.Scale(domain=["C1", "A1"], range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
            legend=alt.Legend(title="Network with G4 minus the network alone", orient="bottom"),
        ),
    )
    layer = alt.layer(
        base.mark_rule().encode(y=alt.Y("lo:Q", title="Coverage gained by the G4 band"), y2="hi:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=55).encode(y="gain:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=W, height=290)
    save(
        layer.facet(
            column=alt.Column("phase:N", sort=["Day 4 to end", "0-7 after"], title=None)
        ).properties(
            title=alt.TitleParams(
                ["Coverage the G4 band adds on injected 15 and 30 day load-shifts, by lead"],
                fontSize=11,
                anchor="start",
            )
        ),
        "gate_value_by_lead.svg",
    )


def event_free() -> None:
    parts = []
    for arm, lab in (("c0", "C0"), ("c1", "C1"), ("a1", "A1")):
        d = pl.read_csv(k.WORK / arm / "event_free.csv", infer_schema_length=None).filter(
            pl.col("group").is_in(list(k.GROUPS_SCORED)) & (pl.col("system").is_in(["plain", "E2"]))
        )
        parts.append(
            d.with_columns(
                net=pl.when(pl.col("system") == "E2")
                .then(pl.lit("E2 (no lags)"))
                .otherwise(pl.lit(lab))
            ).filter((pl.col("system") == "plain") | (pl.lit(arm) == "c1"))
        )
    df = pl.concat(parts).with_columns(
        group_name=pl.col("group").replace({g: k.GROUP_NAME[g] for g in k.GROUPS_SCORED})
    )
    order = ["C0", "C1", "A1", "E2 (no lags)"]
    sc = alt.Scale(
        domain=order,
        range=[theme.DATA_AMBER, theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE],
    )
    ch = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=BANDS, title="Hours ahead", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("net:N", sort=order),
            y=alt.Y("crps:Q", title="CRPS (points of capacity)"),
            color=alt.Color(
                "net:N",
                scale=sc,
                sort=order,
                legend=alt.Legend(title="Forecast", orient="bottom", columns=4),
            ),
        )
        .properties(width=W + 40, height=290)
    )
    save(
        ch.facet(column=alt.Column("group_name:N", title=None)).properties(
            title=alt.TitleParams(
                [
                    "Event-free rows: the error grows with lead, and the lag networks keep their edge over the lag-free network E2"
                ],
                fontSize=11,
                anchor="start",
            )
        ),
        "event_free_crps_by_lead.svg",
    )


# ---------------------------------------------------------------- the December 2025 pair

SWITCH = ("2025-12-30T01:00:00", "2026-01-14T11:30:00")
OBS, SHIFT, WIDE = "#000000", "#2a7d4f", "#7b4ea3"


def series_panel(
    *,
    df: pl.DataFrame,
    strip: pl.DataFrame,
    colour: str,
    label: str,
    title: str,
    xdom: tuple,
    lead_note: str,
):
    """One panel: median and q05-q95 band of one network, observed power, switch times; a thin strip below."""
    leg = alt.Scale(
        domain=["Observed power", "Median", "q05-q95 band", "Switch times", "G4 wide band on"],
        range=[OBS, colour, colour, SHIFT, WIDE],
    )
    cs = alt.Color(
        "series:N",
        scale=leg,
        legend=alt.Legend(title=None, orient="bottom", columns=5, symbolType="stroke"),
    )
    x = alt.X(
        "t:T",
        title=None,
        scale=alt.Scale(domain=list(xdom)),
        axis=alt.Axis(format="%d %b", labelAngle=0),
    )
    base = alt.Chart(df)
    lo = float(min(df["lo"].min(), df["obs"].min())) - 2  # ty: ignore[invalid-argument-type]
    hi = float(max(df["hi"].max(), df["obs"].max())) + 2  # ty: ignore[invalid-argument-type]
    ys = alt.Scale(domain=[lo, hi])
    sw = (
        alt.Chart(pl.DataFrame({"t": list(SWITCH), "series": ["Switch times"] * 2}))  # ty: ignore[unresolved-attribute]
        .transform_calculate(t="toDate(datum.t)")
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.6)
        .encode(x=x, color=cs)
    )
    layers = [
        base.mark_area(opacity=0.25)  # ty: ignore[unresolved-attribute]
        .transform_calculate(series="'q05-q95 band'")
        .encode(x=x, y=alt.Y("lo:Q", scale=ys, title="% of capacity"), y2="hi:Q", color=cs),
        base.mark_line(strokeWidth=1.4)  # ty: ignore[unresolved-attribute]
        .transform_calculate(series="'Median'")
        .encode(x=x, y=alt.Y("med:Q", scale=ys), color=cs),
        base.mark_line(strokeWidth=1.0)  # ty: ignore[unresolved-attribute]
        .transform_calculate(series="'Observed power'")
        .encode(x=x, y=alt.Y("obs:Q", scale=ys), color=cs),
        sw,
    ]
    main = alt.layer(*layers).properties(
        width=640,
        height=300,
        title=alt.TitleParams([f"{title}: {label}", lead_note], fontSize=11, anchor="start"),
    )
    st = (
        alt.Chart(strip)  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.9)
        .transform_calculate(series="'G4 wide band on'")
        .encode(
            x=alt.X("t0:T", scale=alt.Scale(domain=list(xdom)), title=None, axis=None),
            x2="t1:T",
            color=cs,
        )
        .properties(width=640, height=14)
    )
    return alt.vconcat(main, st, spacing=2)


def series_figures() -> None:
    import e00600_score as sc600  # ty: ignore[unresolved-import]

    out = {}
    for arm, lab, colour in (
        ("c1", "C1 + G4", theme.BRAND_ORANGE),
        ("a1", "A1 + G4", theme.DATA_BLUE),
    ):
        inp = sc600.Inputs(arm=arm)
        t = inp.t
        labels = list(t.label)
        panels_long = []
        for pl_name in ("D5", "D16"):
            p = labels.index(pl_name)
            lo_ts, hi_ts = (
                np.datetime64("2025-12-16T00:00:00"),
                np.datetime64("2026-01-28T00:00:00"),
            )
            ts_all = (
                s3.c.k578.prep.GRID_START_US + t.idx.astype(np.int64) * s3.c.k578.prep.STEP_US
            ).astype("datetime64[us]")
            sel = (
                (t.prim == p)
                & (ts_all >= lo_ts)
                & (ts_all < hi_ts)
                & (t.lead >= 216)
                & (t.lead < 240)
            )
            rows = np.flatnonzero(sel)
            rows = rows[np.argsort(t.idx[rows])]
            q = k.k597.assemble(
                v=t.arms["V"][rows],
                m=t.arms["M"][rows],
                wide=inp.mode[t.prim[rows], t.origin_day[rows]],
            )
            iq = k.k597.c.IQ
            ts = ts_all[rows]
            df = pl.DataFrame(
                {
                    "t": pl.Series(ts.astype("datetime64[us]")),
                    "lo": q[:, iq[0.05]] * 100,
                    "hi": q[:, iq[0.95]] * 100,
                    "med": q[:, iq[0.5]] * 100,
                    "obs": t.obs[rows] * 100,
                }
            )
            w = inp.mode[t.prim[rows], t.origin_day[rows]]
            share = float(w.mean())
            print(
                arm,
                pl_name,
                "rows",
                len(rows),
                "share of plotted rows with the wide band on",
                round(share, 3),
                flush=True,
            )
            runs = []
            start = None
            for i, on in enumerate(w):
                if on and start is None:
                    start = i
                if (not on or i == len(w) - 1) and start is not None:
                    end = i if on else i - 1
                    runs.append({"t0": df["t"][start], "t1": df["t"][end]})
                    start = None
            strip = (
                pl.DataFrame(runs)
                if runs
                else pl.DataFrame({"t0": [df["t"][0]], "t1": [df["t"][0]]})
            )
            xdom = (int(df["t"].min().timestamp() * 1000), int(df["t"].max().timestamp() * 1000))  # ty: ignore[unresolved-attribute]
            panels_long.append(
                series_panel(
                    df=df,
                    strip=strip,
                    colour=colour,
                    label=pl_name,
                    title=f"{lab}, 9 to 10 days ahead",
                    xdom=xdom,
                    lead_note="Forecast issued 9 to 10 days before each half-hour.",
                )
            )
        out[arm] = alt.vconcat(*panels_long, spacing=14)
        save(out[arm], f"real_pair_9_to_10_days_ahead_{arm}.svg")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "bars"
    if what == "bars":
        for grp, suffix in (
            ("both groups", ""),
            ("MVA-only eight", "_mva"),
            ("signed-MW six", "_mw"),
        ):
            gn = k.GROUP_NAME.get(grp, "both metering groups")
            coverage_chart(
                metric="cover90",
                ytitle="Share of rows with the NRA inside q05-q95",
                target=0.90,
                domain=[0.5, 1.0],
                name=f"coverage_by_phase_and_lead{suffix}.svg",
                caption=f"Coverage by phase and lead, injected 15 and 30 day load-shifts, {gn}",
                group=grp,
            )
        coverage_chart(
            metric="nra_side95",
            ytitle="Share of rows with the NRA on the safe side",
            target=0.95,
            domain=[0.5, 1.0],
            name="safe_side_by_phase_and_lead.svg",
            caption="Safe-side share by phase and lead, injected 15 and 30 day load-shifts, both metering groups",
            group="both groups",
        )
        gate_value()
        event_free()
    else:
        series_figures()
