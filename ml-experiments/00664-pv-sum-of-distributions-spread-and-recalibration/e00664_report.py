# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 664: draw the figures and print the numbers the write-up quotes.

Usage: python e00664_report.py
"""

import json
import os
from pathlib import Path

import altair as alt
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00664_analyse as an
import numpy as np
import polars as pl

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / "00664-pv-sum-of-distributions-spread-and-recalibration"
)
W = 560
NAMES = {
    "a": "Free network (a)", "b": "Free network, recalibrated (b)", "c": "Sum of distributions (c)", "cc": "Sum, coupled draws (c, coupled)",
    "d": "Sum, recalibrated (d)", "dcc": "Coupled sum, recalibrated", "p": "Placebo: other Primary's capacity", "s": "Placebo: other date's curves", "c0": "Gross draw minus mean curve",
}  # fmt: skip
COLOURS = {
    "a": "#FF4901", "b": "#306BFF", "c": "#B701FF", "cc": "#7BCDF3", "d": "#17E58F", "dcc": "#FFD053", "p": "#808080", "s": "#444444", "c0": "#10C5F7",
}  # fmt: skip


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write the SVG, and a PNG for review when `PNG_DIR` is set."""
    chart.save(DOCS / f"{name}.svg")
    if os.environ.get("PNG_DIR"):
        chart.save(Path(os.environ["PNG_DIR"]) / f"{name}.png", scale_factor=1.0)


def main() -> None:
    """Draw the figures."""
    res = json.loads((an.OUT / "contrasts.json").read_text())
    DOCS.mkdir(parents=True, exist_ok=True)
    # 1. Gain over the free network by lead band, with intervals and bars.
    arms = ("b", "c", "d", "cc", "dcc")
    recs = []
    for band in an.BANDS:
        for arm in arms:
            v = res["contrasts"][f"sunny_{band}"][f"a-{arm}"]
            recs.append(
                {
                    "band": f"Day {band[1:]}",
                    "arm": NAMES[arm],
                    "gain": v["diff"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                    "bar": an.BARS[band],
                }
            )
    df = pl.DataFrame(recs)
    dom = [NAMES[a] for a in arms]
    colour = alt.Color("arm:N", sort=dom, scale=alt.Scale(domain=dom, range=[COLOURS[a] for a in arms]), legend=alt.Legend(title=None, orient="bottom", columns=2))  # fmt: skip
    base = alt.Chart(df).encode(
        y=alt.Y("band:N", title=None, sort=["Day 1-2", "Day 3-5", "Day 6-10"]),
        yOffset=alt.YOffset("arm:N", sort=dom),
    )
    bars = base.mark_bar(height=11).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "gain:Q",
            title="CRPS gain over the free network, sunny middays (points of capacity; positive = better)",
        ),
        color=colour,
    )  # ty: ignore[unresolved-attribute]
    err = base.mark_rule(color="black").encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[2, 2])
        .encode(x="x:Q")
    )
    save(
        (bars + err + zero).properties(
            width=W,
            height=300,
            title="No arm beats the free network; every sum-of-distributions arm is worse",
        ),
        "gain_over_free",
    )
    # 2. PIT histograms of a, b, c on sunny middays at day 1-2 and over all sunny middays.
    recs = []
    for sname, label in (
        ("sunny_d1-2", "Sunny middays, day 1-2"),
        ("sunny_all", "Sunny middays, all leads"),
    ):
        for arm in ("a", "b", "c"):
            for i, h in enumerate(res["pit"][sname][arm]["hist"]):
                recs.append(
                    {"panel": label, "arm": NAMES[arm], "bin": (i + 0.5) / an.PIT_BINS, "share": h}
                )
    ph = pl.DataFrame(recs)
    ref = (
        alt.Chart(pl.DataFrame({"y": [1.0 / an.PIT_BINS]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[3, 3])
        .encode(y="y:Q")
    )
    rows_of_panels = []
    for pi, panel in enumerate(("Sunny middays, day 1-2", "Sunny middays, all leads")):
        cells = []
        for arm in ("a", "b", "c"):
            d = ph.filter((pl.col("panel") == panel) & (pl.col("arm") == NAMES[arm]))
            bar = (
                alt.Chart(d)  # ty: ignore[unresolved-attribute]
                .mark_bar(color=COLOURS[arm])
                .encode(
                    x=alt.X(
                        "bin:Q",
                        title="PIT of the realised power" if pi == 1 else None,
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y(
                        "share:Q",
                        title="Share of half-hours" if arm == "a" else None,
                        scale=alt.Scale(domain=[0, 0.3]),
                    ),
                )
            )  # ty: ignore[unresolved-attribute]
            cells.append(
                (bar + ref).properties(width=170, height=150, title=NAMES[arm] if pi == 0 else "")
            )
        rows_of_panels.append(alt.hconcat(*cells).properties(title=panel))
    save(alt.vconcat(*rows_of_panels), "pit_histograms")
    # 3. CRPS by lead day, all arms that matter.
    rows = a644.load(world="real")
    del rows
    by_day = res["by_lead_day"]
    recs = [
        {"lead_day": int(d), "arm": NAMES[a], "crps": v[a]}
        for d, v in by_day.items()
        for a in ("a", "b", "c", "d")
    ]
    dd = pl.DataFrame(recs)
    dom = [NAMES[a] for a in ("a", "b", "c", "d")]
    line = (
        alt.Chart(dd)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "lead_day:O",
                title="Lead day (from the power-forecast origin)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "crps:Q",
                title="CRPS on sunny middays (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                sort=dom,
                scale=alt.Scale(domain=dom, range=[COLOURS[a] for a in ("a", "b", "c", "d")]),
                legend=alt.Legend(title=None, orient="bottom", columns=2),
            ),
        )
        .properties(width=W, height=300, title="CRPS by lead day, sunny middays")
    )  # ty: ignore[unresolved-attribute]
    save(line, "crps_by_lead_day")
    # 4. Example forecasts: one Primary, one run, arms a and c (seed 0): the first 36 hours and the whole 240 hours.
    rows = a644.load(world="real")
    q_a = an.load_arm(arm="a", kind="q", seed=0, rows=rows)
    q_c = an.load_arm(arm="c", kind="q", seed=0, rows=rows)
    sites = {v: k for k, v in k634.six_indices().items()}
    p0 = int(np.unique(rows["primary"])[0])
    sel = (rows["primary"] == p0) & (rows["cutoff"] == 2) & (rows["lead"] < 240)
    origins = np.unique(rows["origin_us"][sel])
    pick = None
    for o in origins:
        m = sel & (rows["origin_us"] == o)
        if (
            m.sum() >= 400
            and (rows["kt_forecast"][m & rows["midday"] & (rows["lead"] < 36)] >= 0.7).mean() > 0.5
        ):
            pick = o
            break
    assert pick is not None
    m = sel & (rows["origin_us"] == pick)
    t = np.argsort(rows["t_us"][m])
    recs = []
    for arm, q in (("a", q_a), ("c", q_c)):
        qq = q[m][t]
        for i in range(m.sum()):
            recs.append({"hours": float(rows["lead"][m][t][i]), "arm": NAMES[arm], "median": 100 * float(qq[i, 5]), "lo90": 100 * float(qq[i, 1]), "hi90": 100 * float(qq[i, 9]), "lo50": 100 * float(qq[i, 3]), "hi50": 100 * float(qq[i, 7]), "actual": 100 * float(rows["y"][m][t][i])})  # fmt: skip
    ex = pl.DataFrame(recs)
    panels = []
    for title, hi in (("First 36 hours", 36.0), ("Whole 240 hours", 240.0)):
        d = ex.filter(pl.col("hours") <= hi)
        b = alt.Chart(d).encode(x=alt.X("hours:Q", title="Lead (hours)"))
        layer = (
            b.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("lo90:Q", title="Net demand (% of capacity)"), y2="hi90:Q"
            )  # ty: ignore[unresolved-attribute]
            + b.mark_area(opacity=0.3).encode(y="lo50:Q", y2="hi50:Q")  # ty: ignore[unresolved-attribute]
            + b.mark_line().encode(y="median:Q")  # ty: ignore[unresolved-attribute]
            + b.mark_line(color="black", strokeDash=[3, 2]).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        ).properties(width=300 if hi < 100 else 420, height=170, title=title)
        panels.append(layer.facet(row=alt.Row("arm:N", title=None)))
    save(
        alt.hconcat(*panels).properties(
            title=f"One forecast run at {sites[p0]}: median, 50% and 90% bands; dashed line is what happened"
        ),
        "example_forecasts",
    )
    # Print the numbers.
    print("scores", json.dumps(res["scores"], indent=0)[:1500])
    print("coverage sunny d1-2/d3-5/d6-10/all:")
    for s in ("sunny_d1-2", "sunny_d3-5", "sunny_d6-10", "sunny_all", "all"):
        print(
            s,
            {
                a: (round(v["coverage_5_95"], 3), round(v["mean_width_5_95"], 2))
                for a, v in res["coverage"][s].items()
            },
        )
        print(s, "pit dist", {a: round(v["distance"], 3) for a, v in res["pit"][s].items()})
        print(
            s,
            "tails",
            {
                a: {k: round(x, 3) for k, x in v.items()}
                for a, v in res["tails"][s].items()
                if a in ("a", "b", "c", "d")
            },
        )
    v = res["val"]
    for k in (
        "corr_pv_mean_vs_gross_pit_sunny",
        "corr_pv_member_spread_vs_abs_gross_pit_sunny",
        "corr_pv_mean_vs_net_pit_a_sunny",
        "pit_gross_mean_sunny",
        "max_abs_q_a_vs_stored",
        "median_abs_q_c0_minus_stored_d",
    ):
        xs = [r[k] for r in v if r[k] is not None]
        print(
            k,
            round(float(np.mean(xs)), 4),
            round(float(np.min(xs)), 4),
            round(float(np.max(xs)), 4),
        )
    print(
        "fallback cells",
        sum(len(r["fallback"]) for r in v),
        "min cell rows",
        min(min(r["cells"].values()) for r in v),
    )
    print("by primary", res["by_primary"])
    print(
        "mae",
        res["mae_median_sunny"],
        "med-minus-qmean",
        res["median_minus_quantile_mean"],
        "n",
        res["n_rows"],
        res["share_without_realised_kt"],
    )
    print("all-hh contrasts", res["contrasts"]["all_half_hours"])


if __name__ == "__main__":
    main()
