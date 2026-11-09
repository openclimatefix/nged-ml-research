# ruff: noqa: E501
"""Experiment 655: print every number the write-up quotes and draw its figures.

Usage: python e00655_report.py
"""

import json
import os
from pathlib import Path

import altair as alt
import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00655_analyse as an
import e00655_fit as fit
import numpy as np
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / fit.NAME
W = 560
NAMES = {
    "a": "Free network", "b": "Mean-sky term (644)", "k": "Mean-sky, new physics (c0)", "c": "Member-level, air temperature (c)",
    "d": "Member-level, cell temperature (d)", "e": "(d) + curve spread input (e)", "f": "14-day tracker, observed sunniness",
    "tb": "Tracker by forecast sunniness", "tc": "Tracker, continuous weight", "tbs": "Tracker, shuffled errors",
}  # fmt: skip
COLOURS = {
    "a": "#FF4901", "b": "#306BFF", "k": "#10C5F7", "c": "#B701FF", "d": "#17E58F", "e": "#808080",
    "f": "#FFD053", "tb": "#7BCDF3", "tc": "#9CF0C5", "tbs": "#444444",
}  # fmt: skip
SEEDS = tuple(k639.SEEDS)


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write the SVG, and a PNG for review when `PNG_DIR` is set."""
    chart.save(DOCS / f"{name}.svg")
    if os.environ.get("PNG_DIR"):
        chart.save(Path(os.environ["PNG_DIR"]) / f"{name}.png", scale_factor=1.0)


def by_lead_day(*, arms: tuple[str, ...]) -> pl.DataFrame:
    """Return sunny-midday CRPS by lead day and arm, with day-block 95% intervals."""
    rows = a644.load(world="real")
    w = rows["in_window"]
    out = []
    for arm in arms:
        sc = an.arm_crps(arm=arm, rows=rows, seeds=SEEDS).mean(axis=0)
        for day in range(1, 11):
            m = rows["sunny"] & w & (rows["lead_day"] == day)
            mean, lo, hi = a639.bootstrap(diff=sc[m], day=rows["day"][m], level=0.95)
            out.append({"arm": arm, "lead_day": day, "crps": mean, "lo": lo, "hi": hi})
    return pl.DataFrame(out).with_columns(pl.col("arm").replace(NAMES).alias("name"))


def headline(*, arms: tuple[str, ...], title: str, name: str) -> None:
    """Draw CRPS by lead day for the given arms."""
    df = by_lead_day(arms=arms)
    dom = [NAMES[a] for a in arms]
    colour = alt.Color("name:N", sort=dom, scale=alt.Scale(domain=dom, range=[COLOURS[a] for a in arms]), legend=alt.Legend(title=None, orient="bottom", columns=2))  # fmt: skip
    base = alt.Chart(df).encode(
        x=alt.X(
            "lead_day:O",
            title="Lead day (from the power-forecast origin)",
            axis=alt.Axis(labelAngle=0),
        )
    )
    band = base.mark_errorband(opacity=0.15).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", title=None), y2="hi:Q", color=colour
    )  # ty: ignore[unresolved-attribute]
    line = base.mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y(
            "crps:Q",
            title="CRPS on sunny middays (points of capacity)",
            scale=alt.Scale(zero=False),
        ),
        color=colour,
    )  # ty: ignore[unresolved-attribute]
    save((band + line).properties(width=W, height=300, title=title), name)
    print(
        name,
        df.pivot(on="arm", index="lead_day", values="crps")
        .sort("lead_day")
        .with_columns(pl.exclude("lead_day").round(2)),
    )


def gain_chart() -> None:
    """Draw each arm's CRPS gain over the free network (free minus arm) by lead band, with 99.9% intervals."""
    rows = a644.load(world="real")
    masks = a644._masks(rows=rows)
    arms = ("b", "k", "c", "d", "e", "f", "tb", "tc", "tbs")
    scores = {arm: an.arm_crps(arm=arm, rows=rows, seeds=SEEDS) for arm in ("a", *arms)}
    out = []
    for band in an.BANDS:
        for arm in arms:
            r = an.contrast(
                rows=rows, control=scores["a"], arm=scores[arm], mask=masks[f"sunny_{band}"]
            )
            out.append(
                {
                    "band": f"Day {band[1:]}",
                    "arm": NAMES[arm],
                    "gain": r["diff"],
                    "lo": r["lo"],
                    "hi": r["hi"],
                }
            )
    df = pl.DataFrame(out)
    dom = [NAMES[a] for a in arms]
    colour = alt.Color("arm:N", sort=dom, scale=alt.Scale(domain=dom, range=[COLOURS[a] for a in arms]), legend=None)  # fmt: skip
    y = alt.Y("arm:N", sort=dom, title=None, axis=alt.Axis(labelLimit=260))
    rule = (
        alt.Chart(df)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(y=y, x=alt.X("lo:Q", title="CRPS gain (points)"), x2="hi:Q", color=colour)
    )  # ty: ignore[unresolved-attribute]
    point = alt.Chart(df).mark_point(filled=True, size=70).encode(y=y, x="gain:Q", color=colour)  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(df).mark_rule(color="black", strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    save(
        alt.layer(rule, point, zero)
        .properties(width=150, height=260)
        .facet(column=alt.Column("band:N", title=None))
        .properties(
            title="CRPS gain over the free network on sunny middays, 99.9% interval (positive is better)"
        ),
        "gain_over_free",
    )
    print(df.with_columns(pl.col("gain", "lo", "hi").round(3)))


def forecast_examples() -> None:
    """Draw a day-ahead and a 14-day-horizon forecast of the free network (a) and arm d for one Primary."""
    rows = a644.load(world="real")
    q = {arm: an.arm_quantiles(arm=arm, seed=0, rows=rows) for arm in ("a", "d")}
    frames = []
    for cut, season in ((2, "Summer (June 2026)"), (1, "Spring (March 2026)")):
        prim = int(np.unique(rows["primary"])[0])
        m0 = (rows["primary"] == prim) & (rows["cutoff"] == cut)
        origins = np.unique(rows["origin_us"][m0])
        origin = int(origins[len(origins) // 2])
        for arm in ("a", "d"):
            m = m0 & (rows["origin_us"] == origin)
            order = np.argsort(rows["t_us"][m])
            hours = ((rows["t_us"][m] - origin) / 3.6e9)[order]
            frames.append(
                pl.DataFrame(
                    {
                        "hour": hours,
                        **{
                            k: q[arm][m, i][order] * 100
                            for k, i in (("q05", 1), ("q25", 3), ("q50", 5), ("q75", 7), ("q95", 9))
                        },
                        "actual": rows["y"][m][order] * 100,
                        "arm": NAMES[arm],
                        "season": season,
                    }
                )
            )
    df = pl.concat(frames)
    for name, hi, title in (
        ("example_day_ahead", 24.0, "Day-ahead forecast"),
        ("example_14_day", 240.0, "Full horizon of one run"),
    ):
        d = df.filter(pl.col("hour") <= hi)
        base = alt.Chart(d).encode(x=alt.X("hour:Q", title="Hours after the forecast origin"))
        b90 = base.mark_area(opacity=0.2).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Net demand (points of capacity)"),
            y2="q95:Q",
            color=alt.Color("arm:N", legend=alt.Legend(title=None, orient="bottom")),
        )  # ty: ignore[unresolved-attribute]
        b50 = base.mark_area(opacity=0.3).encode(y="q25:Q", y2="q75:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        med = base.mark_line().encode(y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        act = base.mark_line(color="black", strokeDash=[4, 3]).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        save(
            alt.layer(b90, b50, med, act)
            .properties(width=W - 100, height=200)
            .facet(row=alt.Row("season:N", title=None))
            .properties(title=f"{title}: 50% and 90% bands, median, and metered value (dashed)"),
            name,
        )


def curve_example() -> None:
    """Draw one Primary's sunny day: the mean-sky curve, the member-mean curve, and its spread."""
    cutoff, month = "2026-06", "2026-06"
    z = np.load(fit.WORK / "curves" / f"test_curves_{cutoff}_{month}.npz")
    rows = a644.load(world="real")
    sel = rows["cutoff"] == 2
    prim = int(np.unique(z["primary"])[0])
    day = None
    t = z["t_us"]
    m = (z["primary"] == prim) & ((t // 3_600_000_000) % 24 == 12)
    _ = (rows, sel, m)
    days = np.unique(t[m] // 86_400_000_000)
    day = int(days[len(days) // 2])
    keep = (z["primary"] == prim) & (t // 86_400_000_000 == day)
    # One row per valid time: the day-1 run has the shortest lead, which the stored files do not keep, so take the first.
    order = np.argsort(t[keep], kind="stable")
    tt = t[keep][order]
    first = np.unique(tt, return_index=True)[1]
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "hour": (tt[first] % 86_400_000_000) / 3.6e9,
                    "u": z[v][keep][order][first],
                    "curve": label,
                }
            )
            for v, label in (
                ("c0", "Mean sky (c0)"),
                ("c", "Mean over members, air (c)"),
                ("d", "Mean over members, cell (d)"),
            )
        ]
    )
    chart = alt.Chart(df).mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hour of day (UTC)"), y=alt.Y("u:Q", title="Solar curve (share of the 99th percentile)"),
        color=alt.Color("curve:N", legend=alt.Legend(title=None, orient="bottom")),
    ).properties(width=W, height=240, title="One Primary, one day: the three curves")  # fmt: skip
    save(chart, "curve_example")


def main() -> None:
    """Draw every figure and print the contrasts."""
    DOCS.mkdir(parents=True, exist_ok=True)
    gain_chart()
    headline(
        arms=("a", "d", "f", "tb"), title="CRPS on sunny middays by lead day", name="crps_by_lead"
    )
    forecast_examples()
    curve_example()
    res = json.loads((fit.WORK / "analysis" / "contrasts.json").read_text())
    print(json.dumps(res["scores"], indent=1))
    for k, v in res["contrasts"].items():
        print(
            k,
            json.dumps(
                {
                    a: [round(b["diff"], 3), round(b["lo"], 3), round(b["hi"], 3)]
                    for a, b in v.items()
                }
            ),
        )
    print(json.dumps(res["tails_sunny"], indent=1))
    print(json.dumps(res["extra"]["median_minus_mean_q"]))
    for k, v in res["extra"].items():
        if k.startswith("disagree"):
            print(
                k,
                json.dumps(
                    {
                        a: [round(b["diff"], 3), round(b["lo"], 3), round(b["hi"], 3), b["n_rows"]]
                        for a, b in v.items()
                    }
                ),
            )


if __name__ == "__main__":
    main()
