# ruff: noqa: E501
"""Experiment 639: the figures of the write-up, from the stored forecasts and `work/analysis`.

Usage: python e00639_report.py
"""

import json
import os
from pathlib import Path

import altair as alt
import e00639_analyse as a
import e00639_common as k
import numpy as np
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / k.NAME
W = 560
NAMES = {
    "free": "Free irradiance network", "cal": "Explicit, calibrated probe", "raw": "Explicit, raw probe",
    "r4": "Explicit, fixed table", "oracle": "Explicit, exact step", "perm": "Placebo: capacities permuted",
    "shuf": "Placebo: shuffled-day sky",
}  # fmt: skip
GROUP = {
    "free": "Free network", "cal": "Explicit, set by probe", "raw": "Explicit, set by probe",
    "r4": "Explicit, reference capacity", "oracle": "Explicit, reference capacity",
    "perm": "Placebo", "shuf": "Placebo",
}  # fmt: skip
COLOURS = {
    "Free network": "#FF4901", "Explicit, set by probe": "#306BFF",
    "Explicit, reference capacity": "#17E58F", "Placebo": "#B701FF",
}  # fmt: skip
WORLD_TITLE = {"step": "Step world (4 MW added from April 2023)", "real": "Real development months"}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write the SVG, and a PNG for review when `PNG_DIR` is set."""
    chart.save(DOCS / f"{name}.svg")
    if os.environ.get("PNG_DIR"):
        chart.save(Path(os.environ["PNG_DIR"]) / f"{name}.png", scale_factor=1.0)


def scores() -> pl.DataFrame:
    """Return the sunny-midday CRPS of every arm and seed, in points."""
    records = []
    for world in k.WORLDS:
        rows = a.load(world=world)
        for arm in k.ARMS_BY_WORLD[world]:
            s = a.arm_crps(world=world, arm=arm, rows=rows)
            records.extend(
                {
                    "world": world,
                    "arm": NAMES[arm],
                    "group": GROUP[arm],
                    "seed": i,
                    "crps": float(s[i][rows["sunny"]].mean()),
                }
                for i in range(len(k.SEEDS))
            )
    return pl.DataFrame(records)


def crps_figure() -> None:
    """Sunny-midday CRPS by arm, one panel per world."""
    df = (
        scores()
        .group_by("world", "arm", "group")
        .agg(mean=pl.col("crps").mean(), lo=pl.col("crps").min(), hi=pl.col("crps").max())
    )
    order = [NAMES[x] for x in NAMES]
    panels = []
    for world in ("step", "real"):
        d = df.filter(pl.col("world") == world)
        base = alt.Chart(d.to_pandas()).encode(
            y=alt.Y("arm:N", sort=order, title=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        chart = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="CRPS on sunny middays (points of capacity; smaller is better)"),
            x2="hi:Q",
        ) + base.mark_circle(size=70).encode(x="mean:Q")  # ty: ignore[unresolved-attribute]
        panels.append(chart.properties(width=W, height=170, title=WORLD_TITLE[world]))
    fig = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                [
                    "Sunny-middays forecast error by arm. Dot: mean of 5 seeds. Line: range over seeds.",
                    "Six MW Primaries, 60 days after each of 3 cutoffs; recorded weather.",
                ],
                fontSize=12,
                anchor="start",
            )
        )
    )
    save(fig, "crps_by_arm")


def contrast_figure() -> None:
    """Each arm minus the calibrated explicit arm on sunny middays, with the Bonferroni interval."""
    res = [
        r
        for r in json.loads((k.WORK / "analysis/contrasts.json").read_text())
        if r["subset"] == "sunny" and r["a"] != "shuf"
    ]
    df = pl.DataFrame([{**r, "arm": NAMES[r["a"]], "group": GROUP[r["a"]]} for r in res])
    mde = json.loads((k.WORK / "analysis/mde.json").read_text())
    order = [NAMES[x] for x in NAMES if x not in {"cal", "shuf"}]
    panels = []
    for world in ("step", "real"):
        d = df.filter(pl.col("world") == world).to_pandas()
        base = alt.Chart(d).encode(
            y=alt.Y("arm:N", sort=order, title=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#000000")
            .encode(x="x:Q")
        )
        band = (
            alt.Chart(pl.DataFrame({"x": [mde[world]]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(x="x:Q")
        )
        chart = (
            zero
            + band
            + base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "lo:Q",
                    title="Arm minus calibrated explicit arm (CRPS points; positive: explicit arm is better)",
                ),
                x2="hi:Q",
            )
            + base.mark_circle(size=70).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(chart.properties(width=W, height=150, title=WORLD_TITLE[world]))
    fig = (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                [
                    "Dot: difference. Line: 99% day-block interval (Bonferroni over 4 contrasts).",
                    "Black rule: no difference. Dashed rule: minimum detectable effect. Shuffled-sky placebo (+8.8 and +3.3) is off the scale.",
                ],
                fontSize=12,
                anchor="start",
            )
        )
    )
    save(fig, "contrasts_vs_calibrated")


def example_figure() -> None:
    """One Primary, one week: actual demand with the free and calibrated medians and 5-95% bands."""
    cutoff, world, seed = "2026-03", "real", 0
    z = np.load(k.test_rows_path(cutoff=cutoff, world=world))
    six = z["inc"] > 0
    prim = z["primary"][six]
    pick = int(np.unique(prim)[0])
    sel = prim == pick
    t = z["t_us"][six][sel]
    order = np.argsort(t)
    day = (t[order] - t[order].min()) / 86_400_000_000
    first = int(np.floor(day.min())) + 21
    keep = (day >= first) & (day < first + 7)
    frames = []
    for arm in ("free", "cal"):
        q = np.load(k.prediction_path(cutoff=cutoff, world=world, arm=arm, seed=seed))["q"][six][
            sel
        ][order]
        frames.append(
            pl.DataFrame(
                {
                    "day": (day[keep] - first + 1).tolist(),
                    "arm": NAMES[arm],
                    "q05": (100 * q[keep, 1]).tolist(),
                    "q50": (100 * q[keep, 5]).tolist(),
                    "q95": (100 * q[keep, 9]).tolist(),
                    "actual": (100 * z["y"][six][sel][order][keep]).tolist(),
                }
            )
        )
    d = pl.concat(frames).to_pandas()
    base = alt.Chart(d).encode(
        x=alt.X("day:Q", title="Day of the example week (1 to 7), daylight half-hours only")
    )
    band = base.mark_area(opacity=0.25, color="#306BFF").encode(y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
    med = base.mark_line(color="#306BFF").encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q50:Q", title="Net demand (% of capacity)")
    )
    act = base.mark_line(color="#FF4901", strokeDash=[2, 2]).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        (band + med + act)
        .properties(width=W, height=150)
        .facet(row=alt.Row("arm:N", title=None, sort=[NAMES["free"], NAMES["cal"]]))
        .properties(
            title=alt.TitleParams(
                [
                    "Orange dashes: metered demand. Blue line: median forecast; band: 5% to 95%.",
                    "One Primary, one week in March to April 2026, seed 0, recorded weather.",
                ],
                fontSize=12,
                anchor="start",
            )
        )
    )
    save(chart, "example_week")


def main() -> None:
    """Write the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    crps_figure()
    contrast_figure()
    example_figure()


if __name__ == "__main__":
    main()
