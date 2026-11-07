# ruff: noqa: E501, B007
"""Figures of experiment 629, written to the experiment's docs folder."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00629_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / c.SLUG
NET = "K1"
DKEY = "D:" + c.HEADLINE_DIRECT[NET]
ORDER = ["J0", "J2", "J3", "J5", "J0+g", "J2+g", "J3+g", "J5+g", DKEY]
LABEL = {DKEY: "Direct BSP forecast"} | {s: s for s in ORDER[:-1]}
KIND = {
    s: (
        "Direct forecast"
        if s == DKEY
        else "Sum plus gap correction"
        if "+g" in s
        else "Sum of Primaries, as is"
    )
    for s in ORDER
}


def exceed_figure(*, side: str, name: str, title: str) -> None:
    """Draw q95 exceedance rates with month-block intervals for the two BSPs."""
    t = pl.read_parquet(c.WORK / f"scores_{NET}.parquet").filter(
        (pl.col("band") == "0-240h")
        & (pl.col("stat") == "exceed_0.95")
        & (pl.col("side") == side)
        & pl.col("group").is_in([1, 2])
        & pl.col("source").is_in(ORDER)
    )
    df = t.with_columns(
        pl.col("source").replace_strict(LABEL).alias("set"),
        pl.col("source").replace_strict(KIND).alias("kind"),
        pl.col("group").replace_strict({1: "Boston BSP", 2: "Skegness BSP"}).alias("bsp"),
        (pl.col("value") * 100).alias("rate"),
        (pl.col("lo") * 100).alias("lo100"),
        (pl.col("hi") * 100).alias("hi100"),
    )
    rule = alt.Chart().mark_rule(color=theme.GREY_1, strokeDash=[4, 3]).encode(x=alt.datum(5.0))  # ty: ignore[unresolved-attribute]
    base = alt.Chart().encode(y=alt.Y("set:N", sort=[LABEL[s] for s in ORDER], title=None))
    pts = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "rate:Q",
            title="Share of half-hours beyond the q95 (%); nominal 5%",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "kind:N",
            title=None,
            scale=alt.Scale(
                domain=["Sum of Primaries, as is", "Sum plus gap correction", "Direct forecast"],
                range=[theme.DATA_BLUE, theme.DATA_GREEN, theme.BRAND_ORANGE],
            ),
        ),
    )
    bars = base.mark_rule().encode(x="lo100:Q", x2="hi100:Q", color="kind:N")  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, pts, rule, data=df)
        .properties(width=260, height=190)
        .facet(column=alt.Column("bsp:N", title=None))
        .properties(title=title)
    )
    chart.save(DOCS / name)


def example_figure() -> None:
    """Draw one winter and one summer run for each BSP: meter, sum of Primaries (J3), gap-corrected sum, direct."""
    rows = []
    for day, tag in (
        ("2025-12-02", "Winter run, 2 December 2025"),
        ("2026-06-10", "Summer run, 10 June 2026"),
    ):
        z = np.load(c.WORK / "runs" / NET / f"{day}.npz")
        init = datetime.strptime(day, "%Y-%m-%d").replace(hour=9, tzinfo=UTC)
        rr = np.load(c.WORK / f"rows_{NET}.npz")
        sel = rr["init"] == np.datetime64(init.replace(tzinfo=None), "m")
        shift = rr["shift"][sel]  # (valid rows, 4)
        n = shift.shape[0]
        for gi, (g, bname) in enumerate(((1, "Boston BSP"), (2, "Skegness BSP"))):
            q = lambda a, k: np.quantile(a, k, axis=0, method="weibull")  # noqa: E731
            s3 = z["sums"][2, gi][:, :n]
            dq = z["direct"][0, gi][:n]
            hours = (np.arange(n) + 1) / 2
            for name, lo, mid, hi in (
                ("Sum of Primaries (J3), as is", q(s3, 0.05), q(s3, 0.5), q(s3, 0.95)),
                (
                    "Sum plus gap correction (J3+g)",
                    q(s3, 0.05) + shift[:, gi],
                    q(s3, 0.5) + shift[:, gi],
                    q(s3, 0.95) + shift[:, gi],
                ),
                ("Direct BSP forecast", dq[:, 1], dq[:, 5], dq[:, 9]),
            ):
                rows.append(
                    {
                        "run": tag,
                        "bsp": bname,
                        "set": name,
                        "hours": hours,
                        "lo": lo,
                        "mid": mid,
                        "hi": hi,
                    }
                )
            rows.append(
                {
                    "run": tag,
                    "bsp": bname,
                    "set": "Meter",
                    "hours": hours,
                    "lo": z["meter"][gi][:n],
                    "mid": z["meter"][gi][:n],
                    "hi": z["meter"][gi][:n],
                }
            )
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "run": [r["run"]] * len(r["hours"]),
                    "bsp": [r["bsp"]] * len(r["hours"]),
                    "set": [r["set"]] * len(r["hours"]),
                    "hours": r["hours"],
                    "lo": np.asarray(r["lo"], dtype=float),
                    "mid": np.asarray(r["mid"], dtype=float),
                    "hi": np.asarray(r["hi"], dtype=float),
                }
            )
            for r in rows
        ]
    )
    dom = [
        "Sum of Primaries (J3), as is",
        "Sum plus gap correction (J3+g)",
        "Direct BSP forecast",
        "Meter",
    ]
    col = alt.Color(
        "set:N",
        title=None,
        scale=alt.Scale(
            domain=dom, range=[theme.DATA_BLUE, theme.DATA_GREEN, theme.BRAND_ORANGE, theme.BLACK_1]
        ),
    )
    band = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("hours:Q", title="Hours ahead"),
            y=alt.Y("lo:Q", title="Flow into the BSP (MW)"),
            y2="hi:Q",
            color=col,
        )
        .transform_filter(alt.datum.set != "Meter")
    )
    line = alt.Chart().mark_line(strokeWidth=1.5).encode(x="hours:Q", y="mid:Q", color=col)  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(band, line, data=df)
        .properties(width=300, height=170)
        .facet(row=alt.Row("bsp:N", title=None), column=alt.Column("run:N", title=None))
    )
    chart.save(DOCS / "figure_3_example_runs.svg")


def main() -> None:
    """Write the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    exceed_figure(
        side="export",
        name="figure_1_export_q95_exceedance.svg",
        title="Export side: share of actual flows beyond the q95, days 1 to 10, K1 demand",
    )
    exceed_figure(
        side="import",
        name="figure_2_import_q95_exceedance.svg",
        title="Import side: share of actual flows beyond the q95, days 1 to 10, K1 demand",
    )
    example_figure()
    print("written", file=sys.stderr)


if __name__ == "__main__":
    main()
