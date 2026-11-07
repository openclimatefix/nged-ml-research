# ruff: noqa: E501
"""Figures of experiment 631, written to the experiment's docs folder."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import e00631_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / c.SLUG
DKEY = "D:d622_lagged"
SOURCES = [f"{a}|{v}" for a in ("J0", "J3", "J5") for v in ("wo|none", "w|none", "w|tod")] + [DKEY]
KIND = {
    "wo|none": "Primaries only",
    "w|none": "Primaries and farms",
    "w|tod": "Farms and remainder",
}
KINDS = [*KIND.values(), "Direct forecast"]
COLOURS = [theme.DATA_SKY, theme.DATA_BLUE, theme.DATA_PURPLE, theme.BRAND_ORANGE]
BSP = {1: "Boston BSP", 2: "Skegness BSP"}


def label(src: str) -> tuple[str, str]:
    """Return the row label and colour group of a source."""
    if src == DKEY:
        return "Direct forecast", KINDS[3]
    arm, rest = src.split("|", 1)
    return arm, KIND[rest]


def exceed_figure(*, side: str, name: str, title: list[str], xtitle: str) -> None:
    """Draw 5%-tail exceedance rates with month-block intervals for the two BSPs."""
    t = pl.read_parquet(c.WORK / "scores.parquet").filter(
        (pl.col("rowset") == "common")
        & (pl.col("band") == "0-240h")
        & (pl.col("stat") == "exceed_0.05")
        & (pl.col("side") == side)
        & pl.col("source").is_in(SOURCES)
    )
    df = t.with_columns(
        pl.col("source").map_elements(lambda s: label(s)[0], return_dtype=pl.String).alias("set"),
        pl.col("source").map_elements(lambda s: label(s)[1], return_dtype=pl.String).alias("kind"),
        pl.col("group").replace_strict(BSP).alias("bsp"),
        (pl.col("value") * 100).alias("rate"),
        (pl.col("lo") * 100).alias("lo100"),
        (pl.col("hi") * 100).alias("hi100"),
    ).with_columns(
        row=pl.when(pl.col("set").str.starts_with("Direct"))
        .then(pl.col("set"))
        .otherwise(pl.col("set") + ", " + pl.col("kind").str.to_lowercase())
    )
    order = [
        (label(s)[0] if s == DKEY else label(s)[0] + ", " + label(s)[1].lower()) for s in SOURCES
    ]
    rule = (
        alt.Chart(pl.DataFrame({"x": [5.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.BLACK_1, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    base = alt.Chart().encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    col = alt.Color(
        "kind:N",
        title=None,
        scale=alt.Scale(domain=KINDS, range=COLOURS),
        legend=alt.Legend(orient="top", columns=2),
    )
    pts = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("rate:Q", title=xtitle, scale=alt.Scale(domain=[0, 60])), color=col
    )  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule().encode(x="lo100:Q", x2="hi100:Q", color=col)  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(bars, pts, rule, data=df)
        .properties(width=240, height=300)
        .facet(column=alt.Column("bsp:N", title=None))
        .properties(title=alt.TitleParams(title, anchor="start"))
    )
    chart.save(DOCS / name)


def example_figure(*, max_hours: float, file_name: str, ticks: list[int]) -> None:
    """Draw a winter and a summer run for each BSP: meter, J3 with farms and remainder, direct."""
    off = np.load(c.WORK / "offsets.npz")
    rows = []
    for day, tag in (
        ("2025-12-02", "Winter run (December 2025)"),
        ("2026-06-10", "Summer run (June 2026)"),
    ):
        z = np.load(c.WORK / "runs" / f"{day}.npz")
        init = datetime.strptime(day, "%Y-%m-%d").replace(hour=9, tzinfo=UTC)
        month = day[:7]
        hod = np.array(
            [
                ((init + (i + 1) * c.STEP).hour * 2 + (init + (i + 1) * c.STEP).minute // 30)
                for i in range(c.N_STEPS)
            ]
        )  # ty: ignore[unsupported-operator]
        hours = (np.arange(c.N_STEPS) + 1) / 2
        for gi, bname in enumerate(BSP.values()):
            s = z["sums"][2, 1, gi] + off[f"tod_{month}"][1, gi][hod][None, :]
            q = lambda k, s=s: np.quantile(s, k, axis=0, method="weibull")  # noqa: E731
            dq = z["direct"][0, gi]
            for name, lo, mid, hi in (
                ("Sum (J3), farms, remainder", q(0.05), q(0.5), q(0.95)),
                ("Direct forecast", dq[:, 1], dq[:, 5], dq[:, 9]),
                ("Meter", z["meter"][gi], z["meter"][gi], z["meter"][gi]),
            ):
                rows.append(
                    pl.DataFrame(
                        {
                            "run": tag,
                            "bsp": bname,
                            "set": name,
                            "hours": hours,
                            "lo": np.asarray(lo, float),
                            "mid": np.asarray(mid, float),
                            "hi": np.asarray(hi, float),
                        }
                    )
                )
    df = pl.concat(rows).filter(pl.col("hours") <= max_hours)
    dom = ["Sum (J3), farms, remainder", "Direct forecast", "Meter"]
    col = alt.Color(
        "set:N",
        title=None,
        scale=alt.Scale(domain=dom, range=[theme.DATA_PURPLE, theme.BRAND_ORANGE, theme.BLACK_1]),
        legend=alt.Legend(orient="top"),
    )
    band = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("hours:Q", title="Hours ahead", axis=alt.Axis(values=ticks)),
            y=alt.Y("lo:Q", title="Flow into the BSP (MW)"),
            y2="hi:Q",
            color=col,
        )
        .transform_filter(alt.datum.set != "Meter")
    )
    line = alt.Chart().mark_line(strokeWidth=1.5).encode(x="hours:Q", y="mid:Q", color=col)  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(band, line, data=df)
        .properties(width=280, height=170)
        .facet(row=alt.Row("bsp:N", title=None), column=alt.Column("run:N", title=None))
    )
    chart.save(DOCS / file_name)


def main() -> None:
    """Write the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    exceed_figure(
        side="export",
        name="figure_1_export_q05_exceedance.svg",
        title=[
            "Export side: share of actual flows below the sum's q05 (nominal 5%, dashed). Dot: estimate;",
            "line: 95% interval from resampling whole months. Leads 0 to 240 hours; closer to 5% is better.",
        ],
        xtitle="Share of half-hours beyond the q05 (%)",
    )
    exceed_figure(
        side="import",
        name="figure_2_import_q95_exceedance.svg",
        title=[
            "Import side: share of actual flows above the sum's q95 (nominal 5%, dashed). Dot: estimate;",
            "line: 95% interval from resampling whole months. Leads 0 to 240 hours; closer to 5% is better.",
        ],
        xtitle="Share of half-hours beyond the q95 (%)",
    )
    example_figure(
        max_hours=48, file_name="figure_3_example_day_ahead.svg", ticks=[0, 12, 24, 36, 48]
    )
    example_figure(
        max_hours=240, file_name="figure_4_example_ten_days.svg", ticks=[0, 48, 96, 144, 192, 240]
    )
    print("written", file=sys.stderr)


if __name__ == "__main__":
    main()
