# ruff: noqa: E501
"""Experiment 630, part 4: the figures of the write-up (SVG, 680 px wide)."""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00630_common import NETWORK_LABEL, NETWORKS, SLUG, TABLES, load
from plotting import ocf_theme as theme

DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
W: Final[int] = 600
NET_COLOURS: Final[dict[str, str]] = {
    NETWORK_LABEL["K1"]: theme.DATA_BLUE,
    NETWORK_LABEL["Candidate"]: theme.BRAND_ORANGE,
    NETWORK_LABEL["K3"]: theme.DATA_PURPLE,
}


def rows(name: str, network: str) -> pl.DataFrame:
    """Read one slice's exceedance table for a network."""
    return pl.read_csv(TABLES / f"exceedance_by_{name}_{network}.csv").with_columns(
        network_label=pl.lit(NETWORK_LABEL[network])
    )


def dots(
    *,
    d: pl.DataFrame,
    x: str,
    value: str,
    colour: str,
    title: str,
    nominal: float,
    scale: alt.Scale,
    sort: list[str] | None = None,
    offset: str | None = None,
    height: int = 200,
    domain: tuple[float, float] = (0, 30),
) -> alt.LayerChart:
    """Draw dots with 95% interval bars and a dashed nominal line."""
    d = d.with_columns(
        v=pl.col(value) * 100,
        lo=pl.col(f"{value}_lo") * 100,
        hi=pl.col(f"{value}_hi") * 100,
        nominal=pl.lit(nominal),
    )
    base = alt.Chart(d.to_pandas())
    enc: dict = {"x": alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0))}
    if offset:
        enc["xOffset"] = f"{offset}:N"
    col = alt.Color(f"{colour}:N", scale=scale, title=None, legend=alt.Legend(orient="top"))
    rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        **enc,
        y=alt.Y("lo:Q", title=title, scale=alt.Scale(domain=list(domain))),
        y2="hi:Q",
        color=col,
    )
    pts = base.mark_point(filled=True, size=50).encode(**enc, y="v:Q", color=col)  # ty: ignore[unresolved-attribute]
    nom = base.mark_rule(strokeDash=[4, 3], color="#555555").encode(y="nominal:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(nom, rule, pts).properties(width=W, height=height)  # ty: ignore[invalid-return-type]


def save(chart: alt.TopLevelMixin, name: str, title: str, subtitle: list[str]) -> None:
    """Write one chart with a wrapped title and subtitle."""
    DOCS.mkdir(parents=True, exist_ok=True)
    wrapped_title = textwrap.wrap(title, 62)
    wrapped_subtitle = textwrap.wrap(" ".join(subtitle), 92)
    chart.properties(
        title=alt.TitleParams(wrapped_title, subtitle=wrapped_subtitle, anchor="start")
    ).save(DOCS / name)


def fig_months() -> None:
    """Figure 1: the Candidate's exceedance by month, event-free rows against event rows."""
    d = rows("month", "Candidate").filter(pl.col("subset") != "all")
    d = d.with_columns(
        rowset=pl.when(pl.col("subset") == "event-free")
        .then(pl.lit("Event-free rows"))
        .otherwise(pl.lit("Event rows"))
    )
    scale = alt.Scale(
        domain=["Event-free rows", "Event rows"], range=[theme.BRAND_ORANGE, "#FFB599"]
    )
    months = ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]
    p90 = dots(
        d=d,
        x="month",
        value="a90",
        colour="rowset",
        title="Above q90 (%)",
        nominal=10,
        scale=scale,
        sort=months,
        offset="rowset",
        domain=(0, 32),
    )
    p95 = dots(
        d=d,
        x="month",
        value="a95",
        colour="rowset",
        title="Above q95 (%)",
        nominal=5,
        scale=scale,
        sort=months,
        offset="rowset",
        domain=(0, 32),
    )
    save(
        alt.vconcat(p90, p95),
        "figure_1_candidate_exceedance_by_month.svg",
        "The Candidate's upper tail is too low in four of five months, on ordinary days too",
        [
            "Share of observed power above the forecast q90 (top) and q95 (bottom), by month of forecast origin, class-2 Candidate,",
            "16 Primaries. Dot: estimate. Bar: 95% interval from resampling origin days. Dashed line: nominal level (10% and 5%).",
        ],
    )


def fig_slices() -> None:
    """Figure 2: event-free exceedance by lead band and for cold nights."""
    scale = alt.Scale(domain=list(NET_COLOURS), range=list(NET_COLOURS.values()))
    band = pl.concat([rows("band", n) for n in NETWORKS]).filter(pl.col("subset") == "event-free")
    tercile = (
        pl.concat([rows("cold_night", n) for n in NETWORKS])
        .filter(pl.col("subset") == "event-free")
        .with_columns(
            kind=pl.when(pl.col("cold_night"))
            .then(pl.lit("Cold nights"))
            .otherwise(pl.lit("All other hours"))
        )
    )
    p1 = dots(
        d=band,
        x="band",
        value="a90",
        colour="network_label",
        title="Above q90 (%)",
        nominal=10,
        scale=scale,
        sort=["0-24 h", "24-72 h", "72-120 h", "120-240 h"],
        offset="network_label",
        domain=(0, 28),
    )
    p2 = dots(
        d=tercile,
        x="kind",
        value="a90",
        colour="network_label",
        title="Above q90 (%)",
        nominal=10,
        scale=scale,
        sort=["All other hours", "Cold nights"],
        offset="network_label",
        domain=(0, 28),
    )
    save(
        alt.vconcat(p1, p2),
        "figure_2_event_free_exceedance_by_lead_and_cold_night.svg",
        "On ordinary days the excess is larger at long leads and on cold nights",
        [
            "Share of event-free observed power above the forecast q90, by lead band (top) and for cold nights (18:00 to 06:00 UTC,",
            "lowest third of ensemble temperature) against all other hours (bottom). Dot: estimate. Bar: 95% day-resampling interval. Dashed line: 10%.",
        ],
    )


def fig_primaries() -> None:
    """Figure 3: the Candidate's event-free q90 exceedance by Primary."""
    d = (
        rows("primary", "Candidate")
        .filter(pl.col("subset") == "event-free")
        .with_columns(what=pl.lit("Candidate, event-free rows"))
    )
    order = d.sort("a90")["primary_name"].to_list()
    scale = alt.Scale(domain=["Candidate, event-free rows"], range=[theme.BRAND_ORANGE])
    p = dots(
        d=d,
        x="primary_name",
        value="a90",
        colour="what",
        title="Above q90 (%)",
        nominal=10,
        scale=scale,
        sort=order,
        height=180,
        domain=(0, 30),
    )
    save(
        p,
        "figure_3_candidate_exceedance_by_primary.svg",
        "Nine of the 16 Primaries have an event-free q90 exceedance clearly above 10%",
        [
            "Share of event-free observed power above the Candidate's q90, by Primary, sorted by share. D15 and D18 are the two report-only",
            "Primaries with short histories. Dot: estimate. Bar: 95% day-resampling interval. Dashed line: 10%.",
        ],
    )


def fig_correction() -> None:
    """Figure 4: the Candidate's exceedance before and after each correction."""
    c = (
        pl.read_csv(TABLES / "correction_Candidate.csv")
        .filter((pl.col("slice") == "overall") & (pl.col("subset") == "event-free"))
        .row(0, named=True)
    )
    names = {"base": "Uncorrected", "S": "Shift (S)", "R": "Rescale (R)"}
    recs = []
    for arm, label in names.items():
        for lvl, nominal in (("a90", 10.0), ("a95", 5.0)):
            recs.append(
                {
                    "arm": label,
                    "level": lvl,
                    "v": c[f"{lvl}_{arm}"],
                    "lo": c[f"{lvl}_{arm}_lo"],
                    "hi": c[f"{lvl}_{arm}_hi"],
                    "nominal": nominal,
                }
            )
    d = pl.DataFrame(recs)
    scale = alt.Scale(domain=list(names.values()), range=[theme.BRAND_ORANGE] * 3)
    panels = []
    for lvl, title, nominal, dom in (
        ("a90", "Above q90 (%)", 10, (0, 20)),
        ("a95", "Above q95 (%)", 5, (0, 14)),
    ):
        g = d.filter(pl.col("level") == lvl).with_columns(
            v=pl.col("v") * 100, lo=pl.col("lo") * 100, hi=pl.col("hi") * 100
        )
        base = alt.Chart(g.to_pandas())
        col = alt.Color("arm:N", scale=scale, title=None, legend=None)
        x = alt.X("arm:N", sort=list(names.values()), title=None, axis=alt.Axis(labelAngle=0))
        panels.append(
            alt.layer(
                base.mark_rule(strokeDash=[4, 3], color="#555555").encode(y=alt.datum(nominal)),  # ty: ignore[unresolved-attribute]
                base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                    x=x,
                    y=alt.Y("lo:Q", title=title, scale=alt.Scale(domain=list(dom))),
                    y2="hi:Q",
                    color=col,
                ),
                base.mark_point(filled=True, size=60).encode(x=x, y="v:Q", color=col),  # ty: ignore[unresolved-attribute]
            ).properties(width=W, height=200)
        )
    save(
        alt.vconcat(*panels),
        "figure_4_correction_out_of_sample.svg",
        "A shift or rescale fitted on three months removes only part of the excess",
        [
            "Candidate, event-free rows of 2026-03 and 2026-06, with the correction fitted per Primary and lead band on 2025-10, 2025-12, and",
            "2026-01. Dot: estimate. Bar: 95% day-resampling interval. Dashed line: nominal level (10% and 5%).",
        ],
    )


def fig_example() -> None:
    """Figure 5: one Candidate run at one Primary."""
    df = load(network="Candidate").filter(
        (pl.col("primary_name") == "D2") & (pl.col("month") == "2026-01")
    )
    origin = df.filter(pl.col("power_fcst_init_time").dt.day() == 13)["power_fcst_init_time"].min()
    run = (
        df.filter(pl.col("power_fcst_init_time") == origin)
        .sort("valid_time")
        .with_columns(lead_days=pl.col("lead_hours") / 24)
    )
    panels = []
    for title, hi in (("First two days of the run", 2.0), ("Full 10-day run", 10.0)):
        r = (
            run.filter(pl.col("lead_days") <= hi)
            .select("lead_days", "q05", "q25", "q50", "q75", "q95", "q90", "observed_power")
            .to_pandas()
        )
        b = alt.Chart(r).encode(
            x=alt.X(
                "lead_days:Q",
                title="Days since the forecast was issued",
                axis=alt.Axis(values=list(range(11)) if hi > 2 else [0, 0.5, 1, 1.5, 2]),
            )
        )
        band90 = b.mark_area(opacity=0.2, color=theme.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power (MVA)"), y2="q95:Q"
        )
        band50 = b.mark_area(opacity=0.35, color=theme.BRAND_ORANGE).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        med = b.mark_line(color=theme.BRAND_ORANGE).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        obs = b.mark_line(color="#1A1A1A").encode(y="observed_power:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(band90, band50, med, obs).properties(width=W, height=200, title=title)
        )
    save(
        alt.vconcat(*panels),
        "figure_5_example_forecast.svg",
        "Example Candidate forecast at one Primary, January 2026",
        [
            "Black: observed power. Orange line: median. Dark shade: q25 to q75. Pale shade: q05 to q95. Primary D2, one run, event-free."
        ],
    )


if __name__ == "__main__":
    fig_months()
    fig_slices()
    fig_primaries()
    fig_correction()
    fig_example()
