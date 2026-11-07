# ruff: noqa: E501, D103, ANN001
"""Experiment 620 figures (CPU): read the score tables and the export; write SVGs for the page.

Usage: python e00620_figures.py SCORE_DIR EXPORT_DIR MONTH OUT_DIR
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

SYSTEMS = ["K1", "Candidate", "K3", "Manual heuristic", "Climatology"]
COLOUR = {
    "K1": theme.DATA_BLUE,
    "Candidate": theme.DATA_GREEN,
    "K3": theme.DATA_PURPLE,
    "Manual heuristic": theme.BRAND_ORANGE,
    "Climatology": "#7a7a7a",
}
COLOUR_SCALE = alt.Scale(domain=SYSTEMS, range=[COLOUR[s] for s in SYSTEMS])
LEGEND = alt.Legend(title=None, orient="bottom")
W = 640


def save(chart, out: Path, name: str) -> None:
    chart.save(str(out / name))


def lead_chart(
    *,
    d: pl.DataFrame,
    column: str,
    title: str,
    y_title: str,
    rule: float | None = None,
    zoom: bool = False,
) -> alt.Chart:
    if zoom:
        d = d.filter(pl.col("system") != "Climatology")
    d = d.with_columns(hours=pl.col("lead_band") * 24 + 12)
    lines = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "hours:Q",
                title="Forecast lead time (hours ahead of the forecast origin)",
                scale=alt.Scale(domain=[0, 240]),
            ),
            y=alt.Y(
                f"{column}:Q",
                title=y_title,
                scale=alt.Scale(
                    zero=not zoom, domain=[0.6, 1.0] if column == "cover90" else alt.Undefined
                ),
            ),
            color=alt.Color(
                "system:N",
                scale=alt.Scale(domain=SYSTEMS[:-1], range=[COLOUR[x] for x in SYSTEMS[:-1]])
                if zoom
                else COLOUR_SCALE,
                legend=LEGEND,
            ),
        )
    )
    layers = [lines]
    if rule is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"y": [rule]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color=theme.BLACK_2)
            .encode(y="y:Q")
        )
    return alt.layer(*layers).properties(width=W, height=320, title=title)  # ty: ignore[invalid-return-type]


def contrast_chart(*, d: pl.DataFrame, measure: str) -> alt.Chart:
    d = d.filter(pl.col("measure") == measure).with_columns(
        row=pl.col("a") + " minus " + pl.col("b")
    )
    order = d["row"].to_list()
    base = alt.Chart(d).encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=theme.BRAND_ORANGE)
        .encode(x="x:Q")
    )
    bars = base.mark_rule(color=theme.DATA_BLUE, strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo_bonferroni:Q",
            title=f"Difference in {measure.upper()} (percentage points of capacity; negative favours the first system)",
        ),
        x2="hi_bonferroni:Q",
    )
    pts = base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(x="diff_a_minus_b:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, pts).properties(width=W - 200, height=24 * len(order) + 40)  # ty: ignore[invalid-return-type]


def main(score: Path, export: Path, month: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lead = pl.read_csv(score / "by_lead_band.csv")
    save(
        lead_chart(
            d=lead,
            column="mae",
            title="Median forecast error by lead time",
            y_title="Mean absolute error of the median (% of capacity; lower is better)",
        ),
        out,
        "mae_vs_lead.svg",
    )
    save(
        lead_chart(
            d=lead,
            column="mae",
            title="Median forecast error by lead time (climatology left out, axis not from zero)",
            y_title="Mean absolute error of the median (% of capacity; lower is better)",
            zoom=True,
        ),
        out,
        "mae_vs_lead_zoom.svg",
    )
    save(
        lead_chart(
            d=lead,
            column="crps",
            title="CRPS by lead time (climatology left out, axis not from zero)",
            y_title="CRPS (% of capacity; lower is better)",
            zoom=True,
        ),
        out,
        "crps_vs_lead_zoom.svg",
    )
    save(
        lead_chart(
            d=lead,
            column="crps",
            title="Continuous ranked probability score by lead time",
            y_title="CRPS (% of capacity; lower is better)",
        ),
        out,
        "crps_vs_lead.svg",
    )
    save(
        lead_chart(
            d=lead,
            column="cover90",
            title="Share of observed readings inside the q05-q95 band",
            y_title="Coverage of q05-q95 (target 0.90; closer to 0.90 is better)",
            rule=0.9,
        ),
        out,
        "coverage_vs_lead.svg",
    )
    prim = (
        pl.read_csv(score / "by_primary.csv")
        .join(pl.read_csv(score / "primaries.csv"), left_on="primary", right_on="primary")
        .with_columns(
            group=pl.col("group").replace(
                {"report-only (D15 and D18, short history)": "report-only (D15, D18)"}
            )
        )
    )
    order = prim.filter(pl.col("system") == "K1").sort("crps")["label"].to_list()
    for col, name, ttl in (
        ("crps", "crps_by_primary.svg", "CRPS (% of capacity; lower is better)"),
        ("mae", "mae_by_primary.svg", "MAE of the median (% of capacity; lower is better)"),
    ):
        save(
            alt.Chart(prim)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=70)
            .encode(
                y=alt.Y("label:N", sort=order, title=None),
                x=alt.X(f"{col}:Q", title=ttl),
                yOffset=alt.YOffset("system:N", sort=SYSTEMS),
                color=alt.Color("system:N", scale=COLOUR_SCALE, legend=LEGEND),
            )
            .properties(width=W - 120, height=44 * len(order) + 40),
            out,
            name,
        )
    con = pl.read_csv(score / "contrasts.csv")
    for m in ("crps", "mae"):
        save(contrast_chart(d=con, measure=m), out, f"contrast_{m}.svg")
    cost = pl.read_csv(score / "curtailment_cost.csv").rename({"cost_gbp_per_day_0.9": "cost90"})
    if cost.height:
        save(
            alt.Chart(cost)  # ty: ignore[unresolved-attribute]
            .mark_bar()
            .encode(
                y=alt.Y("label:N", title=None),
                x=alt.X(
                    "cost90:Q",
                    title="Avoidable cost, £ per Primary per day (limit at 90% of the pre-window peak; lower is better)",
                ),
                yOffset=alt.YOffset("system:N", sort=SYSTEMS),
                color=alt.Color("system:N", scale=COLOUR_SCALE, legend=LEGEND),
            )
            .properties(width=W - 120, height=40 * cost["label"].n_unique() + 60),
            out,
            "curtailment_cost.svg",
        )
    # example forecasts: one Primary, 14 days, day-ahead segment (24-48 h) of each origin, one forecast arm per panel
    sub = pl.read_csv(
        Path(__import__("os").environ["DATA_PATH_INTERNAL"])
        / "ml-experiments"
        / "backtest_export_roster_2026-11-07"
        / "substations.csv"
    )
    label = "D5" if "D5" in sub["primary_name"].to_list() else sub["primary_name"][0]
    tsid = sub.filter(pl.col("primary_name") == label)["time_series_id"][0]
    start = (
        datetime(2026, 8, 3, tzinfo=UTC) if month == "2026-07" else datetime(2026, 6, 8, tzinfo=UTC)
    )
    panels = []
    names = ["Median", "q05-q95 band", "Observed power"]
    scale = alt.Scale(domain=names, range=[theme.DATA_BLUE, theme.DATA_BLUE_LIGHT, theme.BLACK_1])
    for arm, ttl in (
        ("K1", "K1 (lag network with the G4 gated band)"),
        ("K3", "K3 (lag-free network)"),
    ):
        t = pl.read_parquet(export / f"{arm}_{month}.parquet").filter(
            (pl.col("time_series_id") == tsid)
            & (pl.col("lead_hours") >= 24)
            & (pl.col("lead_hours") < 48)
            & (pl.col("valid_time") >= start)
            & (pl.col("valid_time") < start + timedelta(days=14))
        )
        t = t.with_columns(vt=pl.col("valid_time").dt.replace_time_zone(None))
        unit = t["unit"][0]
        band = (
            alt.Chart(t)  # ty: ignore[unresolved-attribute]
            .transform_calculate(series="'q05-q95 band'")
            .mark_area(opacity=0.6)
            .encode(
                x=alt.X("vt:T", title="Valid time (UTC)"),
                y=alt.Y("q05:Q", title=f"Power ({unit})"),
                y2="q95:Q",
                color=alt.Color(
                    "series:N", scale=scale, legend=alt.Legend(title=None, orient="bottom")
                ),
            )
        )
        med = (
            alt.Chart(t)  # ty: ignore[unresolved-attribute]
            .transform_calculate(series="'Median'")
            .mark_line(strokeWidth=1.5)
            .encode(x="vt:T", y="q50:Q", color=alt.Color("series:N", scale=scale, legend=None))
        )
        obs = (
            alt.Chart(t)  # ty: ignore[unresolved-attribute]
            .transform_calculate(series="'Observed power'")
            .mark_line(strokeWidth=1)
            .encode(
                x="vt:T",
                y="observed_power:Q",
                color=alt.Color("series:N", scale=scale, legend=None),
            )
        )
        panels.append(
            alt.layer(band, med, obs).properties(
                width=W, height=300, title=f"{label}, {ttl}: day-ahead forecasts (24-48 h lead)"
            )
        )
    save(alt.vconcat(*panels).resolve_scale(color="shared"), out, f"example_forecast_{label}.svg")
    print("figures written to", out, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
