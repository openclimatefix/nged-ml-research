# ruff: noqa: E501
"""Figures of experiment 525, from the tables the scorer and `e00525_mass_report` stored.

Figures: 1 the CRPS and q99 pinball contrasts with Bonferroni intervals; 2 the mass-below-zero
ratio of M to C
per fold and seed at D8, D12, and D22; 3 and 4 example forecasts (day-ahead and 10 days) at D8 and D10.
Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00525_fit as fit  # ty: ignore[unresolved-import]
import e00525_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00525-metering-type-flag-lagless-demand-network"
)
SCORES: Final[Path] = fit.W / "scores"
GROUP_LABEL: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"
EXAMPLES: Final[dict[str, str]] = {"D8": "MVA-only Primary", "D10": "signed-MW Primary"}
COLOURS: Final[dict[str, str]] = {
    "L-Ww": theme.DATA_GREEN,
    "C": theme.DATA_BLUE,
    "M": theme.BRAND_ORANGE,
}


def contrasts() -> pl.DataFrame:
    """Return the CRPS and q99 contrasts of both families on the two counted groups."""
    frames = [
        pl.read_csv(SCORES / "contrasts.csv"),
        pl.read_csv(fit.W / "scores_report" / "contrasts.csv"),
    ]
    return pl.concat(frames).filter(
        pl.col("metric").is_in(["crps", "pin_q99"]),
        pl.col("rowset") == "main16",
        pl.col("slice").is_in(list(GROUP_LABEL)),
    )


def figures() -> None:
    """Draw figures 1 and 2, then the examples."""
    con = contrasts().with_columns(
        group=pl.col("slice").replace_strict(GROUP_LABEL),
        metric_label=pl.col("metric").replace_strict(
            {"crps": "CRPS", "pin_q99": "q99 pinball loss"}
        ),
    )
    order = ["M - C", "M - L-Ww", "P - C", "M - P", "C - L-Ww"]
    panels = []
    for metric in ("CRPS", "q99 pinball loss"):
        for group in GROUP_LABEL.values():
            part = con.filter(
                (pl.col("metric_label") == metric) & (pl.col("group") == group)
            ).to_pandas()
            base = alt.Chart(part).encode(y=alt.Y("contrast:N", sort=order, title=None))  # ty: ignore[unresolved-attribute]
            bar = base.mark_rule(strokeWidth=3, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("low_bonferroni:Q", title=f"{metric} difference (points of capacity)"),
                x2="high_bonferroni:Q",
            )  # ty: ignore[unresolved-attribute]
            dot = base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                x="difference:Q"
            )  # ty: ignore[unresolved-attribute]
            zero = (
                alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_rule(color="black")
                .encode(x="x:Q")
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                alt.layer(zero, bar, dot).properties(
                    width=420,
                    height=120,
                    title=alt.TitleParams(f"{metric}, {group}", anchor="start"),
                )
            )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Paired contrasts (negative favours the first arm), five folds, two seeds.",
                "M: metering flag. C: constant flag. P: permuted flag (report-only). Bonferroni intervals: 8 counted tests (M - C, M - L-Ww); P and C - L-Ww rows are report-only.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_1_headline.svg")
    ratio = pl.read_csv(SCORES / "mass_ratio.csv").with_columns(
        cell=pl.col("month") + " s" + pl.col("seed").cast(pl.String)
    )
    base = alt.Chart(ratio.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("cell:N", title=None), y=alt.Y("ratio:Q", title="Mass below zero, M divided by C")
    )
    base.mark_point(filled=True, size=60, color=theme.DATA_BLUE).properties(  # ty: ignore[unresolved-attribute]
        width=200, height=180
    ).facet(column=alt.Column("site:N", title=None)).properties(  # ty: ignore[unresolved-attribute]
        title=alt.TitleParams(
            "Mixture mass below zero, M over C, per fold and seed (no change: 1; pre-registered target: 0.5 or lower)",
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_2_mass_ratio.svg")
    examples()


def examples() -> None:
    """Draw a day-ahead and a 10-day forecast of M with C and L-Ww medians at D8 and D10."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    frames = {}
    for arm in COLOURS:
        pooled = build470.pool_stored(
            folder=score.SEED_PREDS / arm, month=month, sites=sites, seeds=(0, 1)
        )
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
            series=series,
            unit="fraction",
        )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(hour=9, tzinfo=UTC)
    for label, kind in EXAMPLES.items():
        one = {
            a: f.filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin)
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0).sort("valid_time")
            for a, f in frames.items()
        }  # fmt: skip
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == ids[label])
            .join(one["M"].select("valid_time", "day"), on="valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            main = one["M"].filter(pl.col("day").is_between(*window)).to_pandas()
            band = alt.Chart(main).encode(x=x)  # ty: ignore[unresolved-attribute]
            layers = [
                band.mark_area(opacity=0.2, color=COLOURS["M"]).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),
                band.mark_area(opacity=0.35, color=COLOURS["M"]).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
            ]
            for arm, f in one.items():
                sub = f.filter(pl.col("day").is_between(*window)).to_pandas()
                layers.append(
                    alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                    .mark_line(color=COLOURS[arm], strokeWidth=2)
                    .encode(x=x, y="q50:Q")
                )  # ty: ignore[unresolved-attribute]
            act = actual.filter(pl.col("day").is_between(*window)).to_pandas()
            layers.append(
                alt.Chart(act)  # ty: ignore[unresolved-attribute]
                .mark_line(color="black", strokeWidth=1.5, strokeDash=[4, 3])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            alt.layer(*layers).properties(
                width=420, height=200,
                title=alt.TitleParams([f"{label} ({kind}), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: M 50% and 90%. Medians: M orange, C blue, L-Ww green. Dashed: actual."], anchor="start"),
            ).save(DOCS_DIR / f"figure_3_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg")  # fmt: skip


if __name__ == "__main__":
    figures()
