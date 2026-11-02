r"""Draw the figures of experiment 480's write-up.

The contrast numbers are those printed by `e00480_score.py` (pooled rows at the 99.67% level, each
Primary at 95%, 20,000 and 2,000 resamples), typed in so these two charts need no re-score. The
two example-forecast charts read the stored A0 and A1 forecasts.

    PYTHONPATH=ml-experiments/00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions:\\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\\
ml-experiments/00377-demand-v1-first-run-feasibility:\\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \\
        uv run --no-sync python ml-experiments/00480-*/e00480_charts.py
"""

import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00480_score as score  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions"
)
WIDTH: Final[int] = 680
WRAP: Final[int] = 110
GROUP_COLOURS: Final[dict[str, str]] = {
    "signed-MW six": ocf.DATA_BLUE,
    "MVA-only eight": ocf.BRAND_ORANGE,
    "short history": ocf.DATA_PURPLE,
}
HEADLINE_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled MVA-only eight", "MVA-only eight", -0.355, -0.497, -0.228),
    ("Pooled without D8", "MVA-only eight", -0.386, -0.543, -0.241),
    ("Pooled signed-MW six", "signed-MW six", -0.476, -0.627, -0.316),
    ("Pooled D15 and D18", "short history", -0.313, -0.429, -0.201),
    ("D2", "MVA-only eight", -0.528, -0.679, -0.362),
    ("D3", "MVA-only eight", -0.198, -0.264, -0.133),
    ("D4", "MVA-only eight", -0.053, -0.094, -0.016),
    ("D8 (marked)", "MVA-only eight", -0.138, -0.196, -0.085),
    ("D12 (marked, reverse flow)", "MVA-only eight", -0.769, -1.038, -0.514),
    ("D19", "MVA-only eight", -0.330, -0.425, -0.229),
    ("D21", "MVA-only eight", -0.299, -0.366, -0.223),
    ("D22", "MVA-only eight", -0.521, -0.701, -0.376),
    ("D5 (marked)", "signed-MW six", -0.843, -1.019, -0.652),
    ("D9", "signed-MW six", -0.373, -0.457, -0.285),
    ("D10", "signed-MW six", -0.513, -0.665, -0.359),
    ("D11", "signed-MW six", -0.343, -0.427, -0.255),
    ("D14", "signed-MW six", -0.370, -0.447, -0.284),
    ("D16", "signed-MW six", -0.418, -0.529, -0.313),
    ("D15 (marked)", "short history", -0.149, -0.305, 0.018),
    ("D18 (marked)", "short history", -0.477, -0.629, -0.328),
]
BAND_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Day 1, MVA-only eight", "MVA-only eight", -0.243, -0.342, -0.164),
    ("Day 2", "MVA-only eight", -0.268, -0.377, -0.178),
    ("Days 3 to 4", "MVA-only eight", -0.320, -0.447, -0.208),
    ("Days 5 to 7", "MVA-only eight", -0.385, -0.545, -0.244),
    ("Days 8 to 10", "MVA-only eight", -0.415, -0.605, -0.254),
    ("Days 11 to 14", "MVA-only eight", -0.409, -0.604, -0.237),
    ("Day 1, signed-MW six", "signed-MW six", -0.319, -0.429, -0.215),
    ("Day 2 ", "signed-MW six", -0.350, -0.468, -0.239),
    ("Days 3 to 4 ", "signed-MW six", -0.415, -0.554, -0.271),
    ("Days 5 to 7 ", "signed-MW six", -0.518, -0.691, -0.335),
    ("Days 8 to 10 ", "signed-MW six", -0.572, -0.748, -0.378),
    ("Days 11 to 14 ", "signed-MW six", -0.578, -0.765, -0.367),
]
SLICE_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Sunny, unit shape", "MVA-only eight", -0.967, -1.412, -0.611),
    ("Sunny, CAMS index 0.7+", "MVA-only eight", -0.626, -0.861, -0.384),
    ("Winter nights", "MVA-only eight", -0.016, -0.043, 0.003),
    ("Cold nights", "MVA-only eight", -0.012, -0.053, 0.024),
    ("Economy 7 step hours", "MVA-only eight", -0.008, -0.026, 0.003),
    ("Sunny, unit shape ", "signed-MW six", -1.329, -1.774, -0.956),
    ("Sunny, CAMS index 0.7+ ", "signed-MW six", -0.877, -1.139, -0.550),
    ("Winter nights ", "signed-MW six", -0.007, -0.053, 0.021),
    ("Cold nights ", "signed-MW six", 0.006, -0.014, 0.033),
    ("Economy 7 step hours ", "signed-MW six", 0.002, -0.009, 0.012),
]
EXAMPLE_RUNS: Final[dict[str, datetime]] = {
    "Summer (June 2026 run)": datetime(2026, 6, 16, tzinfo=UTC),
    "Winter (January 2026 run)": datetime(2026, 1, 13, tzinfo=UTC),
}


def draw(*, rows: list[tuple[str, str, float, float, float]], title: str, subtitle: str,
         x_title: str, name: str) -> None:  # fmt: skip
    """Write one interval chart to `name`."""
    frame = pl.DataFrame(
        rows, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in rows]
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUP_COLOURS), range=list(GROUP_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q"
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 230,
        height=20 * len(rows),
        title=alt.TitleParams(
            text=textwrap.wrap(title, 80),
            subtitle=textwrap.wrap(subtitle, WRAP),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / name)


def draw_examples(*, horizon_hours: float, name: str, title: str, subtitle: str) -> None:
    """Write the example-forecast chart of D12: A0 and A1 medians and 90% bands against actuals."""
    series = adapter.series_table()
    row = series.filter(pl.col("label") == "D12")
    pieces = []
    for arm in ("a0", "a1"):
        frame = score.own_arm(name=arm, series=row)
        pieces.append(
            frame.with_columns(
                arm=pl.lit("A0 (no CAMS columns)" if arm == "a0" else "A1 (with CAMS columns)")
            )
        )
    forecasts = pl.concat(pieces)
    actuals = adapter.actuals_and_context(
        cache=score.s381.load_or_build_feature_cache(
            paths=score.s381.default_input_paths(), cache_dir=score.base.CACHE_DIR
        ),
        series=row,
        first=datetime(2025, 7, 1, tzinfo=UTC),
        last=datetime(2026, 7, 1, tzinfo=UTC),
    )[0]
    panels = []
    for label, origin in EXAMPLE_RUNS.items():
        available = forecasts["power_fcst_init_time"].unique().sort()
        chosen = available.sort().item(int((available - origin).abs().arg_min() or 0))
        print(f"{label}: run {chosen:%Y-%m-%d}")
        one = (
            forecasts.filter(pl.col("power_fcst_init_time") == chosen)
            .join(actuals, on=["time_series_id", "valid_time"], how="left")
            .with_columns(
                hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60
            )
            .filter(pl.col("hours") <= horizon_hours)
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=["A0 (no CAMS columns)", "A1 (with CAMS columns)"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
        x = alt.X("hours:Q", title="Hours after the forecast was made")
        band = (
            alt.Chart(one)
            .mark_area(opacity=0.2)
            .encode(x=x, y=alt.Y("q05:Q", title="Power (MW)"), y2="q95:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        median = alt.Chart(one).mark_line(strokeWidth=1.2).encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        actual = (
            alt.Chart(one.unique("valid_time"))
            .mark_line(color="black", strokeWidth=1.2)
            .encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (band + median + actual).properties(width=WIDTH - 90, height=150, title=label)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text=textwrap.wrap(title, 80), subtitle=textwrap.wrap(subtitle, WRAP), anchor="start"
        )
    )
    chart.save(DOCS_DIR / name)


def main() -> None:
    """Write the SVG files."""
    draw(
        rows=HEADLINE_ROWS,
        title="Two columns of realised satellite irradiance lowered the error at every group "
        "and at 19 of 20 rows",
        subtitle="A1 (monthly XGBoost model plus the CAMS unit PV shape and clear-sky index) "
        "minus A0 (same model without them), CRPS at days 1 to 10 on the shared half-hours of "
        "2025-07 to 2026-06. Negative is better for A1. Dot: estimate. Line: 99.67% interval "
        "(pooled rows) or 95% interval (single Primary), from resampling whole months. "
        "The columns hold realised irradiance, so a live service cannot supply them for the "
        "first day of lead.",
        x_title="CRPS of A1 minus A0 (points of capacity)",
        name="headline.svg",
    )
    draw(
        rows=BAND_ROWS,
        title="The gain grows with lead time up to days 8 to 10 and stays level after",
        subtitle="A1 minus A0, CRPS by lead band, pooled over each group, 99.67% interval from "
        "resampling whole months. Negative is better for A1.",
        x_title="CRPS of A1 minus A0 (points of capacity)",
        name="lead_bands.svg",
    )
    draw(
        rows=SLICE_ROWS,
        title="The gain sits on sunny half-hours; winter nights, cold nights, and Economy 7 "
        "step hours barely move",
        subtitle="A1 minus A0, CRPS at days 1 to 10, pooled over each group (colour), 99.67% "
        "interval from resampling whole months. Negative is better for A1.",
        x_title="CRPS of A1 minus A0 (points of capacity)",
        name="slices.svg",
    )
    draw_examples(
        horizon_hours=24.0,
        name="example_day_ahead.svg",
        title="Day-ahead forecasts at D12, one summer run and one winter run",
        subtitle="One forecast run, first 24 hours. Black: actual power. Line: median. Band: "
        "90% interval (5th to 95th percentile).",
    )
    draw_examples(
        horizon_hours=336.0,
        name="example_14_day.svg",
        title="Full 14-day forecasts at D12: the CAMS columns lift the summer midday peaks",
        subtitle="One forecast run each, 14 days. Black: actual power. Line: median. Band: 90% "
        "interval (5th to 95th percentile).",
    )


if __name__ == "__main__":
    main()
