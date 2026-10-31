"""Draw the figures of experiment 408's write-up.

- `headline.svg`: A1 minus A0, CRPS at days 1 to 10, per Primary and pooled, numbers typed in from
  `e00408_score.py`.
- `gate.svg`: balanced accuracy of the signed median at the six signed-MW Primaries against the
  shortwave rule, typed in from `e00408_gate.py` at injection level 2.0.
- `example_day_ahead.svg` and `example_14_day.svg`: D12 forecasts from A0 and A1 against the cleaned
  actuals, read from the stored forecasts.

    uv run --no-sync python ml-experiments/00408-*/e00408_charts.py
"""

import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00408_fit as fit
import plotting.ocf_theme as ocf
import polars as pl
from studies.power import scan_power

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00408-xgboost-signed-label-and-cams-columns-at-mva-only-primaries"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
HEADLINE_ROWS: Final[list[tuple[str, str, float, float, float]]] = [
    ("Pooled, eight (99.67%)", "pooled", -0.359, -0.500, -0.229),
    ("D2", "MVA-only", -0.526, -0.676, -0.364),
    ("D3", "MVA-only", -0.200, -0.264, -0.136),
    ("D4", "MVA-only", -0.053, -0.097, -0.014),
    ("D8", "MVA-only", -0.152, -0.206, -0.105),
    ("D12", "MVA-only", -0.772, -1.037, -0.521),
    ("D19", "MVA-only", -0.333, -0.428, -0.231),
    ("D21", "MVA-only", -0.297, -0.364, -0.223),
    ("D22", "MVA-only", -0.521, -0.705, -0.370),
    ("D5 (signed-MW)", "signed-MW", -0.840, -1.013, -0.648),
]
GATE_ROWS: Final[list[tuple[str, str, float]]] = [
    ("D5", "signed median (A2 rule)", 0.788),
    ("D9", "signed median (A2 rule)", 0.841),
    ("D10", "signed median (A2 rule)", 0.825),
    ("D11", "signed median (A2 rule)", 0.849),
    ("D14", "signed median (A2 rule)", 0.846),
    ("D16", "signed median (A2 rule)", 0.843),
    ("D5", "shortwave-threshold rule", 0.868),
    ("D9", "shortwave-threshold rule", 0.875),
    ("D10", "shortwave-threshold rule", 0.873),
    ("D11", "shortwave-threshold rule", 0.860),
    ("D14", "shortwave-threshold rule", 0.869),
    ("D16", "shortwave-threshold rule", 0.873),
]
EXAMPLES: Final[dict[str, datetime]] = {
    "Summer": datetime(2025, 7, 15, tzinfo=UTC),
    "Winter": datetime(2026, 1, 20, tzinfo=UTC),
}
SITE: Final[str] = "D12"
ARM_COLOURS: Final[dict[str, str]] = {"A0": ocf.DATA_BLUE, "A1": ocf.BRAND_ORANGE}


def interval_chart() -> None:
    """Write `headline.svg`."""
    frame = pl.DataFrame(
        HEADLINE_ROWS, schema=["row", "group", "difference", "lower", "upper"], orient="row"
    )
    order = [r[0] for r in HEADLINE_ROWS]
    colours = {"pooled": ocf.BRAND_ORANGE, "MVA-only": ocf.DATA_BLUE, "signed-MW": ocf.DATA_PURPLE}
    base = alt.Chart(frame).encode(
        y=alt.Y("row:N", sort=order, title=None),
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title="CRPS of A1 minus A0 (points of effective capacity)"), x2="upper:Q"
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=WIDTH - 200,
        height=22 * len(HEADLINE_ROWS),
        title=alt.TitleParams(
            text=textwrap.wrap(
                "The half-hour's own satellite solar columns lower the error at every Primary", 60
            ),
            subtitle=textwrap.wrap(
                "Monthly XGBoost model with two CAMS columns (A1) minus without (A0), CRPS at "
                "days 1 to 10, 2025-07 to 2026-06. Negative is better. Lab conditions: the CAMS "
                "value is realised weather at every lead. Line: 95% interval per Primary and "
                "99.67% interval pooled, from resampling whole months.",
                WRAP,
            ),
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "headline.svg")


def gate_chart() -> None:
    """Write `gate.svg`."""
    frame = pl.DataFrame(GATE_ROWS, schema=["site", "estimate", "balanced_accuracy"], orient="row")
    colours = {
        "signed median (A2 rule)": ocf.BRAND_ORANGE,
        "shortwave-threshold rule": ocf.DATA_BLUE,
    }
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", title=None, sort=["D5", "D9", "D10", "D11", "D14", "D16"]),
            xOffset="estimate:N",
            y=alt.Y(
                "balanced_accuracy:Q",
                title="Balanced accuracy",
                scale=alt.Scale(domain=[0.5, 1.0]),
            ),
            color=alt.Color(
                "estimate:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    line = (
        alt.Chart(pl.DataFrame({"y": [0.85]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    (chart + line).properties(
        width=WIDTH - 100,
        height=240,
        title=alt.TitleParams(
            text=textwrap.wrap(
                "At injection level 2.0 the signed median fell short of the 0.85 pass line", 60
            ),
            subtitle=textwrap.wrap(
                "Balanced accuracy of the sign of true reverse flow (injected solar farm output), "
                "days 1 to 10, daylight half-hours. Dashed line: pass line for the pooled value.",
                WRAP,
            ),
            anchor="start",
        ),
    ).save(DOCS_DIR / "gate.svg")


def example_frame(*, arm: str, init: datetime) -> pl.DataFrame:
    """Return one arm's quantiles (points of capacity) for the first run at or after `init`."""
    files = sorted((fit.mode_dir(mode="real") / "preds" / arm).glob(f"{SITE}_*.parquet"))
    stored = pl.concat([pl.read_parquet(f) for f in files]).filter(
        pl.col("power_fcst_init_time") >= init
    )
    first = stored["power_fcst_init_time"].min()
    return stored.filter(pl.col("power_fcst_init_time") == first).with_columns(
        arm=pl.lit(arm),
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        **{q: pl.col(q) * 100 for q in common.LEVEL_COLUMNS},
    )


def example_charts() -> None:
    """Write the day-ahead and 14-day example forecasts at D12."""
    series = common.series_table().filter(pl.col("site") == SITE)
    sid, capacity = series["time_series_id"][0], series["effective_capacity_mw"][0]
    actual = (
        scan_power()
        .filter(pl.col("time_series_id") == sid)
        .select(pl.col("time").alias("valid_time"), actual=pl.col("power") / capacity * 100)
        .collect()
    )
    for kind, window in (("day_ahead", (24.0, 48.0)), ("14_day", (0.0, 336.0))):
        parts = []
        for season, init in EXAMPLES.items():
            for arm in ("A0", "A1"):
                frame = example_frame(arm=arm, init=init)
                parts.append(
                    frame.filter(
                        (pl.col("lead_hours") >= window[0]) & (pl.col("lead_hours") < window[1])
                    )
                    .join(actual, on="valid_time", how="left")
                    .with_columns(season=pl.lit(season))
                )
        data = pl.concat(parts)
        base = alt.Chart(data).encode(x=alt.X("valid_time:T", title=None))
        band90 = base.mark_area(opacity=0.15).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Power (points of effective capacity)"),
            y2="q95:Q",
            color=alt.Color(
                "arm:N", scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values()))
            ),
        )
        band50 = base.mark_area(opacity=0.3).encode(  # ty: ignore[unresolved-attribute]
            y="q25:Q", y2="q75:Q", color="arm:N"
        )
        median = base.mark_line().encode(y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        truth = base.mark_line(color="#000000", strokeDash=[3, 2]).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        layered = (band90 + band50 + median + truth).properties(width=WIDTH - 120, height=180)
        chart = (
            layered.facet(row=alt.Row("season:N", title=None))
            .resolve_scale(x="independent")
            .properties(
                title=alt.TitleParams(
                    text=textwrap.wrap(
                        "Example D12 forecast with and without the CAMS columns: "
                        + ("day ahead" if kind == "day_ahead" else "the full 14 days"),
                        60,
                    ),
                    subtitle=textwrap.wrap(
                        "One run per season. Line: median. Light band: 25% to 75%. Faint band: 5% "
                        "to 95%. Dashed black: cleaned actual. A0 blue, A1 orange.",
                        WRAP,
                    ),
                    anchor="start",
                )
            )
        )
        chart.save(DOCS_DIR / f"example_{kind}.svg")


def main() -> None:
    """Write the four SVG files."""
    interval_chart()
    gate_chart()
    example_charts()


if __name__ == "__main__":
    main()
