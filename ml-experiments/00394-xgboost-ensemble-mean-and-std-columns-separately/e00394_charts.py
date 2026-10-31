# ruff: noqa: E501
"""Draw the figures of experiment 394's write-up.

The numbers are those printed by `e00394_score.py` and `e00394_bins.py`, typed in so the charts
need no re-score.

    uv run --no-sync python ml-experiments/00394-*/e00394_charts.py
"""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00394-xgboost-ensemble-mean-and-std-columns-separately"
)
EXAMPLE_ORIGIN: Final[datetime] = datetime(2026, 1, 12, 9, 0, tzinfo=UTC)
GROUPS: Final[dict[str, str]] = {"signed-MW six": ocf.DATA_BLUE, "MVA-only eight": ocf.BRAND_ORANGE}
ARM_COLOURS: Final[dict[str, str]] = {
    "Mean and spread (S1)": ocf.DATA_BLUE,
    "Mean only": ocf.BRAND_ORANGE,
    "Spread only": "#808080",
}
CONTRASTS: Final[list[tuple[str, str, float, float, float]]] = [
    # arm, group, difference, lower, upper (CRPS, points of capacity; arm minus stored monthly member)
    ("Mean and spread (S1)", "signed-MW six", -0.047, -0.078, -0.008),
    ("Mean and spread (S1)", "MVA-only eight", -0.035, -0.052, -0.019),
    ("Mean only", "signed-MW six", -0.052, -0.082, -0.009),
    ("Mean only", "MVA-only eight", -0.029, -0.044, -0.013),
    ("Spread only", "signed-MW six", -0.001, -0.011, 0.013),
    ("Spread only", "MVA-only eight", -0.008, -0.014, -0.002),
]
BINS: Final[list[tuple[str, str, str, float, float, float]]] = [
    # arm, group, bin, difference, lower, upper (q95 pinball loss, points; arm minus stored monthly member)
    ("Mean and spread (S1)", "signed-MW six", "low", -0.0223, -0.0427, 0.0082),
    ("Mean and spread (S1)", "signed-MW six", "middle", -0.0033, -0.0286, 0.0466),
    ("Mean and spread (S1)", "signed-MW six", "high", -0.0031, -0.0285, 0.0420),
    ("Mean only", "signed-MW six", "low", -0.0233, -0.0418, 0.0038),
    ("Mean only", "signed-MW six", "middle", -0.0113, -0.0325, 0.0319),
    ("Mean only", "signed-MW six", "high", -0.0128, -0.0368, 0.0305),
    ("Spread only", "signed-MW six", "low", -0.0032, -0.0164, 0.0132),
    ("Spread only", "signed-MW six", "middle", 0.0074, -0.0035, 0.0233),
    ("Spread only", "signed-MW six", "high", 0.0110, 0.0020, 0.0218),
    ("Mean and spread (S1)", "MVA-only eight", "low", -0.0212, -0.0276, -0.0141),
    ("Mean and spread (S1)", "MVA-only eight", "middle", -0.0137, -0.0193, -0.0054),
    ("Mean and spread (S1)", "MVA-only eight", "high", -0.0079, -0.0186, 0.0046),
    ("Mean only", "MVA-only eight", "low", -0.0143, -0.0207, -0.0066),
    ("Mean only", "MVA-only eight", "middle", -0.0113, -0.0164, -0.0028),
    ("Mean only", "MVA-only eight", "high", -0.0073, -0.0167, 0.0045),
    ("Spread only", "MVA-only eight", "low", -0.0096, -0.0132, -0.0060),
    ("Spread only", "MVA-only eight", "middle", -0.0035, -0.0065, -0.0007),
    ("Spread only", "MVA-only eight", "high", 0.0009, -0.0035, 0.0056),
]


def titled(*, title: str, subtitle: str) -> alt.TitleParams:
    """Return a title and subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110), anchor="start"
    )


def figure_contrasts() -> None:
    """Write figure 1: the CRPS contrasts with 99% intervals."""
    frame = pl.DataFrame(
        CONTRASTS, schema=["arm", "group", "difference", "lower", "upper"], orient="row"
    )
    order = list(ARM_COLOURS)
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset="group:N",
        color=alt.Color(
            "group:N",
            scale=alt.Scale(domain=list(GROUPS), range=list(GROUPS.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title="Arm minus the stored monthly member (points of capacity)"),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=50).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    chart = (zero + bars + dots).properties(
        width=300,
        height=170,
        title=titled(
            title="The ensemble mean carries most of the CRPS gain; the spread alone carries little",
            subtitle=(
                "Quantile-score CRPS, days 1 to 10, 2025-07 to 2026-06, monthly refits, factor 1.15. Dot: estimate. "
                "Line: 99% interval from resampling whole months. Negative: the arm is better."
            ),
        ),
    )
    chart.save(DOCS_DIR / "figure_1_contrasts.svg")


def figure_bins() -> None:
    """Write figure 2: the q95 pinball loss by tercile of temperature spread."""
    frame = pl.DataFrame(
        BINS, schema=["arm", "group", "bin", "difference", "lower", "upper"], orient="row"
    )
    panels = []
    for group in GROUPS:
        base = alt.Chart(frame.filter(pl.col("group") == group)).encode(
            x=alt.X(
                "bin:N",
                sort=["low", "middle", "high"],
                title="Tercile of the ensemble's temperature spread",
            ),
            xOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        bars = base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("lower:Q", title="Arm minus the stored monthly member (points)"),
            y2="upper:Q",
        )
        dots = base.mark_point(filled=True, size=50).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}))
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append((zero + bars + dots).properties(width=240, height=200, title=group))
    chart = alt.hconcat(*panels).properties(
        title=titled(
            title="The q95 gain is largest where the ensemble agrees, not where it disagrees most",
            subtitle=(
                "Pinball loss at q95, days 1 to 10, all hours, terciles cut within each Primary and lead band. "
                "Dot: estimate. Line: 99% interval from resampling months. Negative: the arm is better."
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_spread_terciles.svg")


def figure_example() -> None:
    """Write figure 3: one 14-day forecast of D5 from two arms, against what happened."""
    import e00389_calibrate as cal  # ty: ignore[unresolved-import]

    work = (
        ML_EXPERIMENTS_DATA_DIR
        / "00394-xgboost-ensemble-mean-and-std-columns-separately"
        / "work"
        / "arms"
    )
    reference = (
        ML_EXPERIMENTS_DATA_DIR
        / "00389-xgboost-tail-calibration-and-ensemble-spread-columns"
        / "work"
        / "arms"
    )
    stored = pl.read_parquet(cal.STORED_BG2).filter(pl.col("site") == "D5")
    series_id = stored["time_series_id"][0]
    actual = stored.filter(pl.col("power_fcst_init_time") == EXAMPLE_ORIGIN).select(
        "valid_time", "actual"
    )
    parts = []
    for arm, path in (
        ("Stored monthly member", reference / "S0M_base.parquet"),
        ("Mean only", work / "MEAN_base.parquet"),
    ):
        parts.append(
            pl.read_parquet(path)
            .filter(
                pl.col("time_series_id") == series_id,
                pl.col("power_fcst_init_time") == EXAMPLE_ORIGIN,
            )
            .join(actual.select("valid_time"), on="valid_time", how="semi")
            .select("valid_time", "q05", "q50", "q95")
            .with_columns(arm=pl.lit(arm))
        )
    forecasts = pl.concat(parts)
    kinds = [("Full 14 days", None), ("First 48 hours", EXAMPLE_ORIGIN + timedelta(hours=48))]
    panels = []
    for title, end in kinds:
        f = forecasts if end is None else forecasts.filter(pl.col("valid_time") < end)
        a = actual if end is None else actual.filter(pl.col("valid_time") < end)
        band = (
            alt.Chart(f)
            .mark_area(opacity=0.25)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("valid_time:T", title=None),
                y=alt.Y("q05:Q", title="Share of effective capacity"),
                y2="q95:Q",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=["Stored monthly member", "Mean only"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        line = (
            alt.Chart(a).mark_line(color="#000000").encode(x="valid_time:T", y="actual:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append((band + line).properties(width=330, height=200, title=title))
    chart = alt.hconcat(*panels).properties(
        title=titled(
            title=f"An example forecast at D5 from the run of {EXAMPLE_ORIGIN:%-d %B %Y}",
            subtitle="Shaded: 5% to 95% band. Black: what happened. The two bands differ little, except for a wider lower band in the stored member on the second afternoon.",
        )
    )
    chart.save(DOCS_DIR / "figure_3_example_forecast.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_bins()
    figure_example()
