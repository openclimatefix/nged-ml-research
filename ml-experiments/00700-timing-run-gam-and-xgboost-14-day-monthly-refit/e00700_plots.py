"""Experiment 700: charts of the provisional scores for origin months 2025-11 to 2026-06.

    uv run --no-sync python ml-experiments/00700-*/e00700_plots.py

Reads `scores/protocol_year/{rows.parquet,scores.csv}` under this experiment's private folder (from
`e00700_protocol.py`) and writes SVG and PNG charts to `plots/` beside them. The Primaries are the 15
that all arms share: D15 has no GAM forecast before 2026-03.
"""

from pathlib import Path

import altair as alt
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Path = (
    ML_EXPERIMENTS_DATA_DIR / "00700-timing-run-gam-and-xgboost-14-day-monthly-refit"
)
SCORES: Path = ROOT / "scores" / "protocol_year"
OUT: Path = ROOT / "plots"
CAPACITY: Path = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10" / "series_capacity.parquet"

ARMS = {
    "GAM": "GAM",
    "XGBoost BG2 (weekly W member)": "XGBoost",
    "Step 8 monthly (9b)": "Neural network (step 8, monthly refits)",
    "Manual heuristic": "Manual heuristic",
    "Climatology": "Climatology",
}
NAMES = list(ARMS.values())
COLOURS = ["#17E58F", "#306BFF", "#FF4901", "#555555", "#B701FF"]
DASH = [[1, 0], [1, 0], [1, 0], [6, 3], [2, 2]]
LEAD_SLICES = {
    "lead_day_1": "Day 1 (0-24 h)",
    "lead_day_2": "Day 2 (24-48 h)",
    "lead_days_3_4": "Days 3-4 (48-96 h)",
    "lead_days_5_7": "Days 5-7 (96-168 h)",
    "lead_days_8_10": "Days 8-10 (168-240 h)",
}
CONDITION_SLICES = {
    "all": "All days",
    "sunny_unit_shape": "Sunny (unit shape)",
    "winter": "Winter (Dec-Feb)",
    "winter_nights": "Winter nights",
    "cold_nights": "Cold nights (3 °C or less, thin)",
    "economy_7_step": "Economy 7 step hours",
    "christmas": "Christmas (24 Dec-1 Jan)",
    "bank_holidays": "Bank holidays",
}
METRICS = {"mae_median": "MAE", "crps": "CRPS", "twcrps": "twCRPS"}
GROUPS = {"signed-MW six": "the 6 Primaries with signed MW", "MVA-only seven": "the 7 MVA-only Primaries"}
SUBTITLE = (
    "Provisional: 8 monthly refits, origins at 12:00 UTC from 2025-11-01 to 2026-06-30, rows that all "
    "five systems share. Points of capacity; smaller is better."
)
COLOUR = alt.Color(
    "arm:N",
    title=None,
    scale=alt.Scale(domain=NAMES, range=COLOURS),
    legend=alt.Legend(orient="top", columns=3, labelLimit=400),
)


def dot_chart(*, group: str, group_text: str, bars: bool = False) -> alt.VConcatChart:
    """Return the two-section chart (by lead time, by condition) of one group of Primaries.

    Each system is a dot, or with `bars` a thin horizontal bar in its own lane of the row.
    """
    row_px = 50 if bars else 22
    scores = pl.read_csv(SCORES / "scores.csv").filter(
        (pl.col("group") == group) & pl.col("arm").is_in(list(ARMS))
    )
    scores = scores.with_columns(arm=pl.col("arm").replace(ARMS))
    sections = []
    for title, slices in (("By forecast horizon (lead time)", LEAD_SLICES), ("By condition", CONDITION_SLICES)):
        data = scores.filter(pl.col("slice").is_in(list(slices))).with_columns(
            label=pl.col("slice").replace(slices)
        )
        order = list(slices.values())
        panels = []
        for metric, metric_name in METRICS.items():
            frame = data.filter(pl.col("metric") == metric).to_pandas()
            mark = alt.Chart(frame).mark_bar(height=7) if bars else alt.Chart(frame).mark_circle(size=80, opacity=0.9)
            lane = {"yOffset": alt.YOffset("arm:N", sort=NAMES)} if bars else {}
            panels.append(
                mark.encode(
                    **lane,
                    y=alt.Y(
                        "label:N",
                        sort=order,
                        title=None,
                        axis=alt.Axis(labels=metric == "mae_median", ticks=False, domain=False),
                    ),
                    x=alt.X("mean:Q", title=f"{metric_name} (points of capacity)", scale=alt.Scale(zero=True)),
                    color=COLOUR,
                )
                .properties(width=170, height=row_px * len(slices), title=metric_name)
            )
        counts = (
            data.filter((pl.col("metric") == "mae_median") & (pl.col("arm") == "GAM"))
            .with_columns(text=pl.col("half_hours").map_elements(lambda n: f"{n:,}", return_dtype=pl.String))
            .to_pandas()
        )
        panels.append(
            alt.Chart(counts)
            .mark_text(align="left")
            .encode(
                y=alt.Y("label:N", sort=order, title=None, axis=None),
                text="text:N",
            )
            .properties(width=60, height=row_px * len(slices), title="Half-hours")
        )
        sections.append(
            alt.hconcat(*panels).resolve_scale(x="independent").properties(title=alt.TitleParams(title, anchor="start", fontSize=14))
        )
    return alt.vconcat(*sections).properties(
        title=alt.TitleParams(
            [f"Error of five forecasts at {group_text}.", SUBTITLE], anchor="start", fontSize=12
        )
    )


def per_day_rows() -> pl.DataFrame:
    """Return the per-row errors of the five systems, with the lead day and the capacity unit."""
    units = pl.read_parquet(CAPACITY).select("time_series_id", "power_unit")
    return (
        pl.scan_parquet(SCORES / "rows.parquet")
        .filter(pl.col("arm").is_in(list(ARMS)))
        .select("time_series_id", "label", "arm", "power_fcst_init_time", "valid_time", "mae_median")
        .with_columns(
            arm=pl.col("arm").replace(ARMS),
            day=((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() // 24)
            .cast(pl.Int32)
            + 0.5,
        )
        .collect()
        .join(units.with_columns(pl.col("time_series_id").cast(pl.Int32)), on="time_series_id")
    )


def encode(*, domain: list[float] | None = None) -> dict:
    """Return the shared encoding of the error-by-horizon line charts."""
    return {
        "x": alt.X(
            "day:Q",
            title="Forecast horizon (days after the forecast origin)",
            axis=alt.Axis(values=[0, 2, 4, 6, 8, 10, 12, 14]),
            scale=alt.Scale(domain=[0, 14]),
        ),
        "y": alt.Y(
            "mae:Q",
            title="MAE of the median forecast (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False) if domain is None else alt.Scale(domain=domain),
        ),
        "color": alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(domain=NAMES, range=COLOURS),
            legend=alt.Legend(orient="top", columns=3, symbolType="stroke", symbolStrokeWidth=3, labelLimit=400),
        ),
        "strokeDash": alt.StrokeDash("arm:N", legend=None, scale=alt.Scale(domain=NAMES, range=DASH)),
    }


def by_day(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the mean absolute error of the median forecast by horizon day (and `by`)."""
    return frame.group_by(*by, "arm", "day").agg(mae=pl.col("mae_median").mean()).sort(*by, "arm", "day")


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG and as a PNG at twice the size."""
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / f"{name}.svg")
    chart.save(OUT / f"{name}.png", scale_factor=2)


def main() -> None:
    """Draw every chart."""
    for group, text in GROUPS.items():
        stem = f"{group.split()[0].lower()}_{group.split()[1]}"
        save(dot_chart(group=group, group_text=text), f"dots_{stem}")
        save(dot_chart(group=group, group_text=text, bars=True), f"bars_{stem}")
    frame = per_day_rows()
    mva = frame.filter(pl.col("power_unit") == "MVA")
    mw = frame.filter(pl.col("power_unit") == "MW")
    groups = {
        "all": (f"All {frame['label'].n_unique()} Primaries with a GAM forecast", frame),
        "mva": (f"The {mva['label'].n_unique()} Primaries that report in MVA", mva),
        "mw": (f"The {mw['label'].n_unique()} Primaries that report in MW (D15 has no GAM forecast yet)", mw),
    }
    for key, (title, part) in groups.items():
        chart = (
            alt.Chart(by_day(part, []).to_pandas())
            .mark_line(point=True, strokeWidth=2.5, clip=True)
            .encode(**encode())
            .properties(
                width=620,
                height=320,
                title=alt.TitleParams([title + ".", SUBTITLE.replace(" Points of capacity; smaller is better.", "")], anchor="start", fontSize=12),
            )
        )
        save(chart, f"mae_by_day_{key}")
    per = by_day(frame, ["label", "power_unit"]).with_columns(
        name=pl.col("label") + pl.lit(" (") + pl.col("power_unit") + pl.lit(")")
    )
    order = sorted(per["name"].unique(), key=lambda n: (n.split("(")[1], int(n.split(" ")[0][1:])))
    chart = (
        alt.Chart(per.to_pandas())
        .mark_line(strokeWidth=1.6, clip=True)
        .encode(
            **{**encode(), "x": alt.X("day:Q", title=None, axis=alt.Axis(values=[0, 7, 14]), scale=alt.Scale(domain=[0, 14])), "y": alt.Y("mae:Q", title=None, scale=alt.Scale(zero=False))},
            facet=alt.Facet("name:N", columns=4, sort=order, title=None),
        )
        .resolve_scale(y="independent")
        .properties(
            width=190,
            height=130,
            title=alt.TitleParams(
                [
                    "Each Primary on its own: MAE of the median forecast by horizon.",
                    "x: forecast horizon in days; y: MAE in points of capacity (own range in each panel). Provisional: origins at 12:00 UTC, 2025-11 to 2026-06.",
                ],
                anchor="start",
                fontSize=12,
            ),
        )
    )
    save(chart, "mae_by_day_per_primary")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
