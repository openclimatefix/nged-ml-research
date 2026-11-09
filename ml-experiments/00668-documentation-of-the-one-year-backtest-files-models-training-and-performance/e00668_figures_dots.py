"""Dot plots of the protocol comparison: MAE, CRPS, and twCRPS side by side, one dot per arm.

uv run --no-sync python ml-experiments/00668-*/e00668_figures_dots.py OUT_DIR

Reads `scores_668/primaries_compare/scores.csv` (`e00668_compare_primaries.py`). One figure per
Primary group and block: "by forecast horizon" (day 1, day 2, days 3 to 4, days 5 to 7, days 8 to
10) and "by condition" (all days, winter, winter nights, cold nights, Economy 7 step, Christmas,
bank holidays, sunny). A column of half-hour counts is printed on the right of each row.
The units are points of capacity; the MVA-only group is in MVA, as apparent power.
"""

import sys
from pathlib import Path

import altair as alt
import e00668_score_lib as lib
import polars as pl

ARMS = {
    "backtest_NRA": ("Backtest, normal running", "#306BFF"),
    "backtest_prevailing": ("Backtest, prevailing", "#B701FF"),
    "default_3": ("Research default 3", "#17E58F"),
    "bg2": ("BG2", "#FC9700"),
    "manual_heuristic": ("Manual heuristic", "#FF4901"),
    "climatology": ("Climatology", "#7F7F7F"),
}
METRICS = {"mae_median": "MAE of the median", "crps": "CRPS", "twcrps": "twCRPS"}
HORIZON = {
    "lead_day_1": "Day 1",
    "lead_day_2": "Day 2",
    "lead_days_3_4": "Days 3 to 4",
    "lead_days_5_7": "Days 5 to 7",
    "lead_days_8_10": "Days 8 to 10",
}
CONDITION = {
    "all": "All days",
    "winter": "Winter (Dec to Feb)",
    "winter_nights": "Winter nights",
    "cold_nights": "Cold nights (3 °C or below)",
    "economy_7_step": "Economy 7 step hours",
    "christmas": "Christmas (24 Dec to 1 Jan)",
    "bank_holidays": "Bank holidays",
    "sunny_unit_shape": "Sunny half-hours",
}
GROUPS = {"signed-MW six": "six MW Primaries", "MVA-only seven": "seven MVA Primaries"}


def block(
    *, scores: pl.DataFrame, group: str, rows: dict[str, str], title: str, unit: str
) -> alt.Chart:
    """Return one block of dot plots: a column per metric, a row per slice."""
    frame = (
        scores.filter(
            (pl.col("group") == group)
            & pl.col("slice").is_in(list(rows))
            & pl.col("arm").is_in(list(ARMS))
        )
        .with_columns(
            row=pl.col("slice").replace(rows),
            system=pl.col("arm").replace({k: v[0] for k, v in ARMS.items()}),
            metric_name=pl.col("metric").replace(METRICS),
        )
        .to_pandas()
    )
    scale = alt.Scale(domain=[v[0] for v in ARMS.values()], range=[v[1] for v in ARMS.values()])
    order = list(rows.values())
    columns = []
    for metric, name in METRICS.items():
        sub = frame[frame["metric"] == metric]
        base = alt.Chart(sub).encode(
            y=alt.Y(
                "row:N",
                sort=order,
                title=None,
                axis=alt.Axis(labels=metric == "mae_median", labelLimit=190),
            ),
        )
        dots = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("mean:Q", title=f"{name} ({unit})", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "system:N",
                scale=scale,
                title=None,
                legend=alt.Legend(orient="top", columns=3) if metric == "mae_median" else None,
            ),
        )
        columns.append(dots.properties(width=280, height=24 * len(rows) + 30))
    counts = (
        alt.Chart(frame[(frame["metric"] == "crps") & (frame["arm"] == "default_3")])  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", fontSize=11)
        .encode(
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labels=False)),
            text=alt.Text("half_hours:Q", format=","),
        )
        .properties(width=70, height=24 * len(rows) + 30, title="Half-hours")
    )
    return alt.hconcat(*columns, counts).properties(  # ty: ignore[invalid-return-type]
        title=alt.TitleParams(title, anchor="start", fontSize=13)
    )


def main(out: Path) -> None:
    """Draw four figures: two blocks for each of the two Primary groups."""
    scores = pl.read_csv(lib.FOLDER / "scores_668" / "primaries_compare" / "scores.csv")
    out.mkdir(parents=True, exist_ok=True)
    for group, label in GROUPS.items():
        unit = "points of capacity"
        for tag, rows, name in (
            ("horizon", HORIZON, "forecast horizon"),
            ("condition", CONDITION, "condition"),
        ):
            title = f"Error by {name}, {label}: one dot per forecast, common rows"
            block(scores=scores, group=group, rows=rows, title=title, unit=unit).save(
                out / f"dots_{'mw' if group.startswith('signed') else 'mva'}_{tag}.svg"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
