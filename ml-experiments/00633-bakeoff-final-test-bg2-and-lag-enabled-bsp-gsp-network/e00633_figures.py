# ruff: noqa: D103, E501
"""Experiment 633, Part 1 figures: MAE and CRPS of six systems by forecast day at the 16 Primaries (CPU)."""

import sys
from pathlib import Path

import altair as alt
import polars as pl

COLOURS = {
    "BG2": "#FF4901",
    "K1": "#306BFF",
    "Candidate": "#10C5F7",
    "K3": "#B701FF",
    "Manual heuristic": "#17E58F",
    "Climatology": "#7F7F7F",
}


def chart(*, frame: pl.DataFrame, measure: str, title: list[str]) -> alt.Chart:
    long = frame.with_columns(day=pl.col("lead_band") + 1)
    return (
        alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "day:Q",
                title="Forecast day (days after the forecast origin)",
                axis=alt.Axis(values=list(range(1, 11))),
            ),
            y=alt.Y(
                f"{measure}:Q",
                title=f"{measure.upper()} (% of capacity; smaller is better)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "system:N",
                title=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", columns=3),
            ),
        )
        .properties(
            width=620, height=300, title=alt.TitleParams(title, anchor="start", fontSize=13)
        )
    )


def main(score_dir: Path, out: Path) -> None:
    frame = pl.read_csv(score_dir / "by_lead_band.csv")
    out.mkdir(parents=True, exist_ok=True)
    sub = [
        "16 Primaries, 123 forecast origins from 2026-07-01 to 2026-10-31, 929,870 half-hourly rows.",
        "Training data end 2026-06-30. Every system is scored once on the same rows.",
    ]
    for measure, name, lead in (
        ("mae", "mae_by_day", "Median-forecast error by forecast day"),
        ("crps", "crps_by_day", "CRPS by forecast day"),
    ):
        c = chart(frame=frame, measure=measure, title=[lead, *sub])
        c.save(out / f"{name}.svg")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
