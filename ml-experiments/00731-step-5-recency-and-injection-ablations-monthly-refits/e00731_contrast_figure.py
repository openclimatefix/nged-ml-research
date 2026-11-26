"""Experiment 731: the contrast figure (SVG) from the scorer's `contrasts.csv`.

README. This script draws `contrasts.svg`, one panel per group of Primaries, showing each planned
contrast (arm minus R0, the replay of arm 9a) of the normalised MAE of the median forecast with its
99.5% interval. It reads `scores/summary/contrasts.csv`, which `e00731_score.py` wrote, and needs no
PYTHONPATH beyond the repository's own environment. Set `DATA_PATH_INTERNAL`.
Usage: `python e00731_contrast_figure.py [--out DIR]`.
"""

import argparse
import os
from pathlib import Path
from typing import Final

import altair as alt
import polars as pl

NAME: Final = "00731-step-5-recency-and-injection-ablations-monthly-refits"
BLUE: Final = "#306BFF"
ORANGE: Final = "#FF4901"
LABELS: Final = {
    "C1": "C1 no recency weights (W0)",
    "C2": "C2 half-life 30 days (W30)",
    "C3": "C3 half-life 270 days (W270)",
    "C4": "C4 no injected events (stored 5wo)",
    "C5": "C5 neither (B0)",
    "C6": "C6 placebo, ages permuted (WP)",
    "C7": "C7 placebo, zero-size shifts (IP)",
}
GROUPS: Final = ("six signed-MW", "eight MVA-only")


def main() -> None:
    """Draw the contrast figure."""
    parser = argparse.ArgumentParser()
    root = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments" / NAME
    parser.add_argument("--out", type=Path, default=root / "figures")
    args = parser.parse_args()
    contrasts = pl.read_csv(root / "scores" / "summary" / "contrasts.csv").filter(
        (pl.col("kind") == "primary")
        & (pl.col("metric") == "mae")
        & (pl.col("setting") == "joint")
        & pl.col("contrast").is_in(list(LABELS))
        & pl.col("group").is_in(list(GROUPS))
    )
    frame = contrasts.with_columns(
        label=pl.col("contrast").replace_strict(LABELS),
        group_label=pl.col("group").replace_strict(
            {
                "six signed-MW": "Six signed-MW Primaries",
                "eight MVA-only": "Eight MVA-only Primaries",
            }
        ),
        placebo=pl.col("contrast").is_in(["C6", "C7"]),
    )
    order = list(LABELS.values())
    colour = alt.Color(
        "placebo:N", scale=alt.Scale(domain=[False, True], range=[BLUE, ORANGE]), legend=None
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    panels = []
    for group_label in frame["group_label"].unique(maintain_order=True):
        base = alt.Chart(frame.filter(pl.col("group_label") == group_label))
        bars = base.mark_rule().encode(
            x=alt.X("lo:Q", title=None),
            x2="hi:Q",
            y=alt.Y("label:N", sort=order, title=None),
            color=colour,
        )
        points = base.mark_point(filled=True, size=70).encode(
            x=alt.X(
                "diff:Q",
                title="Arm minus R0, normalised MAE of the median (points of capacity)",
            ),
            y=alt.Y("label:N", sort=order, title=None),
            color=colour,
        )
        panels.append(
            alt.layer(bars, points, zero).properties(width=420, height=200, title=group_label)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Taking out the recency weights or the injected events, and two placebos",
            subtitle=[
                "Right of the dashed line is worse than arm 9a (lower error is better).",
                "Bars are 99.5% intervals; orange marks the two placebos.",
                "Joint pass of 3,855,253 half-hours, seeds 0 to 2 averaged.",
            ],
            anchor="start",
        )
    )
    args.out.mkdir(parents=True, exist_ok=True)
    chart.save(args.out / "contrasts.svg")
    print("wrote", args.out / "contrasts.svg")


if __name__ == "__main__":
    main()
