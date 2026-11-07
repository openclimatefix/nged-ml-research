# ruff: noqa: D103, E501
"""Experiment 637 figures (CPU): read the window tables; write SVGs for the page.

Usage: python e00637_figures.py WINDOW_DIR OUT_DIR
"""

import sys
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

SYSTEMS = ["K1", "Candidate", "K3"]
COLOUR = {"K1": theme.DATA_BLUE, "Candidate": theme.DATA_GREEN, "K3": theme.DATA_PURPLE}
ORDER = ["all rows", "kept, stricter (k=4.0)", "kept, operating (k=3.5)", "kept, looser (k=3.0)"]


def main(window: Path, out: Path) -> None:
    scores = pl.read_csv(window / "scores.csv").filter(pl.col("system").is_in(SYSTEMS))
    base = alt.Chart(scores).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("reading:N", sort=ORDER, title=None),
        color=alt.Color(
            "system:N",
            scale=alt.Scale(domain=SYSTEMS, range=[COLOUR[s] for s in SYSTEMS]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "diff_lo:Q",
                title="CRPS minus the manual heuristic's CRPS (points of capacity; below 0 beats the heuristic)",
            ),
            x2="diff_hi:Q",
            yOffset="system:N",
        )
        + base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x="crps_minus_heuristic:Q", yOffset="system:N"
        )
        + zero
    ).properties(width=560, height=240)
    chart.save(str(out / "contrast_by_reading.svg"))
    z = pl.read_csv(window / "daily_z.csv", try_parse_dates=True)
    rules = (
        alt.Chart(pl.DataFrame({"y": [-3.5, 3.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#555", strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    panels = []
    for lab in ("D14", "D18"):
        line = (
            alt.Chart(z.filter(pl.col("primary") == lab))
            .mark_line(color=theme.DATA_BLUE, strokeWidth=1.5)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:T", title="Day (2026)"),
                y=alt.Y("z:Q", title=f"{lab}: daily residual (robust SD)"),
            )
        )
        panels.append(alt.layer(line, rules).properties(width=560, height=160))
    alt.vconcat(*panels).save(str(out / "daily_z_d14_d18.svg"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
