"""Draw experiment 444's published-safe figures (rates and shares only) as SVG.

    uv run --no-sync python -I \
        ml-experiments/00444-wind-idle-detectors-label-free-bake-off/e00444_charts.py

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`. No chart shows
a raw farm-day, a calendar date, or a capacity; farms are Farm A, Farm B, and Farm C.
"""

import os
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

DATA_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments"
    / "00444-wind-idle-detectors-label-free-bake-off"
)
OUT_DIR: Final[Path] = Path("docs/ml-experiments/00444-wind-idle-detectors-label-free-bake-off")
FARMS: Final[dict[str, str]] = {"W1": "Farm A", "W2": "Farm B", "W3": "Farm C"}
LABELS: Final[dict[str, str]] = {
    "R0": "Experiment 210 mask",
    "R1": "Power level only",
    "D1": "D1 texture",
    "D4": "D4 texture, smoothed",
    "D3": "D3 residual",
    "D1&D3": "D1 and D3 together",
}
COLOURS: Final[dict[str, str]] = {
    "Experiment 210 mask": "#7F7F7F",
    "Power level only": "#BFBFBF",
    "D1 texture": ocf.DATA_BLUE,
    "D4 texture, smoothed": ocf.DATA_SKY,
    "D3 residual": ocf.BRAND_ORANGE,
    "D1 and D3 together": ocf.DATA_PURPLE,
}
SCALE: Final[alt.Scale] = alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))
WIDTH: Final[int] = 480


def prep(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the plotted candidates and give them farm and candidate labels."""
    return frame.filter(pl.col("cand").is_in(list(LABELS))).with_columns(
        farm=pl.col("site").replace_strict(FARMS), candidate=pl.col("cand").replace_strict(LABELS)
    )


def figure_headline() -> None:
    """Recall on injected idle blocks against false-drop rate, with the two gate lines."""
    gate = prep(pl.read_parquet(DATA_DIR / "gate.parquet"))
    base = alt.Chart(gate).encode(color=alt.Color("candidate:N", scale=SCALE, title=None))  # ty: ignore[unresolved-attribute]
    layers = (
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("fd_gate:Q", title="False-drop rate on genuine low output"),
            y=alt.Y("recall_lo:Q", title="Recall of injected idle blocks"),
            y2="recall_hi:Q",
        )
        + base.mark_point(filled=True, size=70, aria=False).encode(x="fd_gate:Q", y="recall:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.05]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"y": [0.8]}))
        .mark_rule(strokeDash=[4, 3], color="grey")
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=200, height=220)
    layers.facet(column=alt.Column("farm:N", title=None)).save(OUT_DIR / "figure_1_headline.svg")  # ty: ignore[unresolved-attribute]


def figure_by_length() -> None:
    """Recall by length of injected block."""
    cells = (
        prep(pl.read_parquet(DATA_DIR / "cells.parquet"))
        .group_by("farm", "candidate", "length")
        .agg(recall=pl.col("recall").mean())
        .with_columns(hours=pl.col("length") / 2)
    )
    alt.Chart(
        cells.filter(
            pl.col("candidate").is_in(["D1 texture", "D3 residual", "Experiment 210 mask"])
        )
    ).mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hours:O", title="Length of injected idle block (hours)"),
        y=alt.Y("recall:Q", title="Recall"),
        color=alt.Color("candidate:N", scale=SCALE, title=None),
    ).properties(width=200, height=200).facet(column=alt.Column("farm:N", title=None)).save(
        OUT_DIR / "figure_2_recall_by_length.svg"
    )  # ty: ignore[unresolved-attribute]


def figure_false_drop() -> None:
    """False-drop rate on each set of genuine low output."""
    gate = prep(pl.read_parquet(DATA_DIR / "gate.parquet"))
    long = (
        gate.unpivot(
            on=["fd_SG", "fd_SGm", "fd_G0s"],
            index=["farm", "candidate"],
            variable_name="set",
            value_name="rate",
        )
        .with_columns(
            set=pl.col("set").replace_strict(
                {
                    "fd_SG": "Real, 3-8% of capacity, wind 6+ m/s",
                    "fd_SGm": "Real, 3-8% of capacity, wind below 6 m/s",
                    "fd_G0s": "Synthetic, rescaled",
                }
            )
        )
        .filter(pl.col("rate").is_not_nan())
    )
    alt.Chart(long).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("rate:Q", title="Share flagged as idle"),
        y=alt.Y("candidate:N", title=None, sort=list(COLOURS)),
        color=alt.Color("candidate:N", scale=SCALE, legend=None),
        row=alt.Row("set:N", title=None),
        column=alt.Column("farm:N", title=None),
    ).properties(width=160, height=90).save(OUT_DIR / "figure_3_false_drop.svg")


def figure_curve() -> None:
    """Share of real half-hours flagged, by ERA5 wind speed."""
    curve = prep(pl.read_parquet(DATA_DIR / "cut_in_curve.parquet")).filter(
        (pl.col("n") >= 200)
        & (pl.col("wind_bin") <= 20)
        & pl.col("candidate").is_in(["D1 texture", "D3 residual", "Experiment 210 mask"])
    )
    alt.Chart(curve).mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("wind_bin:Q", title="ERA5 100 m wind speed (m/s)"),
        y=alt.Y("share:Q", title="Share of half-hours flagged"),
        color=alt.Color("candidate:N", scale=SCALE, title=None),
    ).properties(width=200, height=200).facet(column=alt.Column("farm:N", title=None)).save(
        OUT_DIR / "figure_4_flagged_by_wind.svg"
    )  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_headline()
    figure_by_length()
    figure_false_drop()
    figure_curve()
