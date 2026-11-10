"""Figures of experiment 689: share of affected cells inside 0.90 to 1.10, and ratio scatters.

DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00689-*/e00689_figures.py
"""

from pathlib import Path

import altair as alt
import polars as pl
from e00689_boundary import OUT

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments"
FOLDER = next(DOCS.glob("00689-*"))
COLOURS = {
    "0 stored": "#7F7F7F",
    "A 90 days": "#306BFF",
    "B per seed, averaged": "#FF4901",
    "B-cross": "#17E58F",
    "C 45 days": "#F9B233",
    "C 180 days": "#9B59B6",
}
LABELS = {"0 stored": "stored", "B per seed, averaged": "B (per seed)", "A 90 days": "A (90 days)"}


def share_chart() -> None:
    """Headline: share of affected cells whose ratio lies in 0.90 to 1.10, by arm."""
    s = pl.read_csv(OUT / "summary.csv").filter(pl.col("lo").is_not_null())
    s = s.filter(pl.col("arm") != "0 seed mean of s0 and s1").with_columns(
        arm=pl.col("arm").replace(LABELS),
        share=pl.col("share") * 100,
        lo=pl.col("lo") * 100,
        hi=pl.col("hi") * 100,
    )
    base = alt.Chart(s.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("arm:N", title=None, sort=None),
    )
    bars = base.mark_bar(size=14, color="#306BFF").encode(
        x=alt.X(
            "share:Q",
            title="Affected cells with a ratio of 0.90 to 1.10 (%)",
            scale=alt.Scale(domain=[0, 100]),
        )
    )
    err = base.mark_rule(strokeWidth=1, color="black").encode(x="lo:Q", x2="hi:Q")
    line = (
        alt.Chart(pl.DataFrame({"x": [90.0]}).to_pandas())
        .mark_rule(  # ty: ignore[unresolved-attribute]
            strokeWidth=1, strokeDash=[4, 3], color="#FF4901"
        )
        .encode(x="x:Q")
    )
    (bars + err + line).properties(width=620, height=220).save(
        FOLDER / "share_inside_ten_percent.svg"
    )


def scatter() -> None:
    """Calibrated ratio against stored ratio for the affected cells, one panel per lead band."""
    r = pl.read_csv(OUT / "ratios.csv")
    aff = pl.read_csv(OUT / "affected_cells.csv").select("time_series_id", "boundary", "band")
    key = ["time_series_id", "boundary", "band"]
    stored = r.filter(pl.col("arm") == "0 stored").select(*key, stored=pl.col("ratio"))
    use = r.filter(pl.col("arm").is_in(["A 90 days", "B per seed, averaged"]))
    d = (
        use.join(stored, on=key)
        .join(aff, on=key)
        .with_columns(
            band_name=pl.col("band").replace_strict(
                {0: "days 0 to 1", 1: "days 1 to 3", 2: "days 3 to 7", 3: "days 7 to 14"}
            ),
            arm=pl.col("arm").replace(LABELS),
        )
    )
    scale = alt.Scale(type="log", domain=[0.5, 4], base=2)
    pts = (
        alt.Chart()
        .mark_point(size=18, filled=True, opacity=0.6)
        .encode(
            x=alt.X("stored:Q", title="Ratio after / before, stored", scale=scale),
            y=alt.Y("ratio:Q", title="Ratio after / before, calibrated", scale=scale),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["A (90 days)", "B (per seed)"], range=["#306BFF", "#FF4901"]
                ),
            ),
        )
    )
    box = (
        alt.Chart(pl.DataFrame({"a": [0.5], "b": [4.0], "lo": [0.9], "hi": [1.1]}).to_pandas())
        .mark_rect(  # ty: ignore[unresolved-attribute]
            color="#D9D9D9", opacity=0.6
        )
        .encode(x="a:Q", x2="b:Q", y="lo:Q", y2="hi:Q")
    )
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.5, 4.0]}).to_pandas())
        .mark_line(  # ty: ignore[unresolved-attribute]
            strokeWidth=1, color="black"
        )
        .encode(x="x:Q", y="x:Q")
    )
    panel = (box + diag + pts).properties(width=300, height=300)
    chart = (
        alt.layer(box, diag, pts, data=d.to_pandas())
        .properties(width=300, height=300)
        .facet(  # ty: ignore[unresolved-attribute]
            facet=alt.Facet(
                "band_name:N",
                title=None,
                sort=["days 0 to 1", "days 1 to 3", "days 3 to 7", "days 7 to 14"],
            ),
            columns=2,
        )
    )
    chart.save(FOLDER / "ratio_scatter.svg")
    _ = panel


if __name__ == "__main__":
    share_chart()
    scatter()
