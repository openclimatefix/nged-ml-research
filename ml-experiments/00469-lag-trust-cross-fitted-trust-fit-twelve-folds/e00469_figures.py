"""Draw the figures of experiment 469 from the tables of `e00469_score.py` (no fit).

```bash
DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00469-*/e00469_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00469-lag-trust-cross-fitted-trust-fit-twelve-folds"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 430
REFERENCES: Final[dict[str, str]] = {
    "B-mix": "vs 50/50 mixture (B-mix)",
    "G1-trust": "vs 6-fold G1-trust",
    "K2-S1": "vs K2-S1",
}
SET_LABELS: Final[dict[tuple[str, str], str]] = {
    ("nolabel", "metered"): "rows with no labelled event",
    ("harm", "metered"): "harm rows",
    ("event", "1.0"): "inside real events (NRA-corrected actuals)",
}
GROUP_LABELS: Final[dict[str, str]] = {"signed_six": "Signed-MW six", "mva_eight": "MVA-only eight"}


def contrast_figure() -> None:
    """Draw G1-trust-12 minus each reference, by group and row set, with month intervals."""
    table = pl.read_csv(OUT / "pooled.csv", schema_overrides={"actual": pl.String})
    rows = [
        table.filter(
            (pl.col("a") == "G1-trust-12")
            & pl.col("b").is_in(list(REFERENCES))
            & (pl.col("set") == setname)
            & (pl.col("actual") == actual)
        ).with_columns(set_label=pl.lit(label))
        for (setname, actual), label in SET_LABELS.items()
    ]
    data = pl.concat(rows).with_columns(
        label=pl.col("b").replace(REFERENCES),
        panel=pl.col("group").replace(GROUP_LABELS) + pl.lit(", ") + pl.col("set_label"),
    )
    y = alt.Y("label:N", sort=list(REFERENCES.values()), title=None)
    base = alt.Chart()
    thin = base.mark_rule(strokeWidth=1, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low_bonf:Q",
            title="G1-trust-12 minus reference (points of capacity)",
            scale=alt.Scale(domain=[-1.0, 1.0], clamp=True),
        ),
        x2="high_bonf:Q",
    )
    thick = base.mark_rule(strokeWidth=4, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="low95:Q", x2="high95:Q"
    )
    dots = base.mark_point(filled=True, size=60, color=theme.BRAND_ORANGE).encode(y=y, x="mean:Q")  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    panels = [f"{GROUP_LABELS[g]}, {label}" for g in GROUP_LABELS for label in SET_LABELS.values()]
    chart = (
        alt.layer(zero, thin, thick, dots, data=data.to_pandas())
        .properties(width=WIDTH, height=90)
        .facet(
            row=alt.Row(
                "panel:N", title=None, sort=panels,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Cross-fitted trust with 12 folds (G1-trust-12) minus each reference",
                    "Left of the dashed line is better. Dot: mean of months.",
                    "Thick line: 95% interval. Thin line: Bonferroni interval for 24 contrasts.",
                ],
                anchor="start", fontSize=12, subtitleFontSize=10,
            )
        )
    )  # fmt: skip
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def weights_figure() -> None:
    """Draw the learned lagged weight by lead band for the 6-fold and 12-fold trust arms."""
    w = (
        pl.read_csv(OUT / "weights.csv")
        .filter(pl.col("arm").is_in(["G1-trust", "G1-trust-12"]))
        .with_columns(group_name=pl.col("group").replace(GROUP_LABELS))
    )
    bands = ["0-12h", "12-36h", "36-72h", "days4-5", "days6-10"]
    line = (
        alt.Chart(w.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("band:N", sort=bands, title="Lead band"),
            y=alt.Y(
                "weight:Q", title="Mean learned weight on the lagged head",
                scale=alt.Scale(domain=[0.0, 1.0]),
            ),
            color=alt.Color("month:N", title="Scored month"),
            strokeDash=alt.StrokeDash("arm:N", title=None),
        )
    )  # fmt: skip
    chart = (
        line.properties(width=WIDTH, height=170)
        .facet(row=alt.Row("group_name:N", title=None))
        .properties(
            title=alt.TitleParams(
                ["Learned weight on the lagged head: 6 folds against 12 folds",
                 "0.5 is the fixed 50/50 mixture."],
                anchor="start", fontSize=12, subtitleFontSize=10,
            )
        )
    )  # fmt: skip
    chart.save(FIGURE_DIR / "figure_2_learned_weights.svg")


if __name__ == "__main__":
    contrast_figure()
    weights_figure()
