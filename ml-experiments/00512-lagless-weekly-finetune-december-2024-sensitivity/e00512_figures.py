# ruff: noqa: E501
"""Draw experiment 512's figure: weekly-over-monthly and recency-weight CRPS differences.

```bash
source <paths>; uv run --no-sync python ml-experiments/00512-*/e00512_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00512_score as score  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00512-lagless-weekly-finetune-december-2024-sensitivity"
)
ORDER: Final[list[str]] = [
    "all", "christmas", "origin_days_1_7", "origin_days_8_14", "origin_days_15_21", "origin_days_22_28", "origin_days_29_31",
]  # fmt: skip
LABELS: Final[dict[str, str]] = {
    "all": "Whole month", "christmas": "Christmas week (valid 22-31 Dec)", "origin_days_1_7": "Origins 1-7 Dec",
    "origin_days_8_14": "Origins 8-14 Dec", "origin_days_15_21": "Origins 15-21 Dec", "origin_days_22_28": "Origins 22-28 Dec", "origin_days_29_31": "Origins 29-31 Dec",
}  # fmt: skip


def main() -> None:
    """Save the figure."""
    t = pl.read_csv(score.WORK_DIR / "scores" / "slices.csv").filter(
        pl.col("kind").str.contains(" - ")
    )
    t = t.with_columns(
        label=pl.col("slice").replace(LABELS),
        group=pl.col("group").replace({"eight": "MVA-only eight", "six": "Signed-MW six"}),
    )
    labels = [LABELS[s] for s in ORDER]
    panels = []
    for g in ("MVA-only eight", "Signed-MW six"):
        part = t.filter(pl.col("group") == g).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("label:N", sort=labels, title=None), yOffset=alt.YOffset("kind:N"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["L-W - L-M", "L-Ww - L-W"], range=[theme.DATA_BLUE, theme.DATA_GREEN]), title=None),
        )  # fmt: skip
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="CRPS difference (points of capacity; lower is better)"),
            x2="high:Q",
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=560, height=190, title=alt.TitleParams(g, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Lag-free arms, fold 2024-12, all rows (no event filter).",
                "Dot: difference; bar: 95% origin-day block bootstrap, uncorrected.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS_DIR / "figure_1_weekly_and_recency_december_2024.svg")


if __name__ == "__main__":
    main()
