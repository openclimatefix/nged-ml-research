"""Experiment 507: the two figures of the write-up, from the contrast tables the scoring step wrote.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_report.py
```
"""

import re
from pathlib import Path

import altair as alt
import e00507_arms as arms  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / Path(__file__).parent.name
GROUP_LABEL = {"six": "Signed-MW six", "eight": "MVA-only eight"}


def gains() -> pl.DataFrame:
    """Return the oracle and TV gain over the plain correction on step rows, with intervals."""
    old = (
        pl.read_csv(arms.SCORE_DIR / "old_contrasts.csv")
        .filter((pl.col("rows") == "step rows") & (pl.col("arm") == "oracle"))
        .with_columns(pl.lit("Old worlds (experiment 428)").alias("worlds"))
    )
    new = (
        pl.read_csv(arms.SCORE_DIR / "new_contrasts.csv")
        .filter((pl.col("rows") == "step (generator mix)") & pl.col("arm").is_in(["oracle", "tv"]))
        .with_columns(pl.lit("New generator worlds").alias("worlds"))
    )
    both = pl.concat([old, new], how="diagonal_relaxed")
    return both.with_columns(
        pl.col("group").replace(GROUP_LABEL).alias("group"),
        pl.col("arm").replace({"oracle": "Oracle repair", "tv": "Total-variation repair"}),
    )


def figure_gains() -> None:
    """Save figure 1: the change in CRPS against the plain correction, with 95% intervals."""
    data = gains()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "worlds:N", title=None, sort=["Old worlds (experiment 428)", "New generator worlds"]
        ),
        yOffset="arm:N",
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(
                domain=["Oracle repair", "Total-variation repair"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
        ),
    )
    chart = (
        (
            base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "lo95:Q",
                    title="Change in CRPS vs plain correction (points; lower is better)",
                ),
                x2="hi95:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        )
        .facet(column=alt.Column("group:N", title=None))
        .properties(title="Step rows")
    )
    chart.save(DOCS / "figure_1_gain_over_plain.svg")


def figure_tuning() -> None:
    """Save figure 2: the tuning error of each total-variation setting."""
    path = next(Path(arms.SCORE_DIR).parent.glob("tv_tuning.txt"))
    rows = []
    unrepaired = float("nan")
    for line in path.read_text().splitlines():
        if m := re.match(r"unrepaired error ([\d.]+)", line):
            unrepaired = float(m.group(1))
        if m := re.match(r"weight\s+([\d.]+) tau ([\d.]+): error ([\d.]+)", line):
            rows.append(
                {
                    "weight": float(m.group(1)),
                    "tau": f"tau {m.group(2)}",
                    "error": float(m.group(3)),
                }
            )
    data = pl.DataFrame(rows)
    lines = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "weight:Q", scale=alt.Scale(type="log"), title="Total-variation weight (spreads)"
            ),
            y=alt.Y("error:Q", title="Mean absolute lag error (spreads)"),
            color=alt.Color("tau:N", title=None),
        )
    )
    ref = (
        alt.Chart(pl.DataFrame({"error": [unrepaired]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(y="error:Q")  # ty: ignore[unresolved-attribute]
    )
    (lines + ref).properties(title="Every repair setting is worse than no repair (dashed)").save(
        DOCS / "figure_2_tv_tuning.svg"
    )


if __name__ == "__main__":
    figure_gains()
    figure_tuning()
