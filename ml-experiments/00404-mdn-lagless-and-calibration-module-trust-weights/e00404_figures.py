# ruff: noqa: E501
"""Draw the pooled-score figures of experiment 404 from the stored pooled tables (no fit).

- Figure 1: score minus B-mix for every arm on real harm rows, real clean rows, and rows inside real
  events, at the signed-MW six, with the month-block 95% interval.
- Figure 2: the same at the MVA-only eight.
- Figure 3: the per-cell difference from B-mix for the arms that come closest to it, one dot per
  month and seed, so the spread behind each interval is visible.
- Figure 4: the absolute score of every arm and of B-mix on the cells the arm has.

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_pool.py \
    --arms A0-inj B0 B1 B1o B3 B4c B6 B7 S1 --reference B-mix
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00404_fit as fit  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
)
ARM_ORDER: Final[list[str]] = ["A0-inj", "B0", "B1", "B1o", "B4c", "B6", "B7", "B3", "S1"]
ARM_LABELS: Final[dict[str, str]] = {
    "A0-inj": "A0-inj: plain network",
    "B0": "B0: constant lag weights",
    "B1": "B1: learned lag weights",
    "B1o": "B1o: B1 + true event flags",
    "B4c": "B4c: B1 + 5 weekly lags",
    "B6": "B6: B1 + learned windows",
    "B7": "B7: B1 + 7-day history",
    "B3": "B3: attention over lags",
    "S1": "S1: plain + lag surprises",
}
FAMILY: Final[dict[str, str]] = {
    "A0-inj": "plain network",
    "S1": "plain network",
    "B0": "constant weights",
    "B1": "learned weights",
    "B1o": "learned weights",
    "B4c": "learned weights",
    "B6": "learned weights",
    "B7": "learned weights",
    "B3": "attention",
}
FAMILY_COLOURS: Final[dict[str, str]] = {
    "plain network": theme.DATA_SKY,
    "constant weights": theme.DATA_PURPLE,
    "learned weights": theme.DATA_BLUE,
    "attention": theme.BRAND_ORANGE,
}
CLASS_TITLES: Final[dict[str, str]] = {
    "harm_real": "Real harm rows",
    "clean_real": "Real clean rows",
    "tgt_real": "Rows inside real events",
}
WIDTH: Final[int] = 680


def summary_frame(*, group: str) -> pl.DataFrame:
    """Return the pooled contrasts of a Primary group for the three real row classes."""
    path = fit.WORK_DIR / "pooled_summary_vs_B-mix.csv"
    frame = pl.read_csv(path).filter(
        (pl.col("group") == group) & pl.col("cls").is_in(list(CLASS_TITLES))
    )
    return frame.with_columns(
        label=pl.col("arm").replace(ARM_LABELS),
        family=pl.col("arm").replace(FAMILY),
        cls_title=pl.col("cls").replace(CLASS_TITLES),
    )


def figure_contrast(*, group: str, group_name: str, number: int, file_stem: str) -> None:
    """Draw the contrast figure of one Primary group."""
    data = summary_frame(group=group)
    labels = [ARM_LABELS[a] for a in ARM_ORDER]
    y = alt.Y("label:N", sort=labels, title=None)
    colour = alt.Color(
        "family:N",
        scale=alt.Scale(domain=list(FAMILY_COLOURS), range=list(FAMILY_COLOURS.values())),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart()
    rules = base.mark_rule(strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low:Q", title="Score minus B-mix (points of capacity)"), x2="high:Q",
        color=colour,
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="minus_ref:Q", color=colour
    )
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, rules, dots, data=data.to_pandas())
        .properties(width=WIDTH - 250, height=190)
        .facet(
            row=alt.Row(
                "cls_title:N",
                title=None,
                sort=list(CLASS_TITLES.values()),
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    f"Each arm minus B-mix at the {group_name}; left of the dashed line is better than B-mix",
                    "Dot: mean over months. Line: 95% interval from resampling whole months (wide: 5 months or fewer).",
                    "B-mix is a fixed 50/50 mixture of the plain network and the lagless table. Score: mass-weighted",
                    "pinball loss, points of capacity. Rows of real half-hours, 2025-10 to 2026-06; one fit per month.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / f"figure_{number}_{file_stem}.svg")


def figure_cells() -> None:
    """Figure 3: the per-cell difference from B-mix on real harm rows at the signed-MW six."""
    data = (
        pl.read_parquet(fit.WORK_DIR / "pooled_vs_B-mix.parquet")
        .filter(
            (pl.col("group") == "signed_six")
            & (pl.col("cls") == "harm_real")
            & pl.col("arm").is_in(["B0", "B1", "B3", "B6", "B4c"])
        )
        .with_columns(
            cell=pl.col("month") + pl.lit(" seed ") + pl.col("seed").cast(pl.String),
            label=pl.col("arm").replace(ARM_LABELS),
            family=pl.col("arm").replace(FAMILY),
        )
    )
    labels = [ARM_LABELS[a] for a in ARM_ORDER if a in {"B0", "B1", "B3", "B6", "B4c"}]
    chart = (
        alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60, aria=False)
        .encode(
            y=alt.Y("label:N", sort=labels, title=None),
            x=alt.X("diff:Q", title="Score minus B-mix on real harm rows (points of capacity)"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=list(FAMILY_COLOURS), range=list(FAMILY_COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape(
                "single_event:N",
                scale=alt.Scale(domain=[False, True], range=["circle", "diamond"]),
                legend=alt.Legend(title="Cell holds one real harm event", orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 250,
            height=180,
            title=alt.TitleParams(
                [
                    "One dot per month and seed: difference from B-mix on real harm rows, signed-MW six",
                    "Left of 0 is better than B-mix. Diamonds mark cells with a single real harm event.",
                ],
                anchor="start",
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.GREY_2, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    (zero + chart).save(FIGURE_DIR / "figure_3_per_cell_harm_rows.svg")


def figure_absolute() -> None:
    """Figure 4: the absolute pinball score of each arm and of B-mix on the arm's cells."""
    data = summary_frame(group="signed_six").select(
        "label", "family", "cls_title", "absolute", "reference_absolute"
    )
    long = pl.concat(
        [
            data.select("label", "cls_title", pl.col("absolute").alias("score")).with_columns(
                kind=pl.lit("Arm")
            ),
            data.select(
                "label", "cls_title", pl.col("reference_absolute").alias("score")
            ).with_columns(kind=pl.lit("B-mix, same cells")),
        ]
    )
    labels = [ARM_LABELS[a] for a in ARM_ORDER]
    chart = (
        alt.Chart(long.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, aria=False)
        .encode(
            y=alt.Y("label:N", sort=labels, title=None),
            x=alt.X(
                "score:Q",
                title="Mass-weighted pinball score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Arm", "B-mix, same cells"],
                    range=[theme.DATA_BLUE, theme.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=WIDTH - 250, height=190)
        .facet(
            row=alt.Row(
                "cls_title:N",
                title=None,
                sort=list(CLASS_TITLES.values()),
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Absolute score of every arm beside B-mix on the cells the arm has, signed-MW six",
                    "Cells differ between arms (B7 has one, B4c and B6 five, the rest seven), so compare an arm",
                    "only with the B-mix dot on its own row.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_4_absolute_scores.svg")


def main() -> None:
    """Draw all pooled-score figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrast(
        group="signed_six", group_name="signed-MW six", number=1, file_stem="signed_mw_six"
    )
    figure_contrast(
        group="mva_eight", group_name="MVA-only eight", number=2, file_stem="mva_only_eight"
    )
    figure_cells()
    figure_absolute()


if __name__ == "__main__":
    main()
