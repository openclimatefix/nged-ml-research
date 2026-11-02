# ruff: noqa: E501
"""Draw the figures of experiment 515 from the per-row metrics that `e00515_score.py rows` stored.

1. `figure_1_headline.svg`: each arm minus W0 on cold nights and on all rows, Bonferroni-20 interval;
2. `figure_2_per_fold.svg`: each arm minus W0 on cold nights, one dot per fold;
3. `figure_3_leave_out_2024_11.svg`: the same contrasts without fold 2024-11 (post hoc).

Also writes `posthoc_leave_out_2024_11.csv` beside the scores.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00515_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00515-lagless-demand-network-trailing-weather-window-features"
)
ARMS: Final[tuple[str, ...]] = ("W-T", "W-S", "W-X", "W-null", "W-ALL", "W-ALLnoDH")
COLOURS: Final[dict[str, str]] = {
    "W-T": theme.DATA_BLUE, "W-S": theme.DATA_GREEN, "W-X": theme.DATA_PURPLE,
    "W-null": "#808080", "W-ALL": theme.BRAND_ORANGE, "W-ALLnoDH": theme.DATA_SKY,
}  # fmt: skip
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final[int] = 560
SLICES: Final[dict[str, str]] = {"cold_night": "cold nights", "all": "all rows"}


def _contrasts(*, frame: pl.DataFrame) -> pl.DataFrame:
    arms = [a for a in ARMS if f"{a}__crps" in frame.columns]
    records = [
        score.contrast(frame=frame, arm=a, ref="W0", group=g, slice_name=s)
        for a in arms for g in GROUPS for s in SLICES
    ]  # fmt: skip
    return pl.DataFrame(records)


def _forest(*, table: pl.DataFrame, title: list[str], file: str) -> None:
    panels = []
    for g, label in GROUPS.items():
        part = table.filter(pl.col("group") == g).with_columns(
            slice=pl.col("slice").replace(SLICES)
        ).to_pandas()  # fmt: skip
        base = alt.Chart(part).encode(
            y=alt.Y("arm:N", sort=list(ARMS), title=None), yOffset="slice:N"
        )  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="CRPS minus W0 (points of capacity; lower is better)"), x2="hi:Q",
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(ARMS), range=[COLOURS[a] for a in ARMS]), legend=None),
            opacity=alt.Opacity("slice:N", scale=alt.Scale(domain=list(SLICES.values()), range=[1.0, 0.4]), legend=alt.Legend(title=None)),
        )  # fmt: skip
        dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x="contrast:Q",
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(ARMS), range=[COLOURS[a] for a in ARMS]), legend=None),
            opacity=alt.Opacity("slice:N", scale=alt.Scale(domain=list(SLICES.values()), range=[1.0, 0.4]), legend=None),
        )  # fmt: skip
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start")).save(
        DOCS_DIR / file
    )


def _per_fold(*, frame: pl.DataFrame) -> None:
    records = []
    for arm in [a for a in ARMS if f"{a}__crps" in frame.columns]:
        for g in GROUPS:
            mask = score.mask_of(frame=frame, group=g, slice_name="cold_night")
            a = score.fold_sums(frame=frame, column=f"{arm}__crps", mask=mask)
            b = score.fold_sums(frame=frame, column="W0__crps", mask=mask)
            records.extend({"arm": arm, "group": GROUPS[g], "fold": m, "contrast": (a[m][0] - b[m][0]) / a[m][1]} for m in sorted(a) if a[m][1])  # fmt: skip
    table = pl.DataFrame(records)
    table.write_csv(score.SCORE_DIR / "per_fold_cold_night.csv")
    panels = []
    for label in GROUPS.values():
        part = table.filter(pl.col("group") == label).to_pandas()
        base = alt.Chart(part).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("fold:N", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("contrast:Q", title="CRPS minus W0, cold nights (points)"),
            color=alt.Color("arm:N", scale=alt.Scale(domain=list(ARMS), range=[COLOURS[a] for a in ARMS])),
        )  # fmt: skip
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(y="y:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, base.mark_point(filled=True, size=55)).properties(width=WIDTH, height=200, title=alt.TitleParams(label, anchor="start")))  # fmt: skip  # ty: ignore[invalid-argument-type]
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Cold-night CRPS minus W0 in each fold (negative favours the arm).",
                "Fold 2024-11 has the weakest W0 and the largest gains for every arm, the placebo W-null included.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_2_per_fold.svg")


def main() -> None:
    """Draw the three figures."""
    frame = score.load_rows()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    full = _contrasts(frame=frame)
    _forest(
        table=full, file="figure_1_headline.svg",
        title=["Each arm against W0 over 10 winter folds, rows with no labelled event.", "Dot: pooled difference; bar: Bonferroni-20 interval over fold blocks."],
    )  # fmt: skip
    _per_fold(frame=frame)
    left_out = _contrasts(frame=frame.filter(pl.col("month") != "2024-11"))
    left_out.write_csv(score.SCORE_DIR / "posthoc_leave_out_2024_11.csv")
    _forest(
        table=left_out, file="figure_3_leave_out_2024_11.svg",
        title=["Post hoc: the same contrasts over the 9 folds without 2024-11.", "Dot: pooled difference; bar: Bonferroni-20 interval over fold blocks."],
    )  # fmt: skip


if __name__ == "__main__":
    main()
