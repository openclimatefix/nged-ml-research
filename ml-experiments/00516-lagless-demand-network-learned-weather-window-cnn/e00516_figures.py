# ruff: noqa: E501
"""Draw the figures of experiment 516 from the contrasts that `e00516_score.py stats` stored.

1. `figure_1_headline.svg`: each CNN arm and the shuffled-day control minus W0 on cold nights and all rows;
2. `figure_2_against_references.svg`: C-CNN and T-CNN against the shuffled-day control and experiment 515's W-T;
3. `figure_3_per_fold.svg`: each arm minus W0 on cold nights, one dot per fold.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00516_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00516-lagless-demand-network-learned-weather-window-cnn"
)
ARMS: Final[tuple[str, ...]] = ("C-CNN", "T-CNN", "C-CNN-null")
COLOURS: Final[dict[str, str]] = {
    "C-CNN": theme.DATA_BLUE,
    "T-CNN": theme.DATA_GREEN,
    "C-CNN-null": "#808080",
}
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}
SLICES: Final[dict[str, str]] = {"cold_night": "cold nights", "all": "all rows"}
WIDTH: Final[int] = 560


def _forest(
    *, table: pl.DataFrame, arms: list[str], ref_label: str, title: list[str], file: str
) -> None:
    panels = []
    colour = alt.Color(
        "arm:N", scale=alt.Scale(domain=arms, range=[COLOURS[a] for a in arms]), legend=None
    )
    for g, label in GROUPS.items():
        part = table.filter(pl.col("group") == g).with_columns(slice=pl.col("slice").replace(SLICES)).to_pandas()  # fmt: skip
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=arms, title=None), yOffset="slice:N")  # ty: ignore[unresolved-attribute]
        opacity = alt.Opacity(
            "slice:N", scale=alt.Scale(domain=list(SLICES.values()), range=[1.0, 0.4])
        )
        bar = base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title=f"CRPS minus {ref_label} (points of capacity; lower is better)"), x2="hi:Q", color=colour, opacity=opacity.legend(alt.Legend(title=None)))  # fmt: skip  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=60).encode(x="contrast:Q", color=colour, opacity=opacity.legend(None))  # fmt: skip  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, bar, dot).properties(width=WIDTH, height=150, title=alt.TitleParams(label, anchor="start")))  # fmt: skip
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start")).save(
        DOCS_DIR / file
    )


def _table(*, contrasts: pl.DataFrame, ref: str, arms: tuple[str, ...]) -> pl.DataFrame:
    return contrasts.filter((pl.col("metric") == "crps") & (pl.col("ref") == ref) & pl.col("arm").is_in(arms) & pl.col("slice").is_in(list(SLICES)) & pl.col("group").is_in(list(GROUPS)))  # fmt: skip


def main() -> None:
    """Draw the three figures."""
    contrasts = pl.read_csv(score.SCORE_DIR / "contrasts.csv")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _forest(
        table=_table(contrasts=contrasts, ref="W0", arms=ARMS), arms=list(ARMS), ref_label="W0", file="figure_1_headline.svg",
        title=["Neither encoder beat the 37-column base W0 over 10 winter folds.", "Dot: pooled difference; bar: Bonferroni-8 interval over fold blocks; rows with no labelled event."],
    )  # fmt: skip
    both = pl.concat(
        [
            _table(contrasts=contrasts, ref="C-CNN-null", arms=("C-CNN", "T-CNN")).with_columns(arm=pl.col("arm") + " vs C-CNN-null"),
            _table(contrasts=contrasts, ref="W-T", arms=("C-CNN", "T-CNN")).with_columns(arm=pl.col("arm") + " vs W-T"),
        ]
    )  # fmt: skip
    COLOURS.update(
        {
            "C-CNN vs C-CNN-null": theme.DATA_BLUE,
            "T-CNN vs C-CNN-null": theme.DATA_GREEN,
            "C-CNN vs W-T": theme.BRAND_ORANGE,
            "T-CNN vs W-T": theme.DATA_PURPLE,
        }
    )
    both_arms = ["C-CNN vs C-CNN-null", "T-CNN vs C-CNN-null", "C-CNN vs W-T", "T-CNN vs W-T"]
    _forest(
        table=both, arms=both_arms, ref_label="the reference", file="figure_2_against_references.svg",
        title=["The encoders against the shuffled-day control and against experiment 515's W-T.", "Positive means the encoder scored worse. Dot: pooled difference; bar: Bonferroni-8 interval."],
    )  # fmt: skip
    frame = score.s515.load_rows()
    records = []
    for arm in ARMS:
        for g, label in GROUPS.items():
            mask = score.s515.mask_of(frame=frame, group=g, slice_name="cold_night")
            a = score.s515.fold_sums(frame=frame, column=f"{arm}__crps", mask=mask)
            b = score.s515.fold_sums(frame=frame, column="W0__crps", mask=mask)
            records.extend({"arm": arm, "group": label, "fold": m, "contrast": (a[m][0] - b[m][0]) / a[m][1]} for m in sorted(a) if a[m][1])  # fmt: skip
    per_fold = pl.DataFrame(records)
    per_fold.write_csv(score.SCORE_DIR / "per_fold_cold_night.csv")
    panels = []
    for label in GROUPS.values():
        base = alt.Chart(per_fold.filter(pl.col("group") == label).to_pandas()).encode(x=alt.X("fold:N", title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y("contrast:Q", title="CRPS minus W0, cold nights (points)"), color=alt.Color("arm:N", scale=alt.Scale(domain=list(ARMS), range=[COLOURS[a] for a in ARMS])))  # fmt: skip  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, base.mark_point(filled=True, size=55)).properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start")))  # fmt: skip  # ty: ignore[invalid-argument-type]
    alt.vconcat(*panels).properties(title=alt.TitleParams("Cold-night CRPS minus W0 in each fold (negative favours the arm).", anchor="start")).save(DOCS_DIR / "figure_3_per_fold.svg")  # fmt: skip


if __name__ == "__main__":
    main()
