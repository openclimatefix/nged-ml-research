# ruff: noqa: E501
"""Draw the stress-cell figures of experiment 556 from the tables `e00556_stats.py` stored.

Figures (SVG, 680 pixels wide, in the experiment's docs folder; optimise each with
`npx svgo@4 --multipass --precision=1 --final-newline`):

1. `figure_1_leak_by_step_length.svg`: first-refit leak share against step length;
2. `figure_2_unlearn.svg`: leak against days since the refit that contained the step;
3. `figure_3_adaptation.svg`: share of a lasting step in the median against days;
4. `figure_4_trade_off.svg`: leak at a 3-day step against adaptation after 14 days.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00556_fit as fit
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / Path(__file__).resolve().parent.name
)
SCORES: Final[Path] = fit.WORK / "scores"
WIDTH: Final[int] = 680
COLOURS: Final[dict[str, str]] = {
    "F1d": theme.DATA_BLUE,
    "F3d": theme.DATA_SKY,
    "F7d": theme.DATA_GREEN,
    "F1d-182": theme.DATA_PURPLE,
    "F3d-E3": theme.BRAND_ORANGE,
    "F3d-R": "#6b6b6b",
}
ARMS: Final[list[str]] = list(COLOURS)
GROUP_NAMES: Final[dict[str, str]] = {"eight": "MVA-only eight", "six": "Signed-MW six"}


def colour(*, arms: list[str]) -> alt.Color:
    """Return the arm colour encoding with a legend above the plot."""
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=arms, range=[COLOURS[a] for a in arms]),
        legend=alt.Legend(orient="top", title=None, columns=6),
    )


def figure_1() -> None:
    """Draw the leak share of F1d's first refit against step length, both groups."""
    t = (
        pl.read_csv(SCORES / "leak_by_scenario.csv")
        .filter(
            (pl.col("arm") == "F1d")
            & pl.col("scenario").str.starts_with("step")
            & pl.col("scenario").str.ends_with("_p10")
            & pl.col("group").is_in(["eight", "six"])
        )
        .with_columns(
            days=pl.col("scenario").str.extract(r"step(\d+)_", 1).cast(pl.Int32),
            share=pl.col("leak") / fit_depth_points(),
            group=pl.col("group").replace(GROUP_NAMES),
        )
    )
    chart = (
        alt.Chart(t.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "days:Q",
                title="Length of the injected step before the refit (days)",
                axis=alt.Axis(values=[1, 3, 7, 14]),
            ),
            y=alt.Y(
                "share:Q",
                title="Share of the step in the next week's median",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=list(GROUP_NAMES.values()), range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=alt.TitleParams(
                [
                    "A 10%-of-capacity step in the fine-tuning targets reaches the next week's median.",
                    "Mean over five folds, two origins, both signs; daily refit; higher means more absorbed.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "figure_1_leak_by_step_length.svg")


def fit_depth_points() -> float:
    """Return the injected depth in points of capacity."""
    return 100.0 * fit.DEPTH_FRACTION


def figure_2() -> None:
    """Draw the leak against days since the step, for L = 3 and L = 7 days at the eight."""
    panels = []
    for scenario, label in (("step3_p10", "3-day step"), ("step7_p10", "7-day step")):
        t = pl.read_csv(SCORES / f"unlearn_{scenario}.csv").filter(pl.col("group") == "eight")
        panels.append(
            alt.Chart(t.to_pandas())
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("refit_off:Q", title="Days since the refit that first contained the step"),
                y=alt.Y("leak:Q", title="Leak (points of capacity)"),
                color=colour(arms=ARMS),
            )
            .properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "How fast each arm unlearns a 10%-of-capacity step once clean data arrive.",
                "Sign-adjusted median shift at the eight MVA-only Primaries, mean over five folds, two origins, both signs; lower is better.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_2_unlearn.svg")


def figure_3() -> None:
    """Draw the share of a lasting 10% step in the median against days, eight and six."""
    t = pl.read_csv(SCORES / "adaptation.csv").filter(pl.col("group").is_in(["eight", "six"]))
    panels = []
    for group, label in GROUP_NAMES.items():
        part = t.filter(pl.col("group") == group)
        panels.append(
            alt.Chart(part.to_pandas())
            .mark_line()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("origin_off:Q", title="Days since the step began"),
                y=alt.Y(
                    "share:Q",
                    title="Share of the step in the median",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                color=colour(arms=ARMS),
            )
            .properties(width=WIDTH, height=190, title=alt.TitleParams(label, anchor="start"))
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Adaptation to a lasting step of 10% of capacity.",
                "Share of the step in the median of the forecast issued on each day, mean over five folds and both signs; higher is faster.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_3_adaptation.svg")


def figure_4() -> None:
    """Draw the leak of a 3-day step against the adaptation share after 14 days, per arm."""
    leak = (
        pl.read_csv(SCORES / "leak_by_scenario.csv")
        .filter((pl.col("scenario") == "step3_p10") & pl.col("group").is_in(["eight", "six"]))
        .select("arm", "group", "leak")
    )
    ad = (
        pl.read_csv(SCORES / "adaptation.csv")
        .filter((pl.col("origin_off") == 14) & pl.col("group").is_in(["eight", "six"]))
        .select("arm", "group", "share")
    )
    t = leak.join(ad, on=["arm", "group"]).with_columns(group=pl.col("group").replace(GROUP_NAMES))
    panels = []
    for label in GROUP_NAMES.values():
        part = t.filter(pl.col("group") == label)
        base = alt.Chart(part.to_pandas())  # ty: ignore[unresolved-attribute]
        points = base.mark_point(filled=True, size=110).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "share:Q",
                title="Share of a lasting step in the median after 14 days (higher is better)",
                scale=alt.Scale(domain=[0.25, 0.55]),
                axis=alt.Axis(format="%", values=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]),
            ),
            y=alt.Y(
                "leak:Q",
                title="Leak of a 3-day step (points; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=colour(arms=ARMS),
        )
        text = base.mark_text(dx=-8, align="right", fontSize=10).encode(  # ty: ignore[unresolved-attribute]
            x="share:Q", y="leak:Q", text="arm:N"
        )
        panels.append(
            (points + text).properties(
                width=WIDTH, height=200, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Cutting the leak and keeping adaptation pull in opposite directions.",
                "Each dot is one arm; the best arm sits at the bottom right of each panel. F1d and F7d overlap: their first refit is the same fit.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_4_trade_off.svg")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    for f in (figure_1, figure_2, figure_3, figure_4):
        f()
