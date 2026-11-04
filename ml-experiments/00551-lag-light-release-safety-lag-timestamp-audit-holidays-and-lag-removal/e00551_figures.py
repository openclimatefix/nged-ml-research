# ruff: noqa: D103, E501
"""Draw experiment 551's two figures from the stored score tables."""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal"
    / "scores"
)
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal"
)
SETS: Final[dict[str, str]] = {
    "eight": "MVA-only eight Primaries",
    "six": "Signed-MW six Primaries",
}
WIDTH: Final[int] = 560
ARMS: Final[dict[str, str]] = {
    "R-DW": theme.DATA_GREEN,
    "R-DROP": theme.BRAND_ORANGE,
    "A-pool": theme.DATA_PURPLE,
    "E (548)": theme.DATA_SKY,
}
SLICES: Final[list[str]] = [
    "Christmas week (24 Dec to 1 Jan)",
    "all bank-holiday windows",
    "cold nights",
    "rows with no labelled event",
]
SHORT: Final[dict[str, str]] = {
    SLICES[0]: "Christmas week (9 days)",
    SLICES[1]: "Bank holidays (13 days)",
    SLICES[2]: "Cold nights",
    SLICES[3]: "No labelled event",
}


def figure_calendar() -> None:
    t = pl.read_csv(SCORES / "calendar_slices.csv").filter(
        pl.col("slice").is_in(SLICES) & pl.col("set").is_in(list(SETS))
    )
    rows = [
        {"set": r["set"], "slice": SHORT[r["slice"]], "arm": a, "diff": r[f"{a} - L-1d"]}
        for r in t.iter_rows(named=True)
        for a in ARMS
    ]
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
        .mark_rule(color=theme.BLACK_2)
        .encode(x="y:Q")
    )  # ty: ignore[unresolved-attribute]
    panels = []
    for key, label in SETS.items():
        part = pl.DataFrame(rows).filter(pl.col("set") == key).to_pandas()
        dots = (
            alt.Chart(part)
            .mark_point(filled=True, size=80)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("slice:N", sort=[SHORT[s] for s in SLICES], title=None),
                x=alt.X("diff:Q", title="CRPS minus L-1d's (points of capacity)"),
                color=alt.Color(
                    "arm:N",
                    title=None,
                    legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=list(ARMS), range=list(ARMS.values())),
                ),
                yOffset="arm:N",
            )
        )
        panels.append(
            alt.layer(zero, dots).properties(
                width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "R-DW and A-pool do worse than L-1d in Christmas week at the eight; R-DROP does not.",
                "Dot: mean CRPS minus L-1d's, all lead times; negative favours the arm. Line: same as L-1d.",
                "No intervals: the holiday slices span a few days of one winter. Five folds; UTC dates.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_1_calendar_slices.svg")


def figure_removal() -> None:
    names = {
        "none": "all readings removed",
        "noday": "day-ago removed",
        "noweek": "week-ago removed",
        "age1": "readings 1 day late",
        "age2": "2 days late",
        "age3": "3 days late",
        "rand25": "25% of rows removed",
        "base": "readings present",
    }
    t = pl.read_csv(SCORES / "removal_contrasts.csv").filter(
        pl.col("band") == "24-120h", pl.col("set").is_in(list(SETS))
    )
    t = t.filter(pl.col("contrast").str.ends_with(" - L-1d")).with_columns(
        arm=pl.col("contrast").str.extract(r"^(R-DW|R-DROP)-"),
        scn=pl.col("contrast").str.extract(r"^R-(?:DW|DROP)-(\w+) - L-1d"),
    )
    t = t.with_columns(label=pl.col("scn").replace(names))
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
        .mark_rule(color=theme.BLACK_2)
        .encode(x="y:Q")
    )  # ty: ignore[unresolved-attribute]
    order = [
        names[s] for s in ("base", "age1", "age2", "age3", "rand25", "noday", "noweek", "none")
    ]
    cols = {"R-DW": theme.DATA_GREEN, "R-DROP": theme.BRAND_ORANGE}
    panels = []
    for key, label in SETS.items():
        part = t.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("label:N", sort=order, title=None), yOffset="arm:N")  # ty: ignore[unresolved-attribute]
        colour = alt.Color(
            "arm:N",
            title=None,
            legend=alt.Legend(orient="top"),
            scale=alt.Scale(domain=list(cols), range=list(cols.values())),
        )
        bar = base.mark_rule(strokeWidth=3).encode(
            x=alt.X("low_month:Q", title="CRPS minus L-1d's (points of capacity)"),
            x2="high_month:Q",
            color=colour,
        )
        dot = base.mark_point(filled=True, size=70).encode(x="difference:Q", color=colour)
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=230, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "With the weights fixed, R-DW loses to L-1d only when both readings are removed.",
                "CRPS minus L-1d's at 24 to 120 h; negative favours the arm. Dot: difference. Bar: 95% interval",
                "from 5 month blocks (near min to max). Line: same as L-1d. Late readings are taken from earlier days.",
            ],
            anchor="start",
            fontSize=12,
        )
    ).save(DOCS / "figure_2_lag_removal_and_late_reading.svg")


if __name__ == "__main__":
    figure_calendar()
    figure_removal()
