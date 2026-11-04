# ruff: noqa: E501
"""Print every number the experiment 554 page quotes, and draw its two figures.

Reads the tables `e00554_score.py` and `e00554_stress.py` stored, plus the per-row CRPS files for the
per-Primary check (a leave-one-out look at whether one Primary carries the M0 gain).
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from e00521_refs import DATA as DATA_521
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
)
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out"
)
SCORES: Final[Path] = ROOT / "scores"
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
COLOURS: Final = {"M0": theme.DATA_BLUE, "M0lr": theme.DATA_SKY}
SETS: Final = {
    "eight": "MVA-only eight",
    "eight_without_D8": "MVA-only eight without D8",
    "six": "Signed-MW six",
}
WIDTH: Final = 600


POOLED: Final = "24-120h"
ARMS: Final = ("M0", "M0lr")
EIGHT: Final = [0, 1, 2, 4, 8, 13, 14, 15]


def tables() -> None:
    """Print the absolute CRPS and the contrasts (with per-fold signs and the MDE)."""
    a = pl.read_csv(SCORES / "absolute.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=220):
        for s in ("eight", "six", "eight_without_D8", "D8_alone", "report"):
            print(f"--- absolute CRPS, set {s}")
            print(
                a.filter(pl.col("set") == s)
                .pivot(on="arm", index="band", values="crps")
                .select(
                    [
                        "band",
                        *[
                            x
                            for x in (
                                "L-1d5",
                                "L-1d",
                                "M0",
                                "M0lr",
                                "R-DW",
                                "A-pool",
                                "manual_heuristic",
                            )
                            if x in a["arm"].unique().to_list()
                        ],
                    ]
                )
                .with_columns(pl.col(pl.Float64).round(3))
            )
        c = pl.read_csv(SCORES / "contrasts.csv")
        for s in ("eight", "six", "eight_without_D8", "D8_alone"):
            print(f"--- contrasts, set {s}")
            print(
                c.filter(
                    (pl.col("set") == s)
                    & pl.col("contrast").is_in(
                        [
                            "M0 - L-1d5",
                            "M0lr - L-1d5",
                            "M0 - M0lr",
                            "L-1d5 - L-1d",
                            "M0 - A-pool",
                            "M0 - R-DW",
                            "L-1d5 - A-pool",
                        ]
                    )
                )
                .select(
                    "band",
                    "contrast",
                    pl.col("difference").round(3),
                    pl.col("low_bonferroni_month").round(3),
                    pl.col("high_bonferroni_month").round(3),
                    "folds_positive",
                    "by_fold",
                    pl.col("mde").round(3),
                    "tests",
                )
                .sort("contrast", "band")
            )


def per_primary() -> pl.DataFrame:
    """Print each Primary's CRPS minus L-1d (5 seeds) over 24-120 h for M0 and M0lr."""
    frames = [
        pl.read_parquet(
            ROOT / "work/scores" / f"rows_{m}.parquet",
            columns=["primary", "lead_hours", "M0__crps", "M0lr__crps", "L-1d5__crps", "main16"],
        )
        for m in ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
    ]
    t = pl.concat(frames).filter(
        pl.col("main16")
        & pl.all_horizontal(
            pl.col(x).is_finite() for x in ("M0__crps", "M0lr__crps", "L-1d5__crps")
        )
        & (pl.col("lead_hours") >= 24)
        & (pl.col("lead_hours") < 120)
    )
    g = (
        t.group_by("primary")
        .agg(
            *[
                (pl.col(f"{a}__crps") - pl.col("L-1d5__crps")).mean().alias(f"{a} - L-1d5")
                for a in ARMS
            ],
            pl.col("L-1d5__crps").mean().alias("L-1d5 level"),
            n=pl.len(),
        )
        .sort("primary")
    )
    sites = (
        pl.read_csv(DATA_521 / "site_map.csv")
        .with_columns(pl.col("primary").cast(pl.Int64))
        .select("primary", "label")
    )
    g = (
        g.with_columns(pl.col("primary").cast(pl.Int64))
        .join(sites, on="primary", how="left")
        .sort("primary")
    )
    print("--- per-Primary CRPS minus L-1d5, leads 24-120 h (negative favours the modular arm)")
    with pl.Config(tbl_rows=30):
        print(g.with_columns(pl.col(pl.Float64).round(3)))
    eight = g.filter(pl.col("primary").is_in(EIGHT))
    for a in ARMS:
        col = f"{a} - L-1d5"
        print(
            f"{a}: negative at {int((eight[col] < 0).sum())} of {eight.height} of the eight; mean of per-Primary means {float(eight[col].mean()):.3f}"
        )  # ty: ignore[invalid-argument-type]
    return g


def figure_gain() -> None:
    """Draw each modular arm's CRPS minus L-1d (5 seeds) by lead band, with Bonferroni intervals."""
    bands = (*BANDS, POOLED)
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("contrast").is_in([f"{a} - L-1d5" for a in ARMS]) & pl.col("band").is_in(bands)
    )
    c = c.with_columns(arm=pl.col("contrast").str.replace(" - L-1d5", ""))
    panels = []
    for key, label in SETS.items():
        part = c.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", sort=list(bands), title="Lead band"),
            yOffset=alt.YOffset("arm:N", sort=list(COLOURS)),
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni_month:Q",
                title="CRPS minus L-1d (points of capacity; more negative is better)",
            ),
            x2="high_bonferroni_month:Q",
            color=colour,
        )
        dot = base.mark_point(filled=True, size=45).encode(x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=200, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Modular arms against the single-network L-1d, both averaged over 5 seeds,",
                "five folds, rows with no labelled event. Dot: difference. Bar: Bonferroni",
                "month-block interval (28 tests). Line at 0: same as L-1d.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_gain_vs_l1d.svg")


def figure_primary(g: pl.DataFrame) -> None:
    """Draw each of the eight Primaries' CRPS minus L-1d at leads of 24 to 120 h."""
    eight = g.filter(pl.col("primary").is_in(EIGHT))
    long = eight.unpivot(
        index=["label"], on=[f"{a} - L-1d5" for a in ARMS], variable_name="arm", value_name="diff"
    )
    long = long.with_columns(arm=pl.col("arm").str.replace(" - L-1d5", "")).to_pandas()
    chart = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort=eight.sort("M0 - L-1d5")["label"].to_list(), title="Primary"),
            yOffset=alt.YOffset("arm:N", sort=list(COLOURS)),
            x=alt.X(
                "diff:Q",
                title="CRPS minus L-1d, 24-120 h (points of capacity; more negative is better)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=300,
            title=alt.TitleParams(
                [
                    "Gain by Primary at the eight MVA-only Primaries, five folds, 5 seeds.",
                    "One Primary (D8) gains far more than the others.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "figure_2_gain_by_primary.svg")


if __name__ == "__main__":
    tables()
    figure_primary(per_primary())
    figure_gain()
