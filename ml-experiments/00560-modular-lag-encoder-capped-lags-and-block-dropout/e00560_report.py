"""Print every number the experiment 560 page quotes, and draw its two figures.

Reads the tables `e00560_score.py` and `e00560_stress.py` stored.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00560-modular-lag-encoder-capped-lags-and-block-dropout"
SCORES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "scores"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
ARMS: Final = {
    "LAG-v1": theme.DATA_BLUE,
    "LAG-v2": theme.DATA_SKY,
    "LAG-v3": theme.BRAND_ORANGE,
    "LAG-v4": "#808080",
    "LAG-b_shared": theme.DATA_GREEN,
    "LAG-b_unshared": "#8c564b",
}
SETS: Final = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final = 600


def tables() -> None:
    """Print absolute CRPS (24-120 h and every band) and the contrasts."""
    a = pl.read_csv(SCORES / "absolute.csv")
    c = pl.read_csv(SCORES / "contrasts.csv")
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_str_lengths=40):
        for s in ("eight", "eight_no_d8", "six", "report"):
            print(f"--- absolute CRPS, set {s}")
            print(
                a.filter(pl.col("set") == s)
                .pivot(on="arm", index="band", values="crps")
                .with_columns(pl.col(pl.Float64).round(3))
            )
        print("--- counted contrasts (eight, 24-120h)")
        print(
            c.filter(pl.col("counted")).select(
                "contrast",
                "difference",
                "low_bonferroni_month",
                "high_bonferroni_month",
                "folds_positive",
                "by_fold",
                "mde",
                "tests",
            )
        )
        for s in ("eight", "eight_no_d8", "six"):
            print(f"--- all contrasts at 24-120h, set {s}")
            print(
                c.filter(
                    (pl.col("set") == s)
                    & (pl.col("band") == "24-120h")
                    & ~pl.col("contrast").str.contains("aged|removed")
                )
                .select(
                    "contrast",
                    pl.col("difference").round(3),
                    pl.col("low_bonferroni_month").round(3),
                    pl.col("high_bonferroni_month").round(3),
                    "folds_positive",
                    pl.col("mde").round(3),
                )
                .unique(maintain_order=True)
            )


def figure_gain() -> None:
    """Draw each arm's CRPS minus M0 by lead band with Bonferroni month-block intervals."""
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("contrast").is_in([f"{a} - M0" for a in ARMS]) & pl.col("band").is_in(BANDS)
    )
    c = c.with_columns(arm=pl.col("contrast").str.replace(" - M0", "")).unique(
        subset=["set", "band", "arm"]
    )
    panels = []
    for key, label in SETS.items():
        part = c.filter(pl.col("set") == key).to_pandas()
        domain = [a for a in ARMS if a in set(part["arm"])]
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", sort=list(BANDS), title="Lead band"),
            yOffset=alt.YOffset("arm:N", sort=domain),
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[ARMS[a] for a in domain]),
            legend=alt.Legend(orient="top", title=None),
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni_month:Q",
                title="CRPS minus M0 (points of capacity; more negative is better)",
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
                width=WIDTH, height=240, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Lag-encoder arms against M0 (no lags), five folds, rows with no labelled event.",
                "Dot: difference. Bar: Bonferroni month-block interval (4 counted tests).",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_gain_vs_m0.svg")


def figure_leak() -> None:
    """Draw the injected-shift leak by block age."""
    t = pl.read_csv(SCORES / "stress_leak_by_age.csv").to_pandas()
    domain = ["L-1d", "M0", "R-DW-24", *ARMS]
    rng = ["#000000", "#555555", "#b0b0b0", *ARMS.values()]
    present = [(d, r) for d, r in zip(domain, rng, strict=True) if d in set(t["arm"])]
    chart = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("block_age_days:O", title="Days since the end of the injected 7-day block"),
            y=alt.Y("leak:Q", title="Leak (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=[d for d, _ in present], range=[r for _, r in present]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                [
                    "Injected-shift leak: the change in the median forecast, signed",
                    "by the injected shift; mean over five folds, seed 0, all leads.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "figure_2_stress_leak.svg")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    tables()
    figure_gain()
    figure_leak()
    print(
        pl.read_csv(SCORES / "stress_leak.csv")
        .pivot(on="band", index="arm", values="leak")
        .with_columns(pl.col(pl.Float64).round(2))
    )
