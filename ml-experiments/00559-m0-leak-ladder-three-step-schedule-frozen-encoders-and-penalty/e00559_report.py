"""Print the contrast and leak tables the experiment 559 page quotes, and draw its three figures.

Reads the tables `e00559_score.py` and `e00559_stress.py` stored.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty"
ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
SCORES: Final[Path] = ROOT / "scores"
WIDTH: Final = 600
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
POOLED: Final = "24-120h"
SHOWN: Final = ("A1", "A2", "A3", "A5", "A8", "A10", "A11")
"""Arms drawn in the figures; the tables hold every arm."""
COLOUR: Final = {
    "M0": theme.DATA_BLUE,
    "A1": theme.DATA_SKY,
    "A2": theme.BRAND_ORANGE,
    "A3": theme.DATA_PURPLE,
    "A5": theme.DATA_GREEN,
    "A8": theme.BRAND_ORANGE_LIGHT,
    "A10": theme.DATA_GREEN_LIGHT,
    "A11": theme.DATA_PURPLE_LIGHT,
}


def tables() -> None:
    """Print contrasts for the sets and bands the page quotes, then the leak and its spread."""
    c = pl.read_csv(SCORES / "contrasts.csv")
    with pl.Config(tbl_rows=300, tbl_cols=20, tbl_width_chars=220, fmt_str_lengths=40):
        for s in ("eight", "six", "eight_without_D8", "eight_cold", "six_cold"):
            print(f"--- contrasts at {POOLED} and all rows, set {s}")
            print(
                c.filter(
                    (pl.col("set") == s)
                    & pl.col("band").is_in([POOLED, "all"])
                    & pl.col("contrast").str.contains("M0$|null$")
                )
                .select(
                    "band",
                    "contrast",
                    pl.col("difference").round(3),
                    pl.col("low_month").round(3),
                    pl.col("high_month").round(3),
                    "folds_positive",
                    pl.col("mde").round(3),
                    "tests",
                    "counted",
                )
                .sort("contrast", "band")
            )
        print("--- injected-shift leak (points of capacity)")
        print(
            pl.read_csv(SCORES / "stress_leak.csv")
            .filter(pl.col("band") == "all")
            .with_columns(pl.col(pl.Float64).round(2))
            .sort("leak")
        )


def figure_leak_vs_gain() -> None:
    """Draw each arm's leak against its 24-120 h CRPS change from M0 at the eight."""
    gain = (
        pl.read_csv(SCORES / "contrasts.csv")
        .filter(
            (pl.col("set") == "eight")
            & (pl.col("band") == POOLED)
            & pl.col("contrast").str.ends_with(" - M0")
        )
        .select(arm=pl.col("contrast").str.replace(" - M0", ""), diff="difference")
    )
    leak = pl.read_csv(SCORES / "stress_leak.csv").filter(pl.col("band") == "all")
    d = leak.join(gain, on="arm", how="inner").to_pandas()
    m0 = pl.DataFrame(
        {
            "arm": ["M0"],
            "leak": [float(leak.filter(pl.col("arm") == "M0")["leak"][0])],
            "diff": [0.0],
        }
    ).to_pandas()
    pts = alt.Chart(d).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "leak:Q",
            title="Leak of an injected 7-day shift (points of capacity; smaller is better)",
        ),
        y=alt.Y("diff:Q", title="CRPS minus M0, 24-120 h, eight (points; more negative is better)"),
    )
    layer = alt.layer(
        pts.mark_point(filled=True, size=60, color=theme.DATA_BLUE),  # ty: ignore[invalid-argument-type]
        pts.mark_text(dy=-9, fontSize=11).encode(text="arm:N"),  # ty: ignore[unresolved-attribute]
        alt.Chart(m0)
        .mark_point(shape="diamond", filled=True, size=120, color="black")
        .encode(x="leak:Q", y="diff:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(color="black").encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            [
                "Leak against forecast skill for each arm, at the eight MVA-only Primaries.",
                "Dot: arm (5-fold mean, leak from seed 0). Diamond: M0. Line at 0: same as M0.",
                "Left of the diamond: less leak. Below the line: better forecast.",
            ],
            anchor="start",
        ),
    )
    layer.save(DOCS / "figure_1_leak_vs_skill.svg")


def figure_bands() -> None:
    """Draw the shown arms' CRPS minus M0 by lead band at the eight, with month-block intervals."""
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("set") == "eight")
        & pl.col("band").is_in(BANDS)
        & pl.col("contrast").is_in([f"{a} - M0" for a in SHOWN])
    )
    c = c.with_columns(arm=pl.col("contrast").str.replace(" - M0", "")).to_pandas()
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(SHOWN), range=[COLOUR[a] for a in SHOWN]),
        legend=alt.Legend(orient="top", title=None, columns=7),
    )
    base = alt.Chart(c).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("band:N", sort=list(BANDS), title="Lead band"),
        yOffset=alt.YOffset("arm:N", sort=list(SHOWN)),
    )
    layer = alt.layer(
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="black").encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_month:Q", title="CRPS minus M0 (points of capacity; more negative is better)"
            ),
            x2="high_month:Q",
            color=colour,
        ),
        base.mark_point(filled=True, size=40).encode(x="difference:Q", color=colour),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=380,
        title=alt.TitleParams(
            [
                "Each arm against M0 by lead band, eight MVA-only Primaries, five folds,",
                "rows with no labelled event. Dot: difference. Line: 95% month-block interval",
                "(5 blocks, near min-to-max). Line at 0: same as M0.",
            ],
            anchor="start",
        ),
    )
    layer.save(DOCS / "figure_2_gain_by_band.svg")


def figure_age() -> None:
    """Draw the leak by days since the injected shift for M0 and the shown arms."""
    d = (
        pl.read_csv(SCORES / "stress_leak_by_age.csv")
        .filter(pl.col("arm").is_in(["M0", *SHOWN]))
        .to_pandas()
    )
    order = ["M0", *SHOWN]
    layer = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("block_age_days:O", title="Days since the injected shift ended"),
            y=alt.Y("leak:Q", title="Leak (points of capacity; smaller is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=order, range=[COLOUR[a] for a in order]),
                legend=alt.Legend(orient="top", title=None, columns=8),
            ),
        )
        .properties(
            width=WIDTH,
            height=280,
            title=alt.TitleParams(
                [
                    "Leak by days since the injected shift, mean of 5 folds, seed 0, both signs.",
                    "Daily refits on days 14 to 20 of each fold. Smaller is better.",
                ],
                anchor="start",
            ),
        )
    )
    layer.save(DOCS / "figure_3_leak_by_age.svg")


if __name__ == "__main__":
    tables()
    figure_leak_vs_gain()
    figure_bands()
    figure_age()
