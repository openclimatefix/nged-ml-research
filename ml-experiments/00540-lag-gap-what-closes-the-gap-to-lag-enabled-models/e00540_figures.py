"""Draw the two figures of experiment 540 from the interim numbers of experiments 533, 538, and 539.

Nothing is fitted or scored here. Every number below is copied from the `interim_numbers.md` file of
the experiment named beside it. Run with `uv run --no-sync python e00540_figures.py`, then optimise
each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00540-lag-gap-what-closes-the-gap-to-lag-enabled-models"
)
WIDTH: Final[int] = 560

# L-1d minus A-pool at the eight, points of capacity (experiment 533): the gap each arm closes.
L1D_GAP: Final[dict[str, float]] = {"24-48 h": 0.288, "48-72 h": 0.254, "72-120 h": 0.200}

# Arm minus A-pool at the eight: difference, interval low, interval high (95% month-block
# bootstrap, 5 blocks). Experiment 538 for R1 and R9, experiment 539 for R-DW.
ARM_MINUS_A_POOL: Final[dict[str, dict[str, tuple[float, float, float]]]] = {
    "R1: newest value only (3 inputs)": {
        "24-48 h": (0.218, 0.129, 0.313),
        "48-72 h": (0.187, 0.094, 0.279),
        "72-120 h": (0.143, 0.055, 0.251),
    },
    "R-DW: day-ago and week-ago lags (6 inputs)": {
        "24-48 h": (0.069, 0.016, 0.122),
        "48-72 h": (0.052, -0.005, 0.109),
        "72-120 h": (0.034, -0.039, 0.105),
    },
    "R9: all 9 lag inputs": {
        "24-48 h": (0.052, -0.005, 0.109),
        "48-72 h": (0.038, -0.020, 0.096),
        "72-120 h": (0.027, -0.043, 0.093),
    },
}
ARM_COLOURS: Final[dict[str, str]] = {
    "R1: newest value only (3 inputs)": theme.BRAND_ORANGE,
    "R-DW: day-ago and week-ago lags (6 inputs)": theme.DATA_GREEN,
    "R9: all 9 lag inputs": theme.DATA_BLUE,
}

# Injected-shift leak, points of capacity, mean over five folds, seed 0 (experiments 538 and 539).
LEAK: Final[dict[str, tuple[float, float, float, float]]] = {
    # arm: all leads, day 1, days 2-3, days 4-10
    "L-1d (no lags)": (4.49, 5.39, 4.62, 4.31),
    "R1: newest value only": (4.52, 4.93, 4.20, 4.55),
    "R-DW: day-ago and week-ago lags": (5.68, 6.13, 5.54, 5.65),
    "R9: all 9 lag inputs": (5.38, 5.74, 5.19, 5.38),
}
LEAK_COLOURS: Final[list[str]] = [
    theme.BLACK_2,
    theme.BRAND_ORANGE,
    theme.DATA_GREEN,
    theme.DATA_BLUE,
]


def gap_closed_frame() -> pl.DataFrame:
    """Return the percentage of the gap closed, with an interval, for each arm and band."""
    rows = []
    for arm, bands in ARM_MINUS_A_POOL.items():
        for band, (difference, low, high) in bands.items():
            gap = L1D_GAP[band]
            rows.append(
                {
                    "arm": arm,
                    "band": band,
                    "closed": 100 * (1 - difference / gap),
                    "closed_low": 100 * (1 - high / gap),
                    "closed_high": 100 * (1 - low / gap),
                }
            )
    return pl.DataFrame(rows)


def figure_1() -> None:
    """Draw the gap closed by arm and band, with intervals."""
    frame = gap_closed_frame().to_pandas()
    arms = list(ARM_MINUS_A_POOL)
    base = alt.Chart(frame).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("band:N", sort=list(L1D_GAP), title=None),
        yOffset=alt.YOffset("arm:N", sort=arms),
        color=alt.Color(
            "arm:N",
            sort=arms,
            scale=alt.Scale(domain=arms, range=[ARM_COLOURS[a] for a in arms]),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    bar = base.mark_rule(strokeWidth=3, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("closed_low:Q", title="Share of L-1d's gap to A-pool that the arm closes (%)"),
        x2="closed_high:Q",
    )
    dot = base.mark_point(filled=True, size=70, aria=False).encode(x="closed:Q")  # ty: ignore[unresolved-attribute]
    rules = (
        alt.Chart(pl.DataFrame({"x": [0.0, 100.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")
    )
    alt.layer(rules, bar, dot).properties(
        width=WIDTH,
        height=190,
        title=alt.TitleParams(
            [
                "Day-ago and week-ago lags close most of the gap; the newest value does not",
                "Eight MVA-only Primaries, CRPS, lead 24 to 120 h. 0% = L-1d; 100% = A-pool.",
                "Dot: estimate. Line: 95% month-block interval (5 blocks, near min-to-max),",
                "rescaled by the point estimate of L-1d minus A-pool. Interim, unaudited.",
            ],
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_1_gap_closed.svg")


def figure_2() -> None:
    """Draw the injected-shift leak by arm and lead group."""
    groups = ["All leads", "Day 1", "Days 2-3", "Days 4-10"]
    rows = [
        {"arm": arm, "group": group, "leak": value}
        for arm, values in LEAK.items()
        for group, value in zip(groups, values, strict=True)
    ]
    arms = list(LEAK)
    alt.Chart(pl.DataFrame(rows).to_pandas()).mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("group:N", sort=groups, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("arm:N", sort=arms),
        y=alt.Y("leak:Q", title="Leak (points of capacity; smaller is better)"),
        color=alt.Color(
            "arm:N",
            sort=arms,
            scale=alt.Scale(domain=arms, range=LEAK_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=500),
        ),
    ).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            [
                "Adding the day-ago and week-ago lags raises the injected load-shift leak",
                "Leak: how far the median forecast moves when plus or minus 3 SD are added",
                "to the targets and lags of a 7-day block. Mean of five folds, one seed.",
                "Fold ranges overlap (all leads: L-1d 2.6 to 10.5; R9 2.2 to 11.3).",
                "Interim, unaudited.",
            ],
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_2_injected_shift_leak.svg")


if __name__ == "__main__":
    figure_1()
    figure_2()
