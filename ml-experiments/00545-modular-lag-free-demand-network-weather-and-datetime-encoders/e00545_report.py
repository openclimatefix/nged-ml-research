# ruff: noqa: E501
"""Print every number the experiment 545 page quotes, and draw its two figures.

Reads the tables `e00545_score.py` and `e00545_stress.py` stored, plus the per-row CRPS files for the
per-Primary check (a leave-one-out look at whether one Primary carries the M0 gain).
"""

from pathlib import Path
from typing import Final

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders"
)
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00545-modular-lag-free-demand-network-weather-and-datetime-encoders"
)
SCORES: Final[Path] = ROOT / "scores"
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
COLOURS: Final = {"M0": theme.DATA_BLUE, "M0lr": theme.DATA_SKY, "M1": theme.BRAND_ORANGE}
SETS: Final = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final = 600


def tables() -> None:
    """Print the absolute CRPS and the contrasts against L-1d."""
    a = pl.read_csv(SCORES / "absolute.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=40):
        for s in ("eight", "six", "report"):
            print(f"--- absolute CRPS, set {s}")
            print(
                a.filter(pl.col("set") == s)
                .pivot(on="arm", index="band", values="crps")
                .with_columns(pl.col(pl.Float64).round(3))
            )
        c = pl.read_csv(SCORES / "contrasts.csv")
        for s in ("eight", "six"):
            print(
                f"--- contrasts, set {s} (difference, Bonferroni month-block interval, folds worse, MDE)"
            )
            print(
                c.filter(
                    (pl.col("set") == s)
                    & pl.col("contrast").is_in(
                        [
                            "M0 - L-1d",
                            "M0lr - L-1d",
                            "M1 - L-1d",
                            "M1 - M0",
                            "M0 - M0lr",
                            "M1 - M0lr",
                            "M0 - A-pool",
                            "M0 - R-DW",
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
                    pl.col("mde").round(3),
                )
                .sort("contrast", "band")
            )


def per_primary() -> None:
    """Print each Primary's CRPS minus L-1d over 24-120 h for M0, M0lr, and M1, and leave-one-out means."""
    frames = [
        pl.read_parquet(
            ROOT / "work/scores" / f"rows_{m}.parquet",
            columns=[
                "primary",
                "lead_hours",
                "M0__crps",
                "M0lr__crps",
                "M1__crps",
                "L-1d__crps",
                "main16",
            ],
        )
        for m in ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
    ]
    t = pl.concat(frames).filter(
        pl.col("main16")
        & pl.all_horizontal(
            pl.col(x).is_finite() for x in ("M0__crps", "M0lr__crps", "M1__crps", "L-1d__crps")
        )
    )
    t = t.filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 120))
    g = (
        t.group_by("primary")
        .agg(
            *[
                (pl.col(f"{a}__crps") - pl.col("L-1d__crps")).mean().alias(f"{a} - L-1d")
                for a in ("M0", "M0lr", "M1")
            ],
            n=pl.len(),
        )
        .sort("primary")
    )
    print(
        "--- per-Primary CRPS minus L-1d, leads 24-120 h, all five folds (primary index; negative favours the modular arm)"
    )
    with pl.Config(tbl_rows=30):
        print(g.with_columns(pl.col(pl.Float64).round(3)))
    leave_one_out_in_sets(g)
    for a in ("M0", "M0lr", "M1"):
        col = f"{a} - L-1d"
        loo = [float(g.filter(pl.col("primary") != p)[col].mean()) for p in g["primary"]]  # ty: ignore[invalid-argument-type]
        print(
            f"{a}: Primaries with a negative difference {int((g[col] < 0).sum())} of {g.height}; leave-one-out mean range {min(loo):.3f} to {max(loo):.3f}"
        )


def leave_one_out_in_sets(g: pl.DataFrame) -> None:
    """Print each arm's mean difference over the eight and the six, and with each Primary left out in turn."""
    sets = {"eight": [0, 1, 2, 4, 8, 13, 14, 15], "six": [3, 5, 6, 7, 9, 11]}
    for name, members in sets.items():
        part = g.filter(pl.col("primary").is_in(members))
        for a in ("M0", "M0lr", "M1"):
            col = f"{a} - L-1d"
            loo = {
                int(p): round(float(part.filter(pl.col("primary") != p)[col].mean()), 3)  # ty: ignore[invalid-argument-type]
                for p in part["primary"]
            }
            print(
                f"{name} {a}: mean {float(part[col].mean()):.3f}; negative at {int((part[col] < 0).sum())} of {part.height}; leave-one-out {loo}"  # ty: ignore[invalid-argument-type]
            )


def figure_gain() -> None:
    """Draw each modular arm's CRPS minus L-1d by lead band, with Bonferroni intervals, eight and six."""
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        pl.col("contrast").is_in([f"{a} - L-1d" for a in COLOURS]) & pl.col("band").is_in(BANDS)
    )
    c = c.with_columns(arm=pl.col("contrast").str.replace(" - L-1d", ""))
    panels = []
    for key, label in SETS.items():
        part = c.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", sort=list(BANDS), title="Lead band"),
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
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=220, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Modular arms against the single-network L-1d, five folds,",
                "rows with no labelled event. Dot: difference. Bar: Bonferroni",
                "month-block interval (28 tests). Line at 0: same as L-1d.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_gain_vs_l1d.svg")


def figure_leak() -> None:
    """Draw the injected-shift leak by block age for L-1d and the modular arms."""
    t = pl.read_csv(SCORES / "stress_leak_by_age.csv").to_pandas()
    domain = ["L-1d", "R-DW", "M0", "M0lr", "M1"]
    rng = [theme.DATA_GREEN, "#808080", theme.DATA_BLUE, theme.DATA_SKY, theme.BRAND_ORANGE]
    chart = (
        alt.Chart(t)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("block_age_days:O", title="Days since the end of the injected 7-day block"),
            y=alt.Y("leak:Q", title="Leak (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=rng),
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
    tables()
    per_primary()
    figure_gain()
    figure_leak()
    print(
        pl.read_csv(SCORES / "stress_leak.csv")
        .pivot(on="band", index="arm", values="leak")
        .with_columns(pl.col(pl.Float64).round(2))
    )
