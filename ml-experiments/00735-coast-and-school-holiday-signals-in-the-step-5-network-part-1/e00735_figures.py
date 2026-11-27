# ruff: noqa: E501, PERF401
"""Experiment 735: the figures of the write-up, drawn from the scorer's outputs (CPU only).

README. Run after `e00735_score.py`, with the experiment folders on `PYTHONPATH` (source
`ml-experiments/00524*/env.sh`). It reads `scores/readings.csv`, `scores/by_fit.csv`, and the cell
files under experiment 735's data folder, and writes SVG files to `figures/`. Colours come from
`plotting.ocf_theme` and are set explicitly (Brand Orange, Data Blue, Data Sky, Data Purple), and the
charts follow the repository's `dataviz` skill: each chart is 680 pixels wide, its title states what
is plotted, which direction is better, and the scope, and no calendar date from a metered
generator appears (only substation demand and forecast errors are drawn).

The figures drawn here: (1) the family and placebo readings with their intervals; (2) each paired
difference by refit month and seed; (3) the signed error of the 12 to 36 hour median on holiday days
against coast distance for R0, S, and SKW. Not drawn here: the forecast time series around each
half-term, the loss against epoch, the per-group tail results by window, and the slope against
distance for SKW and SKWP (their inputs are not in the cell files; see the report).
"""

import os
from pathlib import Path
from typing import Final

import e00735_score as score  # ty: ignore[unresolved-import]
import e00735_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
ROOT: Final[Path] = Path(os.environ.get("R735", ML_EXPERIMENTS_DATA_DIR / NAME))
WIDTH: Final = 620
ORANGE, BLUE, SKY, PURPLE = "#FF4901", "#306BFF", "#10C5F7", "#B701FF"


def forest(*, readings: pl.DataFrame, path: Path) -> None:
    """Draw each family and placebo reading with its interval; a gain is to the left of zero."""
    import altair as alt
    import plotting.ocf_theme  # noqa: F401

    names = [
        "C1 six",
        "C1 eight",
        "C2 six",
        "C2 eight",
        "C4",
        "C5",
        "SP-R0 six",
        "SP-R0 eight",
        "SKWP-S",
        "SEP-S",
        "inland C4",
        "inland C5",
    ]
    frame = readings.filter(pl.col("name").is_in(names) & pl.col("estimate").is_finite()).with_columns(
        kind=pl.when(pl.col("name").is_in(names[:6])).then(pl.lit("family reading (99.17% interval)")).otherwise(pl.lit("placebo reading (95% interval)"))
    )  # fmt: skip
    base = alt.Chart(frame).encode(
        y=alt.Y("name:N", sort=names, title=None),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(range=[ORANGE, BLUE]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    chart = (
        base.mark_rule().encode(x=alt.X("lower:Q", title="CRPS, arm minus reference (points of capacity; more negative is better)"), x2="upper:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=60).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888", strokeDash=[3, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # fmt: skip
    chart.properties(
        width=WIDTH,
        height=22 * len(names),
        title=alt.TitleParams(
            "Planned readings: days 1 to 10, school windows, held-in months",
            subtitle=[
                "Dot: estimate. Line: interval from the two-level bootstrap (refit month, seed, 7-day block). Dashed rule: no difference.",
                "Contrasts: C1 S minus R0, C2 S minus SP, C4 SKW minus SKWP, C5 SE minus SEP.",
            ],
        ),
    ).save(str(path))


def by_fit(*, table: pl.DataFrame, path: Path) -> None:
    """Draw each contrast's difference per refit month and seed."""
    import altair as alt
    import plotting.ocf_theme  # noqa: F401

    frame = table.filter(pl.col("period") == "held-in").with_columns(
        contrast_group=pl.col("contrast") + " " + pl.col("group")
    )
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45)
        .encode(
            x=alt.X("diff:Q", title="CRPS, arm minus reference (points of capacity; more negative is better)"),
            y=alt.Y("month:N", title=None),
            color=alt.Color("seed:N", scale=alt.Scale(range=[ORANGE, BLUE, SKY]), legend=alt.Legend(orient="top", title="seed")),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("contrast_group:N", title=None))
        .properties(title=alt.TitleParams("Each contrast by refit month and seed", subtitle=["Dot: one fit (held-in months, school windows, days 1 to 10)."]))
    )  # fmt: skip
    chart.save(str(path))


def signed_error_by_distance(*, path: Path) -> None:
    """Draw the signed error of the 12 to 36 hour median on school-window days against coast distance, for R0, S, and SKW."""
    import altair as alt
    import plotting.ocf_theme  # noqa: F401

    table = win.sites()
    rows = []
    for arm in ("R0", "S", "SKW"):
        cells = score.load_arm(arm=arm, months=score.MONTHS[: score.HELD_IN_MONTHS])
        if cells is None:
            continue
        codes = cells["vday"].to_numpy().astype(np.int64)
        masks = win.window_masks(codes=codes)
        for wname in (win.SCHOOL_WINDOW, "bank window"):
            sel = masks[wname]
            g = (
                cells.filter(pl.Series(sel))
                .group_by("primary")
                .agg(pl.col("s1236").sum(), pl.col("n1236").sum())
            )
            for r in g.iter_rows(named=True):
                if r["n1236"]:
                    rows.append(
                        {
                            "arm": arm,
                            "window": wname,
                            "primary": r["primary"],
                            "signed": r["s1236"] / r["n1236"],
                        }
                    )
    if not rows:
        print("no cells yet: the signed-error figure is skipped")
        return
    frame = pl.DataFrame(rows).join(table.select("primary", "d_label", "coast_km"), on="primary")
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=45)
        .encode(
            x=alt.X("coast_km:Q", title="Distance from the coast (km)"),
            y=alt.Y("signed:Q", title="Median minus actual (points of capacity; zero is unbiased)"),
            color=alt.Color("arm:N", scale=alt.Scale(domain=["R0", "S", "SKW"], range=[BLUE, ORANGE, PURPLE]), legend=alt.Legend(orient="top", title=None)),
        )
        .properties(width=WIDTH, height=160)
        .facet(row=alt.Row("window:N", title=None))
        .properties(title=alt.TitleParams("Signed error of the 12 to 36 hour median by distance from the coast", subtitle=["Dot: one Primary, held-in months. Negative is an under-forecast."]))
    )  # fmt: skip
    chart.save(str(path))


def main() -> None:
    """Draw every figure whose input exists."""
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    readings = ROOT / "scores" / "readings.csv"
    if readings.exists():
        forest(readings=pl.read_csv(readings, infer_schema_length=None), path=out / "readings.svg")
    fits = ROOT / "scores" / "by_fit.csv"
    if fits.exists():
        by_fit(table=pl.read_csv(fits), path=out / "by_fit.svg")
    signed_error_by_distance(path=out / "signed_error_by_distance.svg")
    print("wrote", out)


if __name__ == "__main__":
    main()
