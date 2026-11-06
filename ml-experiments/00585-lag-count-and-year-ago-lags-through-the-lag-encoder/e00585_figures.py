# ruff: noqa: D103
"""Experiment 585 figures: CRPS contrasts with intervals, slice contrasts, and example forecasts.

Writes SVG files beside `result.md`. Example forecasts are for one Primary (a substation, which may
be named) and compare the control (v3+D5) with Y1 (adds 364-day lag).
"""

import subprocess
from pathlib import Path

import altair as alt
import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00585-lag-count-and-year-ago-lags-through-the-lag-encoder"
WORK = ML_EXPERIMENTS_DATA_DIR / NAME
DOCS = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
COLOURS = {
    "N1": "#777777", "N3": theme.DATA_SKY, "N5": theme.DATA_GREEN,
    "Y1": theme.BRAND_ORANGE, "Y1c": theme.DATA_PURPLE, "Y3": "#222222",
}  # fmt: skip
GRID_START_DAY0 = np.datetime64("2019-09-01")


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"{name}.svg"
    chart.save(out)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(out)],
        check=False,
        capture_output=True,
    )
    print("wrote", out)


def contrasts_figure() -> None:
    c = pl.read_csv(WORK / "scores/part1_contrasts.csv").filter(
        (pl.col("slice") == "all rows") & (pl.col("b") == "V3D5") & pl.col("counted")
    )
    c = c.with_columns(
        group=pl.col("set").replace({"eight": "Eight MVA Primaries", "six": "Six MW Primaries"})
    ).rename({"a": "arm"})
    base = alt.Chart(c).encode(
        y=alt.Y("arm:N", sort=list(COLOURS), title=None),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q", title="CRPS minus control (points of capacity; below zero is better)"
        )
    )
    bars = base.mark_rule().encode(x="low_bonferroni_month:Q", x2="high_bonferroni_month:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(c).mark_rule(color="#999999").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, bars, pts)
        .properties(width=560, height=120)
        .facet(
            row=alt.Row("group:N", title=None),
            column=alt.Column("band:N", sort=["0-24h", "24-120h", "120-240h"], title="Lead band"),
        )
        .properties(
            title=alt.TitleParams(
                text="Change in error against the control (Bonferroni intervals, 36 tests)",
                anchor="start",
                fontSize=13,
            )
        )
    )  # ty: ignore[unresolved-attribute]
    save(chart=chart, name="crps_contrasts")


def slices_figure() -> None:
    c = (
        pl.read_csv(WORK / "scores/part1_contrasts.csv")
        .filter(
            (pl.col("b") == "V3D5")
            & pl.col("set").is_in(["eight", "six"])
            & (pl.col("band") == "24-120h")
            & pl.col("slice").is_in(
                [
                    "no labelled event",
                    "cold winter nights",
                    "Christmas and New Year",
                    "bank-holiday days",
                ]
            )
            & pl.col("a").is_in(["Y1", "Y1c", "N1", "N3", "Y3"])
        )
        .rename({"a": "arm"})
    )
    c = c.with_columns(
        group=pl.col("set").replace({"eight": "Eight MVA Primaries", "six": "Six MW Primaries"})
    )
    base = alt.Chart(c).encode(
        y=alt.Y("arm:N", sort=list(COLOURS), title=None),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="CRPS minus control at 24-120 h (points of capacity)")
    )
    bars = base.mark_rule().encode(x="low_bonferroni_month:Q", x2="high_bonferroni_month:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(c).mark_rule(color="#999999").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, bars, pts)
        .properties(width=250, height=110)
        .facet(row=alt.Row("group:N", title=None), column=alt.Column("slice:N", title=None))
        .properties(
            title=alt.TitleParams(
                text="Error against the control on event types (unadjusted month-block intervals)",
                anchor="start",
                fontSize=13,
            )
        )
    )  # ty: ignore[unresolved-attribute]
    save(chart=chart, name="crps_by_event_type")


def forecasts(*, month: str, arm_dir: Path | None, primary: int) -> pl.DataFrame:
    sites = build470.site_map()
    root, folder = sc.ARM_DIRS["V3D5"] if arm_dir is None else (arm_dir, "V3D5")
    frame = build470.pool_stored(
        folder=root / "seed_preds" / folder, month=month, sites=sites, seeds=(0, 1)
    )
    base = score454.month_base(month=month).select(*d6.KEYS, "lead_hours", "actual")
    return frame.filter(pl.col("primary") == primary).join(base, on=d6.KEYS, how="left")


def example_figure(
    *, name: str, title: str, lo: str, hi: str, day_ahead: bool, origin: str | None, primary: int
) -> None:
    lo_i = int((np.datetime64(lo) - GRID_START_DAY0) / np.timedelta64(30, "m"))
    hi_i = int((np.datetime64(hi) - GRID_START_DAY0) / np.timedelta64(30, "m"))
    parts = []
    for label, arm_dir in (
        ("Control (v3+D5)", None),
        ("Y1 (adds 364-day lag)", WORK / "arms/y1/work"),
        ("Y1c (adds 365-day lag)", WORK / "arms/y1c/work"),
    ):
        f = forecasts(month="2025-12", arm_dir=arm_dir, primary=primary)
        f = f.filter((pl.col("target") >= lo_i) & (pl.col("target") < hi_i))
        if day_ahead:
            f = f.filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 48))
        else:
            o = int((np.datetime64(origin) - GRID_START_DAY0) / np.timedelta64(30, "m"))
            f = f.filter(pl.col("origin") == o)
        parts.append(f.with_columns(series=pl.lit(label)))
    d = pl.concat(parts).with_columns(
        day=(pl.col("target") - lo_i) / 48.0,
        **{c: 100 * pl.col(c) for c in ("q05", "q25", "q50", "q75", "q95", "actual")},
    )
    colours = {
        "Control (v3+D5)": theme.DATA_BLUE,
        "Y1 (adds 364-day lag)": theme.BRAND_ORANGE,
        "Y1c (adds 365-day lag)": theme.DATA_PURPLE,
    }
    cs = alt.Scale(domain=list(colours), range=list(colours.values()))
    base = alt.Chart(d).encode(
        x=alt.X("day:Q", title=f"Days since {lo}"),
        color=alt.Color("series:N", scale=cs, legend=alt.Legend(orient="top", title=None)),
    )  # ty: ignore[unresolved-attribute]
    b90 = base.mark_area(opacity=0.10).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)), y2="q95:Q"
    )
    b50 = base.mark_area(opacity=0.16).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
    med = base.mark_line(strokeWidth=1.3).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
    act = (
        alt.Chart(d.filter(pl.col("series") == "Control (v3+D5)"))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#222222", strokeWidth=1.0)
        .encode(x="day:Q", y="actual:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = alt.layer(b90, b50, med, act).properties(
        width=620,
        height=200,
        title=alt.TitleParams(
            text=title,
            subtitle=[
                "Dark line: what happened. Bands: 50% and 90% intervals of the two-seed mean."
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=10,
        ),
    )  # ty: ignore[unresolved-attribute]
    save(chart=chart, name=name)


def main() -> None:
    contrasts_figure()
    slices_figure()
    for primary, label, group in (
        (3, "D5", "signed-MW"),
        (11, "D16", "signed-MW"),
        (0, "D2", "MVA-only"),
    ):
        example_figure(
            name=f"example_day_ahead_christmas_{label.lower()}",
            title=f"Day-ahead forecasts over Christmas 2025, Primary {label} ({group})",
            lo="2025-12-22",
            hi="2026-01-01",
            day_ahead=True,
            origin=None,
            primary=primary,
        )
    example_figure(
        name="example_full_horizon_d5",
        title="One forecast run (issued 2025-12-15 09:00 UTC), full 10-day horizon, Primary D5",
        lo="2025-12-15 09:00",
        hi="2025-12-25 09:00",
        day_ahead=False,
        origin="2025-12-15 09:00",
        primary=3,
    )


if __name__ == "__main__":
    main()
