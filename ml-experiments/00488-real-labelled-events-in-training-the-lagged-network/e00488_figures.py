# ruff: noqa: E501, PERF401, RUF028
"""Figures of experiment 488: headline contrasts, event time series, example forecasts, sweep.

Primaries are substations, so their series carry no anonymisation requirement. Event plots use the
corrected label table of experiment 400 and day-ahead forecasts (12 to 36 hours ahead).

```bash
PYTHONPATH=<as run488.sh> DATA_PATH_INTERNAL=<private data> \
    uv run --no-sync python ml-experiments/00488-*/e00488_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00404_score as score  # ty: ignore[unresolved-import]
import e00488_fit as fit488  # ty: ignore[unresolved-import]
import e00488_score as score488  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOC_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00488-real-labelled-events-in-training-the-lagged-network"
)
ARMS: Final[tuple[str, ...]] = ("A0-inj", "B-mix", "RealMix", "PT-FT")
COLOURS: Final[list[str]] = [theme.BLACK_1, theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE]
LEAD: Final[tuple[float, float]] = (12.0, 36.0)
SITES: Final[list[str]] = []


def sites() -> list[str]:
    """Return the Primary names."""
    if not SITES:
        SITES.extend(fit488.prep.primaries()["site"].to_list())
    return SITES


def day_ahead(*, arm: str, month: str, primary: int) -> pl.DataFrame:
    """Return an arm's forecasts issued 12 to 36 hours ahead for one Primary, clean world."""
    frame = pl.read_parquet(score.prediction_path(arm=arm, month=month, seed=0, world=-1))
    return frame.filter((pl.col("primary") == primary) & pl.col("lead_hours").is_between(*LEAD))


def pick_events() -> list[tuple[str, str, int, int]]:
    """Return six events (four at the signed-MW six, two at the eight), over several months."""
    events = pl.read_parquet(fit488.EVENTS_PATH).with_columns(
        length=pl.col("end_idx") - pl.col("onset_idx")
    )
    chosen = []
    for month in score488.MONTHS:
        frame = pl.read_parquet(score.prediction_path(arm="B-mix", month=month, seed=0, world=-1))
        lo, hi = int(frame["target"].min()), int(frame["target"].max())  # ty: ignore[invalid-argument-type]
        inside = events.filter(
            (pl.col("onset_idx") >= lo + 150)
            & (pl.col("end_idx") <= hi - 150)
            & (pl.col("length") > 20)
        )
        for group, size in ((score.SIX_SIGNED, 1), (None, 1)):
            part = inside.filter(
                pl.col("label").is_in(group) if group else ~pl.col("label").is_in(score.SIX_SIGNED)
            ).sort("length", descending=True)
            for row in part.head(size).iter_rows(named=True):
                chosen.append((row["label"], month, row["onset_idx"], row["end_idx"]))
    return chosen[:6] if len(chosen) >= 6 else chosen


def event_figure(*, site: str, month: str, onset: int, end: int, number: int) -> None:
    """Draw one real event's day-ahead forecasts for A0-inj, B-mix, RealMix, and PT-FT."""
    primary = sites().index(site)
    lo, hi = onset - 144, end + 144
    parts = []
    for arm in ARMS:
        part = day_ahead(arm=arm, month=month, primary=primary).filter(
            pl.col("target").is_between(lo, hi)
        )
        parts.append(
            part.select(
                "target",
                (pl.col("q50") * 100).alias("v"),
                (pl.col("q05") * 100).alias("low"),
                (pl.col("q95") * 100).alias("high"),
            ).with_columns(series=pl.lit(arm))  # fmt: skip
        )
    first = day_ahead(arm="B-mix", month=month, primary=primary).filter(
        pl.col("target").is_between(lo, hi)
    )
    actual = first.select("target", (pl.col("actual") * 100).alias("v")).with_columns(
        series=pl.lit("Actual")
    )
    lines = pl.concat([actual, *[p.select(actual.columns) for p in parts]]).with_columns(
        day=(pl.col("target") - onset) / 48.0
    )
    band = parts[1].with_columns(day=(pl.col("target") - onset) / 48.0)
    x = alt.X("day:Q", title="Days since the event began (shaded: the event)")
    shade = (
        alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - onset) / 48.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(color=theme.GREY_3, opacity=0.35)
        .encode(x="a:Q", x2="b:Q")
    )
    area = (
        alt.Chart(band.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_area(color=theme.BRAND_ORANGE, opacity=0.18)
        .encode(x=x, y=alt.Y("low:Q", title="Power (% of capacity)"), y2="high:Q")
    )
    domain = ["Actual", *ARMS]
    line = (
        alt.Chart(lines.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(aria=False)
        .encode(
            x=x,
            y="v:Q",
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=[theme.GREY_1, *COLOURS]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )  # fmt: skip
    )
    chart = (shade + area + line).properties(
        width=570, height=190,
        title=alt.TitleParams(
            [f"Primary {site}, real event in {month}: day-ahead forecasts (12 to 36 hours ahead)",
             "Orange band: B-mix 90% interval. Lines: medians. Actual readings include the event."],
            anchor="start",
        ),
    )  # fmt: skip
    chart.save(DOC_DIR / f"figure_{number}_event_{site}_{month}.svg")


def example_figure(*, number: int, month: str, site: str, days: int) -> None:
    """Draw one forecast run: its median and 50% and 90% bands against the actual."""
    primary = sites().index(site)
    frame = pl.read_parquet(score.prediction_path(arm="RealMix", month=month, seed=0, world=-1))
    frame = frame.filter(pl.col("primary") == primary)
    origin = int(frame["origin"].unique().sort()[len(frame["origin"].unique()) // 2])
    run = frame.filter(pl.col("origin") == origin).sort("target")
    run = run.filter(pl.col("lead_hours") <= days * 24.0).with_columns(
        hours=pl.col("lead_hours"), **{k: pl.col(k) * 100 for k in ("q05", "q25", "q50", "q75", "q95", "actual")}
    )  # fmt: skip
    base = alt.Chart(run.to_pandas()).encode(
        x=alt.X("hours:Q", title="Hours ahead of the forecast")
    )  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_area(color=theme.BRAND_ORANGE, opacity=0.15).encode(y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_area(color=theme.BRAND_ORANGE, opacity=0.3).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_line(color=theme.BRAND_ORANGE, aria=False).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_line(color=theme.BLACK_1, aria=False).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=570, height=190,
        title=alt.TitleParams(
            [f"Primary {site}, RealMix forecast issued in {month}, {days}-day horizon",
             "Orange: median, 50% and 90% bands. Black: actual."], anchor="start"),
    )  # fmt: skip
    chart.save(DOC_DIR / f"figure_{number}_example_{site}_{month}_{days}d.svg")


def headline_figure() -> None:
    """Draw the contrasts against A0-inj (signed-MW six, raw) with Bonferroni intervals."""
    t = pl.read_csv(score488.OUT_DIR / "pooled.csv").filter(
        (pl.col("variant") == "raw") & (pl.col("group") == "signed_six")
        & (pl.col("ref") == "A0-inj") & pl.col("arm").is_in(["RealMix", "PT-FT", "RealMix-shuf"])
        & ((pl.col("cls").is_in(["harm_inj", "clean_real", "clean_inj"]) & (pl.col("depth") == "metered"))
           | ((pl.col("cls") == "event") & (pl.col("depth") == "1.0")))
    ).with_columns(
        rowclass=pl.col("cls").replace({"harm_inj": "Injected harm rows", "clean_real": "Rows with no labelled event",
                                        "clean_inj": "Injected worlds, no event", "event": "Inside real events (NRA-corrected)"})
    )  # fmt: skip
    order = ["RealMix", "RealMix-shuf", "PT-FT"]
    palette = alt.Scale(
        domain=order, range=[theme.DATA_BLUE, theme.DATA_PURPLE, theme.BRAND_ORANGE]
    )
    panels = []
    for name in (
        "Injected harm rows", "Inside real events (NRA-corrected)",
        "Rows with no labelled event", "Injected worlds, no event",
    ):  # fmt: skip
        frame = t.filter(pl.col("rowclass") == name).to_pandas()  # ty: ignore[unresolved-attribute]
        colour = alt.Color("arm:N", legend=None, scale=palette)
        y = alt.Y("arm:N", title=None, sort=order)
        bars = (
            alt.Chart(frame).mark_rule().encode(x="low_bonf:Q", x2="high_bonf:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        )
        points = alt.Chart(frame).mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("mean_of_months:Q", title="Arm minus A0-inj (points of capacity; below 0 is better)"),
            y=y, color=colour,
        )  # fmt: skip
        zero = (
            alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color=theme.GREY_1)
            .encode(x="z:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append((zero + bars + points).properties(width=330, height=70, title=name))
    chart = alt.vconcat(*panels).properties(
        title="Five-month mean difference against A0-inj, signed-MW six; bars: month-block interval (Bonferroni, k = 12)"
    )
    chart.save(DOC_DIR / "figure_1_contrasts_against_a0_inj.svg")


def sweep_figure() -> None:
    """Draw the median shift as a share of a 5% step, by number of lags covered."""
    frames = [pl.read_parquet(p) for p in sorted((fit488.WORK_DIR / "sweep").glob("*.parquet"))]
    f = (
        pl.concat(frames)
        .filter(pl.col("kind") == "step")
        .with_columns(share=pl.col("shift") / pl.col("step"))
    )
    t = f.group_by("arm", "k").agg(pl.col("share").mean()).to_pandas()  # ty: ignore[unresolved-attribute]
    chart = alt.Chart(t).mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("arm:N", title=None, sort=None), y=alt.Y("share:Q", title="Median shift as a share of the step"),
        color=alt.Color("arm:N", legend=None, scale=alt.Scale(scheme="tableau10")), column=alt.Column("k:O", title="Lags covered by the step"),
    ).properties(width=150, height=150, title="Encroachment sweep: a 5% step in the newest k lags")  # fmt: skip
    chart.save(DOC_DIR / "figure_2_sweep_shift.svg")


def main() -> None:
    """Draw every figure."""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(0)
    headline_figure()
    sweep_figure()
    for i, (site, month, onset, end) in enumerate(pick_events(), start=3):
        event_figure(site=site, month=month, onset=onset, end=end, number=i)
    example_figure(number=9, month="2026-01", site="D5", days=1)
    example_figure(number=10, month="2026-06", site="D9", days=10)
    example_figure(number=11, month="2026-01", site="D5", days=10)


if __name__ == "__main__":
    main()
