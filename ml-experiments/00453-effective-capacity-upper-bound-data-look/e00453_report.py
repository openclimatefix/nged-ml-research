"""Experiment 453: print the numbers for the write-up and draw its figures as SVG.

Reads `work/*.parquet` from `e00453_tail.py` and `e00453_capacity.py`. Writes four SVG figures to
the experiment's docs folder (optimise them afterwards with `npx svgo@4 --multipass --precision=1
--final-newline f.svg`) and prints tables for the write-up.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00451-*/env.sh
uv run --no-sync python ml-experiments/00453-*/e00453_report.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00453_data as data  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00453-effective-capacity-upper-bound-data-look"
)
RULE_ORDER: Final[list[str]] = ["p99", "p99.9", "p99.99", "max"]
SITES: Final[list[str]] = [
    "Farm A", "Farm B", "Farm C", "Solar A", "Solar B", "Solar C", "Solar D", "Solar E", "Solar F",
]  # fmt: skip
BOOTSTRAPS: Final[int] = 1000


def fig_sorted_tail(*, series: pl.DataFrame) -> None:
    """Sorted upper tail: the top 1,500 half-hours of each series, power over capacity."""
    pieces = []
    for site in SITES:
        x = (
            series.filter(pl.col("site") == site)
            .select(x=pl.col("power") / pl.col("cap"))["x"]
            .sort(descending=True)
            .head(1500)
        )
        pieces.append(pl.DataFrame({"site": site, "rank": np.arange(1, len(x) + 1), "x": x}))
    chart = (
        alt.Chart(pl.concat(pieces))
        .mark_line(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="Rank from the top (log scale)"),
            y=alt.Y("x:Q", scale=alt.Scale(zero=False), title="Power over capacity"),
        )
        .properties(width=170, height=120)
        .facet(facet=alt.Facet("site:N", sort=SITES, title=None), columns=3)
    )
    chart.save(DOCS / "figure_1_sorted_tail.svg")


def fig_time(*, high: pl.DataFrame, summary: pl.DataFrame) -> None:
    """Half-hours above each series' p99.9 or classed implausible, against time."""
    cut = summary.select("site", "p99.9")
    frame = (
        high.join(cut, on="site")
        .filter((pl.col("x") > pl.col("p99.9")) | pl.col("implausible"))
        .with_columns(
            kind=pl.when(pl.col("implausible"))
            .then(pl.lit("implausible"))
            .otherwise(pl.lit("plausible"))
        )
        .select("site", "time", "x", "kind")
    )
    chart = (
        alt.Chart(frame)
        .mark_circle(size=6, opacity=0.6)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%Y", tickCount=4)),
            y=alt.Y("x:Q", scale=alt.Scale(zero=False), title="Power over capacity"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=["plausible", "implausible"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
            ),
        )
        .properties(width=170, height=120)
        .facet(facet=alt.Facet("site:N", sort=SITES, title=None), columns=3)
    )
    chart.save(DOCS / "figure_2_high_halfhours_over_time.svg")


def fig_levels() -> None:
    """The monthly output level at strong wind or clear sky, by series."""
    pieces = [
        pl.read_parquet(p).with_columns(
            site=pl.lit(p.stem.removeprefix("months_").replace("_", " "))
        )
        for p in sorted(data.WORK_DIR.glob("months_*.parquet"))
    ]
    chart = (
        alt.Chart(pl.concat(pieces))
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount=4)),
            y=alt.Y("level:Q", scale=alt.Scale(zero=False), title="Monthly 90th percentile"),
        )
        .properties(width=170, height=120)
        .facet(facet=alt.Facet("site:N", sort=SITES, title=None), columns=3)
    )
    chart.save(DOCS / "figure_3_monthly_level.svg")


def fig_shares(*, rows: pl.DataFrame) -> None:
    """Share of actuals above q99 after table P, by rule and series."""
    table = _summary(rows=rows).filter((pl.col("variant") == "P") & (pl.col("rule") != "current"))
    limit = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = (
        alt.Chart(table)
        .mark_circle(size=60)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("share_gt_q99:Q", title="Share of actuals above q99 (%)"),
            y=alt.Y("site:N", sort=SITES, title=None),
            color=alt.Color(
                "rule:N",
                sort=RULE_ORDER,
                title="Capacity rule",
                scale=alt.Scale(
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, "#555555"]
                ),
            ),
        )
    )
    (points + limit).properties(
        width=330, title="The 1.5% limit is met sooner at higher percentiles"
    ).save(DOCS / "figure_4_share_above_q99.svg")


def _summary(*, rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by("site", "rule", "variant")
        .agg(pl.col("n").sum(), pl.col("n_gt_1").sum(), pl.col("n_gt_q99").sum(),
             pl.col("score_mw_sum").sum(), pl.col("cap_mw").mean())
        .with_columns(share_gt_1=pl.col("n_gt_1") / pl.col("n") * 100,
                      share_gt_q99=pl.col("n_gt_q99") / pl.col("n") * 100,
                      score_mw=pl.col("score_mw_sum") / pl.col("n"))
    )  # fmt: skip


def _change_vs_p99(*, per: pl.DataFrame, sites: list[str], rule: str, sample: list[str]) -> float:
    """Return the mean over sites of the percentage change in MW score of `rule` against p99."""
    weight = pl.DataFrame({"month": sample}).group_by("month").len()
    changes = []
    for site in sites:
        rows = (
            per.filter(pl.col("site") == site)
            .join(weight, on="month")
            .with_columns(score=pl.col("score_mw_sum") * pl.col("len"))
        )
        value = rows.filter(pl.col("rule") == rule)["score"].sum()
        base = rows.filter(pl.col("rule") == "p99")["score"].sum()
        changes.append(value / base - 1.0)
    return float(np.mean(changes)) * 100


def score_ratios(*, rows: pl.DataFrame) -> None:
    """Print each rule's MW score against p99's, per technology, with a month-block interval."""
    rng = np.random.default_rng(0)
    for tech, sites in (("wind", SITES[:3]), ("solar", SITES[3:])):
        part = rows.filter(pl.col("site").is_in(sites) & (pl.col("variant") == "P"))
        months = sorted(part["month"].unique())
        per = part.group_by("site", "month", "rule").agg(pl.col("score_mw_sum").sum())
        for rule in ("p99.9", "p99.99", "max"):
            point = _change_vs_p99(per=per, sites=sites, rule=rule, sample=months)
            boots = [
                _change_vs_p99(
                    per=per, sites=sites, rule=rule, sample=list(rng.choice(months, len(months)))
                )
                for _ in range(BOOTSTRAPS)
            ]
            low, high = np.percentile(boots, [2.5, 97.5])
            print(f"{tech} {rule}: MW score vs p99 {point:+.2f}% (95% {low:+.2f} to {high:+.2f})")


def plausibility_by_threshold(*, high: pl.DataFrame, summary: pl.DataFrame) -> None:
    """Print, per series and threshold, the half-hours at or above it and the plausible share.

    `all flags` counts spike, stuck, level-shift, and weather flags; `no level` leaves the
    level-shift flag out, because a high-level month is real output at an earlier capacity.
    """
    for site in SITES:
        row = summary.filter(pl.col("site") == site).row(0, named=True)
        part = high.filter(pl.col("site") == site)
        cells = []
        for name in ("max", "p99.99", "p99.9"):
            above = part.filter(pl.col("x") >= row[name])
            no_level = above.filter(~(pl.col("spike") | pl.col("stuck") | pl.col("weather_off")))
            plausible = int((~above["implausible"]).sum())
            cells.append(f"{name}: {above.height} half-hours, {plausible} plausible "
                         f"({no_level.height} with level shift ignored)")  # fmt: skip
        print(site, "; ".join(cells))
    wind = high.filter(pl.col("tech") == "wind")
    print(
        wind.group_by("site")
        .agg(
            pl.col("wind").median().alias("median_wind_m_s"),
            pl.col("wind").quantile(0.1).alias("p10_wind_m_s"),
            pl.col("wind").is_null().sum().alias("no_wind"),
        )
        .sort("site")
    )
    solar = high.filter(pl.col("tech") == "solar")
    print(
        solar.group_by("site")
        .agg(
            pl.col("clear").median().alias("median_clear"),
            pl.col("clear").quantile(0.1).alias("p10_clear"),
        )
        .sort("site")
    )


def main() -> None:
    """Print tables and write figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    series = pl.read_parquet(data.SERIES_PATH)
    high = pl.read_parquet(data.WORK_DIR / "high_halfhours.parquet")
    rows = pl.read_parquet(data.WORK_DIR / "capacity_rows.parquet")
    summary = pl.read_parquet(data.WORK_DIR / "tail_summary.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220):
        print(summary)
        # dates: the number of distinct days and the first and last date above each threshold
        for site in SITES:
            row = summary.filter(pl.col("site") == site).row(0, named=True)
            part = series.filter(pl.col("site") == site).with_columns(
                x=pl.col("power") / pl.col("cap")
            )
            for name in ("p99.99", "p99.9"):
                above = part.filter(pl.col("x") > row[name])
                print(
                    site,
                    name,
                    "half-hours",
                    above.height,
                    "days",
                    above["time"].dt.date().n_unique(),
                    "first",
                    above["time"].min(),
                    "last",
                    above["time"].max(),
                )
            top = part.sort("x", descending=True).head(3).select("time", "x")
            print(site, "top three:", top.rows())
        print(
            high.group_by("site")
            .agg(
                pl.len().alias("above_1"),
                (~pl.col("implausible")).sum().alias("plausible"),
                pl.col("spike").sum().alias("spike"),
                pl.col("stuck").sum().alias("stuck"),
                pl.col("step").sum().alias("level_shift"),
                pl.col("weather_off").sum().alias("weather_off"),
                pl.col("implausible").sum().alias("implausible"),
            )
            .sort("site")
        )
        print(
            _summary(rows=rows)
            .sort("site", "variant", "cap_mw")
            .drop("n_gt_1", "n_gt_q99", "score_mw_sum")
        )
    plausibility_by_threshold(high=high, summary=summary)
    score_ratios(rows=rows)
    fig_sorted_tail(series=series)
    fig_time(high=high, summary=summary)
    fig_levels()
    fig_shares(rows=rows)


if __name__ == "__main__":
    main()
