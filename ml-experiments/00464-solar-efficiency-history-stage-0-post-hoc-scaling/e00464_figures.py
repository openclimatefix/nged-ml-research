# ruff: noqa: E501
"""Figures for experiment 464 (anonymised: farms are Solar A to Solar F; axes carry no calendar dates).

PYTHONPATH=<310>:<430>:<464 folders> uv run --no-sync python ml-experiments/00464-*/e00464_figures.py
"""

from pathlib import Path

import altair as alt
import e00464_common as common
import e00464_score as sc
import plotting.ocf_theme as ocf
import polars as pl

DOCS = Path("docs/ml-experiments/00464-solar-efficiency-history-stage-0-post-hoc-scaling")
WIDTH = 680
FARM_COLOURS = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    "#555555",
]
LABELS = {f: f"Solar {f}" for f in common.FARMS}
BEST_ARM = sc.Arm("a2_member", "c", "med14", normalised=True, lag_days=2)


def fig_contrasts() -> None:
    """Point estimate of every primary arm, per farm, days 1 to 10."""
    t = (
        pl.read_csv(common.SCORE_DIR / "contrasts.csv")
        .filter(
            (pl.col("band") == "all")
            & (pl.col("group").str.len_chars() == 1)
            & pl.col("arm").str.starts_with("a2_member|")
            & pl.col("arm").str.contains(r"\|(b|c)\|")
            & pl.col("arm").str.ends_with("|norm|lag2")
        )
        .with_columns(
            farm=pl.col("group").replace_strict(LABELS),
            source=pl.col("arm")
            .str.split("|")
            .list.get(1)
            .replace_strict({"b": "(b) clear-sky", "c": "(c) weather model"}),
        )
    )
    base = alt.Chart(t).encode(
        y=alt.Y("farm:N", title=None),
        x=alt.X("pct:Q", title="Change in mean quantile score (%; negative is better)"),
    )
    points = base.mark_circle(size=60, opacity=0.8).encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "source:N",
            title=None,
            scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top"),
        )
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    margin = (
        alt.Chart(pl.DataFrame({"x": [-1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    chart = (zero + margin + points).properties(
        width=WIDTH - 90,
        height=170,
        title=alt.TitleParams(
            "Scaling by recent efficiency does not lower any farm's score",
            subtitle=[
                "Each dot is one of 18 scaling methods on the stored a2_member forecast, days 1 to 10.",
                "Solid rule: no change. Dashed rule: the 1% gain the plan asks for.",
            ],
        ),
    )
    chart.save(DOCS / "figure_1_contrasts.svg")


def fig_ratio() -> None:
    """Daily ratio and its 56-day trailing median for source (c), per farm."""
    daily = pl.read_parquet(common.RATIO_DIR / "daily_c.parquet")
    est = pl.read_parquet(common.RATIO_DIR / "estimates_c.parquet").filter(
        pl.col("refit") == common.REFITS[-1]
    )
    start = daily.group_by("farm").agg(first=pl.col("date").min())
    daily = daily.join(start, on="farm").with_columns(
        years=(pl.col("date") - pl.col("first")).dt.total_days() / 365.25,
        farm=pl.col("farm").replace_strict(LABELS),
    )
    est = est.join(start, on="farm").with_columns(
        years=(pl.col("date") - pl.col("first")).dt.total_days() / 365.25,
        farm=pl.col("farm").replace_strict(LABELS),
    )
    both = pl.concat(
        [
            daily.group_by("farm", (pl.col("years") * 52).floor().alias("week"))
            .agg(pl.col("years").mean(), pl.col("ratio").median().alias("value"))
            .select("farm", "years", "value", kind=pl.lit("day")),
            est.drop_nulls("med56").select(
                "farm", "years", value=pl.col("med56"), kind=pl.lit("median")
            ),
        ]
    )
    scale = alt.Scale(domain=[0.4, 1.6])
    panels = []
    for farm_label in LABELS.values():
        part = both.filter(pl.col("farm") == farm_label)
        pts = (
            alt.Chart(part.filter(pl.col("kind") == "day"))  # ty: ignore[unresolved-attribute]
            .mark_circle(size=6, opacity=0.25, color=ocf.DATA_BLUE)
            .encode(
                x=alt.X("years:Q", title="Years since the farm's first record"),
                y=alt.Y("value:Q", title=farm_label, scale=scale),
            )
        )
        line = (
            alt.Chart(part.filter(pl.col("kind") == "median"))  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2)
            .encode(x="years:Q", y=alt.Y("value:Q", scale=scale))
        )
        panels.append((pts + line).properties(width=WIDTH - 60, height=80))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Daily efficiency ratio and its 56-day trailing median",
            subtitle=[
                "Blue: weekly median of the daily power over the forward-only weather model's expected power.",
                "Orange: weighted median of the last 56 days. 1.0 is the average over earlier years.",
            ],
        )
    ).save(DOCS / "figure_2_ratio_over_years.svg")


def _base_and_scaled() -> tuple[pl.DataFrame, pl.DataFrame]:
    base = sc.add_refit(frame=sc.s430.load_mine(arm="a2_member")).with_columns(
        refit=pl.col("refit").dt.convert_time_zone("UTC")
    )
    caps = base.group_by("time_series_id", "refit").agg(cap=pl.col("q99").max())
    return base, sc.scale(base=base, arm=BEST_ARM, caps=caps)


def fig_explain(*, base: pl.DataFrame) -> float:
    """Monthly efficiency of source (c) against the unscaled forecast's monthly bias."""
    est = pl.read_parquet(common.RATIO_DIR / "estimates_c.parquet").filter(
        pl.col("refit") == common.REFITS[-1]
    )
    eff = est.filter(pl.col("date").dt.day() == 1).with_columns(
        month=pl.col("date").dt.strftime("%Y-%m"), efficiency=pl.col("med56") / pl.col("reference")
    )
    bias = (
        base.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("time_series_id", "month")
        .agg(bias=(pl.col("q50") - pl.col("actual")).mean() * 100)
        .rename({"time_series_id": "farm"})
    )
    joined = (
        eff.select("farm", "month", "efficiency")
        .join(bias, on=["farm", "month"])
        .filter(pl.col("efficiency").is_finite())
    )
    print("fig3 points", joined.height)
    corr = float(joined.select(pl.corr("efficiency", "bias")).item())
    joined = joined.with_columns(farm=pl.col("farm").replace_strict(LABELS))
    chart = (
        alt.Chart(joined)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X(
                "efficiency:Q",
                scale=alt.Scale(domain=[0.75, 1.2]),
                title="Trailing 56-day efficiency at the start of the month (1.0 = earlier average)",
            ),
            y=alt.Y("bias:Q", title="Mean forecast median minus actual (points of capacity)"),
            color=alt.Color(
                "farm:N",
                title=None,
                scale=alt.Scale(range=FARM_COLOURS),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=260,
            title=alt.TitleParams(
                "Recent efficiency does not explain the forecast's monthly bias",
                subtitle=[
                    f"One dot per farm and month of the scored year. Correlation {corr:+.2f}.",
                    "Positive bias: the forecast median is above what the farm produced.",
                ],
            ),
        )
    )
    chart.save(DOCS / "figure_3_efficiency_vs_bias.svg")
    return corr


def fig_examples(*, base: pl.DataFrame, scaled: pl.DataFrame) -> None:
    """A day-ahead forecast and a full 10-day forecast of one run of Solar B, scaled and unscaled."""
    run = (
        base.filter((pl.col("time_series_id") == "B") & (pl.col("init_time").dt.month() == 6))
        .select("init_time")
        .unique()
        .sort("init_time")["init_time"][15]
    )
    frames = []
    for label, frame in (("unscaled", base), (f"scaled ({BEST_ARM.method}, source c)", scaled)):
        part = frame.filter(
            (pl.col("time_series_id") == "B") & (pl.col("init_time") == run)
        ).with_columns(variant=pl.lit(label), hours=pl.col("lead_hours"))
        frames.append(part.select("variant", "hours", "actual", "q05", "q25", "q50", "q75", "q95"))
    data = pl.concat(frames).with_columns(
        day=pl.col("hours") / 24.0, hour_of_day=pl.col("hours") % 24
    )

    def panel(df: pl.DataFrame, x: str, title: str, xtitle: str, dom: list[float]) -> alt.Chart:
        enc = alt.X(f"{x}:Q", title=xtitle, scale=alt.Scale(domain=dom))
        colour = alt.Color(
            "variant:N",
            title=None,
            scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top"),
        )
        band90 = (
            alt.Chart(df).mark_area(opacity=0.12).encode(x=enc, y="q05:Q", y2="q95:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        band50 = (
            alt.Chart(df).mark_area(opacity=0.25).encode(x=enc, y="q25:Q", y2="q75:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        med = (
            alt.Chart(df)  # ty: ignore[unresolved-attribute]
            .mark_line()
            .encode(x=enc, y=alt.Y("q50:Q", title="Power / capacity"), color=colour)
        )
        act = (
            alt.Chart(df.filter(pl.col("variant") == "unscaled"))  # ty: ignore[unresolved-attribute]
            .mark_line(color="black", strokeDash=[3, 2])
            .encode(x=enc, y="actual:Q")
        )
        return (band90 + band50 + med + act).properties(width=WIDTH - 70, height=170, title=title)

    day1 = data.filter((pl.col("hours") >= 12) & (pl.col("hours") <= 36))
    chart = alt.vconcat(
        panel(day1, "hours", "Day ahead (lead 12 to 36 hours)", "Lead time (hours)", [12, 36]),
        panel(data, "day", "Full 10-day horizon of the same run", "Lead time (days)", [0.5, 10]),
    ).properties(
        title=alt.TitleParams(
            "Example forecasts for Solar B, scaled and unscaled",
            subtitle=[
                "Line: median. Dark band: 50% interval. Light band: 90% interval. Dashed black: actual.",
                "Daylight and night both drawn; one 00 UTC run in June 2026.",
            ],
        )
    )
    chart.save(DOCS / "figure_4_examples.svg")


def main() -> None:
    """Write every figure and print the correlation of figure 3."""
    fig_contrasts()
    fig_ratio()
    base, scaled = _base_and_scaled()
    print("corr", fig_explain(base=base))
    fig_examples(base=base, scaled=scaled)


if __name__ == "__main__":
    main()
