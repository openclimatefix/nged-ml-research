# ruff: noqa: E501
"""Figures and numbers for experiment 409's write-up.

Reads the stored seed forecasts, solar outputs, and capacities, and the printed scoring report
(`work/score_output.txt`), and writes the figures beside `result.md`.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00409_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00409-mdn-solar-term-day-of-year-and-anomaly-known-answer"
)
WORK: Final[Path] = sc.WORK
SIX: Final = sc.SIGNED
STUDIES: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs/studies/assets/embedded_pv_capacity_table.csv"
)


def contrasts() -> pl.DataFrame:
    """Parse the contrast lines of the printed scoring report."""
    rows = []
    pattern = re.compile(
        r"(\S+) minus (\S+) \[(\w+)\]: ([+-][\d.]+) \(([+-][\d.]+) to ([+-][\d.]+)\) MDE ([\d.]+|nan)"
    )
    for line in (WORK / "score_output.txt").read_text().splitlines():
        m = pattern.match(line)
        if m:
            a, b, sl, d, lo, hi, mde = m.groups()
            rows.append(
                {
                    "contrast": f"{a} minus {b}",
                    "slice": sl,
                    "difference": float(d),
                    "lower": float(lo),
                    "upper": float(hi),
                    "mde": float(mde),
                }
            )
    return pl.DataFrame(rows)


def figure_contrasts() -> None:
    """Draw each contrast with its interval, by slice."""
    data = contrasts().filter(~pl.col("contrast").str.ends_with("minus A0")).to_pandas()
    names = {
        "lead_days_1_10": "All half-hours",
        "sunny_unit_shape": "Sunniest third",
        "sunny_april_june": "Sunny Apr-Jun (1 month)",
    }
    data["slice"] = data["slice"].map(names)
    y = alt.Y("contrast:N", sort=list(dict.fromkeys(data["contrast"])), title=None)
    colour = alt.Color(
        "slice:N",
        scale=alt.Scale(
            domain=list(names.values()),
            range=[theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_PURPLE],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    bars = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X(
                "lower:Q",
                title="CRPS difference (points of capacity; below zero favours the first arm)",
            ),
            x2="upper:Q",
            color=colour,
            yOffset="slice:N",
        )
    )  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(y=y, x="difference:Q", color=colour, yOffset="slice:N")
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (bars + dots + zero).properties(
        width=560,
        height=330,
        title=alt.TitleParams(
            [
                "Contrasts at the six signed-direction Primaries, five test months (interim)",
                "Line: 99.29% month-block interval (Bonferroni across seven contrasts).",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS / "figure_1_contrasts.svg")


def figure_solar() -> None:
    """Draw the learned solar output by hour for a winter and a summer month, C3 against C1."""
    series = adapter.series_table()
    label = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    pieces = []
    for arm in ("C3", "C1"):
        for month in ("2025-12", "2026-06"):
            frames = [
                pl.read_parquet(WORK / "seed_preds" / arm / f"{month}_s{s}.solar.parquet")
                for s in sc.SEEDS
            ]
            frame = pl.concat(frames).with_columns(
                site=pl.col("time_series_id").replace_strict(label, default=None)
            )
            pieces.append(
                frame.filter(pl.col("site").is_in(list(SIX)))
                .with_columns(
                    hour=pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60
                )
                .group_by("site", "hour")
                .agg(pl.col("solar_mw").mean())
                .with_columns(arm=pl.lit(arm), month=pl.lit(month))
            )
    data = pl.concat(pieces).with_columns(series=pl.col("arm") + " " + pl.col("month")).to_pandas()
    domain = ["C3 2025-12", "C1 2025-12", "C3 2026-06", "C1 2026-06"]
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hour:Q", title="Hour of day (UTC)"),
            y=alt.Y("solar_mw:Q", title="Solar output (MW)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[
                        theme.DATA_BLUE,
                        theme.DATA_BLUE_LIGHT,
                        theme.BRAND_ORANGE,
                        theme.BRAND_ORANGE_LIGHT,
                    ],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(width=170, height=130)
        .facet(facet=alt.Facet("site:N", sort=list(SIX), title=None), columns=3)
        .properties(
            title=alt.TitleParams(
                [
                    "Learned solar output (C3, learned capacity) against the pinned-capacity solar term (C1)",
                    "Mean over test half-hours and 3 seeds, December 2025 and June 2026.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(DOCS / "figure_2_learned_solar_output.svg")


def capacities() -> pl.DataFrame:
    """Return each arm's learned capacity at the as-of season-year, per Primary, fold, and seed."""
    rows = []
    for arm in ("C2", "C3", "C4"):
        for month in sc.MONTHS:
            for s in sc.SEEDS:
                cap = json.loads(
                    (WORK / "seed_preds" / arm / f"{month}_s{s}.capacity.json").read_text()
                )
                rows += [
                    {
                        "arm": arm,
                        "month": month,
                        "seed": s,
                        "site": site,
                        "capacity_mw": cap[site][2],
                    }
                    for site in SIX
                ]
    return pl.DataFrame(rows)


def figure_capacity() -> pl.DataFrame:
    """Draw learned capacity per Primary against the studies page, and return the summary."""
    cap = capacities()
    studies = (
        pl.read_csv(STUDIES)
        .filter(pl.col("season_year") == 2025, pl.col("site").is_in(list(SIX)))
        .select("site", "capacity_mw", "capacity_low_mw", "capacity_high_mw")
    )
    summary = cap.group_by("arm", "site").agg(
        mean=pl.col("capacity_mw").mean(),
        lo=pl.col("capacity_mw").min(),
        hi=pl.col("capacity_mw").max(),
    )
    s = studies.rename(
        {"capacity_mw": "mean", "capacity_low_mw": "lo", "capacity_high_mw": "hi"}
    ).with_columns(arm=pl.lit("Studies page (95% interval)"))
    data = pl.concat([summary, s.select(summary.columns)]).to_pandas()
    domain = ["Studies page (95% interval)", "C2", "C3", "C4"]
    y = alt.Y("site:N", sort=list(SIX), title=None)
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=domain,
            range=[
                theme.GREY_5 if hasattr(theme, "GREY_5") else "#666666",
                theme.DATA_BLUE,
                theme.BRAND_ORANGE,
                theme.DATA_PURPLE,
            ],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    bars = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X("lo:Q", title="Solar capacity, 2025/26 season-year (MW AC)"),
            x2="hi:Q",
            color=colour,
            yOffset="arm:N",
        )
    )  # ty: ignore[unresolved-attribute]
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50)
        .encode(y=y, x="mean:Q", color=colour, yOffset="arm:N")
    )  # ty: ignore[unresolved-attribute]
    chart = (bars + dots).properties(
        width=520,
        height=260,
        title=alt.TitleParams(
            [
                "Learned capacity against the studies page",
                "Dot: mean over 5 folds and 3 seeds. Line: range over folds and seeds (studies page: 95% interval).",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS / "figure_3_learned_capacity.svg")
    return summary.join(studies, on="site", suffix="_studies").sort("arm", "site")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of C3 and A0-noirr-doy for one Primary, two seasons."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    sid = {r[1]: r[0] for r in series.iter_rows()}
    actuals = ctx["actuals"]
    pieces = []
    for arm in ("A0-noirr-doy", "C3"):
        frame = adapter.forecast_frame(
            stored=sc.arm_frame(arm=arm, seeds=sc.SEEDS), series=series, unit="fraction"
        )
        for site, origin in (
            ("D10", datetime(2026, 1, 15, 9, tzinfo=UTC)),
            ("D10", datetime(2026, 6, 15, 9, tzinfo=UTC)),
        ):
            f = frame.filter(
                pl.col("time_series_id") == sid[site], pl.col("power_fcst_init_time") == origin
            )
            pieces.append(f.with_columns(arm=pl.lit(arm), origin=pl.lit(origin), site=pl.lit(site)))
    data = pl.concat(pieces).join(
        actuals.filter(pl.col("time_series_id") == sid["D10"]),
        on=["time_series_id", "valid_time"],
        how="left",
    )
    data.write_parquet(WORK / "example_forecasts.parquet")
    print(data.columns, data.height)
    for _window, hours, name in (
        ("day_ahead", 36, "figure_4_example_day_ahead.svg"),
        ("fourteen_day", 336, "figure_5_example_14_day.svg"),
    ):
        d = (
            data.with_columns(
                hours=(pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60
            )
            .filter(pl.col("hours") <= hours, pl.col("hours") >= 0)
            .to_pandas()
        )
        d["origin_label"] = d["origin"].dt.strftime("%Y-%m-%d")
        base = alt.Chart(d).encode(x=alt.X("valid_time:T", title=None))
        c90 = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Net demand (MW; negative is export)"),
            y2="q95:Q",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["A0-noirr-doy", "C3"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )  # ty: ignore[unresolved-attribute]
        c50 = base.mark_area(opacity=0.35).encode(y="q25:Q", y2="q75:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        med = base.mark_line().encode(y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        act = base.mark_line(color="black", strokeWidth=1.2).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        layered = (
            alt.layer(c90, c50, med, act)
            .properties(width=320 if hours == 36 else 560, height=170)
            .facet(column=alt.Column("origin_label:N", title="Forecast origin (UTC)"))
        )
        layered.resolve_scale(x="independent").save(DOCS / name)


if __name__ == "__main__":
    import sys

    todo = sys.argv[1:] or ["contrasts", "solar", "capacity", "examples"]
    if "contrasts" in todo:
        figure_contrasts()
    if "solar" in todo:
        figure_solar()
    if "capacity" in todo:
        print(figure_capacity())
    if "examples" in todo:
        figure_examples()
