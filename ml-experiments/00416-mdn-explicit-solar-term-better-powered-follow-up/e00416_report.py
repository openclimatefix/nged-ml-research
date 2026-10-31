# ruff: noqa: E501
"""Figures and numbers for experiment 416's write-up.

Reads the stored seed forecasts, solar outputs, and capacities, and `work/score_results.json` (from
`e00416_score.py`), and writes the figures beside `result.md`.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import e00416_score as sc  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCS: Final[Path] = (
    ROOT / "docs" / "ml-experiments" / "00416-mdn-explicit-solar-term-better-powered-follow-up"
)
WORK: Final[Path] = sc.WORK
SIX: Final = sc.SIGNED
STUDIES: Final[Path] = ROOT / "docs/studies/assets/embedded_pv_capacity_table.csv"
CAPACITY_ARMS: Final[tuple[str, ...]] = ("C3", "H1", "C3p", "C3pa", "C3p-weak")
ARM_COLOURS: Final[dict[str, str]] = {
    "A0-doy": theme.DATA_BLUE,
    "C3": theme.BRAND_ORANGE,
    "H1": theme.DATA_PURPLE,
    "A0-noirr-doy": theme.BRAND_ORANGE_LIGHT,
}


def seed_dir(*, arm: str, month: str, seed: int) -> Path:
    """Return the folder holding a seed's files (experiment 409's for its reused seeds)."""
    return sc.seed_file(arm=arm, month=month, seed=seed).parent


def contrast_table() -> pl.DataFrame:
    """Return every stored contrast as a row with its slice and metric."""
    stored = json.loads(sc.RESULTS_PATH.read_text())
    rows = []
    for group in ("primary", "secondary"):
        for key, value in stored[group].items():
            name, rest = key.split(" [")
            slice_name, metric = rest.rstrip("]").split(", ")
            rows.append(
                {
                    "group": group,
                    "contrast": name,
                    "slice": slice_name,
                    "metric": metric,
                    "difference": value["difference"],
                    "lower": value["lower"],
                    "upper": value["upper"],
                }
            )
    return pl.DataFrame(rows)


def figure_contrasts() -> None:
    """Draw the contrasts against A0-doy, sunny-midday and all half-hours, CRPS and q99 pinball."""
    data = contrast_table().filter(pl.col("contrast").str.ends_with("minus A0-doy"))
    names = {"sunny_midday": "Sunny middays", "lead_days_1_10": "All half-hours"}
    panels = []
    for metric, title in (("crps", "CRPS"), ("pinball_q99", "q99 pinball loss")):
        d = (
            data.filter(pl.col("metric") == metric)
            .with_columns(pl.col("slice").replace(names))
            .to_pandas()
        )
        y = alt.Y("contrast:N", sort=list(dict.fromkeys(d["contrast"])), title=None)
        colour = alt.Color(
            "slice:N",
            scale=alt.Scale(
                domain=list(names.values()), range=[theme.BRAND_ORANGE, theme.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
        bars = (
            alt.Chart(d)
            .mark_rule(strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                y=y,
                x=alt.X(
                    "lower:Q", title=f"{title} difference from A0-doy (below zero favours the arm)"
                ),
                x2="upper:Q",
                color=colour,
                yOffset="slice:N",
            )
        )
        dots = (
            alt.Chart(d)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60)
            .encode(y=y, x="difference:Q", color=colour, yOffset="slice:N")
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 4], color="grey")
            .encode(x="x:Q")
        )
        panels.append((bars + dots + zero).properties(width=330, height=170))
    chart = alt.hconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Differences from the free irradiance input (A0-doy) at the six signed-direction Primaries",
                "Line: 99.17% month-block interval (Bonferroni across six tests for the sunny-midday contrasts).",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS / "figure_1_contrasts.svg")


def figure_solar() -> None:
    """Draw the learned solar output by hour for a winter and a summer month, C3 against H1."""
    series = adapter.series_table()
    label = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    pieces = []
    for arm in ("C3", "H1"):
        for month in ("2025-12", "2026-06"):
            frames = [
                pl.read_parquet(
                    seed_dir(arm=arm, month=month, seed=s) / f"{month}_s{s}.solar.parquet"
                )
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
    domain = ["C3 2025-12", "H1 2025-12", "C3 2026-06", "H1 2026-06"]
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
                    "Learned solar output of the explicit term (C3) and the hybrid (H1)",
                    "Mean over test half-hours and 5 seeds, December 2025 and June 2026.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(DOCS / "figure_2_learned_solar_output.svg")


def capacities() -> pl.DataFrame:
    """Return each arm's learned capacity at the as-of season-year, per Primary, fold, and seed."""
    rows = []
    for arm in CAPACITY_ARMS:
        for month in sc.MONTHS:
            for s in sc.SEEDS:
                path = seed_dir(arm=arm, month=month, seed=s) / f"{month}_s{s}.capacity.json"
                cap = json.loads(path.read_text())
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
        lo=pl.col("capacity_mw").quantile(0.1),
        hi=pl.col("capacity_mw").quantile(0.9),
    )
    s = studies.rename(
        {"capacity_mw": "mean", "capacity_low_mw": "lo", "capacity_high_mw": "hi"}
    ).with_columns(arm=pl.lit("Studies page (95% interval)"))
    data = pl.concat([summary, s.select(summary.columns)]).to_pandas()
    domain = ["Studies page (95% interval)", *CAPACITY_ARMS]
    y = alt.Y("site:N", sort=list(SIX), title=None)
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(
            domain=domain,
            range=[
                "#666666",
                theme.BRAND_ORANGE,
                theme.DATA_PURPLE,
                theme.DATA_BLUE,
                theme.DATA_BLUE_LIGHT,
                theme.BRAND_ORANGE_LIGHT,
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
    )
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50)
        .encode(y=y, x="mean:Q", color=colour, yOffset="arm:N")
    )
    chart = (bars + dots).properties(
        width=520,
        height=300,
        title=alt.TitleParams(
            [
                "Learned capacity against the studies page",
                "Dot: mean over folds and 5 seeds. Line: 10th to 90th percentile (studies page: 95% interval).",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS / "figure_3_learned_capacity.svg")
    return summary.join(studies, on="site", suffix="_studies").sort("arm", "site")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of A0-doy and C3: D10 (signed) and D12 (MVA-only)."""
    series = adapter.series_table()
    ctx = s381.build_context(series=series)
    sid = {r[1]: r[0] for r in series.iter_rows()}
    pieces = []
    for arm in ("A0-doy", "C3"):
        frame = adapter.forecast_frame(
            stored=sc.arm_frame(arm=arm, seeds=sc.SEEDS), series=series, unit="fraction"
        )
        for site in ("D10", "D12"):
            for origin in (
                datetime(2026, 1, 15, 9, tzinfo=UTC),
                datetime(2026, 6, 15, 9, tzinfo=UTC),
            ):
                f = frame.filter(
                    pl.col("time_series_id") == sid[site], pl.col("power_fcst_init_time") == origin
                )
                pieces.append(
                    f.with_columns(arm=pl.lit(arm), origin=pl.lit(origin), site=pl.lit(site))
                )
    data = pl.concat(pieces).join(ctx["actuals"], on=["time_series_id", "valid_time"], how="left")
    data.write_parquet(WORK / "example_forecasts.parquet")
    for hours, name in (
        (36, "figure_4_example_day_ahead.svg"),
        (336, "figure_5_example_14_day.svg"),
    ):
        d = (
            data.with_columns(
                hours=(pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60
            )
            .filter(pl.col("hours") <= hours, pl.col("hours") >= 0)
            .with_columns(panel=pl.col("site") + " " + pl.col("origin").dt.strftime("%Y-%m-%d"))
            .to_pandas()
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=["A0-doy", "C3"], range=[theme.DATA_BLUE, theme.BRAND_ORANGE]),
            legend=alt.Legend(title=None, orient="top"),
        )
        base = alt.Chart(d).encode(x=alt.X("valid_time:T", title=None))
        c90 = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("q05:Q", title="Net demand (MW; negative is export)"), y2="q95:Q", color=colour
        )
        c50 = base.mark_area(opacity=0.35).encode(y="q25:Q", y2="q75:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        med = base.mark_line().encode(y="q50:Q", color="arm:N")  # ty: ignore[unresolved-attribute]
        act = base.mark_line(color="black", strokeWidth=1.2).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        layered = (
            alt.layer(c90, c50, med, act)
            .properties(width=300 if hours == 36 else 520, height=150)
            .facet(facet=alt.Facet("panel:N", title=None), columns=2)
        )
        layered.resolve_scale(x="independent").save(DOCS / name)


def figure_per_primary() -> None:
    """Draw sunny-midday CRPS and q99 pinball loss per Primary and arm from the stored report."""
    stored = json.loads(sc.RESULTS_PATH.read_text())
    d = pl.DataFrame(stored["per_primary_sunny_midday"]).filter(pl.col("label").is_in(list(SIX)))
    panels = []
    for metric, title in (
        ("crps", "Sunny-midday CRPS"),
        ("pinball_q99", "Sunny-midday q99 pinball loss"),
    ):
        panels.append(
            alt.Chart(d.to_pandas())
            .mark_bar()
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("label:N", sort=list(SIX), title=None),
                y=alt.Y(f"{metric}:Q", title=title),
                xOffset="arm:N",
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            .properties(width=300, height=170)
        )
    alt.hconcat(*panels).save(DOCS / "figure_6_per_primary.svg")


if __name__ == "__main__":
    sc.MONTHS[:] = sc.TEST_MONTHS_12
    todo = sys.argv[1:] or ["contrasts", "solar", "capacity", "examples", "per_primary"]
    if "contrasts" in todo:
        figure_contrasts()
    if "solar" in todo:
        figure_solar()
    if "capacity" in todo:
        print(figure_capacity())
    if "examples" in todo:
        figure_examples()
    if "per_primary" in todo:
        figure_per_primary()
