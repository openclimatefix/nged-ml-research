# ruff: noqa: E501
r"""Draw the figures of experiment 448 from the stored forecasts and `e00448_refs.py`'s tables.

- Figure 1: CRPS against the lead day, S1 family beside the references.
- Figure 2: leave-one-out arms minus S1-pool5, by lead day.
- Figure 3: the ladder arms minus S1, with 95% intervals (experiment 448's contrast table).
- Figure 4: tail coverage of every arm and reference.
- Figures 5 and 6: a day-ahead and a 10-day forecast of S1-pool5, winter and summer, for one Primary.
- Figure 7: six real lag-jump events (three per metering group).

Forecasts stop at a 240-hour lead, so no figure shows 14 days. Tables go to `work/figure_tables/`.

```bash
source /tmp/e448_env.sh   # PYTHONPATH of every ml-experiments folder
uv run --no-sync python ml-experiments/00448-*/e00448_figures.py
```
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00448_refs as refs  # ty: ignore[unresolved-import]
import e00448_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401
from studies.charts import CONTENT_WIDTH_PX, wrapped

FIGURE_DIR: Final[Path] = next(
    (Path(__file__).resolve().parents[2] / "docs" / "ml-experiments").glob("00448-*")
)
TABLE_DIR: Final[Path] = refs.TABLE_DIR
METRICS: Final[Path] = TABLE_DIR / "metrics_four_months.parquet"
WIDTH: Final[int] = CONTENT_WIDTH_PX - 110
BRAND: Final[dict[str, str]] = {
    "orange": "#FF4901", "blue": "#306BFF", "sky": "#10C5F7", "purple": "#B701FF", "green": "#17E58F",
}  # fmt: skip
GREY_DARK: Final[str] = "#444444"
GREY_MID: Final[str] = "#888888"
GREY_LIGHT: Final[str] = "#BBBBBB"
GROUP_ORDER: Final[tuple[str, str]] = ("signed-MW six", "MVA-only eight")
GROUP_COLOURS: Final[list[str]] = [BRAND["blue"], BRAND["orange"]]
LEAD_ARMS: Final[dict[str, tuple[str, str, int]]] = {
    "S1-pool5": (BRAND["orange"], "solid", 3),
    "S1": (BRAND["blue"], "solid", 2),
    "S1PT": (BRAND["sky"], "solid", 2),
    "BG2": (GREY_DARK, "solid", 2),
    "XGB-monthly": (GREY_MID, "dashed", 2),
    "manual_heuristic": (GREY_MID, "dotted", 2),
    "climatology": (GREY_LIGHT, "solid", 2),
}
"""Colour, line style, and width of each line of figure 1."""
LOO_ARMS: Final[dict[str, str]] = {
    "S1PT-pool5": BRAND["purple"],
    "S1PT-pool5-Bmix": BRAND["green"],
    "S1PT-noemb-pool5": BRAND["blue"],
    "RinjPT-pool5": BRAND["sky"],
    "S1": GREY_MID,
}
LADDER_CONTRASTS: Final[dict[str, str]] = {
    "C1 S1PT - S1": "S1PT (pretraining)",
    "C1b S1PT-fullLR - S1": "S1PT-fullLR (full learning rate)",
    "C5 S1-w256 - S1": "S1-w256 (wider network)",
    "C2 S1-Bmix - S1": "S1-Bmix (B-mix output)",
    "C3b S1-pool3 - S1": "S1-pool3 (3 seeds)",
    "C3 S1-pool5 - S1": "S1-pool5 (5 seeds)",
}
EXAMPLE_ARM: Final[str] = "S1-pool5"
EXAMPLE_SEEDS: Final[tuple[int, ...]] = refs.SEEDS
POINTS: Final[float] = 100.0
FIRST_FOLD: Final[dict[str, datetime]] = {
    m: datetime.fromisoformat(f"{m}-01").replace(tzinfo=UTC) for m in refs.FOUR_MONTHS
}


def titled(*, title: str, subtitle: str, width_chars: int = 105) -> alt.TitleParams:
    """Return a left-aligned title and subtitle wrapped to the plot's width."""
    return alt.TitleParams(
        wrapped(text=title, width=width_chars) + wrapped(text=subtitle, width=width_chars + 12),
        anchor="start",
        fontSize=12,
        subtitleFontSize=11,
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a figure's SVG beside the result page."""
    chart.save(FIGURE_DIR / name)  # ty: ignore[unresolved-attribute]


def figure_lead() -> None:
    """Figures 1 and 2: CRPS against the lead day (pooled over rows), two metering groups."""
    m = pl.read_parquet(METRICS).filter(
        pl.col("subset").str.starts_with("day "), pl.col("name").is_in(list(GROUP_ORDER))
    )
    m = m.with_columns(day=pl.col("subset").str.slice(4).cast(pl.Int32))
    m.select("arm", "name", "day", "n", "crps").sort("arm", "name", "day").write_csv(
        TABLE_DIR / "crps_by_lead_day.csv"
    )
    shown = m.filter(pl.col("arm").is_in(list(LEAD_ARMS)))
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(LEAD_ARMS), range=[v[0] for v in LEAD_ARMS.values()]),
        legend=alt.Legend(title=None, orient="top", columns=4, symbolStrokeWidth=3),
    )
    dash = alt.StrokeDash(
        "arm:N",
        scale=alt.Scale(domain=list(LEAD_ARMS), range=[[1, 0] if v[1] == "solid" else ([6, 3] if v[1] == "dashed" else [2, 2]) for v in LEAD_ARMS.values()]),
        legend=None,
    )  # fmt: skip
    chart = (
        alt.Chart(shown.to_pandas())
        .mark_line(point=alt.OverlayMarkDef(size=25, filled=True))
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:O", title="Lead (days after the forecast origin)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
            color=colour,
            strokeDash=dash,
        )
        .properties(width=WIDTH, height=170)
        .facet(row=alt.Row("name:N", title=None, sort=list(GROUP_ORDER)))
        .resolve_scale(y="independent")
        .properties(
            title=titled(
                title="CRPS rises with the lead; the S1 family and BG2 sit close together, well below manual_heuristic and climatology",
                subtitle="Mean CRPS over clean half-hours of 4 test months (2025-10, 2026-01, 2026-03, 2026-06), real data, 1.15 widening of the S1 family, 5 seeds. Smaller is better. Only half-hours that every reference also holds.",
            )
        )
    )
    save(chart=chart, name="figure_1_crps_by_lead.svg")

    base = m.filter(pl.col("arm") == "S1-pool5").select("name", "day", base_crps="crps")
    loo = (
        m.filter(pl.col("arm").is_in(list(LOO_ARMS)))
        .join(base, on=["name", "day"])
        .with_columns(difference=pl.col("crps") - pl.col("base_crps"))
    )
    loo.select("arm", "name", "day", "difference").sort("arm", "name", "day").write_csv(
        TABLE_DIR / "loo_minus_s1_pool5_by_lead_day.csv"
    )
    shown_arms = [a for a in LOO_ARMS if a != "S1PT-noemb-pool5"]
    panels = []
    for group in GROUP_ORDER:
        part = loo.filter(pl.col("name") == group, pl.col("arm") != "S1PT-noemb-pool5")
        zero = (
            alt.Chart(pl.DataFrame({"y": [0.0]}).to_pandas())
            .mark_rule(color="#888", strokeDash=[3, 3])
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        lines = (
            alt.Chart(part.to_pandas())
            .mark_line(point=alt.OverlayMarkDef(size=25, filled=True))
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "day:O",
                    title="Lead (days after the forecast origin)",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y("difference:Q", title="CRPS minus S1-pool5's (points)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(domain=shown_arms, range=[LOO_ARMS[a] for a in shown_arms]),
                    legend=alt.Legend(title=None, orient="top", columns=3, symbolStrokeWidth=3)
                    if group == GROUP_ORDER[0]
                    else None,
                ),
            )
        )
        panels.append(
            (zero + lines).properties(
                width=WIDTH, height=150, title=alt.TitleParams(group, anchor="start", fontSize=11)
            )
        )
    chart2 = alt.vconcat(*panels).properties(
        title=titled(
            title="Pretraining, B-mix, and the surprise inputs each add 0.0 to 0.2 points of CRPS to S1-pool5 at every lead, so none of them helps",
            subtitle="Each arm's CRPS minus S1-pool5's by lead day (negative is better), clean half-hours, 4 test months, 5 seeds. The dashed line is S1-pool5 (no pretraining). S1PT-pool5 adds pretraining, S1PT-pool5-Bmix adds pretraining and B-mix, RinjPT-pool5 adds pretraining and drops the surprise inputs. S1 (grey) is one fit without pretraining. S1PT-noemb-pool5 (pretraining, no Primary embedding) is 0.6 to 0.9 points above S1-pool5 and is left off to keep the scale.",
        )
    )
    save(chart=chart2, name="figure_2_loo_by_lead.svg")


def figure_ladder() -> None:
    """Figure 3: each ladder arm minus S1, per group, with 95% intervals."""
    scores_dir = run.WORK_DIR / "scores"
    # The three-seed pool is compared with S1's seeds 0 to 2, so its contrast comes from the
    # seeds-0-to-2 table; every other contrast uses all five seeds.
    contrasts = pl.concat(
        [
            pl.read_parquet(scores_dir / "contrasts_seeds_0-4.parquet").filter(
                ~pl.col("contrast").str.starts_with("C3b")
            ),
            pl.read_parquet(scores_dir / "contrasts_seeds_0-2.parquet").filter(
                pl.col("contrast").str.starts_with("C3b")
            ),
        ]
    )
    c = (
        contrasts.filter(
            pl.col("row_set") == "clean",
            pl.col("contrast").is_in(list(LADDER_CONTRASTS)),
            pl.col("group").is_in(list(GROUP_ORDER)),
        )
        .with_columns(arm=pl.col("contrast").replace_strict(LADDER_CONTRASTS))
        .drop("fold_differences")
    )
    c.write_csv(TABLE_DIR / "ladder_contrasts_clean.csv")
    order = list(LADDER_CONTRASTS.values())
    y = alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(GROUP_ORDER), range=GROUP_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    offset = alt.YOffset("group:N", scale=alt.Scale(domain=list(GROUP_ORDER)))
    base = alt.Chart(c.to_pandas())
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=y, yOffset=offset, x=alt.X("low:Q", title="CRPS minus S1's (points; negative is better)"), x2="high:Q", color=colour
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
        y=y, yOffset=offset, x="difference:Q", color=colour
    )  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (zero + rules + dots).properties(
        width=WIDTH - 150, height=240,
        title=titled(
            title="Pretraining and B-mix raise CRPS; the 5-seed pool and the wider network lower it",
            subtitle="Dot: estimate. Line: 95% interval from resampling origin days (interim: 4 months). Clean half-hours, 5 seeds, 1.15 widening. Each arm minus the single-network S1; the dashed line is S1. The 3-seed pool is compared with seeds 0 to 2 of S1, every other arm with all 5 seeds.",
        ),
    )  # fmt: skip
    save(chart=chart, name="figure_3_ladder_contrasts.svg")


def figure_tails() -> None:
    """Figure 4: coverage of the q01, q05, q95, and q99 forecasts, per arm and group."""
    m = pl.read_parquet(METRICS).filter(
        pl.col("subset") == "days 1 to 10",
        pl.col("scope") == "group",
        pl.col("weighting") == "pooled",
        pl.col("name").is_in(list(GROUP_ORDER)),
    )
    long = m.unpivot(
        on=[f"coverage_{t}" for t in refs.TAILS], index=["arm", "name"], variable_name="tail", value_name="coverage"
    ).with_columns(
        tail=pl.col("tail").str.replace("coverage_", ""),
        nominal=pl.col("tail").str.replace("coverage_", "").replace_strict({"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}),
    )  # fmt: skip
    long.write_csv(TABLE_DIR / "tail_coverage.csv")
    arms = [*refs.LADDER_ARMS, *refs.REFERENCES]
    panels = []
    for tail, nominal, domain in (
        ("q01", 0.01, [0, 0.08]),
        ("q05", 0.05, [0, 0.14]),
        ("q95", 0.95, [0.8, 1.0]),
        ("q99", 0.99, [0.85, 1.0]),
    ):
        d = long.filter(pl.col("tail") == tail)
        base = alt.Chart(d.to_pandas())
        dots = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=arms, title=None, axis=alt.Axis(labels=tail == "q01", labelLimit=150)),
            x=alt.X("coverage:Q", title=f"{tail}: share of actuals at or below", scale=alt.Scale(domain=domain), axis=alt.Axis(format=".2f", tickCount=3)),
            color=alt.Color("name:N", scale=alt.Scale(domain=list(GROUP_ORDER), range=GROUP_COLOURS), legend=alt.Legend(title=None, orient="top") if tail == "q01" else None),
        )  # fmt: skip
        rule = (
            alt.Chart(pl.DataFrame({"x": [nominal]}).to_pandas())
            .mark_rule(color="#888", strokeDash=[3, 3])
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=domain)))  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append((rule + dots).properties(width=(WIDTH - 150) // 4 - 8, height=290))
    chart = alt.hconcat(*panels, spacing=6).properties(
        title=titled(
            title="The S1 family's q01 and q05 forecasts are wider than needed and its q95 slightly too low; manual_heuristic and climatology miss by more",
            subtitle="Share of actuals at or below each quantile, mean over clean half-hours, 4 test months. The dashed line is the nominal level, so a dot left of it means too few actuals fell below.",
            width_chars=100,
        )
    )
    save(chart=chart, name="figure_4_tail_coverage.svg")


def read_quantiles(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's world-0 rows for a month (seed-mean widened quantiles, all rows kept)."""
    frames = [
        pl.read_parquet(run.WORK_DIR / "predictions" / arm / f"{month}_s{s}_w0.parquet")
        for s in EXAMPLE_SEEDS
    ]
    first = frames[0]
    q = np.zeros((first.height, len(refs.LEVEL_NAMES)))
    for f in frames:
        q += score.widened(
            q=f.select(list(refs.LEVEL_NAMES)).to_numpy().astype(np.float64), factor=score.WIDENING
        ) / len(frames)
    return first.select(
        "primary", "origin_us", "valid_us", "lead", "actual", "tgt_real", "win_real", "tgt_post"
    ).with_columns(*[pl.Series(n, q[:, i]) for i, n in enumerate(refs.LEVEL_NAMES)])


def figure_examples() -> None:
    """Figures 5 and 6: a day-ahead and a 10-day forecast, winter (2026-01) and summer (2026-06)."""
    sites = refs.sites_table()
    capacity = dict(zip(sites["primary"].to_list(), sites["capacity_mw"].to_list(), strict=True))
    label = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    months = ("2026-01", "2026-06")
    data_by_month = {m: read_quantiles(arm=EXAMPLE_ARM, month=m) for m in months}
    # The Primary: a signed-MW one with no real-event flag in either month's windows.
    best, best_flags = None, 1.0
    for primary in (p for p, lab in label.items() if lab in score.GROUPS["signed-MW six"]):
        flags = np.mean(
            [
                d.filter(pl.col("primary") == primary).select((pl.col("win_real") | pl.col("tgt_post") | pl.col("actual").is_nan()).mean()).item()
                for d in data_by_month.values()
            ]
        )  # fmt: skip
        if flags < best_flags:
            best, best_flags = primary, float(flags)
    assert best is not None
    site = label[best]
    print(f"example Primary {site} (flag share {best_flags:.3f})")
    pieces = []
    for month, d in data_by_month.items():
        rows = d.filter(pl.col("primary") == best)
        origins = sorted(rows["origin_us"].unique().to_list())
        origin = origins[len(origins) // 2]
        one = rows.filter(pl.col("origin_us") == origin).sort("valid_us")
        pieces.append(one.with_columns(month=pl.lit(month), site=pl.lit(site)))
    ex = pl.concat(pieces).with_columns(
        time=pl.from_epoch("valid_us", time_unit="us"),
        season=pl.col("month").replace_strict({"2026-01": "Winter", "2026-06": "Summer"}),
        **{c: pl.col(c) * capacity[best] for c in [*refs.LEVEL_NAMES, "actual"]},
    )
    ex.drop("primary").write_csv(TABLE_DIR / "example_forecasts.csv")
    unit = "MVA" if site in score.GROUPS["MVA-only eight"] else "MW"
    for number, name, hours, width_note in (
        (5, "day_ahead", 48.0, "first 48 hours"),
        (6, "10_day", 240.0, "full 10 days; forecasts stop at a 240-hour lead"),
    ):
        panels = []
        for season in ("Winter", "Summer"):
            d = ex.filter(pl.col("season") == season, pl.col("lead") <= hours)
            origin = d["time"].min()
            pdata = d.select("time", "q05", "q25", "q50", "q75", "q95", "actual").to_pandas()
            x = alt.X(
                "time:T",
                title=None,
                axis=alt.Axis(format="%d %b %H:%M" if hours <= 48 else "%d %b", tickCount=6),
            )
            band90 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.18, color=BRAND["orange"])
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x,
                    y=alt.Y("q05:Q", title=f"Power ({unit})", scale=alt.Scale(zero=False)),
                    y2="q95:Q",
                )
            )  # ty: ignore[unresolved-attribute]
            band50 = (
                alt.Chart(pdata)
                .mark_area(opacity=0.32, color=BRAND["orange"])
                .encode(x=x, y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
            )  # ty: ignore[unresolved-attribute]
            median = (
                alt.Chart(pdata)
                .mark_line(color=BRAND["orange"], strokeWidth=2)
                .encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
            )  # ty: ignore[unresolved-attribute]
            truth = (
                alt.Chart(pdata).mark_line(color="#222", strokeWidth=1.3).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
            )  # ty: ignore[unresolved-attribute]
            panels.append(
                (band90 + band50 + median + truth).properties(
                    width=WIDTH,
                    height=170,
                    title=alt.TitleParams(
                        f"{season}: forecast issued {origin:%d %b %Y} 09:00 UTC",
                        anchor="start",
                        fontSize=11,
                    ),
                )
            )
        chart = alt.vconcat(*panels).properties(
            title=titled(
                title=f"S1-pool5's {name.replace('_', '-')} forecast for Primary {site} against the metered power, winter and summer",
                subtitle=f"Window: {width_note}. Orange line: median. Dark orange band: 50% interval. Light orange band: 90% interval. Black line: cleaned reading. One origin per season (the middle one of the test month), 5 seeds, 1.15 widening.",
            )
        )
        save(chart=chart, name=f"figure_{number}_example_{name}.svg")


def event_list(*, count: int = 3) -> list[dict[str, Any]]:
    """Return the longest padded real events of each metering group, one per Primary and month."""
    padded = data.real_event_mask()
    times = common.grid_times()
    sites = data.sites()
    chosen: list[dict[str, Any]] = []
    for group in score.GROUPS:
        if group == "D15 and D18":
            continue
        found: list[dict[str, Any]] = []
        for site in score.GROUPS[group]:
            row = padded[sites.index(site)]
            edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            for start, end in zip(
                np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True
            ):
                start_time = times[int(start)]
                month = next(
                    (
                        m
                        for m, t in FIRST_FOLD.items()
                        if t + timedelta(days=1) <= start_time < t + timedelta(days=28)
                    ),
                    None,
                )
                if month is None or end >= len(times) or end - start < 48 or end - start > 48 * 8:
                    continue
                found.append(
                    {
                        "group": group,
                        "site": site,
                        "start": int(start),
                        "end": int(end),
                        "month": month,
                    }
                )
        found.sort(key=lambda e: e["end"] - e["start"], reverse=True)
        used: set[str] = set()
        picked = []
        for e in found:
            if e["site"] in used:
                continue
            used.add(e["site"])
            picked.append(e)
            if len(picked) == count:
                break
        chosen += picked
    return chosen


def figure_events() -> None:
    """Figure 7: six real events with S1 and S1-pool5 median and 90% band, and the actuals."""
    times = common.grid_times()
    sites = refs.sites_table()
    capacity = dict(zip(sites["label"].to_list(), sites["capacity_mw"].to_list(), strict=True))
    index = {
        lab: p for p, lab in zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True)
    }
    arms = {"S1": BRAND["blue"], "S1-pool5": BRAND["orange"]}
    events = event_list()
    cache: dict[tuple[str, str], pl.DataFrame] = {}
    panels, table = [], []
    for i, e in enumerate(events, start=1):
        site, month = e["site"], e["month"]
        t_start, t_end = times[e["start"]], times[e["end"]]
        t0, t1 = t_start - timedelta(days=3), t_end + timedelta(days=2)
        pieces = []
        for arm in arms:
            if (arm, month) not in cache:
                cache[arm, month] = read_quantiles(arm=arm, month=month)
            f = cache[arm, month].filter(pl.col("primary") == index[site])
            origins = np.array(sorted(f["origin_us"].unique().to_list()))
            wanted = int((t_start - timedelta(days=1)).timestamp() * 1_000_000)
            origin = int(origins[np.argmin(np.abs(origins - wanted))])
            pieces.append(f.filter(pl.col("origin_us") == origin).with_columns(arm=pl.lit(arm)))
        fc = pl.concat(pieces).with_columns(
            time=pl.from_epoch("valid_us", time_unit="us").dt.replace_time_zone("UTC"),
            **{c: pl.col(c) * capacity[site] for c in [*refs.LEVEL_NAMES, "actual"]},
        ).filter(pl.col("time").is_between(t0, t1))  # fmt: skip
        actual = (
            cache["S1", month].filter(pl.col("primary") == index[site])
            .group_by("valid_us").agg(pl.col("actual").first())
            .with_columns(time=pl.from_epoch("valid_us", time_unit="us").dt.replace_time_zone("UTC"), actual=pl.col("actual") * capacity[site])
            .filter(pl.col("time").is_between(t0, t1), pl.col("actual").is_finite())
            .sort("time")
        )  # fmt: skip
        unit = "MVA" if site in score.GROUPS["MVA-only eight"] else "MW"
        x = alt.X(
            "time:T",
            title=None,
            scale=alt.Scale(domain=[t0, t1]),
            axis=alt.Axis(format="%d %b", tickCount=6),
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(arms), range=list(arms.values())),
            legend=alt.Legend(title=None, orient="top") if i == 1 else None,
        )
        pdata = fc.to_pandas()
        shade = (
            alt.Chart(pl.DataFrame({"a": [t_start], "b": [t_end]}).to_pandas())
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        start_rule = (
            alt.Chart(pl.DataFrame({"t": [t_start, t_end]}).to_pandas())
            .mark_rule(color="#555", strokeDash=[4, 3])
            .encode(x="t:T")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(pdata)
            .mark_area(opacity=0.10)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x,
                y=alt.Y("q05:Q", title=unit, scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = (
            alt.Chart(fc.filter(pl.col("arm") == "S1-pool5").to_pandas())
            .mark_line(strokeWidth=2)
            .encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
            + alt.Chart(fc.filter(pl.col("arm") == "S1").to_pandas())
            .mark_line(strokeWidth=1.5, strokeDash=[5, 3])
            .encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
        truth = (
            alt.Chart(actual.to_pandas())
            .mark_line(color="#222", strokeWidth=1.2)
            .encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            (shade + start_rule + band + median + truth).properties(
                width=WIDTH, height=120,
                title=alt.TitleParams(f"Event {i}: Primary {site} ({e['group']}), {month}; forecast from the day before the start", anchor="start", fontSize=11),
            )
        )  # fmt: skip
        table.append(
            {"event": i, "site": site, "group": e["group"], "month": month, "start": t_start, "end": t_end,
             "hours": (e["end"] - e["start"]) / 2}
        )  # fmt: skip
        fc.drop("primary").write_csv(TABLE_DIR / f"event_{i}_forecasts.csv")
    pl.DataFrame(table).write_csv(TABLE_DIR / "events_shown.csv")
    chart = alt.vconcat(*panels).properties(
        title=titled(
            title="Forecasts made the day before six labelled lag-jump events, three Primaries per metering group",
            subtitle="Dashed lines: event start and end (padded). Shaded: the event. Solid orange: S1-pool5 median; dashed blue: S1 median. Bands: 90% interval of each. Black: cleaned reading. Events are labelled by hand; some real events are not logged. 5 seeds, 1.15 widening.",
        )
    )
    save(chart=chart, name="figure_7_real_events.svg")


def main() -> None:
    """Draw every figure."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figure_lead()
    figure_ladder()
    figure_tails()
    figure_examples()
    figure_events()


if __name__ == "__main__":
    main()
