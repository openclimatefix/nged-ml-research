# ruff: noqa: E501
r"""Draw the figures of experiment 438 from the stored scores and forecasts (no fit).

- Figure 1: O1, O2, and O3 (E-oracle minus E3, S1, and R-inj) with 95% intervals and the minimum
  detectable effect, per row set (signed-MW six and MVA-only eight).
- Figure 2: absolute CRPS of the four arms per row set.
- Figure 3: O1 in each test month on clean rows and injected harm rows.
- Figures 4 and 5: a day-ahead and a 10-day forecast of one Primary with E3, E-oracle, and S1.
- Figure 6: six real events (three Primaries from each metering group) with E3 and E-oracle.

```bash
PYTHONPATH=<the folders of e00438_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00438-*/e00438_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00438_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00438-lag-event-flag-oracle-in-the-day-window-encoder"
)
SCORES: Final[Path] = run.WORK_DIR / "scores"
MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[list[str]] = ["R-inj", "S1", "E3", "E-oracle"]
ARM_COLOURS: Final[list[str]] = ["#7BCDF3", "#14A38B", "#306BFF", "#FF4901"]
EXAMPLE_ARMS: Final[list[str]] = ["E3", "E-oracle", "S1"]
EXAMPLE_COLOURS: Final[list[str]] = ["#306BFF", "#FF4901", "#14A38B"]
EVENT_ARMS: Final[list[str]] = ["E3", "E-oracle"]
EVENT_COLOURS: Final[list[str]] = ["#306BFF", "#FF4901"]
GROUPS: Final[list[str]] = ["signed-MW six", "MVA-only eight"]
ROW_SETS: Final[list[str]] = ["clean", "inj_harm", "real_harm", "real_event"]
CONTRAST_NAMES: Final[list[str]] = [
    "O1 E-oracle - E3",
    "O2 E-oracle - S1",
    "O3 E-oracle - R-inj",
]
WIDTH: Final[int] = 700
POINTS: Final[float] = 100.0


def figure_contrasts() -> None:
    """Figure 1: O1 to O3 with 95% intervals and the minimum detectable effect."""
    mde = pl.read_parquet(SCORES / "mde.parquet").select("row_set", "group", "mde")
    c = (
        pl.read_parquet(SCORES / "contrasts.parquet")
        .drop("fold_differences")
        .filter(
            pl.col("contrast").is_in(CONTRAST_NAMES),
            pl.col("group").is_in(GROUPS),
            pl.col("row_set").is_in(ROW_SETS),
        )
        .join(mde, on=["row_set", "group"])
        .with_columns(neg_mde=-pl.col("mde"))
    )
    base = alt.Chart(c.to_pandas())
    y = alt.Y("contrast:N", sort=CONTRAST_NAMES, title=None)
    colour = alt.Color("contrast:N", legend=None)
    band = base.mark_rect(opacity=0.12, color="#888").encode(x="neg_mde:Q", x2="mde:Q")  # ty: ignore[unresolved-attribute]
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low:Q", title="Difference in CRPS (points; positive means E-oracle is worse)"), x2="high:Q",
        color=colour,
    )  # fmt: skip
    dots = base.mark_point(filled=True, size=40).encode(y=y, x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color="#888", strokeDash=[3, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        (band + zero + rules + dots)
        .properties(width=WIDTH - 200, height=70)
        .facet(
            row=alt.Row("row_set:N", title=None, sort=ROW_SETS),
            column=alt.Column("group:N", title=None, sort=GROUPS),
        )
        .properties(
            title=alt.TitleParams(
                [
                    "E-oracle minus E3, S1, and R-inj at days 1 to 10 (four test months, three seeds)",
                    "Dot: estimate. Line: 95% interval from resampling origin days. Grey: plus or minus the minimum detectable effect.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def figure_absolute() -> None:
    """Figure 2: absolute CRPS of the four arms per row set."""
    a = pl.read_parquet(SCORES / "absolute.parquet").filter(
        pl.col("group").is_in(GROUPS), pl.col("row_set").is_in(ROW_SETS)
    )
    chart = (
        alt.Chart(a.to_pandas())
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("arm:N", sort=ARMS, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("crps:Q", title="CRPS (points)"),
            color=alt.Color("arm:N", scale=alt.Scale(domain=ARMS, range=ARM_COLOURS), legend=None),
            column=alt.Column("row_set:N", title=None, sort=ROW_SETS),
            row=alt.Row("group:N", title=None, sort=GROUPS),
        )
        .properties(
            width=(WIDTH - 140) // 4, height=130,
            title=alt.TitleParams(["Absolute CRPS of the four arms at days 1 to 10 (four test months, three seeds)", "Three seeds, 1.15 widening. Smaller is better."], anchor="start"),
        )
        .resolve_scale(y="independent")
    )  # fmt: skip
    chart.save(FIGURE_DIR / "figure_2_absolute_crps.svg")


def figure_folds() -> None:
    """Figure 3: O1 in each test month on clean rows and injected harm rows."""
    c = pl.read_parquet(SCORES / "contrasts.parquet").filter(
        pl.col("contrast") == CONTRAST_NAMES[0],
        pl.col("row_set").is_in(["clean", "inj_harm"]),
        pl.col("group").is_in(GROUPS),
    )
    rows = [
        {"group": r["group"], "row_set": r["row_set"], "month": month, "difference": v}
        for r in c.iter_rows(named=True)
        for month, v in r["fold_differences"].items()
        if v is not None
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows).to_pandas())
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Test month", sort=list(MONTHS)),
            y=alt.Y("difference:Q", title="E-oracle minus E3, CRPS (points)"),
            color=alt.Color("row_set:N", legend=alt.Legend(title=None, orient="top")),
            xOffset="row_set:N",
            column=alt.Column("group:N", title=None, sort=GROUPS),
        )
        .properties(
            width=(WIDTH - 120) // 2, height=200,
            title=alt.TitleParams(["E-oracle minus E3 in each test month (four test months, three seeds)", "Positive means E-oracle is worse. The flag is exact on injected harm rows."], anchor="start"),
        )
    )  # fmt: skip
    chart.save(FIGURE_DIR / "figure_3_folds.svg")


def forecasts(*, month: str, site: str, arms: list[str]) -> pl.DataFrame:
    """Return the world-0 quantile rows of the example arms for one Primary and month."""
    primary = data.sites().index(site)
    frames = []
    for arm in arms:
        f = score.read_arm(arm=arm, month=month, world=0, seeds=SEEDS, factor=score.WIDENING)
        frames.append(f.filter(pl.col("primary") == primary).with_columns(arm=pl.lit(arm)))
    return pl.concat(frames)


def figure_examples(*, month: str, site: str) -> None:
    """Figures 4 and 5: a day-ahead and a 10-day forecast from one origin."""
    arms, colours = EXAMPLE_ARMS, EXAMPLE_COLOURS
    f = forecasts(month=month, site=site, arms=EXAMPLE_ARMS)
    origins = sorted(f["origin_us"].unique().to_list())
    origin = origins[len(origins) // 2]
    one = f.filter(pl.col("origin_us") == origin)
    for number, name, hours in ((4, "day_ahead", 48.0), (5, "10_day", 240.0)):
        d = one.filter(pl.col("lead") <= hours).with_columns(
            time=pl.from_epoch("valid_us", time_unit="us"),
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS,
            actual=pl.col("actual") * POINTS,
        )  # fmt: skip
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=arms, range=colours),
            legend=alt.Legend(title=None, orient="top"),
        )
        x = alt.X("time:T", title="Valid time (UTC)", axis=alt.Axis(format="%a %d %b %H:%M" if name == "day_ahead" else "%a %d %b", labelOverlap=True))
        pdata = d.to_pandas()
        band = (
            alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.12)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        truth = (
            alt.Chart(d.filter(pl.col("arm") == "E3").to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color="#333", strokeDash=[2, 2])
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        chart = (band + median + truth).properties(
            width=WIDTH - 90, height=220,
            title=alt.TitleParams(
                [f"{name.replace('_', '-').capitalize()} forecast of Primary {site}, origin {d['time'].min():%Y-%m-%d}", "Line: median. Band: 5% to 95%. Dashed: the cleaned reading."],
                anchor="start",
            ),
        )  # fmt: skip
        chart.save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")


def real_events(*, count: int = 3) -> list[dict]:
    """Return the longest padded real events (with forecast rows) of each metering group."""
    padded = data.real_event_mask()
    sites = data.sites()
    chosen: list[dict] = []
    for label, members in (
        ("signed-MW six", score.SIGNED_MW_LABELS),
        ("MVA-only eight", score.MVA_ONLY_LABELS),
    ):
        found = []
        for site in members:
            row = padded[sites.index(site)]
            edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
            for start, end in zip(
                np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True
            ):
                found.append({"group": label, "site": site, "start": int(start), "end": int(end)})
        found.sort(key=lambda e: e["end"] - e["start"], reverse=True)  # ty: ignore[unsupported-operator]
        chosen += found[: count * 4]
    return chosen


def figure_events() -> None:
    """Figure 6: six real events, three per metering group, with E3 and E-oracle."""
    arms, colours = EVENT_ARMS, EVENT_COLOURS
    from datetime import UTC, datetime, timedelta

    import e00087_common as common  # ty: ignore[unresolved-import]

    times = common.grid_times()
    power, _ = data.load_stack(include_eval=False)
    month_of = {m: (datetime.fromisoformat(f"{m}-01").replace(tzinfo=UTC)) for m in MONTHS}
    picked: list[dict] = []
    per_group = {"signed-MW six": 0, "MVA-only eight": 0}
    for e in real_events(count=14):
        if per_group[e["group"]] >= 3:
            continue
        start_time = times[e["start"]]
        month = next(
            (m for m, t in month_of.items() if t <= start_time < (t + timedelta(days=31))), None
        )
        if month is None:
            continue
        e["month"] = month
        picked.append(e)
        per_group[e["group"]] += 1
    panels = []
    for i, e in enumerate(picked, start=1):
        f = forecasts(month=e["month"], site=e["site"], arms=EVENT_ARMS).with_columns(
            time=pl.from_epoch("valid_us", time_unit="us")
        )
        t0, t1 = (
            times[e["start"]] - timedelta(days=3),
            times[min(e["end"], len(times) - 1)] + timedelta(days=2),
        )
        origin_us = int((times[e["start"]] - timedelta(days=1)).timestamp() * 1_000_000)
        origins = np.array(sorted(f["origin_us"].unique().to_list()))
        if not len(origins):
            continue
        origin = int(origins[np.argmin(np.abs(origins - origin_us))])
        d = f.filter(pl.col("origin_us") == origin, pl.col("lead") <= 120).with_columns(
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS
        )
        grid = (
            pl.DataFrame({"time": times})
            .with_columns(actual=pl.Series(power[0][data.sites().index(e["site"])] * POINTS))
            .filter(pl.col("time").is_between(t0, t1))
            .drop_nulls()
            .drop_nans()
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=arms, range=colours),
            legend=alt.Legend(title=None, orient="top") if i == 1 else None,
        )
        x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1]), axis=alt.Axis(format="%a %d %b", labelOverlap=True, tickCount="day"))
        pdata = d.to_pandas()
        shade = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"a": [times[e["start"]]], "b": [times[min(e["end"], len(times) - 1)]]}
                ).to_pandas()
            )
            .mark_rect(opacity=0.12, color="#888")
            .encode(x="a:T", x2="b:T")
        )  # ty: ignore[unresolved-attribute]
        band = (
            alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.12)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        truth = alt.Chart(grid.to_pandas()).mark_line(color="#333").encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            (shade + band + median + truth).properties(
                width=WIDTH - 90,
                height=130,
                title=f"Event {i}: Primary {e['site']} ({e['group']}), forecast from the day before the event starts",
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Labelled load-shift events: forecasts made the day before each event starts",
                "Shaded: the padded event. Black: the cleaned reading. Events are labelled by hand, and some real events are not logged.",
            ],
            anchor="start",
        )
    )
    chart.save(FIGURE_DIR / "figure_6_real_events.svg")


def main() -> None:
    """Draw every figure."""
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_absolute()
    figure_folds()
    figure_examples(month="2026-01", site=score.SIGNED_MW_LABELS[0])
    figure_events()


if __name__ == "__main__":
    main()
