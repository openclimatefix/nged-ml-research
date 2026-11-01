# ruff: noqa: E501
r"""Draw the figures of experiment 440 from the stored scores and forecasts (no fit).

Scores are read from `work/scores` (arms S1, R-inj, S2, S3, S3-rw, and H-lvl; seeds 0 to 2; four
test months; the 1.15 widening) and the sweep tables from `work/sweep` (seed 0 only). Figures:

- Figure 1: the contrasts W1 to W5, H1, and H2 with 95% intervals, and the margin
  max(0.03, minimum detectable effect), per row set, for the signed-MW six and the MVA-only eight.
- Figure 2: absolute CRPS of the six arms per row set.
- Figure 3: each contrast on clean rows in each test month.
- Figure 4: CRPS against the lead, clean rows (absolute, and minus S1).
- Figure 5: the share of a step in the lags that reaches the forecast median, by k.
- Figures 6 and 7: a day-ahead and a 10-day forecast of one Primary (S1, S2, S3).
- Figure 8: six real events (S1, S3, S3-rw).

The script also prints every number the page quotes that the stored score tables do not give
directly: the absolute table, the contrasts, the tail table, the sweep report, and the H-lvl
contrasts on clean rows at 12 to 36 hours.

```bash
source <paths>; PYTHONPATH=<the folders of e00440_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00440-*/e00440_figures.py
```
"""

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00440_run as run  # ty: ignore[unresolved-import]
import e00440_sweep as sweep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00440-windowed-and-neighbour-relative-surprise-on-the-lag-network"
)
SCORES: Final[Path] = run.WORK_DIR / "scores"
MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SIGNED: Final[str] = "signed-MW six"
MVA: Final[str] = "MVA-only eight"
WIDTH: Final[int] = 700
POINTS: Final[float] = 100.0
TITLE_CHARS: Final[int] = 78
MARGIN_FLOOR: Final[float] = 0.03
BOOTSTRAP_DRAWS: Final[int] = 2000

ARMS: Final[list[str]] = ["R-inj", "S1", "S2", "S3", "S3-rw", "H-lvl"]
COLOURS: Final[list[str]] = ["#888888", "#306BFF", "#FF4901", "#B701FF", "#D57AFF", "#17E58F"]
EXAMPLE_ARMS: Final[list[str]] = ["S1", "S2", "S3"]
EXAMPLE_COLOURS: Final[list[str]] = ["#306BFF", "#FF4901", "#B701FF"]
EVENT_ARMS: Final[list[str]] = ["S1", "S3", "S3-rw"]
EVENT_COLOURS: Final[list[str]] = ["#306BFF", "#B701FF", "#D57AFF"]
CONTRAST_ORDER: Final[list[str]] = [
    "W1 S2 - S1", "W2 S3 - S2", "W3 S3 - S3-rw", "W4 S3 - S1", "W5 S3-rw - S2",
    "H1 H-lvl - R-inj", "H2 H-lvl - S1",
]  # fmt: skip
ROW_NAMES: Final[dict[str, str]] = {
    "clean": "Clean rows",
    "inj_harm": "Injected harm rows",
    "real_harm": "Real harm rows",
    "real_event": "Real event rows",
}
ROW_ORDER: Final[list[str]] = list(ROW_NAMES)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG, wrapping its title lines to the plot width."""
    title = chart.title  # ty: ignore[unresolved-attribute]
    if isinstance(title, alt.TitleParams) and isinstance(title.text, list):
        lines = [w for t in title.text for w in textwrap.wrap(t, width=TITLE_CHARS)]
        chart = chart.properties(title=alt.TitleParams(lines, anchor="start"))  # ty: ignore[unresolved-attribute]
    chart.save(FIGURE_DIR / name)  # ty: ignore[unresolved-attribute]


def arm_colour(*, arms: list[str], colours: list[str], legend: bool = True) -> alt.Color:
    """Return the colour encoding of an arm."""
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=arms, range=colours),
        legend=alt.Legend(title=None, orient="top", columns=len(arms)) if legend else None,
    )


def figure_contrasts() -> None:
    """Figure 1: contrasts with 95% intervals and the margin, per row set and metering group."""
    c = pl.read_parquet(SCORES / "contrasts.parquet").drop("fold_differences")
    m = pl.read_parquet(SCORES / "mde.parquet")
    rows = []
    for row_set in ROW_ORDER:
        panels = []
        sub_row = c.filter(pl.col("row_set") == row_set, pl.col("group").is_in([SIGNED, MVA]))
        margins = {
            g: max(MARGIN_FLOOR, float(m.filter(pl.col("row_set") == row_set, pl.col("group") == g)["mde"][0]))
            for g in (SIGNED, MVA)
        }  # fmt: skip
        lo = min(float(sub_row["low"].min()), -max(margins.values())) - 0.01  # ty: ignore[invalid-argument-type]
        hi = max(float(sub_row["high"].max()), *margins.values()) + 0.01  # ty: ignore[invalid-argument-type]
        for i, group in enumerate((SIGNED, MVA)):
            d = sub_row.filter(pl.col("group") == group).to_pandas()
            margin = pl.DataFrame({"x": [-margins[group], margins[group]]}).to_pandas()
            y = alt.Y(
                "contrast:N", sort=CONTRAST_ORDER, title=None,
                axis=alt.Axis(labels=i == 0, ticks=i == 0),
            )  # fmt: skip
            x = alt.X(
                "low:Q", title="Contrast, CRPS (points; negative is better)",
                scale=alt.Scale(domain=[lo, hi]),
            )  # fmt: skip
            rules = (
                alt.Chart(d)  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeWidth=1.5, color="#306BFF")
                .encode(y=y, x=x, x2="high:Q")
            )  # ty: ignore[unresolved-attribute]
            dots = (
                alt.Chart(d)  # ty: ignore[unresolved-attribute]
                .mark_point(filled=True, size=40, color="#306BFF")
                .encode(y=y, x="difference:Q")
            )  # ty: ignore[unresolved-attribute]
            zero = alt.Chart(d).mark_rule(color="#888").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
            edges = alt.Chart(margin).mark_rule(color="#FF4901", strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
            panels.append(
                (zero + edges + rules + dots).properties(
                    width=(WIDTH - 200) // 2, height=100,
                    title=alt.TitleParams(f"{ROW_NAMES[row_set]}: {group}", anchor="start", fontSize=11),
                )
            )  # fmt: skip
        rows.append(alt.hconcat(*panels))
    chart = alt.vconcat(*rows).properties(
        title=alt.TitleParams(
            [
                "Windowed and neighbour-relative surprise minus the base network, days 1 to 10 (interim)",
                "Dot: estimate. Line: 95% interval from resampling origin days. Orange dashes: the margin, the larger of 0.03 and the minimum detectable effect. Grey: zero. Four test months, three seeds.",
            ],
            anchor="start",
        )
    )
    save(chart=chart, name="figure_1_contrasts.svg")


def figure_absolute() -> None:
    """Figure 2: absolute CRPS of the six arms per row set and group."""
    a = (
        pl.read_parquet(SCORES / "absolute.parquet")
        .filter(pl.col("group").is_in([SIGNED, MVA]), pl.col("row_set").is_in(ROW_ORDER))
        .with_columns(row=pl.col("row_set").replace_strict(ROW_NAMES))
    )
    chart = (
        alt.Chart(a.to_pandas())
        .mark_point(filled=True, size=50)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=ARMS, title=None),
            x=alt.X("crps:Q", title="CRPS (points of capacity; smaller is better)", scale=alt.Scale(zero=False)),
            color=arm_colour(arms=ARMS, colours=COLOURS, legend=False),
        )
        .properties(width=(WIDTH - 140) // 2, height=110)
        .facet(
            row=alt.Row("row:N", title=None, sort=list(ROW_NAMES.values())),
            column=alt.Column("group:N", title=None, sort=[SIGNED, MVA]),
        )
        .resolve_scale(x="independent")
        .properties(
            title=alt.TitleParams(
                ["Absolute CRPS of the six arms at days 1 to 10 (interim)", "One dot per arm. Four test months, three seeds, 1.15 widening. Scales differ between panels."],
                anchor="start",
            )
        )
    )  # fmt: skip
    save(chart=chart, name="figure_2_absolute_crps.svg")


def figure_folds() -> None:
    """Figure 3: each contrast on clean rows in each test month."""
    c = pl.read_parquet(SCORES / "contrasts.parquet").filter(
        pl.col("row_set") == "clean", pl.col("group").is_in([SIGNED, MVA])
    )
    rows = [
        {"group": r["group"], "contrast": r["contrast"], "month": month, "difference": v}
        for r in c.iter_rows(named=True)
        for month, v in r["fold_differences"].items()
    ]
    base = alt.Chart(pl.DataFrame(rows).to_pandas())
    dots = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("contrast:N", sort=CONTRAST_ORDER, title=None),
        x=alt.X("difference:Q", title="Contrast, CRPS (points; negative is better)"),
        color=alt.Color(
            "month:N",
            scale=alt.Scale(domain=sorted(MONTHS), range=["#306BFF", "#FF4901", "#B701FF", "#17E58F"]),
            legend=alt.Legend(title="Test month", orient="top", columns=4),
        ),
    )  # fmt: skip
    zero = base.mark_rule(color="#888").encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        (zero + dots)
        .properties(width=(WIDTH - 200) // 2, height=170)
        .facet(column=alt.Column("group:N", title=None, sort=[SIGNED, MVA]))
        .properties(
            title=alt.TitleParams(
                ["Each contrast on clean rows in each test month (interim)", "One dot per test month, three seeds pooled. Grey line: zero."],
                anchor="start",
            )
        )
    )  # fmt: skip
    save(chart=chart, name="figure_3_folds.svg")


def clean_signed(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's world-0 clean-row forecasts at the signed-MW six for one month."""
    f = score.read_arm(arm=arm, month=month, world=0, seeds=SEEDS, factor=score.WIDENING)
    keep = score.row_mask(frame=f, rule=score.ROW_SETS["clean"][1])
    keep &= score.group_of(primary=f["primary"].to_numpy())[SIGNED]
    return f.filter(pl.Series(keep)).with_columns(arm=pl.lit(arm), month=pl.lit(month))


def lead_table() -> pl.DataFrame:
    """Return the mean CRPS by arm and lead day, clean rows, signed-MW six."""
    parts = []
    for arm in ARMS:
        for month in MONTHS:
            g = clean_signed(arm=arm, month=month).with_columns(
                day=(pl.col("lead") / 24.0).floor().cast(pl.Int32) + 1
            )
            parts.append(g.group_by("arm", "day").agg(pl.col("score").sum(), pl.len().alias("n")))
    return (
        pl.concat(parts)
        .group_by("arm", "day")
        .agg(pl.col("score").sum(), pl.col("n").sum())
        .with_columns(crps=pl.col("score") / pl.col("n") * POINTS)
        .filter(pl.col("day") <= 10)
        .sort("arm", "day")
    )


def figure_lead() -> None:
    """Figure 4: CRPS against the lead day, absolute and minus S1."""
    t = lead_table()
    s1 = t.filter(pl.col("arm") == "S1").select("day", s1=pl.col("crps"))
    t = t.join(s1, on="day").with_columns(minus_s1=pl.col("crps") - pl.col("s1"))
    colour = arm_colour(arms=ARMS, colours=COLOURS)
    top = (
        alt.Chart(t.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:O", title="Lead (days)"),
            y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
            color=colour,
        )
        .properties(width=WIDTH - 90, height=180)
    )
    bottom = (
        alt.Chart(t.filter(pl.col("arm") != "S1").to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:O", title="Lead (days)"),
            y=alt.Y("minus_s1:Q", title="CRPS minus S1 (points)"),
            color=arm_colour(arms=ARMS, colours=COLOURS, legend=False),
        )
        .properties(width=WIDTH - 90, height=180)
    )
    chart = alt.vconcat(top, bottom).properties(
        title=alt.TitleParams(
            ["CRPS against the lead, clean rows (signed-MW six, interim)", "Top: CRPS. Bottom: the same minus S1's (negative is better than S1). Four test months, three seeds, 1.15 widening."],
            anchor="start",
        )
    )  # fmt: skip
    save(chart=chart, name="figure_4_crps_by_lead.svg")


def figure_sweep() -> None:
    """Figure 5: median share of a 5% step in the lags that reaches the forecast median, by k."""
    t = sweep.share_report(
        table=sweep.load_tables(), step="c050", band="24-72 h", widened=False
    ).filter(pl.col("arm").is_in(["R-inj", "S1", "S2", "S3", "S3-rw"]))
    arms = ["R-inj", "S1", "S2", "S3", "S3-rw"]
    chart = (
        alt.Chart(t.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("k:O", title="Lags stepped (1: newest lag; 2: and the daily lag; 3: and the weekly lag)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("median_share:Q", title="Median share of the step in the forecast median"),
            color=arm_colour(arms=arms, colours=[COLOURS[ARMS.index(a)] for a in arms]),
        )
        .properties(
            width=WIDTH - 90, height=220,
            title=alt.TitleParams(
                ["Share of a step in the recent power that reaches the forecast (seed 0)", "Step of 5% of capacity, both signs, forecasts 24 to 72 hours ahead, four test months. 1 is the full step. S2, S3, and S3-rw overlap."],
                anchor="start",
            ),
        )
    )  # fmt: skip
    save(chart=chart, name="figure_5_sweep.svg")


def forecasts(*, month: str, site: str, arms: list[str]) -> pl.DataFrame:
    """Return the world-0 quantile rows of the given arms for one Primary and month."""
    primary = data.sites().index(site)
    frames = []
    for arm in arms:
        f = score.read_arm(arm=arm, month=month, world=0, seeds=SEEDS, factor=score.WIDENING)
        frames.append(f.filter(pl.col("primary") == primary).with_columns(arm=pl.lit(arm)))
    return pl.concat(frames)


def figure_examples(*, month: str, site: str) -> None:
    """Figures 6 and 7: a day-ahead and a 10-day forecast from one origin."""
    f = forecasts(month=month, site=site, arms=EXAMPLE_ARMS)
    origins = sorted(f["origin_us"].unique().to_list())
    origin = origins[len(origins) // 2]
    one = f.filter(pl.col("origin_us") == origin)
    for number, name, hours in ((6, "day_ahead", 48.0), (7, "10_day", 240.0)):
        d = one.filter(pl.col("lead") <= hours).with_columns(
            time=pl.from_epoch("valid_us", time_unit="us"),
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS,
            actual=pl.col("actual") * POINTS,
        )  # fmt: skip
        colour = arm_colour(arms=EXAMPLE_ARMS, colours=EXAMPLE_COLOURS)
        x = alt.X("time:T", title="Valid time (UTC)")
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
            alt.Chart(d.filter(pl.col("arm") == "S1").to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color="#333", strokeDash=[2, 2])
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        chart = (band + median + truth).properties(
            width=WIDTH - 90, height=220,
            title=alt.TitleParams(
                [f"{name.replace('_', '-')} forecast of Primary {site}, origin {d['time'].min():%Y-%m-%d}", "Line: median. Band: 5% to 95%. Dashed: the cleaned reading."],
                anchor="start",
            ),
        )  # fmt: skip
        save(chart=chart, name=f"figure_{number}_example_{name}.svg")


def real_events(*, count: int = 3) -> list[dict]:
    """Return the longest padded real events (with forecast rows) of each metering group."""
    padded = data.real_event_mask()
    sites = data.sites()
    chosen: list[dict] = []
    for label, members in ((SIGNED, score.SIGNED_MW_LABELS), (MVA, score.MVA_ONLY_LABELS)):
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
    """Figure 8: six real events, three per metering group, with S1, S3, and S3-rw."""
    import e00087_common as common  # ty: ignore[unresolved-import]

    times = common.grid_times()
    power, _ = data.load_stack(include_eval=False)
    month_of = {m: (datetime.fromisoformat(f"{m}-01").replace(tzinfo=UTC)) for m in MONTHS}
    picked: list[dict] = []
    per_group = {SIGNED: 0, MVA: 0}
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
        colour = arm_colour(arms=EVENT_ARMS, colours=EVENT_COLOURS, legend=i == 1)
        x = alt.X("time:T", title=None, scale=alt.Scale(domain=[t0, t1]))
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
            .mark_area(opacity=0.10)
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
    save(chart=chart, name="figure_8_real_events.svg")


def lead_window_contrast(*, arm: str, base: str, low: float, high: float) -> pl.DataFrame:
    """Return `arm` minus `base` on clean rows at the signed-MW six for leads in [low, high) hours.

    The interval resamples (test month, origin day) blocks with replacement.
    """
    blocks = []
    for month in MONTHS:
        a = clean_signed(arm=arm, month=month)
        b = clean_signed(arm=base, month=month)
        assert a["valid_us"].equals(b["valid_us"])
        assert a["primary"].equals(b["primary"])
        diff = ((a["score"] - b["score"]) * POINTS).alias("diff")
        d = a.select("month", "origin_us", "lead").with_columns(diff)
        d = d.filter(pl.col("lead") >= low, pl.col("lead") < high).with_columns(
            day=pl.col("origin_us") // 86_400_000_000
        )
        blocks.append(d.group_by("month", "day").agg(pl.col("diff").sum(), pl.len().alias("n")))
    t = pl.concat(blocks).sort("month", "day")
    total, n = t["diff"].to_numpy(), t["n"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(t), size=(BOOTSTRAP_DRAWS, len(t)))
    draws = total[idx].sum(axis=1) / n[idx].sum(axis=1)
    per_month = t.group_by("month").agg(
        (pl.col("diff").sum() / pl.col("n").sum()).alias("difference")
    )
    return pl.DataFrame(
        {
            "contrast": f"{arm} - {base}",
            "lead_hours": f"{low:g} to {high:g}",
            "n": int(n.sum()),
            "difference": float(total.sum() / n.sum()),
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            **{
                f"fold_{m}": float(per_month.filter(pl.col("month") == m)["difference"][0])
                for m in MONTHS
            },
        }
    )


def median_checks() -> pl.DataFrame:
    """Return, per arm and group, the clean-row error of the median forecast and median minus mean.

    The stored forecasts hold 11 quantiles and no mean, so the mean is the average of the 11
    quantiles (a proxy). Both columns are in points of capacity.
    """
    rows = []
    for arm in ARMS:
        parts = []
        for month in MONTHS:
            f = score.read_arm(arm=arm, month=month, world=0, seeds=SEEDS, factor=score.WIDENING)
            keep = score.row_mask(frame=f, rule=score.ROW_SETS["clean"][1])
            parts.append(f.filter(pl.Series(keep)))
        f = pl.concat(parts)
        mean_q = f.select(pl.mean_horizontal(*pr.LEVEL_NAMES).alias("m"))["m"]
        for group in (SIGNED, MVA):
            mask = pl.Series(score.group_of(primary=f["primary"].to_numpy())[group])
            g = f.filter(mask)
            rows.append(
                {
                    "arm": arm,
                    "group": group,
                    "mae_q50": float(np.abs((g["q50"] - g["actual"]).to_numpy()).mean()) * POINTS,
                    "median_minus_mean": float((g["q50"] - mean_q.filter(mask)).to_numpy().mean())
                    * POINTS,
                }
            )
    return pl.DataFrame(rows)


def print_numbers() -> None:
    """Print every table whose numbers the page quotes."""
    with pl.Config(
        tbl_rows=200, tbl_cols=-1, tbl_width_chars=240, fmt_str_lengths=60, fmt_float="mixed"
    ):
        a = pl.read_parquet(SCORES / "absolute.parquet").filter(
            pl.col("group").is_in([SIGNED, MVA])
        )
        print("ABSOLUTE (CRPS, pinball and coverage at the tails)")
        print(a.sort("group", "row_set", "arm").with_columns(pl.col(pl.Float64).round(3)))
        print("MINIMUM DETECTABLE EFFECT")
        print(
            pl.read_parquet(SCORES / "mde.parquet")
            .sort("group", "row_set")
            .with_columns(pl.col(pl.Float64).round(3))
        )
        c = pl.read_parquet(SCORES / "contrasts.parquet")
        print("CONTRASTS")
        print(
            c.drop("fold_differences").sort("group", "row_set", "contrast")
            .with_columns(pl.col(pl.Float64).round(3))
        )  # fmt: skip
        print("CLEAN-ROW CONTRASTS BY TEST MONTH")
        print(
            c.filter(pl.col("row_set") == "clean").unnest("fold_differences")
            .sort("group", "contrast").with_columns(pl.col(pl.Float64).round(3))
        )  # fmt: skip
        print("SWEEP (c050, 24-72 h, seed 0)")
        print(
            sweep.share_report(table=sweep.load_tables(), step="c050", band="24-72 h", widened=False)
            .with_columns(pl.col(pl.Float64).round(3))
        )  # fmt: skip
        print("LEAD TABLE (clean rows, signed-MW six)")
        print(lead_table().with_columns(pl.col(pl.Float64).round(3)))
        print("CLEAN ROWS: ERROR OF THE MEDIAN FORECAST, MEDIAN MINUS MEAN")
        print(median_checks().with_columns(pl.col(pl.Float64).round(3)))
        print("H-LVL ON CLEAN ROWS BY LEAD")
        lead_rows = [
            lead_window_contrast(arm=arm, base=base, low=low, high=high)
            for arm, base in (("H-lvl", "R-inj"), ("H-lvl", "S1"))
            for low, high in ((12.0, 36.0), (0.0, 240.5))
        ]
        print(pl.concat(lead_rows).with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:
    """Draw every figure and print the numbers."""
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_absolute()
    figure_folds()
    figure_lead()
    figure_sweep()
    figure_examples(month="2026-01", site=score.SIGNED_MW_LABELS[0])
    figure_events()
    print_numbers()


if __name__ == "__main__":
    main()
