# ruff: noqa: E501
"""Draw the figures of experiment 460 from the stored score tables and forecasts (no fit).

- Figures 1 and 2: each arm minus B-mix on rows with no labelled event, real harm rows, and rows
  inside real events (NRA-corrected actuals), at the signed-MW six and the MVA-only eight.
- Figure 3: the AUC of the spread signals with their month-block and event-cluster intervals.
- Figure 4: absolute scores of every arm on rows with no labelled event.
- Figures 5 and 6: example forecasts (7 days of day-ahead forecasts; one run's whole horizon).
- Figures 7 to 12: day-ahead forecasts around six real load-shift events, three per group.

```bash
source /tmp/e460_env.sh
uv run --no-sync python ml-experiments/00460-*/e00460_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00460_arms as arms  # ty: ignore[unresolved-import]
import e00460_score as scoring  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00460-lag-dropout-network-forecast-disagreement-as-a-trust-and-width-signal"
)
WIDTH: Final[int] = 680
LABELS: Final[dict[str, str]] = {
    "S1d": "b: lag-dropout, all lags",
    "S1d-c": "c: b, spread widening",
    "S1d-d": "d: b, spread mixture",
    "S1": "a: stored S1",
    "S1w": "f: S1, widening 1.15",
    "S1-ens3": "g: S1 3-seed ensemble",
    "S1d-c0": "c0: b, table P only",
}
CONTRAST_OF: Final[dict[str, str]] = {
    "S1d": "C8",
    "S1d-c": "C2",
    "S1d-d": "C3",
    "S1": "C11",
    "S1-ens3": "C7",
}
CLASS_TITLES: Final[dict[str, str]] = {
    "clean_real": "Rows with no labelled event",
    "harm_real": "Real harm rows",
    "tgt_real": "Rows inside real events (NRA-corrected actuals)",
}
EVENTS: Final[tuple[tuple[str, int, int], ...]] = (
    ("D5", 112321, 112550),
    ("D5", 111746, 111922),
    ("D14", 106721, 106748),
    ("D8", 111596, 111696),
    ("D12", 107836, 107933),
    ("D8", 110276, 110612),
)
"""Primary, onset, and end grid index of each plotted real event: three signed-MW, three MVA-only."""
MARGIN: Final[int] = 3 * 48
LINE_COLOURS: Final[dict[str, str]] = {
    "Actual": theme.BLACK_1,
    "B-mix": theme.BRAND_ORANGE,
    "S1": theme.DATA_PURPLE,
    "S1d (b)": theme.DATA_BLUE,
}


def figure_contrast(*, group: str, group_name: str, number: int) -> None:
    """Draw the contrast figure of a group: each arm minus B-mix on the three row classes."""
    table = pl.read_csv(scoring.SCORE_DIR / "contrasts.csv").filter(
        (pl.col("group") == group)
        & (pl.col("actual") == "nra")
        & (pl.col("band") == "all")
        & pl.col("contrast").is_in(list(CONTRAST_OF.values()))
        & pl.col("cls").is_in(list(CLASS_TITLES))
    )
    table = table.with_columns(
        label=pl.col("a").replace(LABELS)
        + pl.lit(" (")
        + pl.col("months").cast(pl.String)
        + pl.lit(" months)"),
        cls_title=pl.col("cls").replace(CLASS_TITLES),
    )
    order = [
        f"{LABELS[a]} ({m} months)"
        for a in CONTRAST_OF
        for m in sorted(table.filter(pl.col("a") == a)["months"].unique().to_list(), reverse=True)
    ]
    base = alt.Chart()
    y = alt.Y("label:N", sort=order, title=None)
    rules = base.mark_rule(strokeWidth=2, color=theme.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low:Q", title="Score minus B-mix (points of capacity)"), x2="high:Q"
    )
    dots = base.mark_point(filled=True, size=60, color=theme.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="diff:Q"
    )
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, rules, dots, data=table.to_pandas())
        .properties(width=WIDTH - 290, height=130)
        .facet(
            row=alt.Row(
                "cls_title:N",
                title=None,
                sort=list(CLASS_TITLES.values()),
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=alt.TitleParams(
                [
                    f"Each arm minus B-mix at the {group_name}; left of the dashed line is better than B-mix",
                    "Dot: mean over months. Line: 95% interval (months resampled on the first row,",
                    "real events resampled on the other two). Arms c and d have no 2025-10 forecast.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / f"figure_{number}_contrast_{group}.svg")


def figure_auc() -> None:
    """Figure 3: AUC of the spread signals per group, with the injected-world control."""
    auc = pl.read_csv(scoring.SCORE_DIR / "auc.csv").filter(pl.col("signal") != "max")
    auc = auc.with_columns(
        signal_label=pl.col("signal").replace(
            {"mean": "Mean spread", "lead": "Lead in hours (no model)"}
        ),
    )
    base = alt.Chart()
    y = alt.Y("signal_label:N", title=None, sort=["Mean spread", "Lead in hours (no model)"])
    rules = base.mark_rule(strokeWidth=2, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "month_low:Q",
            title="Area under the curve (0.5 = no separation)",
            scale=alt.Scale(domain=[0.25, 0.75]),
        ),
        x2="month_high:Q",
    )
    cluster = base.mark_rule(strokeWidth=2, color=theme.DATA_SKY, opacity=0.6).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="event_low:Q", x2="event_high:Q"
    )
    dots = base.mark_point(filled=True, size=70, color=theme.DATA_BLUE).encode(y=y, x="auc:Q")  # ty: ignore[unresolved-attribute]
    bar = [
        base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(v))  # ty: ignore[unresolved-attribute]
        for v in (0.5, 0.65)
    ]
    chart = (
        alt.layer(*bar, cluster, rules, dots, data=auc.to_pandas())  # ty: ignore[unresolved-attribute]
        .properties(width=WIDTH - 260, height=70)
        .facet(
            row=alt.Row("group:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Can the spread separate real harm rows from rows with no labelled event?",
                    "Dark line: 95% interval, months resampled. Pale: real events resampled.",
                    "Dashed lines mark 0.5 (no separation) and the 0.65 bar.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_3_auc.svg")


def figure_absolute() -> None:
    """Figure 4: absolute scores on rows with no labelled event, per group and arm."""
    table = pl.read_csv(scoring.SCORE_DIR / "absolute.csv").filter(
        (pl.col("actual") == "metered")
        & (pl.col("band") == "all")
        & (pl.col("cls") == "clean_real")
        & pl.col("arm").is_in([*LABELS, "B-mix"])
        & (pl.col("group") != "report_only")
    )
    labels = {**LABELS, "B-mix": "e: B-mix"}
    table = table.with_columns(
        label=pl.col("arm").replace(labels)
        + pl.lit(" (")
        + pl.col("months").cast(pl.String)
        + pl.lit(" months)")
    )
    chart = (
        alt.Chart(table.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=theme.DATA_BLUE, aria=False)
        .encode(
            y=alt.Y("label:N", title=None),
            x=alt.X(
                "crps:Q",
                title="Mass-weighted pinball score (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(width=WIDTH - 290, height=190)
        .facet(
            row=alt.Row("group:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Absolute score on rows with no labelled event; smaller is better",
                    "Arms with 4 months lack 2025-10, so compare only arms with the same month count.",
                ],
                anchor="start",
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_4_absolute_scores.svg")


def load_arm(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's cell-0 forecast table for a month."""
    return pl.read_parquet(arms.prediction_path(arm=arm, month=month, cell=0))


def month_of(*, primary: int, onset: int, end: int) -> str:
    """Return the scored month with the most day-ahead rows of a Primary inside the event window."""
    counts = {}
    for month in arms.MONTHS:
        frame = pl.read_parquet(
            arms.prediction_path(arm="B-mix", month=month, cell=0),
            columns=["primary", "target", "lead_hours"],
        )
        counts[month] = frame.filter(
            (pl.col("primary") == primary)
            & pl.col("target").is_between(onset - MARGIN, end + MARGIN)
            & pl.col("lead_hours").is_between(12.0, 36.0)
        ).height
    return max(counts, key=counts.__getitem__)


def band_panel(
    *,
    frame: pl.DataFrame,
    x: str,
    x_title: str,
    title: str,
    shade: tuple[float, float] | None = None,
) -> alt.Chart:
    """Return one panel: actual, median, 50% and 90% bands, optionally a shaded event."""
    layers = []
    if shade is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"a": [shade[0]], "b": [shade[1]]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(color=theme.GREY_3, opacity=0.35)
            .encode(x="a:Q", x2="b:Q")
        )
    data = frame.to_pandas()  # ty: ignore[unresolved-attribute]
    layers.append(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_area(color=theme.BRAND_ORANGE, opacity=0.15)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y("q05:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
            y2="q95:Q",
        )
    )
    layers.append(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_area(color=theme.BRAND_ORANGE, opacity=0.3)
        .encode(x=f"{x}:Q", y="q25:Q", y2="q75:Q")
    )
    layers.append(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(color=theme.BRAND_ORANGE, aria=False)
        .encode(x=f"{x}:Q", y="q50:Q")
    )
    layers.append(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(color=theme.BLACK_1, strokeWidth=1.4, aria=False)
        .encode(x=f"{x}:Q", y="actual:Q")
    )
    return alt.layer(*layers).properties(width=WIDTH - 110, height=150, title=title)  # ty: ignore[invalid-return-type]


def pct(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Express the actual and quantile columns in percent of capacity."""
    return frame.with_columns(
        [(pl.col(c) * 100) for c in ("actual", "q05", "q25", "q50", "q75", "q95")]
    )


def figure_examples() -> None:
    """Figures 5 and 6: day-ahead forecasts for one week and one run's full horizon, two seasons."""
    primary = fit.prep.primaries()["site"].to_list().index("D5")
    day_ahead, horizon = [], []
    for month, label in (("2026-01", "winter, 2026-01"), ("2026-06", "summer, 2026-06")):
        for arm, arm_label in (("B-mix", "B-mix"), ("S1d-c", "arm c")):
            if arm == "S1d-c" and month == "2025-10":
                continue
            frame = load_arm(arm=arm, month=month).filter(pl.col("primary") == primary)
            da = (
                frame.filter(pl.col("lead_hours").is_between(12.0, 36.0))
                .unique(subset="target", keep="first")
                .sort("target")
            )
            first = int(da["target"].min())  # ty: ignore[invalid-argument-type]
            week = pct(
                frame=da.filter(pl.col("target").is_between(first, first + 7 * 48))
            ).with_columns(x=(pl.col("target") - first) / 48.0)
            day_ahead.append(
                band_panel(
                    frame=week,
                    x="x",
                    x_title="Days into the month",
                    title=f"D5, {label}: day-ahead, {arm_label}",
                )
            )
            origins = sorted(frame["origin"].unique().to_list())
            origin = origins[len(origins) // 2]
            run = pct(frame=frame.filter(pl.col("origin") == origin).sort("target")).with_columns(
                x=pl.col("lead_hours") / 24.0
            )
            horizon.append(
                band_panel(
                    frame=run,
                    x="x",
                    x_title="Lead (days)",
                    title=f"D5, {label}: one run, {arm_label}",
                )
            )
    note = "Orange line: median. Dark band: 50% interval. Pale band: 90%. Black: actual."
    alt.vconcat(*day_ahead).properties(
        title=alt.TitleParams(
            ["Example day-ahead forecasts (12 to 36 hours ahead)", note], anchor="start"
        )
    ).save(FIGURE_DIR / "figure_5_example_day_ahead.svg")
    alt.vconcat(*horizon).properties(
        title=alt.TitleParams(
            ["Example forecasts over the whole horizon of one run", note], anchor="start"
        )
    ).save(FIGURE_DIR / "figure_6_example_horizon.svg")


def figure_event(*, site: str, onset: int, end: int, number: int) -> None:
    """Draw one real event: actual, B-mix band, and the medians of S1, S1d, and B-mix."""
    primary = fit.prep.primaries()["site"].to_list().index(site)
    month = month_of(primary=primary, onset=onset, end=end)
    lo, hi = onset - MARGIN, end + MARGIN
    parts = {}
    for arm, name in (("B-mix", "B-mix"), ("S1", "S1"), ("S1d", "S1d (b)")):
        frame = (
            load_arm(arm=arm, month=month)
            .filter((pl.col("primary") == primary) & pl.col("lead_hours").is_between(12.0, 36.0))
            .unique(subset="target", keep="first")
        )
        parts[name] = pct(frame=frame.filter(pl.col("target").is_between(lo, hi))).with_columns(
            x=(pl.col("target") - onset) / 48.0
        )
    lines = pl.concat(
        [
            parts["B-mix"]
            .select("x", pl.col("actual").alias("median"))
            .with_columns(series=pl.lit("Actual"))
        ]
        + [
            p.select("x", pl.col("q50").alias("median")).with_columns(series=pl.lit(n))
            for n, p in parts.items()
        ]
    )
    band = parts["B-mix"]
    x = alt.X("x:Q", title="Days since the event began (shaded: the event)")
    shade = (
        alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - onset) / 48.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rect(color=theme.GREY_3, opacity=0.35)
        .encode(x="a:Q", x2="b:Q")
    )  # ty: ignore[unresolved-attribute]
    area = (
        alt.Chart(band.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_area(color=theme.BRAND_ORANGE, opacity=0.18)
        .encode(
            x=x,
            y=alt.Y("q05:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
            y2="q95:Q",
        )
    )  # ty: ignore[unresolved-attribute]
    line = alt.Chart(lines.to_pandas()).mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y="median:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=list(LINE_COLOURS), range=list(LINE_COLOURS.values())), legend=alt.Legend(title=None, orient="top")),
    )  # fmt: skip
    group = "MVA-only eight" if site in fit.MVA_ONLY else "signed-MW six"
    chart = (shade + area + line).properties(
        width=WIDTH - 110, height=190,
        title=alt.TitleParams([f"Primary {site} ({group}), real event, {month} forecasts, 12 to 36 h ahead", "Orange band: B-mix 90% interval. Lines: medians. Shaded: the event."], anchor="start"),
    )  # fmt: skip
    chart.save(FIGURE_DIR / f"figure_{number}_event_{site}_{onset}.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrast(group="signed_six", group_name="signed-MW six", number=1)
    figure_contrast(group="mva_eight", group_name="MVA-only eight", number=2)
    figure_auc()
    figure_absolute()
    figure_examples()
    for i, (site, onset, end) in enumerate(EVENTS, start=7):
        figure_event(site=site, onset=onset, end=end, number=i)


if __name__ == "__main__":
    main()
