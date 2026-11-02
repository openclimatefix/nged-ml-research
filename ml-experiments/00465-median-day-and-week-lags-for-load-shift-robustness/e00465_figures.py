# ruff: noqa: E501
r"""Draw experiment 465's figures from the scored rows, the sweep sums, and the stored forecasts.

Writes `docs/ml-experiments/00465-*/figure_*.svg` and `contrasts.csv` (the month-block contrasts of
`e00465_score.py`, recomputed from `work/scores/rows.parquet`). Arms: (a) original lags, (c) median
lags, (f) the 50/50 mixture of (a) and the lagless forecast.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\\
ml-experiments/00379-bg2-components-all-16-primaries-to-14-days:\\
ml-experiments/00383-load-shift-xgboost-oracle-first-part-1:\\
ml-experiments/00392-xgboost-nra-load-shift-masked-targets:\\
ml-experiments/00402-lag-gate-learned-weight-injected-load-shifts-part-1:\\
ml-experiments/00465-median-day-and-week-lags-for-load-shift-robustness \\
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00465-*/e00465_figures.py
```
"""

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00379_adapter as adapter  # ty: ignore[unresolved-import]
import e00392_score as s392  # ty: ignore[unresolved-import]
import e00402_mixture as mixture  # ty: ignore[unresolved-import]
import e00465_early as early  # ty: ignore[unresolved-import]
import e00465_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.eval_protocol import metrics
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS

SLUG: Final[str] = "00465-median-day-and-week-lags-for-load-shift-robustness"
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 600
ARM_LABELS: Final[dict[str, str]] = {
    "a": "(a) original lags",
    "c": "(c) median lags",
    "f": "(f) mixture of (a) and lagless",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a": theme.DATA_BLUE,
    "c": theme.BRAND_ORANGE,
    "f": theme.DATA_PURPLE,
}
GROUP_DESCRIPTIONS: Final[dict[str, str]] = {
    "signed-MW six": "one of the six Primaries metered in signed MW",
    "MVA-only eight": "one of the eight Primaries metered in MVA only",
}
GROUP_NAMES: Final[dict[str, str]] = {
    "signed-MW six": "Six Primaries metered in signed MW",
    "MVA-only eight": "Eight Primaries metered in MVA only",
}


def scale() -> alt.Scale:
    """Return the colour scale of the three plotted arms."""
    return alt.Scale(domain=[ARM_LABELS[a] for a in ARM_COLOURS], range=list(ARM_COLOURS.values()))


def contrasts() -> pl.DataFrame:
    """Compute the month-block contrasts at both groups and write `contrasts.csv`."""
    rows = pl.read_parquet(score.OUT / "rows.parquet").filter(pl.col("widening") == 1.0)
    out = []
    for group in score.GROUPS:
        for row_set in score.ROW_SETS:
            for band, band_cond in score.BANDS.items():
                cond = band_cond & s392.GROUP_VIEWS[group] & s392.ROW_SETS[row_set]
                for arm, control in score.CONTRASTS:
                    try:
                        interval = metrics.month_bootstrap_contrast(
                            treatment=rows.filter(pl.col("arm") == arm),
                            reference=rows.filter(pl.col("arm") == control),
                            metric="crps",
                            where=cond,
                            level_percent=score.LEVEL_PERCENT,
                            n_resamples=score.RESAMPLES,
                        )
                    except ValueError:
                        continue
                    table = s392.contrast_table(
                        rows=rows, arm=arm, control=control, where=cond, by=[]
                    )
                    months = [c for c in table.columns if c != "pooled"]
                    out.append(
                        {
                            "group": group,
                            "row_set": row_set,
                            "band": band,
                            "arm": arm,
                            "control": control,
                            "difference": interval["difference"],
                            "lower": interval["lower"],
                            "upper": interval["upper"],
                            "n_months": interval["n_months"],
                            "n_half_hours": interval["n_half_hours"],
                            "months_worse": sum(float(table[m][0]) > 0 for m in months),
                            "months": len(months),
                        }
                    )
    frame = pl.DataFrame(out)
    frame.write_csv(FIGURE_DIR / "contrasts.csv")
    return frame


def contrast_figure(*, frame: pl.DataFrame) -> None:
    """Draw arms (c) and (e) minus the controls, with intervals, for rows with no labelled event."""
    data = frame.filter(
        pl.col("row_set").is_in(["outside", "during", "after7d"]),
        pl.col("band") == "0-240 h",
        pl.col("control") != "e",
        pl.col("group") == "signed-MW six",
    ).with_columns(
        label=pl.lit("(") + pl.col("arm") + pl.lit(") minus (") + pl.col("control") + pl.lit(")"),
        panel=pl.col("row_set").replace(
            {
                "outside": "Rows with no labelled event",
                "during": "Inside labelled events",
                "after7d": "Seven days after an event",
            }
        ),
    )
    base = alt.Chart(data.to_pandas())
    rule = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    line = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q"),
        x2="upper:Q",
        y=alt.Y("label:N", title=None),
        color=alt.value(theme.DATA_BLUE),
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="CRPS difference (points of capacity; negative = first arm better)",
        ),
        y=alt.Y("label:N", title=None),
        color=alt.value(theme.BRAND_ORANGE),
    )
    chart = (
        alt.layer(rule, line, dots)
        .properties(width=WIDTH - 160, height=130)
        .facet(
            row=alt.Row("panel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Median day and week lags (c) and the 50/50 mixture (e) against the controls,",
                    "six Primaries metered in signed MW, leads up to 240 hours.",
                    "Dot: estimate. Line: Bonferroni month-block interval over 4 months (3 inside events).",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def sweep_figure() -> None:
    """Draw the median shift as a share of the step against the step's length."""
    sweep = pl.read_parquet(early.fit.WORK_DIR / "sweep" / "sweep.parquet").with_columns(
        group=pl.col("site").replace_strict(
            {
                **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW six"),
                **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only eight"),
            },
            default=None,
        )
    )
    data = (
        sweep.filter(
            pl.col("kind") == "step",
            pl.col("widening") == 1.0,
            pl.col("share") == 0.05,
            pl.col("band") == "24-72 h",
            pl.col("arm").is_in(["a", "c"]),
            pl.col("group").is_not_null(),
        )
        .group_by("group", "arm", "variant", "sign")
        .agg(pl.col("shift_sum").sum(), pl.col("n").sum())
        .with_columns(share=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * 0.05))
        .group_by("group", "arm", "variant")
        .agg(pl.col("share").mean())
        .with_columns(
            hours=pl.col("variant").replace_strict(
                {"6h": 6, "24h": 24, "48h": 48, "3d": 72, "5d": 120, "7d": 168},
                return_dtype=pl.Int64,
            ),
            arm_label=pl.col("arm").replace(ARM_LABELS),
            panel=pl.col("group").replace(GROUP_NAMES),
        )
    )
    base = alt.Chart(data.to_pandas()).encode(
        x=alt.X(
            "hours:Q",
            title="Length of the injected step, ending at the forecast origin (hours)",
            axis=alt.Axis(values=[6, 24, 48, 72, 120, 168]),
        ),
        y=alt.Y("share:Q", title="Median shift / step", scale=alt.Scale(domain=[0, 0.5])),
        color=alt.Color("arm_label:N", scale=scale(), legend=alt.Legend(title=None, orient="top")),
    )
    chart = (
        (base.mark_line(point=True))
        .properties(width=WIDTH - 60, height=170)  # ty: ignore[unresolved-attribute]
        .facet(
            row=alt.Row("panel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "A step of 5% of capacity in the last readings: how far the median forecast moves,",
                    "as a share of the step, at 24 to 72 hours lead (smaller is more robust).",
                ],
                anchor="start",
                fontSize=11,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_sweep_share.svg")


def level_figure() -> None:
    """Draw the response to a persistent step of 5% of capacity against days since it began."""
    sweep = pl.read_parquet(early.fit.WORK_DIR / "sweep" / "sweep.parquet").filter(
        pl.col("kind") == "level",
        pl.col("widening") == 1.0,
        pl.col("band") == "24-72 h",
        pl.col("arm").is_in(["a", "c"]),
        pl.col("site").is_in(sorted(SIGNED_MW_LABELS)),
    )
    data = (
        sweep.group_by("arm", "sign", "day")
        .agg(pl.col("shift_sum").sum(), pl.col("n").sum())
        .with_columns(share=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * 0.05))
        .group_by("arm", "day")
        .agg(pl.col("share").mean())
        .with_columns(arm_label=pl.col("arm").replace(ARM_LABELS))
    )
    chart = (
        alt.Chart(data.to_pandas())
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Days since the persistent step began"),
            y=alt.Y("share:Q", title="Median shift / step", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm_label:N", scale=scale(), legend=alt.Legend(title=None, orient="top")
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=200,
            title=alt.TitleParams(
                [
                    "A persistent step of 5% of capacity in the readings: the share the forecast",
                    "median follows, six Primaries metered in signed MW, at 24 to 72 hours lead.",
                ],
                anchor="start",
                fontSize=11,
            ),
        )
    )
    chart.save(FIGURE_DIR / "figure_3_level_change.svg")


def forecast_window(
    *, label: str, origin: datetime, days: int, actuals: pl.DataFrame, series: pl.DataFrame
) -> pl.DataFrame:
    """Return the quantiles of arms a, c, f and the actual for one Primary and one origin."""
    capacity = series.filter(pl.col("label") == label)["capacity_mw"][0]
    series_id = series.filter(pl.col("label") == label)["time_series_id"][0]
    refit = max(r for r in s392.FOLD_STARTS if r <= origin)
    fold = s392.FOLD_STARTS.index(refit)
    one = series.filter(pl.col("label") == label)
    frames = {}
    for arm, source in (("a", "a_r0_lagged"), ("b", "b_r0_lagless"), ("c", "c_robust")):
        stored = s392.read_fold(directory=early.ARMS[source], sites=[label], fold=fold)
        assert stored is not None
        frames[arm] = (
            adapter.forecast_frame(stored=stored, series=one, unit="fraction")
            .filter(pl.col("power_fcst_init_time") == origin)
            .sort("valid_time")
        )
    keys = list(s392.KEY_COLUMNS)
    q = list(s392.QUANTILES)
    joined = frames["a"].join(
        frames["b"].select(*keys, *[pl.col(c).alias(f"{c}_2") for c in q]), on=keys, validate="1:1"
    )
    pair = mixture.Pair(
        q1=joined.select(q).to_numpy(), q2=joined.select([f"{c}_2" for c in q]).to_numpy()
    )
    mix = joined.select(*keys).hstack(
        pl.DataFrame(pair.quantiles(w=np.full(joined.height, 0.5)), schema=q, orient="row")
    )
    pieces = []
    for arm, frame in (("a", frames["a"]), ("c", frames["c"]), ("f", mix)):
        pieces.append(
            frame.with_columns(arm=pl.lit(ARM_LABELS[arm]))
            .filter(pl.col("valid_time") <= origin + pl.duration(days=days))
            .select("valid_time", "arm", *[(pl.col(c) * 100.0 / capacity).alias(c) for c in q])
        )
    out = pl.concat(pieces).with_columns(
        hour=(pl.col("valid_time") - origin).dt.total_minutes() / 60.0
    )
    act = actuals.filter(pl.col("time_series_id") == series_id).select(
        "valid_time", actual=pl.col("actual") * 100.0 / capacity
    )
    return out.join(act, on="valid_time", how="left")


def series_chart(
    *, data: pl.DataFrame, title: list[str], span: tuple[float, float] | None
) -> alt.LayerChart | alt.FacetChart:
    """Draw one forecast window: median, 50% and 90% bands of three arms, the actual, the span."""
    base = alt.Chart(data.to_pandas())
    colour = alt.Color(
        "arm:N", scale=scale(), legend=alt.Legend(title=None, orient="top", labelLimit=400)
    )
    band90 = base.mark_area(opacity=0.10).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q", y="q05:Q", y2="q95:Q", color=colour
    )
    band50 = base.mark_area(opacity=0.20).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q", y="q25:Q", y2="q75:Q", color=alt.Color("arm:N", scale=scale(), legend=None)
    )
    median = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hours after the forecast origin"),
        y=alt.Y("q50:Q", title="Demand (% of effective capacity)", scale=alt.Scale(zero=False)),
        color=alt.Color("arm:N", scale=scale(), legend=None),
    )
    actual = (
        alt.Chart(data.filter(pl.col("arm") == ARM_LABELS["a"]).to_pandas())
        .mark_line(color="black", strokeWidth=1.5)
        .encode(x="hour:Q", y="actual:Q")  # ty: ignore[unresolved-attribute]
    )
    layers = [band90, band50, median, actual]
    if span is not None:
        rect = (
            alt.Chart(pl.DataFrame({"a": [span[0]], "b": [span[1]]}).to_pandas())
            .mark_rect(opacity=0.10, color="grey")
            .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
        )
        layers.insert(0, rect)
    return alt.layer(*layers).properties(
        width=WIDTH - 60, height=190, title=alt.TitleParams(title, anchor="start", fontSize=11)
    )


def origin_for(*, event: dict, origins: list[datetime]) -> datetime | None:
    """Return the latest scored origin at least 12 hours before the event, in the event's fold."""
    for first, last in zip(s392.FOLD_STARTS, s392.FOLD_ENDS, strict=True):
        if first <= event["start"] < last:
            usable = [
                o
                for o in origins
                if first <= o <= event["start"] - timedelta(hours=12) and o < last
            ]
            return max(usable) if usable else None
    return None


def pick_events(*, events: pl.DataFrame, origins: list[datetime]) -> list[tuple[dict, datetime]]:
    """Choose up to three labelled events at the signed-MW six and three at the MVA-only eight.

    An event qualifies if a scored origin lies in its fold at least 12 hours before it starts.
    Within a group, the deepest event of each Primary comes first.
    """
    chosen: list[tuple[dict, datetime]] = []
    for group in (SIGNED_MW_LABELS, MVA_ONLY_LABELS):
        part = events.filter(
            pl.col("label").is_in(sorted(group)),
            pl.col("kind").is_in(["labelled", "labelled low-confidence"]),
            (pl.col("end") - pl.col("start")) >= pl.duration(hours=4),
            (pl.col("end") - pl.col("start")) <= pl.duration(days=7),
        ).sort("depth_abs", descending=True)
        seen: set[str] = set()
        count = 0
        for event in part.to_dicts():
            origin = origin_for(event=event, origins=origins)
            if origin is None or event["label"] in seen or count == 3:
                continue
            seen.add(event["label"])
            count += 1
            chosen.append((event, origin))
    return chosen


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-events", action="store_true", help="skip the contrast figures")
    args = parser.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.only_events:
        frame = contrasts()
        with pl.Config(tbl_rows=400, tbl_cols=20, tbl_width_chars=220, float_precision=3):
            print(frame)
        contrast_figure(frame=frame)
        sweep_figure()
        level_figure()
    series = adapter.series_table(labels=adapter.PRIMARY_LABELS)
    actuals, _, _ = s392.load_actuals(source="observed", series=series)
    _, events = s392.event_tables(series=series)
    origins = common.all_origins()
    # Example forecasts without a labelled event: D5 in winter and D3 in summer.
    for name, label, month in (
        ("figure_4_day_ahead_winter", "D5", (2026, 1, 15)),
        ("figure_5_fourteen_day_summer", "D3", (2026, 6, 15)),
    ):
        origin = min(
            origins,
            key=lambda o: abs((o - datetime(*month, 12, tzinfo=UTC)).total_seconds()),
        )
        days = 1 if "day_ahead" in name else 14
        data = forecast_window(
            label=label, origin=origin, days=days, actuals=actuals, series=series
        )
        series_chart(
            data=data,
            title=[
                f"{label}, forecast issued {origin:%B %Y}: median, 50% and 90% bands, and the",
                f"actual (black); {days} day{'s' if days > 1 else ''} ahead.",
            ],
            span=None,
        ).save(FIGURE_DIR / f"{name}.svg")
    for number, (event, origin) in enumerate(pick_events(events=events, origins=origins), start=6):
        start, end = event["start"], event["end"]
        data = forecast_window(
            label=event["label"], origin=origin, days=14, actuals=actuals, series=series
        )
        span = ((start - origin).total_seconds() / 3600.0, (end - origin).total_seconds() / 3600.0)
        group = "signed-MW six" if event["label"] in SIGNED_MW_LABELS else "MVA-only eight"
        series_chart(
            data=data,
            title=[
                f"{event['label']} ({GROUP_DESCRIPTIONS[group]}), forecast issued {origin:%B %Y}:",
                "median, 50% and 90% bands and the actual; labelled event shaded grey.",
            ],
            span=span,
        ).save(FIGURE_DIR / f"figure_{number}_event_{event['label']}.svg")


if __name__ == "__main__":
    main()
