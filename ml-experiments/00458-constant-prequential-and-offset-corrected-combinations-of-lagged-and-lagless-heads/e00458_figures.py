"""Experiment 458: draw the figures from the tables of `e00458_report.py` (no fit).

```bash
DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00458-*/e00458_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = (
    "00458-constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads"
)
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FIGURE_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
LABELS: Final[dict[str, str]] = {
    "b-A0": "(b) band weight, A0-inj", "b-S1": "(b) band weight, S1",
    "c-A0": "(c) prequential, A0-inj", "c-S1": "(c) prequential, S1",
    "d": "(d) lagless + offset", "lagless": "lagless alone",
    "lagged-A0": "A0-inj alone",
}  # fmt: skip
SITES: Final[list[str]] = [
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D15", "D16", "D18", "D19",
    "D21", "D22",
]  # fmt: skip
WIDTH: Final[int] = 430


def contrast_figure() -> None:
    """Draw each arm minus the 50/50 mixture of its own lagged head, by group and row set."""
    parts = [pl.read_csv(OUT / f) for f in ("contrasts_vs_half_a0.csv", "contrasts_vs_half_s1.csv")]
    data = (
        pl.concat(parts, how="vertical_relaxed")
        .filter(
            pl.col("arm").is_in(list(LABELS))
            & (pl.col("group").is_in(["signed_six", "mva_eight"]))
            & (
                ((pl.col("rowset") == "event") & (pl.col("col") == "crps_nra1.0"))
                | ((pl.col("rowset") == "nolabel") & (pl.col("col") == "crps"))
            )
        )
        .unique(["arm", "group", "rowset"])
    )
    data = data.with_columns(
        label=pl.col("arm").replace(LABELS),
        panel=pl.col("group").replace(
            {"signed_six": "Signed-MW six", "mva_eight": "MVA-only eight"}
        )
        + pl.lit(", ")
        + pl.col("rowset").replace(
            {
                "event": "inside real events (NRA-corrected)",
                "nolabel": "rows with no labelled event",
            }
        ),
    )
    y = alt.Y("label:N", sort=list(LABELS.values()), title=None)
    base = alt.Chart()
    thin = base.mark_rule(strokeWidth=1, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "low_bonf:Q",
            title="Arm minus 50/50 mixture (points of capacity)",
            scale=alt.Scale(domain=[-1.0, 1.5], clamp=True),
        ),
        x2="high_bonf:Q",
    )
    thick = base.mark_rule(strokeWidth=4, color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="low95:Q", x2="high95:Q"
    )  # ty: ignore[unresolved-attribute]
    dots = base.mark_point(filled=True, size=60, color=theme.BRAND_ORANGE).encode(y=y, x="diff:Q")  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(color=theme.GREY_2, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    chart = (
        alt.layer(zero, thin, thick, dots, data=data.to_pandas())
        .properties(width=WIDTH, height=150)
        .facet(
            row=alt.Row("panel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Each combination minus the 50/50 mixture of its lagged and lagless heads",
                    "Left of the dashed line is better than 50/50. Dot: mean of month means.",
                    "Thick line: 95% interval. Thin line: Bonferroni interval for 20 contrasts.",
                    "Both resample the scored months (4, or 3 where a month has no real event).",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=10,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_1_contrasts.svg")


def weights_figure() -> None:
    """Draw the fitted lagged weight of arm (b) by lead band and month at the signed-MW six."""
    w = (
        pl.read_csv(OUT / "weights.csv")
        .filter(
            pl.col("rule").is_in(["b-A0", "b-S1"]) & (pl.col("seed") == 0) & (pl.col("group") < 2)
        )
        .with_columns(
            band_name=pl.col("band").replace_strict(
                {0: "0-12h", 1: "12-36h", 2: "36-72h", 3: "days 4-5", 4: "days 6-10"},
                return_dtype=pl.String,
            ),
            group_name=pl.col("group").replace_strict(
                {0: "Signed-MW six", 1: "MVA-only eight"}, return_dtype=pl.String
            ),
            head=pl.col("rule").replace({"b-A0": "A0-inj", "b-S1": "S1"}),
        )
    )
    line = (
        alt.Chart(w.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "band_name:N",
                sort=["0-12h", "12-36h", "36-72h", "days 4-5", "days 6-10"],
                title="Lead band",
            ),
            y=alt.Y(
                "value:Q",
                title="Fitted weight on the lagged head",
                scale=alt.Scale(domain=[0.2, 1.0]),
            ),
            color=alt.Color("month:N", title="Scored month (weights fitted on earlier months)"),
            strokeDash=alt.StrokeDash("head:N", title=None),
        )
    )
    chart = (
        line.properties(width=WIDTH, height=170)
        .facet(row=alt.Row("group_name:N", title=None))
        .properties(
            title=alt.TitleParams(
                [
                    "The fitted weight swings by 0.2 or more between months in the same lead band",
                    "Weights fall with lead time; 0.5 is the 50/50 mixture.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=10,
            )
        )
    )
    chart.save(FIGURE_DIR / "figure_2_fitted_weights.svg")


def series_figure(
    *, name: str, primary: int, origin: int, month: str, horizon_steps: int, title: str
) -> None:
    """Draw one forecast origin: actual, median, and 50% and 90% bands of B-mix and arm (b)."""
    frames = []
    for arm, label in (
        ("B-mix", "B-mix (50/50)"),
        ("b-A0", "(b) weight per lead band"),
        ("d", "(d) lagless plus offset"),
    ):
        f = (
            pl.read_parquet(OUT / "quantiles" / arm / f"{month}.parquet")
            .filter(
                (pl.col("primary") == primary)
                & (pl.col("origin") == origin)
                & ((pl.col("target") - pl.col("origin")) <= horizon_steps)
            )
            .with_columns(arm=pl.lit(label), hour=(pl.col("target") - pl.col("origin")) / 2.0)
        )
        frames.append(f)
    data = pl.concat(frames)
    actual = data.filter(pl.col("arm") == "B-mix (50/50)").select(
        "hour", "actual", "tgt_real", "tgt_post"
    )
    colours = {
        "B-mix (50/50)": theme.DATA_BLUE,
        "(b) weight per lead band": theme.BRAND_ORANGE,
        "(d) lagless plus offset": theme.DATA_GREEN,
    }
    scale = alt.Scale(domain=list(colours), range=list(colours.values()))
    base = alt.Chart(data.to_pandas())
    band90 = base.mark_area(opacity=0.12).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q",
        y="q05:Q",
        y2="q95:Q",
        color=alt.Color("arm:N", scale=scale, legend=alt.Legend(title=None, orient="top")),
    )
    band50 = base.mark_area(opacity=0.25).encode(  # ty: ignore[unresolved-attribute]
        x="hour:Q", y="q25:Q", y2="q75:Q", color=alt.Color("arm:N", scale=scale, legend=None)
    )
    med = base.mark_line().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hours after the forecast origin"),
        y=alt.Y(
            "q50:Q", title="Demand (fraction of effective capacity)", scale=alt.Scale(zero=False)
        ),
        color=alt.Color("arm:N", scale=scale, legend=None),
    )
    act = (
        alt.Chart(actual.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(color="black", strokeWidth=1.5)
        .encode(x="hour:Q", y="actual:Q")
    )
    ev = actual.filter(pl.col("tgt_real"))
    layers = [band90, band50, med, act]
    if ev.height:
        span = (
            alt.Chart(pl.DataFrame({"a": [ev["hour"].min()], "b": [ev["hour"].max()]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.08, color="grey")
            .encode(x="a:Q", x2="b:Q")
        )
        layers.insert(0, span)
    alt.layer(*layers).properties(
        width=WIDTH, height=220, title=alt.TitleParams(title, anchor="start", fontSize=12)
    ).save(FIGURE_DIR / f"{name}.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    contrast_figure()
    weights_figure()
    events = pl.read_parquet(OUT / "events.parquet").filter(
        (pl.col("arm") == "B-mix") & (pl.col("group") == 0) & (pl.col("seed") == 0)
    )
    # The labelled event with most rows at the signed-MW six, in a month after the warm-up.
    top = (
        events.filter(pl.col("month") != "2025-10")
        .group_by("month", "cluster")
        .agg(pl.len().alias("n"), pl.col("primary").first(), pl.col("origin").min().alias("o"))
        .sort("n", descending=True)[0]
    )
    month, primary = top["month"][0], int(top["primary"][0])
    sel = events.filter((pl.col("month") == month) & (pl.col("cluster") == top["cluster"][0]))
    # Day-ahead: an origin whose first 24 hours hold event rows; 10-day: the same origin.
    first = (
        sel.filter((pl.col("target") - pl.col("origin")) <= 48)
        .group_by("origin")
        .len()
        .sort("len", descending=True)["origin"][0]
    )
    label = SITES[primary]
    series_figure(
        name="figure_3_day_ahead_event",
        primary=primary,
        origin=int(first),
        month=month,
        horizon_steps=48,
        title=f"Day ahead, {label}, {month}: real labelled event rows shaded grey",
    )
    series_figure(
        name="figure_4_ten_day_event",
        primary=primary,
        origin=int(first),
        month=month,
        horizon_steps=480,
        title=f"Ten days, {label}, {month}, same origin",
    )
    q = pl.read_parquet(OUT / "quantiles" / "B-mix" / "2026-06.parquet").filter(
        pl.col("primary") == 3
    )
    flagged = q.filter(pl.col("tgt_real") | pl.col("tgt_post"))
    clean_origin = int(np.sort(q["origin"].unique().to_numpy())[0])
    if flagged.height:
        ok = [
            o
            for o in np.sort(q["origin"].unique().to_numpy())
            if q.filter(
                (pl.col("origin") == o)
                & (pl.col("target") - pl.col("origin") <= 480)
                & (pl.col("tgt_real") | pl.col("tgt_post"))
            ).is_empty()
        ]
        clean_origin = int(ok[len(ok) // 2]) if ok else clean_origin
    series_figure(
        name="figure_5_day_ahead_no_label",
        primary=3,
        origin=clean_origin,
        month="2026-06",
        horizon_steps=48,
        title=f"Day ahead, {SITES[3]}, 2026-06: no labelled event in 10 days",
    )
    series_figure(
        name="figure_6_ten_day_no_label",
        primary=3,
        origin=clean_origin,
        month="2026-06",
        horizon_steps=480,
        title=f"Ten days, {SITES[3]}, 2026-06, same origin",
    )


if __name__ == "__main__":
    main()
