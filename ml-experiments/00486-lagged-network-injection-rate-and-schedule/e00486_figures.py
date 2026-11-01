"""Draw the figures of experiment 486 from the stored scores and forecasts (no fit).

- Figure 1: each new arm minus A0-inj, and minus B-mix, in points of capacity, on rows with no
  labelled event and on injected harm rows at the signed-MW six.
- Figure 2: the median forecast's shift as a share of a 5%-of-capacity step in its lags, by number
  of lags stepped (written once the sweep has run).
- Figure 4: six forecasts before real labelled events, three at each metering group.
- Figure 3: a day-ahead forecast and the full 10-day horizon of one forecast issued before a real
  labelled event at a signed-MW Primary, for A0-inj, B-mix, and the arm with the best score on rows
  with no labelled event.

```bash
PYTHONPATH=<as e00486_fit.py shows> uv run --no-sync python ml-experiments/00486-*/e00486_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00404_report as report  # ty: ignore[unresolved-import]
import e00486_fit as fit486  # ty: ignore[unresolved-import]
import e00486_score as score486  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00486-lagged-network-injection-rate-and-schedule"
)
WORK: Final[Path] = fit486.WORK_DIR
WIDTH: Final[int] = 680
ARMS: Final[list[str]] = ["R05", "R50", "Cur-down", "Cur-up"]
COLOURS: Final[list[str]] = ["#FF4901", "#306BFF", "#17E58F", "#B701FF"]
ROWS: Final[dict[str, str]] = {
    "clean_real": "Rows with no labelled event",
    "harm_inj": "Injected harm rows",
}


def figure_contrasts() -> None:
    """Figure 1: arm minus A0-inj and arm minus B-mix with month-block intervals."""
    parts = []
    for reference in ("A0-inj", "B-mix"):
        frame = pl.read_csv(WORK / f"pooled_summary_vs_{reference}.csv").filter(
            (pl.col("group") == "signed_six")
            & pl.col("cls").is_in(list(ROWS))
            & pl.col("arm").is_in(ARMS)
        )
        parts.append(frame.with_columns(reference=pl.lit(f"minus {reference}")))
    data = pl.concat(parts).with_columns(rows=pl.col("cls").replace(ROWS))
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=ARMS, title=None),
        color=alt.Color("arm:N", scale=alt.Scale(domain=ARMS, range=COLOURS), legend=None),
    )
    chart = (
        (
            base.mark_rule().encode(
                x=alt.X(
                    "low:Q", title="Arm minus reference (points of capacity; negative is better)"
                ),
                x2="high:Q",
            )
            + base.mark_point(filled=True, size=70).encode(x="minus_ref:Q")
            + alt.Chart(pl.DataFrame({"z": [0.0]}))
            .mark_rule(color="#888", strokeDash=[4, 3])
            .encode(x="z:Q")
        )
        .properties(width=WIDTH - 150, height=90)
        .facet(
            row=alt.Row("rows:N", title=None, sort=list(ROWS.values())),
            column=alt.Column("reference:N", title=None),
        )
        .properties(
            title=alt.TitleParams(
                "Each new arm minus A0-inj and minus B-mix at the signed-MW six",
                subtitle=[
                    "Dot: pooled mean over 5 months. Line: 95% interval, resampling whole months.",
                    "Dashed line: same score as the reference. Seven cells (June at 3 seeds).",
                ],
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_contrasts.svg")


def figure_sweep() -> None:
    """Figure 2: median shift as a share of a 5% step, by arm and number of lags stepped."""
    frames = [
        pl.read_parquet(p).with_columns(month=pl.lit(p.name[:7]))
        for p in sorted((WORK / "sweep").glob("*.parquet"))
    ]
    data = (
        pl.concat(frames)
        .filter(~pl.col("arm").str.ends_with("masked") & (pl.col("lead_band") == 0))
        .group_by("arm", "k", "month")
        .agg(pl.col("shift").mean())
        .group_by("arm", "k")
        .agg(pl.col("shift").mean().alias("share"))
        .with_columns(lags=pl.col("k").cast(pl.Utf8))
    )
    order = ["A0-inj", *ARMS]
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "lags:N",
                title="Lags inside the 5%-of-capacity step (1, 2, or 3)",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("arm:N", sort=order),
            y=alt.Y("share:Q", title="Median shift as a share of the step (smaller is better)"),
            color=alt.Color(
                "arm:N",
                sort=order,
                scale=alt.Scale(domain=order, range=["#888888", *COLOURS]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=220,
            title=alt.TitleParams(
                "How far the median follows a step in the lags",
                subtitle=[
                    "Mean over both signs, leads under 48 h, rows with no labelled event,",
                    "mean of the 5 months. The lagless head follows 0.",
                ],
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_sweep.svg")


def figure_examples(*, best: str) -> None:
    """Figure 3: day-ahead and 10-day forecast before a real event, three arms."""
    score486.point_at_486()
    arms = ["A0-inj", "B-mix", best]
    frames = {a: report.load(arm=a, month="2026-06", seed=0, worlds=[-1]) for a in arms}
    ref = frames["A0-inj"].filter(pl.col("group") == "signed_six")
    inside = ref.filter(pl.col("tgt_real"))
    origin = int(inside.group_by("origin").len().sort("len", descending=True)["origin"][0])
    primary = int(inside.filter(pl.col("origin") == origin)["primary"][0])
    parts = []
    for arm, frame in frames.items():
        one = frame.filter((pl.col("origin") == origin) & (pl.col("primary") == primary))
        parts.append(
            one.select(
                "target", "lead_hours", "actual", "q05", "q50", "q95", "tgt_real"
            ).with_columns(
                arm=pl.lit(arm),
                days=pl.col("lead_hours") / 24.0,
            )
        )
    data = pl.concat(parts).with_columns(
        actual=pl.col("actual") * 100,
        q05=pl.col("q05") * 100,
        q50=pl.col("q50") * 100,
        q95=pl.col("q95") * 100,
    )
    colours = {"A0-inj": "#888888", "B-mix": "#306BFF", best: "#FF4901"}
    layers = []
    for limit, name, height in ((2.0, "Day ahead", 200), (10.0, "Full 10-day horizon", 200)):
        part = data.filter(pl.col("days") <= limit)
        line = alt.Chart(part).encode(
            x=alt.X("days:Q", title="Days after the forecast was issued"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        band = line.mark_area(opacity=0.12).encode(
            y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"
        )
        median = line.mark_line().encode(y="q50:Q")
        actual = (
            alt.Chart(part)
            .mark_line(color="black", strokeDash=[2, 2])
            .encode(x="days:Q", y="actual:Q", detail="arm:N")
        )
        layers.append(
            (band + median + actual).properties(width=WIDTH - 90, height=height, title=name)
        )
    chart = alt.vconcat(*layers).properties(
        title=alt.TitleParams(
            "Forecasts issued before a real labelled event at a signed-MW Primary",
            subtitle=[
                "Line: median. Shade: 5% to 95% band. Black dashes: actual power.",
                "June 2026 forecast, seed 0, real data; the same forecast origin in all arms.",
            ],
        )
    )
    chart.save(DOCS_DIR / "figure_3_examples.svg")
    print("example Primary index", primary, "origin", origin)


def figure_events(*, best: str) -> None:
    """Figure 4: six forecasts issued before real labelled events, three at each metering group."""
    score486.point_at_486()
    arms = ["A0-inj", "B-mix", best]
    colours = {"A0-inj": "#888888", "B-mix": "#306BFF", best: "#FF4901"}
    panels = []
    for group, label in (("signed_six", "signed-MW Primary"), ("mva_eight", "MVA-only Primary")):
        picked: list[tuple[str, int, int]] = []
        for month in ("2026-06", "2026-03", "2026-01", "2025-12", "2025-10"):
            frames = {a: report.load(arm=a, month=month, seed=0, worlds=[-1]) for a in arms}
            ref = frames["A0-inj"].filter((pl.col("group") == group) & pl.col("tgt_real"))
            counts = ref.group_by("primary", "origin").len().sort("len", descending=True)
            for primary, origin, _ in counts.iter_rows():
                if len(picked) < 3 and primary not in [p[1] for p in picked]:
                    picked.append((month, primary, origin))
                    parts = []
                    for arm, frame in frames.items():
                        one = frame.filter(
                            (pl.col("origin") == origin) & (pl.col("primary") == primary)
                        )
                        parts.append(
                            one.with_columns(arm=pl.lit(arm), days=pl.col("lead_hours") / 24.0)
                        )
                    data = pl.concat(parts).with_columns(
                        pl.col("actual", "q05", "q50", "q95") * 100
                    )
                    event = data.filter((pl.col("arm") == "A0-inj") & pl.col("tgt_real"))
                    span = pl.DataFrame(
                        {"start": [event["days"].min()], "end": [event["days"].max()]}
                    )
                    line = alt.Chart(data).encode(
                        x=alt.X("days:Q", title="Days after the forecast was issued"),
                        color=alt.Color(
                            "arm:N",
                            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                    )
                    shade = (
                        alt.Chart(span)
                        .mark_rect(color="#FC9700", opacity=0.15)
                        .encode(x="start:Q", x2="end:Q")
                    )
                    panel = (
                        shade
                        + line.mark_area(opacity=0.1).encode(
                            y=alt.Y("q05:Q", title="% of capacity"), y2="q95:Q"
                        )
                        + line.mark_line().encode(y="q50:Q")
                        + alt.Chart(data)
                        .mark_line(color="black", strokeDash=[2, 2])
                        .encode(x="days:Q", y="actual:Q", detail="arm:N")
                    ).properties(
                        width=WIDTH - 90,
                        height=130,
                        title=f"{label} {len(picked)} of 3, {month} forecast",
                    )
                    panels.append(panel)
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Forecasts issued before a real labelled event",
            subtitle=[
                "Orange band: the real event (rows whose target lies in a labelled event).",
                "Line: median. Shade: 5% to 95% band. Black dashes: actual power. Seed 0.",
            ],
        )
    )
    chart.save(DOCS_DIR / "figure_4_events.svg")


def best_arm() -> str:
    """Return the new arm with the lowest score on rows with no labelled event at the six."""
    frame = pl.read_csv(WORK / "pooled_summary_vs_A0-inj.csv").filter(
        (pl.col("group") == "signed_six") & (pl.col("cls") == "clean_real")
        & pl.col("arm").is_in(ARMS)
    )  # fmt: skip
    return frame.sort("absolute")["arm"][0]


def main() -> None:
    """Draw the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    if list((WORK / "sweep").glob("*.parquet")):
        figure_sweep()
    best = best_arm()
    figure_examples(best=best)
    figure_events(best=best)


if __name__ == "__main__":
    main()
