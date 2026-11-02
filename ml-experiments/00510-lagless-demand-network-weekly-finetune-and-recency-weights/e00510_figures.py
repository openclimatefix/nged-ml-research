# ruff: noqa: E501
"""Draw the figures of experiment 510 from the tables the scorer, slices, and stress scripts stored.

Figures (SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline`):

1. `figure_1_headline.svg`: each arm's CRPS minus K2-S1 at the eight and the six, Bonferroni interval;
2. `figure_2_week_of_month.svg`: L-W minus L-M and L-Ww minus L-W by week of the month;
3. `figure_3_christmas_january.svg`: L-W and L-Ww minus K2-S1 in Christmas week and in January;
4. `figure_4_stress_leak.svg`: the injected-shift leak of each arm in each fold;
5. `figure_5_example_*.svg`: a day-ahead and a 10-day forecast of L-W, L-Ww, and K2-S1;
6. `figure_6_event_*.svg`: day-ahead medians around six labelled events.

```bash
source <paths>; uv run --no-sync python ml-experiments/00510-*/e00510_figures.py
```
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00510_common as c
import e00510_score as score
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00510-lagless-demand-network-weekly-finetune-and-recency-weights"
)
SCORES: Final[Path] = c.WORK_DIR / "scores"
WIDTH: Final[int] = 600
COLOURS: Final[dict[str, str]] = {
    "L-M": theme.DATA_SKY, "L-W": theme.DATA_BLUE, "L-Ww": theme.DATA_GREEN,
    "L-Ww-D": theme.DATA_PURPLE, "L-Ww-O": theme.BRAND_ORANGE, "K2-S1": "#000000", "A-pool": "#808080",
}  # fmt: skip
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"


def _scale(*, arms: list[str]) -> alt.Scale:
    return alt.Scale(domain=arms, range=[COLOURS[a] for a in arms])


def figure_headline() -> None:
    """Draw each arm's CRPS minus K2-S1 with its Bonferroni interval."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("metric") == "crps")
        & pl.col("contrast").str.ends_with(" - K2-S1")
        & pl.col("slice").is_in(list(GROUPS))
    )
    t = t.with_columns(
        arm=pl.col("contrast").str.replace(" - K2-S1", ""), group=pl.col("slice").replace(GROUPS)
    )
    arms = ["L-M", "L-W", "L-Ww", "L-Ww-D", "L-Ww-O"]
    panels = []
    for label in GROUPS.values():
        part = t.filter(pl.col("group") == label).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=arms, title=None))  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low_bonferroni:Q", title="CRPS minus K2-S1 (points of capacity; lower is better)"),
            x2="high_bonferroni:Q", color=alt.Color("arm:N", scale=_scale(arms=arms), legend=None),
        )  # fmt: skip
        dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x="difference:Q", color=alt.Color("arm:N", scale=_scale(arms=arms), legend=None)
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=130, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Lag-free arms against K2-S1 over five folds, rows with no labelled event.",
                "Dot: difference; bar: Bonferroni interval (56 tests).",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def _slice_chart(
    *, names: list[str], kinds: list[str], title: list[str], file: str, order: list[str]
) -> None:
    t = pl.read_csv(SCORES / "slices.csv").filter(
        pl.col("slice").is_in(names) & pl.col("kind").is_in(kinds)
    )
    panels = []
    for g, label in (("eight", GROUPS["eight"]), ("six", GROUPS["six_clean"])):
        part = t.filter(pl.col("group") == g).to_pandas()
        kind_order = kinds
        base = alt.Chart(part).encode(
            y=alt.Y("slice:N", sort=order, title=None),
            yOffset=alt.YOffset("kind:N", sort=kind_order),
        )  # ty: ignore[unresolved-attribute]
        colour = alt.Color(
            "kind:N", sort=kind_order, legend=alt.Legend(orient="top", title=None, columns=1)
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="CRPS difference (points of capacity)"),
            x2="high:Q",
            color=colour,
        )
        dot = base.mark_point(filled=True, size=50).encode(x="value:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=40 * len(order), title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start")).save(
        DOCS_DIR / file
    )


def figure_week_of_month() -> None:
    """Draw the weekly-over-monthly and recency gains by week of the month."""
    _slice_chart(
        names=[f"week_{w}" for w in range(1, 6)], kinds=["L-W - L-M", "L-Ww - L-W"],
        order=[f"week_{w}" for w in range(1, 6)],
        title=["Gain from a weekly refit (L-W minus L-M) and from the 90-day weight", "(L-Ww minus L-W) by scored week of the month; bars: 95% origin-day interval."],
        file="figure_2_week_of_month.svg",
    )  # fmt: skip


def figure_christmas_january() -> None:
    """Draw L-W and L-Ww minus K2-S1 in Christmas week and January."""
    _slice_chart(
        names=["christmas", "jan_1_14", "jan_15_24", "jan_25_31"], kinds=["L-W - K2-S1", "L-Ww - K2-S1"],
        order=["christmas", "jan_1_14", "jan_15_24", "jan_25_31"],
        title=["Lag-free arms minus K2-S1 (CRPS, points of capacity) in Christmas week", "2025 and in three parts of January 2026; bars: 95% origin-day interval."],
        file="figure_3_christmas_january.svg",
    )  # fmt: skip


def figure_stress() -> None:
    """Draw the injected-shift leak by arm and fold."""
    t = pl.read_csv(SCORES / "stress_leak_by_fold.csv").to_pandas()
    arms = ["L-W", "L-Ww", "L-Ww-D", "L-Ww-O"]
    base = alt.Chart(t).encode(y=alt.Y("arm:N", sort=arms, title=None))  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("all:Q", title="Leak: shift of the median forecast, points of capacity"),
            color=alt.Color("month:N", legend=alt.Legend(orient="top", title="Fold", columns=5)),
        )
    ).properties(
        width=WIDTH, height=140,
        title=alt.TitleParams(["How much of a 3-standard-deviation shift in the 7 days before a refit", "reaches the next week's median forecast (one seed, both signs averaged)."], anchor="start"),
    )  # fmt: skip
    chart.save(DOCS_DIR / "figure_4_stress_leak.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 10-day forecast of L-W, L-Ww, and K2-S1 against the actuals."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    frames = {}
    for arm in ("L-W", "L-Ww"):
        pooled = build470.pool_stored(
            folder=score.SEED_PREDS / arm, month=month, sites=sites, seeds=(0, 1)
        )
        frames[arm] = adapter.forecast_frame(
            stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
            series=series,
            unit="fraction",
        )
    pooled = build470.pool_stored(
        folder=build470.E425 / "seed_preds" / "A", month=month, sites=sites, seeds=(0, 1)
    )
    frames["A-pool"] = adapter.forecast_frame(
        stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
        series=series,
        unit="fraction",
    )
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(hour=9, tzinfo=UTC)
    for label, kind in EXAMPLES.items():
        one = {
            arm: f.filter(pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin)
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0).sort("valid_time")
            for arm, f in frames.items() if f is not None
        }  # fmt: skip
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == ids[label])
            .join(one["L-Ww"].select("valid_time", "day"), on="valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            main = one["L-Ww"].filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            band = alt.Chart(main).encode(x=x)
            layers = [
                band.mark_area(opacity=0.2, color=COLOURS["L-Ww"]).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),  # ty: ignore[unresolved-attribute]
                band.mark_area(opacity=0.35, color=COLOURS["L-Ww"]).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
            ]
            for arm in one:  # noqa: PLC0206
                sub = one[arm].filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
                layers.append(
                    alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                    .mark_line(color=COLOURS[arm], strokeWidth=2)
                    .encode(x=x, y="q50:Q")
                )  # ty: ignore[unresolved-attribute]
            act = actual.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            layers.append(
                alt.Chart(act)  # ty: ignore[unresolved-attribute]
                .mark_line(color="black", strokeWidth=1.5, strokeDash=[4, 3])
                .encode(x=x, y="actual:Q")
            )  # ty: ignore[unresolved-attribute]
            chart = alt.layer(*layers).properties(
                width=WIDTH, height=200,
                title=alt.TitleParams([f"{label} ({kind}), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: L-Ww 50% and 90%. Medians: L-W blue, L-Ww green, experiment 425 arm A (two seeds, with lags) grey. Dashed: actual."], anchor="start"),
            )  # fmt: skip
            chart.save(
                DOCS_DIR
                / f"figure_5_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg"
            )


def figure_events() -> None:
    """Draw day-ahead medians around six labelled events (three per metering group where available)."""
    names = c.sites()
    clean = c.clean_stack()
    events = pl.read_parquet(worlds402.cf.EVENTS)
    rows = pl.concat(
        [pl.read_parquet(p) for p in sorted(SCORES.glob("rows_*.parquet"))], how="diagonal_relaxed"
    )
    group_of = dict(rows.select("primary", "group").unique().iter_rows())
    chosen = []
    for grp in ("mva_eight", "signed_six"):
        pool = []
        for r in events.iter_rows(named=True):
            p = names.index(r["label"])
            onset, end = int(r["onset_idx"]), int(r["end_idx"])
            if group_of.get(p) != grp or end - onset < 96 or end - onset > 48 * 14:
                continue
            m = rows.filter(
                (pl.col("primary") == p)
                & (pl.col("target") >= onset - 96)
                & (pl.col("target") <= end + 96)
            )
            if m.height > 300:
                pool.append((p, onset, end, grp))
        chosen += pool[:: max(1, len(pool) // 3)][:3]
    start = common.GRID_START
    for i, (p, onset, end, grp) in enumerate(chosen, 1):
        sel = rows.filter(
            (pl.col("primary") == p)
            & (pl.col("target") >= onset - 192)
            & (pl.col("target") <= end + 192)
            & (pl.col("lead_hours") >= 24)
            & (pl.col("lead_hours") < 48)
        )
        if sel.height == 0:
            continue
        t = sel.sort("target").select(
            "target", **{a: pl.col(f"{a}__q50") * 100 for a in ("L-W", "L-Ww", "K2-S1")}
        )
        long = t.unpivot(index="target", variable_name="arm", value_name="median").with_columns(
            day=(pl.col("target") - onset) / 48.0
        )
        act = pl.DataFrame(
            {"target": t["target"], "actual": clean[p][t["target"].to_numpy()] * 100}
        ).with_columns(day=(pl.col("target") - onset) / 48.0)
        arms = ["L-W", "L-Ww", "K2-S1"]
        lines = (
            alt.Chart(long.to_pandas())
            .mark_line(strokeWidth=2)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("day:Q", title="Days since the event began"),
                y=alt.Y("median:Q", title="Share of capacity (%)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "arm:N",
                    scale=_scale(arms=arms),
                    legend=alt.Legend(orient="top", title=None, columns=3),
                ),
            )
        )
        actual_line = (
            alt.Chart(act.to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color="black", strokeDash=[4, 3])
            .encode(x="day:Q", y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        span = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - onset) / 48.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color="grey")
            .encode(x="a:Q", x2="b:Q")
        )  # ty: ignore[unresolved-attribute]
        date = (start + timedelta(minutes=30 * onset)).date()
        kind = "MVA-only" if grp == "mva_eight" else "signed-MW"
        alt.layer(span, lines, actual_line).properties(
            width=WIDTH, height=170,
            title=alt.TitleParams([f"Labelled event {i}: a {kind} Primary, event began {date}.", "Day-ahead medians (lead 24 to 48 h); grey: the labelled event; dashed: actual."], anchor="start"),
        ).save(DOCS_DIR / f"figure_6_event_{i}.svg")  # fmt: skip


def main() -> None:
    """Draw every figure."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    score.configure()
    figure_headline()
    figure_week_of_month()
    figure_christmas_january()
    figure_stress()
    figure_examples()
    figure_events()
    _ = np


if __name__ == "__main__":
    main()
