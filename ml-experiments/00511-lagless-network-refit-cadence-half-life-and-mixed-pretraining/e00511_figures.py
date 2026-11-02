# ruff: noqa: E501
"""Draw the figures of experiment 511 from the tables the scorer, slices, stale, and leak scripts stored.

Figures (SVG, in the experiment's docs folder; optimise each with `npx svgo@4 --multipass
--precision=1 --final-newline`):

1. `figure_1_headline.svg`: each refitted arm's CRPS minus L-Ww at the eight and the six, with
   the Bonferroni interval;
2. `figure_2_staleness.svg`: CRPS by the age of the model that scores each origin;
3. `figure_3_horizon.svg`: L-3d and L-1d minus L-Ww by horizon band;
4. `figure_4_stress_leak.svg`: the injected-shift leak of L-1d by block age and of L-Ww-30;
5. `figure_5_example_*.svg`: a day-ahead and a 10-day forecast of L-Ww and L-1d;
6. `figure_6_event_*.svg`: day-ahead medians around six labelled events.
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
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k
import e00511_score as score
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00511-lagless-network-refit-cadence-half-life-and-mixed-pretraining"
)
SCORES: Final[Path] = k.WORK_DIR / "scores"
WIDTH: Final[int] = 600
COLOURS: Final[dict[str, str]] = {
    "L-Ww": theme.DATA_GREEN, "L-3d": theme.DATA_SKY, "L-1d": theme.DATA_BLUE,
    "L-Ww-30": theme.BRAND_ORANGE, "L-Ww-60": theme.DATA_PURPLE, "L-mix": "#a0522d",
    "K2-S1": "#000000", "A-pool": "#808080",
}  # fmt: skip
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}
EXAMPLES: Final[dict[str, str]] = {"D10": "signed-MW Primary", "D4": "MVA-only Primary"}
EXAMPLE_ORIGIN: Final[str] = "2026-01-12"


def _scale(*, arms: list[str]) -> alt.Scale:
    return alt.Scale(domain=arms, range=[COLOURS[a] for a in arms])


def _interval_chart(
    *, t: pl.DataFrame, arms: list[str], x_title: str, title: list[str], file: str
) -> None:
    """Draw dots and Bonferroni bars per arm for the eight and the six."""
    panels = []
    for label in GROUPS.values():
        part = t.filter(pl.col("group") == label).to_pandas()
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=arms, title=None))  # ty: ignore[unresolved-attribute]
        colour = alt.Color("arm:N", scale=_scale(arms=arms), legend=None)
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low_bonferroni:Q", title=x_title), x2="high_bonferroni:Q", color=colour
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=70).encode(x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
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
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start")).save(
        DOCS_DIR / file
    )


def figure_headline() -> None:
    """Draw each refitted arm's CRPS minus L-Ww, then minus K2-S1, with Bonferroni intervals."""
    t = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("metric") == "crps") & pl.col("slice").is_in(list(GROUPS))
    )
    t = t.with_columns(group=pl.col("slice").replace(GROUPS))
    vs_ww = t.filter(pl.col("contrast").str.ends_with(" - L-Ww")).with_columns(
        arm=pl.col("contrast").str.replace(" - L-Ww", "")
    )
    _interval_chart(
        t=vs_ww, arms=["L-3d", "L-1d", "L-Ww-30", "L-Ww-60", "L-mix"],
        x_title="CRPS minus L-Ww (points of capacity; lower is better)",
        title=["Refitted arms against L-Ww over five folds, rows with no labelled event.", "Dot: difference; bar: Bonferroni interval (56 tests)."],
        file="figure_1_headline.svg",
    )  # fmt: skip
    vs_k2 = t.filter(pl.col("contrast").str.ends_with(" - K2-S1")).with_columns(
        arm=pl.col("contrast").str.replace(" - K2-S1", "")
    )
    _interval_chart(
        t=vs_k2, arms=["L-Ww", "L-3d", "L-1d", "L-Ww-30", "L-Ww-60"],
        x_title="CRPS minus K2-S1 (points of capacity; lower is better)",
        title=["Lag-free arms against K2-S1 (reads lagged power), rows with no labelled event.", "Dot: difference; bar: Bonferroni interval (56 tests)."],
        file="figure_1b_gap_to_k2s1.svg",
    )  # fmt: skip


def figure_staleness() -> None:
    """Draw CRPS against model age for each arm, eight and six."""
    t = pl.read_csv(SCORES / "staleness_curve.csv").with_columns(
        group=pl.col("group").replace({"eight": GROUPS["eight"], "six": GROUPS["six_clean"]})
    )
    arms = ["L-3d", "L-1d", "L-Ww-30", "L-Ww-60", "L-mix"]
    panels = []
    for label in GROUPS.values():
        part = t.filter(pl.col("group") == label).to_pandas()
        panels.append(
            alt.Chart(part).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("age:Q", title="Age of the scoring model (days since its cutoff)"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color("arm:N", scale=_scale(arms=arms), legend=alt.Legend(orient="top", title=None, columns=5)),
            ).properties(width=WIDTH, height=170, title=alt.TitleParams(label, anchor="start"))
        )  # fmt: skip
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS by the age of the model that scores each origin day.",
                "Same rows at every age: origins whose age-7 model exists in every fold.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_2_staleness.svg")


def figure_horizon() -> None:
    """Draw L-3d and L-1d minus L-Ww by horizon band."""
    t = pl.read_csv(SCORES / "slices.csv").filter(
        pl.col("kind").is_in(["L-3d - L-Ww", "L-1d - L-Ww"])
        & pl.col("slice").is_in(["day1", "days2_3", "days4_10"])
    )
    t = t.with_columns(
        arm=pl.col("kind").str.replace(" - L-Ww", ""),
        group=pl.col("group").replace({"eight": GROUPS["eight"], "six": GROUPS["six_clean"]}),
    )
    arms = ["L-3d", "L-1d"]
    panels = []
    for label in GROUPS.values():
        part = t.filter(pl.col("group") == label).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("slice:N", sort=["day1", "days2_3", "days4_10"], title=None),
            color=alt.Color(
                "arm:N",
                scale=_scale(arms=arms),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
        )  # ty: ignore[unresolved-attribute]
        bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="CRPS minus L-Ww (points of capacity)"),
            x2="high:Q",
            yOffset="arm:N",
        )
        dot = base.mark_point(filled=True, size=60).encode(x="value:Q", yOffset="arm:N")  # ty: ignore[unresolved-attribute]
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
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Gain over L-Ww by forecast horizon (day 1, days 2 to 3, days 4 to 10).",
                "95% origin-day block-bootstrap interval, no multiplicity correction.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_3_horizon.svg")


def figure_stress() -> None:
    """Draw the injected-shift leak: L-1d by block age against L-Ww-30 and experiment 510's L-Ww."""
    t = pl.read_csv(SCORES / "stress_leak.csv")
    ww = pl.read_csv(c.WORK_DIR / "scores" / "stress_leak.csv").filter(pl.col("arm") == "L-Ww")[
        "all"
    ][0]
    daily = t.filter(pl.col("arm") == "L-1d").to_pandas()
    line = alt.Chart(daily).mark_line(point=True, color=COLOURS["L-1d"]).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("block_age_days:Q", title="Days between the end of the injected block and the refit"),
        y=alt.Y("all:Q", title="Leak (points of capacity)", scale=alt.Scale(zero=True)),
    )  # fmt: skip
    marks = pl.DataFrame(
        {
            "y": [ww, float(t.filter(pl.col("arm") == "L-Ww-30")["all"][0])],
            "arm": ["L-Ww (experiment 510, weekly)", "L-Ww-30 (weekly)"],
        }
    ).to_pandas()
    rules = (
        alt.Chart(marks)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3])
        .encode(y="y:Q", color=alt.Color("arm:N", legend=alt.Legend(orient="top", title=None)))
    )  # ty: ignore[unresolved-attribute]
    (line + rules).properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            [
                "Injected-shift leak into the next forecasts (mean of five folds, both signs, one seed).",
                "L-1d: next-day window by age of the block; dashed: the weekly arms' mean over the next week.",
            ],
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_4_stress_leak.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 10-day forecast of L-W, L-Ww, and K2-S1 against the actuals."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    month = EXAMPLE_ORIGIN[:7]
    frames = {}
    for arm in ("L-Ww", "L-1d"):
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
            .join(one["L-1d"].select("valid_time", "day"), on="valid_time")
        )
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("10 days", (0.0, 10.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            main = one["L-1d"].filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            band = alt.Chart(main).encode(x=x)
            layers = [
                band.mark_area(opacity=0.2, color=COLOURS["L-1d"]).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),  # ty: ignore[unresolved-attribute]
                band.mark_area(opacity=0.35, color=COLOURS["L-1d"]).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
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
                title=alt.TitleParams([f"{label} ({kind}), {horizon} forecast made on {EXAMPLE_ORIGIN}, widened 1.15.", "Bands: L-1d 50% and 90%. Medians: L-Ww green, L-1d blue, experiment 425 arm A (two seeds, with lags) grey. Dashed: actual."], anchor="start"),
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
            "target", **{a: pl.col(f"{a}__q50") * 100 for a in ("L-Ww", "L-1d", "K2-S1")}
        )
        long = t.unpivot(index="target", variable_name="arm", value_name="median").with_columns(
            day=(pl.col("target") - onset) / 48.0
        )
        act = pl.DataFrame(
            {"target": t["target"], "actual": clean[p][t["target"].to_numpy()] * 100}
        ).with_columns(day=(pl.col("target") - onset) / 48.0)
        arms = ["L-Ww", "L-1d", "K2-S1"]
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
    figure_staleness()
    figure_horizon()
    figure_stress()
    figure_examples()
    figure_events()


if __name__ == "__main__":
    main()
