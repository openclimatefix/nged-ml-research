# ruff: noqa: E501, PERF401
"""Experiment 500: figures for the write-up (680 pixels wide, OCF theme).

Reads the tables written by `e00500_conformal.py score`, `e00500_posthoc.py`, and the arm frames.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00500_conformal as conformal  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00500-rolling-conformal-recalibration-of-the-demand-tails"
)
WIDTH: Final[int] = 680
ARM_ORDER: Final[list[str]] = [
    "raw", "w1.15", "P", "two-constant", "RC7", "RC14", "RC28", "RC14+w1.05", "RC28-bias",
]  # fmt: skip
ARM_COLOURS: Final[list[str]] = [
    theme.GREY_3, theme.DATA_SKY, theme.DATA_BLUE, theme.DATA_PURPLE, theme.BRAND_ORANGE_LIGHT,
    theme.BRAND_ORANGE, theme.BLACK_1, theme.DATA_GREEN_LIGHT, theme.DATA_GREEN,
]  # fmt: skip
TARGETS: Final[dict[str, tuple[float, float]]] = {
    "q01": (0.005, 0.02), "q05": (0.03, 0.07), "q95": (0.93, 0.97), "q99": (0.98, 0.995),
}  # fmt: skip
LEVEL_BONFERRONI: Final[float] = 100.0 * (1.0 - 0.05 / conformal.BONFERRONI_COUNT)


def colour() -> alt.Color:
    """Return the arm colour encoding."""
    return alt.Color(
        "arm:N", sort=ARM_ORDER, scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLOURS),
        legend=alt.Legend(orient="top", columns=5, title=None),
    )  # fmt: skip


def figure_coverage() -> None:
    """Coverage of each arm at the six with 95% intervals, by tail, against the target range."""
    c = pl.read_csv(conformal.WORK / "coverage_intervals.csv").filter(
        (pl.col("rowset") == "five") & (pl.col("group") == "signed_six")
        & pl.col("arm").is_in(ARM_ORDER)
    ).rename({"low_95.0": "low95", "high_95.0": "high95"}).with_columns(
        tail=pl.col("metric").str.slice(-3)
    )  # fmt: skip
    target = pl.DataFrame([{"tail": k, "lo": v[0], "hi": v[1]} for k, v in TARGETS.items()])
    panels = []
    for tail in TARGETS:
        part = c.filter(pl.col("tail") == tail)
        lo, hi = TARGETS[tail]
        pad = 0.01 if tail in ("q95", "q99") else 0.005
        scale = alt.Scale(
            domain=[min(lo, part["low95"].min()) - pad, max(hi, part["high95"].max()) + pad]  # ty: ignore[invalid-argument-type]
        )
        band = (
            alt.Chart(target.filter(pl.col("tail") == tail)).mark_rect(  # ty: ignore[unresolved-attribute]
                opacity=0.18, color=theme.DATA_GREEN
            ).encode(x=alt.X("lo:Q", scale=scale), x2="hi:Q")
        )  # fmt: skip
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=ARM_ORDER, title=None))
        rule = base.mark_rule().encode(x=alt.X("low95:Q", title=f"{tail} coverage", scale=scale),  # ty: ignore[unresolved-attribute]
                                       x2="high95:Q", color=colour())  # fmt: skip
        pts = base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("estimate:Q", scale=scale), color=colour()
        )
        panels.append(
            alt.layer(band, rule, pts).properties(width=WIDTH - 140, height=150, title=tail)
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Coverage of each tail quantile at the signed-MW six (target range shaded)",
            subtitle=[
                "Dot: share of actuals at or below the quantile, clean rows, five folds. Line: 95% interval",
                "from resampling origin days. Target: q01 0.005-0.02, q05 0.03-0.07, q95 0.93-0.97, q99 0.98-0.995.",
            ],
        )
    )
    chart.save(FIGURE_DIR / "figure_1_coverage_by_arm.svg")


def figure_folds() -> None:
    """q95 and q99 coverage of selected arms by fold."""
    s = pl.read_parquet(conformal.WORK / "scores.parquet").filter(
        (pl.col("rowset") == "five") & (pl.col("slice") == "clean") & (pl.col("by") == "group+month")
        & (pl.col("group") == "signed_six") & pl.col("arm").is_in(["raw", "P", "RC28", "RC28-bias"])
    )  # fmt: skip
    panels = []
    for tail in ("q95", "q99"):
        lo, hi = TARGETS[tail]
        band = alt.Chart(pl.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(  # ty: ignore[unresolved-attribute]
            opacity=0.18, color=theme.DATA_GREEN
        ).encode(y="lo:Q", y2="hi:Q")  # fmt: skip
        line = alt.Chart(s).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Fold (origin month)"),
            y=alt.Y(f"cov_{tail}:Q", title=f"{tail} coverage", scale=alt.Scale(zero=False)),
            color=colour(),
        )  # fmt: skip
        panels.append(alt.layer(band, line).properties(width=WIDTH - 80, height=170, title=tail))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Coverage by fold at the signed-MW six (target range shaded)",
            subtitle="Clean rows. January 2026 is the hardest fold for every arm.",
        )
    ).save(FIGURE_DIR / "figure_2_coverage_by_fold.svg")


def figure_contrasts() -> None:
    """CRPS and q99 pinball loss against raw, with Bonferroni intervals."""
    c = pl.read_csv(conformal.WORK / "contrasts.csv").filter(
        (pl.col("rowset") == "five") & (pl.col("group") == "signed_six")
        & (pl.col("against") == "raw") & pl.col("arm").is_in(ARM_ORDER)
    )  # fmt: skip
    c = c.with_columns(
        metric_label=pl.col("metric").replace(
            {"crps": "CRPS (points)", "pin_q99": "q99 pinball loss (points)"}
        ),
    )
    panels = []
    for metric, label in (("crps", "CRPS"), ("pin_q99", "q99 pinball loss")):
        part = c.filter(pl.col("metric") == metric)
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=theme.BLACK_1, strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        base = alt.Chart(part).encode(y=alt.Y("arm:N", sort=ARM_ORDER, title=None))
        rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low_bonferroni:Q", title=f"{label}: arm minus raw (points of effective capacity; negative is better)"),
            x2="high_bonferroni:Q", color=colour(),
        )  # fmt: skip
        pts = base.mark_point(filled=True, size=45).encode(x="difference:Q", color=colour())  # ty: ignore[unresolved-attribute]
        panels.append(alt.layer(zero, rule, pts).properties(width=WIDTH - 140, height=170))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Change in CRPS and q99 pinball loss against the raw forecast, signed-MW six",
            subtitle=[
                "Dot: estimate on clean rows. Line: 99.375% interval (Bonferroni, 8 tests for the four",
                "candidate arms; the two bias-only arms are post hoc and use the same level).",
            ],
        )
    ).save(FIGURE_DIR / "figure_3_contrasts_against_raw.svg")


def figure_shifts() -> None:
    """Mean buffer shift of q50, q95, and q99 over the scored origins, 28-day buffer, 12-36 h."""
    m = pl.read_csv(conformal.WORK / "shift_means.csv").filter(
        (pl.col("days") == 28) & (pl.col("lead_band") == "12-36h")
    )
    long = m.unpivot(
        on=["s50", "s95", "s99"], index=["origin"], variable_name="level", value_name="shift"
    )
    long = long.with_columns(points=pl.col("shift") * 100.0, day=pl.col("origin") // 48)
    alt.Chart(long).mark_line(point=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("day:Q", title="Origin day (grid index)", scale=alt.Scale(zero=False)),
        y=alt.Y("points:Q", title="Mean shift (points of effective capacity)"),
        color=alt.Color(
            "level:N",
            scale=alt.Scale(
                domain=["s50", "s95", "s99"],
                range=[theme.DATA_BLUE, theme.BRAND_ORANGE, theme.DATA_PURPLE],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    ).properties(
        width=WIDTH - 80,
        height=200,
        title=alt.TitleParams(
            "Rolling shift of the median, q95, and q99 (28-day buffer, 12 to 36 hours ahead)",
            subtitle="Mean over the 16 Primaries. A positive shift raises the forecast quantile.",
        ),
    ).save(FIGURE_DIR / "figure_4_shifts_over_time.svg")


def example_frames() -> dict[str, pl.DataFrame]:
    """Return the raw, RC28, and RC28-bias frames."""
    return {a: conformal.load_arm(arm=a) for a in ("raw", "RC28", "RC28-bias")}


def forecast_panel(
    *,
    frames: dict[str, pl.DataFrame],
    primary: int,
    origin: int,
    bands: tuple[str, ...] | None,
    title: str,
    event: bool = False,
) -> alt.Chart:
    """Return one forecast panel: actual, and the median and 90% band of raw and RC28-bias."""
    parts = []
    for arm in ("raw", "RC28-bias"):
        f = frames[arm].filter((pl.col("primary") == primary) & (pl.col("origin") == origin))
        if bands:
            f = f.filter(pl.col("lead_band").is_in(list(bands)))
        parts.append(
            f.with_columns(arm=pl.lit(arm)).select(
                "arm", "valid_time", "actual", "nra", "tgt_real", "q05", "q50", "q95"
            )
        )
    d = pl.concat(parts).with_columns(pl.col("actual", "nra", "q05", "q50", "q95") * 100.0)
    x = alt.X("valid_time:T", title="Valid time (UTC)")
    scale = alt.Scale(domain=["raw", "RC28-bias"], range=[theme.GREY_3, theme.DATA_GREEN])
    band = alt.Chart(d).mark_area(opacity=0.35).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("q05:Q", title="Share of capacity (%)", scale=alt.Scale(zero=False)), y2="q95:Q",
        color=alt.Color("arm:N", scale=scale, legend=alt.Legend(orient="top", title=None)))  # fmt: skip
    med = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(x=x, y="q50:Q", color=alt.Color("arm:N", scale=scale, legend=None))
    )
    act = (
        alt.Chart(d.filter(pl.col("arm") == "raw"))  # ty: ignore[unresolved-attribute]
        .mark_line(color=theme.BLACK_1, strokeWidth=1.5)
        .encode(x=x, y="actual:Q")
    )
    layers = [band, med, act]
    if event:
        ev = (
            d.filter(pl.col("arm") == "raw")
            .sort("valid_time")
            .with_columns(
                run=(pl.col("tgt_real") != pl.col("tgt_real").shift(1)).cum_sum().fill_null(0)
            )
            .filter(pl.col("tgt_real"))
            .group_by("run")
            .agg(start=pl.col("valid_time").min(), end=pl.col("valid_time").max())
        )
        if ev.height:
            layers.insert(
                0,
                alt.Chart(ev)  # ty: ignore[unresolved-attribute]
                .mark_rect(color=theme.BRAND_ORANGE, opacity=0.15)
                .encode(x=alt.X("start:T"), x2="end:T"),
            )
    return alt.layer(*layers).properties(width=WIDTH - 90, height=160, title=title)  # ty: ignore[invalid-return-type]


def figure_examples() -> None:
    """A day-ahead forecast and a full 10-day forecast, in winter and summer."""
    frames = example_frames()
    sites = refs448.sites_table()
    labels = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    six_primary = (
        frames["raw"]
        .filter(pl.col("group") == "signed_six")["primary"]
        .unique()
        .sort()
        .to_list()[0]
    )
    raw = frames["raw"]
    pick = lambda month: int(  # noqa: E731
        raw.filter((pl.col("primary") == six_primary) & (pl.col("month") == month))["origin"]
        .unique()
        .sort()[10]
    )
    name = labels[six_primary]
    panels = [
        forecast_panel(
            frames=frames,
            primary=six_primary,
            origin=pick("2026-01"),
            bands=("12-36h",),
            title=f"{name}, day ahead (12 to 36 hours), January 2026",
        ),
        forecast_panel(
            frames=frames,
            primary=six_primary,
            origin=pick("2026-06"),
            bands=("12-36h",),
            title=f"{name}, day ahead (12 to 36 hours), June 2026",
        ),
        forecast_panel(
            frames=frames,
            primary=six_primary,
            origin=pick("2026-01"),
            bands=None,
            title=f"{name}, full 10-day horizon from one origin, January 2026",
        ),
        forecast_panel(
            frames=frames,
            primary=six_primary,
            origin=pick("2026-06"),
            bands=None,
            title=f"{name}, full 10-day horizon from one origin, June 2026",
        ),
    ]
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Example forecasts at a signed-MW Primary: raw A-pool and rolling median correction",
            subtitle=[
                "Black: actual. Lines: median. Shade: 90% band (q05 to q95). Grey: raw. Green: the 28-day",
                "median-only correction (post hoc). The stored horizon is 10 days.",
            ],
        )
    ).save(FIGURE_DIR / "figure_5_example_forecasts.svg")


def figure_events() -> None:
    """Six forecasts that cross a real event (three at the six, three at the eight)."""
    frames = example_frames()
    raw = frames["raw"]
    sites = refs448.sites_table()
    labels = dict(zip(sites["primary"].to_list(), sites["label"].to_list(), strict=True))
    panels = []
    for group in ("signed_six", "mva_eight"):
        ev = (
            raw.filter((pl.col("group") == group) & pl.col("tgt_real"))
            .group_by("primary", "origin")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        chosen, seen = [], set()
        for r in ev.iter_rows(named=True):
            if r["primary"] in seen:
                continue
            seen.add(r["primary"])
            chosen.append(r)
            if len(chosen) == 3:
                break
        for r in chosen:
            panels.append(
                forecast_panel(
                    frames=frames,
                    primary=r["primary"],
                    origin=r["origin"],
                    bands=None,
                    event=True,
                    title=f"{labels[r['primary']]} ({'signed-MW six' if group == 'signed_six' else 'MVA-only eight'}): {r['n']} event half-hours",
                )
            )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Forecasts that cross a labelled load shift: raw A-pool and the 28-day median correction",
            subtitle=[
                "Black: metered actual. Orange shading: periods inside a labelled event (labels are incomplete,",
                "so unlabelled events may also appear). Grey: raw median and 90% band. Green: corrected.",
            ],
        )
    ).save(FIGURE_DIR / "figure_6_event_forecasts.svg")


def main() -> None:
    """Draw every figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_coverage()
    figure_folds()
    figure_contrasts()
    figure_shifts()
    figure_examples()
    figure_events()


if __name__ == "__main__":
    main()
