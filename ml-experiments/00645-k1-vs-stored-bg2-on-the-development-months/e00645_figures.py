# ruff: noqa: D103, E501
"""Experiment 645 figures (CPU): read the score tables; write SVGs for the page.

The window rows (experiment 633, July to October 2026) are typed in from the leakage audit's
section 3b, with the sign flipped to K1 minus BG2; they are quoted, not recomputed.

Usage: python e00645_figures.py SCORES_DIR OUT_DIR
"""

import sys
from pathlib import Path

import altair as alt
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BANDS = ["day 1", "day 2", "days 3-5", "days 6-10"]
# (K1 minus BG2, lo, hi) day-block 95%, from the audit's section 3b table.
WINDOW = {
    "Window, 16 Primaries, all rows": [
        (0.78, 0.68, 0.88),
        (0.71, 0.60, 0.81),
        (0.69, 0.60, 0.78),
        (0.71, 0.61, 0.81),
    ],
    "Window, 13 Primaries, event-free": [
        (0.03, -0.03, 0.09),
        (0.01, -0.05, 0.07),
        (0.01, -0.03, 0.06),
        (0.04, -0.02, 0.09),
    ],
}
COLOURS = {
    "Development months, 16 Primaries, all rows": theme.DATA_BLUE,
    "Development months, 13 Primaries, event-free": theme.DATA_SKY,
    "Window, 16 Primaries, all rows": theme.BRAND_ORANGE,
    "Window, 13 Primaries, event-free": "#FFA27D",
}


def headline(scores: Path, out: Path) -> None:
    c = pl.read_csv(scores / "contrasts.csv").filter(
        (pl.col("a") == "K1") & (pl.col("measure") == "crps") & pl.col("band").is_in(BANDS)
    )
    rows = []
    for name, cut, reading in (
        ("Development months, 16 Primaries, all rows", "16 Primaries", "all rows"),
        (
            "Development months, 13 Primaries, event-free",
            "13 (no D14, D15, D18)",
            "event-free (637 detector)",
        ),
    ):
        for r in c.filter((pl.col("cut") == cut) & (pl.col("reading") == reading)).iter_rows(
            named=True
        ):
            rows.append(
                {
                    "series": name,
                    "band": r["band"],
                    "d": r["a_minus_bg2"],
                    "lo": r["month_lo"],
                    "hi": r["month_hi"],
                }
            )
    for name, v in WINDOW.items():
        rows += [
            {"series": name, "band": b, "d": x[0], "lo": x[1], "hi": x[2]}
            for b, x in zip(BANDS, v, strict=True)
        ]
    df = pl.DataFrame(rows)
    base = alt.Chart(df).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("band:N", sort=BANDS, title=None),
        yOffset=alt.YOffset("series:N", sort=list(COLOURS)),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=alt.Legend(
                title=None, orient="top", columns=2, labelLimit=400, symbolType="circle"
            ),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#555", strokeDash=[4, 3])
        .encode(x="x:Q")
    )  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X(
                "lo:Q",
                title="K1's CRPS minus BG2's CRPS (points of capacity; below 0 means K1 is better, dashed line: equal)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(filled=True, size=60).encode(x="d:Q")
        + zero
    ).properties(
        width=560,
        height=300,
        title=alt.TitleParams(
            [
                "K1 is ahead of BG2 on the development months and behind it in the window",
                "Dot: estimate. Line: 95% interval (development months: resampling the 5 origin months; window: resampling origin days).",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        ),
    )  # ty: ignore[unresolved-attribute]
    chart.save(str(out / "k1_minus_bg2_by_lead_band.svg"))


def by_primary(scores: Path, out: Path) -> None:
    t = pl.read_csv(scores / "by_primary_and_month.csv").filter(
        (pl.col("by") == "primary")
        & (pl.col("reading") == "event-free (637 detector)")
        & pl.col("system").is_in(["K1", "BG2"])
    )
    w = (
        t.pivot(on="system", index="key", values="crps")
        .with_columns((pl.col("K1") - pl.col("BG2")).alias("d"))
        .sort("d")
    )
    order = w["key"].to_list()
    chart = (
        alt.Chart(w)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=theme.DATA_BLUE)
        .encode(
            y=alt.Y("key:N", sort=order, title=None),
            x=alt.X(
                "d:Q",
                title="K1's CRPS minus BG2's CRPS (points of capacity; below 0 means K1 is better)",
            ),
        )
        .properties(
            width=560,
            height=340,
            title=alt.TitleParams(
                [
                    "K1 beats BG2 at 14 of 16 Primaries; BG2 leads at D15 (short history) and, slightly, at D5",
                    "Development months, event-free rows (experiment 637's detector), days 1 to 10.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            ),
        )
    )
    chart.save(str(out / "k1_minus_bg2_by_primary.svg"))


def example(out: Path) -> None:
    exp = ML_EXPERIMENTS_DATA_DIR / "backtest_export_roster_2026-11-07"
    bg = pl.read_parquet(
        next(ML_EXPERIMENTS_DATA_DIR.glob("00087-*"))
        / "predictions"
        / "p_s6_bg2"
        / "predictions.parquet"
    )
    origin = pl.datetime(2025, 12, 10, 9, time_zone="UTC")
    k1 = pl.read_parquet(exp / "K1_2025-12.parquet").filter(
        (pl.col("time_series_id") == 17)
        & (pl.col("power_fcst_init_time") == origin)
        & (pl.col("lead_hours") < 240)
    )
    b2 = bg.filter(
        (pl.col("time_series_id") == 17)
        & (pl.col("power_fcst_init_time") == origin)
        & (pl.col("lead_hours") < 240)
    )
    cap = float(b2["capacity_mw"][0])
    panels = []
    for name, d, scale in (("K1", k1, 1.0), ("BG2", b2, cap)):
        d = d.with_columns(
            *(pl.col(q) * scale for q in ("q05", "q50", "q95")), hours=pl.col("lead_hours")
        ).select("hours", "q05", "q50", "q95")
        act = k1.select(
            pl.col("lead_hours").alias("hours"), pl.col("observed_power").alias("actual")
        )
        x = alt.X(
            "hours:Q",
            title="Hours after the forecast origin (09:00 UTC)",
            axis=alt.Axis(values=[0, 48, 96, 144, 192, 240]),
        )
        band = (
            alt.Chart(d)
            .mark_area(opacity=0.25, color=theme.DATA_BLUE)
            .encode(x=x, y=alt.Y("q05:Q", title="Power (MW)"), y2="q95:Q")
        )  # ty: ignore[unresolved-attribute]
        med = alt.Chart(d).mark_line(color=theme.DATA_BLUE).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
        obs = alt.Chart(act).mark_line(color="#222", strokeWidth=1).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
        panels.append((band + med + obs).properties(width=560, height=170, title=name))
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Ten-day forecasts of D5 issued on 10 December 2025",
                "Black: observed power. Blue line: median. Shade: 90% band (q05 to q95). First 24 hours are the day-ahead forecast.",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        )
    ).save(str(out / "example_d5.svg"))


if __name__ == "__main__":
    s, o = Path(sys.argv[1]), Path(sys.argv[2])
    o.mkdir(parents=True, exist_ok=True)
    headline(s, o)
    by_primary(s, o)
    example(o)
