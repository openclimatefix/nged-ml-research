# ruff: noqa: E501
"""Draw experiment 485's figures from the stored scores (no fit).

Figure 1: each arm minus H0 on the gate and Christmas slices (Bonferroni intervals).
Figure 2: CRPS by valid date, 17 December to 6 January, at the eight and the six.
Figures 3 and 4: a day-ahead and a 14-day forecast of one Primary (H0 and H2) over Christmas.

```bash
source /home/jack/e485env.sh
uv run --no-sync python ml-experiments/00485-*/e00485_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as s470  # ty: ignore[unresolved-import]
import e00485_fit as fit485  # ty: ignore[unresolved-import]
import e00485_score as score485  # ty: ignore[unresolved-import]
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00485-christmas-period-demand-forecast-regime-arms"
)
SCORES: Final[Path] = fit485.WORK_DIR / "scores"
COLOURS: Final[dict[str, str]] = {
    "H0": theme.BLACK_1, "H1": theme.DATA_BLUE, "H2": theme.BRAND_ORANGE, "H3": theme.DATA_GREEN,
    "H4": theme.DATA_PURPLE, "P": theme.GREY_3, "BG2": "#c0392b",
}  # fmt: skip
LABELS: Final[dict[str, str]] = {
    "eight": "Eight, days outside the window", "six_clean": "Six, no labelled event, outside window",
    "cold_eight": "Eight, cold nights, outside window", "cold_six": "Six, cold nights, outside window",
    "xmas_eight": "Eight, 22-31 Dec 2025 (suggestive)", "xmas_six": "Six, 22-31 Dec 2025 (suggestive)",
}  # fmt: skip


def figure_contrasts() -> None:
    """Draw each arm minus H0 with Bonferroni intervals (CRPS, points of capacity)."""
    c = pl.read_csv(SCORES / "contrasts.csv").filter(
        (pl.col("metric") == "crps") & pl.col("slice").is_in(list(LABELS))
    )
    c = c.with_columns(
        label=pl.col("slice").replace(LABELS), arm=pl.col("contrast").str.replace(" - H0", "")
    )
    base = alt.Chart(c.to_pandas())  # ty: ignore[unresolved-attribute]
    y = alt.Y("label:N", sort=list(LABELS.values()), title=None, axis=alt.Axis(labelLimit=300))
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(COLOURS)[1:6], range=list(COLOURS.values())[1:6]),
        legend=alt.Legend(title=None, orient="top", columns=5),
    )
    bars = base.mark_rule().encode(
        y=y,
        x=alt.X("low_bonferroni:Q", title="Arm minus H0 (CRPS points; below zero is better)"),
        x2="high_bonferroni:Q",
        color=colour,
        yOffset="arm:N",
    )
    dots = base.mark_point(filled=True, size=40).encode(
        y=y, x="difference:Q", color=colour, yOffset="arm:N"
    )
    zero = (
        alt.Chart(pl.DataFrame({"z": [0.0]}).to_pandas())
        .mark_rule(strokeDash=[3, 3], color=theme.BLACK_1)
        .encode(x="z:Q")
    )  # ty: ignore[unresolved-attribute]
    (bars + dots + zero).properties(
        width=520,
        height=300,
        title=alt.TitleParams(
            "No arm gains over Christmas by more than its minimum detectable effect",
            subtitle="Bonferroni intervals (count 45 for the gate rows); Christmas rows use one holiday and are descriptive.",
            anchor="start",
        ),
    ).save(FIGURE_DIR / "figure_1_contrasts.svg")


def figure_by_day() -> None:
    """Draw CRPS by valid date over the holiday for H0, H2, H3, P, and BG2."""
    parts = []
    for m in score485.MONTHS:
        t = pl.read_parquet(SCORES / f"rows_{m}.parquet").filter(pl.col("ref"))
        parts.append(
            t.select(
                "group",
                "month",
                "valid_day",
                *[f"{a}__crps" for a in ("H0", "H2", "H3", "P", "BG2")],
            )
        )
    t = pl.concat(parts).with_columns(
        date=(pl.col("valid_day") * 86_400_000).cast(pl.Datetime("ms"))
    )
    t = t.filter(pl.col("date").is_between(pl.datetime(2025, 12, 17), pl.datetime(2026, 1, 6)))
    long = t.unpivot(
        index=["group", "date"],
        on=[c for c in t.columns if c.endswith("__crps")],
        variable_name="arm",
        value_name="crps",
    ).with_columns(pl.col("arm").str.replace("__crps", ""))
    long = long.group_by("group", "date", "arm").agg(pl.col("crps").mean())
    panels = []
    for group, title in (
        ("mva_eight", "Eight MVA-only Primaries"),
        ("signed_six", "Signed-MW six"),
    ):
        d = long.filter(pl.col("group") == group)
        panels.append(
            alt.Chart(d.to_pandas())
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Valid date (UTC)"),
                y=alt.Y("crps:Q", title="CRPS (points of capacity)"),
                color=alt.Color(
                    "arm:N",
                    scale=alt.Scale(
                        domain=["H0", "H2", "H3", "P", "BG2"],
                        range=[COLOURS[a] for a in ("H0", "H2", "H3", "P", "BG2")],
                    ),
                    legend=alt.Legend(title=None, orient="top", columns=5),
                ),
            )
            .properties(width=520, height=170, title=title)
        )  # ty: ignore[unresolved-attribute]
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "CRPS by valid date over the holiday, 17 December 2025 to 6 January 2026",
            anchor="start",
        )
    ).save(FIGURE_DIR / "figure_2_crps_by_date.svg")


def figure_examples(*, site: str = "D4") -> None:
    """Draw a day-ahead and a 14-day forecast of one Primary from the 24 December 2025 run: H0 and H2."""
    sites = build470.site_map()
    ctx = prep425.cleaned_scoring_context(series=adapter.series_table())
    base = s470.month_context(month="2025-12", sites=sites, ctx=ctx)
    primary = int(base.filter(pl.col("group") == "mva_eight")["primary"].unique().sort()[0])
    rows = base.filter(pl.col("primary") == primary)
    origin = (
        rows.filter(pl.col("valid_time").dt.date() == pl.date(2025, 12, 25))["power_fcst_init_time"]
        .dt.date()
        .min()
    )
    del origin
    first = rows.filter(pl.col("power_fcst_init_time").dt.date() == pl.date(2025, 12, 23)).sort(
        "valid_time"
    )
    out = {}
    for arm in ("H0", "H2"):
        f = score485.seed_frame(arm=arm, month="2025-12", seed=0, sites=sites)
        q = first.select(*s470.KEYS, "valid_time", "actual").join(
            f.select(*s470.KEYS, *s470.LEVELS), on=s470.KEYS, how="inner"
        )
        out[arm] = widen(frame=q, factor=s470.HEADLINE).with_columns(arm=pl.lit(arm))
    data = pl.concat(
        [o.with_columns(pl.col(list(s470.LEVELS)).cast(pl.Float64)) for o in out.values()]
    )
    for name, hours, number in (("day_ahead", 36, 3), ("fourteen_day", 14 * 24, 4)):
        d = data.filter(pl.col("valid_time") <= data["valid_time"].min() + pl.duration(hours=hours))
        x = alt.X("valid_time:T", title="Valid time (UTC)")
        col = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=["H0", "H2"], range=[COLOURS["H0"], COLOURS["H2"]]),
            legend=alt.Legend(title=None, orient="top"),
        )
        band90 = (
            alt.Chart(d.to_pandas())
            .mark_area(opacity=0.15)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=col,
            )
        )  # ty: ignore[unresolved-attribute]
        band50 = (
            alt.Chart(d.to_pandas())
            .mark_area(opacity=0.3)
            .encode(x=x, y="q25:Q", y2="q75:Q", color=col)
        )  # ty: ignore[unresolved-attribute]
        med = alt.Chart(d.to_pandas()).mark_line().encode(x=x, y="q50:Q", color=col)  # ty: ignore[unresolved-attribute]
        act = (
            alt.Chart(d.filter(pl.col("arm") == "H0").to_pandas())
            .mark_line(color="#333", strokeDash=[2, 2])
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        (band90 + band50 + med + act).properties(
            width=560,
            height=240,
            title=alt.TitleParams(
                f"One run (23 December 2025 origin) at a Primary: H0 and H2, {name.replace('_', ' ')}",
                subtitle="Median, 50% and 90% bands; dashed line is the actual.",
                anchor="start",
            ),
        ).save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")
    del site


if __name__ == "__main__":
    figure_contrasts()
    figure_by_day()
    figure_examples()
