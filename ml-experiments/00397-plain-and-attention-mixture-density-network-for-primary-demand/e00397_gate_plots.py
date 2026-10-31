# ruff: noqa: E501
"""Gate-weight and token-norm charts of the attention arms of experiment 397 (interim or final).

Reads the gate logs the fit script wrote for the arms `A0` and `A2`, averages the seeds, and writes
four charts (SVG and PNG) to `work/charts/`:

1. `weights_by_lead`: A2's mean weight of each input group against lead time.
2. `token_norms_by_lead`: each group's token norm against lead time, for A0 and A2.
3. `weights_by_day_type`: A2's mean weight by day type.
4. `christmas_vs_december`: A2's mean weight on 24 December to 1 January against 1 to 23
   December, by lead band (needs the 2025-12 fold).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
uv run --no-sync python ml-experiments/00397-*/e00397_gate_plots.py --months 2025-10 2025-12
```
"""

import argparse
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

WIDTH: Final[int] = 680
HEIGHT: Final[int] = 220
SEED_DIR: Final[Path] = prep.WORK_DIR / "work" / "seed_preds" / "summary"
CHART_DIR: Final[Path] = prep.WORK_DIR / "work" / "charts"
GROUPS: Final[tuple[str, ...]] = ("weather", "solar", "calendar", "holidays", "lags")
GROUP_COLOURS: Final[tuple[str, ...]] = (
    theme.DATA_BLUE,
    theme.DATA_SKY,
    theme.BRAND_ORANGE,
    theme.DATA_PURPLE,
    theme.DATA_GREEN,
)
BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("day 1", 0.0, 24.0),
    ("days 2-3", 24.0, 72.0),
    ("days 4-7", 72.0, 168.0),
    ("days 8-10", 168.0, 240.0),
)
BIN_HOURS: Final[float] = 12.0
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
INTERIM: Final[str] = "Interim: screen months only, 3 seeds, no interval."


def load(*, arm: str, months: list[str]) -> pl.DataFrame:
    """Return an arm's gate logs, averaged over seeds, with local calendar columns."""
    pieces = [
        pl.read_parquet(f)
        for m in months
        for f in sorted((SEED_DIR / arm).glob(f"{m}_s*_gate.parquet"))
    ]
    value_columns = [f"{p}_{g}" for p in ("w", "n") for g in GROUPS]
    frame = (
        pl.concat(pieces)
        .with_columns(pl.col(value_columns).cast(pl.Float32))
        .group_by(KEYS)
        .agg(pl.col("lead_hours").first(), *[pl.col(c).mean() for c in value_columns])
    )
    local = pl.col("valid_time").dt.convert_time_zone("Europe/London")
    gaps = common.holiday_gaps().select("date", "days_to_nearest_holiday")
    frame = frame.with_columns(date=local.dt.date(), weekday=local.dt.weekday()).join(
        gaps, on="date", how="left"
    )
    return frame.with_columns(
        lead_bin=(pl.col("lead_hours") // BIN_HOURS * BIN_HOURS + BIN_HOURS / 2).cast(pl.Float64)
    )


def long(*, frame: pl.DataFrame, prefix: str, by: list[str]) -> pl.DataFrame:
    """Return the mean of each group's column, one row per group and `by` value."""
    return (
        frame.group_by(by)
        .agg(*[pl.col(f"{prefix}_{g}").mean().alias(g) for g in GROUPS])
        .unpivot(index=by, variable_name="group", value_name="value")
        .sort(by)
    )


def colour(*, title: str = "Input group") -> alt.Color:
    """Return the colour encoding of the five input groups."""
    return alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(GROUPS), range=list(GROUP_COLOURS)),
        legend=alt.Legend(title=title, orient="top", direction="horizontal"),
    )


def line_chart(*, data: pl.DataFrame, y_title: str, title: list[str], zero: bool) -> alt.Chart:
    """Return a line chart of a value against lead time, one line per group."""
    return (
        alt.Chart(data.to_pandas())
        .mark_line(strokeWidth=2, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lead_bin:Q", title="Lead time (hours; 240 hours = day 10)",
                    axis=alt.Axis(values=[0, 24, 72, 120, 168, 240, 336]),
                    scale=alt.Scale(domain=[0, 336])),
            y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=zero)),
            color=colour(),
        )
        .properties(width=WIDTH - 90, height=HEIGHT, title=alt.TitleParams(title, anchor="start"))
    )  # fmt: skip


def save(*, chart: alt.Chart | alt.VConcatChart, name: str) -> None:
    """Write a chart as SVG and PNG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(CHART_DIR / f"{name}.svg")
    chart.save(CHART_DIR / f"{name}.png", scale_factor=2)
    print(CHART_DIR / f"{name}.png")


def day_type(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add a day-type column: Christmas slice, bank holiday, Saturday, Sunday, or weekday."""
    month_day = pl.col("date").dt.strftime("%m-%d")
    return frame.with_columns(
        day_type=pl.when((month_day >= "12-24") | (month_day <= "01-01"))
        .then(pl.lit("24 Dec to 1 Jan"))
        .when(pl.col("days_to_nearest_holiday") == 0)
        .then(pl.lit("Bank holiday"))
        .when(pl.col("weekday") == 6)
        .then(pl.lit("Saturday"))
        .when(pl.col("weekday") == 7)
        .then(pl.lit("Sunday"))
        .otherwise(pl.lit("Ordinary weekday"))
    )


def main() -> None:
    """Draw the charts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    args = parser.parse_args()
    months = args.months
    a2 = load(arm="A2", months=months)
    a0 = load(arm="A0", months=months)
    scope = f"Folds {', '.join(months)}; pooled over 16 Primaries."

    by_lead = long(frame=a2, prefix="w", by=["lead_bin"])
    save(
        chart=line_chart(
            data=by_lead, y_title="Mean gate weight (uniform = 0.2)", zero=False,
            title=["A2 gate weight of each input group against lead time", f"{scope} {INTERIM}"],
        ) + alt.Chart(pl.DataFrame({"y": [0.2]}).to_pandas()).mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
        name="weights_by_lead",
    )  # fmt: skip

    norms = pl.concat(
        [
            long(frame=f, prefix="n", by=["lead_bin"]).with_columns(arm=pl.lit(a))
            for a, f in (("A0 (weights fixed at 0.2)", a0), ("A2", a2))
        ]
    )
    norm_chart = (
        alt.Chart(norms.to_pandas())
        .mark_line(strokeWidth=2, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lead_bin:Q", title="Lead time (hours; 240 hours = day 10)",
                    axis=alt.Axis(values=[0, 24, 72, 120, 168, 240, 336]),
                    scale=alt.Scale(domain=[0, 336])),
            y=alt.Y("value:Q", title="Mean token norm", scale=alt.Scale(zero=False)),
            color=colour(),
            row=alt.Row("arm:N", title=None),
        )
        .properties(width=WIDTH - 90, height=HEIGHT // 1.6,
                    title=alt.TitleParams(["Token norm of each input group against lead time", f"A larger norm means a larger contribution at the same weight. {scope} {INTERIM}"], anchor="start"))
    )  # fmt: skip
    save(chart=norm_chart, name="token_norms_by_lead")

    types = long(frame=day_type(frame=a2), prefix="w", by=["day_type"])
    order = ["Ordinary weekday", "Saturday", "Sunday", "Bank holiday", "24 Dec to 1 Jan"]
    dots = (
        alt.Chart(types.to_pandas())
        .mark_point(filled=True, size=90, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Mean gate weight (uniform = 0.2)", scale=alt.Scale(zero=False)),
            y=alt.Y("day_type:N", sort=order, title=None),
            color=colour(),
            shape=alt.Shape("group:N", legend=None),
        )
        .properties(width=WIDTH - 140, height=HEIGHT,
                    title=alt.TitleParams(["A2 gate weight by type of day", f"Bank holidays include Christmas Day, Boxing Day, and New Year's Day. {scope} {INTERIM}"], anchor="start"))
    )  # fmt: skip
    save(chart=dots, name="weights_by_day_type")

    december = a2.filter(pl.col("date").dt.month() == 12)
    if december.height:
        month_day = pl.col("date").dt.strftime("%m-%d")
        parts = []
        for name, where in (
            ("1 to 23 December", month_day < "12-24"),
            ("24 December to 1 January", (month_day >= "12-24") | (month_day <= "01-01")),
        ):
            sub = a2.filter(
                where & ((pl.col("date").dt.month() == 12) | (pl.col("date").dt.month() == 1))
            )
            for band, lo, hi in BANDS:
                cut = sub.filter(pl.col("lead_hours").is_between(lo, hi, closed="right"))
                row = {"period": name, "band": band} | {
                    g: float(cut[f"w_{g}"].to_numpy().mean()) for g in GROUPS
                }
                parts.append(pl.DataFrame([row]))
        both = pl.concat(parts).unpivot(
            index=["period", "band"], variable_name="group", value_name="value"
        )
        xmas = (
            alt.Chart(both.to_pandas())
            .mark_point(filled=True, size=80, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("value:Q", title="Mean gate weight (uniform = 0.2)", scale=alt.Scale(zero=False)),
                y=alt.Y("band:N", sort=[b[0] for b in BANDS], title=None),
                color=colour(),
                shape=alt.Shape("group:N", legend=None),
                row=alt.Row("period:N", title=None),
            )
            .properties(width=WIDTH - 140, height=HEIGHT // 1.8,
                        title=alt.TitleParams(["A2 gate weight on 24 December to 1 January against 1 to 23 December, by lead band", f"December fold only (one Christmas per Primary in the test month). {INTERIM}"], anchor="start"))
        )  # fmt: skip
        save(chart=xmas, name="christmas_vs_december")


if __name__ == "__main__":
    main()
