# ruff: noqa: E501, C408
"""Experiment 468, Part A: time-series graphs of the best wind and solar forecasters.

Reads experiment 461's stored, un-adjusted rows (`work/rows_*.parquet`: quantiles and actuals in MW,
true days 1 to 10), applies table P month by month exactly as experiment 461 did (factors fitted on
earlier scored months only, clip in MW), and draws charts for Farm A, B, and C (wind
`member_plain`, p99.9 capacity) and Solar A and B (`a2_member` and `g0_onepass`, old capacity
rule, because experiment 461 did not adopt p99.9 for Solar A). The x-axes count days and carry no
calendar date. Charts are private: they carry megawatts of an anonymised generator.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
uv run --no-sync python ml-experiments/00468-*/e00468_part_a.py
```

Writes charts to `<experiment data folder>/part_a/` and tables to `part_a/tables/`.
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK461: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00461-wind-and-solar-retrain-with-p99-9-effective-capacity" / "work"
)
OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00468-best-models-performance-analysis-with-time-series-graphs"
    / "part_a"
)
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVEL_VALUES: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
)
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
MIN_FIT_ROWS: Final[int] = 200
SITES: Final[dict[str, list[str]]] = {
    "wind": ["Farm A", "Farm B", "Farm C"],
    "solar": ["Solar A", "Solar B"],
}
ARMS: Final[dict[str, list[tuple[str, str]]]] = {
    "wind": [("member_plain", "p999")],
    "solar": [("a2_member", "old"), ("g0_onepass", "old")],
}
WIDTH: Final[int] = 640


def fit_factors(*, fit: pl.DataFrame) -> dict[tuple[str, str, str], float]:
    """Return table P: one factor per (series, lead band, tail quantile), as experiment 450 does."""
    out: dict[tuple[str, str, str], float] = {}
    for (site, band), part in fit.group_by("site", "lead_band"):
        for name, tau in TAILS.items():
            spread = (pl.col(name) - pl.col("q50")).abs()
            upper = name in UPPER
            ratio = (
                (pl.col("actual") - pl.col("q50")) / spread
                if upper
                else (pl.col("q50") - pl.col("actual")) / spread
            )
            use = part.filter(spread > 1e-6)
            if use.height >= MIN_FIT_ROWS:
                out[site, band, name] = float(
                    use.select(ratio.quantile(tau if upper else 1.0 - tau)).item()
                )
    return out


def apply_table_p(*, rows: pl.DataFrame, factors: dict, upper: dict[str, float]) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its factor, clip in MW."""
    out = rows
    for name in TAILS:
        lookup = pl.DataFrame(
            [(s, b, f) for (s, b, n), f in factors.items() if n == name],
            schema={"site": pl.String, "lead_band": pl.String, "factor": pl.Float64},
            orient="row",
        )
        out = (
            out.join(lookup, on=["site", "lead_band"], how="left")
            .with_columns(
                (pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")))
                .clip(0.0, None)
                .alias(name)
            )
            .drop("factor")
        )
    upper_expr = pl.col("site").replace_strict(upper, return_dtype=pl.Float64)
    return out.with_columns(*[pl.col(c).clip(upper_bound=upper_expr) for c in LEVELS])


def table_p_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the scored months (all but the first) after table P, fitted on earlier months."""
    months = sorted(frame["month"].unique())
    parts = []
    for i, month in enumerate(months[1:], start=1):
        earlier = frame.filter(pl.col("month").is_in(months[:i]))
        held = frame.filter(pl.col("month") == month)
        cap_now = {
            r["site"]: float(r["c"])
            for r in held.group_by("site")
            .agg(pl.col("cap_mw").max().alias("c"))
            .iter_rows(named=True)
        }
        earlier_max = {
            r["site"]: float(r["m"])
            for r in earlier.group_by("site")
            .agg(pl.col("actual").max().alias("m"))
            .iter_rows(named=True)
        }
        upper = {s: max(cap_now[s], earlier_max.get(s, 0.0)) for s in cap_now}
        parts.append(apply_table_p(rows=held, factors=fit_factors(fit=earlier), upper=upper))
    return pl.concat(parts)


def load(*, tech: str) -> pl.DataFrame:
    """Return the table-P rows of every headline arm of `tech` for the shown series."""
    frames = []
    for arm, rule in ARMS[tech]:
        raw = pl.read_parquet(WORK461 / f"rows_{arm}_{rule}.parquet").filter(
            pl.col("site").is_in(SITES[tech])
        )
        frames.append(
            table_p_rows(frame=raw).with_columns(
                valid=pl.col("init")
                + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
            )
        )
    return pl.concat(frames)


def chart_title(*, lines: list[str]) -> alt.TitleParams:
    """Return a wrapped title."""
    return alt.TitleParams(text=lines, anchor="start", fontSize=13)


def save(*, chart: alt.Chart | alt.LayerChart | alt.VConcatChart, name: str) -> None:
    """Write an SVG chart to the private folder."""
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / f"{name}.svg")
    (OUT / "png").mkdir(exist_ok=True)
    chart.save(OUT / "png" / f"{name}.png", scale_factor=1.5)


def band_chart(
    *, frame: pl.DataFrame, x_hours: str, title: list[str], height: int = 200, cap: float
) -> alt.LayerChart:
    """Return median, 50% band, 90% band, and actual against hours since the first row."""
    frame = frame.sort(x_hours).with_columns(
        seg=(pl.col(x_hours).diff().fill_null(0.0) > 1.01).cum_sum()
    )
    base = alt.Chart(frame)
    x = alt.X(
        f"{x_hours}:Q",
        title="Days since the start of the period shown (day 1 is the first 24 hours)",
        axis=alt.Axis(values=list(range(0, 24 * 15, 24)), labelExpr="datum.value / 24 + 1"),
    )
    y = alt.Y("y:Q", title="Power (MW)", scale=alt.Scale(domain=[0, cap]))
    wide = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(
        x=x, y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(domain=[0, cap])), y2="q95:Q"
    )
    mid = base.mark_area(opacity=0.35, color=ocf.DATA_BLUE).encode(x=x, y="q25:Q", y2="q75:Q")
    med = base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1.6).encode(
        x=x, y="q50:Q", detail="seg:N"
    )
    act = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.4).encode(
        x=x, y="actual:Q", detail="seg:N"
    )
    del y
    return (wide + mid + med + act).properties(
        width=WIDTH - 90, height=height, title=chart_title(lines=title)
    )


def run_charts(*, rows: pl.DataFrame, tech: str, arm: str) -> list[dict]:
    """Draw the stormy, calm, and worst-tail forecast runs (wind) or weeks (solar)."""
    notes = []
    for site in SITES[tech]:
        s = rows.filter((pl.col("site") == site) & (pl.col("arm") == arm)).sort("valid")
        cap = float(s["cap_mw"].max())
        if tech == "wind":
            runs = (
                s.group_by("init")
                .agg(
                    mean=pl.col("actual").mean() / cap,
                    n=pl.len(),
                    over99=(pl.col("actual") > pl.col("q99")).sum(),
                )
                .filter(pl.col("n") > 300)
            )
            picks = {
                "stormy": runs.sort("mean").tail(1)["init"][0],
                "calm": runs.sort("mean").head(1)["init"][0],
                "worst_tail": runs.sort("over99").tail(1)["init"][0],
            }
            for kind, init in picks.items():
                part = s.filter(pl.col("init") == init).with_columns(
                    hours=(pl.col("lead_hours") - 24.0)
                )
                m = float(part["actual"].mean()) / cap
                mae = float((part["q50"] - part["actual"]).abs().mean()) / cap * 100
                cov = float(
                    ((part["actual"] >= part["q05"]) & (part["actual"] <= part["q95"])).mean()
                )
                title = [
                    f"{site}: one {kind.replace('_', '-')} forecast run, 24 to 240 hours ahead.",
                    f"Mean output {m * 100:.0f}% of capacity; error of the median forecast {mae:.1f} points;",
                    f"{cov * 100:.0f}% of actuals inside the 90% band ({part.height} half-hours).",
                ]
                save(
                    chart=band_chart(frame=part, x_hours="hours", title=title, cap=cap * 1.05),
                    name=f"run_{kind}_{site.replace(' ', '_')}",
                )
                notes.append(
                    dict(site=site, kind=kind, mean_cf=m, mae_pts=mae, cov90=cov, n=part.height)
                )
        else:
            near = s.filter(pl.col("lead_hours") < 48).with_columns(
                day=pl.col("valid").dt.truncate("1d")
            )
            full = (
                near.with_columns(week=pl.col("valid").dt.truncate("1w"))
                .group_by("week")
                .len()["len"]
                .max()
            )
            wk_floor = int(0.95 * full)
            wk = (
                near.with_columns(
                    week=pl.col("valid").dt.truncate("1w"), month=pl.col("valid").dt.month()
                )
                .filter(pl.col("month").is_between(4, 8))
                .group_by("week")
                .agg(mean=pl.col("actual").mean(), n=pl.len())
                .filter(pl.col("n") >= wk_floor)
            )
            picks = {
                "clear_sky": wk.sort("mean").tail(1)["week"][0],
                "cloudy": wk.sort("mean").head(1)["week"][0],
            }
            for kind, week in picks.items():
                part = near.filter(pl.col("valid").dt.truncate("1w") == week)
                t0 = part["valid"].min()
                part = part.with_columns(
                    hours=((pl.col("valid") - t0).dt.total_minutes() / 60.0)
                ).sort("valid")
                mae = float((part["q50"] - part["actual"]).abs().mean()) / cap * 100
                title = [
                    f"{site}: a {kind.replace('_', ' ')} week, day-ahead forecast (24 to 48 hours).",
                    f"Error of the median forecast {mae:.1f} points of capacity over {part.height} daylight",
                    "half-hours scored (the stored rows hold the half-hours the scoring kept).",
                ]
                save(
                    chart=band_chart(frame=part, x_hours="hours", title=title, cap=cap * 1.05),
                    name=f"week_{kind}_{site.replace(' ', '_')}",
                )
                notes.append(dict(site=site, kind=kind, mae_pts=mae, n=part.height))
    return notes


def coverage_table(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the share of actuals at or below each tail quantile, per arm, series, and month."""
    return (
        rows.with_columns(month=pl.col("valid").dt.strftime("%Y-%m"))
        .group_by("arm", "site", "month")
        .agg(
            n=pl.len(),
            **{
                f"below_{t}": (
                    (pl.col("actual") > pl.col(t)).mean() * 100
                    if t in UPPER
                    else (pl.col("actual") < pl.col(t)).mean() * 100
                )
                for t in TAILS
            },
        )
        .sort("arm", "site", "month")
    )


def coverage_chart(*, table: pl.DataFrame, tech: str, arm: str) -> None:
    """Draw below-quantile shares by month for the four tail levels."""
    long = (
        table.filter(pl.col("arm") == arm)
        .unpivot(
            index=["site", "month", "n"],
            on=[f"below_{t}" for t in TAILS],
            variable_name="level",
            value_name="share",
        )
        .with_columns(
            level=pl.col("level").str.replace("below_", ""),
        )
    )
    long.write_csv(OUT / f"coverage_{tech}.csv")
    charts = []
    for level, tau in TAILS.items():
        d = long.filter(pl.col("level") == level)
        line = (
            alt.Chart(d)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "month:O",
                    title="Month scored" if level == "q99" else None,
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(
                    "share:Q",
                    title=f"% above {level}" if level in UPPER else f"% at or below {level}",
                    scale=alt.Scale(domain=[0, max(6.0, float(d["share"].max()) * 1.1)]),
                ),
                color=alt.Color(
                    "site:N",
                    title=None,
                    legend=alt.Legend(orient="top"),
                    scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                ),
            )
        )
        nominal = (1 - tau) * 100 if level in UPPER else tau * 100
        rule = (
            alt.Chart(pl.DataFrame({"y": [nominal]})).mark_rule(strokeDash=[4, 3]).encode(y="y:Q")
        )
        charts.append((line + rule).properties(width=WIDTH - 90, height=110))
    save(
        chart=alt.vconcat(*charts).properties(
            title=chart_title(
                lines=[
                    f"{tech.capitalize()} ({arm}, after table P): share of actuals beyond each tail",
                    "quantile, by month (below q01 and q05, above q95 and q99). Dashed rule: the",
                    "nominal share (a calibrated forecast sits on it). One dot is one month.",
                ]
            )
        ),
        name=f"coverage_by_month_{tech}",
    )


def lead_chart(*, rows: pl.DataFrame, tech: str) -> None:
    """Draw the median error and quantile score by lead day for each arm."""
    d = (
        rows.with_columns(
            day=(pl.col("lead_hours") // 24).cast(pl.Int32) + 0.5,
            ae=(pl.col("q50") - pl.col("actual")).abs() / pl.col("cap_mw") * 100,
        )
        .group_by("arm", "site", "day")
        .agg(mae=pl.col("ae").mean(), n=pl.len())
        .sort("arm", "site", "day")
    )
    d.write_csv(OUT / f"error_by_lead_{tech}.csv")
    chart = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:Q", title="Lead time (days ahead)", axis=alt.Axis(values=list(range(0, 11, 2)))
            ),
            y=alt.Y(
                "mae:Q",
                title="Mean absolute error of the median forecast (% of capacity; smaller is better)",
            ),
            color=alt.Color("site:N", title=None, legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("arm:N", title=None),
        )
        .properties(
            width=WIDTH - 90,
            height=200,
            title=chart_title(
                lines=[
                    f"{tech.capitalize()}: mean absolute error of the median forecast by lead time, per series, after table P.",
                    "Each point averages that lead day's half-hours (see the CSV for counts).",
                ]
            ),
        )
    )
    save(chart=chart, name=f"error_by_lead_{tech}")


def worst_tail_table(*, rows: pl.DataFrame, tech: str, arm: str) -> pl.DataFrame:
    """Return the 20 worst half-hours by q99 and q01 pinball loss per series, as day-in-run counts."""
    out = []
    for site in SITES[tech]:
        s = rows.filter((pl.col("site") == site) & (pl.col("arm") == arm))
        for t, tau in (("q99", 0.99), ("q01", 0.01)):
            d = pl.col("actual") - pl.col(t)
            pin = pl.when(d >= 0).then(tau * d).otherwise((tau - 1) * d)
            w = s.with_columns(pin=pin / pl.col("cap_mw")).sort("pin", descending=True).head(20)
            out.append(
                w.select(
                    "site",
                    level=pl.lit(t),
                    pin_cf=pl.col("pin"),
                    lead_hours="lead_hours",
                    actual_cf=pl.col("actual") / pl.col("cap_mw"),
                    quantile_cf=pl.col(t) / pl.col("cap_mw"),
                    run=pl.col("init").dt.strftime("%Y-%m"),
                    hour_of_day=pl.col("valid").dt.hour(),
                )
            )
    return pl.concat(out)


def main() -> None:
    """Build Part A tables and charts."""
    OUT.mkdir(parents=True, exist_ok=True)
    notes: list[dict] = []
    for tech in ("wind", "solar"):
        rows = load(tech=tech)
        rows.write_parquet(OUT / f"rows_after_p_{tech}.parquet")
        arm = ARMS[tech][0][0]
        notes += run_charts(rows=rows.filter(pl.col("arm") == arm), tech=tech, arm=arm)
        cov = coverage_table(rows=rows)
        coverage_chart(table=cov, tech=tech, arm=arm)
        lead_chart(rows=rows.filter(pl.col("arm") == arm), tech=tech)
        w = worst_tail_table(rows=rows, tech=tech, arm=arm)
        w.write_csv(OUT / f"worst_20_{tech}.csv")
        print(tech, "rows", rows.height)
        print(
            w.group_by("site", "level").agg(
                pl.col("run").n_unique().alias("runs_months"), pl.col("pin_cf").mean()
            )
        )
    pl.DataFrame(notes, strict=False).write_csv(OUT / "chart_notes.csv")
    print(pl.DataFrame(notes, strict=False))


if __name__ == "__main__":
    main()
