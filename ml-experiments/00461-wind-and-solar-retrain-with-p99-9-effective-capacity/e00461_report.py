"""Experiment 461: draw the figures and print the capacity and interval tables for the write-up.

Reads `work/cells.parquet` and `work/rows_*.parquet` from `e00461_score.py`. Writes SVG figures to
the experiment's docs folder (Farm A, Solar A, and Solar B only) and, for every series, the week
fans and the 20 worst half-hours by q99 pinball loss to `work/private_charts/` (the private
store). No chart carries a calendar date: the x-axes count hours after the run was issued.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00429-*/env.sh   # plus the 00461 folder
uv run --no-sync python ml-experiments/00461-*/e00461_report.py
```
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00461_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00461-wind-and-solar-retrain-with-p99-9-effective-capacity"
)
WORK: Final[Path] = common.DATA_DIR / "work"
PRIVATE: Final[Path] = WORK / "private_charts"
SERIES: Final[list[str]] = [
    "Farm A", "Farm B", "Farm C", "Solar A", "Solar B", "Solar C", "Solar D", "Solar E", "Solar F",
]  # fmt: skip
DRAWS: Final[int] = 2000
WIDTH: Final[int] = 640
OLD_COLOUR, NEW_COLOUR = ocf.DATA_BLUE, ocf.BRAND_ORANGE


def headline_arm(*, site: str) -> str:
    """Return the headline arm of a series' technology."""
    return "member_plain" if site.startswith("Farm") else "a2_member"


def month_block(
    *, cells: pl.DataFrame, site: str, arm: str, rule: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return (point, low, high) of the share above q99 (%) after table P, month-block."""
    sub = cells.filter(
        (pl.col("site") == site) & (pl.col("arm") == arm) & (pl.col("rule") == rule)
        & (pl.col("variant") == "P")
    ).sort("month")  # fmt: skip
    above = (sub["n"] - sub["hit_q99"]).to_numpy().astype(float)
    n = sub["n"].to_numpy().astype(float)
    draws = np.empty(DRAWS)
    for d in range(DRAWS):
        pick = rng.integers(0, len(n), size=len(n))
        draws[d] = above[pick].sum() / n[pick].sum() * 100
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(above.sum() / n.sum() * 100), float(low), float(high)


def contrast(
    *, cells: pl.DataFrame, site: str, arm: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return (percent change, low, high) of the new minus old score after table P, month-block."""
    sub = cells.filter(
        (pl.col("site") == site) & (pl.col("arm") == arm) & (pl.col("variant") == "P")
    )
    wide = {rule: sub.filter(pl.col("rule") == rule).sort("month") for rule in ("old", "p999")}
    s_old, n = wide["old"]["score_mw_sum"].to_numpy(), wide["old"]["n"].to_numpy().astype(float)
    s_new = wide["p999"]["score_mw_sum"].to_numpy()
    point = (s_new.sum() - s_old.sum()) / s_old.sum() * 100
    draws = np.empty(DRAWS)
    for d in range(DRAWS):
        pick = rng.integers(0, len(n), size=len(n))
        draws[d] = (s_new[pick].sum() - s_old[pick].sum()) / s_old[pick].sum() * 100
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def fig_share_and_score(*, cells: pl.DataFrame) -> None:
    """Figures 1 and 2: share above q99 after table P, and the score contrast, per series."""
    rng = np.random.default_rng(0)
    share_rows, score_rows = [], []
    for site in SERIES:
        arm = headline_arm(site=site)
        for rule, label in (
            ("old", "old (p99 of whole record)"),
            ("p999", "p99.9 of earlier power"),
        ):
            p, lo, hi = month_block(cells=cells, site=site, arm=arm, rule=rule, rng=rng)
            share_rows.append({"series": site, "rule": label, "share": p, "low": lo, "high": hi})
        p, lo, hi = contrast(cells=cells, site=site, arm=arm, rng=rng)
        score_rows.append({"series": site, "change": p, "low": lo, "high": hi})
    share = pl.DataFrame(share_rows)
    print(share)
    print(pl.DataFrame(score_rows))
    colour = alt.Color(
        "rule:N",
        title=None,
        scale=alt.Scale(
            domain=["old (p99 of whole record)", "p99.9 of earlier power"],
            range=[OLD_COLOUR, NEW_COLOUR],
        ),
        legend=alt.Legend(orient="top"),
    )
    base = alt.Chart(share).encode(y=alt.Y("series:N", sort=SERIES, title=None))
    limit = alt.Chart(pl.DataFrame({"x": [1.5]})).mark_rule(strokeDash=[4, 3]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q", title="Share of actuals above q99 (%; smaller is better; aim at most 1.5%)"
            ),
            x2="high:Q",
            color=colour,
            yOffset=alt.YOffset("rule:N"),
        )
        + base.mark_circle(size=60, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x="share:Q", color=colour, yOffset=alt.YOffset("rule:N")
        )
        + limit
    ).properties(
        width=WIDTH - 90,
        height=230,
        title=[
            "Share of actuals above q99 after table P, true days 1 to 10 (wind: Farm A to C,",
            "member_plain; solar: Solar A to F, a2_member). Dot: estimate. Line: 95% interval",
            "from resampling whole months. Dashed rule: the 1.5% limit.",
        ],
    )
    chart.save(DOCS / "figure_1_share_above_q99.svg")
    score = pl.DataFrame(score_rows)
    rule0 = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule().encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    margin = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))
        .mark_rule(strokeDash=[4, 3], color=ocf.GREY_1)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )  # ty: ignore[unresolved-attribute]
    base = alt.Chart(score).encode(y=alt.Y("series:N", sort=SERIES, title=None))
    (
        base.mark_rule(strokeWidth=2, color=NEW_COLOUR).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="New minus old quantile score (% of the old score; negative is better)",
            ),
            x2="high:Q",
        )
        + base.mark_circle(size=60, opacity=1, color=NEW_COLOUR).encode(x="change:Q")  # ty: ignore[unresolved-attribute]
        + rule0
        + margin
    ).properties(
        width=WIDTH - 90,
        height=200,
        title=[
            "Change in quantile score in MW after table P: p99.9 capacity minus old capacity.",
            "Dot: estimate. Line: 95% interval from resampling whole months. Solid rule: no",
            "change. Dashed rule: the solar margin of 1% (the wind margin is 0.10 points of",
            "capacity, about 0.1% of the score).",
        ],
    ).save(DOCS / "figure_2_score_change.svg")


def pick_run(*, rows: pl.DataFrame, site: str, month_prefix: str | None) -> pl.DataFrame:
    """Return one run's rows (true days 1 to 10) of a series, optionally from a month."""
    part = rows.filter(pl.col("site") == site)
    if month_prefix:
        part = part.filter(pl.col("month").str.starts_with(month_prefix))
    counts = part.group_by("init").agg(pl.len().alias("n")).sort("n", descending=True)
    full = counts.filter(pl.col("n") == counts["n"].max()).sort("init")["init"]
    init = full[len(full) // 2]
    return part.filter(pl.col("init") == init).sort("lead_hours")


def fan(
    *, run: pl.DataFrame, lo_h: float, hi_h: float, title: list[str], height: int = 140
) -> alt.LayerChart:
    """Return a fan chart (q01 to q99, q05 to q95, q25 to q75, median, actual) in MW."""
    part = run.filter(pl.col("lead_hours").is_between(lo_h, hi_h))
    base = alt.Chart(part).encode(
        x=alt.X(
            "lead_hours:Q",
            scale=alt.Scale(domain=[lo_h, hi_h], nice=False),
            title="Hours after the run was issued",
        )
    )

    def band(low: str, high: str, opacity: float) -> alt.Chart:
        return base.mark_area(color=NEW_COLOUR, opacity=opacity, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(f"{low}:Q", title="Power (MW)"), y2=f"{high}:Q"
        )

    return (
        band("q01", "q99", 0.15)
        + band("q05", "q95", 0.2)
        + band("q25", "q75", 0.3)
        + base.mark_line(color=NEW_COLOUR, aria=False).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_line(color="black", aria=False).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH - 80, height=height, title=title)


def fig_examples(*, rows: dict[str, pl.DataFrame]) -> None:
    """Figures 3 and 4: a day-ahead forecast and a 10-day run for Farm A and Solar A."""
    spec = [("Farm A", "2026-01", "winter"), ("Farm A", "2025-07", "summer"),
            ("Solar A", "2026-06", "summer"), ("Solar A", "2025-12", "winter")]  # fmt: skip
    day, horizon = [], []
    for site, month, season in spec:
        frame = rows["wind" if site.startswith("Farm") else "solar"]
        run = pick_run(rows=frame, site=site, month_prefix=month[:4] + "-" + month[5:])
        label = f"{site}, one {season} run"
        day.append(
            fan(run=run, lo_h=24, hi_h=48, title=[f"{label}: day ahead (true day 1)"], height=110)
        )
        horizon.append(
            fan(run=run, lo_h=24, hi_h=240, title=[f"{label}: true days 1 to 10"], height=110)
        )
    note = [
        "Line: actual. Orange line: median. Bands: 50%, 90%, and 98% (q01 to q99) of the new",
        "arm before table P. Runs start at the hour shown; no dates are drawn.",
    ]
    for name, charts in (
        ("figure_3_day_ahead.svg", day),
        ("figure_4_ten_day_horizon.svg", horizon),
    ):
        alt.vconcat(*charts).properties(title=note).save(DOCS / name)


def private_charts(*, rows: dict[str, pl.DataFrame]) -> None:
    """Write each series' high-actual and low-actual week fans and worst 20 half-hours."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    worst_all = []
    for site in SERIES:
        frame = rows["wind" if site.startswith("Farm") else "solar"].filter(pl.col("site") == site)
        week = frame.filter(pl.col("lead_hours").is_between(24, 192))
        per_run = week.group_by("init").agg(
            pl.col("actual").mean().alias("mean_actual"), pl.len().alias("n")
        )
        per_run = per_run.filter(pl.col("n") >= float(per_run["n"].to_numpy().max()) * 0.9)
        picks = {
            "high": per_run.sort("mean_actual")["init"][-1],
            "low": per_run.sort("mean_actual")["init"][0],
        }
        charts = [
            fan(
                run=frame.filter(pl.col("init") == init).sort("lead_hours"),
                lo_h=24,
                hi_h=192,
                title=[f"{site}: run with the {kind} mean actual, true days 1 to 7"],
            )
            for kind, init in picks.items()
        ]
        alt.vconcat(*charts).save(PRIVATE / f"fans_{site.replace(' ', '_')}.svg")
        pin = (
            frame.with_columns(
                loss=pl.when(pl.col("actual") >= pl.col("q99"))
                .then(0.99 * (pl.col("actual") - pl.col("q99")))
                .otherwise(-0.01 * (pl.col("actual") - pl.col("q99")))
            )
            .sort("loss", descending=True)
            .head(20)
            .select("site", "init", "lead_hours", "actual", "q50", "q99", "loss")
        )
        worst_all.append(pin)
    pl.concat(worst_all).write_csv(PRIVATE / "worst_20_by_q99_pinball.csv")


def print_tables(*, cells: pl.DataFrame) -> None:
    """Print the capacity tables and the rows behind them."""
    wind = json.loads((common.DATA_DIR / "wind_capacity.json").read_text())
    print("wind capacity ratio (new / old):", {k: round(v["ratio"], 4) for k, v in wind.items()})
    solar = json.loads((common.DATA_DIR / "solar_capacity.json").read_text())
    for refit, table in solar.items():
        print(refit, {s: (round(v["ratio"], 3), v["n_history"]) for s, v in table.items()})
    rows = cells.filter(
        (pl.col("variant") == "P")
        & (pl.col("rule") == "old")
        & pl.col("arm").is_in(["member_plain", "a2_member"])
    )
    print("rows per series:", rows.group_by("site").agg(pl.col("n").sum()).sort("site"))


def main() -> None:
    """Draw every figure."""
    cells = pl.read_parquet(WORK / "cells.parquet")
    rows = {
        "wind": pl.read_parquet(WORK / "rows_member_plain_p999.parquet"),
        "solar": pl.read_parquet(WORK / "rows_a2_member_p999.parquet"),
    }
    print_tables(cells=cells)
    fig_share_and_score(cells=cells)
    fig_examples(rows=rows)
    private_charts(rows=rows)


if __name__ == "__main__":
    main()
