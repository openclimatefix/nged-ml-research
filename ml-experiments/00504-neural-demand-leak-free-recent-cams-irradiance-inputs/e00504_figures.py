# ruff: noqa: E501
"""Draw the figures of experiment 504 from the tables `e00504_score.py` and `e00504_report.py` stored.

Figures (SVG, in the experiment's docs folder; optimise each with
`npx svgo@4 --multipass --precision=1 --final-newline`):

1. `figure_1_headline.svg`: each arm's CRPS difference to K3 at the eight and the six, with the
   origin-day Bonferroni interval, the interval across the folds, and the minimum detectable effect;
2. `figure_2_per_fold.svg`: the same difference in each fold;
3. `figure_3_tail_calibration.svg`: coverage at q01, q05, q95, and q99;
4. `figure_4_autocorrelation.svg`: the correlation between the daily cloudiness of two days;
5. `figure_5_example_*.svg`: a day-ahead and a 14-day forecast of K3, A-cams, and A-cams-real.

```bash
source <paths>
uv run --no-sync python ml-experiments/00494-*/e00494_figures.py
```
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00407_score as s407  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00504_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from scipy import stats

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00504-neural-demand-leak-free-recent-cams-irradiance-inputs"
)
WIDTH: Final[int] = 600
ARMS: Final[dict[str, str]] = {
    "A-cams": "Leak-free columns (A-cams)",
    "A-cams-lag-only": "Cloudiness column only (A-cams-lag-only)",
    "A-cams-real": "Realised target-time columns (A-cams-real, upper bound)",
}
COLOURS: Final[dict[str, str]] = {
    "K3": theme.BRAND_ORANGE,
    "A-cams": theme.DATA_BLUE,
    "A-cams-lag-only": theme.DATA_PURPLE,
    "A-cams-real": theme.DATA_GREEN,
}
GROUPS: Final[dict[str, str]] = {"eight": "MVA-only eight", "six_clean": "Signed-MW six"}
EXAMPLES: Final[dict[str, tuple[str, str]]] = {
    "D4": ("MVA-only Primary", "2026-01-12"),
    "D12": ("MVA-only Primary", "2026-06-10"),
}


def contrast_table() -> pl.DataFrame:
    """Return each arm's CRPS contrast at the eight and the six, with the MDE and fold intervals."""
    rows = []
    folder = score.WORK_DIR / "scores" / "main"
    mde = pl.read_csv(folder / "mde_k3.csv")
    mdes = {"eight": mde["mde"][0], "six_clean": mde["mde"][1]}
    contrasts = pl.read_csv(folder / "contrasts.csv")
    for arm in ARMS:
        for sname in GROUPS:
            row = contrasts.filter(
                (pl.col("contrast") == f"{arm} - K3")
                & (pl.col("slice") == sname)
                & (pl.col("metric") == "crps")
            ).row(0, named=True)
            folds = list(json.loads(row["by_month"]).values())
            half = stats.t.ppf(0.975, len(folds) - 1) * np.std(folds, ddof=1) / np.sqrt(len(folds))
            rows.append(
                {
                    "arm": ARMS[arm], "group": GROUPS[sname], "diff": row["difference"],
                    "low": row["low_bonferroni"], "high": row["high_bonferroni"],
                    "fold_low": float(np.mean(folds) - half), "fold_high": float(np.mean(folds) + half),
                    "limit": -mdes[sname],
                }
            )  # fmt: skip
    return pl.DataFrame(rows)


def figure_headline() -> None:
    """Draw the CRPS difference to K3 with both intervals and the equivalence limit."""
    table = contrast_table().to_pandas()  # ty: ignore[unresolved-attribute]
    order = list(ARMS.values())
    colour = alt.Color(
        "arm:N",
        sort=order,
        scale=alt.Scale(domain=order, range=[COLOURS[a] for a in ARMS]),
        legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=500),
    )
    y = alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False))
    panels = []
    for label in GROUPS.values():
        part = table[table["group"] == label]
        base = alt.Chart(part).encode(y=y)  # ty: ignore[unresolved-attribute]
        fold = base.mark_rule(strokeWidth=2, opacity=0.35).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "fold_low:Q",
                title="CRPS minus K3 (points of effective capacity; negative is better)",
            ),
            x2="fold_high:Q",
            color=colour,
        )
        interval = base.mark_rule(strokeWidth=5).encode(x="low:Q", x2="high:Q", color=colour)  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=90, opacity=1).encode(x="diff:Q", color=colour)  # ty: ignore[unresolved-attribute]
        limit = base.mark_tick(color="black", thickness=2, size=22).encode(x="limit:Q")  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        panels.append(
            alt.layer(zero, fold, interval, dot, limit).properties(
                width=WIDTH, height=70, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "CRPS difference to the stored three-seed pool (K3), five folds, rows with no",
                "labelled event at the six. Dot: estimate. Thick line: Bonferroni interval over",
                "origin days. Thin line: 95% interval over folds. Black tick: minus the minimum",
                "detectable effect, the gain the gate needs. Dashed line: same as K3.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def figure_per_fold() -> None:
    """Draw each arm's CRPS difference to K3 in each fold, at the eight and the six."""
    rows = []
    contrasts = pl.read_csv(score.WORK_DIR / "scores" / "main" / "contrasts.csv")
    for name in ARMS:
        for sname, label in GROUPS.items():
            row = contrasts.filter(
                (pl.col("contrast") == f"{name} - K3")
                & (pl.col("slice") == sname)
                & (pl.col("metric") == "crps")
            ).row(0, named=True)
            for month, value in json.loads(row["by_month"]).items():
                rows.append({"arm": ARMS[name], "group": label, "fold": month, "diff": value})
    table = pl.DataFrame(rows).to_pandas()  # ty: ignore[unresolved-attribute]
    order = list(ARMS.values())
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X("fold:N", title="Scored month (fold)"),
            y=alt.Y("diff:Q", title="CRPS minus K3 (points; negative is better)"),
            color=alt.Color(
                "arm:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[COLOURS[a] for a in ARMS]),
                legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=500),
            ),
            xOffset=alt.XOffset("arm:N", sort=order),
        )
        .properties(width=WIDTH, height=150)
        .facet(
            row=alt.Row(
                "group:N",
                sort=list(GROUPS.values()),
                title=None,
                header=alt.Header(labelAnchor="start", labelAngle=0),
            ),
            title=alt.TitleParams(
                [
                    "Fold-by-fold CRPS difference to K3: the leak-free arms sit near zero in",
                    "every fold; the realised arm gains in every fold at the eight.",
                ],
                anchor="start",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_per_fold.svg")


def figure_tails() -> None:
    """Draw coverage at q01, q05, q95, and q99 for K3 and the three arms, one panel per group."""
    rows = []
    absolute = pl.read_csv(score.WORK_DIR / "scores" / "main" / "absolute.csv")
    for name in ARMS:
        for arm in (name, "K3"):
            for sname, label in GROUPS.items():
                row = absolute.filter(
                    (pl.col("slice") == sname)
                    & (pl.col("arm") == arm)
                    & (pl.col("rowset") == "main16")
                ).row(0, named=True)
                rows.extend(
                    {
                        "series": "K3" if arm == "K3" else ARMS[name],
                        "group": label,
                        "level": level,
                        "coverage": row[f"cov_{level}"],
                    }
                    for level in ("q01", "q05", "q95", "q99")
                )
    table = pl.DataFrame(rows).unique(subset=["series", "group", "level"], keep="first").to_pandas()  # ty: ignore[unresolved-attribute]
    order = ["K3", *ARMS.values()]
    nominal = pl.DataFrame(
        {"level": ["q01", "q05", "q95", "q99"], "nominal": [0.01, 0.05, 0.95, 0.99]}
    ).to_pandas()  # ty: ignore[unresolved-attribute]
    panels = []
    for label in GROUPS.values():
        points = (
            alt.Chart(table[table["group"] == label])  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60)
            .encode(
                x=alt.X("level:N", sort=["q01", "q05", "q95", "q99"], title="Quantile level"),
                y=alt.Y("coverage:Q", title="Share of actuals below the quantile"),
                color=alt.Color(
                    "series:N",
                    sort=order,
                    scale=alt.Scale(domain=order, range=[COLOURS[a] for a in COLOURS]),
                    legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=500),
                ),
                xOffset=alt.XOffset("series:N", sort=order),
            )
        )
        rule = (
            alt.Chart(nominal)  # ty: ignore[unresolved-attribute]
            .mark_tick(color="black", thickness=2, size=WIDTH / 5)
            .encode(x=alt.X("level:N"), y="nominal:Q")
        )
        panels.append(
            (rule + points).properties(
                width=WIDTH, height=160, title=alt.TitleParams(label, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Tail calibration over the five folds, widened by 1.15: dot is the share of",
                "actuals below the quantile; black tick is the nominal level.",
            ],
            anchor="start",
        )
    )
    chart.save(DOCS_DIR / "figure_3_tail_calibration.svg")


def figure_autocorrelation() -> None:
    """Draw the correlation between the daily cloudiness of two days `lag` days apart."""
    table = pl.read_csv(score.WORK_DIR / "scores" / "report" / "autocorrelation.csv").to_pandas()  # ty: ignore[unresolved-attribute]
    mean = (
        pl.read_csv(score.WORK_DIR / "scores" / "report" / "autocorrelation.csv")
        .group_by("lag_days")
        .agg(pl.col("correlation").mean())
        .sort("lag_days")
        .to_pandas()  # ty: ignore[unresolved-attribute]
    )
    x = alt.X("lag_days:O", title="Days between the two days")
    sites = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=30, opacity=0.4, color=theme.DATA_BLUE)
        .encode(x=x, y=alt.Y("correlation:Q", title="Correlation of daily cloudiness"))
    )
    line = (
        alt.Chart(mean)  # ty: ignore[unresolved-attribute]
        .mark_line(color=theme.BRAND_ORANGE, strokeWidth=3, point=True)
        .encode(x=x, y="correlation:Q")
    )
    chart = (sites + line).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            [
                "Satellite cloudiness three days earlier says little about today: the daily",
                "correlation falls from 0.25 at one day to 0.16 at three days. Blue: each of the",
                "16 Primaries. Orange: their mean. July 2023 to June 2026.",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "figure_4_autocorrelation.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of K3, A-cams, and A-cams-real against the actuals."""
    series = adapter.series_table()
    ctx = prep425.cleaned_scoring_context(series=series)
    sites = build470.site_map()
    ids = dict(zip(series["label"].to_list(), series["time_series_id"].to_list(), strict=True))
    for label, (kind, origin_date) in EXAMPLES.items():
        month = origin_date[:7]
        frames = {}
        for arm in ("K3", "A-cams", "A-cams-real"):
            pooled = build470.pool_stored(
                folder=score.ARM_FOLDERS[arm], month=month, sites=sites, seeds=score.SEEDS
            )
            frames[arm] = adapter.forecast_frame(
                stored=s407.widen_global(frame=pooled.drop("primary", "origin", "target")),
                series=series,
                unit="fraction",
            )
        origin = datetime.fromisoformat(origin_date).replace(hour=9, tzinfo=UTC)
        one = {
            arm: frame.filter(
                pl.col("time_series_id") == ids[label], pl.col("power_fcst_init_time") == origin
            )
            .with_columns(day=(pl.col("valid_time") - origin).dt.total_minutes() / 1440.0)
            .sort("valid_time")
            for arm, frame in frames.items()
        }
        actual = (
            ctx["actuals"]
            .filter(pl.col("time_series_id") == ids[label])
            .join(one["K3"].select("valid_time", "day"), on="valid_time")
        )
        season = "January 2026" if month == "2026-01" else "June 2026"
        for horizon, window in (("day-ahead", (1.0, 2.0)), ("14 days", (0.0, 14.0))):
            x = alt.X(
                "day:Q",
                title="Days after the forecast was made",
                scale=alt.Scale(domain=list(window), nice=False),
            )
            k3 = one["K3"].filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            band = alt.Chart(k3).encode(x=x)
            layers = [
                band.mark_area(opacity=0.2, color=COLOURS["K3"]).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("q05:Q", title="Power (MW)", scale=alt.Scale(zero=False)), y2="q95:Q"
                ),
                band.mark_area(opacity=0.35, color=COLOURS["K3"]).encode(y="q25:Q", y2="q75:Q"),  # ty: ignore[unresolved-attribute]
            ]
            for arm in ("K3", "A-cams", "A-cams-real"):
                sub = one[arm].filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
                layers.append(
                    alt.Chart(sub)  # ty: ignore[unresolved-attribute]
                    .mark_line(color=COLOURS[arm], strokeWidth=2)
                    .encode(x=x, y="q50:Q")
                )
            act = actual.filter(pl.col("day").is_between(*window)).to_pandas()  # ty: ignore[unresolved-attribute]
            layers.append(
                alt.Chart(act).mark_line(color="black", strokeWidth=1.5).encode(x=x, y="actual:Q")  # ty: ignore[unresolved-attribute]
            )
            chart = alt.layer(*layers).properties(
                width=WIDTH,
                height=200,
                title=alt.TitleParams(
                    [
                        f"{label} ({kind}), {horizon} forecast made in {season}, widened 1.15.",
                        "Bands: K3's 50% and 90%. Median lines: K3 orange, leak-free columns",
                        "blue, realised columns green. Black: what happened.",
                    ],
                    anchor="start",
                ),
            )
            chart.save(
                DOCS_DIR
                / f"figure_5_example_{label}_{horizon.replace('-', '_').replace(' ', '_')}.svg"
            )


def main() -> None:
    """Draw every figure."""
    figure_headline()
    figure_per_fold()
    figure_tails()
    figure_autocorrelation()
    figure_examples()


if __name__ == "__main__":
    main()
