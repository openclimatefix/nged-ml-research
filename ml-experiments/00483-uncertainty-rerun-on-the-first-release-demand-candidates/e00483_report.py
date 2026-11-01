"""Experiment 483: intervals, minimum detectable effect, and figures.

Reads this experiment's cache (`e00483_tails.py cache`), rebuilds the widened and table P frames,
and writes `work/headline.csv` (coverage and pinball loss with month-by-origin-day block bootstrap
intervals), `work/contrasts.csv` (table P minus the 1.15 widening), `work/mde.csv`, and the figures
under `docs/ml-experiments/00483-*/`.

The counted contrasts are table P minus the 1.15 widening on CRPS and q99 pinball loss, for A-pool
at the eight and the six, A-gam-in at the eight, and K2-425A at the eight (4 cells x 2 metrics),
plus K2-425A minus A-pool on q99 pinball loss at the eight, raw and widened by 1.15 (2), so
`TESTS` = 10.

```bash
source /tmp/e483env.sh
uv run --no-sync python ml-experiments/00483-*/e00483_report.py
```
"""

import math
from pathlib import Path
from typing import Final

import altair as alt
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00483_curtailment as curt  # ty: ignore[unresolved-import]
import e00483_tails as tails  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00435_tailcore import widen  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00483-uncertainty-rerun-on-the-first-release-demand-candidates"
)
TESTS: Final[int] = 10
RESAMPLES: Final[int] = 20_000
CELLS: Final[tuple[tuple[str, str], ...]] = (
    ("A-pool", "mva_eight"),
    ("A-pool", "signed_six"),
    ("A-gam-in", "mva_eight"),
    ("K2-425A", "mva_eight"),
)
TARGETS: Final[dict[str, tuple[float, float]]] = {
    "cov_q01": (0.005, 0.02),
    "cov_q05": (0.03, 0.07),
    "cov_q95": (0.93, 0.97),
    "cov_q99": (0.98, 0.995),
}
WIDTH: Final[int] = 640
assert len(CELLS) * 2 + 2 == TESTS


def variant_frame(*, arm: str, variant: str) -> pl.DataFrame:
    """Return clean rows of an arm for `raw`, `w1.15`, or `P`, with row-level metrics attached."""
    frame = curt.adjusted(arm=arm, variant=variant)
    m = score470.row_metrics(frame=frame)
    return frame.with_columns([pl.Series(k, v) for k, v in m.items()])


def block_ci(*, frame: pl.DataFrame, column: str, rng: np.random.Generator, level: float) -> tuple:
    """Return the mean and its percentile interval from month-by-day blocks."""
    blocks = frame.group_by("month", "day").agg(
        pl.len().alias("n"), pl.col(column).sum().alias("s")
    )
    n, s = blocks["n"].to_numpy().astype(float), blocks["s"].to_numpy()
    draws = rng.integers(0, len(n), (RESAMPLES, len(n)))
    boot = s[draws].sum(1) / n[draws].sum(1)
    alpha = 100.0 * (1.0 - level) / 2.0
    return s.sum() / n.sum(), np.percentile(boot, alpha), np.percentile(boot, 100.0 - alpha)


def stats() -> None:
    """Write the headline table with intervals, the contrasts, and the minimum detectable effect."""
    rng = np.random.default_rng(0)
    head, contrasts = [], []
    frames = {(a, v): variant_frame(arm=a, variant=v) for a in tails.ARMS for v in curt.VARIANTS}
    metrics = ["crps", *[f"cov_{t}" for t in ("q01", "q05", "q95", "q99")], "pin_q99"]
    for (arm, variant), frame in frames.items():
        for group in ("mva_eight", "signed_six", "report_only"):
            part = frame.filter(pl.col("group") == group)
            if part.is_empty():
                continue
            for metric in metrics:
                mean, low, high = block_ci(frame=part, column=metric, rng=rng, level=0.95)
                head.append(
                    {
                        "arm": arm,
                        "variant": variant,
                        "group": group,
                        "metric": metric,
                        "n": part.height,
                        "mean": mean,
                        "low": low,
                        "high": high,
                    }
                )
    pl.DataFrame(head).write_csv(tails.WORK / "headline.csv")

    def diff(a: pl.DataFrame, b: pl.DataFrame, group: str, metric: str) -> pl.DataFrame:
        keys = [*build470.KEYS, "month", "day"]
        a1 = a.filter(pl.col("group") == group).select(*keys, x=pl.col(metric))
        b1 = b.filter(pl.col("group") == group).select(*keys, y=pl.col(metric))
        return a1.join(b1, on=keys).with_columns(d=pl.col("x") - pl.col("y"))

    for arm, group in CELLS:
        for metric in ("crps", "pin_q99"):
            d = diff(frames[(arm, "P")], frames[(arm, "w1.15")], group, metric)
            mean, low, high = block_ci(frame=d, column="d", rng=rng, level=1 - 0.05 / TESTS)
            contrasts.append(
                {
                    "contrast": f"{arm}: P minus w1.15",
                    "group": group,
                    "metric": metric,
                    "n": d.height,
                    "difference": mean,
                    "low_bonferroni": low,
                    "high_bonferroni": high,
                }
            )
    for variant in ("raw", "w1.15"):
        d = diff(frames[("K2-425A", variant)], frames[("A-pool", variant)], "mva_eight", "pin_q99")
        mean, low, high = block_ci(frame=d, column="d", rng=rng, level=1 - 0.05 / TESTS)
        contrasts.append(
            {
                "contrast": f"K2-425A minus A-pool ({variant})",
                "group": "mva_eight",
                "metric": "pin_q99",
                "n": d.height,
                "difference": mean,
                "low_bonferroni": low,
                "high_bonferroni": high,
            }
        )
    table = pl.DataFrame(contrasts)
    table.write_csv(tails.WORK / "contrasts.csv")
    print(table.with_columns(pl.col(pl.Float64).round(4)))
    mde()


def mde() -> None:
    """Return the minimum detectable effect from A-pool's five single seeds (widened by 1.15)."""
    base = {m: pl.read_parquet(tails.cache_path(arm="A-pool", month=m)) for m in tails.MONTHS}
    rows = []
    for group in ("mva_eight", "signed_six"):
        for metric in ("crps", "cov_q95", "cov_q99", "pin_q99"):
            values = []
            for month in tails.MONTHS:
                clean = base[month].filter((pl.col("group") == group) & (pl.col("cls") == "clean"))
                per_seed = []
                for seed in build470.POOL_SEEDS:
                    pred = pl.read_parquet(
                        build470.prediction_path(arm=f"A-s{seed}", month=month, cell=0)
                    )
                    joined = clean.select(*build470.KEYS, "actual").join(
                        pred.select(*build470.KEYS, *tails.LEVELS), on=build470.KEYS, how="inner"
                    )
                    joined = widen(frame=joined, factor=1.15)
                    m = score470.row_metrics(frame=joined)
                    per_seed.append(float(np.mean(m[metric])))
                values.append(per_seed)
            v = np.array(values)
            folds, k = v.shape
            resid = v - v.mean(axis=1, keepdims=True)
            sd = math.sqrt((resid**2).sum() / (folds * (k - 1)))
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "seed_sd": sd,
                    "mde": 1.96 * math.sqrt(2.0) * sd / math.sqrt(k),
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(tails.WORK / "mde.csv")
    print(out)


def figures() -> None:
    """Draw coverage by variant, the beyond-q99 share by month, and the worst-20 half-hours."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    head = pl.read_csv(tails.WORK / "headline.csv").filter(
        pl.col("metric").str.starts_with("cov_") & (pl.col("group") != "report_only")
    )
    targets = pl.DataFrame(
        [{"metric": k, "low_t": v[0], "high_t": v[1]} for k, v in TARGETS.items()]
    )
    head = head.join(targets, on="metric").with_columns(
        label=pl.col("arm")
        + ", "
        + pl.col("group").replace({"mva_eight": "eight", "signed_six": "six"})
    )
    base = alt.Chart(head).encode(
        y=alt.Y("variant:N", title=None, sort=["raw", "w1.15", "P"]),
        color=alt.Color(
            "variant:N",
            legend=None,
            scale=alt.Scale(
                domain=["raw", "w1.15", "P"],
                range=[theme.GREY_3, theme.DATA_BLUE, theme.BRAND_ORANGE],
            ),
        ),
    )
    points = base.mark_point(filled=True, size=50).encode(x=alt.X("mean:Q", title="Coverage"))  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    band = (
        alt.Chart(head)
        .mark_rect(opacity=0.15, color=theme.DATA_GREEN)
        .encode(x="low_t:Q", x2="high_t:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        alt.layer(band, bars, points)
        .properties(width=WIDTH // 4, height=70)
        .facet(row=alt.Row("label:N", title=None), column=alt.Column("metric:N", title=None))
        .resolve_scale(x="independent")
    )
    chart.save(FIGURE_DIR / "figure_1_coverage_by_variant.svg")
    beyond = pl.read_parquet(tails.WORK / "beyond.parquet").filter(
        (pl.col("widening").is_in([1.0, 1.15])) & (pl.col("group") != "report_only")
    )
    alt.Chart(beyond.with_columns(widening=pl.col("widening").cast(pl.String))).mark_line(
        point=True
    ).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("month:N", title="Fold"),
        y=alt.Y("above_q99:Q", title="Share of actuals above q99"),
        color=alt.Color("arm:N", title=None),
        strokeDash="widening:N",
    ).properties(width=WIDTH // 2, height=140).facet(column="group:N").save(
        FIGURE_DIR / "figure_2_above_q99_by_month.svg"
    )
    worst = pl.read_parquet(tails.WORK / "worst20.parquet").filter(
        (pl.col("arm") == "A-pool") & (pl.col("widening") == 1.15)
    )
    alt.Chart(worst.with_columns(rank=pl.int_range(pl.len()) + 1)).mark_bar(
        color=theme.BRAND_ORANGE
    ).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("rank:O", title="Rank by q99 pinball loss"),
        y=alt.Y("pin_q99:Q", title="q99 pinball loss (points)"),
        tooltip=["month", "lead_hours", "actual"],
    ).properties(width=WIDTH // 2, height=140).save(FIGURE_DIR / "figure_3_worst20_pinball.svg")
    context_charts(worst=worst)


def context_charts(*, worst: pl.DataFrame) -> None:
    """Draw each of the worst 12 half-hours as a forecast from its origin with the target marked."""
    labels = dict(enumerate(tails.prov.LABELS))
    panels = []
    full = tails.load(arm="A-pool")
    for i, row in enumerate(worst.head(12).iter_rows(named=True), start=1):
        run = full.filter(
            (pl.col("primary") == row["primary"]) & (pl.col("origin") == row["origin"])
        ).sort("target")
        run = widen(frame=run, factor=1.15)
        pdata = run.select("valid_time", "actual", "q05", "q50", "q95", "q99").with_columns(
            [pl.col(c) * 100.0 for c in ("actual", "q05", "q50", "q95", "q99")]
        )
        title = f"{i}. {labels[int(row['primary'])]}, lead {row['lead_hours']:.0f} h"
        x = alt.X("valid_time:T", title=None)
        band = (
            alt.Chart(pdata)
            .mark_area(opacity=0.25, color=theme.DATA_BLUE)
            .encode(x=x, y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
        )
        line = alt.Chart(pdata).mark_line(color=theme.DATA_BLUE).encode(x=x, y="q50:Q")  # ty: ignore[unresolved-attribute]
        q99 = (
            alt.Chart(pdata)
            .mark_line(color=theme.DATA_BLUE, strokeDash=[3, 3])
            .encode(x=x, y="q99:Q")  # ty: ignore[unresolved-attribute]
        )
        act = (
            alt.Chart(pdata)
            .mark_line(color=theme.BLACK_1)
            .encode(x=x, y=alt.Y("actual:Q", title="% of capacity"))  # ty: ignore[unresolved-attribute]
        )
        mark = (
            alt.Chart(pl.DataFrame({"t": [row["valid_time"]]}))
            .mark_rule(color=theme.BRAND_ORANGE)
            .encode(x="t:T")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(band, q99, line, act, mark).properties(
                width=WIDTH // 2, height=110, title=title
            )
        )
    rows = [alt.hconcat(*panels[j : j + 2]) for j in range(0, len(panels), 2)]
    alt.vconcat(*rows).save(FIGURE_DIR / "figure_4_worst20_context.svg")


if __name__ == "__main__":
    stats()
    figures()
