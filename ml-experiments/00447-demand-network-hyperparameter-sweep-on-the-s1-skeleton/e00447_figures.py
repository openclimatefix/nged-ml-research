# ruff: noqa: E501
r"""Draw the figures of experiment 447 from the stored scores and forecasts, and print the audit checks.

- Figure 1: each arm minus S1 (CRPS points, 95% intervals) on clean rows, pooled 14, with the margin.
- Figure 2: the same on injected harm and real harm rows.
- Figure 3: each arm's gain in each of the three folds, clean pooled rows.
- Figures 4 and 5: a day-ahead and a 10-day forecast of one Primary, S1 and the widest network.
- Checks: the error of the median forecast and the median minus mean forecast, S1 and the widest arm.

```bash
PYTHONPATH=<the folders of e00447_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00447-*/e00447_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_score as score  # ty: ignore[unresolved-import]
import e00447_common as common
import e00447_run as run
import numpy as np
import polars as pl
from plotting import ocf_theme as theme  # noqa: F401

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / common.SLUG
)
SCORES: Final[Path] = run.WORK_DIR / "scores"
WIDTH: Final[int] = 700
POINTS: Final[float] = 100.0
EXAMPLE_ARMS: Final[list[str]] = ["S1", "X-w256"]
COLOURS: Final[list[str]] = ["#306BFF", "#FF4901"]


def figure_contrasts(*, stage: str) -> None:
    """Figures 1 and 2: arm minus S1 with 95% intervals."""
    c = pl.read_parquet(SCORES / stage / "decisions.parquet").drop("fold_differences")
    margin = float(c["margin"].max())
    for number, row_sets, name in (
        (1, ["clean"], "clean"), (2, ["inj_harm", "real_harm"], "harm"),
    ):  # fmt: skip
        d = c.filter(pl.col("row_set").is_in(row_sets), pl.col("group") != "D15 and D18")
        d = d.with_columns(gain=pl.col("gain") * 1.0, arm=pl.col("arm").str.replace("X-", ""))
        order = (
            c.filter((pl.col("row_set") == "clean") & (pl.col("group") == "pooled 14"))
            .sort("gain", descending=True)["arm"]
            .str.replace("X-", "")
            .to_list()
        )
        base = alt.Chart(d.to_pandas())
        y = alt.Y("arm:N", sort=order, title=None)
        colour = alt.Color("group:N", legend=alt.Legend(title=None, orient="top"))
        points = base.mark_point(filled=True, size=40).encode(
            x=alt.X("gain:Q", title="Gain over S1 (CRPS points; positive is better)"),
            y=y, color=colour, yOffset="group:N",
        )  # fmt: skip
        rules = base.mark_rule().encode(
            x="gain_low:Q", x2="gain_high:Q", y=y, color=colour, yOffset="group:N"
        )  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0, margin]}).to_pandas())
            .mark_rule(strokeDash=[3, 3], color="#333")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        layered = (rules + points + zero).properties(width=WIDTH - 150)  # ty: ignore[unresolved-attribute]
        chart = (
            layered.facet(row=alt.Row("row_set:N", title=None)) if len(row_sets) > 1 else layered
        )
        chart.save(FIGURE_DIR / f"figure_{number}_{name}.svg")  # ty: ignore[unresolved-attribute]


def figure_folds(*, stage: str, name: str) -> None:
    """Figure 3: gain in each fold on clean pooled rows."""
    c = pl.read_parquet(SCORES / stage / "decisions.parquet")
    c = c.filter((pl.col("row_set") == "clean") & (pl.col("group") == "pooled 14"))
    rows = [
        {"arm": r["arm"].replace("X-", ""), "fold": m, "gain": -v}
        for r in c.iter_rows(named=True)
        for m, v in r["fold_differences"].items()
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows).to_pandas())
        .mark_point(filled=True, size=50)
        .encode(
            x=alt.X("gain:Q", title="Gain over S1 (CRPS points; positive is better)"),
            y=alt.Y("arm:N", title=None), color=alt.Color("fold:N", legend=alt.Legend(title=None, orient="top")),
        )
    )  # fmt: skip
    chart.properties(width=WIDTH - 150).save(FIGURE_DIR / name)  # ty: ignore[unresolved-attribute]


def figure_examples(*, month: str, site: str, seeds: tuple[int, ...]) -> None:
    """Figures 4 and 5: a day-ahead and a 10-day forecast from one origin."""
    primary = data.sites().index(site)
    frames = [
        score.read_arm(arm=a, month=month, world=0, seeds=seeds, factor=score.WIDENING)
        .filter(pl.col("primary") == primary)
        .with_columns(arm=pl.lit(a))
        for a in EXAMPLE_ARMS
    ]
    f = pl.concat(frames)
    origins = sorted(f["origin_us"].unique().to_list())
    one = f.filter(pl.col("origin_us") == origins[len(origins) // 2])
    for number, name, hours in ((4, "day_ahead", 48.0), (5, "10_day", 240.0)):
        d = one.filter(pl.col("lead") <= hours).with_columns(
            time=pl.from_epoch("valid_us", time_unit="us"),
            q05=pl.col("q05") * POINTS, q50=pl.col("q50") * POINTS, q95=pl.col("q95") * POINTS,
            actual=pl.col("actual") * POINTS,
        )  # fmt: skip
        colour = alt.Color(
            "arm:N", scale=alt.Scale(domain=EXAMPLE_ARMS, range=COLOURS),
            legend=alt.Legend(title=None, orient="top"),
        )  # fmt: skip
        x = alt.X("time:T", title="Valid time (UTC)")
        pdata = d.to_pandas()
        band = (
            alt.Chart(pdata)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.12)
            .encode(
                x=x,
                y=alt.Y("q05:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False)),
                y2="q95:Q",
                color=colour,
            )
        )  # ty: ignore[unresolved-attribute]
        median = alt.Chart(pdata).mark_line().encode(x=x, y="q50:Q", color=colour)  # ty: ignore[unresolved-attribute]
        truth = (
            alt.Chart(d.filter(pl.col("arm") == "S1").to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_line(color="#333", strokeDash=[2, 2])
            .encode(x=x, y="actual:Q")
        )  # ty: ignore[unresolved-attribute]
        chart = (band + median + truth).properties(
            width=WIDTH - 90, height=220,
            title=alt.TitleParams(
                [f"{name.replace('_', '-')} forecast of Primary {site}, origin {d['time'].min():%Y-%m-%d}", "Line: median. Band: 5% to 95%. Dashed: the cleaned reading."],
                anchor="start",
            ),
        )  # fmt: skip
        chart.save(FIGURE_DIR / f"figure_{number}_example_{name}.svg")


def checks(*, months: tuple[str, ...], seeds: tuple[int, ...], arms: list[str]) -> None:
    """Print the median-forecast error and the median minus mean forecast on clean rows."""
    rows = []
    for arm in arms:
        for month in months:
            f = score.read_arm(arm=arm, month=month, world=0, seeds=seeds, factor=score.WIDENING)
            keep = score.row_mask(frame=f, rule=score.ROW_SETS["clean"][1])
            q = f.select(list(pr.LEVEL_NAMES)).to_numpy()
            sites = np.array(data.sites())[f["primary"].to_numpy()]
            rows.append(
                pl.DataFrame(
                    {
                        "arm": arm, "site": sites, "keep": keep,
                        "abs_median": np.abs(f["actual"].to_numpy() - f["q50"].to_numpy()) * POINTS,
                        "median_minus_mean": (f["q50"].to_numpy() - q.mean(axis=1)) * POINTS,
                    }
                ).filter(pl.col("keep") & pl.col("abs_median").is_finite())
            )  # fmt: skip
    t = pl.concat(rows).with_columns(
        group=pl.when(pl.col("site").is_in(list(score.SIGNED_MW_LABELS)))
        .then(pl.lit("signed-MW six"))
        .when(pl.col("site").is_in(list(score.MVA_ONLY_LABELS)))
        .then(pl.lit("MVA-only eight"))
        .otherwise(pl.lit("D15 and D18"))
    )
    with pl.Config(tbl_rows=60):
        print(
            t.group_by("group", "arm")
            .agg(pl.col("abs_median").mean().alias("mae_q50"), pl.col("median_minus_mean").mean())
            .sort("group", "arm")
        )


def main() -> None:
    """Draw every figure and print the checks."""
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts(stage="screen")
    figure_folds(stage="screen", name="figure_3_folds.svg")
    figure_folds(stage="confirm", name="figure_6_confirmation_folds.svg")
    figure_examples(month="2026-01", site=score.SIGNED_MW_LABELS[0], seeds=common.SCREEN_SEEDS)
    checks(months=common.SCREEN_FOLDS, seeds=common.SCREEN_SEEDS, arms=["S1", "X-w256"])


if __name__ == "__main__":
    main()
