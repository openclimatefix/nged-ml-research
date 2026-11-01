# ruff: noqa: E501
"""Draw the figures of experiment 436 from the files the scoring and diagnostic scripts wrote.

1. `figure_1_variogram_contrasts.svg`: each construction's variogram score relative to the
   independent draws, per block, with month-block intervals and the coherence margin;
2. `figure_2_dependence.svg`: the Gaussian-score correlations of each construction against the
   realised ones;
3. `figure_3_rank_histograms.svg`: average-rank histograms of the spatial wind block and the joint
   block;
4. `figure_4_example_*.svg`: a day-ahead and a 14-day example, in winter and summer, of the 51
   scenarios of one Primary, one wind farm, and one solar farm under three constructions.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
uv run --no-sync python ml-experiments/00436-*/e00436_figures.py
```
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00436_build as build  # ty: ignore[unresolved-import]
import e00436_contrasts as contrasts  # ty: ignore[unresolved-import]
import e00436_report as report  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / build.SLUG
WIDTH: Final[int] = 600
ARM_COLOURS: Final[dict[str, str]] = {
    "J1": "#1A1A1A",
    "J2": theme.BRAND_ORANGE,
    "J3": theme.DATA_BLUE,
    "J4": theme.DATA_SKY,
    "J5": theme.DATA_PURPLE,
    "J6": theme.DATA_GREEN,
}
ROWS: Final[list[tuple[str, str, str, str]]] = [
    ("Joint: all 25 nodes, 6-hourly leads", "joint", "all", "all"),
    ("Across Primaries (demand)", "spatial", "all", "demand"),
    ("Across wind farms", "spatial", "all", "wind"),
    ("Across solar farms", "spatial", "all", "solar"),
    ("Across the three families (daylight)", "cross", "daylight", "all"),
]
EXAMPLES: Final[dict[str, str]] = {"winter": "2026-01-13", "summer": "2026-06-16"}
EXAMPLE_NODES: Final[list[tuple[int, str]]] = [
    (0, "Primary D1"),
    (16, "Wind farm W1"),
    (19, "Solar farm A"),
]
EXAMPLE_ARMS: Final[list[str]] = ["J0", "J2", "J5"]


def figure_contrasts() -> None:
    """Draw the variogram score relative to independent draws, per block."""
    scores = pl.read_parquet(build.WORK_DIR / "scores_all.parquet")
    rows = []
    for name, block, slice_name, label in ROWS:
        wide = contrasts.month_means(
            scores=scores,
            window="days_1_10",
            block=block,
            slice_name=slice_name,
            label=label,
        )
        margin = report.margin(
            ind=float(wide["J0"].to_numpy().mean()), co=float(wide["J1"].to_numpy().mean())
        )
        for arm in ARM_COLOURS:
            point, lo, hi = report.bootstrap_relative(
                base=wide["J0"].to_numpy(), other=wide[arm].to_numpy()
            )
            rows.append(
                {"row": name, "arm": arm, "change": point * 100, "lo": lo * 100, "hi": hi * 100,
                 "margin": margin * 100}
            )  # fmt: skip
    data = pl.DataFrame(rows).to_pandas()  # ty: ignore[unresolved-attribute]
    order = [r[0] for r in ROWS]
    margins = {r["row"]: r["margin"] for r in rows}
    base = alt.Chart(data).encode(
        y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_COLOURS)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None, columns=6),
        ),
    )
    bars = base.mark_rule(strokeWidth=2, clip=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lo:Q",
            scale=alt.Scale(domain=[-60, 40]),
            title="Variogram score relative to independent draws (%); lower is better",
        ),
        x2="hi:Q",
    )
    points = base.mark_point(filled=True, size=50, clip=True).encode(x="change:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#1A1A1A").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    ticks = (
        alt.Chart(
            pl.DataFrame(
                [{"row": r, "x": s * m} for r, m in margins.items() for s in (-1, 1)]
            ).to_pandas()  # ty: ignore[unresolved-attribute]
        )
        .mark_tick(color="#8A8A8A", size=26, thickness=2)
        .encode(x="x:Q", y=alt.Y("row:N", sort=order))  # ty: ignore[unresolved-attribute]
    )
    chart = (zero + ticks + bars + points).properties(
        width=WIDTH,
        height=330,
        title=alt.TitleParams(
            [
                "Most constructions beat independent draws across wind and solar farms,",
                "none beats them on the joint block. Days 1 to 10. Points: mean over 10 months.",
                "Lines: 98.75% month-block intervals. Grey ticks: the margin of each row.",
            ],
            anchor="start",
        ),
    )
    chart.save(DOCS_DIR / "figure_1_variogram_contrasts.svg")


def figure_dependence() -> None:
    """Draw each construction's mean score correlation against the realised one."""
    dep = pl.read_parquet(build.WORK_DIR / "diag_dependence.parquet")
    items = {
        "lag0_demand_solar": "Demand with solar (same half-hour)",
        "lag0_demand_wind": "Demand with wind (same half-hour)",
        "lag0_wind_solar": "Wind with solar (same half-hour)",
        "lag0_within_demand": "Between Primaries",
        "lag0_within_wind": "Between wind farms",
        "lag0_within_solar": "Between solar farms",
    }
    arms = ["J0", *ARM_COLOURS]
    data = dep.filter(pl.col("item").is_in(list(items)) & pl.col("arm").is_in(arms)).with_columns(
        row=pl.col("item").replace_strict(items)
    )
    observed = data.group_by("row").agg(pl.col("observed").first())
    colours = {"J0": "#8A8A8A", **ARM_COLOURS}
    order = list(items.values())
    points = (
        alt.Chart(data.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .encode(
            x=alt.X(
                "scenario:Q",
                scale=alt.Scale(domain=[-1, 1]),
                title="Mean correlation of Gaussian scores",
            ),
            y=alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(orient="top", title=None, columns=7),
            ),
        )
    )
    realised = (
        alt.Chart(observed.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_tick(color="#1A1A1A", thickness=3, size=26)
        .encode(x="observed:Q", y=alt.Y("row:N", sort=order))
    )
    (realised + points).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            [
                "J2 overshoots the realised correlation in most pairs; the nearest construction differs by pair.",
                "Black tick: realised. Dots: each construction, mean over 44 runs.",
            ],
            anchor="start",
        ),
    ).save(DOCS_DIR / "figure_2_dependence.svg")


def figure_ranks() -> None:
    """Draw average-rank histograms of the spatial wind block and the joint block."""
    ranks = pl.read_parquet(build.WORK_DIR / "diag_rank.parquet").filter(
        (pl.col("kind") == "average_rank") & pl.col("block").is_in(["spatial_wind", "joint"])
    )
    ranks = (
        ranks.with_columns(bin=(pl.col("bin") // 5).clip(upper_bound=9))
        .group_by("arm", "block", "kind", "bin")
        .agg(pl.col("count").sum())
    )
    ranks = ranks.with_columns(
        share=pl.col("count")
        / pl.col("count").sum().over("arm", "block")
        * ranks["bin"].n_unique(),
        block=pl.col("block").replace({"spatial_wind": "Across wind farms", "joint": "Both"}),
    )
    arms = ["J0", "J1", "J3", "J5", "J6"]
    colours = {"J0": "#8A8A8A", **ARM_COLOURS}
    charts = []
    for block in ("Across wind farms", "Both"):
        panels = []
        for arm in arms:
            sub = ranks.filter((pl.col("arm") == arm) & (pl.col("block") == block))
            panels.append(
                alt.Chart(sub.to_pandas())  # ty: ignore[unresolved-attribute]
                .mark_bar(color=colours[arm])
                .encode(
                    x=alt.X(
                        "bin:O",
                        axis=alt.Axis(
                            values=[0, 5, 9],
                            title="Rank of the observation (10 groups)"
                            if arm == arms[-1]
                            else None,
                        ),
                    ),
                    y=alt.Y(
                        "share:Q",
                        scale=alt.Scale(domain=[0, 3], clamp=True),
                        axis=alt.Axis(values=[0, 1, 2, 3], title=arm),
                    ),
                )
                .properties(width=WIDTH // 2 - 60, height=60)
            )
        charts.append(alt.vconcat(*panels).properties(title=block))
    alt.hconcat(*charts).properties(
        title=alt.TitleParams(
            [
                "Average-rank histograms. Flat at 1 means the observation is exchangeable with the",
                "scenarios. Bars clipped at 3.",
            ],
            anchor="start",
        )
    ).save(DOCS_DIR / "figure_3_rank_histograms.svg")


def figure_examples() -> None:
    """Draw day-ahead and 14-day example scenarios for each season."""
    for season, date in EXAMPLES.items():
        run = np.load(build.WORK_DIR / "runs" / f"{date}.npz")
        actual = run["actual"]
        paths = {
            arm: np.load(build.WORK_DIR / "paths" / arm / f"{date}.npy") for arm in EXAMPLE_ARMS
        }
        for horizon, (lo, hi) in {"day-ahead": (30, 78), "14_days": (0, build.N_STEPS)}.items():
            rows = []
            for arm in EXAMPLE_ARMS:
                for node, name in EXAMPLE_NODES:
                    for k in range(12):
                        for t in range(lo, hi):
                            rows.append(  # noqa: PERF401
                                (
                                    arm,
                                    name,
                                    k,
                                    (t - lo) / 2.0 / (1 if horizon == "day-ahead" else 24.0),
                                    float(paths[arm][k, node, t]),
                                )
                            )
            scen = pl.DataFrame(rows, schema=["arm", "node", "k", "x", "y"], orient="row")
            obs = pl.DataFrame(
                [(name, (t - lo) / 2.0 / (1 if horizon == "day-ahead" else 24.0), float(actual[node, t]))
                 for node, name in EXAMPLE_NODES for t in range(lo, hi) if not np.isnan(actual[node, t])],
                schema=["node", "x", "y"], orient="row",
            )  # fmt: skip
            ymax = {
                name: float(max(paths[a][:12, node, lo:hi].max() for a in EXAMPLE_ARMS)) * 1.05
                for node, name in EXAMPLE_NODES
            }
            xtitle = (
                "Hours after midnight (tomorrow)"
                if horizon == "day-ahead"
                else "Days after the forecast starts"
            )
            xaxis = (
                alt.Axis(values=[0, 6, 12, 18, 24])
                if horizon == "day-ahead"
                else alt.Axis(values=[0, 5, 10, 14])
            )
            panels = []
            for _, name in EXAMPLE_NODES:
                row = []
                for arm in EXAMPLE_ARMS:
                    lines = (
                        alt.Chart(
                            scen.filter(
                                (pl.col("arm") == arm) & (pl.col("node") == name)
                            ).to_pandas()
                        )
                        .mark_line(strokeWidth=0.8, opacity=0.6, color=theme.DATA_BLUE, aria=False)
                        .encode(  # ty: ignore[unresolved-attribute]
                            x=alt.X(
                                "x:Q",
                                axis=xaxis,
                                title=xtitle if name == EXAMPLE_NODES[-1][1] else None,
                            ),
                            y=alt.Y(
                                "y:Q",
                                scale=alt.Scale(domain=[0, ymax[name]]),
                                title=name if arm == EXAMPLE_ARMS[0] else None,
                            ),
                            detail="k:N",
                        )
                    )
                    truth = (
                        alt.Chart(obs.filter(pl.col("node") == name).to_pandas())  # ty: ignore[unresolved-attribute]
                        .mark_line(strokeWidth=1.6, color="#1A1A1A", aria=False)
                        .encode(x="x:Q", y="y:Q")
                    )
                    row.append(
                        (lines + truth).properties(
                            width=WIDTH // 3 - 50,
                            height=90,
                            title=arm if name == EXAMPLE_NODES[0][1] else "",
                        )
                    )
                panels.append(alt.hconcat(*row))
            alt.vconcat(*panels).properties(
                title=alt.TitleParams(
                    [f"12 of the 51 scenarios ({season} Tuesday run, {horizon.replace('_', ' ')}). Black: observed.",
                     "Values in % of capacity (Primary: of its peak). J0 independent, J2 by ensemble rank, J5 copula."],
                    anchor="start",
                )
            ).save(DOCS_DIR / f"figure_4_example_{season}_{horizon}.svg")  # fmt: skip


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts()
    figure_dependence()
    figure_ranks()
    figure_examples()
