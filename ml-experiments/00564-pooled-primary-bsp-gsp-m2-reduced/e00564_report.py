# ruff: noqa: E501
"""Print every number the experiment 564 page quotes, and draw its figures.

Reads the tables `e00564_score.py` and `e00564_nodes.py` stored. The example forecasts are M2-a's
seed 0 at one bulk supply point (BSP) and one grid supply point (GSP).
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00087_common as common  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00564_pool as pool
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00564-pooled-primary-bsp-gsp-m2-reduced"
DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00564-pooled-primary-bsp-gsp-m2-reduced"
)
SCORES: Final[Path] = ROOT / "scores"
BANDS: Final = ("0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h")
ARMS: Final = {"M2-a": theme.DATA_BLUE, "M2-b": theme.DATA_SKY, "M2-c": theme.DATA_AMBER}
SETS: Final = {"eight": "MVA-only eight", "six": "Signed-MW six"}
WIDTH: Final = 600


def tables() -> None:
    """Print absolute CRPS, contrasts, and the extra series' scores."""
    a = pl.read_csv(SCORES / "absolute.csv")
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=60):
        for s in ("eight", "six", "eight_without_D8", "D8_alone", "report"):
            print(f"--- absolute CRPS, set {s}")
            have = [
                x
                for x in (
                    "M2-a",
                    "M2-b",
                    "M2-c",
                    "M0-2",
                    "M0",
                    "L-1d5",
                    "A-pool",
                    "manual_heuristic",
                )
                if x in a["arm"].unique().to_list()
            ]
            print(
                a.filter(pl.col("set") == s)
                .pivot(on="arm", index="band", values="crps")
                .select(["band", *have])
                .with_columns(pl.col(pl.Float64).round(3))
            )
        c_ = pl.read_csv(SCORES / "contrasts.csv")
        for s in ("eight", "six", "eight_without_D8", "D8_alone"):
            print(f"--- contrasts, set {s}")
            print(
                c_.filter(pl.col("set") == s)
                .select(
                    "band",
                    "contrast",
                    pl.col("difference").round(3),
                    pl.col("low_bonferroni_month").round(3),
                    pl.col("high_bonferroni_month").round(3),
                    "folds_positive",
                    "by_fold",
                    pl.col("mde").round(3),
                    "tests",
                )
                .sort("contrast", "band")
            )
        n = pl.read_csv(ROOT / "work/scores/nodes.csv")
        print("--- extra series, CRPS by arm and band (count-weighted over months)")
        print(
            n.group_by("site", "kind", "arm", "band")
            .agg((pl.col("crps") * pl.col("n")).sum() / pl.col("n").sum())
            .pivot(on="arm", index=["site", "kind", "band"], values="crps")
            .sort("site", "band")
            .with_columns(pl.col(pl.Float64).round(3))
        )


def figure_gain() -> None:
    """Draw each M2 arm's CRPS minus M0 (same 2 seeds) by lead band with Bonferroni intervals."""
    c_ = pl.read_csv(SCORES / "contrasts.csv")
    names = [f"{a} - M0-2" for a in ARMS]
    c_ = c_.filter(pl.col("contrast").is_in(names) & pl.col("band").is_in([*BANDS, "24-120h"]))
    c_ = c_.with_columns(arm=pl.col("contrast").str.replace(" - M0-2", ""))
    panels = []
    for key, label in SETS.items():
        part = c_.filter(pl.col("set") == key).to_pandas()
        base = alt.Chart(part).encode(
            y=alt.Y("band:N", sort=[*BANDS, "24-120h"], title="Lead band"),
            yOffset=alt.YOffset("arm:N", sort=list(ARMS)),
        )
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARMS), range=list(ARMS.values())),
            legend=alt.Legend(orient="top", title=None),
        )
        bar = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low_bonferroni_month:Q",
                title="CRPS minus M0 (points of capacity; negative favours the pooled arm)",
            ),
            x2="high_bonferroni_month:Q",
            color=colour,
        )  # ty: ignore[unresolved-attribute]
        dot = base.mark_point(filled=True, size=45).encode(x="difference:Q", color=colour)  # ty: ignore[unresolved-attribute]
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
            .mark_rule(color="black")
            .encode(x="x:Q")
        )  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(zero, bar, dot).properties(
                width=WIDTH, height=220, title=alt.TitleParams(label, anchor="start")
            )
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Pooled arms against M0, both averaged over seeds 0 and 1, five folds. Dot: difference.",
                "Bar: Bonferroni month-block interval (42 tests). Line at 0: same as M0.",
            ],
            anchor="start",
        )
    ).save(DOCS / "figure_1_gain_vs_m0.svg")


def figure_nodes() -> None:
    """Draw the extra series' CRPS at 24-120 h by arm, beside the climatology reference."""
    n = pl.read_csv(ROOT / "work/scores/nodes.csv").filter(pl.col("band") == "24-120h")
    g = (
        n.group_by("site", "kind", "arm")
        .agg((pl.col("crps") * pl.col("n")).sum() / pl.col("n").sum())
        .with_columns(label=pl.col("kind") + " " + pl.col("site"))
        .to_pandas()
    )
    domain = ["climatology", *ARMS]
    colours = [theme.GREY_2, *ARMS.values()]
    chart = (
        alt.Chart(g)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("label:N", title=None),
            yOffset=alt.YOffset("arm:N", sort=domain),
            x=alt.X("crps:Q", title="CRPS at 24-120 h (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                "Forecast error at the two grid and two bulk supply points, five folds, seeds 0 and 1.",
                anchor="start",
            ),
        )
    )
    chart.save(DOCS / "figure_2_extra_series.svg")


def figure_examples() -> None:
    """Draw a day-ahead and a 14-day forecast of M2-a at one BSP and one GSP, against the actual."""
    pool.install()
    clean = c.clean_stack()
    table = pool.pooled_table()
    origin = datetime(2026, 1, 12, 9, tzinfo=UTC)
    pred = pl.read_parquet(ROOT / "work/node_preds/M2-a/2026-01_s0.parquet")
    for site in ("D13", "D1"):
        row = table.filter(pl.col("site") == site).row(0, named=True)
        f = pred.filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & (pl.col("power_fcst_init_time") == origin)
        ).sort("valid_time")
        idx = np.asarray(common.grid_index(times=f["valid_time"]))
        f = f.with_columns(
            actual=pl.Series(100 * clean[row["primary"]][idx]),
            **{q: pl.col(q) * 100 for q in ("q05", "q25", "q50", "q75", "q95")},
        ).to_pandas()
        for name, hours in (("day_ahead", 48), ("14_days", 240)):
            part = f[f["valid_time"] <= origin + timedelta(hours=hours)]
            base = alt.Chart(part).encode(x=alt.X("valid_time:T", title="Valid time (UTC)"))
            band = base.mark_area(opacity=0.2, color=theme.DATA_BLUE).encode(y="q05:Q", y2="q95:Q")  # ty: ignore[unresolved-attribute]
            inner = base.mark_area(opacity=0.3, color=theme.DATA_BLUE).encode(y="q25:Q", y2="q75:Q")  # ty: ignore[unresolved-attribute]
            med = base.mark_line(color=theme.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("q50:Q", title="Power (% of capacity)")
            )  # ty: ignore[unresolved-attribute]
            act = base.mark_line(color="black", strokeWidth=1).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
            kind = "BSP" if row["substation_type"] == "BSP" else "GSP"
            (band + inner + med + act).properties(
                width=WIDTH,
                height=200,
                title=alt.TitleParams(
                    f"M2-a forecast at a {kind} issued {origin:%Y-%m-%d %H:%M} UTC. Black: actual. Bands: 90% and 50%.",
                    anchor="start",
                ),
            ).save(DOCS / f"figure_3_example_{kind}_{name}.svg")


if __name__ == "__main__":
    tables()
    figure_gain()
    figure_nodes()
    figure_examples()
