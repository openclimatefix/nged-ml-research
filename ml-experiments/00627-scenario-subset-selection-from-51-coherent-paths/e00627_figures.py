# ruff: noqa: D103, E501
"""Draw experiment 627's figures from the tables `e00627_score.py` and `e00627_bsp.py` wrote."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00627_report as rep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00627-scenario-subset-selection-from-51-coherent-paths"
)
CHART: Any = alt.Chart
W: Final[int] = 600
ARM_ORDER: Final[list[str]] = ["R", "S-wind", "S-solar", "S-net", "S-demand", "C"]
COLOUR: Final[dict[str, str]] = {
    "R": "#888888",
    "S-wind": theme.DATA_BLUE,
    "S-solar": theme.BRAND_ORANGE,
    "S-net": theme.DATA_PURPLE,
    "S-demand": theme.DATA_GREEN,
    "C": theme.DATA_SKY,
    "S-bsp": theme.DATA_BLUE,
    "Q": theme.BRAND_ORANGE,
    "All": "#1A1A1A",
}
ALPHA: Final[float] = 0.05


def ci(x: np.ndarray) -> tuple[float, float, float]:
    idx = rep.RNG.integers(0, len(x), size=(rep.N_BOOT, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [ALPHA / 2, 1 - ALPHA / 2])
    return float(x.mean()), float(lo), float(hi)


def monthly_rows(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return (
        df.with_columns(rep.month())
        .group_by("month", "arm", "k")
        .agg(pl.col(cols).mean())
        .sort("month")
    )


def interval_rows(m: pl.DataFrame, col: str) -> pl.DataFrame:
    rows = []
    for (arm, k), g in m.group_by(["arm", "k"]):
        est, lo, hi = ci(g.sort("month")[col].drop_nulls().to_numpy())
        rows.append({"arm": arm, "k": k, "est": est, "lo": lo, "hi": hi})
    return pl.DataFrame(rows)


def dot_chart(
    rows: pl.DataFrame, *, title: str, ytitle: str, rule: float | None, height: int = 220
) -> Any:
    d = rows.to_pandas()
    order = [a for a in [*ARM_ORDER, "S-bsp", "Q", "All"] if a in set(rows["arm"])]
    base = CHART(d).encode(
        x=alt.X("arm:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40)),
        color=alt.Color(
            "arm:N", scale=alt.Scale(domain=list(COLOUR), range=list(COLOUR.values())), legend=None
        ),
    )
    layers = [
        base.mark_rule().encode(y=alt.Y("lo:Q", title=ytitle), y2="hi:Q"),
        base.mark_point(filled=True, size=70).encode(y="est:Q"),
    ]
    if rule is not None:
        layers.insert(
            0, CHART(d).mark_rule(color="#888888", strokeDash=[4, 3]).encode(y=alt.datum(rule))
        )
    return (
        alt.layer(*layers)
        .properties(width=150, height=height)
        .facet(column=alt.Column("k:O", title="Number of paths K"))
        .properties(title=textwrap.wrap(title, 55))
    )


def head_frame() -> pl.DataFrame:
    sel = pl.read_parquet(rep.W / "selection.parquet").with_columns(
        pl.col(pl.Float64).fill_nan(None)
    )
    return sel.filter(
        (pl.col("coupling") == "J3")
        & (pl.col("selection") == "d1_10")
        & (pl.col("band") == "d1_10")
    ).with_columns(
        ((pl.col("tail_low") + pl.col("tail_high")) / 2).alias("tail"),
        ((pl.col("gap05") + pl.col("gap50") + pl.col("gap95")) / 3).alias("gap"),
    )


def per_run(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.group_by("power_fcst_init_time", "arm", "k").agg(pl.col(cols).mean())


def fig1(h: pl.DataFrame) -> None:
    m = monthly_rows(per_run(h.filter(pl.col("arm") != "Oracle"), ["tail"]), ["tail"])
    chart = dot_chart(
        interval_rows(m, "tail"),
        title="Share of draws that include a path in the 10% tails of the 51 (higher is better)",
        ytitle="Tail coverage",
        rule=None,
    )
    chart.save(DOCS / "figure_1_tail_coverage.svg")


def fig2(h: pl.DataFrame) -> None:
    m = monthly_rows(per_run(h.filter(pl.col("arm") != "Oracle"), ["gap"]), ["gap"])
    chart = dot_chart(
        interval_rows(m, "gap"),
        title="Gap to the 51-path distribution, mean of 5th, 50th, 95th percentile (lower is better)",
        ytitle="Gap / spread of the 51",
        rule=0.10,
    )
    chart.save(DOCS / "figure_2_percentile_gap.svg")


def fig3() -> None:
    j = pl.read_parquet(rep.W / "joint.parquet").filter(
        (pl.col("coupling") == "J3") & (pl.col("band") == "d1_10")
    )
    base = (
        j.filter(pl.col("arm") == "All")
        .with_columns(rep.month())
        .group_by("month")
        .agg(pl.col("variogram").mean())
        .sort("month")["variogram"]
        .to_numpy()
    )
    rows = []
    for arm in ARM_ORDER:
        for k in (3, 5, 10):
            f = j.filter(
                (pl.col("arm") == arm) & (pl.col("selection") == "d1_10") & (pl.col("k") == k)
            )
            a = (
                f.with_columns(rep.month())
                .group_by("month")
                .agg(pl.col("variogram").mean())
                .sort("month")["variogram"]
                .to_numpy()
            )
            idx = rep.RNG.integers(0, len(a), size=(rep.N_BOOT, len(a)))
            r = a[idx].mean(1) / base[idx].mean(1) - 1
            lo, hi = np.quantile(r, [ALPHA / 2, 1 - ALPHA / 2])
            rows.append({"arm": arm, "k": k, "est": a.mean() / base.mean() - 1, "lo": lo, "hi": hi})
    chart = dot_chart(
        pl.DataFrame(rows),
        title="Variogram score of the K paths against the observed vector, relative to all 51 (lower is better)",
        ytitle="Difference to all 51 paths",
        rule=0.0,
    )
    chart.save(DOCS / "figure_3_joint_variogram.svg")


def fig4() -> None:
    b = pl.read_parquet(rep.W / "bsp.parquet").with_columns(pl.col(pl.Float64).fill_nan(None))
    d = b.filter((pl.col("coupling") == "J3") & (pl.col("band") == "d1_10"))
    pr = d.group_by("power_fcst_init_time", "arm", "k").agg(pl.col("exceed").mean())
    m = monthly_rows(pr, ["exceed"])
    rows = interval_rows(m, "exceed").filter(pl.col("k").is_in([4, 5, 10, 51]))
    all_row = rows.filter(pl.col("arm") == "All")
    allv = float(all_row["est"][0])
    rows = rows.filter(pl.col("arm") != "All")
    chart = dot_chart(
        rows,
        title="Share of observed daily peak exports above the largest of the K paths' daily peaks (lower is better)",
        ytitle="Share of days exceeded",
        rule=allv + 0.05,
    )
    chart.save(DOCS / "figure_4_bsp_gate.svg")


def fig5() -> None:
    summ = pl.read_parquet(rep.W / "summ.parquet")
    rows = []
    for coupling in ("J3", "J2"):
        for name in ("wind", "solar", "net", "demand"):
            f = summ.filter((pl.col("coupling") == coupling) & (pl.col("summary") == name))
            a = f.filter(pl.col("band") == "d1_3").sort("power_fcst_init_time")["values"].to_list()
            c = f.filter(pl.col("band") == "d4_10").sort("power_fcst_init_time")["values"].to_list()
            rho = [
                np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
                for x, y in zip(a, c, strict=True)
            ]
            rho = np.array([r for r in rho if np.isfinite(r)])
            est, lo, hi = ci(rho)
            rows.append(
                {
                    "coupling": "Schaake shuffle (J3)"
                    if coupling == "J3"
                    else "Ensemble copula (J2)",
                    "summary": name,
                    "est": est,
                    "lo": lo,
                    "hi": hi,
                }
            )
    d = pl.DataFrame(rows).to_pandas()
    base = CHART(d).encode(
        x=alt.X(
            "summary:N",
            title=None,
            sort=["wind", "solar", "net", "demand"],
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color("coupling:N", legend=alt.Legend(title=None, orient="top")),
        xOffset="coupling:N",
    )
    chart = alt.layer(
        CHART(pl.DataFrame({"y": [0.0]}).to_pandas()).mark_rule(color="#888888").encode(y="y:Q"),
        base.mark_rule().encode(
            y=alt.Y("lo:Q", title="Rank correlation, days 1-3 vs days 4-10"), y2="hi:Q"
        ),
        base.mark_point(filled=True, size=70).encode(y="est:Q"),
    ).properties(
        width=W - 100,
        height=240,
        title=textwrap.wrap("Does a path keep its rank between lead bands within one run?", 55),
    )
    chart.save(DOCS / "figure_5_label_carry_over.svg")


def main() -> None:
    h = head_frame()
    fig1(h)
    fig2(h)
    fig3()
    fig4()
    fig5()


if __name__ == "__main__":
    main()
