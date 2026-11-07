# ruff: noqa: D103
"""Draw experiment 625's figures from the tables `e00625_score.py` wrote.

1. `figure_1_variogram_vs_independent.svg`: variogram score of each coupling against independent
   draws, by group, with month-block intervals;
2. `figure_2_k_sweep.svg`: variogram score of K paths against K = 51, by group;
3. `figure_3_opposite_sky.svg`: share of opposite-sky-state farm pairs;
4. `figure_4_step_autocorrelation.svg`: lag-1 autocorrelation of step changes against the observed.
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import e00625_report as rep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00625-coherent-members-from-quantiles-k-sweep-and-path-checks"
)
CHART: Any = alt.Chart
"""`alt.Chart` typed loosely, because `ty` loses the chart type after a `mark_*()` call."""
W: Final[int] = 600
COLOUR: Final[dict[str, str]] = {
    "J2": theme.BRAND_ORANGE,
    "J3": theme.DATA_BLUE,
    "SSR": theme.DATA_SKY,
    "J5": theme.DATA_PURPLE,
    "J6": theme.DATA_GREEN,
}
NAME: Final[dict[str, str]] = {
    "J2": "ECC (ensemble ranks)",
    "J3": "SS (observed paths)",
    "SSR": "SS-R (anomaly paths)",
    "J5": "ECC + error copula",
    "J6": "MW (member-wise)",
}
GROUPS: Final[list[tuple[str, str]]] = [
    ("Primaries", "demand"),
    ("Wind farms", "wind"),
    ("Solar farms", "solar"),
    ("Joint, all 25 nodes", "joint"),
]


def group_frames() -> dict[str, pl.DataFrame]:
    """Return per-run variogram scores (days 1 to 10, all slice) for each group."""
    coh = pl.read_parquet(rep.W / "coherence.parquet").filter(
        (pl.col("window") == "days_1_10") & (pl.col("slice") == "all")
    )
    out = {}
    for _, key in GROUPS:
        if key == "joint":
            f = coh.filter(pl.col("block") == "joint")
        else:
            f = coh.filter((pl.col("block") == "spatial") & (pl.col("label") == key))
        out[key] = f.select("power_fcst_init_time", "arm", "k", "variogram")
    return out


def contrast_rows(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for title, key in GROUPS:
        m = rep.monthly(frames[key], "variogram")
        for arm in ("J2", "J3", "SSR", "J5", "J6"):
            if key == "demand" and arm == "J6":
                continue
            a = m.filter((pl.col("arm") == arm) & (pl.col("k") == 51)).sort("month")
            b = m.filter((pl.col("arm") == "J0") & (pl.col("k") == 51)).sort("month")
            est, lo, hi = rep.boot(a["variogram"].to_numpy(), b["variogram"].to_numpy())
            rows.append(
                {"group": title, "arm": NAME[arm], "est": est * 100, "lo": lo * 100, "hi": hi * 100}
            )
    return pl.DataFrame(rows)


def k_rows(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows = []
    for title, key in GROUPS:
        m = rep.monthly(frames[key], "variogram")
        for arm in ("J3", "SSR"):
            if key == "demand" and arm == "J3":
                continue
            if key in ("wind", "solar") and arm == "SSR":
                continue
            ref = m.filter((pl.col("arm") == arm) & (pl.col("k") == 51)).sort("month")
            for k in (5, 10, 20):
                a = m.filter((pl.col("arm") == arm) & (pl.col("k") == k)).sort("month")
                est, lo, hi = rep.boot(a["variogram"].to_numpy(), ref["variogram"].to_numpy())
                rows.append(
                    {"group": title, "k": k, "est": est * 100, "lo": lo * 100, "hi": hi * 100}
                )
    return pl.DataFrame(rows)


def save(chart: Any, name: str) -> None:
    chart.save(DOCS / name)


def fig1(rows: pl.DataFrame) -> None:
    base = CHART(rows.to_pandas()).encode(
        y=alt.Y("arm:N", title=None, sort=[NAME[a] for a in ("J2", "J3", "SSR", "J5", "J6")]),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=[NAME[a] for a in COLOUR], range=list(COLOUR.values())),
            legend=None,
        ),
    )
    pts = base.mark_point(filled=True, size=70).encode(
        x=alt.X("est:Q", title="Variogram score against independent draws (%, lower is better)")
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")
    zero = CHART(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#1A1A1A").encode(x="x:Q")
    margin = (
        CHART(pl.DataFrame({"x": [-5.0]}).to_pandas())
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        alt.layer(zero, margin, bars, pts, data=None)
        .properties(width=W - 150, height=110)
        .facet(
            row=alt.Row("group:N", title=None, sort=[g for g, _ in GROUPS]), data=rows.to_pandas()
        )
        if False
        else alt.vconcat(
            *[
                alt.layer(
                    zero,
                    margin,
                    bars.transform_filter(alt.datum.group == g),
                    pts.transform_filter(alt.datum.group == g),
                ).properties(width=W - 160, height=100, title=g)
                for g, _ in GROUPS
            ]
        )
    )
    save(chart.configure_view(strokeWidth=0), "figure_1_variogram_vs_independent.svg")


def fig2(rows: pl.DataFrame) -> None:
    data = rows.to_pandas()
    line = CHART(data).encode(
        x=alt.X("k:O", title="Number of member paths K"),
        color=alt.Color("group:N", legend=alt.Legend(title=None, orient="top")),
    )
    chart = alt.layer(
        line.mark_line().encode(y=alt.Y("est:Q", title="Variogram score against K = 51 (%)")),
        line.mark_rule().encode(y=alt.Y("lo:Q", title=None), y2="hi:Q"),
        line.mark_point(filled=True, size=60).encode(y=alt.Y("est:Q", title=None)),
    ).properties(width=W - 80, height=260)
    save(chart, "figure_2_k_sweep.svg")


def fig3() -> None:
    sky = pl.read_parquet(rep.W / "sky.parquet")
    thr = sky.filter(pl.col("arm") == "OBS")["spread"].quantile(0.75)
    days = sky.filter((pl.col("arm") == "OBS") & (pl.col("spread") >= thr)).select(
        "power_fcst_init_time", "day"
    )
    top = sky.join(days, on=["power_fcst_init_time", "day"]).with_columns(rep.month())
    rows = []
    for arm in ("OBS", "J0", "J2", "J3", "SSR", "J5", "J6"):
        k = 0 if arm == "OBS" else 51
        d = (
            top.filter((pl.col("arm") == arm) & (pl.col("k") == k))
            .group_by("month")
            .agg(pl.col("opposite").sum(), pl.col("total").sum())
        )
        sh = (d["opposite"] / d["total"]).to_numpy()
        idx = rep.RNG.integers(0, len(sh), size=(rep.N_BOOT, len(sh)))
        q = np.quantile(sh[idx].mean(1), [0.025, 0.975])
        label = {"OBS": "Observed", "J0": "Independent draws", **{a: NAME[a] for a in COLOUR}}[arm]
        rows.append({"arm": label, "share": sh.mean() * 100, "lo": q[0] * 100, "hi": q[1] * 100})
    data = pl.DataFrame(rows).to_pandas()
    base = CHART(data).encode(y=alt.Y("arm:N", title=None, sort=None))
    chart = alt.layer(
        base.mark_bar(color=theme.DATA_BLUE).encode(
            x=alt.X("share:Q", title="Share of farm-pair half-hours in opposite sky states (%)")
        ),
        base.mark_rule(color="#1A1A1A").encode(x="lo:Q", x2="hi:Q"),
    ).properties(width=W - 160, height=200)
    save(chart, "figure_3_opposite_sky.svg")


def fig4() -> None:
    acf = pl.read_parquet(rep.W / "acf.parquet").filter(pl.col("window") == "d1_3")
    rows = (
        acf.filter(pl.col("k").is_in([0, 51]))
        .filter(pl.col("acf1").is_not_nan())
        .group_by(["family", "arm"])
        .agg(pl.col("acf1").mean())
        .with_columns(
            pl.col("arm").replace_strict(
                {"OBS": "Observed", "J0": "Independent draws", **{a: NAME[a] for a in COLOUR}},
                default=pl.col("arm"),
            )
        )
    )
    chart = (
        CHART(rows.to_pandas())
        .mark_bar(color=theme.DATA_BLUE)
        .encode(
            x=alt.X("acf1:Q", title="Lag-1 autocorrelation of half-hourly change, days 1 to 3"),
            y=alt.Y("arm:N", title=None, sort=None),
            row=alt.Row("family:N", title=None),
        )
        .properties(width=W - 160, height=150)
    )
    save(chart, "figure_4_step_autocorrelation.svg")


def main() -> None:
    frames = group_frames()
    fig1(contrast_rows(frames=frames))
    fig2(k_rows(frames=frames))
    fig3()
    fig4()


if __name__ == "__main__":
    main()
