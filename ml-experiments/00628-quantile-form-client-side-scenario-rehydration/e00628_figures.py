# ruff: noqa: D103, E501
"""Draw experiment 628's figures from the tables the scoring scripts wrote.

1. `figure_1_tail_rules.svg`: share of actuals beyond the 0.2% and 99.8% quantiles, by tail rule;
2. `figure_2_client_against_pipeline.svg`: variogram score of the client's 51 stratified paths
   against experiment 436's 51 in-pipeline paths, by coupling;
3. `figure_3_bsp_pick.svg`: movement of the day-3 95th-percentile BSP demand pick;
4. `figure_4_example_day_ahead.svg` and 5. `figure_5_example_ten_days.svg`: client scenarios beside
   the stored quantiles of one Primary.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00628_neural as neural  # ty: ignore[unresolved-import]
import e00628_report as rep  # ty: ignore[unresolved-import]
import e00628_score as s628  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00628-quantile-form-client-side-scenario-rehydration"
)
CHART: Any = alt.Chart
W: Final[int] = 680
TAIL_COLOUR: Final[dict[str, str]] = {
    "clamp": theme.BRAND_ORANGE,
    "linear": theme.DATA_SKY,
    "gaussian": theme.DATA_BLUE,
}
TAIL_NAME: Final[dict[str, str]] = {
    "clamp": "Clamp at q01 and q99",
    "linear": "Linear extrapolation",
    "gaussian": "Gaussian tail",
}
COUPLING_COLOUR: Final[dict[str, str]] = {
    "IND": theme.BRAND_ORANGE,
    "COP": theme.DATA_BLUE,
    "SS": theme.DATA_GREEN,
}
COUPLING_NAME: Final[dict[str, str]] = {
    "IND": "Independent draws",
    "COP": "Fitted copula",
    "SS": "Schaake shuffle",
}
FAMILY_NAME: Final[dict[str, str]] = {
    "demand": "16 Primaries",
    "wind": "3 wind farms",
    "solar": "6 solar farms",
}


def save(chart: Any, name: str) -> None:
    chart.save(DOCS / name)


def fig1() -> None:
    e = pl.read_parquet(rep.W / "exceed_analytic.parquet")
    rows = []
    for fam in ("demand", "wind", "solar"):
        for tail in rep.TAILS:
            for side, level in (
                ("Below the 0.2% quantile (nominal 0.2%)", 0.002),
                ("Above the 99.8% quantile (nominal 0.2%)", 0.998),
            ):
                s = e.filter(
                    (pl.col("arm") == f"Q11-{tail}")
                    & (pl.col("family") == fam)
                    & (pl.col("level") == level)
                )
                rows.append(
                    {
                        "family": FAMILY_NAME[fam],
                        "tail": TAIL_NAME[tail],
                        "side": side,
                        "share": s["hits"].sum() / s["n"].sum() * 100,  # ty: ignore[unsupported-operator]
                    }
                )
    data = pl.DataFrame(rows).to_pandas()
    panels = []
    for side in sorted(data["side"].unique(), reverse=True):
        bars = (
            CHART(data[data["side"] == side])
            .mark_bar()
            .encode(
                y=alt.Y("tail:N", title=None, sort=list(TAIL_NAME.values()), axis=None),
                x=alt.X(
                    "share:Q",
                    title="Share of actuals beyond the quantile (%; nearer 0.2 is better)",
                ),
                row=alt.Row(
                    "family:N",
                    title=None,
                    sort=list(FAMILY_NAME.values()),
                    header=alt.Header(labelAngle=0, labelAlign="left", labelPadding=4),
                ),
                color=alt.Color(
                    "tail:N",
                    scale=alt.Scale(
                        domain=list(TAIL_NAME.values()), range=list(TAIL_COLOUR.values())
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            .properties(width=W - 200, height=34, title=side)
        )
        panels.append(bars)
    save(alt.vconcat(*panels).resolve_scale(color="shared"), "figure_1_tail_rules.svg")


def fig2() -> None:
    c = pl.read_parquet(rep.W / "coherence.parquet").with_columns(rep.month())
    base = c.filter((pl.col("window") == "days_1_10") & (pl.col("slice") == "all"))
    pairs = {"IND": "J0", "COP": "J5", "SS": "J3"}
    rows = []
    for gname, df in {
        "16 Primaries": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "demand")),
        "3 wind farms": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "wind")),
        "6 solar farms": base.filter((pl.col("block") == "spatial") & (pl.col("label") == "solar")),
    }.items():
        per = (
            df.group_by("month", "power_fcst_init_time", "arm")
            .agg(pl.col("variogram").mean())
            .group_by("month", "arm")
            .agg(pl.col("variogram").mean())
        )
        for coupling, ref in pairs.items():
            r = per.filter(pl.col("arm") == ref).sort("month")
            a = per.filter(pl.col("arm") == f"{coupling}-gaussian").sort("month")
            est, lo, hi = rep.boot_ratio(a["variogram"].to_numpy(), r["variogram"].to_numpy())
            rows.append(
                {
                    "group": gname,
                    "coupling": COUPLING_NAME[coupling],
                    "est": est * 100,
                    "lo": lo * 100,
                    "hi": hi * 100,
                }
            )
    all_rows = pl.DataFrame(rows)
    zero = CHART(pl.DataFrame({"x": [0.0]}).to_pandas()).mark_rule(color="#1A1A1A").encode(x="x:Q")
    panels = []
    for g in ("16 Primaries", "3 wind farms", "6 solar farms"):
        base_chart = CHART(all_rows.filter(pl.col("group") == g).to_pandas()).encode(
            y=alt.Y("coupling:N", title=None, sort=list(COUPLING_NAME.values()), axis=None),
            color=alt.Color(
                "coupling:N",
                scale=alt.Scale(
                    domain=list(COUPLING_NAME.values()), range=list(COUPLING_COLOUR.values())
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        panels.append(
            alt.layer(
                zero,
                base_chart.mark_rule().encode(x="lo:Q", x2="hi:Q"),
                base_chart.mark_point(filled=True, size=70).encode(
                    x=alt.X(
                        "est:Q",
                        title="Client variogram score minus in-pipeline score (% of the in-pipeline score; lower is better)",
                    )
                ),
            ).properties(width=W - 80, height=70, title=g)
        )
    save(alt.vconcat(*panels).configure_view(strokeWidth=0), "figure_2_client_against_pipeline.svg")


def pick_of(q: pl.DataFrame, arm: str, pick: str, which: str) -> pl.DataFrame:
    return q.filter((pl.col("arm") == arm) & (pl.col("set") == which)).select(
        "power_fcst_init_time", pl.col(pick).alias(which)
    )


def fig3() -> None:
    rows = []
    for net in ("K1",):
        q = pl.read_parquet(rep.W / f"neural_{net}_query.parquet")
        for pick in ("bsp_q95_boston", "bsp_q95_skegness"):
            for coupling in ("IND", "COP", "SS"):
                for tail in rep.TAILS:
                    arm = f"{coupling}-{tail}"
                    j = (
                        pick_of(q, arm, pick, "A")
                        .join(pick_of(q, arm, pick, "B"), on="power_fcst_init_time")
                        .join(pick_of(q, arm, pick, "A51"), on="power_fcst_init_time")
                    )
                    rows.append(
                        {
                            "network": net,
                            "bsp": pick.split("_")[-1].capitalize(),
                            "coupling": COUPLING_NAME[coupling],
                            "tail": TAIL_NAME[tail],
                            "between_sets": float((j["A"] - j["B"]).abs().mean()),  # ty: ignore[invalid-argument-type]
                            "from_51": float((j["A"] - j["A51"]).abs().mean()),  # ty: ignore[invalid-argument-type]
                        }
                    )
    data = pl.DataFrame(rows).filter(pl.col("network") == "K1").to_pandas()
    panels = []
    for col, title in (
        (
            "between_sets",
            "Boston BSP: mean change of the pick between two independent sets of 1,000 draws (MW)",
        ),
        (
            "from_51",
            "Boston BSP: mean distance of the pick from the pick of 51 stratified paths (MW)",
        ),
    ):
        panels.append(
            CHART(data[data["bsp"] == "Boston"])
            .mark_bar()
            .encode(
                x=alt.X(
                    "tail:N",
                    title=None,
                    sort=list(TAIL_NAME.values()),
                    axis=alt.Axis(labelAngle=0, labelLimit=140),
                ),
                xOffset=alt.XOffset("coupling:N", sort=list(COUPLING_NAME.values())),
                y=alt.Y(f"{col}:Q", title="MW (smaller is steadier)"),
                color=alt.Color(
                    "coupling:N",
                    scale=alt.Scale(
                        domain=list(COUPLING_NAME.values()), range=list(COUPLING_COLOUR.values())
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            .properties(width=W - 100, height=150, title=title)
        )
    save(alt.vconcat(*panels).resolve_scale(color="shared"), "figure_3_bsp_pick.svg")


def example_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    runs = neural.load_runs(network="K1")
    z = [neural.pit_z(q=r["q"], actual=r["actual"]) for r in runs]
    node = neural.NODE_IDS.index(3)
    run = max(runs, key=lambda r: np.isfinite(r["actual"][node]).mean())
    month = run["month"]
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    train = [
        i for i, r in enumerate(runs) if r["month"] != month and abs((r["init"] - start).days) > 15
    ]
    chol, phi = neural.fit_copula(z_runs=[z[i] for i in train])
    u = s628.client.copula_uniforms(
        chol=chol, phi=phi, k=1000, steps=neural.T, rng=np.random.default_rng(7)
    )
    paths = s628.client.scenarios(
        q=run["q"], u=u, tail="gaussian", bounds=np.full((neural.P, 2), np.nan)
    )[:, node]
    hours = (np.arange(neural.T) + 1) * 0.5
    q = run["q"][node]
    band = pl.DataFrame(
        {
            "hours": hours,
            "q05": q[:, 1],
            "q25": q[:, 3],
            "q50": q[:, 5],
            "q75": q[:, 7],
            "q95": q[:, 9],
            "actual": run["actual"][node],
        }
    )
    lines = pl.DataFrame(
        {
            "hours": np.tile(hours, 4),
            "scenario": np.repeat(np.arange(4), neural.T).astype(str),
            "value": paths[:4].ravel(),
        }
    )
    return band, lines


def example_fig(
    band: pl.DataFrame, lines: pl.DataFrame, *, last_hour: float, name: str, title: str | list[str]
) -> None:
    b = band.filter(pl.col("hours") <= last_hour).to_pandas()
    ln = lines.filter(pl.col("hours") <= last_hour).to_pandas()
    x = alt.X(
        "hours:Q",
        title="Hours after the forecast was issued",
        scale=alt.Scale(domain=[0, last_hour]),
    )
    chart = alt.layer(
        CHART(b)
        .mark_area(color=theme.DATA_BLUE, opacity=0.15)
        .encode(x=x, y=alt.Y("q05:Q", title="Power (% of capacity)"), y2="q95:Q"),
        CHART(b).mark_area(color=theme.DATA_BLUE, opacity=0.25).encode(x=x, y="q25:Q", y2="q75:Q"),
        CHART(ln)
        .mark_line(strokeWidth=0.8, opacity=0.8, color=theme.DATA_PURPLE)
        .encode(x=x, y="value:Q", detail="scenario:N"),
        CHART(b).mark_line(color=theme.DATA_BLUE, strokeWidth=2).encode(x=x, y="q50:Q"),
        CHART(b[b["actual"].notna()])
        .mark_line(color="#1A1A1A", strokeWidth=1.5)
        .encode(x=x, y="actual:Q"),
    ).properties(width=W - 80, height=240, title=title)
    save(chart, name)


def main() -> None:
    fig1()
    fig2()
    fig3()
    band, lines = example_data()
    example_fig(
        band,
        lines,
        last_hour=24,
        name="figure_4_example_day_ahead.svg",
        title=[
            "One Primary, day ahead: black is the reading, blue line the median,",
            "dark and pale blue bands the 50% and 90% ranges, purple four client scenarios",
        ],
    )
    example_fig(
        band,
        lines,
        last_hour=240,
        name="figure_5_example_ten_days.svg",
        title=["The same forecast over the full 10 days: the bands widen with lead time"],
    )


if __name__ == "__main__":
    main()
