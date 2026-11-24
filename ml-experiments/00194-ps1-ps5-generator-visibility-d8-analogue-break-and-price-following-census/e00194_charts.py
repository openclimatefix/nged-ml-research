# ruff: noqa: E501
"""Draw the figures for experiment 194 from the tables the other scripts wrote."""

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import altair as alt
import e00194_common as common
import numpy as np
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine_figure_guard as guard

DOCS_DIR: Final[Path] = Path("docs/ml-experiments/00194-" + common.SLUG)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
CHANGE: Final[date] = date(2023, 8, 14)
FALL: Final[date] = date(2026, 3, 9)

SITE_NAMES: Final[dict[str, str]] = {
    "D8": "D8",
    "D3": "D3",
    "D5": "D5",
    common.REGISTER_PRIMARY: "register's Primary",
}
"""Chart label of each site, naming the register's Primary generically."""


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # Figure 1: shortfall share pooled over three eras; no dates, no megawatt values.
    era_order = ["Before the shortfall starts", "While the shortfall lasts", "After it fades"]
    rows = []
    for label in ("D8", *common.CONTROL_LABELS):
        w = pl.read_parquet(common.DATA_DIR / f"weekly_{label}.parquet").filter(pl.col("n") >= 168)
        day = w["time"].dt.date()
        era = (
            pl.when(day < CHANGE)
            .then(pl.lit(era_order[0]))
            .when(day < FALL)
            .then(pl.lit(era_order[1]))
            .otherwise(pl.lit(era_order[2]))
        )
        pooled = (
            w.with_columns(era=era)
            .group_by("era")
            .agg(k=pl.col("k").sum(), n=pl.col("n").sum())
            .with_columns(
                share=pl.col("k") / pl.col("n"), site=pl.lit(SITE_NAMES.get(label, label))
            )
        )
        rows.append(pooled.select("era", "site", "share"))
    fig1 = (
        alt.Chart(pl.concat(rows))
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "site:N", title=None, sort=list(SITE_NAMES.values()), axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("share:Q", title="Share of half-hours in shortfall"),
            color=alt.Color(
                "site:N",
                legend=None,
                scale=alt.Scale(
                    domain=list(SITE_NAMES.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE],
                ),
            ),
            column=alt.Column("era:N", title=None, sort=era_order),
        )
        .properties(
            width=WIDTH // 4,
            height=160,
            title="D8's shortfall share is high only while the shortfall lasts (pooled eras)",
        )
    )
    guard.save_chart(chart=fig1, path=DOCS_DIR / "figure_1.svg")

    # Figure 2: pooled before and after the break; no dates, no megawatt values, no counts per day.
    both = pl.read_parquet(common.DATA_DIR / "d8_vs_g2.parquet").filter(
        pl.col("time").is_between(
            datetime(2023, 7, 15, tzinfo=UTC), datetime(2023, 9, 15, tzinfo=UTC), closed="left"
        )
    )
    last_g2 = both.filter(pl.col("g2").is_not_null())["time"].dt.date().max()
    pooled = (
        both.with_columns(
            period=pl.when(pl.col("time").dt.date() <= last_g2)
            .then(pl.lit("Before the break"))
            .otherwise(pl.lit("After the break")),
        )
        .group_by("period")
        .agg(
            tail=pl.col("tail").mean(),
            readings=pl.col("g2").is_not_null().mean(),
        )
    )
    order = ["Before the break", "After the break"]
    top = (
        alt.Chart(pooled)
        .mark_bar(color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("period:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tail:Q", title="D8 shortfall share"),
        )
        .properties(
            width=WIDTH // 2,
            height=110,
            title=[
                "D8's shortfall share is higher after the break, once G2's output",
                "readings stop arriving (the weeks either side, pooled)",
            ],
        )
    )
    bot = (
        alt.Chart(pooled)
        .mark_bar(color=ocf.DATA_BLUE_LIGHT, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("period:N", title=None, sort=order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("readings:Q", title="Share of half-hours with a G2 reading"),
        )
        .properties(width=WIDTH // 2, height=90)
    )
    guard.save_chart(chart=alt.vconcat(top, bot), path=DOCS_DIR / "figure_2.svg")

    # Figure 3: G2 running by price fifth.
    g2 = pl.read_parquet(common.DATA_DIR / "g2_window.parquet")
    edges = np.quantile(g2["price"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    fifth = np.searchsorted(edges, g2["price"].to_numpy(), side="right")
    running = (g2["units"] >= 1).to_numpy()
    rows = []
    for f in range(5):
        k, n = int(running[fifth == f].sum()), int((fifth == f).sum())
        lo, hi = common.wilson(k=k, n=n)
        rows.append({"fifth": f + 1, "share": k / n, "low": lo, "high": hi})
    d = pl.DataFrame(rows)
    bars = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "fifth:O",
                title="Fifth of the system price (1 = cheapest)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "share:Q", title="Share of half-hours G2 ran", scale=alt.Scale(domain=[0, 0.35])
            ),
        )
    )
    err = alt.Chart(d).mark_rule(color="black").encode(x="fifth:O", y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    fig3 = (bars + err).properties(
        width=WIDTH // 2,
        height=170,
        title="G2 ran in 9% of the cheapest and 26% of the dearest fifth (95% intervals)",
    )
    guard.save_chart(chart=fig3, path=DOCS_DIR / "figure_3.svg")

    # Figure 4: residual histograms per period; no dates, residual in multiples of the threshold.
    d8 = pl.read_parquet(common.DATA_DIR / "d8_resid.parquet")
    periods = {
        "1 before the engine": (
            datetime(2022, 1, 1, tzinfo=UTC),
            common.ENGINE_CONNECTED,
        ),
        "2 G2 output arrives": (common.G2_FIRST, common.G2_LAST),
        "3 after": (
            datetime(2023, 9, 1, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
        ),
        "4 the shortfall has faded": (
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 7, 1, tzinfo=UTC),
        ),
    }
    bins = np.arange(-8, 4.01, 0.5)
    hist_rows = []
    for name, (a, b) in periods.items():
        arr = d8.filter(pl.col("time").is_between(a, b, closed="left"))["resid"].to_numpy()
        h, _ = np.histogram(arr, bins=bins)
        hist_rows += [
            {
                "period": name,
                "scaled": float((bins[i] + 0.25) / abs(common.LEFT_TAIL_MW)),
                "share": float(h[i] / arr.size),
            }
            for i in range(h.size)
        ]
    fig4 = (
        alt.Chart(pl.DataFrame(hist_rows))
        .mark_bar(color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "scaled:Q",
                title="D8 metered minus expected power, in multiples of the shortfall threshold",
            ),
            y=alt.Y("share:Q", title=None),
            row=alt.Row("period:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 120,
            height=70,
            title="D8's residual gains a long negative tail after G2's output stops arriving (share of half-hours)",
        )
    )
    guard.save_chart(chart=fig4, path=DOCS_DIR / "figure_4.svg")

    # Figure 5: shortfall share in the scored year against the earlier history, pooled.
    monthly = pl.read_parquet(common.DATA_DIR / "d8_monthly.parquet")
    scored_start = datetime(2025, 7, 1, tzinfo=UTC)
    names = ["Earlier history", "Scored year"]
    pooled5 = (
        monthly.with_columns(
            part=pl.when(pl.col("time") >= scored_start)
            .then(pl.lit(names[1]))
            .otherwise(pl.lit(names[0]))
        )
        .group_by("part")
        .agg(k=pl.col("k").sum(), n=pl.col("n").sum())
        .with_columns(share=pl.col("k") / pl.col("n"))
    )
    fig5 = (
        alt.Chart(pooled5)
        .mark_bar(color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("part:N", title=None, sort=names, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("share:Q", title="Share of half-hours in shortfall"),
        )
        .properties(
            width=WIDTH // 2,
            height=160,
            title="Experiment 191's scored year against the earlier history (pooled)",
        )
    )
    guard.save_chart(chart=fig5, path=DOCS_DIR / "figure_5.svg")


if __name__ == "__main__":
    main()
