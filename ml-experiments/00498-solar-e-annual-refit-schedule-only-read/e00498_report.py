# ruff: noqa: E501
"""Figures of experiment 498 from the stored forecasts of experiments 430, 493, and 498.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00493-solar-annual-against-quarterly-refits:ml-experiments/00498-solar-e-annual-refit-schedule-only-read \
        uv run --no-sync python ml-experiments/00498-*/e00498_report.py

Farms appear only as Solar A to F.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00430_score as s430  # ty: ignore[unresolved-import]
import e00498_read as rd  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00498-solar-e-annual-refit-schedule-only-read"
)
WIDTH: Final[int] = 440
SEEDS: Final[tuple[int, ...]] = tuple(range(10))


def rules() -> alt.Chart:
    """Return dashed rules at 0 and at the 1% limit."""
    return (
        alt.Chart(alt.Data(values=[{"x": 0}, {"x": 1}]))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(x="x:Q")
    )


def averaged(*, name: str, seeds: tuple[int, ...]) -> pl.DataFrame:
    """Return a Solar E arm with the given seeds averaged."""
    frame = pl.concat([rd.load_seed(name=name, seed=s) for s in seeds])
    return s430.average_seeds(frame=frame, arm=name).with_columns(time_series_id=pl.lit(rd.FARM))


def main() -> None:
    """Write every figure."""
    DOCS.mkdir(parents=True, exist_ok=True)
    a2, q2 = "annual_a2_member", "q_a2_member"
    g0 = "annual_g0_onepass"
    # Per-seed paired contrasts.
    per_seed = []
    for name_a, name_b, label, seeds in (
        (a2, q2, "C2: encoder schedule", SEEDS),
        (g0, q2, "C3: schedule and design", SEEDS[:6]),
    ):
        for s in seeds:
            a, b = rd.paired(a=rd.load_seed(name=name_a, seed=s), b=rd.load_seed(name=name_b, seed=s))
            pct = 100.0 * (rd.score_rows(frame=a)["score"].mean() / rd.score_rows(frame=b)["score"].mean() - 1.0)
            per_seed.append({"contrast": label, "seed": s, "pct": float(pct)})
    seed_table = pl.DataFrame(per_seed)
    seed_table.write_csv(rd.NEW_DIR / "scores" / "per_seed_contrasts.csv")

    # Seed-averaged arms.
    arms = {
        "annual": averaged(name=a2, seeds=SEEDS),
        "quarterly": averaged(name=q2, seeds=SEEDS),
    }
    cut = s430.common_rows(frames=arms)
    rows = {n: rd.score_rows(frame=f) for n, f in cut.items()}
    va, months = rd.by_block(rows=rows["annual"], column="month")
    vb, _ = rd.by_block(rows=rows["quarterly"], column="month")
    point, lo, hi, _ = rd.block_interval(a=va, b=vb)
    summary = pl.DataFrame(
        [
            {"contrast": "C2 ten seeds, month-block 97.5% interval", "pct": point, "lo": lo, "hi": hi}
        ]
    )
    print(summary)
    mean_by_contrast = seed_table.group_by("contrast", maintain_order=True).agg(
        pl.col("pct").mean().alias("mean"), pl.col("pct").min().alias("lo"), pl.col("pct").max().alias("hi")
    )
    dots = (
        alt.Chart(alt.Data(values=seed_table.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60, color=ocf.DATA_BLUE, opacity=0.8)
        .encode(x=alt.X("pct:Q", title="Change in quantile score (%, negative is better)"), y=alt.Y("contrast:N", title=None, sort=None))
    )
    means = (
        alt.Chart(alt.Data(values=mean_by_contrast.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_point(shape="diamond", filled=True, size=160, color=ocf.BRAND_ORANGE)
        .encode(x="mean:Q", y=alt.Y("contrast:N", sort=None))
    )
    interval = (
        alt.Chart(alt.Data(values=summary.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=4, color=ocf.BRAND_ORANGE)
        .encode(x="lo:Q", x2="hi:Q", y=alt.Y("contrast:N", sort=None))
    )
    summary_row = summary.with_columns(contrast=pl.lit("C2: encoder schedule"))
    interval = (
        alt.Chart(alt.Data(values=summary_row.to_dicts()))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=4, color=ocf.BRAND_ORANGE, opacity=0.5)
        .encode(x="lo:Q", x2="hi:Q", y=alt.Y("contrast:N", sort=None))
    )
    (rules() + dots + means + interval).properties(
        width=WIDTH,
        height=120,
        title="Solar E, annual against quarterly refits",
    ).configure_title(fontSize=11, anchor="start").save(str(DOCS / "headline.svg"))

    # By month and by hour of day (ten seeds).
    month_rows = []
    for m in months:
        a_ = rows["annual"].filter(pl.col("month") == m)["score"].mean()
        b_ = rows["quarterly"].filter(pl.col("month") == m)["score"].mean()
        month_rows.append({"month": m, "pct": 100.0 * (a_ / b_ - 1.0)})
    (
        alt.Chart(alt.Data(values=month_rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(x=alt.X("month:N", title="Valid-time month", sort=None), y=alt.Y("pct:Q", title="Change in quantile score (%)"))
        .properties(width=WIDTH, height=160, title="Solar E, C2 by month (ten seeds). The first three months are zero by construction.")
        .configure_title(fontSize=11, anchor="start")
        .save(str(DOCS / "per_month.svg"))
    )
    hour_rows = []
    total = rows["annual"]["score"].sum() - rows["quarterly"]["score"].sum()
    for h in sorted(rows["annual"]["hour"].unique().to_list()):
        sa = rows["annual"].filter(pl.col("hour") == h)
        sb = rows["quarterly"].filter(pl.col("hour") == h)
        hour_rows.append(
            {"hour": h, "share": float((sa["score"].sum() - sb["score"].sum()) / total), "pct": 100.0 * float(sa["score"].mean() / sb["score"].mean() - 1.0)}
        )
    pl.DataFrame(hour_rows).write_csv(rd.NEW_DIR / "scores" / "by_hour_ten_seeds.csv")
    (
        alt.Chart(alt.Data(values=hour_rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(x=alt.X("hour:O", title="Hour of day (UTC)"), y=alt.Y("pct:Q", title="Change in quantile score (%)"))
        .properties(width=WIDTH, height=160, title="Solar E, C2 by hour of day (ten seeds)")
        .configure_title(fontSize=11, anchor="start")
        .save(str(DOCS / "per_hour.svg"))
    )

    # Example forecasts: Solar E, ten-seed average, annual (band) against quarterly (dashed median).
    pieces = []
    for season, init in (("summer", "2026-06-15"), ("winter", "2026-01-15")):
        for name, frame in cut.items():
            sel = frame.filter(pl.col("init_time").dt.strftime("%Y-%m-%d") == init)
            pieces.append(sel.with_columns(season=pl.lit(season), hours=pl.col("lead_hours"), schedule=pl.lit(name)))
    data = pl.concat(pieces)
    for fname, bands, title in (
        ("example_day_ahead.svg", ["d1"], "Day ahead (12 to 36 hours), Solar E, ten-seed average"),
        ("example_10_day.svg", list(c310.BAND_ORDER), "Days 1 to 10, Solar E, ten-seed average"),
    ):
        d = data.filter(pl.col("band").is_in(bands)).sort("hours")
        base = alt.Chart().mark_line().encode(x=alt.X("hours:Q", title="Hours since the run"))  # ty: ignore[unresolved-attribute]
        is_annual = alt.datum.schedule == "annual"
        band90 = base.transform_filter(is_annual).mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(y=alt.Y("q05:Q", title="Output (share of capacity)"), y2="q95:Q")  # ty: ignore[unresolved-attribute]
        median = base.transform_filter(is_annual).mark_line(color=ocf.DATA_BLUE).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        quarterly = base.transform_filter(~is_annual).mark_line(color=ocf.DATA_PURPLE, strokeDash=[4, 3]).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
        actual = base.transform_filter(is_annual).mark_point(size=12, color=ocf.BRAND_ORANGE, filled=True).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
        alt.layer(band90, median, quarterly, actual).properties(width=WIDTH, height=150).facet(
            row=alt.Row("season:N", title=None), data=alt.Data(values=d.to_dicts()), title=title
        ).save(str(DOCS / fname))
    print(np.round(seed_table["pct"].to_numpy(), 2))


if __name__ == "__main__":
    main()
