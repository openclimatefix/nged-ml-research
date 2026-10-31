# ruff: noqa: E501
"""Tables and figures for experiment 430: the stage contrasts, per-member quantiles, tails, examples.

Run after every fit has finished:

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00430-*/e00430_report.py

Writes `scores/headline.csv`, `scores/per_member.csv`, `scores/tails.csv`, `scores/checks.csv`, and
the SVG figures under `docs/ml-experiments/00430-*/`. Farms appear only as labels A to F, and the
example forecasts are drawn against hours since the run, with no calendar date.
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00310_common as c310  # ty: ignore[unresolved-import]
import e00430_common as common
import e00430_score as sc
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
)
WIDTH: Final[int] = 440
HEADLINE: Final[tuple[tuple[str, str, str, str], ...]] = (
    # (stage label, arm, reference, row set)
    ("A2 pooled-encoder member route vs one pass", "a2_member", "g0_onepass", "d3_10"),
    ("A2 mean only vs one pass", "a2_mean_only", "g0_onepass", "d3_10"),
    ("A2 member-wise vs one pass", "a2_memberwise", "g0_onepass", "d3_10"),
    ("A6 pretrain and fine-tune vs ensemble only", "a6_finetune", "a6_ens_only", "all"),
    ("A6 ERA5 and CAMS only vs ensemble only", "a6_pretrain_only", "a6_ens_only", "all"),
    ("A5 frozen plant vs plain", "a5_frozen", "a5_plain", "d6_10"),
    ("A5 trained plant vs plain", "a5_e2e", "a5_plain", "d6_10"),
    ("A5 clear-sky column alone vs one pass", "a5_plain", "g0_onepass", "d6_10"),
    ("A7 point masses vs clipped mixture", "a7_atoms", "g0_onepass", "all"),
)


def headline() -> pl.DataFrame:
    """Return the headline contrasts with their 99% intervals."""
    rows = []
    for label, arm, ref, band in HEADLINE:
        arms = {n: sc.load_mine(arm=n) for n in {arm, ref}}
        table = sc.contrast_table(arms=arms, pairs=[(arm, ref)])
        for chosen in dict.fromkeys((band, "d1", "all")):
            r = table.filter(pl.col("band") == chosen).row(0, named=True)
            rows.append(
                {
                    "label": label,
                    "arm": arm,
                    "ref": ref,
                    "set": chosen,
                    "headline": chosen == band,
                    **{
                        k: r[k]
                        for k in (
                            "score_arm",
                            "score_ref",
                            "pct_change",
                            "lo99",
                            "hi99",
                            "pin01_diff",
                            "pin99_diff",
                        )
                    },
                }
            )
    return pl.DataFrame(rows)


def per_member() -> pl.DataFrame:
    """Return the member-wise route's per-member quantile scores beside the pooled mixture's."""
    pooled_cov = np.zeros(len(c310.LEVELS))
    member_cov = np.zeros(len(c310.LEVELS))
    pooled_pin = np.zeros(len(c310.LEVELS))
    member_pin = np.zeros(len(c310.LEVELS))
    n = 0
    levels = np.array(c310.LEVELS)
    for path in sorted((common.PRED_DIR / "a2_memberwise").glob("s*_*.parquet")):
        table = pl.read_parquet(path)
        z = np.load(path.with_name("paths_" + path.name.replace(".parquet", ".npz")))
        member_q = z["member_q"].astype(np.float64)
        y = table["actual"].to_numpy()[:, None, None]
        pooled = table.select(sc.QCOLS).to_numpy().astype(np.float64)[:, None, :]
        for q, cov_sum, pin_sum in (
            (member_q, member_cov, member_pin),
            (pooled, pooled_cov, pooled_pin),
        ):
            err = y - q
            np.add(cov_sum, (y <= q).mean(axis=1).sum(axis=0), out=cov_sum)
            pinball = np.maximum(levels * err, (levels - 1.0) * err).mean(axis=1).sum(axis=0)
            np.add(pin_sum, pinball, out=pin_sum)
        n += len(table)
    return pl.DataFrame(
        {
            "level": levels,
            "coverage_member": member_cov / n,
            "coverage_pooled": pooled_cov / n,
            "pinball_member": 100 * member_pin / n,
            "pinball_pooled": 100 * pooled_pin / n,
        }
    )


def tails() -> pl.DataFrame:
    """Return tail pinball loss and coverage of the main arms over all bands."""
    names = [
        "g0_onepass",
        "a2_mean_only",
        "a2_member",
        "a2_memberwise",
        "a6_ens_only",
        "a6_finetune",
        "a6_pretrain_only",
        "a5_plain",
        "a5_frozen",
        "a5_e2e",
        "a7_atoms",
    ]
    return sc.absolute_table(arms={n: sc.load_mine(arm=n) for n in names})


def checks() -> pl.DataFrame:
    """Return the audit numbers: median minus mean forecast, and the mean-forecast error beside the score."""
    arms = {n: sc.load_mine(arm=n) for n in ("g0_onepass", "a2_member", "a5_plain", "a7_atoms")}
    aligned = sc.common_rows(frames=arms)
    out = []
    for name, frame in aligned.items():
        rows = sc.losses(frame=frame)
        per_farm = (
            rows.group_by("time_series_id")
            .agg(
                pl.col("median_minus_mean").mean(), pl.col("ae_mean").mean(), pl.col("score").mean()
            )
            .with_columns(arm=pl.lit(name))
        )
        out.append(per_farm)
    return pl.concat(out).sort("arm", "time_series_id")


def save(chart: alt.Chart | alt.LayerChart | alt.VConcatChart, name: str) -> None:
    """Save a chart as SVG in the docs folder."""
    chart.save(str(DOCS / name))


def headline_figure(table: pl.DataFrame) -> None:
    """Draw every stage contrast with its 99% interval and the 1% margin."""
    data = table.filter(pl.col("headline"))
    frame = data.to_dicts()
    base = alt.Chart(alt.Data(values=frame)).encode(
        y=alt.Y("label:N", sort=[h[0] for h in HEADLINE], title=None, axis=alt.Axis(labelLimit=330))
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.DATA_BLUE).encode(
        x=alt.X("lo99:Q", title="Change in quantile score (%, negative is better)"), x2="hi99:Q"
    )
    points = base.mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE).encode(x="pct_change:Q")
    zero = (
        alt.Chart(alt.Data(values=[{"x": 0}, {"x": -1}, {"x": 1}]))
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(x="x:Q")
    )
    save(
        (zero + bars + points)
        .properties(
            width=WIDTH,
            height=260,
            title="Each ablation against its reference, with 99% intervals; dashed lines mark 0 and the 1% margin",
        )
        .configure_title(fontSize=12, anchor="start"),
        "headline.svg",
    )


def coherence_figure() -> None:
    """Draw the energy and variogram score of each route against one pass with ECC."""
    path = max((common.DATA_DIR / "scores").glob("coherence_*.csv"))
    t = (
        pl.read_csv(path)
        .filter(pl.col("set") == "joint")
        .with_columns(label=pl.col("arm") + " + " + pl.col("route"))
    )
    long = pl.concat(
        [
            t.select(
                "label",
                pl.lit("Energy score").alias("metric"),
                pl.col("energy_pct_vs_ref").alias("change"),
                pl.col("energy_lo99").alias("lo"),
                pl.col("energy_hi99").alias("hi"),
            ),
            t.select(
                "label",
                pl.lit("Variogram score").alias("metric"),
                pl.col("variogram_pct_vs_ref").alias("change"),
                pl.lit(None, dtype=pl.Float64).alias("lo"),
                pl.lit(None, dtype=pl.Float64).alias("hi"),
            ),
        ]
    )
    base = alt.Chart(alt.Data(values=long.to_dicts())).encode(
        y=alt.Y("label:N", title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "metric:N",
            scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(
        x=alt.X("change:Q", title="Change against one pass with ECC (%, negative is better)"),
        yOffset="metric:N",
    )
    bars = (
        base.transform_filter("datum.lo != null")
        .mark_rule(strokeWidth=2)
        .encode(x="lo:Q", x2="hi:Q", yOffset="metric:N")
    )
    save(
        (bars + points)
        .properties(
            width=WIDTH,
            height=300,
            title="Coherence of 51 member paths over six farms and 10 days (energy score has 99% intervals)",
        )
        .configure_title(fontSize=12, anchor="start"),
        "coherence.svg",
    )


def example_figures() -> None:
    """Draw a day-ahead and a 10-day forecast of farm A from the one-pass network (seed average)."""
    frame = sc.load_mine(arm="g0_onepass").filter(pl.col("time_series_id") == "A")
    pieces = []
    for season, init in (("summer", "2026-06-15"), ("winter", "2026-01-15")):
        rows = frame.filter(pl.col("init_time").dt.strftime("%Y-%m-%d") == init)
        pieces.append(rows.with_columns(season=pl.lit(season), hours=pl.col("lead_hours")))
    data = pl.concat(pieces)
    for name, bands, title in (
        ("example_day_ahead.svg", ["d1"], "Day ahead (12 to 36 hours), farm A"),
        ("example_10_day.svg", list(c310.BAND_ORDER), "Days 1 to 10, farm A"),
    ):
        d = data.filter(pl.col("band").is_in(bands)).sort("hours")
        base = alt.Chart(alt.Data(values=d.to_dicts())).encode(
            x=alt.X("hours:Q", title="Hours since the run")
        )
        band90 = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(
            y=alt.Y("q05:Q", title="Output (share of capacity)"), y2="q95:Q"
        )
        band50 = base.mark_area(opacity=0.35, color=ocf.DATA_BLUE).encode(y="q25:Q", y2="q75:Q")
        median = base.mark_line(color=ocf.DATA_BLUE).encode(y="q50:Q")
        actual = base.mark_point(size=12, color=ocf.BRAND_ORANGE, filled=True).encode(y="actual:Q")
        chart = (
            alt.layer(band90, band50, median, actual)
            .properties(width=WIDTH, height=150)
            .facet(row=alt.Row("season:N", title=None), title=title)
        )
        save(chart, name)


def main() -> None:
    """Write every table and figure."""
    out = common.DATA_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(14)
    head = headline()
    head.write_csv(out / "headline.csv")
    print(head.with_columns(pl.col(pl.Float64).round(3)))
    members = per_member()
    members.write_csv(out / "per_member.csv")
    print(members.with_columns(pl.col(pl.Float64).round(4)))
    tail = tails()
    tail.write_csv(out / "tails.csv")
    print(tail.with_columns(pl.col(pl.Float64).round(4)))
    check = checks()
    check.write_csv(out / "checks.csv")
    print(check.with_columns(pl.col(pl.Float64).round(4)))
    headline_figure(head)
    coherence_figure()
    example_figures()


if __name__ == "__main__":
    main()
