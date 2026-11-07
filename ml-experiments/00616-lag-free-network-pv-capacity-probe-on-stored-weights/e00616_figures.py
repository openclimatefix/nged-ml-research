# ruff: noqa: E501
"""Experiment 616: the figures of the write-up, drawn from the private result tables (no generator series).

Usage: python e00616_figures.py DOCS_DIR PNG_DIR
"""

import sys
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00616_analyse as an
import e00616_regress as reg
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

OUT = reg.OUT
WIDTH: Final[int] = 640
SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")


def legend() -> alt.Legend:
    """Return a legend placed above the plot."""
    return alt.Legend(orient="top", title=None, labelLimit=640, columns=1)


def save(*, chart: alt.TopLevelMixin, name: str, docs: Path, png: Path) -> None:
    """Write the SVG for the docs and a 2x PNG for the figure review."""
    chart.save(docs / f"{name}.svg")
    chart.save(png / f"{name}.png", scale_factor=2)


def fig_headline(*, tables: pl.DataFrame, refs: pl.DataFrame) -> alt.Chart:
    """Probe capacity per Primary beside experiments 474 and 479 (season-year July 2025 to June 2026)."""
    t = (
        tables.filter(
            (pl.col("fold") == "2026-06")
            & (pl.col("arm") == "D5")
            & (pl.col("variant") == "T")
            & (pl.col("date_set") == "SY2025_dates")
        )
        .select(site="site", lo="p10", mid="mean", hi="p90")
        .with_columns(source=pl.lit("Probe of the stored network (seeds and bootstrap)"))
    )
    r4 = (
        refs.filter(pl.col("season_year") == 2025)
        .select(site="site", lo="r4_p10", mid="r4_p50", hi="r4_p90")
        .with_columns(source=pl.lit("Experiment 474, rung R4 with cloud terms"))
    )
    r479 = (
        refs.filter(pl.col("season_year") == 2025)
        .select(site="site", lo="r479_p05", mid="r479_p50", hi="r479_p95")
        .with_columns(source=pl.lit("Experiment 479, anomaly regression with cloud terms"))
    )
    d = pl.concat([t, r4, r479], how="vertical_relaxed").filter(pl.col("site").is_in(SITES))
    domain = list(d["source"].unique(maintain_order=True))
    colour = alt.Color(
        "source:N",
        scale=alt.Scale(domain=domain, range=[theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_SKY]),
        legend=legend(),
    )
    base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("site:N", sort=list(SITES), title="Primary (six MW Primaries)"),
        xOffset=alt.XOffset("source:N", sort=domain),
        color=colour,
    )
    bar = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lo:Q", title="AC capacity estimate (MW)"), y2="hi:Q"
    )
    dot = base.mark_point(filled=True, size=70).encode(y="mid:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bar, dot).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            [
                "The probe matches the earlier estimates at D5 and D9, and is 15 to 35% below them at the other four"
            ],
            anchor="start",
            subtitle=[
                "Probe dot: mean over seeds, sky-and-temperature fit, dates July 2025 to June 2026. Other dots: medians; both earlier estimates use cloud terms.",
                "Lines: 10th to 90th percentile (5th to 95th for experiment 479). The probe line counts seeds and donor days only; the others count other uncertainty.",
            ],
        ),
    )


def fig_dates(*, per_date: pl.DataFrame) -> alt.Chart:
    """Each Primary's capacity on each probe date as a share of its own mean, to show the seasonal pattern."""
    d = per_date.filter(
        (pl.col("fold") == "2026-06")
        & (pl.col("arm") == "D5")
        & pl.col("site").is_in(SITES)
        & pl.col("capacity_mw").is_not_null()
    )
    d = d.with_columns(share=pl.col("capacity_mw") / pl.col("capacity_mw").mean().over("site"))
    d = d.with_columns(
        window=pl.when(pl.col("date") < date(2026, 1, 1))
        .then(pl.lit("2025"))
        .otherwise(pl.lit("2026"))
    )
    agg = (
        d.group_by("date", "window")
        .agg(median=pl.col("share").median(), low=pl.col("share").min(), high=pl.col("share").max())
        .sort("date")
    )
    band = (
        alt.Chart(agg.to_pandas())
        .mark_area(opacity=0.35, color=theme.BRAND_ORANGE_LIGHT)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "date:T",
                title="Probe date (Tuesdays; 2025-04 to 2025-09, then 2026-04 to 2026-05)",
                axis=alt.Axis(format="%b %Y"),
            ),
            y=alt.Y(
                "low:Q",
                title="Capacity as a share of the Primary's mean",
                scale=alt.Scale(zero=False),
            ),
            y2="high:Q",
            detail="window:N",
        )
    )
    line = (
        alt.Chart(agg.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(
            point=alt.OverlayMarkDef(color=theme.BRAND_ORANGE, filled=True),
            color=theme.BRAND_ORANGE,
        )
        .encode(x="date:T", y="median:Q", detail="window:N")
    )  # ty: ignore[unresolved-attribute]
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.DATA_BLUE, strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    return alt.layer(band, line, one).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            ["The probed capacity is largest in April and smallest in June and July"],
            anchor="start",
            subtitle=[
                "Orange line: median over the six Primaries. Pale orange band: lowest to highest Primary. Blue dashed line: each Primary's own mean. No probes between September 2025 and April 2026."
            ],
        ),
    )


def fig_placebo(*, tables: pl.DataFrame) -> alt.Chart:
    """The placebos and the joint fit as a share of the real capacity."""
    t = tables.filter(
        (pl.col("fold") == "2026-06")
        & (pl.col("arm") == "D5")
        & (pl.col("date_set") == "SY2025_dates")
        & pl.col("site").is_in(SITES)
    )
    real = t.filter(pl.col("variant") == "T").select("site", real="mean")
    names = {
        "P_shuffle": "Shuffled sky (placebo)",
        "P_profile": "Climatological sky alone (placebo)",
        "J_profile": "Climatological sky's slope when fitted beside the real sky",
    }
    d = (
        t.filter(pl.col("variant").is_in(list(names)))
        .join(real, on="site")
        .with_columns(share=pl.col("mean") / pl.col("real"), label=pl.col("variant").replace(names))
    )
    domain = list(names.values())
    base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("site:N", sort=list(SITES), title="Primary (six MW Primaries)"),
        xOffset=alt.XOffset("label:N", sort=domain),
        y=alt.Y(
            "share:Q", title="Placebo capacity as a share of the real capacity (smaller is better)"
        ),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=domain, range=[theme.DATA_PURPLE, theme.DATA_BLUE, theme.DATA_GREEN]
            ),
            legend=legend(),
        ),
    )
    limit = (
        alt.Chart(pl.DataFrame({"y": [0.2]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_rule(color=theme.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    label = (
        alt.Chart(pl.DataFrame({"y": [0.2], "t": ["0.20 limit"]}).to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=4, dy=-6, color=theme.BRAND_ORANGE)
        .encode(y="y:Q", text="t:N", x=alt.value(0))
    )  # ty: ignore[unresolved-attribute]
    return alt.layer(base.mark_point(filled=True, size=80), limit, label).properties(  # ty: ignore[invalid-argument-type,invalid-return-type]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            [
                "A shuffled sky reads zero; the seasonal trend of the sky alone reads most of the real capacity"
            ],
            anchor="start",
            subtitle=[
                "A placebo above the orange dashed 0.20 line counts as a failure. Stored 2026-06 network, probe dates in July 2025 to June 2026."
            ],
        ),
    )


def fig_variants(*, tables: pl.DataFrame) -> alt.Chart:
    """The capacity under each regression variant."""
    names = {
        "noT": "Sky only",
        "T": "Sky and temperature (headline)",
        "tm": "Donors within 2 °C of the median temperature",
        "cloud": "Sky, temperature, and two cloud terms",
    }
    d = tables.filter(
        (pl.col("fold") == "2026-06")
        & (pl.col("arm") == "D5")
        & (pl.col("date_set") == "SY2025_dates")
        & pl.col("site").is_in(SITES)
        & pl.col("variant").is_in(list(names))
    )
    d = d.with_columns(label=pl.col("variant").replace(names))
    domain = list(names.values())
    base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("site:N", sort=list(SITES), title="Primary (six MW Primaries)"),
        xOffset=alt.XOffset("label:N", sort=domain),
        y=alt.Y("mean:Q", title="AC capacity estimate (MW)"),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=domain,
                range=[theme.DATA_SKY, theme.BRAND_ORANGE, theme.DATA_BLUE, theme.DATA_PURPLE],
            ),
            legend=legend(),
        ),
    )
    return base.mark_point(filled=True, size=80).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            [
                "Adding temperature changes the capacity little; the cloud terms change it a lot at D5"
            ],
            anchor="start",
            subtitle=["Stored 2026-06 network, seed mean, probe dates in July 2025 to June 2026."],
        ),
    )


def fig_folds(*, tables: pl.DataFrame) -> alt.Chart:
    """The capacity from each fold's stored network, for arms D5 and M0."""
    panels = []
    for arm, label in (
        ("D5", "Design-5 networks (experiment 565)"),
        ("M0", "M0 networks (experiment 554)"),
    ):
        d = tables.filter(
            (pl.col("arm") == arm)
            & (pl.col("variant") == "T")
            & (pl.col("date_set") == "all_dates")
            & pl.col("site").is_in(SITES)
        )
        base = alt.Chart(d.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=list(SITES), title="Primary (six MW Primaries)"),
            xOffset=alt.XOffset("fold:N", sort=list(FOLDS)),
            y=alt.Y("mean:Q", title="AC capacity estimate (MW)", scale=alt.Scale(domain=[0, 4])),
            color=alt.Color(
                "fold:N",
                title="Fold (test month)",
                scale=alt.Scale(
                    domain=list(FOLDS),
                    range=[
                        theme.DATA_SKY,
                        theme.DATA_PURPLE,
                        theme.DATA_GREEN,
                        theme.DATA_BLUE,
                        theme.BRAND_ORANGE,
                    ],
                ),
                legend=alt.Legend(
                    orient="top", title="Fold (test month)", labelLimit=640, direction="horizontal"
                ),
            ),
        )
        panels.append(
            base.mark_point(filled=True, size=70).properties(  # ty: ignore[unresolved-attribute]
                width=WIDTH, height=280, title=alt.TitleParams(label, anchor="start")
            )
        )
    return alt.vconcat(*panels).properties(  # ty: ignore[invalid-return-type]
        title=alt.TitleParams(
            ["The capacity estimate barely depends on the fold or the network design"],
            anchor="start",
            subtitle=[
                "Mean over the 10 seeds and over every probe date a fold can be probed on (12 dates; 15 for fold 2026-06, 14 at D5), so values differ from figure 1."
            ],
        )
    )


def response_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the binned response (Primary D11) and the hour profile of the contrast, from the 2026-06 fold."""
    rows = pl.read_parquet(OUT / "probe_2026-06.parquet").with_row_index("global_row")
    y = np.load(OUT / "probe_2026-06_y.npy")[0].mean(0)
    rows = an.group_table(rows=rows, table=reg.unit_table())
    import e00397_prep as prep  # ty: ignore[unresolved-import]

    pid = prep.primaries()["site"].to_list().index("D11")
    g = rows.filter(
        (pl.col("primary") == pid) & pl.col("date").is_between(date(2025, 7, 1), date(2026, 6, 30))
    )
    idx = g["global_row"].to_numpy()
    yy, u = y[idx].astype(float), g["u"].to_numpy()
    key = np.unique(
        g["date"].cast(pl.String).to_numpy().astype(str)
        + "_"
        + g["slot"].cast(pl.String).to_numpy().astype(str),
        return_inverse=True,
    )[1]
    n = np.bincount(key)
    yd, ud = (
        yy - (np.bincount(key, weights=yy) / n)[key],
        u - (np.bincount(key, weights=u) / n)[key],
    )
    edges = np.quantile(ud, np.linspace(0, 1, 11))
    which = np.clip(np.searchsorted(edges, ud, side="right") - 1, 0, 9)
    binned = pl.DataFrame(
        {
            "u": [ud[which == b].mean() for b in range(10)],
            "y": [yd[which == b].mean() for b in range(10)],
        }
    )
    slope = np.polyfit(ud, yd, 1)[0]
    binned = binned.with_columns(fit=pl.col("u") * slope)
    slot, dates = g["slot"].to_numpy(), g["date"].to_numpy()
    prof = []
    for s in np.unique(slot):
        dy, du = [], []
        for d in np.unique(dates):
            m = (slot == s) & (dates == d)
            if m.sum() < 30:
                continue
            q1, q2 = np.quantile(u[m], [1 / 3, 2 / 3])
            dy.append(yy[m & (u <= q1)].mean() - yy[m & (u >= q2)].mean())
            du.append(u[m & (u >= q2)].mean() - u[m & (u <= q1)].mean())
        if dy:
            prof.append(
                {
                    "hour": s / 2,
                    "response": float(np.median(dy)),
                    "unit": float(np.median(du)) * -slope,
                }
            )
    return binned, pl.DataFrame(prof)


def fig_response(*, binned: pl.DataFrame) -> alt.Chart:
    """The slot-demeaned response against the unit curve, binned, with the fitted line."""
    pts = (
        alt.Chart(binned.to_pandas())
        .mark_point(filled=True, size=80, color=theme.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "u:Q",
                title="Unit PV curve u (MW per MW of capacity), minus its slot mean",
                axis=alt.Axis(tickCount=6, format=".2f"),
            ),
            y=alt.Y(
                "y:Q", title="Probed net demand, minus its slot mean (MW; lower means more PV)"
            ),
        )
    )
    line = alt.Chart(binned.to_pandas()).mark_line(color=theme.DATA_BLUE).encode(x="u:Q", y="fit:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(pts, line).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            ["Probed net demand falls almost in a straight line as the donor sky gets sunnier"],
            anchor="start",
            subtitle=[
                "Primary D11, stored design-5 network (seed mean), probe dates in July 2025 to June 2026. Orange dots: ten equal-count bins. Blue line: the fitted slope."
            ],
        ),
    )


def fig_profile(*, prof: pl.DataFrame) -> alt.Chart:
    """The hour profile of the sunny-minus-dull contrast against the unit curve's, scaled by the fitted capacity."""
    long = prof.unpivot(index="hour", variable_name="series", value_name="mw").with_columns(
        series=pl.col("series").replace(
            {"response": "Probed response", "unit": "Unit PV curve times the fitted capacity"}
        )
    )
    domain = ["Probed response", "Unit PV curve times the fitted capacity"]
    base = alt.Chart(long.to_pandas()).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "hour:Q",
            title="Hour of day (UTC)",
            scale=alt.Scale(domain=[4, 20]),
            axis=alt.Axis(values=[4, 8, 12, 16, 20]),
        ),
        y=alt.Y("mw:Q", title="Fall in net demand, dull minus sunny third of donors (MW)"),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=domain, range=[theme.BRAND_ORANGE, theme.DATA_BLUE]),
            legend=legend(),
        ),
    )
    return alt.layer(base.mark_line(), base.mark_point(filled=True)).properties(  # ty: ignore[invalid-argument-type,invalid-return-type]
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            [
                "The probed response follows the PV unit curve from 9 h to 16 h, and lags it before 9 h"
            ],
            anchor="start",
            subtitle=[
                "Primary D11, median over the probe dates in July 2025 to June 2026; seed-mean design-5 network. Both lines are estimates."
            ],
        ),
    )


def main() -> None:
    """Draw every figure."""
    docs, png = Path(sys.argv[1]), Path(sys.argv[2])
    docs.mkdir(parents=True, exist_ok=True)
    png.mkdir(parents=True, exist_ok=True)
    tables = pl.read_parquet(OUT / "tables.parquet")
    refs = pl.read_parquet(OUT / "references.parquet")
    per_date = pl.read_parquet(OUT / "per_date.parquet")
    binned, prof = response_data()
    for name, chart in (
        ("figure_1_capacity_per_primary", fig_headline(tables=tables, refs=refs)),
        ("figure_2_response_to_the_sky", fig_response(binned=binned)),
        ("figure_3_hour_profile", fig_profile(prof=prof)),
        ("figure_4_placebos", fig_placebo(tables=tables)),
        ("figure_5_regression_variants", fig_variants(tables=tables)),
        ("figure_6_probe_dates", fig_dates(per_date=per_date)),
        ("figure_7_folds_and_designs", fig_folds(tables=tables)),
    ):
        save(chart=chart, name=name, docs=docs, png=png)
        print("saved", name, flush=True)


if __name__ == "__main__":
    main()
