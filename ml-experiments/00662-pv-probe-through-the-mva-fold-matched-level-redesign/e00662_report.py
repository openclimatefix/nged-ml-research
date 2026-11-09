# ruff: noqa: E501, PERF401
"""Experiment 662: the printed report and the figures of the write-up.

Reads `work/analysis/recoveries_2026-06.csv` and the probe parquets. Every number the write-up
quotes is printed here. The six signed-MW Primaries carry the injected worlds; the eight MVA
Primaries are read from their uninjected twin networks only.

Usage: python e00662_report.py
"""

from pathlib import Path

import altair as alt
import e00662_common as k
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00662-pv-probe-through-the-mva-fold-matched-level-redesign"
)
CUTOFF = "2026-06"
W = 640
METHODS = {
    "L0": ("Linear (L0)", theme.BRAND_ORANGE),
    "L1": ("One-sided linear (L1)", theme.DATA_GREEN),
    "FA": ("Fold model (FA)", theme.DATA_BLUE),
}
LEVEL_LABEL = {"l05": "0.5", "l1": "1", "l2": "2", "l6": "6"}


def six_levels(rec: pl.DataFrame) -> pl.DataFrame:
    """Return the PF-arm fleet recoveries: median over seeds and rotations per Primary, level, and method."""
    sel = rec.filter(pl.col("condition").str.starts_with("pf_fleet_") & pl.col("site").is_in(k.SIX))
    return (
        sel.with_columns(level_label=pl.col("level").cast(pl.String))
        .group_by("site", "level", "method")
        .agg(rec=pl.col("rec").median())
    )


def fig_levels(rec: pl.DataFrame) -> None:
    """Draw recovery by injection level and method.

    One dot per Primary, and a line through the median of the six.
    """
    d = (
        six_levels(rec)
        .filter(pl.col("method").is_in(list(METHODS)))
        .with_columns(method_label=pl.col("method").replace({m: v[0] for m, v in METHODS.items()}))
    )
    med = d.group_by("level", "method_label").agg(rec=pl.col("rec").median())
    scale = alt.Scale(
        domain=[v[0] for v in METHODS.values()], range=[v[1] for v in METHODS.values()]
    )
    x = alt.X(
        "level:O",
        title="Injected capacity, as a multiple of the Primary's median sunny-midday reading",
        axis=alt.Axis(labelAngle=0),
    )
    y = alt.Y("rec:Q", title="Share of injected MW recovered", scale=alt.Scale(domain=[-1, 1.2]))
    colour = alt.Color(
        "method_label:N", scale=scale, title=None, legend=alt.Legend(orient="top", columns=1)
    )
    dots = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40, opacity=0.5, aria=False)
        .encode(x=x, y=y, color=colour)
    )
    line = alt.Chart(med).mark_line(strokeWidth=2.5, aria=False).encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
    ref = (
        alt.Chart(pl.DataFrame({"y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.5, color=theme.BLACK_2)
        .encode(y="y:Q")
    )
    chart = (ref + dots + line).properties(
        width=W,
        height=300,
        title=alt.TitleParams(
            "Recovery of injected solar capacity, six megawatt Primaries converted to apparent power",
            subtitle=[
                "Power-factor-0.97 world, 2026-06 networks, median of 3 seeds and 2 farm rotations per Primary. Dot: one MW Primary. Line: median of the six. Dashed lines: 0 (nothing found) and 1 (all found)."
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    chart.save(DOCS / "recovery_by_level.svg")


def fig_placebos(rec: pl.DataFrame) -> None:
    """Capacity read from each placebo, per injected MW of the placebo's size."""
    names = {
        "pf_clim": "Climatological profile",
        "pf_boxcar": "Load-shift boxcar",
        "pf_cloud": "Cloud-following load",
        "pf_sunload": "Sun-rising load",
        "pf_fleet_l05_r0": "Real fleet, 0.5× (reference)",
    }
    d = rec.filter(
        pl.col("condition").is_in(list(names))
        & pl.col("site").is_in(k.SIX)
        & pl.col("method").is_in(["L0", "FA"])
    )
    d = (
        d.group_by("site", "condition", "method")
        .agg(rec=pl.col("rec").median())
        .with_columns(
            case=pl.col("condition").replace(names),
            method_label=pl.col("method").replace({m: METHODS[m][0] for m in ("L0", "FA")}),
        )
    )
    scale = alt.Scale(
        domain=[METHODS[m][0] for m in ("L0", "FA")], range=[METHODS[m][1] for m in ("L0", "FA")]
    )
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50, opacity=0.7, aria=False)
        .encode(
            y=alt.Y("case:N", sort=list(names.values()), title=None),
            x=alt.X(
                "rec:Q",
                title="Capacity read, as a share of the size of the added output or load",
                scale=alt.Scale(domain=[-1, 1.2]),
            ),
            color=alt.Color(
                "method_label:N", scale=scale, title=None, legend=alt.Legend(orient="top")
            ),
            yOffset="method_label:N",
        )
        .properties(
            width=W,
            height=240,
            title=alt.TitleParams(
                "Capacity read from worlds with no real solar farm",
                subtitle=[
                    "2026-06 networks, one dot per Primary (median of 3 seeds). A placebo should read near 0; the reference row is a real fleet."
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            ),
        )
    )
    chart.save(DOCS / "placebos.svg")


def real_table(res: pl.DataFrame) -> pl.DataFrame:
    """Return the capacity of each Primary's uninjected PF-arm twin network, per method.

    Each capacity is a p10, p50, and p90 over seeds and bootstrap draws.
    """
    tw = res.filter(pl.col("condition") == "pf_twin")
    rows = []
    for site in (*k.SIX, *k.NATIVE, *k.FOLDERS):
        t = tw.filter(pl.col("site") == site)
        for meth in ("L0", "L1", "F4", "F1", "FA"):
            draws = np.concatenate([np.array(r[f"{meth}_boot"]) for r in t.iter_rows(named=True)])
            draws = draws[np.isfinite(draws)]
            rows.append(
                {
                    "site": site,
                    "method": meth,
                    "p10": float(np.percentile(draws, 10)),
                    "p50": float(t[meth].median()),  # ty: ignore[invalid-argument-type]
                    "p90": float(np.percentile(draws, 90)),
                    "seed_sd": float(t[meth].std()),  # ty: ignore[invalid-argument-type]
                }
            )
    return pl.DataFrame(rows)


def fig_real(table: pl.DataFrame) -> None:
    """Capacity at all 14 Primaries from the linear probe and the fold model."""
    label = {
        site: (f"{site} (report only)" if site in k.FOLDERS else site)
        for site in (*k.SIX, *k.NATIVE, *k.FOLDERS)
    }
    order = [label[site] for site in (*k.SIX, *k.NATIVE, *k.FOLDERS)]
    d = (
        table.with_columns(site=pl.col("site").replace(label))
        .filter(pl.col("method").is_in(["L0", "FA"]))
        .with_columns(
            method_label=pl.col("method").replace({m: METHODS[m][0] for m in ("L0", "FA")}),
            group=pl.when(pl.col("site").is_in(list(k.SIX)))
            .then(pl.lit("MW Primary"))
            .when(pl.col("site").is_in(list(k.NATIVE)))
            .then(pl.lit("MVA, does not fold"))
            .otherwise(pl.lit("MVA, folds (report only)")),
        )
    )
    scale = alt.Scale(
        domain=[METHODS[m][0] for m in ("L0", "FA")], range=[METHODS[m][1] for m in ("L0", "FA")]
    )
    base = alt.Chart(d).encode(
        x=alt.X(
            "site:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelAlign="right")
        ),
        color=alt.Color("method_label:N", scale=scale, title=None, legend=alt.Legend(orient="top")),
        xOffset="method_label:N",
    )
    chart = (
        base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("p10:Q", title="Capacity (MW of AC, raw, uncalibrated)"), y2="p90:Q"
        )
        + base.mark_point(filled=True, size=50, aria=False).encode(y="p50:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=W,
        height=300,
        title=alt.TitleParams(
            "Capacity read from the uninjected 2026-06 networks",
            subtitle=[
                "Dot: median of 3 seeds. Line: 10th to 90th percentile of bootstrap draws over days. MW Primaries: D5 to D16; MVA Primaries: D2 to D22. D8, D12, and D22 fold (report only)."
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    chart.save(DOCS / "capacity_at_fourteen_primaries.svg")


def fig_cross(res: pl.DataFrame) -> None:
    """Share of probe cells the fold model flags as crossing zero, by month and half-hour, at D12 and D22."""
    parts = []
    for site in ("D12", "D22"):
        for r in res.filter(
            (pl.col("condition") == "pf_twin") & (pl.col("site") == site)
        ).iter_rows(named=True):
            parts.append(
                pl.DataFrame(
                    {
                        "site": site,
                        "month": r["cell_month"],
                        "hour": [h / 2 for h in r["cell_hh"]],
                        "cross": [float(c) for c in r["FA_cross"]],
                    }
                )
            )
    d = pl.concat(parts).group_by("site", "month", "hour").agg(cross=pl.col("cross").mean())
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_rect(aria=False)
        .encode(
            x=alt.X("month:O", title="Month of the probe date (all 2025)"),
            y=alt.Y(
                "hour:O",
                title="Hour of day (UTC)",
                sort="descending",
                axis=alt.Axis(values=[6, 9, 12, 15, 18]),
            ),
            color=alt.Color(
                "cross:Q",
                title="Share of cells flagged",
                scale=alt.Scale(domain=[0, 1], range=["#E6E6E6", theme.BRAND_ORANGE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=W, height=170)
        .facet(row=alt.Row("site:N", title=None))
        .properties(
            title=alt.TitleParams(
                "Where the fold model says the reading crosses zero (unvalidated)",
                subtitle=[
                    "Uninjected 2026-06 networks, 3 seeds. Orange: fold model (FA) finds demand below solar output in the probe cell. Bar B6 failed, so treat the map as unvalidated."
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            )
        )
    )
    chart.save(DOCS / "crossing_map.svg")


def main() -> None:
    """Print the numbers and write the figures."""
    DOCS.mkdir(parents=True, exist_ok=True)
    rec = pl.read_csv(k.WORK / "analysis" / f"recoveries_{CUTOFF}.csv")
    res = pl.concat(
        [pl.read_parquet(p) for p in sorted((k.WORK / "probes" / CUTOFF).glob("*.parquet"))],
        how="diagonal_relaxed",
    )
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    six = rec.filter(pl.col("site").is_in(k.SIX))
    print("== Median recovery over the six, by condition and method ==")
    print(
        six.group_by("condition", "method")
        .agg(rec=pl.col("rec").median())
        .pivot(on="method", index="condition", values="rec")
        .sort("condition")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Native (MVA, does not fold) Primaries, PF arm ==")
    nat = rec.filter(
        pl.col("site").is_in(k.NATIVE)
        & pl.col("condition").is_in(["pf_fleet_l05_r0", "pf_fleet_l1_r0", "pf_fleet_l6_r0"])
    )
    print(
        nat.group_by("condition", "method")
        .agg(rec=pl.col("rec").median())
        .pivot(on="method", index="condition", values="rec")
        .sort("condition")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Per-Primary L0 recovery, L0.5 and L1 (PF) and L1 (signed) ==")
    pp = rec.filter(
        pl.col("site").is_in(k.SIX)
        & (pl.col("method") == "L0")
        & pl.col("condition").is_in(["pf_fleet_l05_r0", "pf_fleet_l1_r0", "sig_fleet_l1_r0"])
    )
    print(
        pp.group_by("site", "condition")
        .agg(rec=pl.col("rec").median(), inj_mw=pl.col("inj_mw").first())
        .pivot(on="condition", index=["site"], values=["rec", "inj_mw"])
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Fold-loss ratio at L1 (PF over signed), median over the six ==")
    for meth in ("L0", "L1", "FA", "F4"):
        a = (
            six.filter((pl.col("condition") == "pf_fleet_l1_r0") & (pl.col("method") == meth))
            .group_by("site")
            .agg(r=pl.col("rec").median())
        )
        b = (
            six.filter((pl.col("condition") == "sig_fleet_l1_r0") & (pl.col("method") == meth))
            .group_by("site")
            .agg(r=pl.col("rec").median())
        )
        j = a.join(b, on="site", suffix="_sig")
        print(meth, round(float((j["r"] / j["r_sig"]).median()), 2))  # ty: ignore[invalid-argument-type]
    table = real_table(res)
    print("== Capacity of the uninjected PF-arm twin networks (MW, raw) ==")
    print(
        table.with_columns(pl.col(pl.Float64).round(2))
        .pivot(on="method", index="site", values="p50")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print(
        table.filter(pl.col("method").is_in(["L0", "FA"])).with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Curve class, q0, q1, share of cells flagged as crossing (FA), twins ==")
    tw = res.filter(pl.col("condition") == "pf_twin")
    summary = tw.group_by("site", maintain_order=True).agg(
        curve=pl.col("curve_class").first(),
        q0=pl.col("FA_q0").median(),
        q1=pl.col("FA_q1").median(),
        n_cells=pl.col("cell_umax").first().list.len(),
        cross_share=pl.col("FA_cross")
        .map_elements(lambda s: float(np.mean(s.to_list())), return_dtype=pl.Float64)
        .mean(),
        mix=pl.col("FA_mix").first(),
    )
    print(summary.with_columns(pl.col(pl.Float64).round(2)))
    table.write_csv(k.WORK / "analysis" / "capacity_table_2026-06.csv")
    fig_levels(rec)
    fig_placebos(rec)
    fig_real(table)
    fig_cross(res)
    print("figures written to", DOCS)


if __name__ == "__main__":
    main()
