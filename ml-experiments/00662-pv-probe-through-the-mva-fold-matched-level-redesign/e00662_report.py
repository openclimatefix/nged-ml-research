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


def level_summary(rec: pl.DataFrame, *, arm: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return per-Primary recoveries and the pooled spread over seeds and Primaries.

    Recoveries are medians over seeds and rotations. The spread is the 10th to 90th percentile.
    """
    sel = rec.filter(
        pl.col("condition").str.starts_with(f"{arm}_fleet_")
        & pl.col("site").is_in(k.SIX)
        & pl.col("method").is_in(list(METHODS))
    )
    dots = sel.group_by("site", "level", "method").agg(rec=pl.col("rec").median())
    per_seed = sel.group_by("site", "level", "method", "seed").agg(rec=pl.col("rec").median())
    spread = per_seed.group_by("level", "method").agg(
        lo=pl.col("rec").quantile(0.1),
        hi=pl.col("rec").quantile(0.9),
        med=pl.col("rec").median(),
    )
    return dots, spread


def fig_levels(rec: pl.DataFrame) -> None:
    """Draw recovery by injection level and method, for the apparent-power (PF) world and the signed world.

    One dot per Primary, a diamond at the median of the six, and a bar from the 10th to the 90th
    percentile over seeds and Primaries.
    """
    scale = alt.Scale(
        domain=[v[0] for v in METHODS.values()], range=[v[1] for v in METHODS.values()]
    )
    panels = []
    for arm, title in (
        ("pf", "Apparent-power (PF) world: the six MW Primaries converted to MVA"),
        ("sig", "Signed world: the same farms and levels, no fold"),
    ):
        dots, spread = level_summary(rec, arm=arm)
        lab = {m: v[0] for m, v in METHODS.items()}
        dots = dots.with_columns(
            method_label=pl.col("method").replace(lab), lv=pl.col("level").cast(pl.String)
        )
        spread = spread.with_columns(
            method_label=pl.col("method").replace(lab), lv=pl.col("level").cast(pl.String)
        )
        order = ["0.5", "1.0", "2.0", "6.0"]
        x = alt.X(
            "lv:O",
            title="Injected capacity, as a multiple of the midday reading (2.0 MW floor)",
            sort=order,
            axis=alt.Axis(labelAngle=0),
        )
        y = alt.Y(
            "rec:Q", title="Share of injected MW recovered", scale=alt.Scale(domain=[-1.1, 1.3])
        )
        colour = alt.Color(
            "method_label:N",
            scale=scale,
            title=None,
            legend=alt.Legend(orient="top") if arm == "pf" else None,
        )
        off = alt.XOffset("method_label:N")
        pts = (
            alt.Chart(dots)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=30, opacity=0.35, aria=False)
            .encode(x=x, y=y, color=colour, xOffset=off)
        )
        bars = (
            alt.Chart(spread)  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeWidth=2.5, aria=False)
            .encode(x=x, y="lo:Q", y2="hi:Q", color=colour, xOffset=off)
        )
        meds = (
            alt.Chart(spread)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=90, shape="diamond", aria=False)
            .encode(x=x, y=alt.Y("med:Q"), color=colour, xOffset=off)
        )
        ref = (
            alt.Chart(pl.DataFrame({"y": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[5, 3], strokeWidth=1.2, color=theme.BLACK_2)
            .encode(y="y:Q")
        )
        panels.append(
            (ref + pts + bars + meds).properties(
                width=W, height=260, title=alt.TitleParams(title, anchor="start", fontSize=12)
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Recovery of injected solar capacity by injection level",
            subtitle=[
                "2026-06 networks, 5 seeds. Faint dot: one MW Primary. Diamond: median of the six. Bar: 10th to 90th percentile over seeds and Primaries.",
                "Dashed lines: 0 (nothing found) and 1 (all found).",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
    )
    chart.save(DOCS / "recovery_by_level.svg")


def fig_bound(res: pl.DataFrame) -> None:
    """Share of fold fits that end on a bound, by condition level and fold-model arm."""
    sel = res.filter(
        pl.col("condition").str.starts_with("pf_fleet_") | (pl.col("condition") == "pf_twin")
    ).filter(pl.col("site").is_in([*k.SIX, *k.NATIVE]))
    rows = []
    for arm, name in (
        ("FA", "FA (night prior)"),
        ("F4", "F4 (four curves)"),
        ("F1", "F1 (one curve)"),
    ):
        g = (
            sel.with_columns(
                lv=pl.col("condition")
                .str.extract(r"_l(05|1|2|6)_", 1)
                .replace({"05": "0.5", "1": "1", "2": "2", "6": "6"})
                .fill_null("0 (twin)"),
            )
            .group_by("lv")
            .agg(share=pl.col(f"{arm}_bound").cast(pl.Float64).mean())
        )
        rows.append(g.with_columns(arm=pl.lit(name)))
    d = pl.concat(rows)
    chart = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            x=alt.X(
                "lv:O",
                sort=["0 (twin)", "0.5", "1", "2", "6"],
                title="Injected capacity, as a multiple of the midday reading",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of fits on a bound", scale=alt.Scale(domain=[0, 0.2])),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(d["arm"].unique(maintain_order=True)),
                    range=[theme.DATA_BLUE, theme.DATA_SKY, theme.DATA_PURPLE],
                ),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
            xOffset="arm:N",
        )
        .properties(
            width=W,
            height=200,
            title=alt.TitleParams(
                "How often the fold fit ends on a bound",
                subtitle=[
                    "2026-06 networks, PF-arm real-farm fleets and twins, 11 Primaries, 5 seeds. A bound means every fold capacity is zero,",
                    "or most cell levels sit at their lower limit. Dashed line: the 10% target. FA and F4 bars have zero height.",
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            ),
        )
    )
    target = (
        alt.Chart(pl.DataFrame({"y": [0.10]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.2, color=theme.BLACK_2)
        .encode(y="y:Q")
    )
    (chart + target).save(DOCS / "fits_on_a_bound.svg")


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
                title="Capacity read, as a share of the added size (2 times, 6 times for the sun-rising load, 0.5 times for the reference)",
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
                    "2026-06 networks, one dot per Primary (median of 5 seeds). A placebo should read near the dashed zero line; the reference row is a real fleet, not a placebo."
                ],
                anchor="start",
                fontSize=13,
                subtitleFontSize=11,
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[5, 3], strokeWidth=1.2, color=theme.BLACK_2)
        .encode(x="x:Q")
    )
    (chart + zero).save(DOCS / "placebos.svg")


def real_table(res: pl.DataFrame) -> pl.DataFrame:
    """Return the capacity of each Primary's uninjected PF-arm twin network, per cutoff and method.

    The interval is the day-block bootstrap half-width and 1.2816 times the between-seed standard
    deviation, summed in quadrature, about the median over seeds.
    """
    rows = []
    for cutoff in sorted(res["cutoff"].unique(), reverse=True):
        tw = res.filter((pl.col("condition") == "pf_twin") & (pl.col("cutoff") == cutoff))
        for site in (*k.SIX, *k.NATIVE, *k.FOLDERS):
            t = tw.filter(pl.col("site") == site)
            for meth in ("L0", "L1", "F4", "F1", "FA"):
                draws = np.concatenate(
                    [np.array(r[f"{meth}_boot"]) for r in t.iter_rows(named=True)]
                )
                draws = draws[np.isfinite(draws)]
                half_boot = float(np.subtract(*np.percentile(draws, [90, 10])) / 2)
                sd_seed = float(t[meth].std())  # ty: ignore[invalid-argument-type]
                half = float(np.hypot(half_boot, 1.2816 * sd_seed))
                centre = float(t[meth].median())  # ty: ignore[invalid-argument-type]
                rows.append(
                    {
                        "cutoff": cutoff,
                        "site": site,
                        "method": meth,
                        "p50": centre,
                        "p10": centre - half,
                        "p90": centre + half,
                        "seed_sd": sd_seed,
                    }
                )
    return pl.DataFrame(rows)


def calibrate(table: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """Divide L0 by its leave-one-farm-out factor at 0.5 times and FA by its factor at 1 times.

    Both factors come from the six MW Primaries in the PF arm.
    """
    out = []
    for meth, conds in (("L0", ["pf_fleet_l05_r0"]), ("FA", ["pf_fleet_l1_r0", "pf_fleet_l1_r1"])):
        per = (
            rec.filter(
                pl.col("condition").is_in(conds)
                & (pl.col("method") == meth)
                & pl.col("site").is_in(k.SIX)
            )
            .group_by("site")
            .agg(r=pl.col("rec").median())
        )
        vals = dict(zip(per["site"].to_list(), per["r"].to_list(), strict=True))
        loo = [float(np.median([v for s2, v in vals.items() if s2 != s])) for s in vals]
        f_mid, f_lo, f_hi = float(np.median(list(vals.values()))), min(loo), max(loo)
        sub = table.filter(pl.col("method") == meth)
        out.append(
            sub.with_columns(
                factor=pl.lit(f_mid),
                factor_lo=pl.lit(f_lo),
                factor_hi=pl.lit(f_hi),
                cal_p50=pl.col("p50") / f_mid,
                cal_p10=pl.when(pl.col("p10") > 0)
                .then(pl.col("p10") / f_hi)
                .otherwise(pl.col("p10") / f_lo),
                cal_p90=pl.when(pl.col("p90") > 0)
                .then(pl.col("p90") / f_lo)
                .otherwise(pl.col("p90") / f_hi),
            )
        )
    return pl.concat(out)


def fig_real(table: pl.DataFrame) -> None:
    """Capacity at all 14 Primaries from the linear probe and the fold model, at both cutoffs."""
    label = {
        site: (f"{site} (report only)" if site in k.FOLDERS else site)
        for site in (*k.SIX, *k.NATIVE, *k.FOLDERS)
    }
    order = [label[site] for site in (*k.SIX, *k.NATIVE, *k.FOLDERS)]
    scale = alt.Scale(
        domain=[METHODS[m][0] for m in ("L0", "FA")], range=[METHODS[m][1] for m in ("L0", "FA")]
    )
    panels = []
    for cutoff in sorted(table["cutoff"].unique(), reverse=True):
        d = (
            table.filter((pl.col("cutoff") == cutoff) & pl.col("method").is_in(["L0", "FA"]))
            .with_columns(site=pl.col("site").replace(label))
            .with_columns(
                method_label=pl.col("method").replace({m: METHODS[m][0] for m in ("L0", "FA")})
            )
        )
        base = alt.Chart(d).encode(
            x=alt.X(
                "site:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelAlign="right")
            ),
            color=alt.Color(
                "method_label:N", scale=scale, title=None, legend=alt.Legend(orient="top")
            ),
            xOffset="method_label:N",
        )
        panels.append(
            (
                base.mark_rule(aria=False).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y(
                        "p10:Q", title="Capacity (MW of AC, raw)", scale=alt.Scale(domain=[-6, 12])
                    ),
                    y2="p90:Q",
                )
                + base.mark_point(filled=True, size=50, aria=False).encode(y="p50:Q")  # ty: ignore[unresolved-attribute]
            ).properties(
                width=W,
                height=230,
                title=alt.TitleParams(f"{cutoff} networks", anchor="start", fontSize=12),
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Capacity read from the uninjected networks, before calibration",
            subtitle=[
                "Dot: median over seeds. Line: bootstrap and between-seed spread, summed in quadrature (80%).",
                "MW Primaries D5 to D16; MVA Primaries D2 to D22. D8, D12, and D22 fold (report only).",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        )
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
        .properties(width=W, height=250)
        .facet(row=alt.Row("site:N", title=None))
        .properties(
            title=alt.TitleParams(
                "Where the fold model says the reading crosses zero",
                subtitle=[
                    "Uninjected 2026-06 networks, 5 seeds. Orange: fold model (FA) finds demand below solar output in the probe cell."
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
    res_all = pl.concat(
        [
            pl.read_parquet(p)
            for c in k.CUTOFFS
            for p in sorted((k.WORK / "probes" / c).glob("*.parquet"))
        ],
        how="diagonal_relaxed",
    )
    res = res_all.filter(pl.col("cutoff") == CUTOFF)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    six = rec.filter(pl.col("site").is_in(k.SIX))
    print("== Median recovery over the six, by condition and method (2026-06) ==")
    print(
        six.group_by("condition", "method")
        .agg(rec=pl.col("rec").median())
        .pivot(on="method", index="condition", values="rec")
        .sort("condition")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Injected MW per Primary at each level ==")
    print(
        rec.filter(
            pl.col("condition").str.contains("^pf_fleet_l\\d+_r0$") & (pl.col("method") == "L0")
        )
        .group_by("site", "level")
        .agg(inj_mw=pl.col("inj_mw").first())
        .pivot(on="level", index="site", values="inj_mw")
        .sort("site")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Per-Primary recovery (PF and signed, matched level; median over seeds) ==")
    for meth in ("L0", "L1", "FA"):
        pp = rec.filter(
            pl.col("site").is_in(k.SIX)
            & (pl.col("method") == meth)
            & pl.col("condition").is_in(
                [f"{a}_fleet_{lv}_r0" for a in ("pf", "sig") for lv in ("l05", "l1", "l2")]
            )
        )
        print(meth)
        print(
            pp.group_by("site", "condition")
            .agg(rec=pl.col("rec").median())
            .pivot(on="condition", index="site", values="rec")
            .sort("site")
            .with_columns(pl.col(pl.Float64).round(2))
        )
    print("== Native (MVA, does not fold) Primaries, PF arm ==")
    nat = rec.filter(
        pl.col("site").is_in(k.NATIVE)
        & pl.col("condition").is_in(
            ["pf_fleet_l05_r0", "pf_fleet_l1_r0", "pf_fleet_l2_r0", "pf_fleet_l6_r0"]
        )
    )
    print(
        nat.group_by("condition", "method")
        .agg(rec=pl.col("rec").median())
        .pivot(on="method", index="condition", values="rec")
        .sort("condition")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print("== Share of fits on a bound, by arm and condition kind ==")
    for arm in ("FA", "F4", "F1"):
        print(
            arm,
            round(
                float(
                    res.filter(pl.col("condition").str.starts_with("pf_"))[f"{arm}_bound"]
                    .cast(pl.Float64)
                    .mean()  # ty: ignore[invalid-argument-type]
                ),
                3,
            ),
        )
    table = real_table(res_all)
    cal = calibrate(table, rec)
    print(
        "== Capacity of the uninjected PF-arm twin networks (MW, raw p50 with interval, and calibrated) =="
    )
    print(
        cal.filter(pl.col("method").is_in(["L0", "FA"]))
        .select(
            "cutoff",
            "site",
            "method",
            "p50",
            "p10",
            "p90",
            "seed_sd",
            "cal_p50",
            "cal_p10",
            "cal_p90",
        )
        .sort("method", "site", "cutoff")
        .with_columns(pl.col(pl.Float64).round(2))
    )
    print(
        "calibration factors:",
        cal.select("method", "factor", "factor_lo", "factor_hi").unique().to_dicts(),
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
    )
    print(summary.with_columns(pl.col(pl.Float64).round(2)))
    cal.write_csv(k.WORK / "analysis" / "capacity_table.csv")
    fig_levels(rec)
    fig_placebos(rec)
    fig_real(table)
    fig_bound(res)
    fig_cross(res)
    print("figures written to", DOCS)


if __name__ == "__main__":
    main()
