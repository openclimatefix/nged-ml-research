# ruff: noqa: E501
"""Experiment 673: the NGED page on unmetered solar capacity per Primary, with register cross-checks.

Reads the stored probe tables of experiments 649, 662, and 671, the registers (Feed-in Tariff, the
Renewable Energy Planning Database, and NGED's Embedded Capacity Register), and each Primary's
service-area polygon (`area_wkt` in `data/NGED/metadata.parquet`). It fits nothing. It writes the
tables to this experiment's private `work/` folder, prints every number the page quotes, and draws
figure (a). Figure (b) is drawn by `e00673_s_of_u.py`.

Usage: python e00673_deliverable.py
"""

import json
import sys
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
from plotting import ocf_theme as theme
from pyproj import Transformer
from shapely import points, wkt
from shapely.prepared import prep
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies/embedded_battery_forecast"))
from ecr import read_ecr  # ty: ignore[unresolved-import]

NAME: Final = "00673-pv-capacity-for-nged-per-primary-with-register-cross-check"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
DOCS: Final[Path] = Path(__file__).resolve().parents[2] / "docs/ml-experiments" / NAME
E: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUTSIDE: Final[Path] = E / "00045-pvdis-4-weekly-capacity-asymmetric-l1" / "outside"
ECR_PATH: Final[Path] = REPO_DATA_DIR.parent / "literature" / "NGED" / "NGED ECR AUG 2026.xlsx"
REPD_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/solar_bmu_census/inputs/raw/repd_register.json"
)
MW_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_MAIN: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D19", "D21")
REPORT_ONLY: Final[tuple[str, ...]] = ("D8", "D22")
ALL_SITES: Final[tuple[str, ...]] = (
    "D2", "D3", "D4", "D5", "D8", "D9", "D10", "D11", "D12", "D14", "D16", "D19", "D21", "D22",
)  # fmt: skip
SMALL_FIT_KW: Final = 50.0
"""The Embedded Capacity Register starts at 50 kW, so Feed-in Tariff sites up to this size cannot be in it."""
MATCH_M: Final = 1000.0
"""Two register rows within this distance count as one site; a metered farm removes register rows within it."""
SIGNED_RECOVERY: Final = 0.7858
"""Experiment 662's median recovery of the linear probe at 0.5 times in the signed world (no apparent-power conversion).

The five non-folding MVA Primaries divide the raw reading by this value, because experiment 679 finds the conversion loss under power factor 1 with a small reactive floor to be 0.5 points (80% interval -1.0 to 7.7)."""
PV_METERED_IDS: Final[tuple[int, ...]] = (21, 22, 23, 29, 30, 31, 32)
"""Metadata ids of NGED's metered solar generators. They are lettered A to G by id; the page names none."""


def load_polygons() -> dict[str, object]:
    """Return each Primary's service-area polygon, keyed by site label."""
    series = pl.read_parquet(
        E / "00103-differentiable-physics-1-data-look" / "series_table.parquet"
    ).filter(pl.col("site").is_in(ALL_SITES))
    meta = pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet")
    joined = series.join(meta.select("time_series_id", "area_wkt"), on="time_series_id")
    return {r["site"]: wkt.loads(r["area_wkt"]) for r in joined.iter_rows(named=True)}


def metered_pv() -> pl.DataFrame:
    """Return the metered solar generators with their position and a letter, for matching only."""
    meta = pl.read_parquet(REPO_DATA_DIR / "NGED" / "metadata.parquet")
    sel = meta.filter(pl.col("time_series_id").is_in(PV_METERED_IDS)).sort("time_series_id")
    letters = "ABCDEFG"
    return sel.select("latitude", "longitude").with_columns(
        letter=pl.Series([letters[i] for i in range(sel.height)])
    )


def to_lonlat(*, easting: np.ndarray, northing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert British National Grid metres to longitude and latitude."""
    return Transformer.from_crs(27700, 4326, always_xy=True).transform(easting, northing)


def _rescale(column: pl.Expr, *, limit: float) -> pl.Expr:
    """Divide a coordinate by 10 until it is under `limit` metres.

    203 of the Embedded Capacity Register's connected solar rows carry a British National Grid
    coordinate inflated by a power of 10 (for example 4.9e6 for a Lincoln postcode). Without this
    repair those rows fall outside every polygon.
    """
    expr = column
    for _ in range(12):
        expr = pl.when(expr > limit).then(expr / 10.0).otherwise(expr)
    return expr


def read_ecr_solar() -> pl.DataFrame:
    """Return the connected solar rows of the Embedded Capacity Register with a position."""
    ecr = read_ecr(path=ECR_PATH).filter(
        pl.col("energy_source_1") == "Solar", pl.col("connection_status") == "Connected"
    )
    ecr = ecr.with_columns(
        easting=_rescale(pl.col("easting"), limit=700_000.0),
        northing=_rescale(pl.col("northing"), limit=1_300_000.0),
    )
    return ecr.select(
        "easting",
        "northing",
        "primary_substation",
        "connection_voltage_kv",
        "sheet",
        capacity_mw=pl.col("connected_registered_capacity_mw"),
        index=pl.int_range(pl.len()),
    )


def read_repd_solar() -> pl.DataFrame:
    """Return the operational solar rows of the Renewable Energy Planning Database."""
    body = json.loads(REPD_PATH.read_text())["body"]
    repd = pl.read_csv(body.encode(), infer_schema_length=0)
    return (
        repd.filter(
            pl.col("Technology Type") == "Solar Photovoltaics",
            pl.col("Development Status (short)") == "Operational",
        )
        .select(
            easting=pl.col("X-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
            northing=pl.col("Y-coordinate").str.replace_all(",", "").cast(pl.Float64, strict=False),
            capacity_mw=pl.col("Installed Capacity (MWelec)").cast(pl.Float64, strict=False),
        )
        .drop_nulls()
        .with_columns(index=pl.int_range(pl.len()))
    )


def inside(*, frame: pl.DataFrame, polygon: object) -> np.ndarray:
    """Return a boolean mask of the rows whose position lies in the polygon."""
    lon, lat = to_lonlat(easting=frame["easting"].to_numpy(), northing=frame["northing"].to_numpy())
    prepared = prep(polygon)
    return np.array([prepared.contains(p) for p in points(lon, lat)])


def near(*, a: pl.DataFrame, b: pl.DataFrame) -> np.ndarray:
    """Return a mask over `a`'s rows that lie within `MATCH_M` metres of any row of `b` (British National Grid)."""
    if b.height == 0 or a.height == 0:
        return np.zeros(a.height, dtype=bool)
    ax, ay = a["easting"].to_numpy(), a["northing"].to_numpy()
    bx, by = b["easting"].to_numpy(), b["northing"].to_numpy()
    d = np.hypot(ax[:, None] - bx[None, :], ay[:, None] - by[None, :])
    return (d <= MATCH_M).any(axis=1)


def fit_by_primary() -> pl.DataFrame:
    """Return Feed-in Tariff solar capacity per Primary (AC and DC, MW), by size band, by district share."""
    series = pl.read_parquet(
        E / "00103-differentiable-physics-1-data-look" / "series_table.parquet"
    )
    ids = dict(series.select("time_series_id", "site").iter_rows())
    installs = pl.read_parquet(OUTSIDE / "fit_solar_pv_installations.parquet").filter(
        pl.col("technology") == "Photovoltaic"
    )
    shares = pl.read_parquet(OUTSIDE / "fit_district_polygon_shares.parquet")
    bands = {"fit_ac_le50kw": (0, SMALL_FIT_KW), "fit_ac_50kw_1mw": (SMALL_FIT_KW, 1000.0)}
    frames = []
    for name, (lo, hi) in bands.items():
        by_district = (
            installs.filter(
                pl.col("installed_capacity_kw") > lo, pl.col("installed_capacity_kw") <= hi
            )
            .group_by("postcode")
            .agg(
                ac=pl.col("declared_net_capacity_kw").sum(),
                dc=pl.col("installed_capacity_kw").sum(),
                n=pl.len(),
            )
        )
        frames.append(
            shares.join(by_district, left_on="district", right_on="postcode")
            .group_by("time_series_id")
            .agg(
                **{
                    name: (pl.col("share") * pl.col("ac")).sum() / 1000.0,
                    name.replace("_ac_", "_dc_"): (pl.col("share") * pl.col("dc")).sum() / 1000.0,
                }
            )
        )
    out = frames[0].join(frames[1], on="time_series_id", how="full", coalesce=True)
    return (
        out.with_columns(site=pl.col("time_series_id").replace_strict(ids, default=None))
        .drop_nulls("site")
        .filter(pl.col("site").is_in(ALL_SITES))
        .fill_null(0.0)
    )


def register_totals() -> pl.DataFrame:
    """Return the register totals per Primary: each register, double-counting, and the metered removal."""
    polygons = load_polygons()
    ecr, repd, metered = read_ecr_solar(), read_repd_solar(), metered_pv()
    mx, my = Transformer.from_crs(4326, 27700, always_xy=True).transform(
        metered["longitude"].to_numpy(), metered["latitude"].to_numpy()
    )
    met = pl.DataFrame({"easting": mx, "northing": my, "letter": metered["letter"]})
    rows = []
    ecr_xy = ecr.filter(pl.col("easting").is_not_null())
    for site in ALL_SITES:
        poly = polygons[site]
        e = ecr_xy.filter(pl.Series(inside(frame=ecr_xy, polygon=poly)))
        r = repd.filter(pl.Series(inside(frame=repd, polygon=poly)))
        # REPD rows that are the same site as an ECR row anywhere in the polygon.
        r_dup = near(a=r, b=e)
        r = r.with_columns(dup=pl.Series(r_dup))
        # Rows removed because a metered NGED solar generator sits there.
        e = e.with_columns(metered=pl.Series(near(a=e, b=met)))
        r = r.with_columns(metered=pl.Series(near(a=r, b=met)))
        large_voltage = pl.col("connection_voltage_kv").is_in(["33", "66", "132"])
        e_small = e.filter(pl.col("sheet").str.contains("<1MW"), ~pl.col("metered"))
        e_large_low = e.filter(
            ~pl.col("sheet").str.contains("<1MW"), ~large_voltage, ~pl.col("metered")
        )
        e_large_high = e.filter(
            ~pl.col("sheet").str.contains("<1MW"), large_voltage, ~pl.col("metered")
        )
        r_new = r.filter(~pl.col("dup"), ~pl.col("metered"))
        r_new_small = r_new.filter(pl.col("capacity_mw") < 1.0)
        r_new_large = r_new.filter(pl.col("capacity_mw") >= 1.0)
        rows.append(
            {
                "site": site,
                "ecr_rows_in_polygon": e.height,
                "ecr_mw_all": e["capacity_mw"].sum(),
                "ecr_mw_lt1": e.filter(pl.col("sheet").str.contains("<1MW"))["capacity_mw"].sum(),
                "ecr_mw_ge1": e.filter(~pl.col("sheet").str.contains("<1MW"))["capacity_mw"].sum(),
                "ecr_mw_ge1_ge33kv": e.filter(~pl.col("sheet").str.contains("<1MW"), large_voltage)[
                    "capacity_mw"
                ].sum(),
                "ecr_mw_metered_removed": e.filter(pl.col("metered"))["capacity_mw"].sum(),
                "ecr_small_net": e_small["capacity_mw"].sum(),
                "ecr_large_low_net": e_large_low["capacity_mw"].sum(),
                "ecr_large_high_net": e_large_high["capacity_mw"].sum(),
                "repd_rows_in_polygon": r.height,
                "repd_mw_all": r["capacity_mw"].sum(),
                "repd_mw_dup_of_ecr": r.filter(pl.col("dup"))["capacity_mw"].sum(),
                "repd_mw_metered_removed": r.filter(pl.col("metered"), ~pl.col("dup"))[
                    "capacity_mw"
                ].sum(),
                "repd_new_lt1_net": r_new_small["capacity_mw"].sum(),
                "repd_new_ge1_net": r_new_large["capacity_mw"].sum(),
                "metered_letters_inside": "".join(
                    met.filter(pl.Series(inside(frame=met, polygon=poly)))["letter"].to_list()
                ),
            }
        )
    return pl.DataFrame(rows)


def probe_table() -> pl.DataFrame:
    """Return the probe interval per Primary: stored calibrated values and the group each belongs to."""
    c649 = pl.read_parquet(
        E / "00649-pv-capacity-table-season-year-from-stored-probes/work/capacity_table.parquet"
    )
    c649 = c649.filter(pl.col("probe_year") == 2025, pl.col("cutoff") == "2026-06")
    mw = c649.select(
        "site",
        probe="cal_p50",
        p10="cal_p10",
        p90="cal_p90",
        raw_p10="raw_p10",
        raw_p50="raw_p50",
        raw_p90="raw_p90",
    ).with_columns(group=pl.lit("MW Primary"), method=pl.lit("neural probe (experiment 649)"))
    t662 = pl.read_csv(
        E
        / "00662-pv-probe-through-the-mva-fold-matched-level-redesign/work/analysis/capacity_table.csv"
    ).filter(pl.col("cutoff") == "2026-06")
    lin = t662.filter(pl.col("method") == "L0")
    fold = t662.filter(pl.col("method") == "FA")
    mva_rows = []
    for site in ("D2", "D3", "D4", "D19", "D21", "D8", "D22", "D12"):
        group = (
            "MVA main"
            if site in MVA_MAIN
            else ("D12 fold only" if site == "D12" else "report only")
        )
        src = fold if site == "D12" else lin
        r = src.filter(pl.col("site") == site).row(0, named=True)
        f = fold.filter(pl.col("site") == site).row(0, named=True)
        rescale = r["factor"] / SIGNED_RECOVERY if site in MVA_MAIN else 1.0
        mva_rows.append(
            {
                "site": site,
                "probe": r["cal_p50"] * rescale,
                "p10": r["cal_p10"] * rescale,
                "p90": r["cal_p90"] * rescale,
                "pf_probe": r["cal_p50"],
                "pf_p10": r["cal_p10"],
                "pf_p90": r["cal_p90"],
                "raw_p10": r["p10"],
                "raw_p50": r["p50"],
                "raw_p90": r["p90"],
                "group": group,
                "method": "fold model (experiment 662)"
                if site == "D12"
                else "linear probe (experiment 662)",
                "fold_upper_p10": f["cal_p10"],
                "fold_upper_p50": f["cal_p50"],
                "fold_upper_p90": f["cal_p90"],
            }
        )
    return pl.concat([mw, pl.DataFrame(mva_rows)], how="diagonal")


def reactive_arms() -> pl.DataFrame:
    """Return experiment 662's median recovery at the six MW Primaries per reactive arm, probe, and level."""
    r = pl.read_csv(
        E
        / "00662-pv-probe-through-the-mva-fold-matched-level-redesign/work/analysis/recoveries_2026-06.csv"
    )
    return (
        r.filter(
            pl.col("kind") == "fleet",
            pl.col("rot") == 0,
            pl.col("site").is_in(MW_SITES),
            pl.col("method").is_in(["L0", "FA"]),
        )
        .group_by("arm", "level", "method", "site")
        .agg(rec=pl.col("rec").median())
        .group_by("arm", "level", "method")
        .agg(median=pl.col("rec").median(), lo=pl.col("rec").min(), hi=pl.col("rec").max())
        .sort("method", "level", "arm")
    )


def flag(*, p10: float, p90: float, floor: float, group: str) -> str:
    """Return the plain flag fixed in the plan."""
    if group == "report only":
        return "not comparable"
    if p10 >= floor:
        return "probe above floor"
    if p90 < floor:
        return "below floor"
    return "overlaps"


def main() -> None:
    """Write the tables and figure (a), and print every number the page quotes."""
    WORK.mkdir(parents=True, exist_ok=True)
    probe = probe_table()
    totals = register_totals()
    fit = fit_by_primary()
    table = (
        probe.join(totals, on="site", how="left")
        .join(fit.drop("time_series_id"), on="site", how="left")
        .with_columns(
            floor_main=pl.col("fit_ac_le50kw")
            + pl.col("ecr_small_net")
            + pl.col("ecr_large_low_net")
            + pl.col("repd_new_lt1_net"),
            floor_all=pl.col("fit_ac_le50kw")
            + pl.col("ecr_small_net")
            + pl.col("ecr_large_low_net")
            + pl.col("ecr_large_high_net")
            + pl.col("repd_new_lt1_net")
            + pl.col("repd_new_ge1_net"),
        )
        .with_columns(ratio=pl.col("probe") / pl.col("floor_main"))
    )
    table = table.with_columns(
        flag=pl.struct("p10", "p90", "floor_main", "group").map_elements(
            lambda s: flag(p10=s["p10"], p90=s["p90"], floor=s["floor_main"], group=s["group"]),
            return_dtype=pl.String,
        )
    ).sort(pl.col("site").str.slice(1).cast(pl.Int32))
    table.write_parquet(WORK / "deliverable_table.parquet")
    arms = reactive_arms()
    arms.write_csv(WORK / "reactive_arms.csv")
    with pl.Config(tbl_rows=60, tbl_cols=40, tbl_width_chars=300, float_precision=2):
        print(
            table.select(
                "site", "group", "p10", "probe", "p90", "floor_main", "floor_all", "ratio", "flag"
            )
        )
        print(
            table.select(
                "site",
                "fit_ac_le50kw",
                "fit_ac_50kw_1mw",
                "fit_dc_le50kw",
                "ecr_rows_in_polygon",
                "ecr_mw_lt1",
                "ecr_mw_ge1",
                "ecr_mw_ge1_ge33kv",
                "ecr_mw_metered_removed",
                "repd_rows_in_polygon",
                "repd_mw_all",
                "repd_mw_dup_of_ecr",
                "repd_mw_metered_removed",
                "metered_letters_inside",
            )
        )
        print(arms)
    draw_figure_a(table=table)


def draw_figure_a(*, table: pl.DataFrame) -> None:
    """Draw figure (a): probe interval against the register floor for all 14 Primaries."""
    df = table.select(
        "site", "group", "p10", "probe", "p90", "floor_main", "floor_all", "flag"
    ).with_columns(
        kind=pl.when(pl.col("group") == "MW Primary")
        .then(pl.lit("MW Primary"))
        .when(pl.col("group") == "report only")
        .then(pl.lit("Report only (MVA)"))
        .otherwise(pl.lit("MVA Primary")),
    )
    order = list(ALL_SITES)
    kinds = ["MW Primary", "MVA Primary", "Report only (MVA)", "Register floor"]
    colours = [theme.BRAND_ORANGE, theme.DATA_BLUE, "#6b6b6b", "#000000"]
    scale = alt.Scale(domain=kinds, range=colours)
    legend = alt.Legend(orient="top", title=None, symbolType="square", symbolSize=220)
    y = alt.Y("site:N", sort=order, title="Primary substation")
    base = alt.Chart(df).encode(y=y)
    bar = base.mark_bar(height=12).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("p10:Q", title="AC capacity of unmetered solar (MW)", scale=alt.Scale(zero=True)),
        x2="p90:Q",
        color=alt.Color("kind:N", scale=scale, legend=legend),
    )
    median = base.mark_tick(color="white", thickness=2, size=12).encode(x="probe:Q")  # ty: ignore[unresolved-attribute]
    floor_df = df.with_columns(kind=pl.lit("Register floor"))
    floor = (
        alt.Chart(floor_df)  # ty: ignore[unresolved-attribute]
        .mark_tick(thickness=3, size=22)
        .encode(y=y, x="floor_main:Q", color=alt.Color("kind:N", scale=scale, legend=legend))
    )
    chart = (bar + median + floor).properties(
        width=420,
        height=320,
        title=alt.TitleParams(
            "Probe interval (80%) and register floor per Primary",
            subtitle=[
                "Bar: probe 10th to 90th percentile; white tick: probe median.",
                "Report-only Primaries (D8, D22) are grey. MVA Primaries are read as AC MW.",
            ],
            anchor="start",
            fontSize=13,
            subtitleFontSize=11,
        ),
    )
    chart.save(str(DOCS / "probe_against_register_floor.svg"))


if __name__ == "__main__":
    main()


def markdown_tables() -> str:
    """Return the page's tables as markdown, for pasting into `result.md`."""
    t = pl.read_parquet(WORK / "deliverable_table.parquet")
    d = t.to_dicts()
    by = {r["site"]: r for r in d}
    out = []
    f = lambda x: "none" if x is None else f"{x:.2f}"  # noqa: E731
    out.append(
        "MW|"
        + "\n".join(
            f"| {s} | {f(by[s]['raw_p10'])} / {f(by[s]['raw_p50'])} / {f(by[s]['raw_p90'])} | {f(by[s]['p10'])} / {f(by[s]['probe'])} / {f(by[s]['p90'])} |"
            for s in MW_SITES
        )
    )
    out.append(
        "MVA|"
        + "\n".join(
            f"| {s} | {f(by[s]['raw_p50'])} | {f(by[s]['p10'])} / {f(by[s]['probe'])} / {f(by[s]['p90'])} | {f(by[s]['pf_p10'])} / {f(by[s]['pf_probe'])} / {f(by[s]['pf_p90'])} | {f(by[s]['fold_upper_p10'])} / {f(by[s]['fold_upper_p50'])} / {f(by[s]['fold_upper_p90'])} |"
            for s in (*MVA_MAIN, "D12", *REPORT_ONLY)
        )
    )
    out.append(
        "REG|"
        + "\n".join(
            f"| {s} | {f(by[s]['fit_ac_le50kw'])} ({f(by[s]['fit_dc_le50kw'])}) | {f(by[s]['fit_ac_50kw_1mw'])} | {f(by[s]['ecr_mw_lt1'])} | {f(by[s]['ecr_mw_ge1'])} ({f(by[s]['ecr_mw_ge1_ge33kv'])}) | {f(by[s]['repd_mw_all'])} ({f(by[s]['repd_mw_dup_of_ecr'])}) |"
            for s in ALL_SITES
        )
    )
    out.append(
        "CHK|"
        + "\n".join(
            f"| {s} | {f(by[s]['p10'])} to {f(by[s]['p90'])} | {f(by[s]['floor_main'])} | {f(by[s]['ecr_large_low_net'] + by[s]['repd_new_ge1_net'] * 0)} | {f(by[s]['floor_all'])} | {f(by[s]['ratio'])} | {by[s]['flag']} |"
            for s in ALL_SITES
        )
    )
    return "\n\n".join(out)


if __name__ == "__main__":
    (WORK / "tables.md").write_text(markdown_tables())
    print(markdown_tables())
