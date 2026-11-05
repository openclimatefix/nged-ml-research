# ruff: noqa: E501, PT018, PERF401
"""Experiment 576: where the lag gap sits, by half-hour type, Primary, hour, season, and lead band.

Re-scores experiment 572's stored forecasts (E2, v3, v3+D5; two seeds, five folds) row by row. The
gap is `E2 minus lag arm`, so a positive gap means the lag arm is better. Each half-hour is labelled
sunny or cloudy from the ensemble-mean shortwave radiation of the weather run issued at the origin
minus 9 h (the run behind the forecast, never the target's observed weather); the label is a slicing
variable, not a model input. Solar elevation is computed for 53.0 N, 2.0 W for every Primary.

Bonferroni count 12: 2 contrasts (E2 minus v3, E2 minus v3+D5) x 2 groups x 3 lead bands on all
rows. Every other interval is uncorrected and descriptive.

Outputs under `data/ml-experiments/00576-*/work/`: `gap_tables.csv` (long format), `gap_rows.parquet`.
"""

import math
from pathlib import Path
from typing import Final

import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00572_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00576-lag-gap-by-half-hour-type-eight-mva-and-six-mw-primaries/work"
)
SUMMARY: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00397-plain-and-attention-mixture-density-network-for-primary-demand/work/ens_summary"
)
ARMS: Final = {k: sc.ARM_DIRS[k] for k in ("E2", "V3", "V3D5")}
CONTRASTS: Final = (("E2", "V3"), ("E2", "V3D5"))
LAT: Final[float] = 53.0
LON: Final[float] = -2.0
SUNNY_INDEX: Final[float] = 0.6
BANDS: Final = tuple(sc.BANDS)
SEASONS: Final = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def sun(*, valid_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return solar elevation (degrees) and Haurwitz clear-sky irradiance (W/m2) at UTC microseconds."""
    days = valid_us / 86_400e6
    doy = (days % 365.2425) + 1.0  # day of year is only needed to about a day
    hour = (days % 1.0) * 24.0
    g = 2.0 * math.pi / 365.0 * (doy - 1.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
    )
    eqt = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    solar_hour = hour + (eqt + 4.0 * LON) / 60.0
    ha = np.radians(15.0 * (solar_hour - 12.0))
    lat = math.radians(LAT)
    sin_e = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    elev = np.degrees(np.arcsin(np.clip(sin_e, -1, 1)))
    cs = np.where(sin_e > 0.02, 1098.0 * sin_e * np.exp(-0.057 / np.maximum(sin_e, 0.02)), 0.0)
    return elev, cs


def build() -> pl.DataFrame:
    """Return the scored rows with the two gap columns, weather labels, and calendar columns."""
    t = sc.build_table(arms=ARMS)
    t = t.filter(pl.col("main16") & pl.col("group").is_in(["mva_eight", "signed_six"]))
    t = t.filter(~((pl.col("group") == "signed_six") & (pl.col("cls") != "clean")))
    sites = refs448.sites_table().select(
        pl.col("primary").cast(pl.Int16), "label", "time_series_id"
    )
    us = lambda c: pl.from_epoch(
        pl.col(c).cast(pl.Int64) * heads470.STEP_US + heads470.GRID_START_US, time_unit="us"
    ).dt.replace_time_zone("UTC")  # noqa: E731
    t = t.join(sites, on="primary", how="left", validate="m:1").with_columns(
        power_fcst_init_time=us("origin"), valid_time=us("target")
    )
    lo, hi = t["power_fcst_init_time"].min(), t["power_fcst_init_time"].max()
    w = (
        pl.scan_parquet(str(SUMMARY / "init=*.parquet"))
        .filter(pl.col("power_fcst_init_time").is_between(lo, hi))
        .select("time_series_id", "power_fcst_init_time", "valid_time", "mean_shortwave")
        .collect()
    )
    t = t.join(
        w, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="left", validate="m:1"
    )
    # Standing rule: the weather run is the one issued at or before the origin (init - 9 h), never the target's.
    miss = t["mean_shortwave"].is_null().mean()
    print("share of rows with no weather-run row:", round(float(miss), 5))  # ty: ignore[invalid-argument-type]
    init_us = t["power_fcst_init_time"].dt.epoch("us").to_numpy()
    run_us = init_us - 9 * 3_600_000_000
    assert (run_us <= init_us).all() and (t["valid_time"] > t["power_fcst_init_time"]).all()
    ex = t.row(0, named=True)
    print(
        "example row:",
        {
            k: ex[k]
            for k in ("label", "power_fcst_init_time", "valid_time", "lead_hours", "mean_shortwave")
        },
        "weather run issued",
        ex["power_fcst_init_time"] - __import__("datetime").timedelta(hours=9),
    )
    vus = t["valid_time"].dt.epoch("us").to_numpy()
    elev, cs = sun(valid_us=vus)
    sw = t["mean_shortwave"].to_numpy()
    kt = np.where(cs > 50.0, sw / np.maximum(cs, 1.0), np.nan)
    sky = np.where(
        elev <= 5.0,
        "night or low sun",
        np.where(np.isnan(kt), "unknown", np.where(kt >= SUNNY_INDEX, "sunny", "cloudy")),
    )
    ebin = np.select(
        [elev <= 0, elev <= 10, elev <= 25],
        ["below horizon", "0-10 deg", "10-25 deg"],
        "above 25 deg",
    )
    t = t.with_columns(
        sky=pl.Series(sky),
        elev_band=pl.Series(ebin),
        clear_sky_index=pl.Series(kt, dtype=pl.Float64),
        hour=pl.col("valid_time").dt.hour().cast(pl.Int32),
        season=pl.col("valid_time").dt.month().replace_strict(SEASONS, return_dtype=pl.String),
        gap_V3=pl.mean_horizontal([pl.col(f"E2_s{x}") - pl.col(f"V3_s{x}") for x in sc.SEEDS]),
        gap_V3D5=pl.mean_horizontal([pl.col(f"E2_s{x}") - pl.col(f"V3D5_s{x}") for x in sc.SEEDS]),
        **{
            f"d_E2_{b}_s{x}": pl.col(f"E2_s{x}") - pl.col(f"{b}_s{x}")
            for b in ("V3", "V3D5")
            for x in sc.SEEDS
        },
    )
    return t.with_columns(
        set=pl.when(pl.col("group") == "mva_eight").then(pl.lit("eight")).otherwise(pl.lit("six"))
    )


def stat(*, sub: pl.DataFrame, b: str, tests: int, rng: np.random.Generator) -> dict:
    """Return the gap `E2 minus b` with its month-block interval, MDE, and per-fold sign count."""
    d = sub.select("month", "day", d=pl.col(f"gap_{b}"))
    st = s.block_stats(diff=d, rng=rng, tests=tests)
    st["mde"] = sc.mde(sub=sub, a="E2", b=b)
    return st


def tables(t: pl.DataFrame) -> pl.DataFrame:
    """Return every descriptive and counted table in long format."""
    rng = np.random.default_rng(0)
    out = []
    groups = {
        "eight": t.filter(pl.col("set") == "eight"),
        "eight without D8": t.filter((pl.col("set") == "eight") & (pl.col("label") != "D8")),
        "six": t.filter(pl.col("set") == "six"),
    }
    dims = {
        "all": None,
        "sky": "sky",
        "elev_band": "elev_band",
        "hour": "hour",
        "season": "season",
        "primary": "label",
    }
    for gname, g in groups.items():
        for band in (*BANDS, "all"):
            gb = g if band == "all" else g.filter(pl.col("band") == band)
            for dim, col in dims.items():
                if dim == "primary" and gname == "eight without D8":
                    continue
                parts = (
                    [("all rows", gb)]
                    if col is None
                    else [(str(k[0]), v) for k, v in gb.partition_by(col, as_dict=True).items()]
                )
                for val, sub in parts:
                    if sub.height < 200:
                        continue
                    for _, b in CONTRASTS:
                        counted = dim == "all" and gname in ("eight", "six") and band != "all"
                        st = stat(sub=sub, b=b, tests=12 if counted else 1, rng=rng)
                        out.append({"group": gname, "band": band, "dim": dim, "value": val, "arm": b, "counted": counted, "tests": 12 if counted else 1,
                                    "mean_crps_E2": float(np.mean([sub[f"E2_s{x}"].mean() for x in sc.SEEDS])), "sum_gap": float(sub[f"gap_{b}"].sum()), **st})  # fmt: skip
    return pl.DataFrame(out)


def shares(t: pl.DataFrame) -> pl.DataFrame:
    """Return the share of the 24-120 h total gap held by each (sky, season) half-hour type, and by D8."""
    rows = []
    for gname in ("eight", "six"):
        g = t.filter((pl.col("set") == gname) & (pl.col("band") == "24-120h"))
        for b in ("V3", "V3D5"):
            tot = float(g[f"gap_{b}"].sum())
            by = (
                g.group_by("sky", "season")
                .agg(n=pl.len(), s=pl.col(f"gap_{b}").sum())
                .with_columns(share=pl.col("s") / tot, share_rows=pl.col("n") / g.height)
                .sort("s", descending=True)
            )
            for rank, r in enumerate(by.iter_rows(named=True), start=1):
                rows.append(
                    {
                        "group": gname,
                        "arm": b,
                        "type": f"{r['sky']}, {r['season']}",
                        "rank": rank,
                        "share_of_total_gap": r["share"],
                        "share_of_rows": r["share_rows"],
                    }
                )
            lab = (
                g.group_by("label")
                .agg(n=pl.len(), s=pl.col(f"gap_{b}").sum())
                .with_columns(share=pl.col("s") / tot, share_rows=pl.col("n") / g.height)
            )
            for r in lab.iter_rows(named=True):
                rows.append(
                    {
                        "group": gname,
                        "arm": b,
                        "type": f"Primary {r['label']}",
                        "rank": 0,
                        "share_of_total_gap": r["share"],
                        "share_of_rows": r["share_rows"],
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    """Build the rows, write the tables, and print the headline numbers."""
    OUT.mkdir(parents=True, exist_ok=True)
    t = build()
    t.drop(
        [c for c in t.columns if c.endswith(("_s0", "_s1")) and not c.startswith("d_")]
    ).write_parquet(OUT / "gap_rows.parquet")
    tab = tables(t)
    tab.write_csv(OUT / "gap_tables.csv")
    shares(t).write_csv(OUT / "gap_shares.csv")
    with pl.Config(tbl_rows=80, tbl_cols=14, tbl_width_chars=220):
        print(
            tab.filter((pl.col("dim") == "all") & pl.col("counted"))
            .select(
                "group",
                "band",
                "arm",
                "n",
                "difference",
                "low_bonferroni_month",
                "high_bonferroni_month",
                "mde",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(
            tab.filter((pl.col("dim") == "sky") & (pl.col("band") == "24-120h"))
            .select(
                "group",
                "value",
                "arm",
                "n",
                "difference",
                "low_month",
                "high_month",
                "mde",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(
            tab.filter((pl.col("dim") == "primary") & (pl.col("band") == "24-120h"))
            .select(
                "group",
                "value",
                "arm",
                "n",
                "difference",
                "low_month",
                "high_month",
                "folds_positive",
            )
            .with_columns(pl.col(pl.Float64).round(3))
        )
        print(shares(t).filter(pl.col("rank") <= 3).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
