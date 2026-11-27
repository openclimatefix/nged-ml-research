# ruff: noqa: E501, PLR0915
"""Experiment 735: distance of each Primary from the coast, and holiday-day errors of the stored step 5.

Run with the folders of experiments 667, 668, 677, 521, and 704 on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00667-*:ml-experiments/00668-*:ml-experiments/00521-*:ml-experiments/00704-*:ml-experiments/00011-* \
        uv run --no-sync python ml-experiments/00735-*/e00735_diagnostic.py

Reads the stored step-5 forecasts of experiment 677 (`backtests/step5w_with_injected_sp1.parquet`, seed
1 only, monthly refits, contract origin 12:00 UTC), the cleaned power, and the GB outline in
`studies/beam_diffuse_split/domains/great_britain.geojson`. Writes private tables under
`$DATA_PATH_INTERNAL/ml-experiments/00735-*/diagnostic/` and prints every number the page quotes.

The score is the signed error of the median forecast (median minus observed, in points of capacity,
so a negative number is an under-forecast) at leads of 12 to 36 hours.
"""

import json
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Final

import e00011_term_dates as term_dates  # ty: ignore[unresolved-import]
import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import holidays
import numpy as np
import polars as pl
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final = "00735-coast-and-school-holiday-signals-in-the-step-5-network-part-1"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "diagnostic"
BUNDLE: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00705-*")) / "bundles" / "drop" / "2026-10"
OUTLINE: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "studies"
    / "beam_diffuse_split"
    / "domains"
    / "great_britain.geojson"
)
HOLDOUT: Final = date(2026, 7, 1)
HALF_TERM_FIRST_SATURDAYS: Final = (date(2025, 10, 25), date(2026, 2, 14), date(2026, 5, 23))
OCT_2026_FIRST_SATURDAY: Final = date(2026, 10, 24)
"""First day of the October 2026 half-term, from the colleague's message. It is after the holdout
start, and the 2026-27 term dates are not in `lincolnshire_term_dates.csv`."""
KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
QCOLS: Final = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")
CUT_START: Final = datetime(2025, 3, 1, tzinfo=UTC)
COAST_KM: Final = 10.0
LEAD_LO, LEAD_HI = 12.0, 36.0


def primaries() -> pl.DataFrame:
    """Return the 16 Primaries with their coordinates, from the production bundle's `primaries.json`."""
    rows = json.loads((BUNDLE / "primaries.json").read_text())["primaries"]
    return pl.DataFrame(
        [
            {
                k: r[k]
                for k in ("time_series_id", "d_label", "unit", "latitude", "longitude", "group")
            }
            for r in rows
        ]
    ).with_columns(pl.col("time_series_id").cast(pl.Int64))


def coast_distances(*, table: pl.DataFrame) -> pl.DataFrame:
    """Add `coast_km`: the distance from each Primary to the nearest point of the GB outline."""
    geo = json.loads(OUTLINE.read_text())
    land = shape(geo["geometry"])
    proj = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform
    boundary = transform(proj, land.boundary)
    km = [
        boundary.distance(transform(proj, Point(lon, lat))) / 1000.0
        for lat, lon in zip(table["latitude"], table["longitude"], strict=True)
    ]
    return table.with_columns(pl.Series("coast_km", km).round(2)).with_columns(
        (pl.col("coast_km") <= COAST_KM).alias("coastal")
    )


def holiday_calendar() -> pl.DataFrame:
    """Return one row per local date with `kind`: school holiday, bank holiday, or ordinary."""
    terms = term_dates.read_terms().sort("first_day")
    spans: list[tuple[date, date, str]] = []
    ordered = list(terms.iter_rows(named=True))
    for a, b in pairwise(ordered):
        gap = (b["first_day"] - a["last_day"]).days
        if a["academic_year"] == b["academic_year"] or gap < 60:
            name = {
                1: "autumn half-term",
                2: "Christmas",
                3: "February half-term",
                4: "Easter",
                5: "May half-term",
            }.get(a["term"], "summer")
            spans.append(
                (a["last_day"] + timedelta(days=1), b["first_day"] - timedelta(days=1), name)
            )
    bank = holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028))
    days = pl.date_range(date(2025, 11, 1), date(2026, 10, 31), eager=True).to_list()
    out = []
    for d in days:
        kind = "ordinary"
        for lo, hi, name in spans:
            if lo <= d <= hi:
                kind = f"school: {name}"
        if d in bank:
            kind = "bank holiday"
        out.append({"date": d, "kind": kind})
    return pl.DataFrame(out)


def main() -> None:
    """Print and write the coast distances and the holiday-day errors."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = coast_distances(table=primaries())
    table.write_csv(OUT / "coast_distance.csv")
    print("=== distance to the coast (km), nearest first")
    print(
        table.sort("coast_km").select(
            "time_series_id", "d_label", "latitude", "longitude", "coast_km", "coastal", "group"
        )
    )

    stored = (
        pl.read_parquet(
            next(ML_EXPERIMENTS_DATA_DIR.glob("00677-*"))
            / "backtests"
            / "step5w_with_injected_sp1.parquet"
        )
        .filter(pl.col("seed") == 1)
        .select(*KEYS, *QCOLS)
    )
    print("stored rows", stored.height)
    ids = sorted(stored["time_series_id"].unique().to_list())
    cap = (
        pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        .filter(pl.col("time_series_id").is_in(ids))
        .sort("time_series_id")["capacity"]
        .to_numpy()
        .astype(np.float64)
    )
    grid = lib.cleaned_grid(ids=ids)
    for label in ("D15", "D18"):
        tid = int(table.filter(pl.col("d_label") == label)["time_series_id"][0])
        grid[ids.index(tid), : lib.grid_index(times=pl.Series([CUT_START]))[0]] = np.nan
    row = stored["time_series_id"].replace_strict({t: i for i, t in enumerate(ids)}).to_numpy()
    vidx = lib.grid_index(times=stored["valid_time"])
    oidx = lib.grid_index(times=stored["power_fcst_init_time"])
    grid = np.pad(
        grid, ((0, 0), (0, max(0, int(vidx.max()) + 1 - grid.shape[1]))), constant_values=np.nan
    )
    scale = 100.0 / cap[row]
    y = grid[row, vidx] * scale
    lead_h = (vidx - oidx) / 2.0
    q = stored.select(list(QCOLS)).to_numpy().astype(np.float64) * scale[:, None]
    ok = np.isfinite(y)
    err = q[:, 5] - y
    pin = lib.pinball(y=y, q=q)
    crps = 2.0 * pin.mean(axis=1)
    rows = type("R", (), {"keys": stored, "y": y, "lead_h": lead_h})
    valid_local = stored["valid_time"].dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame(
        {
            "time_series_id": rows.keys["time_series_id"],
            "date": valid_local.dt.date(),
            "dow": valid_local.dt.weekday(),
            "lead_h": rows.lead_h,
            "err": err,
            "crps": crps,
            "y": rows.y,
            "ok": ok,
        }
    ).filter(pl.col("ok") & (pl.col("lead_h") >= LEAD_LO) & (pl.col("lead_h") < LEAD_HI))
    print("rows scored (12 to 36 h, finite truth):", frame.height)
    cal = holiday_calendar()
    frame = frame.join(cal, on="date", how="left").join(
        table.select("time_series_id", "d_label", "coast_km", "coastal"), on="time_series_id"
    )
    frame = frame.with_columns(
        pl.when(pl.col("dow") == 6)
        .then(pl.lit("Sunday"))
        .when(pl.col("dow") == 5)
        .then(pl.lit("Saturday"))
        .otherwise(pl.lit("Mon-Fri"))
        .alias("daytype"),
        (pl.col("date") >= HOLDOUT).alias("holdout"),
    )
    frame.write_parquet(OUT / "rows_12_36h.parquet")
    scored = frame.filter(~pl.col("holdout"))

    # 1. Excess signed error per Primary on school-holiday days over ordinary days of the same day type.
    ordinary = (
        scored.filter(pl.col("kind") == "ordinary")
        .group_by("d_label", "daytype")
        .agg(pl.col("err").mean().alias("err_ord"), pl.col("crps").mean().alias("crps_ord"))
    )
    for label, cond in (
        ("school holiday days", pl.col("kind").str.starts_with("school")),
        ("school half-term days", pl.col("kind").str.contains("half-term")),
        ("bank holiday days", pl.col("kind") == "bank holiday"),
    ):
        sub = (
            scored.filter(cond)
            .group_by("d_label", "daytype")
            .agg(
                pl.col("err").mean().alias("err_hol"),
                pl.col("crps").mean().alias("crps_hol"),
                pl.len().alias("n"),
            )
            .join(ordinary, on=["d_label", "daytype"])
        )
        sub = sub.with_columns(
            (pl.col("err_hol") - pl.col("err_ord")).alias("excess_err"),
            (pl.col("crps_hol") - pl.col("crps_ord")).alias("excess_crps"),
        )
        per = (
            sub.group_by("d_label")
            .agg(
                (pl.col("excess_err") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("excess_crps") * pl.col("n")).sum() / pl.col("n").sum(),
                pl.col("n").sum(),
            )
            .join(table.select("d_label", "time_series_id", "coast_km", "coastal"), on="d_label")
            .sort("coast_km")
        )
        per.write_csv(OUT / f"excess_{label.replace(' ', '_')}.csv")
        print(
            f"\n=== {label}: signed error excess over ordinary days of the same day type (points of capacity; negative = under-forecast more than usual)"
        )
        print(per.with_columns(pl.col("excess_err").round(2), pl.col("excess_crps").round(2)))
        coast = per.filter(pl.col("coastal"))["excess_err"].mean()
        inland = per.filter(~pl.col("coastal"))["excess_err"].mean()
        corr = float(np.corrcoef(per["coast_km"].to_numpy(), per["excess_err"].to_numpy())[0, 1])
        print(
            f"mean excess signed error: coastal {coast:.2f}, inland {inland:.2f}; correlation with coast distance {corr:.2f}"
        )

    # 2. The first Saturdays of half-term, and the first Saturday of October 2026 half-term.
    for d in (*HALF_TERM_FIRST_SATURDAYS, OCT_2026_FIRST_SATURDAY):
        day = (
            frame.filter(pl.col("date") == d)
            .group_by("d_label", "coast_km")
            .agg(pl.col("err").mean().round(2), pl.col("y").mean().round(1))
            .sort("err")
        )
        print(
            f"\n=== signed error of the median on {d} (a Saturday), by Primary, 12 to 36 h leads"
            + (" [HOLDOUT day: report only]" if d >= HOLDOUT else "")
        )
        print(day)

    # 3. All days: the Primaries with the largest absolute error on each school-holiday span.
    spans = (
        scored.filter(pl.col("kind") != "ordinary")
        .group_by("kind", "d_label")
        .agg(pl.col("err").mean().round(2), pl.len().alias("n"))
        .sort("kind", "err")
    )
    print("\n=== mean signed error by holiday kind and Primary")
    print(spans)
    spans.write_csv(OUT / "kind_by_primary.csv")


if __name__ == "__main__":
    main()
