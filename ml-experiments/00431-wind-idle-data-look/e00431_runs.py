"""Experiment 431: run lengths of idle half-hours at high wind (the whole-farm outage test, I2).

    systemd-run --user --scope -p MemoryMax=4G -p OOMPolicy=continue nice -n 19 \
        uv run --no-sync python -I ml-experiments/00431-wind-idle-data-look/e00431_runs.py

Reads `features_private.parquet` (from `e00431_analysis.py`) and experiment 210's `report.json`.
A run is an unbroken stretch of half-hours (30 minutes apart) that are in the idle band with ERA5
100 m wind above 1.5 times the cut-in. The reference for "longer than weather explains" is the
farm's own ERA5 series: the longest spell with wind below the cut-in, which is the longest idle
spell that low wind alone could produce. Writes published-safe tables (lengths in hours, shares,
counts by year; no dates) under `data/ml-experiments/00431-wind-idle-data-look/`.
"""

import json
import os
from pathlib import Path
from typing import Any, Final

import polars as pl

DATA_ROOT: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
SRC: Final[Path] = DATA_ROOT / "00210-wind-idle-noise-mva-artefact" / "report.json"
OUT_DIR: Final[Path] = DATA_ROOT / "00431-wind-idle-data-look"
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
HIGH_WIND_FACTOR: Final[float] = 1.5
BUCKETS: Final[list[tuple[str, float, float]]] = [
    ("up to 3 h", 0.0, 3.0),
    ("3 to 12 h", 3.0, 12.0),
    ("12 to 24 h", 12.0, 24.0),
    ("1 to 7 days", 24.0, 168.0),
    ("over 7 days", 168.0, float("inf")),
]


def num(value: Any) -> float:
    """Convert a Polars scalar (or None) to a float, with None as NaN."""
    return float("nan") if value is None else float(value)


def runs_of(*, frame: pl.DataFrame, flag: pl.Expr, max_gap: int = 0) -> pl.DataFrame:
    """Return one row per run of flagged half-hours: site, year, half-hour count, length in hours.

    Two flagged half-hours belong to one run when at most `max_gap` unflagged (or missing)
    half-hours lie between them. The length counts the span from the first to the last flagged
    half-hour of the run, gaps included.
    """
    step = pl.duration(minutes=30 * (max_gap + 1))
    return (
        frame.sort("site", "valid_time")
        .filter(flag)
        .with_columns(
            run=(pl.col("valid_time").diff().over("site") > step).fill_null(value=True).cum_sum()
        )
        .group_by("site", "run")
        .agg(
            n=pl.len(),
            first=pl.col("valid_time").min(),
            last=pl.col("valid_time").max(),
            median_wind=pl.col("wind_speed").median(),
        )
        .with_columns(
            hours=(pl.col("last") - pl.col("first")).dt.total_minutes() / 60 + 0.5,
            year=pl.col("first").dt.year(),
        )
    )


def bucket_table(*, runs: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Share of flagged half-hours that sit in runs of each length bucket, per farm."""
    rows = []
    for site in SITES:
        sub = runs.filter(pl.col("site") == site)
        total = sub["n"].sum()
        for name, lo, hi in BUCKETS:
            part = sub.filter((pl.col("hours") > lo) & (pl.col("hours") <= hi))
            rows.append(
                {
                    "site": site,
                    "kind": kind,
                    "bucket": name,
                    "n_runs": part.height,
                    "n_half_hours": int(part["n"].sum()),
                    "share_of_half_hours": num(part["n"].sum()) / num(total) if total else 0.0,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Compute run-length tables for the pre-registered and the empirical cut-in speeds."""
    report = json.loads(SRC.read_text())
    frame = pl.read_parquet(
        OUT_DIR / "features_private.parquet",
        columns=["site", "valid_time", "wind_speed", "in_band", "power"],
    )
    buckets, summary, calm = [], [], []
    p95 = {s: float(report[s]["idle_floor"]["p95"]) for s in SITES}
    top = pl.col("site").replace_strict(p95, return_dtype=pl.Float64)
    variants = (
        ("used cut-in", "cut_in_used", pl.col("in_band"), 0),
        ("empirical cut-in", "cut_in_empirical", pl.col("in_band"), 0),
        ("post hoc: used cut-in, at or below band top, gaps up to 3", "cut_in_used", None, 3),
        (
            "post hoc: empirical cut-in, at or below band top, gaps up to 3",
            "cut_in_empirical",
            None,
            3,
        ),
    )
    for kind, key, in_band, max_gap in variants:
        speeds = {s: float(report[s][key]) for s in SITES}
        thr = pl.col("site").replace_strict(
            {s: HIGH_WIND_FACTOR * c for s, c in speeds.items()}, return_dtype=pl.Float64
        )
        idle = in_band if in_band is not None else pl.col("power") <= top
        runs = runs_of(frame=frame, flag=idle & (pl.col("wind_speed") > thr), max_gap=max_gap)
        buckets.append(bucket_table(runs=runs, kind=kind))
        for site in SITES:
            sub = runs.filter(pl.col("site") == site)
            summary.append(
                {
                    "site": site,
                    "kind": kind,
                    "n_runs": sub.height,
                    "longest_hours": num(sub["hours"].max()),
                    "p90_hours": num(sub["hours"].quantile(0.9)),
                    "n_over_24h": int(sub.filter(pl.col("hours") > 24).height),
                    "n_over_7d": int(sub.filter(pl.col("hours") > 168).height),
                    "n_over_24h_by_year": json.dumps(
                        sub.filter(pl.col("hours") > 24).group_by("year").len().sort("year").rows()
                    ),
                    "median_wind_in_runs_over_24h": num(
                        sub.filter(pl.col("hours") > 24)["median_wind"].median()
                    ),
                }
            )
        # Reference: spells of ERA5 wind below the cut-in (the idle that low wind could explain).
        cut = pl.col("site").replace_strict(speeds, return_dtype=pl.Float64)
        calm_runs = runs_of(frame=frame, flag=pl.col("wind_speed") < cut, max_gap=max_gap)
        for site in SITES:
            sub = calm_runs.filter(pl.col("site") == site)
            calm.append(
                {
                    "site": site,
                    "kind": kind,
                    "n_spells": sub.height,
                    "p99_hours": num(sub["hours"].quantile(0.99)),
                    "longest_hours": num(sub["hours"].max()),
                    "n_over_24h": int(sub.filter(pl.col("hours") > 24).height),
                }
            )
    pl.concat(buckets).write_parquet(OUT_DIR / "run_buckets.parquet")
    pl.DataFrame(summary).write_parquet(OUT_DIR / "run_summary.parquet")
    pl.DataFrame(calm).write_parquet(OUT_DIR / "calm_spell_reference.parquet")


if __name__ == "__main__":
    main()
