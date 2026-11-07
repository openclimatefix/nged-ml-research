# ruff: noqa: E501
"""Shared pieces of experiment 630: loading the joined frames and the day-block bootstrap."""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
TABLES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "tables"
NETWORKS: Final[tuple[str, ...]] = ("K1", "Candidate", "K3")
NETWORK_LABEL: Final[dict[str, str]] = {
    "K1": "K1 (class 1)",
    "Candidate": "Candidate (class 2)",
    "K3": "K3 (class 3)",
}
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
TRAIN_MONTHS: Final[tuple[str, ...]] = MONTHS[:3]
TEST_MONTHS: Final[tuple[str, ...]] = MONTHS[3:]
Q: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
TAU: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
BAND_EDGES: Final[tuple[float, ...]] = (0.0, 24.0, 72.0, 120.0, 240.0)
BAND_LABELS: Final[tuple[str, ...]] = ("0-24 h", "24-72 h", "72-120 h", "120-240 h")
N_BOOT: Final[int] = 20_000
SEED: Final[int] = 630
INDICATORS: Final[tuple[str, ...]] = ("a90", "a95", "b10", "b05")
"""Above q90, above q95, below q10, below q05."""


def load(*, network: str) -> pl.DataFrame:
    """One network's rows with indicator columns, lead band, hour, tercile and cold-night flags."""
    df = pl.read_parquet(WORK / f"{network}.parquet")
    y = pl.col("observed_power")
    return (
        df.with_columns(
            a90=(y > pl.col("q90")),
            a95=(y > pl.col("q95")),
            b10=(y < pl.col("q10")),
            b05=(y < pl.col("q05")),
            gap=(y - pl.col("q50")).cast(pl.Float64),
            band=pl.when(pl.col("lead_hours") < BAND_EDGES[1])
            .then(pl.lit(BAND_LABELS[0]))
            .when(pl.col("lead_hours") < BAND_EDGES[2])
            .then(pl.lit(BAND_LABELS[1]))
            .when(pl.col("lead_hours") < BAND_EDGES[3])
            .then(pl.lit(BAND_LABELS[2]))
            .otherwise(pl.lit(BAND_LABELS[3])),
            hour=pl.col("valid_time").dt.hour(),
            origin_date=pl.col("power_fcst_init_time").dt.date(),
            subset=pl.when(pl.col("event_free"))
            .then(pl.lit("event-free"))
            .otherwise(pl.lit("event")),
            median_demand=y.median().over("time_series_id"),
        )
        .with_columns(
            tercile=(
                pl.col("temperature_c").rank("ordinal").over("time_series_id")
                * 3
                / pl.len().over("time_series_id")
            )
            .ceil()
            .cast(pl.Int8)
            .replace_strict({1: "cold third", 2: "middle third", 3: "warm third"}),
            night=((pl.col("hour") >= 18) | (pl.col("hour") < 6)),
        )
        .with_columns(
            cold_night=(pl.col("tercile") == "cold third") & pl.col("night"),
            gap_share=pl.col("gap") / pl.col("median_demand"),
        )
    )


def day_table(*, df: pl.DataFrame, keys: list[str], values: list[str]) -> pl.DataFrame:
    """Per (keys, month, origin day) row counts and sums of the 0/1 or numeric `values` columns."""
    return df.group_by(list(dict.fromkeys([*keys, "month", "origin_date"]))).agg(
        pl.len().alias("n"), *[pl.col(v).cast(pl.Float64).sum().alias(v) for v in values]
    )


def bootstrap_ratio(
    *,
    days: pl.DataFrame,
    keys: list[str],
    values: list[str],
    ratios: dict[str, tuple[str, str]] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pl.DataFrame:
    """Mean of each `values` column per cell, pooled over months, with 95% day-resampling intervals.

    Days are resampled with replacement within each month (a stratum) and the cell's numerators and
    counts are summed over strata before dividing, so months with more days weigh more, as in the point
    estimate. The unit is the forecast-origin day because half-hour rows within a day are
    autocorrelated. `ratios` maps an output name to a (numerator, denominator) pair of `values`
    columns whose ratio of sums also gets an interval.
    """
    ratios = ratios or {}
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    cells = days.partition_by(keys, as_dict=True) if keys else {(): days}
    for cell, cell_days in cells.items():
        n_tot = np.zeros(n_boot)
        s_tot = {v: np.zeros(n_boot) for v in values}
        n_pt = cell_days["n"].sum()
        s_pt = {v: cell_days[v].sum() for v in values}
        for stratum in cell_days.partition_by("month", as_dict=True).values():
            d = stratum.height
            counts = rng.multinomial(d, np.full(d, 1.0 / d), size=n_boot).astype(np.float64)
            n_tot += counts @ stratum["n"].to_numpy().astype(np.float64)
            for v in values:
                s_tot[v] += counts @ stratum[v].to_numpy()
        row = dict(zip(keys, cell if isinstance(cell, tuple) else (cell,), strict=False))
        row["n"] = int(n_pt)
        row["n_days"] = cell_days.select(pl.struct("month", "origin_date").n_unique()).item()
        for v in values:
            ratio = s_tot[v] / np.maximum(n_tot, 1)
            row[v] = s_pt[v] / n_pt  # ty: ignore[unsupported-operator]
            row[f"{v}_lo"], row[f"{v}_hi"] = np.quantile(ratio, [0.025, 0.975])
        for name, (num, den) in ratios.items():
            row[name] = s_pt[num] / s_pt[den]  # ty: ignore[unsupported-operator]
            row[f"{name}_lo"], row[f"{name}_hi"] = np.quantile(
                s_tot[num] / s_tot[den], [0.025, 0.975]
            )
        out.append(row)
    return pl.DataFrame(out).sort(keys) if keys else pl.DataFrame(out)
