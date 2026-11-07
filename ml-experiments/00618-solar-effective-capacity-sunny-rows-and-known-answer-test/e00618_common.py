"""Experiment 618: shared code for the solar effective-capacity test.

Rows come from experiment 218's `farm_rows_cams.parquet` (CAMS irradiance, no new fetch). A
half-hour is *sunny* when the clear-sky index is at least 0.9, the clear-sky irradiance is at least
700 W/m2, and it sits in a run of at least six consecutive sunny half-hours. The capacity estimator
works on one number per sunny day and fits a piecewise-constant level by exact optimal partitioning
(an L0 change-point penalty). Farms appear only as labels A to F.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00618-solar-effective-capacity-sunny-rows-and-known-answer-test"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
TESTABLE: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""Solar E has seven months of sunny rows, so it is fitted but not known-answer tested."""
END_EXCLUSIVE: Final[str] = "2026-07-01"
CSI_MIN: Final[float] = 0.9
CLEAR_SKY_MIN_W_M2: Final[float] = 700.0
RUN_MIN: Final[int] = 6
MIN_SUNNY_PER_MONTH: Final[int] = 20
MIN_ROWS_PER_DAY: Final[int] = 6
MIN_SEGMENT_DAYS: Final[int] = 5
WINSOR_SIGMAS: Final[float] = 4.0
PENALTY_GRID: Final[tuple[float, ...]] = (3.0, 5.0, 8.0, 12.0, 18.0, 27.0, 40.0, 60.0)
"""Penalty per step, in units of sigma^2 of one daily value."""
Z90: Final[float] = 1.6449


def load_rows() -> pl.DataFrame:
    """Return every half-hour of the six farms to 2026-06-30 with the `sunny` flag set."""
    d = (
        pl.read_parquet(FARM_ROWS)
        .filter(pl.col("time") < pl.lit(END_EXCLUSIVE).str.to_datetime(time_zone="UTC"))
        .sort("site", "time")
    )
    ok = (
        (pl.col("ghi_w_m2") / pl.col("clear_sky") >= CSI_MIN)
        & (pl.col("clear_sky") >= CLEAR_SKY_MIN_W_M2)
        & pl.col("output").is_not_null()
    )
    d = d.with_columns(ok=ok)
    d = d.with_columns(
        brk=(
            (pl.col("time").diff().over("site") != pl.duration(minutes=30)).fill_null(True)
            | (pl.col("ok") != pl.col("ok").shift().over("site"))
        )
    )
    d = d.with_columns(run=pl.col("brk").cum_sum().over("site"))
    d = d.with_columns(run_len=pl.col("ok").sum().over("site", "run"))
    return d.with_columns(
        sunny=pl.col("ok") & (pl.col("run_len") >= RUN_MIN),
        month=pl.col("time").dt.truncate("1mo"),
        day=pl.col("time").dt.date(),
    ).drop("ok", "brk", "run", "run_len")


def expected_output(*, sunny: pl.DataFrame) -> np.ndarray:
    """Return the fitted expectation of output on the sunny rows of one farm (robust regression).

    The regression (irradiance, its square, air temperature, sun azimuth, sun elevation) is fitted
    once over the farm's whole record to 2026-06-30, so a ratio of 1 is the farm's long-run level.
    """
    g = sunny["ghi_w_m2"].to_numpy() / 1000.0
    az = np.deg2rad(sunny["sun_azimuth_deg"].to_numpy())
    x = np.column_stack(
        [
            np.ones_like(g),
            g,
            g**2,
            sunny["temperature_c"].to_numpy(),
            np.sin(az),
            np.cos(az),
            sunny["elevation_deg"].to_numpy() / 90.0,
        ]
    )
    y = sunny["output"].to_numpy().astype(float)
    w = np.ones_like(y)
    beta = np.zeros(x.shape[1])
    for _ in range(20):
        beta = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        r = y - x @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-9))
    return x @ beta


def daily_index(*, rows: pl.DataFrame, farm: str) -> pl.DataFrame:
    """Return one row per sunny day: `day`, `value` (median output / expectation), `n_rows`."""
    sunny = rows.filter((pl.col("site") == farm) & pl.col("sunny"))
    expected = expected_output(sunny=sunny)
    sunny = sunny.with_columns(ratio=pl.col("output") / pl.Series(expected))
    return (
        sunny.group_by("day")
        .agg(value=pl.col("ratio").median(), n_rows=pl.len())
        .filter(pl.col("n_rows") >= MIN_ROWS_PER_DAY)
        .sort("day")
    )


def sigma_from_differences(*, values: np.ndarray) -> float:
    """Return a step-robust noise sigma of one daily value, from the MAD of first differences."""
    d = np.diff(values)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)) + 1e-9


def fit_levels(*, values: np.ndarray, penalty: float) -> tuple[list[int], np.ndarray, float]:
    """Fit a piecewise-constant level by exact optimal partitioning with an L0 penalty per step.

    Values are winsorised to the median plus or minus `WINSOR_SIGMAS` sigma first. Returns the
    index of each step (the first day of a new segment), the level at every day, and sigma.
    """
    sigma = sigma_from_differences(values=values)
    med = float(np.median(values))
    v = np.clip(values, med - WINSOR_SIGMAS * sigma, med + WINSOR_SIGMAS * sigma)
    n = len(v)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v**2)])
    beta = penalty * sigma**2
    best = np.full(n + 1, np.inf)
    best[0] = -beta
    last = np.zeros(n + 1, dtype=int)
    for end in range(1, n + 1):
        starts = np.arange(0, end - MIN_SEGMENT_DAYS + 1)
        if len(starts) == 0:
            continue
        m = end - starts
        sse = (c2[end] - c2[starts]) - (c1[end] - c1[starts]) ** 2 / m
        cost = best[starts] + sse + beta
        j = int(np.argmin(cost))
        best[end] = cost[j]
        last[end] = int(starts[j])
    cuts: list[int] = []
    end = n
    while end > 0:
        cuts.append(last[end])
        end = last[end]
    cuts = sorted(set(cuts))
    steps = [c for c in cuts if c > 0]
    bounds = [*cuts, n]
    level = np.empty(n)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        level[a:b] = np.mean(values[a:b])
    return steps, level, sigma


def segment_halfwidth(
    *, steps: list[int], n: int, sigma: float, inflate: float = 1.0
) -> np.ndarray:
    """Return the 90% half-width of the level at each day: 1.645 sigma / sqrt(days in segment)."""
    bounds = [0, *steps, n]
    half = np.empty(n)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        half[a:b] = Z90 * sigma * inflate / np.sqrt(b - a)
    return half
