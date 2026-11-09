"""Scoring of the one-year backtest files against observed power (CPU; no forecast is written).

Definitions follow experiments 620 and 633 (`e00633_score.py`): CRPS is twice the mean pinball loss
over the 11 stored quantile levels, and the normalised mean absolute error of the median is the
absolute error of `q50`, both in percent of the series' capacity (points of capacity). The manual
heuristic is the ensemble of the observed powers at the valid time minus 1 to 6 weeks and minus 49
55 weeks, and the climatology is the quantiles of the series' own cleaned power in the cell (local
month, local half-hour, local weekend flag) pooled over the 9 neighbouring cells. Intervals are
day-block bootstraps over forecast-origin days.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

FOLDER: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
STEP_MINUTES: Final[int] = 30
TAUS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
QCOLS: Final[list[str]] = [
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
]
LAGS_H: Final[tuple[int, ...]] = (
    168,
    336,
    504,
    672,
    840,
    1008,
    8232,
    8400,
    8568,
    8736,
    8904,
    9072,
    9240,
)
DELAY_H: Final[float] = 6.0
"""Power is available only up to 6 hours before the origin."""
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "days 0 to 1": (0.0, 24.0),
    "days 1 to 3": (24.0, 72.0),
    "days 3 to 7": (72.0, 168.0),
    "days 7 to 10": (168.0, 240.0),
    "days 10 to 14": (240.0, 336.0),
}
N_BOOT: Final[int] = 2000


def grid_index(*, times: pl.Series) -> np.ndarray:
    """Return each timestamp's half-hour index on the grid that starts at `GRID_START`."""
    start_us = int(GRID_START.timestamp() * 1_000_000)
    return (
        ((times.dt.epoch("us") - start_us) // (STEP_MINUTES * 60_000_000))
        .to_numpy()
        .astype(np.int64)
    )


def cleaned_grid(*, ids: list[int]) -> np.ndarray:
    """Return the cleaned power of each series on the dense half-hourly grid (NaN where absent)."""
    scan = pl.scan_delta(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
    frame = (
        scan.filter(pl.col("drop_reason").is_null() & pl.col("time_series_id").is_in(ids))
        .select("time_series_id", "time", "power")
        .collect()
    )
    last = int(grid_index(times=frame["time"]).max())
    grid = np.full((len(ids), last + 1), np.nan, np.float64)
    for row, tid in enumerate(ids):
        part = frame.filter(pl.col("time_series_id") == tid)
        grid[row, grid_index(times=part["time"])] = part["power"].to_numpy()
    return grid


def manual_heuristic(
    *, grid: np.ndarray, row: np.ndarray, vidx: np.ndarray, lead_h: np.ndarray
) -> np.ndarray:
    """Return the heuristic's 11 quantiles, using only lags that exist at the origin.

    A lag of `h` hours reads the power at the valid time minus `h`, which must be at least
    `DELAY_H` hours before the origin, so `h >= lead + 6`. The 1-week lag (168 h) therefore drops
    out beyond a lead of 162 hours.
    """
    members = np.full((len(row), len(LAGS_H)), np.nan, np.float64)
    for k, h in enumerate(LAGS_H):
        j = vidx - 2 * h
        ok = (lead_h + DELAY_H <= h) & (j >= 0) & (j < grid.shape[1])
        members[ok, k] = grid[row[ok], j[ok]]
    have = np.isfinite(members).any(axis=1)
    out = np.full((len(row), len(TAUS)), np.nan, np.float64)
    out[have] = np.nanquantile(members[have], TAUS, axis=1, method="linear").T
    return out


def climatology(
    *,
    grid: np.ndarray,
    row: np.ndarray,
    vidx: np.ndarray,
    cutoff_idx: np.ndarray,
    use_weekend: bool = True,
) -> np.ndarray:
    """Return the climatology's 11 quantiles, trained on the power before each row's cutoff.

    `use_weekend` is `False` for generators, whose power does not follow the working week.
    """
    n = grid.shape[1]
    times = pl.Series(
        pl.datetime_range(
            GRID_START,
            datetime.fromtimestamp(GRID_START.timestamp() + (n - 1) * 1800, tz=UTC),
            interval="30m",
            time_zone="UTC",
            eager=True,
        )
    )
    local = times.dt.convert_time_zone("Europe/London")
    month = (local.dt.month() - 1).to_numpy().astype(np.int64)
    half = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy().astype(np.int64)
    weekend = (local.dt.weekday() >= 6).to_numpy().astype(int) * int(use_weekend)
    out = np.full((len(row), len(TAUS)), np.nan, np.float64)
    offsets = [(dm, dh) for dm in (-1, 0, 1) for dh in (-1, 0, 1)]
    for series in np.unique(row):
        for cutoff in np.unique(cutoff_idx[row == series]):
            train = np.flatnonzero(np.isfinite(grid[series, :cutoff]))
            pooled = pl.concat(
                [
                    pl.DataFrame(
                        {
                            "cell": (((month[train] + dm) % 12) * 48 + (half[train] + dh) % 48) * 2
                            + weekend[train],
                            "v": grid[series, train],
                        }
                    )
                    for dm, dh in offsets
                ]
            )
            table = pooled.group_by("cell").agg(
                [
                    pl.col("v").quantile(float(t), "linear").alias(f"q{i}")
                    for i, t in enumerate(TAUS)
                ]
            )
            look = {r[0]: r[1:] for r in table.iter_rows()}
            rows = np.flatnonzero((row == series) & (cutoff_idx == cutoff))
            cells = (month[vidx[rows]] * 48 + half[vidx[rows]]) * 2 + weekend[vidx[rows]]
            out[rows] = np.array(
                [look.get(int(c), (np.nan,) * len(TAUS)) for c in cells], np.float64
            )
    return out


def pinball(*, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the pinball loss of each of the 11 quantiles, shape (rows, 11)."""
    d = y[:, None] - q
    return np.maximum(TAUS * d, (TAUS - 1.0) * d)


def losses(*, y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row CRPS, median absolute error, and q05-q95 coverage (all in percent)."""
    return {
        "crps": 2.0 * pinball(y=y, q=q).mean(axis=1),
        "mae": np.abs(y - q[:, 5]),
        "coverage": ((y >= q[:, 1]) & (y <= q[:, 9])).astype(np.float64) * 100.0,
    }


def day_bootstrap(
    *, day: np.ndarray, values: dict[str, np.ndarray], seed: int = 668
) -> dict[str, np.ndarray]:
    """Return bootstrap draws of each system's mean, resampling origin days with shared picks."""
    days, inv = np.unique(day, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    return {
        name: np.bincount(inv, weights=v)[pick].sum(axis=1) / counts[pick].sum(axis=1)
        for name, v in values.items()
    }


def summarise(
    *,
    mask: np.ndarray,
    day: np.ndarray,
    per_system: dict[str, dict[str, np.ndarray]],
    reference: str,
    label: dict[str, str],
) -> list[dict]:
    """Return one table row per (system, metric) on the masked rows, with the model's gains.

    `reference` is the model's name. Every other system gets a gain of the model over it, as a
    percentage of that system's mean, with a day-block bootstrap 95% interval.
    """
    rows: list[dict] = []
    if not mask.any():
        return rows
    for metric in ("crps", "mae", "coverage"):
        draws = day_bootstrap(
            day=day[mask], values={s: per_system[s][metric][mask] for s in per_system}
        )
        for system in per_system:
            mean = float(per_system[system][metric][mask].mean())
            lo, hi = np.quantile(draws[system], [0.025, 0.975])
            row = {
                **label,
                "system": system,
                "metric": metric,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "rows": int(mask.sum()),
                "origin_days": len(np.unique(day[mask])),
            }
            if system != reference and metric != "coverage":
                gain = 100.0 * (1.0 - draws[reference] / draws[system])
                row["gain_pct"] = 100.0 * (
                    1.0 - float(per_system[reference][metric][mask].mean()) / mean
                )
                row["gain_lo"], row["gain_hi"] = (
                    float(x) for x in np.quantile(gain, [0.025, 0.975])
                )
            rows.append(row)
    return rows
