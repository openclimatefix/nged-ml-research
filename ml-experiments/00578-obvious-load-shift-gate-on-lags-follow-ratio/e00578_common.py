# ruff: noqa: E501, RET504
"""Experiment 578: shared loaders (labels, cleaned and corrected power, stored forecasts, daily gaps).

Time is a half-hour grid index from `GRID_START` (2019-09-01 UTC). Power is in fractions of
effective capacity, as in experiments 556, 565, and 572. All arrays are (16 Primaries, grid).
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00422_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00565_design6 as d6  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME: Final[str] = "00578-obvious-load-shift-gate-on-lags-follow-ratio"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / NAME / "work"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
EVENTS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored/work/events_corrected.parquet"
)
NRA: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails/work/nra_correction_share.npz"
)
PER_DAY: Final[int] = 48
GRID_START: Final[datetime] = datetime.fromtimestamp(prep.GRID_START_US / 1e6, UTC)


def sites() -> pl.DataFrame:
    """Return the site map (primary index, D-label, capacity, group)."""
    return pl.read_csv(s.DATA / "site_map.csv")


def day_of_idx(idx: np.ndarray | int) -> np.ndarray | int:
    """Return the UTC day number (days since the grid start) of a half-hour grid index."""
    return idx // PER_DAY


def clean() -> np.ndarray:
    """Return cleaned power over capacity (16, grid), NaN where flagged or cut."""
    return c.clean_stack()


def corrected_shift() -> np.ndarray:
    """Return the additive NRA correction (capacity fraction) that turns metered power into the counterfactual."""
    return -np.load(NRA)["f1.0"].astype(np.float32)


def expected_lag_free(
    *, shape: tuple[int, int], arm: tuple[Path, str] | None = None, seeds: tuple[int, ...] = (0, 1)
) -> np.ndarray:
    """Return the lag-free D5 (E2) median forecast of each half-hour (16, grid), shortest lead per time.

    Only the five fold months' origins exist, so other times are NaN. The forecast for time t comes from
    the latest stored origin at or before t, so the weather run is at or before t minus 9 h.
    """
    root, tag = arm or (d6.W, "D5")
    return fill_grid(root=root, tag=tag, seeds=seeds, grid=np.full(shape, np.nan, np.float32))


def fill_grid(*, root: Path, tag: str, seeds: tuple[int, ...], grid: np.ndarray) -> np.ndarray:
    """Fill `grid` with the seed-mean q50 of the shortest-lead forecast of each time."""
    sm = sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    acc = np.zeros_like(grid)
    cnt = np.zeros(grid.shape, np.int8)
    for seed in seeds:
        best = np.full(grid.shape, np.inf, np.float32)
        val = np.full(grid.shape, np.nan, np.float32)
        for m in MONTHS:
            for d in ("d0", "d14"):
                df = pl.read_parquet(
                    root / "fits" / tag / f"{m}_{d}_s{seed}.parquet",
                    columns=["time_series_id", "power_fcst_init_time", "valid_time", "q50"],
                )
                lead = (
                    (df["valid_time"] - df["power_fcst_init_time"]).dt.total_seconds() / 3600.0
                ).to_numpy()
                prim = np.array([prim_of[t] for t in df["time_series_id"].to_list()])
                idx = (
                    (df["valid_time"].dt.epoch("us").to_numpy() - prep.GRID_START_US)
                    // prep.STEP_US
                ).astype(np.int64)
                order = np.argsort(
                    -lead, kind="stable"
                )  # longest lead first, so the shortest lead is written last
                prim, idx, lead_o, q50 = (
                    prim[order],
                    idx[order],
                    lead[order],
                    df["q50"].to_numpy()[order],
                )
                better = lead_o < best[prim, idx]
                best[prim[better], idx[better]] = lead_o[better]
                val[prim[better], idx[better]] = q50[better]
        ok = np.isfinite(val)
        acc[ok] += val[ok]
        cnt[ok] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return out
