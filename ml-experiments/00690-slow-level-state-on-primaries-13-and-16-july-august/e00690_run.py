"""Does the slow level of experiment 653 track the July to August 2026 change at ids 13 and 16?

Arm 1 describes the miss of the stored normal-running q50. Arm 2 fits the slow level (masked,
frozen design of experiment 653, noise units frozen on the pre-July residual as in experiment 685)
on experiment 685's residual panel and compares it with arm 1. Arm 3 repeats that on two control
Primaries chosen by a rule fixed in the plan. Arm 4 adds the level to q50 as a diagnostic.

Run from the repository root:

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search:\
$P/00685-event-table-on-the-final-test-window-all-primaries
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      env OMP_NUM_THREADS=1 uv run --no-sync python $P/00690-*/e00690_run.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00653_common as c653  # isort: skip
import e00685_run as r685  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

from e00653_common import block_means  # isort: skip
from e00653_core import freeze  # isort: skip
from e00653_slow import fit_slow, noise_unit, to_half_hours  # isort: skip
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BACKTEST: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "backtest_year_2025-11_to_2026-10"
OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00690-slow-level-state-on-primaries-13-and-16-july-august"
)
TARGETS: Final[tuple[int, ...]] = (13, 16)
EXCLUDED_FROM_CONTROL: Final[frozenset[int]] = frozenset({11, 13, 16, 18})
"""Ids with a flag in the one-year backtest; the controls are chosen from the rest."""
ARMS: Final[tuple[str, ...]] = ("masked", "huber", "rw30")
"""`masked` is the recommended design (the main arm); `huber` is S1 and `rw30` is S0."""
BASE_START: Final[datetime] = datetime(2026, 5, 1, tzinfo=UTC)
BASE_END: Final[datetime] = datetime(2026, 6, 15, tzinfo=UTC)
POST_START: Final[datetime] = datetime(2026, 7, 15, tzinfo=UTC)
POST_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2026, 10, 1, tzinfo=UTC)
SCORE_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_NAIVE: Final[datetime] = datetime(2022, 1, 1)  # noqa: DTZ001
SCORE_NAIVE: Final[datetime] = datetime(2026, 7, 1)  # noqa: DTZ001
N_BOOT: Final[int] = 2000
BLOCK_DAYS: Final[int] = 7
SEED: Final[int] = 690


def index_of(*, when: datetime) -> int:
    """Half-hour index of a UTC datetime on experiment 685's grid."""
    return int((when - r685.GRID_START).total_seconds() // 1800)


def stored_normal_running() -> pl.DataFrame:
    """One normal-running row per half-hour and id (smallest lead) joined to cleaned power."""
    fc = (
        pl.scan_parquet(BACKTEST / "primaries.parquet")
        .filter(pl.col("NRA_or_prevailing") == "NRA")
        .with_columns(lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")))
        .sort("time_series_id", "valid_time", "lead")
        .group_by("time_series_id", "valid_time", maintain_order=True)
        .first()
        .select("time_series_id", "valid_time", "q05", "q50", "q95")
    )
    truth = pl.scan_parquet(BACKTEST / "ground_truth_primaries.parquet").select(
        "time_series_id", "valid_time", "power"
    )
    return (
        fc.join(truth, on=["time_series_id", "valid_time"], how="inner")
        .filter(pl.col("power").is_not_null())
        .with_columns(resid=pl.col("power") - pl.col("q50"))
        .collect()
    )


def choose_controls(*, nr: pl.DataFrame, candidates: list[int]) -> tuple[list[int], pl.DataFrame]:
    """The two candidate ids whose coverage in 2026-07 and 2026-08 is closest to 0.90."""
    rows = []
    for s in candidates:
        d = nr.filter(
            (pl.col("time_series_id") == s)
            & (pl.col("valid_time") >= SCORE_START)
            & (pl.col("valid_time") < datetime(2026, 9, 1, tzinfo=UTC))
        )
        covs = [
            float(
                d.filter(pl.col("valid_time").dt.month() == m)
                .select(
                    ((pl.col("power") >= pl.col("q05")) & (pl.col("power") <= pl.col("q95"))).mean()
                )
                .item()
            )
            for m in (7, 8)
        ]
        rows.append(
            {
                "time_series_id": s,
                "cov_jul": covs[0],
                "cov_aug": covs[1],
                "distance": float(np.mean([abs(c - 0.9) for c in covs])),
            }
        )
    table = pl.DataFrame(rows).sort("distance")
    return table["time_series_id"].head(2).to_list(), table


def half_crossing(*, daily: np.ndarray, days: list[datetime]) -> tuple[float, float | None]:
    """Change between the base and post spans, and the first day after the base that passes half.

    Args:
        daily: A daily series, already smoothed.
        days: The date of each value.

    Returns:
        The change in megawatts, and days from 2026-07-01 to the first day from 2026-06-15 on at
        which the series passes half of the change in its direction (None if it never does).
    """
    d = np.array(days)
    base = np.nanmean(daily[(d >= BASE_START) & (d < BASE_END)])
    post = np.nanmean(daily[(d >= POST_START) & (d < POST_END)])
    change = float(post - base)
    half = base + 0.5 * change
    for v, day in zip(daily, d, strict=True):
        if day < BASE_END or not np.isfinite(v):
            continue
        if (change >= 0 and v >= half) or (change < 0 and v <= half):
            return change, (day - SCORE_START).total_seconds() / 86400
    return change, None


def smooth7(*, x: np.ndarray) -> np.ndarray:
    """Trailing-free centred 7-day mean of a daily series, NaN-aware."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - 3) : i + 4]
        if np.isfinite(w).sum() >= 4:
            out[i] = np.nanmean(w)
    return out


def slow_levels() -> dict[int, dict[str, np.ndarray]]:
    """Half-hour slow levels (three arms) and the panel residual per id, for the 15 panel series."""
    panel = r685.ExtendedPanel()
    residual = panel.mask_common_mode(residual=panel.residual)
    # Noise units frozen on the residual before 2026-07-01, with the old block count (as in 685).
    c653.N_BLOCKS = r685.OLD_LENGTH // c653.BLOCK
    try:
        frozen = freeze(
            panel=panel, residual={s: v[: r685.OLD_LENGTH] for s, v in residual.items()}, upto=None
        )
    finally:
        c653.N_BLOCKS = r685.NEW_LENGTH // c653.BLOCK
    info = pl.read_parquet(c653.SERIES_PATH).select("time_series_id", "site")
    site_of = dict(zip(info["time_series_id"], info["site"], strict=True))
    levels: dict[int, dict[str, np.ndarray]] = {}
    for s in c653.SERIES_IDS:
        site = site_of[s]
        y = block_means(x=residual[site])
        sigma = frozen.sigma[site]
        if not np.isfinite(sigma):
            msg = f"no frozen noise unit for id {s}"
            raise ValueError(msg)
        levels[s] = {arm: to_half_hours(slow=fit_slow(y=y, sigma=sigma, arm=arm)) for arm in ARMS}
        print(s, site, "sigma", round(sigma, 4), "noise", round(noise_unit(y=y), 4))
    return levels


def daily_series(*, values: np.ndarray, days: list[datetime], agg: str) -> np.ndarray:
    """Daily mean or median of a half-hourly array on experiment 685's grid."""
    out = []
    for d in days:
        i = index_of(when=d)
        chunk = values[i : i + 48]
        chunk = chunk[np.isfinite(chunk)]
        out.append(
            np.nan if len(chunk) < 24 else (np.median(chunk) if agg == "median" else chunk.mean())
        )
    return np.array(out)


def weekly(*, x: np.ndarray) -> np.ndarray:
    """Median of each run of 7 days, starting 2026-05-01."""
    return np.array([np.nanmedian(x[k : k + 7]) for k in range(0, len(x) - 6, 7)])


def best_lag(*, a: np.ndarray, b: np.ndarray, max_lag: int = 28) -> tuple[float, int]:
    """Correlation of daily series `a` with `b` shifted by each lag in days; the best one.

    A positive lag means `a` (the level estimate) trails `b` (the residual) by that many days.
    """
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: len(b) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30:
            continue
        c = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if c > best[0]:
            best = (c, lag)
    return best


def _stat(*, rows: np.ndarray, arrays: tuple[np.ndarray, ...]) -> tuple[float, float, float]:
    """Percent change in median absolute error, and the two coverages, over `rows`."""
    ok, e0, e1, cov0, cov1 = arrays
    r = rows[ok[rows]]
    return (
        float(100 * (np.median(e1[r]) / np.median(e0[r]) - 1)),
        float(100 * cov0[r].mean()),
        float(100 * cov1[r].mean()),
    )


def diagnostic(*, nr: pl.DataFrame, s: int, level: np.ndarray) -> dict[str, object]:
    """Median absolute error and coverage in 2026-07 to 2026-09 with and without the level added."""
    d = nr.filter(
        (pl.col("time_series_id") == s)
        & (pl.col("valid_time") >= SCORE_START)
        & (pl.col("valid_time") < WINDOW_END)
    ).sort("valid_time")
    idx = (
        (d["valid_time"].dt.replace_time_zone(None) - GRID_NAIVE).dt.total_minutes() // 30
    ).to_numpy()
    shift = level[idx]
    base_shift = np.nanmean(level[index_of(when=BASE_START) : index_of(when=BASE_END)])
    out: dict[str, object] = {"time_series_id": s, "n": d.height}
    p, q50, q05, q95 = (d[c].to_numpy().astype(float) for c in ("power", "q50", "q05", "q95"))
    day = ((d["valid_time"].dt.replace_time_zone(None) - SCORE_NAIVE).dt.total_days()).to_numpy()
    blocks = day // BLOCK_DAYS
    rng = np.random.default_rng(SEED)
    block_ids = np.unique(blocks)
    groups = [np.flatnonzero(blocks == b) for b in block_ids]
    for name, sh in (("level", shift), ("level_change", shift - base_shift)):
        ok = np.isfinite(sh)
        e0 = np.abs(p - q50)
        e1 = np.abs(p - (q50 + sh))
        cov0 = (p >= q05) & (p <= q95)
        cov1 = (p >= q05 + sh) & (p <= q95 + sh)

        arrays = (ok, e0, e1, cov0, cov1)
        point = _stat(rows=np.arange(len(p)), arrays=arrays)
        draws = np.array(
            [
                _stat(
                    arrays=arrays,
                    rows=np.concatenate(
                        [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                    ),
                )
                for _ in range(N_BOOT)
            ]
        )
        lo, hi = np.percentile(draws[:, 0], [2.5, 97.5])
        clo, chi = np.percentile(draws[:, 2] - draws[:, 1], [2.5, 97.5])
        out[name] = {
            "mae_change_pct": point[0],
            "mae_change_ci": [float(lo), float(hi)],
            "mae_base": float(np.median(e0[ok])),
            "mae_with": float(np.median(e1[ok])),
            "cov_base_pct": point[1],
            "cov_with_pct": point[2],
            "cov_change_pts_ci": [float(clo), float(chi)],
            "mean_shift_mw": float(np.nanmean(sh)),
        }
    return out


def main() -> None:
    """Run all four arms and write tables to the experiment's data folder."""
    OUT.mkdir(parents=True, exist_ok=True)
    nr = stored_normal_running()
    candidates = [s for s in c653.SERIES_IDS if s not in EXCLUDED_FROM_CONTROL]
    controls, control_table = choose_controls(nr=nr, candidates=candidates)
    control_table.write_csv(OUT / "control_choice.csv")
    print("controls", controls)
    levels = slow_levels()
    days = [BASE_START + timedelta(days=i) for i in range((WINDOW_END - BASE_START).days)]
    ids = [*TARGETS, *controls]
    results: dict[str, object] = {"controls": controls, "ids": ids, "per_id": {}}
    daily_rows = []
    for s in ids:
        d = nr.filter(pl.col("time_series_id") == s).filter(
            (pl.col("valid_time") >= BASE_START) & (pl.col("valid_time") < WINDOW_END)
        )
        # arm 1: daily median of the stored-q50 residual, then a centred 7-day mean
        by_day = (
            d.with_columns(day=pl.col("valid_time").dt.truncate("1d"))
            .group_by("day")
            .agg(resid=pl.col("resid").median(), power=pl.col("power").mean(), n=pl.len())
            .sort("day")
        )
        lookup = {r["day"]: r for r in by_day.iter_rows(named=True)}
        r1 = smooth7(
            x=np.array(
                [
                    lookup[x]["resid"] if x in lookup and lookup[x]["n"] >= 24 else np.nan
                    for x in days
                ]
            )
        )
        pw = np.array([lookup[x]["power"] if x in lookup else np.nan for x in days])
        change1, cross1 = half_crossing(daily=r1, days=days)
        per: dict[str, object] = {
            "residual_change_mw": change1,
            "residual_half_cross_day": cross1,
            "mean_power_mw": float(np.nanmean(pw)),
            "mean_power_base_mw": float(np.nanmean(pw[: (BASE_END - BASE_START).days])),
            "arms": {},
        }
        for arm in ARMS:
            lv = smooth7(x=daily_series(values=levels[s][arm], days=days, agg="mean"))
            change2, cross2 = half_crossing(daily=lv, days=days)
            corr, lag = best_lag(a=lv, b=r1)
            wl, wr = weekly(x=lv), weekly(x=r1)
            ok = np.isfinite(wl) & np.isfinite(wr)
            corr0 = float(np.corrcoef(wl[ok], wr[ok])[0, 1]) if ok.sum() > 3 else float("nan")
            per["arms"][arm] = {  # type: ignore[index]
                "level_change_mw": change2,
                "level_half_cross_day": cross2,
                "weekly_corr": corr0,
                "best_daily_corr": corr,
                "best_lag_days": lag,
                "within_14_days": (
                    cross1 is not None and cross2 is not None and abs(cross2 - cross1) <= 14
                ),
            }
            for k, x in enumerate(days):
                daily_rows.append(
                    {
                        "time_series_id": s,
                        "arm": arm,
                        "day": x,
                        "level": lv[k],
                        "residual_7d": r1[k],
                    }
                )
        per["diagnostic"] = {arm: diagnostic(nr=nr, s=s, level=levels[s][arm]) for arm in ARMS}
        results["per_id"][str(s)] = per  # type: ignore[index]
    (
        nr.filter(pl.col("time_series_id").is_in(ids))
        .filter((pl.col("valid_time") >= BASE_START) & (pl.col("valid_time") < WINDOW_END))
        .with_columns(week=pl.col("valid_time").dt.offset_by("0d").dt.truncate("1w"))
        .group_by("time_series_id", "week")
        .agg(
            median_resid_mw=pl.col("resid").median(),
            q05_resid_mw=pl.col("resid").quantile(0.05),
            q95_resid_mw=pl.col("resid").quantile(0.95),
            mean_power_mw=pl.col("power").mean(),
        )
        .sort("time_series_id", "week")
        .write_csv(OUT / "weekly_miss.csv")
    )
    pl.DataFrame(daily_rows).write_parquet(OUT / "daily_level_and_residual.parquet")
    nr.filter(pl.col("time_series_id").is_in(ids)).filter(
        (pl.col("valid_time") >= BASE_START) & (pl.col("valid_time") < WINDOW_END)
    ).write_parquet(OUT / "stored_residual_halfhours.parquet")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
