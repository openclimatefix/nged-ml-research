"""Walk-forward calibration of the quantile bands in the one-year backtest files (CPU only).

PYTHONPATH=ml-experiments/00668-* uv run --no-sync python \
    ml-experiments/00683-*/e00683_calibrate.py bsps
PYTHONPATH=ml-experiments/00668-* uv run --no-sync python \
    ml-experiments/00683-*/e00683_calibrate.py primaries NRA

Arms: 0 the file as stored; 1 every quantile moved away from `q50` by 1.0934 (experiment 613);
2 a per-series, per-output, per-lead-band scale on the distance from `q50`, fitted at each origin
month start on the 90 days of valid times that end 6 hours before the month starts; 3 arm 2 plus a
per-series, per-output level shift on every quantile. The scale is the 0.90 quantile of `z`, the
distance of the actual power from `q50` divided by the stored half-width on the side where the
actual lies, so the scaled q05-to-q95 band covers 90% of the fitting rows.

Writes, under `<data>/00683-*/` (private): `predictions/<arm>/<name>[_<kind>].parquet` (the
calibrated quantiles in the file's own units), `daily/<name>[_<kind>].parquet` (per arm, series,
origin day, and lead band: row count and sums of each metric, so any slice and any day-block
bootstrap follow from this table), and `fits/<name>[_<kind>].parquet` (every fitted parameter).
Scored rows follow experiments 666 to 669: finite cleaned observed power (wind idle half-hours set
to zero; solar daylight half-hours only).
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00668_score_lib as lib
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.solar import zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00683-walk-forward-calibration-of-the-quantile-bands-in-the-one-year-backtest-files"
)
WIDTH_FACTOR: Final[float] = 1.0934
"""Experiment 613's fixed factor on the distance of every quantile from the median."""
LOOKBACK_DAYS: Final[int] = 90
"""Length of the fitting window in days of valid time (45 and 180 are sensitivity checks)."""
DELAY_HALF_HOURS: Final[int] = 12
"""Power is available only up to 6 hours before an origin or a month start."""
MIN_FIT_DAYS: Final[int] = 20
"""A cell with fewer distinct valid days of fitting rows keeps a scale of 1."""
MIN_HALF_WIDTH: Final[float] = 0.05
"""Rows whose stored half-width is under this many points of capacity are left out of the fit."""
SCALE_BOUNDS: Final[tuple[float, float]] = (0.5, 3.0)
TARGET_COVERAGE: Final[float] = 0.90
FIRST_SCORED_MONTH: Final[str] = "2025-12"
"""2025-11 has no earlier forecast to fit on, so no arm is scored on it."""
ARMS: Final[tuple[int, ...]] = (0, 1, 2, 3)
TAIL_LEVELS: Final[dict[str, int]] = {"01": 0, "05": 1, "95": 9, "99": 10}
"""Index of each tail level in the 11 stored quantiles."""
WIND_IDLE_AS_ZERO: Final[bool] = True


def band_index(*, lead_h: np.ndarray) -> np.ndarray:
    """Return the lead band of each row: 0 for days 0 to 1, 1 for 1 to 3, 2 for 3 to 7, 3 beyond."""
    return np.searchsorted(np.array([24.0, 72.0, 168.0]), lead_h, side="right")


BAND_LABELS: Final[tuple[str, ...]] = ("days 0 to 1", "days 1 to 3", "days 3 to 7", "days 7 to 14")


def z_scores(*, y: np.ndarray, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> np.ndarray:
    """Return the distance of `y` from `q50` over the stored half-width on the side `y` lies."""
    above = y >= q50
    half = np.where(above, q95 - q50, q50 - q05)
    return np.where(half >= MIN_HALF_WIDTH, np.abs(y - q50) / np.maximum(half, 1e-9), np.nan)


def fit_month(
    *,
    y: np.ndarray,
    q: np.ndarray,
    vidx: np.ndarray,
    row: np.ndarray,
    band: np.ndarray,
    ok: np.ndarray,
    cutoff_v: int,
    n_series: int,
) -> dict[str, np.ndarray]:
    """Fit the scale (arm 2), the shift, and the scale after the shift (arm 3) for one month."""
    window = (vidx <= cutoff_v) & (vidx > cutoff_v - 2 * 24 * LOOKBACK_DAYS) & ok
    scale2 = np.ones((n_series, 4))
    scale3 = np.ones((n_series, 4))
    shift = np.zeros(n_series)
    count = np.zeros((n_series, 4), dtype=np.int64)
    clipped = np.zeros((n_series, 4), dtype=bool)
    for i in range(n_series):
        sel_i = window & (row == i)
        if len(np.unique(vidx[sel_i] // 48)) < MIN_FIT_DAYS:
            continue
        d = float(np.median(y[sel_i] - q[sel_i, 5]))
        shift[i] = d
        for b in range(4):
            sel = sel_i & (band == b)
            count[i, b] = int(sel.sum())
            if len(np.unique(vidx[sel] // 48)) < MIN_FIT_DAYS:
                continue
            args = {"q05": q[sel, 1], "q50": q[sel, 5], "q95": q[sel, 9]}
            z2 = z_scores(y=y[sel], **args)
            z3 = z_scores(y=y[sel] - d, **args)
            z2, z3 = z2[np.isfinite(z2)], z3[np.isfinite(z3)]
            if len(z2) > 0 and len(z3) > 0:
                raw2, raw3 = np.quantile(z2, TARGET_COVERAGE), np.quantile(z3, TARGET_COVERAGE)
                scale2[i, b] = np.clip(raw2, *SCALE_BOUNDS)
                scale3[i, b] = np.clip(raw3, *SCALE_BOUNDS)
                clipped[i, b] = not (SCALE_BOUNDS[0] < raw2 < SCALE_BOUNDS[1])
    return {"scale2": scale2, "scale3": scale3, "shift": shift, "count": count, "clipped": clipped}


def apply_arm(
    *, q: np.ndarray, factor: np.ndarray, shift: np.ndarray, floor_zero: np.ndarray
) -> np.ndarray:
    """Return `q50 + shift + factor * (q - q50)` per row, clipped at zero where `floor_zero`."""
    median = q[:, 5:6]
    out = median + shift[:, None] + factor[:, None] * (q - median)
    out = np.where(floor_zero[:, None], np.maximum(out, 0.0), out)
    return np.sort(out, axis=1)


def row_metrics(*, y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row CRPS, MAE, coverage, one-sided misses, and tail pinball loss and hits."""
    out = lib.losses(y=y, q=q)
    out["below"] = (y < q[:, 1]).astype(np.float64) * 100.0
    out["above"] = (y > q[:, 9]).astype(np.float64) * 100.0
    for tag, k in TAIL_LEVELS.items():
        tau = lib.TAUS[k]
        d = y - q[:, k]
        out[f"pin{tag}"] = np.maximum(tau * d, (tau - 1.0) * d)
        out[f"hit{tag}"] = (y <= q[:, k]).astype(np.float64) * 100.0
    out["width"] = q[:, 9] - q[:, 1]
    return out


def load_truth(*, name: str, ids: list[int]) -> np.ndarray:
    """Return the cleaned observed power of each series on the dense half-hourly grid."""
    truth = pl.read_parquet(lib.FOLDER / f"ground_truth_{name}.parquet").filter(
        pl.col("time_series_id").is_in(ids)
    )
    index = {t: i for i, t in enumerate(ids)}
    gi = lib.grid_index(times=truth["valid_time"])
    gr = truth["time_series_id"].replace_strict(index).to_numpy()
    grid = np.full((len(ids), int(gi.max()) + 1), np.nan, np.float64)
    grid[gr, gi] = truth["power"].to_numpy().astype(np.float64)
    if name == "wind":
        idle = truth.filter(pl.col("research_flag").is_not_null())
        grid[
            idle["time_series_id"].replace_strict(index).to_numpy(),
            lib.grid_index(times=idle["valid_time"]),
        ] = 0.0 if WIND_IDLE_AS_ZERO else np.nan
    return grid


def main(name: str, kind: str | None) -> None:  # noqa: PLR0915
    """Calibrate and score one file (and one output of the Primaries)."""
    label = name if kind is None else f"{name}_{kind}"
    forecast = pl.read_parquet(lib.FOLDER / f"{name}.parquet")
    if kind is not None:
        forecast = forecast.filter(pl.col("NRA_or_prevailing") == kind)
    ids = sorted(forecast["time_series_id"].unique().to_list())
    index = {t: i for i, t in enumerate(ids)}
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    capacity = capacity.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")
    cap = capacity["capacity"].to_numpy().astype(np.float64)
    nonneg = np.array(
        [(name in ("wind", "solar")) or (u == "MVA") for u in capacity["power_unit"].to_list()]
    )
    grid = load_truth(name=name, ids=ids)
    row = forecast["time_series_id"].replace_strict(index).to_numpy()
    vidx = lib.grid_index(times=forecast["valid_time"])
    oidx = lib.grid_index(times=forecast["power_fcst_init_time"])
    if vidx.max() >= grid.shape[1]:
        grid = np.pad(
            grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
        )
    lead_h = (vidx - oidx) / 2.0
    band = band_index(lead_h=lead_h)
    scale = 100.0 / cap[row]
    y = grid[row, vidx] * scale
    q = forecast.select(lib.QCOLS).to_numpy().astype(np.float64) * scale[:, None]
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    if name == "solar":
        sites = pv_sites().sort("time_series_id")
        sites = sites.filter(pl.col("time_series_id").is_in(ids))
        daylight = np.zeros(len(y), dtype=bool)
        for i, rec in enumerate(sites.iter_rows(named=True)):
            sel = np.flatnonzero(row == i)
            mid = forecast["valid_time"].gather(sel.tolist()).dt.offset_by("-15m")
            daylight[sel] = (
                np.asarray(zenith(stamps=mid, latitude=rec["latitude"], longitude=rec["longitude"]))
                < 90.0
            )
        ok &= daylight
    y = np.where(ok, y, 0.0)
    origin_month = forecast["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    day = oidx // 48
    months = sorted(set(origin_month))
    fits: list[dict] = []
    arm_q = {a: np.empty_like(q) for a in ARMS[1:]}
    arm_q[0] = q
    for m in months:
        start = pl.Series([datetime(int(m[:4]), int(m[5:]), 1, tzinfo=UTC)])
        cutoff_v = int(lib.grid_index(times=start)[0]) - DELAY_HALF_HOURS
        fit = fit_month(
            y=y, q=q, vidx=vidx, row=row, band=band, ok=ok, cutoff_v=cutoff_v, n_series=len(ids)
        )
        sel = np.flatnonzero(origin_month == m)
        r, b = row[sel], band[sel]
        zero = np.zeros(len(sel))
        arm_q[1][sel] = apply_arm(
            q=q[sel], factor=np.full(len(sel), WIDTH_FACTOR), shift=zero, floor_zero=nonneg[r]
        )
        arm_q[2][sel] = apply_arm(
            q=q[sel], factor=fit["scale2"][r, b], shift=zero, floor_zero=nonneg[r]
        )
        arm_q[3][sel] = apply_arm(
            q=q[sel], factor=fit["scale3"][r, b], shift=fit["shift"][r], floor_zero=nonneg[r]
        )
        fits.extend(
            {
                "file": label,
                "time_series_id": tid,
                "origin_month": m,
                "band": BAND_LABELS[k],
                "fit_rows": int(fit["count"][i, k]),
                "arm2_clipped": bool(fit["clipped"][i, k]),
                "scale_arm2": float(fit["scale2"][i, k]),
                "scale_arm3": float(fit["scale3"][i, k]),
                "shift_arm3_pct_capacity": float(fit["shift"][i]),
            }
            for i, tid in enumerate(ids)
            for k in range(4)
        )
        print(label, m, "fitted", flush=True)
    for sub in ("predictions", "daily", "fits"):
        (EXPERIMENT_DIR / sub).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(fits).write_parquet(EXPERIMENT_DIR / "fits" / f"{label}.parquet")
    key_columns = ["time_series_id", "valid_time", "nwp_init_time", "power_fcst_init_time"]
    keys = forecast.select(key_columns + (["NRA_or_prevailing"] if kind is not None else []))
    daily: list[pl.DataFrame] = []
    for a in ARMS:
        metrics = row_metrics(y=y, q=arm_q[a])
        frame = pl.DataFrame(
            {
                "arm": np.full(len(y), a, dtype=np.int8),
                "time_series_id": np.asarray(ids, dtype=np.int8)[row],
                "origin_day": day.astype(np.int32),
                "origin_month": origin_month,
                "band": band.astype(np.int8),
                "ok": ok,
                **metrics,
            }
        ).filter(pl.col("ok"))
        names = [
            c
            for c in frame.columns
            if c not in ("arm", "time_series_id", "origin_day", "origin_month", "band", "ok")
        ]
        daily.append(
            frame.group_by("arm", "time_series_id", "origin_day", "origin_month", "band").agg(
                pl.len().alias("n"), *[pl.col(c).sum().alias(f"sum_{c}") for c in names]
            )
        )
        if a > 0:
            out = keys.with_columns(
                *[
                    pl.Series(c, (arm_q[a][:, k] / scale).astype(np.float32))
                    for k, c in enumerate(lib.QCOLS)
                ]
            )
            folder = EXPERIMENT_DIR / "predictions" / f"arm{a}"
            folder.mkdir(parents=True, exist_ok=True)
            out.write_parquet(folder / f"{label}.parquet")
        print(label, "arm", a, "scored", int(ok.sum()), flush=True)
    pl.concat(daily).write_parquet(EXPERIMENT_DIR / "daily" / f"{label}.parquet")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
