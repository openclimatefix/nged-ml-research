"""Experiment 456, step 4: does injecting into the half-hourly series and refitting change recall?

The injection validation adds steps to the daily residual, which leaves the cross-fitted model
unchanged. Here one Primary (D2) gets a 30-day step of 1.5 times the detection threshold added to
its half-hourly power at two quiet windows of `injection_quiet.csv`; every 6-month block model is
refitted on the injected series, and the scan is run on the new residuals. The model can absorb
part of the step, so detection here is the realistic figure.
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00456_residuals as resid  # ty: ignore[unresolved-import]
import e00456_scan as scan  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb

SITE: Final[str] = "D2"
SIZE: Final[float] = 1.5
DURATION: Final[int] = 30
N_WINDOWS: Final[int] = 2
WORK_DIR: Final[Path] = resid.WORK_DIR


def main() -> None:
    """Refit with an injected step and write the detection outcome."""
    series = common.series_table().filter(pl.col("site") == SITE)
    capacity = float(series["effective_capacity_mw"][0])
    data = common.load_series_arrays(series=series)[SITE]
    base = pl.read_parquet(WORK_DIR / "residuals_b.parquet").filter(pl.col("site") == SITE)
    levels = scan.daily_levels(
        grid=base["grid"].to_numpy(),
        residual=(base["power"] - base["prediction"]).to_numpy().astype(np.float64),
        n_days=121_248 // 48,
    )
    spread, sigma = scan.spread_of(levels=levels), scan.noise_sigma(levels=levels)
    points = SIZE * max(3.0 * spread, 2.0)
    windows = (
        pl.read_csv(WORK_DIR / "injection_quiet.csv")
        .filter(
            (pl.col("site") == SITE) & (pl.col("size") == SIZE) & (pl.col("duration") == DURATION)
            & ~pl.col("mirrored_injection")
        )["start"].to_list()[:N_WINDOWS]
    )  # fmt: skip
    power = data.power.astype(np.float64)
    keep = np.isfinite(power)
    block = resid.block_ids(valid_time=data.valid_time)
    grid = np.arange(len(power))
    rows = []
    for start in windows:
        injected = power.copy()
        injected[start * 48 : (start + DURATION) * 48] += points / 100.0
        prediction = np.full(len(power), np.nan)
        for current in sorted(set(block[keep])):
            train, test = keep & (block != current), keep & (block == current)
            booster = xgb.train(
                resid.PARAMS,
                xgb.DMatrix(data.features[train], label=injected[train]),
                num_boost_round=resid.ROUNDS,
            )
            prediction[test] = booster.predict(xgb.DMatrix(data.features[test]))
        lv = scan.daily_levels(
            grid=grid[keep], residual=injected[keep] - prediction[keep], n_days=len(levels)
        )
        found, _ = scan.detect(
            site=SITE, levels=lv, capacity_mw=capacity, params=scan.Params(),
            spread=spread, sigma=sigma,
        )  # fmt: skip
        hit = [s for s in found if abs(s.start_day - start) <= 2 and s.points * points > 0]
        measured = float(
            np.nanmean(lv[start : start + DURATION]) - np.nanmean(levels[start : start + DURATION])
        )
        rows.append(
            {"start_day": int(start), "injected_points": points, "detected": bool(hit),
             "measured_points": measured}
        )  # fmt: skip
    pl.DataFrame(rows).write_csv(WORK_DIR / "refit_check.csv")


if __name__ == "__main__":
    main()
