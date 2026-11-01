"""Experiment 456, step 1: block cross-fitted lagless residuals of the 16 Primaries (method b).

For each Primary, a lagless XGBoost median forecast (the 13 calendar and ERA5 features of
experiment 5, absolute-error objective) is fitted on every 6-month block except one and predicts
that block, so no block is predicted by a model that saw it. The residual is the observed power
(share of effective capacity, flagged zeros null) minus the prediction. D15 and D18 drop every
row before 2025-03-01. Output: `residuals_b.parquet` (Primary label, grid index, power, prediction)
and one booster per Primary and block in `boosters/`, both in the experiment's private work folder.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G \
    -p OOMPolicy=continue uv run --no-sync python \
    ml-experiments/00456-primary-demand-unmirrored-level-step-scan/e00456_residuals.py
```
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00456-primary-demand-unmirrored-level-step-scan" / "work"
)
SHORT_HISTORY_SITES: Final[frozenset[str]] = frozenset({"D15", "D18"})
SHORT_HISTORY_CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
N_THREADS: Final[int] = 2
MIN_TRAIN_ROWS: Final[int] = 5_000
PARAMS: Final[dict[str, object]] = {
    "objective": "reg:absoluteerror",
    "eta": 0.1,
    "max_depth": 6,
    "tree_method": "hist",
    "nthread": N_THREADS,
    "seed": 0,
}
ROUNDS: Final[int] = 300


def block_ids(*, valid_time: pl.Series) -> np.ndarray:
    """Return the 6-month block number (half-years since 2019) of each timestamp."""
    year = valid_time.dt.year().to_numpy()
    half = (valid_time.dt.month().to_numpy() > 6).astype(int)
    return (year - 2019) * 2 + half


def main() -> None:
    """Fit the cross-fitted lagless models and write the residual table."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    (WORK_DIR / "boosters").mkdir(exist_ok=True)
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    arrays = common.load_series_arrays(series=series)
    cut = int(common.grid_index(times=SHORT_HISTORY_CUT))
    frames: list[pl.DataFrame] = []
    for site in series["site"].to_list():
        started = time.time()
        data = arrays[site]
        power = data.power.astype(np.float64)
        grid = np.arange(len(power))
        keep = np.isfinite(power)
        if site in SHORT_HISTORY_SITES:
            keep &= grid >= cut
        block = block_ids(valid_time=data.valid_time)
        prediction = np.full(len(power), np.nan)
        for current in sorted(set(block[keep])):
            train = keep & (block != current)
            test = keep & (block == current)
            if train.sum() < MIN_TRAIN_ROWS:
                continue
            booster = xgb.train(
                PARAMS,
                xgb.DMatrix(data.features[train], label=power[train]),
                num_boost_round=ROUNDS,
            )
            booster.save_model(WORK_DIR / "boosters" / f"{site}_block{int(current):02d}.ubj")
            prediction[test] = booster.predict(xgb.DMatrix(data.features[test]))
        frames.append(
            pl.DataFrame(
                {
                    "site": site,
                    "grid": grid[keep].astype(np.int32),
                    "power": power[keep].astype(np.float32),
                    "prediction": prediction[keep].astype(np.float32),
                }
            )
        )
        print(f"{site}: {keep.sum()} rows in {time.time() - started:.0f} s", flush=True)
    pl.concat(frames).write_parquet(WORK_DIR / "residuals_b.parquet")


if __name__ == "__main__":
    main()
