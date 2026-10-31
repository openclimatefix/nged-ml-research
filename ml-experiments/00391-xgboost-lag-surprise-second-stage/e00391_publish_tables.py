"""Publish the ENS-driven stage 1 surprise tables for experiment 392's arm R5.

Copies `work/stage1/ens/<site>.parquet` to `surprise_tables/half_hourly/<site>.parquet` and writes
`surprise_tables/origins/<site>.parquet` (6-hourly `z`, `r_mw`, `z_noise`, `r_sigma`, from
`e00391_features.origin_statistics`), plus a README that says how the tables were built and which
stage 1 fits did not train on each row.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00391-*/e00391_publish_tables.py
```
"""

import shutil
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00391_features as feats
import e00391_stage1 as stage1
import numpy as np
import polars as pl

OUT: Final[Path] = stage1.WORK_DIR.parent / "surprise_tables"
README: Final[str] = """# Stage 1 surprise tables of experiment 391 (ENS-driven)

Written by `ml-experiments/00391-xgboost-lag-surprise-second-stage/e00391_publish_tables.py`.
These are the tables experiment 391's arm N1 reads and experiment 392's arm R5 should read.

## How they were built

- Stage 1 is a lagless XGBoost quantile model (experiment 87's `n1h` inputs without the three
  origin lags, with `lead_hours`), 11 quantile levels, 150 rounds, eta 0.15, `colsample_*` 1.0,
  seed 0, no label filter on the training rows, one model per Primary and per month.
- The model is refitted at the first of each month from 2024-10-01 to 2026-06-01. A refit trains
  on control-member ENS rows whose valid time is at least 3.5 hours before the refit and whose
  forecast origin is at or before the refit, with weights halving every half year.
- A refit predicts the half-hours from its refit time to the next refit. **Every row of
  `half_hourly/<site>.parquet` was predicted by the refit named in its `refit` column, and that
  fit trained on no row at or after its refit time, so it never saw the half-hour it predicts.**
  The first month, 2024-10, is predicted by a model trained on 2024-04 to 2024-09 rows only.
- The weather for a half-hour comes from the control member of the ENS run with the latest
  forecast origin before the half-hour (lead 0.5 to 24 hours).
- `u` is the normal score of the reading's probability integral transform in the 11 quantiles
  (linear between levels, clipped to 0.005 to 0.995). `r_mw` is the reading minus the median in
  MW. `q01` to `q99` and `median_mw` are in MW.
- `origins/<site>.parquet` holds, per 6-hourly origin (00, 06, 12, and 18 UTC), `z` (mean `u`
  over the 48 half-hours before the origin, at least 36 present), `r_mw` (mean residual over the
  same window), `z_noise` (z minus the expanding median of earlier origins, over 1.4826 times
  their median absolute deviation; missing before 120 earlier origins), and `r_sigma`.
- D15 and D18 have a short history, so their tables start later and hold no row for the first
  months.
"""


def main() -> None:
    """Copy the half-hourly tables, write the origin statistics, and write the README."""
    (OUT / "half_hourly").mkdir(parents=True, exist_ok=True)
    (OUT / "origins").mkdir(parents=True, exist_ok=True)
    size = common.grid_times().len()
    for path in sorted((stage1.STAGE1_DIR / "ens").glob("*.parquet")):
        shutil.copy(path, OUT / "half_hourly" / path.name)
        table = pl.read_parquet(path)
        index = np.asarray(common.grid_index(times=table["valid_time"]))
        u, r = np.full(size, np.nan, np.float32), np.full(size, np.nan, np.float32)
        u[index], r[index] = table["u"].to_numpy(), table["r_mw"].to_numpy()
        stats = feats.origin_statistics(u=u, r=r)
        times = common.grid_times()[::12]
        pl.DataFrame(
            {
                "origin_time": times + timedelta(0),
                "z": stats.z,
                "r_mw": stats.r,
                "z_noise": stats.z_noise,
                "r_sigma": stats.r_sigma,
            }
        ).write_parquet(OUT / "origins" / path.name)
    (OUT / "README.md").write_text(README)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
