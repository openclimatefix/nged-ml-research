"""Experiment 628: write the 11 stored quantiles of every joint run of experiment 436.

Repeats the demand, wind, and solar reading of experiment 436's `e00436_build.py` and keeps the 11
quantiles (25, 702, 11), in points of capacity, instead of interpolating them to 51 values.
Writes `work/q11/<date>.npy` as float32. Unscored half-hours hold the constant fallback that
experiment 436 used (their actual is NaN, so they are never scored).

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00628-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00628-*/e00628_build.py
```
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00087_common as common  # noqa: E402  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # noqa: E402  # ty: ignore[unresolved-import]
import e00434_score as s434  # noqa: E402  # ty: ignore[unresolved-import]
import e00436_build as build  # noqa: E402  # ty: ignore[unresolved-import]
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from studies.sources import ML_EXPERIMENTS_DATA_DIR  # noqa: E402

SLUG: Final[str] = "00628-quantile-form-client-side-scenario-rehydration"
OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work" / "q11"


def main() -> None:
    """Write one 11-quantile array per joint run."""
    OUT.mkdir(parents=True, exist_ok=True)
    series = adapter.series_table()
    ids = series["time_series_id"].to_list()
    wind, wind_q, _ = build.wind_rows()
    solar, solar_q, _ = solar_rows_quantiles()
    joint = [
        datetime.strptime(line, "%Y-%m-%d").replace(hour=9, tzinfo=UTC)
        for line in (build.WORK_DIR / "runs_found.txt").read_text().splitlines()[1:]
    ]
    stored: dict[str, pl.DataFrame] = {}
    for init in joint:
        out = OUT / f"{init:%Y-%m-%d}.npy"
        if out.exists():
            continue
        month = f"{init:%Y-%m}"
        if month not in stored:
            stored = {month: s434.stored_arm_a(month=month)}
        sub = stored[month].filter(pl.col("power_fcst_init_time") == init)
        order = {sid: i for i, sid in enumerate(ids)}
        sub = sub.with_columns(
            p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64)
        ).sort("p", "valid_time")
        n_steps = build.N_STEPS
        if sub.height != len(ids) * n_steps:
            msg = f"{init}: demand rows {sub.height}"
            raise ValueError(msg)
        q = np.zeros((build.N_NODES, n_steps, 11))
        demand = sub.select(common.LEVEL_COLUMNS).to_numpy().reshape(build.N_PRIMARIES, n_steps, 11)
        q[: build.N_PRIMARIES] = np.sort(demand, axis=-1) * 100.0
        for f, farm in enumerate(build.WIND_FARMS):
            node = build.N_PRIMARIES + f
            rows = wind.filter((pl.col("power_fcst_init_time") == init) & (pl.col("site") == farm))
            step = np.rint(rows["lead_hours"].to_numpy() * 2).astype(int) - 1
            ridx = rows["row"].to_numpy()
            fallback = np.median(wind_q[ridx], axis=0)[5] * 100.0
            q[node] = fallback
            q[node, step] = np.sort(wind_q[ridx], axis=-1) * 100.0
        init00 = init.replace(hour=0)
        for f, farm in enumerate(build.SOLAR_FARMS):
            node = build.N_PRIMARIES + len(build.WIND_FARMS) + f
            rows = solar.filter(
                (pl.col("init_time") == init00) & (pl.col("time_series_id") == farm)
            )
            step = ((rows["time"] - init).dt.total_minutes() // 30).to_numpy().astype(int) - 1
            ok = (step >= 0) & (step < n_steps)
            ridx = rows["row"].to_numpy()[ok]
            q[node] = 0.0
            q[node, step[ok]] = np.sort(solar_q[ridx], axis=-1) * 100.0
        np.save(out, q.astype(np.float32))
        print(f"{init:%Y-%m-%d} written", flush=True)


def solar_rows_quantiles() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Return experiment 436's solar rows and quantiles (the same call, kept here for clarity)."""
    return build.solar_rows()


if __name__ == "__main__":
    main()
