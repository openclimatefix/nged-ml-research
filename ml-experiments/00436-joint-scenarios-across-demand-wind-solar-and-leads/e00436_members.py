"""Write the member-wise quantiles of wind and solar for the joint runs of experiment 436.

No fit. For each joint run (`work/runs/<date>.npz`), writes `work/members/<date>.npz` with
`member_q` (9, 702, 51, 11): for the three wind farms (experiment 429 `memberwise`, seeds averaged)
and the six solar farms (experiment 430 `a2_memberwise`, seeds 0 to 2 averaged), the 11
quantiles of the forecast that each ensemble member produces alone, in points of capacity, on the
same 702 half-hour grid as the run file. Half-hours a node has no row for hold zero.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00436-*/env.sh
nice -n 19 uv run --no-sync python ml-experiments/00436-*/e00436_members.py
```
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00436_build as build  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

MEMBER_DIR: Final[Path] = build.WORK_DIR / "members"
SOLAR_MEMBERWISE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
    / "a2_memberwise"
)
SOLAR_MEMBERWISE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
"""Experiment 430 ran the member-wise solar arm with three seeds."""
N_MEMBER_NODES: Final[int] = len(build.WIND_FARMS) + len(build.SOLAR_FARMS)


def wind_member_quantiles() -> np.ndarray:
    """Return the seed-mean member-wise quantiles (rows, 51, 11) of the scored wind rows."""
    total = None
    for seed in build.WIND_SEEDS:
        part = np.load(build.WIND_PARTS / f"memberwise_members_h{seed}.npy", mmap_mode="r")
        total = np.array(part, dtype=np.float64) if total is None else total + part
    assert total is not None
    return total / len(build.WIND_SEEDS)


def solar_member_quantiles() -> tuple[pl.DataFrame, np.ndarray]:
    """Return the member-wise solar rows and their seed-mean quantiles (rows, 51, 11)."""
    tables, arrays = [], []
    for path in sorted(SOLAR_MEMBERWISE.glob("s0_*.parquet")):
        base = pl.read_parquet(path).select("time", "init_time", "time_series_id")
        stack = []
        for seed in SOLAR_MEMBERWISE_SEEDS:
            name = path.name.replace("s0_", f"s{seed}_")
            other = pl.read_parquet(path.with_name(name)).select(
                "time", "init_time", "time_series_id"
            )
            if not other.equals(base):
                msg = f"{name}: rows differ from seed 0"
                raise ValueError(msg)
            z = np.load(path.with_name("paths_" + name.replace(".parquet", ".npz")))
            stack.append(z["member_q"].astype(np.float64))
        tables.append(base)
        arrays.append(np.mean(stack, axis=0))
    return pl.concat(tables).with_row_index("row"), np.concatenate(arrays)


def main() -> None:
    """Write one member file per joint run."""
    MEMBER_DIR.mkdir(parents=True, exist_ok=True)
    wind, _, _ = build.wind_rows()
    wind_mq = wind_member_quantiles()
    solar, _, _ = build.solar_rows()
    solar_check, solar_mq = solar_member_quantiles()
    if not solar.select("time", "init_time", "time_series_id").equals(
        solar_check.select("time", "init_time", "time_series_id")
    ):
        msg = "a2_memberwise rows differ from g0_onepass rows"
        raise ValueError(msg)
    for file in sorted((build.WORK_DIR / "runs").glob("*.npz")):
        out = MEMBER_DIR / file.name
        if out.exists():
            continue
        init = datetime.fromisoformat(f"{np.load(file)['init']}+00:00").astimezone(UTC)
        member_q = np.zeros((N_MEMBER_NODES, build.N_STEPS, build.N_MEMBERS, 11), dtype=np.float32)
        for f, farm in enumerate(build.WIND_FARMS):
            rows = wind.filter((pl.col("power_fcst_init_time") == init) & (pl.col("site") == farm))
            step = np.rint(rows["lead_hours"].to_numpy() * 2).astype(int) - 1
            member_q[f, step] = np.sort(wind_mq[rows["row"].to_numpy()], axis=-1) * 100.0
        for f, farm in enumerate(build.SOLAR_FARMS):
            rows = solar.filter(
                (pl.col("init_time") == init.replace(hour=0)) & (pl.col("time_series_id") == farm)
            )
            step = ((rows["time"] - init).dt.total_minutes() // 30).to_numpy().astype(int) - 1
            ok = (step >= 0) & (step < build.N_STEPS)
            member_q[len(build.WIND_FARMS) + f, step[ok]] = (
                np.sort(solar_mq[rows["row"].to_numpy()[ok]], axis=-1) * 100.0
            )
        np.savez_compressed(out, member_q=member_q)
        print(f"{init:%Y-%m-%d} written", flush=True)


if __name__ == "__main__":
    main()
