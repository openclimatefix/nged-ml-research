"""Experiment 532, `e00532_a.py`: kept for the record.

Part A of experiment 532: refit 404's lag-free "further" module for seeds 0 to 4 and score
ensembles of 3, 4 and 5 heads on origin-time weather (origin mode only; reuses 531's code).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e00454_build as b454  # ty: ignore[unresolved-import]
import e00521_refs as r521  # ty: ignore[unresolved-import]
import e00531_a as a531  # ty: ignore[unresolved-import]
import e00531_common as m531  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00532-origin-time-weather-check-and-lag-free-rescore" / "work"
SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    """Fit five heads per month and write the ensemble cells."""
    levels = list(r521.LEVELS)
    for month in m531.MONTHS:
        started = time.monotonic()
        ref = pl.read_parquet(r521.E454 / "K2-S1" / f"{month}_s0_w-1.parquet").select(
            *r521.KEYS, "actual"
        )
        raw = m531.raw_test(month=month, mode="origin", dense={})
        heads = {}
        for seed in SEEDS:
            module, scaler, _ = a531.fit_further(month=month, seed=seed, device="cuda")
            heads[seed] = a531.head(
                module=module,
                scaler=scaler,
                x_raw=raw["x"][:, :35],
                site=raw["primary"],
                device="cuda",
            )
            del module
            torch.cuda.empty_cache()
        keys = pl.DataFrame(
            {
                "primary": raw["primary"].astype("int16"),
                "origin": raw["origin_idx"].astype("int32"),
                "target": raw["target_idx"].astype("int32"),
            }
        )
        for k in (3, 4, 5):
            q = b454.mix_equal(heads=[heads[s] for s in SEEDS[:k]])
            frame = keys.hstack(
                pl.DataFrame(q.astype("float32"), schema=levels, orient="row")
            ).join(ref, on=r521.KEYS, how="left")
            path = WORK / "cells" / f"LL-ens{k}-origin" / f"{month}_c0.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            frame.write_parquet(path)
        print(f"{month}: rows {len(keys):,}, {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
