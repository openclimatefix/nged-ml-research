"""Put experiment 487's replay arms and their references on experiment 470's scored rows.

Experiment 487's seed-level forecasts (`work/seed_preds/<arm>/<month>_s<seed>.parquet`, raw 11
quantiles in capacity fractions) are mapped onto experiment 454's rows (`primary`, `origin`,
`target` on experiment 404's grid) the way experiment 470's `build_singles` does. Each arm is
written to `work/predictions/<arm>/<month>_c<cell>.parquet`, where the cell is the fit seed.

- `R25`, `R50`, `Rcold`: this experiment's fits, cells 0 to 2 (seeds 0 to 2).
- `R0`: experiment 425's stored arm A fits at seeds 0 to 2 (not refitted), copied from experiment
  470's `A-s0`, `A-s1`, and `A-s2`, which are the same fits on the same rows.
- `A-pool` (five-seed quantile average) and `A-s3`, `A-s4` (for the minimum detectable effect) are
  copied from experiment 470; `A-s0` is `R0` cell 0.

```bash
source .e_replay_env.sh   (sets PYTHONPATH; run from the repo root)
uv run --no-sync python ml-experiments/00487-*/e00487_build.py
```
"""

import shutil
from pathlib import Path
from typing import Final

import e00470_build as build470  # ty: ignore[unresolved-import]
import e00487_fit as fit  # ty: ignore[unresolved-import]
import polars as pl

KEYS: Final[list[str]] = build470.KEYS
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
NEW_ARMS: Final[tuple[str, ...]] = ("R25", "R50", "Rcold")
COPIED: Final[dict[str, tuple[str, ...]]] = {
    "R0": ("A-s0", "A-s1", "A-s2"),
    "A-pool": ("A-pool",),
    "A-s3": ("A-s3",),
    "A-s4": ("A-s4",),
}


def prediction_path(*, arm: str, month: str, cell: int) -> Path:
    """Return where this experiment stores an arm's cell on the scored rows."""
    return fit.WORK_DIR / "predictions" / arm / f"{month}_c{cell}.parquet"


def build() -> None:
    """Write every arm and month on experiment 470's rows."""
    sites = build470.site_map()
    for month in fit.MONTHS:
        reference = pl.read_parquet(build470.prediction_path(arm="A-s0", month=month, cell=0))
        rows = reference.select(KEYS)
        for arm in NEW_ARMS:
            for seed in fit.SEEDS:
                stored = build470.pool_stored(
                    folder=fit.WORK_DIR / "seed_preds" / arm,
                    month=month,
                    sites=sites,
                    seeds=(seed,),
                )
                frame = build470.on_rows(rows=rows, frame=stored)
                target = prediction_path(arm=arm, month=month, cell=seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".tmp")
                frame.write_parquet(temporary)
                temporary.rename(target)
        for arm, sources in COPIED.items():
            for cell, source in enumerate(sources):
                target = prediction_path(arm=arm, month=month, cell=cell)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(build470.prediction_path(arm=source, month=month, cell=0), target)
        print(f"{month}: wrote arms on {rows.height} rows", flush=True)


if __name__ == "__main__":
    build()
