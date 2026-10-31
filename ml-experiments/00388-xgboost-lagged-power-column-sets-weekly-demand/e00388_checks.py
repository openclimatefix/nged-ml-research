"""Print the median-minus-mean forecast per Primary for each arm (the write-up's Checks bullet).

The mean of a forecast is the mass-weighted mean of its 11 quantiles (`e00087_common.ATOM_MASSES`),
in points of effective capacity; the median is `q50`. A spread that shifts the forecast shows as a
non-zero gap.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations \
    uv run --no-sync python ml-experiments/00388-*/e00388_checks.py
```
"""

from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00388-xgboost-lagged-power-column-sets-weekly-demand" / "work"
)
E381: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00381-simple-weekly-xgboost-demand-against-gam-stack" / "work"
)
ARMS: Final[dict[str, Path]] = {
    "baseline": E381 / "arm_preds" / "monthly" / "control",
    **{a: WORK / "arm_preds" / "monthly" / a for a in ("a", "b", "c", "e", "f")},
}


def main() -> None:
    """Print per-Primary gaps."""
    weights = np.asarray(common.ATOM_MASSES, dtype=np.float64)
    weights = weights / weights.sum()
    series = common.series_table()
    label_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    table = {}
    for name, folder in ARMS.items():
        frame = pl.concat([pl.read_parquet(f) for f in sorted(folder.glob("*.parquet"))])
        quantiles = frame.select(list(common.LEVEL_COLUMNS)).to_numpy().astype(np.float64)
        gap = (frame["q50"].to_numpy() - quantiles @ weights) * 100
        by = pl.DataFrame({"id": frame["time_series_id"], "gap": gap}).group_by("id").mean()
        table[name] = {label_of[i]: g for i, g in by.iter_rows()}
    labels = sorted(table["baseline"])
    print("median minus mean forecast, points of capacity, mean over all runs")
    print("site  " + "  ".join(f"{a:>8s}" for a in ARMS))
    for lab in labels:
        print(f"{lab:5s} " + "  ".join(f"{table[a][lab]:+8.3f}" for a in ARMS))


if __name__ == "__main__":
    main()
