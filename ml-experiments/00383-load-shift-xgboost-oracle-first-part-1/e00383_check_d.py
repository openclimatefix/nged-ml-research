"""Reproduction gate: does the refit of arm `d` reproduce experiment 87's stored weekly member?

Compares the quantiles that `e00383_fit.py --arms d` wrote at a subset of the weekly refits with
experiment 87's stored `n1h` files at the same refits and origins, and the CRPS of both against the
observed power. The gate passes when the mean CRPS differs by at most 0.01 points of effective
capacity at every Primary; the quantile gaps are printed beside it.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00383-* \
    uv run --no-sync python ml-experiments/00383-*/e00383_check_d.py
```
"""

import sys
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00383_assemble as asm
import numpy as np
import polars as pl

GATE_POINTS: Final[float] = 0.01
LEVEL_VALUES: Final[np.ndarray] = np.asarray(common.LEVELS, dtype=np.float64)


def crps_points(*, frame: pl.DataFrame, actual: np.ndarray) -> np.ndarray:
    """Return the quantile-score CRPS (points) of each row from its 11 quantiles (fractions)."""
    quantiles = frame.select(asm.LEVELS).to_numpy().astype(np.float64)
    diff = actual[:, None] - quantiles
    pinball = np.maximum(LEVEL_VALUES * diff, (LEVEL_VALUES - 1.0) * diff)
    return 2.0 * pinball.mean(axis=1) * 100.0


def main() -> None:
    """Print the gaps per Primary and raise if the gate fails."""
    series = common.series_table()
    sites = series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    mine = asm.read_arm(directory=asm.ARM_DIR, name="d", sites=sites, stored=False)
    stored = asm.read_arm(
        directory=asm.STORED_87 / "anchor_preds", name="n1h", sites=sites, stored=True
    )
    keys = list(common.KEYS)
    joined = mine.join(stored, on=keys, suffix="_stored", validate="1:1")
    print(f"{joined.height:,} shared rows of {mine.height:,} refitted, {stored.height:,} stored")
    actual = asm._actual_fraction(frame=joined).to_numpy()
    finite = np.isfinite(actual)
    mine_crps = crps_points(frame=joined.select(asm.LEVELS), actual=actual)
    stored_crps = crps_points(
        frame=joined.select({c: f"{c}_stored" for c in asm.LEVELS}.values()).rename(
            {f"{c}_stored": c for c in asm.LEVELS}
        ),
        actual=actual,
    )
    gaps = np.stack(
        [100.0 * (joined[c] - joined[f"{c}_stored"]).abs().to_numpy() for c in asm.LEVELS]
    )
    print(f"quantile gap, points: max {gaps.max():.4f}, mean {gaps.mean():.5f}")
    ids = joined["time_series_id"].to_numpy()
    label_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    worst = 0.0
    for series_id in np.unique(ids):
        rows = (ids == series_id) & finite
        gap = float(mine_crps[rows].mean() - stored_crps[rows].mean())
        worst = max(worst, abs(gap))
        print(
            f"{label_of[series_id]}: CRPS mine {mine_crps[rows].mean():.4f}, "
            f"stored {stored_crps[rows].mean():.4f}, gap {gap:+.4f}"
        )
    print(f"worst |CRPS gap| = {worst:.4f} points (gate {GATE_POINTS})")
    if worst > GATE_POINTS:
        print("REPRODUCTION GATE FAILED")
        sys.exit(1)
    print("reproduction gate passed")


if __name__ == "__main__":
    main()
