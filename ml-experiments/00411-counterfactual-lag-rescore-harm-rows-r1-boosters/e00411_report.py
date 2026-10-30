"""Experiment 411: summarise the harm-row scores written by `e00411_rescore.py`.

The gain of the counterfactual is the mean CRPS with real lags minus the mean CRPS with
counterfactual lags, on harm rows, in points of effective capacity. A positive gain means that
removing the event from the lags improves the forecast. The 95% interval resamples whole events
(a Primary's event with all its harm rows) with 2,000 resamples, as in the plan of experiment 404.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00411-*/e00411_report.py
```
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00411-counterfactual-lag-rescore-harm-rows-r1-boosters"
    / "work"
    / "rows"
)
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
N_BOOT: Final[int] = 2000
SEED: Final[int] = 0
DAY_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("days 1-2", 0, 48), ("days 3-5", 48, 120), ("days 6-10", 120, 240),
)  # fmt: skip
THRESHOLD: Final[float] = 0.1
"""Batches 1 and 2 of experiment 404 are skipped if the gain at the six is below this."""


def gain_with_interval(
    *, frame: pl.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the gain and its 95% event-cluster bootstrap interval."""
    cluster = frame.group_by("site", "event_id").agg(
        n=pl.len(), real=pl.col("crps_real").sum(), cf=pl.col("crps_cf").sum()
    )
    n, real, cf = (cluster[c].to_numpy().astype(float) for c in ("n", "real", "cf"))
    point = (real.sum() - cf.sum()) / n.sum()
    draws = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    boot = (real[draws].sum(axis=1) - cf[draws].sum(axis=1)) / n[draws].sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(point), float(low), float(high)


def main() -> None:
    """Print the gain for the six, the eight, and each lead band."""
    frames = [pl.read_parquet(f) for f in sorted(ROWS.glob("*.parquet"))]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        print("no harm rows written")
        return
    rows = pl.concat(frames)
    rng = np.random.default_rng(SEED)
    groups = {"six signed-MW": SIX, "eight MVA-only": EIGHT, "all": SIX + EIGHT}
    for name, sites in groups.items():
        sel = rows.filter(pl.col("site").is_in(sites))
        if sel.is_empty():
            continue
        events = sel.select("site", "event_id").unique().height
        print(f"== {name}: {sel.height} harm rows, {events} events")
        print(
            f"  CRPS real lags {sel['crps_real'].mean():.3f}, "
            f"counterfactual lags {sel['crps_cf'].mean():.3f}"
        )
        point, low, high = gain_with_interval(frame=sel, rng=rng)
        print(f"  gain {point:+.3f} (95% {low:+.3f} to {high:+.3f}); threshold {THRESHOLD}")
        for label, lo, hi in DAY_BANDS:
            in_band = (pl.col("lead_hours") > lo) & (pl.col("lead_hours") <= hi)
            band = sel.filter(in_band)
            if band.is_empty():
                continue
            point, low, high = gain_with_interval(frame=band, rng=rng)
            print(
                f"  {label}: {band.height} rows, gain {point:+.3f} (95% {low:+.3f} to {high:+.3f})"
            )
        for site in sites:
            one = sel.filter(pl.col("site") == site)
            if not one.is_empty():
                site_gain = float(np.mean(one["crps_real"]) - np.mean(one["crps_cf"]))
                print(f"  {site}: {one.height} rows, gain {site_gain:+.3f}")


if __name__ == "__main__":
    main()
