"""Experiment 414: summarise the harm-row scores written by `e00414_rescore.py`.

The gain of the counterfactual is the mean CRPS with real lags minus the mean CRPS with
counterfactual lags, on harm rows, in points of effective capacity. A positive gain means that
removing the event from the lags improves the forecast. The 95% interval resamples whole events
(a Primary's event with all its harm rows) with 2,000 resamples, as in the plan of experiment 404.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00414-*/e00414_report.py
```
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ORIGINAL_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00411-counterfactual-lag-rescore-harm-rows-r1-boosters"
    / "work"
    / "rows"
)
"""Experiment 411's scored rows, whose events are the original 15."""
ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00414-post-hoc-gap-filled-counterfactual-lag-rescore"
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


def print_groups(*, rows: pl.DataFrame, rng: np.random.Generator) -> None:
    """Print the gain for the six, the eight, all 14, each lead band, and each Primary."""
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
                site_gain = float(
                    one["crps_real"].to_numpy().mean() - one["crps_cf"].to_numpy().mean()
                )
                print(f"  {site}: {one.height} rows, gain {site_gain:+.3f}")


def leave_one_event_out(*, rows: pl.DataFrame, sites: tuple[str, ...]) -> None:
    """Print the range of the gain when each event is dropped in turn."""
    sel = rows.filter(pl.col("site").is_in(sites))
    keys = sel.select("site", "event_id").unique().iter_rows()
    gains = []
    for site, event in keys:
        rest = sel.filter(~((pl.col("site") == site) & (pl.col("event_id") == event)))
        if not rest.is_empty():
            gains.append(rest["crps_real"].to_numpy().mean() - rest["crps_cf"].to_numpy().mean())
    if gains:
        print(f"  leave-one-event-out gain range: {min(gains):+.3f} to {max(gains):+.3f}")


def main() -> None:
    """Print the results for all events, the original events, and the added events."""
    frames = [
        pl.read_parquet(f).with_columns(pl.col("event_id").cast(pl.Int64))
        for f in sorted(ROWS.glob("*.parquet"))
    ]
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        print("no harm rows written")
        return
    rows = pl.concat(frames)
    original = (
        pl.concat(
            [
                pl.read_parquet(f).with_columns(pl.col("event_id").cast(pl.Int64))
                for f in sorted(ORIGINAL_ROWS.glob("*.parquet"))
            ]
        )
        .select("site", "event_id")
        .unique()
    )
    keyed = rows.join(
        original.with_columns(original=pl.lit(True)), on=["site", "event_id"], how="left"
    ).with_columns(pl.col("original").fill_null(False))
    rng = np.random.default_rng(SEED)
    subsets = {
        "ALL SCORED EVENTS": keyed,
        "ORIGINAL EVENTS (the 15 of experiment 411)": keyed.filter(pl.col("original")),
        "ADDED EVENTS (gaps filled)": keyed.filter(~pl.col("original")),
    }
    for name, subset in subsets.items():
        print(f"##### {name}")
        print_groups(rows=subset, rng=rng)
        for label, sites in (("six", SIX), ("eight", EIGHT)):
            print(f"  {label} signed-MW" if label == "six" else f"  {label} MVA-only")
            leave_one_event_out(rows=subset, sites=sites)


if __name__ == "__main__":
    main()
