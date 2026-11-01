"""Fit experiment 430's solar arms with effective capacity at p99.9 of earlier power.

The fit loop is experiment 430's (`e00430_fit.run_job`), with one change: each series' `output`
and `lag_power`, stored as power over the old capacity, are divided by that series' `ratio` (new
capacity over old) before anything reads them, so the training target, the lagged-power input,
the plant scale, the output ceiling, and the scored `actual` are all in units of the new capacity.
The new capacity of a refit is the 99.9th percentile of the series' cleaned power before the
refit's cutoff (the refit start minus experiment 310's refit gap). Predictions go to this
experiment's `solar_predictions/<arm>/`, in units of the new capacity; the capacity table is
`solar_capacity.json`. `--identity` uses ratio 1 as a gate against experiment 430's stored
forecasts.

    PYTHONPATH=<310>:<430>:<461 folders> OMP_NUM_THREADS=1 systemd-run --user --scope \
        -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00461-*/e00461_solar_fit.py ARM [ARM ...] \
        [--seeds 0,1,2] [--refits 2025-09,...] [--identity]
"""

import argparse
import dataclasses
import json
from datetime import datetime, timedelta
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_features as feats  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00430_common as common430  # ty: ignore[unresolved-import]
import e00430_fit as fit430  # ty: ignore[unresolved-import]
import e00461_common as common
import polars as pl

PRED_DIR: Final = common.DATA_DIR / "solar_predictions"
CAPACITY_FILE: Final[str] = "solar_capacity.json"
_RATIO: dict[str, float] = {}
_ORIGINAL: Final = feats.load_band_farm.__wrapped__  # ty: ignore[unresolved-attribute]
_CACHE: dict[tuple[str, str, float], object] = {}


def load_band_farm(site: str, band: str) -> feats.BandFarm:
    """Return experiment 310's band rows with `output` and `lag_power` in the new capacity."""
    ratio = _RATIO[site]
    key = (site, band, ratio)
    if key not in _CACHE:
        original = _ORIGINAL(site, band)
        frame = original.frame.with_columns(
            output=pl.col("output") / ratio, lag_power=pl.col("lag_power") / ratio
        )
        _CACHE[key] = dataclasses.replace(original, frame=frame)
    return _CACHE[key]  # ty: ignore[invalid-return-type]


def set_ratios(*, start: datetime, identity: bool) -> None:
    """Set the per-series ratio for the refit starting at `start` and record the capacities."""
    cutoff = start - timedelta(days=f310.REFIT_GAP_DAYS)
    table = {}
    for site in c310.FARMS:
        row = common.capacity(site=f"Solar {site}", cutoff=cutoff)
        table[site] = row
        _RATIO[site] = 1.0 if identity else float(row["ratio"])
    fit430.solar_base.cache_clear()  # these two caches hold values in the previous refit's units
    fit430.farm_ceiling.cache_clear()
    path = common.DATA_DIR / CAPACITY_FILE
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored[f"{start:%Y-%m}"] = table
    common.write_capacities(name=CAPACITY_FILE, table=stored)


def main() -> None:
    """Run the named arms, seeds, and refits one after another in this process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--refits", default=",".join(f"{r:%Y-%m}" for r in fit430.REFITS))
    parser.add_argument("--identity", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    refits = [r for r in fit430.REFITS if f"{r:%Y-%m}" in args.refits.split(",")]
    feats.load_band_farm = load_band_farm  # ty: ignore[invalid-assignment]
    common430.PRED_DIR = PRED_DIR / ("identity" if args.identity else "p999")
    for arm in args.arms:
        for seed in seeds:
            for start in refits:
                set_ratios(start=start, identity=args.identity)
                config = common430.ARMS[arm]
                print(fit430.run_job(arm=arm, config=config, seed=seed, start=start), flush=True)


if __name__ == "__main__":
    main()
