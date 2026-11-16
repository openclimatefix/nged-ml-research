# ruff: noqa: E501
"""Experiment 699: the held-out injected bank drawn from experiment 660's generator (GPU).

Runs experiment 677's `e00677_cand.py bank` unchanged, after replacing `e00581_bank.synthetic_clean`
for the bank's seeds (9,000,000 or more) by a function that returns the cleaned power plus the
change that experiment 660's generator makes to the real power panel. A training seed (below
9,000,000) still gets experiment 581's generator, which the bank command never calls with one.

The generator injects into the 15 Primaries of experiment 660's panel (D15 is missing, so D15 gets
no shift). The panel ends on 2026-06-30, and the wrapper sets every reading from 2026-07-01 on to
NaN before the generator runs. The wrapper slices the panel to the half-hours of the bank's scored
window (`first` to `last`) before injecting, so every event starts inside the window. A world drawn
over the whole panel would carry the permanent shifts of earlier years into the window as offsets
of several times the capacity. The change at each Primary, in MW or MVA, is divided by the
Primary's capacity to give a share of capacity, and the share is added to the cleaned power where
both the cleaned power and the panel hold a reading. The seed is 660,000 plus the fold month as a
number, and the wrapper writes it to `bank660/<month>_seed.json` under the chain's root.

Usage (through `gpu699.sh`, with `E00677_PHASE=predict`): python e00699_bank660.py P0+D50 bank
--models-from DIR --work-suffix SUFFIX --synthetic 0|1 --months M --seeds ... The arguments pass to
`e00677_cand.py`.
"""

import dataclasses
import json
import os
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import e00521_stats as s521  # ty: ignore[unresolved-import]
import e00581_bank as bank  # ty: ignore[unresolved-import]
import e00660_data as data660  # ty: ignore[unresolved-import]
import e00660_generator as gen660  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00699_common import BANK_BASE_581, BANK_BASE_660

EVENTS_PER_PRIMARY_YEAR = 24.0
CLEAN_GRID_START = datetime(2019, 9, 1, tzinfo=UTC)
OFFSET = int((data660.GRID_START - CLEAN_GRID_START).total_seconds() // 1800)
"""Half-hours from the cleaned power's grid start (2019-09-01) to the panel's (2022-01-01)."""
CLEAN_ROOT = Path(os.environ["E00677_ROOT"])
_original = bank.synthetic_clean
RECORD: dict = {}


def site_rows() -> dict[str, tuple[int, float]]:
    """Return, for each panel site label, the row of the 16-Primary stack and the capacity in MW or MVA."""
    sites = pl.read_csv(s521.DATA / "site_map.csv")
    by_id = {int(r["time_series_id"]): r for r in sites.iter_rows(named=True)}
    out = {}
    for label, info in data660.series_info().items():
        row = by_id[int(info["id"])]
        assert row["label"] == label, (label, row["label"])
        out[label] = (int(row["primary"]), float(row["capacity_mw"]))
    assert len(out) == 15
    return out


def synthetic_clean_660(
    *, clean: np.ndarray, first: int, last: int, seed: int
) -> tuple[np.ndarray, int]:
    """Return the cleaned power with experiment 660's injected changes in `[first, last]` (bank seeds only)."""
    if seed < BANK_BASE_581:
        return _original(clean=clean, first=first, last=last, seed=seed)
    month_number = seed - BANK_BASE_581
    own_seed = BANK_BASE_660 + month_number
    panel = data660.load_panel()
    assert panel.length == data660.GRID_LENGTH
    for series in panel.power.values():
        series[data660.idx_of(when=gen660.CUTOFF) :] = np.nan  # nothing at or after 2026-07-01
        assert not np.isfinite(series[data660.idx_of(when=gen660.CUTOFF) :]).any()
    lo, hi = max(first - OFFSET, 0), min(last - OFFSET + 1, panel.length)
    assert hi > lo, (first, last)
    # inject into the scored window only: a world drawn over the whole history would carry the permanent
    # shifts of earlier years into the window as offsets far larger than any event of the window
    window_panel = dataclasses.replace(
        panel,
        power={site: series[lo:hi].copy() for site, series in panel.power.items()},
        origin=panel.origin + timedelta(minutes=30 * lo),
    )
    config = dataclasses.replace(
        gen660.default_config(), events_per_primary_year=EVENTS_PER_PRIMARY_YEAR
    )
    world = gen660.inject(
        panel=window_panel, config=config, seed=own_seed, start_times=data660.logged_start_times()
    )
    fixed = clean.copy()
    rows = site_rows()
    n_changed = 0
    for site, (primary, capacity) in rows.items():
        base, moved = window_panel.power[site], world.power[site]
        delta = np.where(np.isfinite(moved) & np.isfinite(base), moved - base, 0.0)
        share = (delta / capacity).astype(np.float32)
        target = slice(OFFSET + lo, OFFSET + hi)
        block = fixed[primary, target]
        fixed[primary, target] = np.where(np.isfinite(block), block + share, block)
        n_changed += int((share != 0).sum())
    events = world.events.filter(pl.col("label") != gen660.HARD_NEGATIVE)
    n_events = events.height
    RECORD.update(
        seed=own_seed, month=month_number, events_per_primary_year=EVENTS_PER_PRIMARY_YEAR,
        events_in_window=n_events, half_hours_changed=n_changed, first=first, last=last,
        d15_unshifted=True,
    )  # fmt: skip
    folder = CLEAN_ROOT / "bank660"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{month_number // 100}-{month_number % 100:02d}_seed.json").write_text(
        json.dumps(RECORD, indent=1)
    )
    print(
        f"bank 660: seed {own_seed}, {n_events} events start in the window, {n_changed} half-hours changed",
        flush=True,
    )
    return fixed, n_events


if __name__ == "__main__":
    bank.synthetic_clean = synthetic_clean_660
    args = sys.argv[1:]
    assert args[1] == "bank"
    assert "_b660" in args[args.index("--work-suffix") + 1]
    cand = next(Path(__file__).resolve().parent.parent.glob("00677-*/e00677_cand.py"))
    sys.argv = [str(cand), *args]
    runpy.run_path(str(cand), run_name="__main__")
