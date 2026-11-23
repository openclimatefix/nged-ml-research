# ruff: noqa: E501
"""Experiment 718: build the row tables of one arm and fold (WP5a). About 1.4 CPU hours for the arms with a weather measure.

The tables are built from the prediction-phase power array of the fold, because the training phase
blanks the fold month's power (design section 2.4). Each build writes the `era5` table (stage 1) and the
`ens` table (stages 2 and 3, and prediction) under `data/ml-experiments/00718-.../work/tables`. A
table covers all 16 Primaries and every origin day from the first with a full history to the last
scored origin. `--limit-days N` keeps the last N days only, and exists for tests.

Usage: run.sh e00718_build_tables.py ARM MONTH [--seed 0] [--limit-days N]
"""

import argparse
import sys
import time
from pathlib import Path

import e00713_inputs as inp  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_fit as fit_mod
import e00718_layout as lay
import e00718_retrieval as ret
import e00718_tables as tab
import numpy as np
from e00713_batch import Source  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR


def build_tables(
    *, source: Source, arm: str, days: np.ndarray, holidays: set | frozenset
) -> dict[str, ret.RowTable]:
    """Return the `era5` and `ens` tables of `arm` for all Primaries and `days`, built from `source`."""
    assert source.ens is not None
    arrays = tab.pool_arrays_from_source(
        source=source, with_calendar=arm == "UX", holidays=set(holidays)
    )
    primaries = np.arange(source.power_z.shape[0])
    return {
        fb.TABLE_ERA5: ret.build_table(arm=arm, arrays=arrays, primaries=primaries, days=days, target_provider=tab.era5_provider(arrays=arrays)),
        fb.TABLE_ENS: ret.build_table(arm=arm, arrays=arrays, primaries=primaries, days=days, target_provider=tab.ens_provider(ens=source.ens)),
    }  # fmt: skip


def days_of_fold(*, source: Source, last_origin_idx: int, limit: int | None) -> np.ndarray:
    """Return the origin day numbers from the first with a full history to the day of `last_origin_idx`."""
    first = (lay713.STEPS_PER_DAY * (lay713.MAX_AGE + 1) + 24) // lay713.STEPS_PER_DAY
    days = np.arange(first, last_origin_idx // lay713.STEPS_PER_DAY + 1)
    assert days.max() < source.n_grid // lay713.STEPS_PER_DAY
    return days if limit is None else days[-limit:]


def build(
    *, arm: str, month: str, seed: int, data_dir: Path, limit: int | None, device: str = "cpu"
) -> list[Path]:
    """Build and save the two tables of `arm` for the fold `month`."""
    paths = [
        arms.table_file(data_dir=data_dir, arm=arm, month=month, version=v)
        for v in (fb.TABLE_ERA5, fb.TABLE_ENS)
    ]
    if all(p.exists() for p in paths):
        print(f"{arm} {month}: tables exist, skipped", flush=True)
        return paths
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    assert data.test_origin_idx is not None
    last = int(np.max(data.test_origin_idx)) + lay713.NATIVE_OFFSET_STEPS
    days = days_of_fold(source=data.source, last_origin_idx=last, limit=limit)
    started = time.monotonic()
    tables = build_tables(
        source=data.source,
        arm=arm,
        days=days,
        holidays=lay713.england_bank_holidays(years=range(2019, 2029)),
    )
    for version, path in zip((fb.TABLE_ERA5, fb.TABLE_ENS), paths, strict=True):
        fit_mod.save_table(table=tables[version], path=path)
    print(
        f"{arm} {month}: {len(days)} days built in {time.monotonic() - started:.0f} s", flush=True
    )
    return paths


def main() -> int:
    """Parse the command line and build."""
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(lay.ROWS_PER_ARM))
    parser.add_argument("month")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-days", type=int, default=None)
    args = parser.parse_args()
    arms.install_own_cache(data_dir=ML_EXPERIMENTS_DATA_DIR)
    build(
        arm=args.arm,
        month=args.month,
        seed=args.seed,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        limit=args.limit_days,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
