# ruff: noqa: E501
"""Experiment 718: build the row tables of one arm and fold (WP5a). About 1.4 CPU hours for the arms
with a weather measure.

The tables are built from the prediction-phase power array of the fold, because the training
phase blanks the fold month's power (design section 2.4). Each build writes the `era5` table
(stage 1) and the `ens` table (stages 2 and 3, and prediction) under
`data/ml-experiments/00718-.../work/tables`. A table covers all 16 Primaries and every origin day
from the first with a full history to the last scored origin. `--limit-days N` keeps the last N
days only, and exists for tests.

Usage: run.sh e00718_build_tables.py ARMS MONTH [--seed 0] [--limit-days N]
"""  # noqa: D205

import argparse
import resource
import sys
import time
from collections.abc import Sequence
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
    *, source: Source, arms_to_build: Sequence[str], days: np.ndarray, holidays: set | frozenset
) -> dict[str, dict[str, ret.RowTable]]:
    """Return the `era5` and `ens` tables of every arm in `arms_to_build` for all Primaries and
    `days`, built in one pass from `source`.
    """  # noqa: D205
    assert source.ens is not None
    arrays = tab.pool_arrays_from_source(
        source=source, with_calendar="UX" in arms_to_build, holidays=set(holidays)
    )
    primaries = np.arange(source.power_z.shape[0])
    common = {"arms": list(arms_to_build), "arrays": arrays, "primaries": primaries, "days": days}
    era5 = ret.build_tables(target_provider=tab.era5_provider(arrays=arrays), **common)  # ty: ignore[invalid-argument-type]
    ens = ret.build_tables(target_provider=tab.ens_provider(ens=source.ens), **common)  # ty: ignore[invalid-argument-type]
    return {a: {fb.TABLE_ERA5: era5[a], fb.TABLE_ENS: ens[a]} for a in arms_to_build}


def days_of_fold(*, source: Source, last_origin_idx: int, limit: int | None) -> np.ndarray:
    """Return the origin day numbers from the first with a full history to the day of `last_origin_idx`."""
    first = (lay713.STEPS_PER_DAY * (lay713.MAX_AGE + 1) + 24) // lay713.STEPS_PER_DAY
    days = np.arange(first, last_origin_idx // lay713.STEPS_PER_DAY + 1)
    assert days.max() < source.n_grid // lay713.STEPS_PER_DAY
    return days if limit is None else days[-limit:]


def build(
    *, arms_to_build: Sequence[str], month: str, seed: int, data_dir: Path, limit: int | None,
    device: str = "cpu",
) -> None:  # fmt: skip
    """Build and save the two tables of every arm in `arms_to_build` for the fold `month`, in one
    pass over the fold.

    A table that exists is kept only when its provenance (builder code, phase, arm, month, and
    standardiser fingerprint, input digest, and `--limit-days`) matches. A table that differs
    only in `--limit-days` is rebuilt; any other difference stops the build.
    """  # noqa: D205
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    assert data.test_origin_idx is not None
    assert pipe.fold.PHASE == "predict", "tables are built from the prediction-phase power"
    digest = fit_mod.inputs_digest(source=data.source, cutoff_idx=data.cutoff_idx)
    todo = []
    for arm in arms_to_build:
        paths = [
            arms.table_file(data_dir=data_dir, arm=arm, month=month, version=v)
            for v in (fb.TABLE_ERA5, fb.TABLE_ENS)
        ]
        meta = fit_mod.table_meta(
            arm=arm, month=month, fingerprint=data.fingerprint, phase=pipe.fold.PHASE,
            digest=digest, limit_days=limit,
        )  # fmt: skip
        diffs = [fit_mod.provenance_diff(path=p, expect=meta) for p in paths if p.exists()]
        if len(diffs) == len(paths) and not any(diffs):
            print(f"{arm} {month}: tables exist with matching provenance, skipped", flush=True)
        elif diffs and all(set(d) <= {"limit_days"} for d in diffs):
            print(
                f"{arm} {month}: a table of another --limit-days exists; it is rebuilt", flush=True
            )
            todo.append(arm)
        else:
            assert not any(diffs), f"{arm} {month}: an existing table has other provenance: {diffs}"
            todo.append(arm)
    if not todo:
        return
    last = int(np.max(data.test_origin_idx)) + lay713.NATIVE_OFFSET_STEPS
    days = days_of_fold(source=data.source, last_origin_idx=last, limit=limit)
    started = time.monotonic()
    tables = build_tables(
        source=data.source,
        arms_to_build=todo,
        days=days,
        holidays=lay713.england_bank_holidays(years=range(2019, 2029)),
    )
    for arm, by_version in tables.items():
        meta = fit_mod.table_meta(
            arm=arm, month=month, fingerprint=data.fingerprint, phase=pipe.fold.PHASE,
            digest=digest, limit_days=limit,
        )  # fmt: skip
        for version, table in by_version.items():
            path = arms.table_file(data_dir=data_dir, arm=arm, month=month, version=version)
            fit_mod.save_table(table=table, path=path, meta=meta)
    cpu = time.process_time()
    print(
        f"{','.join(todo)} {month}: {len(days)} days built in {time.monotonic() - started:.0f} s ({cpu:.0f} CPU s)",
        flush=True,
    )


def main() -> int:
    """Parse the command line and build."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arms", help=f"comma-separated arms, or `all`; arms: {sorted(lay.ROWS_PER_ARM)}"
    )
    parser.add_argument("month")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument(
        "--max-cpu-hours",
        type=float,
        default=3.0,
        help="CPU-time cap of one build, enforced with RLIMIT_CPU",
    )
    args = parser.parse_args()
    arms.cpu_only_cuda_calls()
    arms.install_own_cache(data_dir=ML_EXPERIMENTS_DATA_DIR)
    wanted = sorted(lay.ROWS_PER_ARM) if args.arms == "all" else args.arms.split(",")
    assert set(wanted) <= set(lay.ROWS_PER_ARM), wanted
    if args.arms == "all" and args.month not in arms.UX_FOLDS:
        wanted.remove("UX")  # arm UX is fitted on two folds only
        print(
            f"arm UX is not built for fold {args.month}: it is fitted on {arms.UX_FOLDS}",
            flush=True,
        )
    limit_s = int(args.max_cpu_hours * 3600)
    resource.setrlimit(
        resource.RLIMIT_CPU, (limit_s, limit_s + 60)
    )  # the cap is enforced: the kernel stops the build
    build(
        arms_to_build=wanted,
        month=args.month,
        seed=args.seed,
        data_dir=ML_EXPERIMENTS_DATA_DIR,
        limit=args.limit_days,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
