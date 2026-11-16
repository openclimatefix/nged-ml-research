# ruff: noqa: E501
"""Experiment 699: cut every forecast file and bank file at the holdout before any score script reads it.

The forecasts of the 2026-06 fold run to mid-July, and the demand protocol keeps valid times on or
after 2026-07-01 out of every experiment that chooses between training recipes. This script removes
every row whose valid time is on or after 2026-07-01.

- `e00699_filter.py new SUBFOLDER`: cuts, in place, every forecast file (`arms/*/work/fits/*/*.parquet`
  and `backtests/*.parquet`) and every bank file (`P0_D50_*/work/bank/*.parquet`) under
  `roots/SUBFOLDER/`.
- `e00699_filter.py stored`: writes cut copies of experiment 677's stored forecast files (seeds 0 to
  2, monthly refits) and of its stored bank files for the eight folds under `roots/stored/`. Nothing
  under experiment 677's data folder is written.

Every cut asserts that the written file holds no valid time on or after the holdout.
"""

import sys
from pathlib import Path

import polars as pl
from e00699_common import (
    ARMS,
    FOLDS,
    HOLDOUT,
    NEW_ARMS,
    ROOT,
    ROOT_677,
    STORED_SEEDS,
    bank_folder,
)


def write_cut(*, frame: pl.DataFrame, target: Path) -> int:
    """Write `frame` without rows at or after the holdout, atomically, and return the rows removed."""
    kept = frame.filter(pl.col("valid_time") < HOLDOUT)
    assert kept.height > 0, target
    assert kept["valid_time"].max() < HOLDOUT  # ty: ignore[unsupported-operator]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".cut.tmp")
    kept.write_parquet(tmp)
    tmp.rename(target)
    return frame.height - kept.height


def cut_in_place(*, path: Path) -> int:
    """Cut one forecast or bank file in place and return the number of rows removed."""
    return write_cut(frame=pl.read_parquet(path), target=path)


def new_files(*, subfolder: str) -> list[Path]:
    """Return every forecast and bank file under `roots/<subfolder>`."""
    base = ROOT / "roots" / subfolder
    found = sorted(base.glob("arms/*/work/fits/*/*.parquet"))
    found += sorted(base.glob("backtests/*.parquet"))
    found += sorted(base.glob("P0_D50_*/work/bank/*.parquet"))
    return found


def cut_new(*, subfolder: str) -> None:
    """Cut every file of one sub-folder in place."""
    removed = 0
    files = new_files(subfolder=subfolder)
    for path in files:
        removed += cut_in_place(path=path)
    print(f"{subfolder}: {len(files)} files cut in place, {removed:,} rows removed")


def copy_stored() -> None:
    """Write cut copies of experiment 677's stored forecast and bank files."""
    for key in ("none_stored", "s3_stored"):
        arm = ARMS[key]
        source = ROOT_677 / "backtests" / f"{arm.file_stem}.parquet"
        frame = pl.read_parquet(source).filter(pl.col("seed").is_in(list(STORED_SEEDS)))
        removed = write_cut(frame=frame, target=arm.folder / "backtests" / source.name)
        print(f"{arm.label}: forecast file cut, {removed:,} rows removed")
        for month in FOLDS:
            source_dir = ROOT_677 / f"P0_D50_{month}_{arm.step}_sp1" / "work" / "bank"
            for seed in STORED_SEEDS:
                name = f"{month}_d0_s{seed}.parquet"
                write_cut(
                    frame=pl.read_parquet(source_dir / name),
                    target=bank_folder(arm=arm, month=month) / name,
                )
        print(f"{arm.label}: bank files cut for {len(FOLDS)} folds")


def main() -> None:
    """Run the command named on the command line."""
    command = sys.argv[1]
    if command == "stored":
        copy_stored()
    elif command == "new":
        assert sys.argv[2] in {ARMS[k].root for k in NEW_ARMS}, sys.argv[2]
        cut_new(subfolder=sys.argv[2])
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
