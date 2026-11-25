"""Check that `load_all_lean` of `e00717_score` returns the values of `e00716_score.load_all`.

Read-only. For one fold and seeds 0 and 1 of one arm, the script calls both loaders and prints,
per column that `s713.contrast` reads, the largest absolute difference, the row counts, and
whether the row order is identical, for the per-seed frames and for the seed-averaged frames.
It writes no file.
Run `run.sh e00717_check_lean_loader.py [--arm HT] [--fold 2026-05]`.
Expected: a largest difference of at most 1e-12 and identical row order in every table.
"""

import argparse
import importlib
import sys

import e00716_score as score  # ty: ignore[unresolved-import]
import e00717_guard as guard
import numpy as np
import polars as pl

guard.require_part1(score)

SEEDS: tuple[int, ...] = (0, 1)
READ_BY_CONTRAST = (*score.KEYS, "seed", "lead_hours", "crps", "abs_err")
"""The columns of a per-seed frame that `s713.contrast` reads."""


def largest_difference(*, old: pl.Series, new: pl.Series) -> float:
    """Return the largest absolute difference of two aligned columns (timestamps as integers)."""
    if old.dtype.is_temporal():
        old, new = old.dt.epoch(time_unit="us"), new.dt.epoch(time_unit="us")
    return float(np.max(np.abs(old.cast(pl.Float64).to_numpy() - new.cast(pl.Float64).to_numpy())))


def compare(*, title: str, old: pl.DataFrame, new: pl.DataFrame, columns: tuple[str, ...]) -> bool:
    """Print one table's comparison and return whether it passes."""
    same_rows = old.height == new.height
    key_columns = [c for c in (*score.KEYS, "seed") if c in old.columns and c in new.columns]
    same_order = same_rows and bool(old.select(key_columns).equals(new.select(key_columns)))
    print(
        f"\n## {title}: rows old {old.height}, new {new.height}, row order identical {same_order}"
    )
    worst = 0.0
    if same_rows:
        for column in columns:
            diff = largest_difference(old=old[column], new=new[column])
            worst = max(worst, diff)
            print(f"{column:>22}: largest absolute difference {diff:.3e}")
    ok = same_rows and same_order and worst <= 1e-12
    print(f"{title}: {'PASS' if ok else 'FAIL'} (largest {worst:.3e})")
    return ok


def main() -> int:
    """Load one arm with both loaders and compare them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="HT")
    parser.add_argument("--fold", default="2026-05")
    args = parser.parse_args()
    old_loader = score.load_all
    assert old_loader.__module__ == "e00716_score", "the original loader is already replaced"
    lean_loader = importlib.import_module("e00717_score").load_all_lean
    names, months = [args.arm], (args.fold,)
    seeds: dict[str, tuple[int, ...]] = {args.arm: SEEDS}
    old_per_seed, old_averaged, old_groups = old_loader(names=names, months=months, seeds=seeds)
    new_per_seed, new_averaged, new_groups = lean_loader(names=names, months=months, seeds=seeds)
    ok = old_groups == new_groups
    print(f"Primary groups identical: {ok}")
    ok &= compare(
        title="per-seed frame",
        old=old_per_seed[args.arm],
        new=new_per_seed[args.arm],
        columns=READ_BY_CONTRAST,
    )
    ok &= compare(
        title="seed-averaged frame",
        old=old_averaged[args.arm],
        new=new_averaged[args.arm],
        columns=tuple(c for c in old_averaged[args.arm].columns if c != "seed"),
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
