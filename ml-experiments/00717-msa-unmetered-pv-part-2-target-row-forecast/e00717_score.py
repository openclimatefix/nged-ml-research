"""Score part 2 of F7 (arms `HT`, `HT-S`, `T`) with part 1's scoring code.

The contrasts are B1 (`HT` minus `H`) and B2 (`HT` minus `HT-S`). The Z rule applies to part 1 only.
Run `run.sh e00717_score.py part2 --n-seeds 5` (or `--n-seeds 2`) and
`run.sh e00717_score.py secondary2 --n-seeds 5`. The scoring code lives in `e00716_score`
(see `EXTRA_PYTHONPATH` in `run.sh`).
"""

import argparse
import sys

import e00713_score as s713  # ty: ignore[unresolved-import]
import e00716_score as score  # ty: ignore[unresolved-import]
import e00717_guard as guard
import numpy as np
import polars as pl

guard.require_part1(score)

PER_SEED_COLUMNS = (*score.KEYS, "seed", "lead_hours", "crps", "abs_err")
"""The only columns that `s713.contrast` reads from a per-seed frame."""


def load_all_lean(
    *,
    names: list[str],
    months: tuple[str, ...],
    seeds: dict[str, tuple[int, ...]],
    truth: np.ndarray | None = None,
) -> tuple[dict, dict, dict]:
    """Return what `e00716_score.load_all` returns, fold by fold, with 7 columns per seed row.

    The values are those of `load_all`: the losses are row-wise, the seed average is per key, and a
    key lives in one fold. The per-seed frames hold only the columns that `s713.contrast` reads.
    """
    truth = s713.truth_stack() if truth is None else truth
    per_seed, averaged = {}, {}
    for name in names:
        seed_parts, avg_parts = [], []
        for month in months:
            raw = score.load_arm(arm=name, seeds=seeds[name], months=(month,))
            seed_parts.append(s713.add_losses(frame=raw, truth=truth).select(PER_SEED_COLUMNS))
            avg_parts.append(s713.add_losses(frame=s713.seed_average(frame=raw), truth=truth))
            del raw
        per_seed[name] = pl.concat(seed_parts)
        averaged[name] = pl.concat(avg_parts).sort(score.KEYS)
        del seed_parts, avg_parts
    s713.assert_same_keys(frames=averaged)
    return per_seed, averaged, s713.group_ids()


score.load_all = load_all_lean


def selftest() -> int:
    """Check that part 2's pairs name the arms of the design."""
    pairs = score.pairs_of_part(part=2, star="M*")
    assert set(pairs) == {"first", "placebo", "control"}, pairs
    names = {n for p in pairs.values() for n in p}
    assert {"HT", "HT-S", "H"} <= names, names
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("part2", "secondary2"))
    parser.add_argument(
        "--n-seeds", type=int, choices=(2, 5), help="the manager's seed set: 5 seeds or 2"
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    assert args.n_seeds is not None, "pass --n-seeds 5 or --n-seeds 2"
    if args.command == "secondary2":
        return score.cmd_secondary(part=2, n_seeds=args.n_seeds)
    return score.cmd_part(part=2, n_seeds=args.n_seeds)


if __name__ == "__main__":
    sys.exit(main())
