# ruff: noqa: E501
"""Experiment 717: the S1 secondary reading, `A1` on the solar block of the sunny folds, INTERIM (CPU only).

Usage: run.sh e00717_s1_read.py --smoke FOLD   # load one fold at seed 0 and print row counts; no contrast
       run.sh e00717_s1_read.py read           # the reading at the seeds that exist (2 seeds)
       run.sh e00717_s1_read.py --selftest

S1 is one of 14 secondary readings. It is report-only, read at 95%, and changes no verdict. It reads
`A1` (`H` minus `M*`, treatment minus reference in CRPS points of capacity, negative favours `H`)
on the solar block of the folds 2026-04 to 2026-06, for the six signed-MW Primaries and for the eight
MVA-only Primaries separately. S1 holds in a group when the point is at most -0.05 and the interval
lies below zero. The command uses the reviewed `apply_slice`, `load_all`, `pinned_seeds`, and
`contrast` of `e00716_score` without change, and writes `score/s1_interim.json` of part 1. It
uses one thread and no GPU.
"""

import argparse
import sys

import e00716_arms as arms  # ty: ignore[unresolved-import]
import e00716_score as score  # ty: ignore[unresolved-import]
import e00717_guard as guard
from e00713_arms import m_star  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

guard.require_part1(score, arms)

S1_POINT_LIMIT = -0.05
"""S1 holds when the point is at most this value (design section 7.5)."""


def holds(*, point: float, upper: float) -> bool:
    """Return whether S1 holds in a group: the point is at most -0.05 and the interval is below zero."""
    return point <= S1_POINT_LIMIT and upper < 0


def read(*, n_seeds: int = 2) -> dict:
    """Return A1 on the solar block of the sunny folds per counted group at 95%."""
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "H"]
    seeds = score.pinned_seeds(n_seeds=n_seeds, names=names, months=score.SUNNY_FOLDS)
    per_seed, averaged, groups = score.load_all(
        names=names, months=score.SUNNY_FOLDS, seeds=dict.fromkeys(names, seeds)
    )
    ps = score.apply_slice(per_seed, solar=True, folds=score.SUNNY_FOLDS)
    av = score.apply_slice(averaged, solar=True, folds=score.SUNNY_FOLDS)
    out: dict = {"interim": True, "reading": "S1 (A1)", "level": score.LEVEL_SECONDARY, "folds": list(score.SUNNY_FOLDS), "seeds": list(seeds), "contrast": "H minus M*"}
    for group in score.GROUPS_COUNTED:
        c = score.contrast(
            per_seed=ps, averaged=av, treatment="H", reference=star, group=group, groups=groups,
            level=score.LEVEL_SECONDARY, n_seeds=n_seeds,
        )  # fmt: skip
        out[group] = {"point": c["point"], "lower": c["lower"], "upper": c["upper"], "holds": holds(point=c["point"], upper=c["upper"])}
    return out


def smoke(*, fold: str) -> int:
    """Load one fold at seed 0 for `M*` and `H`, apply the solar-block slice, and print row counts only."""
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "H"]
    _, averaged, groups = score.load_all(names=names, months=(fold,), seeds=dict.fromkeys(names, (0,)))
    sliced = score.apply_slice(averaged, solar=True, folds=(fold,))
    for name in names:
        print(f"smoke {name} {fold} seed 0: {averaged[name].height} rows, {sliced[name].height} in the solar block")
    assert sliced[names[0]].height == sliced[names[1]].height, "the slice changed the alignment"
    print(f"smoke: {len(groups)} Primary groups")
    return 0


def selftest() -> int:
    """Check the rule that decides whether S1 holds, with no data."""
    assert holds(point=-0.06, upper=-0.01)
    assert not holds(point=-0.04, upper=-0.01)
    assert not holds(point=-0.06, upper=0.01)
    assert set(score.SUNNY_FOLDS) == {"2026-04", "2026-05", "2026-06"}
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("read",))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--smoke", metavar="FOLD")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        return smoke(fold=args.smoke)
    assert args.command == "read", "pass read, --smoke FOLD, or --selftest"
    out = read()
    score.write(name="s1_interim", obj=out, score_dir=score.score_dir_of(part=1))
    for group in score.GROUPS_COUNTED:
        c = out[group]
        print(f"INTERIM S1 (one of 14 secondaries, 95%) A1, {group}: point {c['point']:+.4f}, interval {c['lower']:+.4f} to {c['upper']:+.4f}, holds {c['holds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
