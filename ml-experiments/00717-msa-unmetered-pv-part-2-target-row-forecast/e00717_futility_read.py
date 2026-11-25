# ruff: noqa: E501
"""Experiment 717: read the counted contrasts of F7 at 2 seeds, for the futility rule (CPU only).

Usage: run.sh e00717_futility_read.py part1    # A1 (`H` minus `M*`) and A2 (`H` minus `H-S`)
       run.sh e00717_futility_read.py part2    # B1 (`HT` minus `H`) and B2 (`HT` minus `HT-S`)
       run.sh e00717_futility_read.py --selftest

The staging rule on the plan pages opens seeds 2 to 4 only if the point estimate of a counted
treatment contrast is below zero in at least one group. This command computes those point estimates
and their 98.75% intervals, as `e00716_score.cmd_part` does for the verdict, but for the two seeds
that exist, on all 8 folds, and for the two counted contrasts of one part. It calls the reviewed
functions of `e00716_score` (`pinned_seeds`, `load_all`, `contrast`) without change, so it reads no
arm that a later phase writes (no read-out, no `H-X`, no `T`). It prints each contrast's point,
interval, and per-fold points, and writes `score/futility_part<N>.json` of the part's data folder.
Every number is interim. A contrast is treatment minus reference in CRPS points of capacity, so a
negative point estimate favours the treatment.

Memory: the command loads three arms x 8 folds x 2 seeds = 48 forecast files at one time, about 6
to 9 GB (an estimate from the screen run, which held 16 files under a 12 GB cap). Run it under
`run.sh` with `SLOT_MEM=14G`, which is the cap `run.sh` passes to `systemd-run` with no swap. It
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

LABELS = {1: {"first": "A1", "placebo": "A2"}, 2: {"first": "B1", "placebo": "B2"}}
"""The name of the counted treatment contrast and of the placebo contrast in each part."""


def read_part(*, part: int) -> dict:
    """Return the points, intervals, and per-fold points of a part's two counted contrasts."""
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    pairs = score.pairs_of_part(part=part, star=star)
    contrasts = {k: pairs[k] for k in ("first", "placebo")}
    names = sorted({n for pair in contrasts.values() for n in pair})
    seeds = score.pinned_seeds(n_seeds=2, names=names, months=arms.ALL_FOLDS)
    per_seed, averaged, groups = score.load_all(
        names=names, months=arms.ALL_FOLDS, seeds=dict.fromkeys(names, seeds)
    )
    out: dict = {"interim": True, "part": part, "folds": list(arms.ALL_FOLDS), "seeds": list(seeds)}
    for key, (treatment, reference) in contrasts.items():
        label = LABELS[part][key]
        out[label] = {"treatment": treatment, "reference": reference}
        for group in score.GROUPS_COUNTED:
            c = score.contrast(
                per_seed=per_seed,
                averaged=averaged,
                treatment=treatment,
                reference=reference,
                group=group,
                groups=groups,
                level=score.LEVEL_COUNTED,
                n_seeds=2,
            )
            out[label][group] = {
                "point": c["point"],
                "lower": c["lower"],
                "upper": c["upper"],
                "monthly": [float(x) for x in c["monthly"]],
                "level": score.LEVEL_COUNTED,
            }
    return out


def futility_opens_seeds(*, out: dict, part: int) -> bool:
    """Return whether the counted treatment contrast (A1 or B1) is below zero in at least one group.

    A2 and B2 are the placebo contrasts and do not count. The staging rule on the plan pages opens
    seeds 2 to 4 only when this holds for some counted treatment contrast of either part.
    """
    label = LABELS[part]["first"]
    return any(out[label][group]["point"] < 0 for group in score.GROUPS_COUNTED)


def smoke(*, fold: str) -> int:
    """Load one fold at seed 0 for the arms of both parts and print the row counts; no contrast, no score."""
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "H", "H-S"]
    _, averaged, groups = score.load_all(
        names=names, months=(fold,), seeds=dict.fromkeys(names, (0,))
    )
    for name in names:
        print(f"smoke {name} {fold} seed 0: {averaged[name].height} rows")
    print(f"smoke: keys equal across the arms, {len(groups)} Primary groups")
    return 0


def selftest() -> int:
    """Check the pairs and the labels, with no data."""
    for part, labels in LABELS.items():
        pairs = score.pairs_of_part(part=part, star="M*")
        assert set(labels) == {"first", "placebo"} <= set(pairs), (part, pairs)
    assert score.pairs_of_part(part=1, star="M*")["first"] == ("H", "M*")
    assert score.pairs_of_part(part=1, star="M*")["placebo"] == ("H", "H-S")
    assert score.pairs_of_part(part=2, star="M*")["first"] == ("HT", "H")
    assert score.pairs_of_part(part=2, star="M*")["placebo"] == ("HT", "HT-S")
    assert len(arms.ALL_FOLDS) == 8
    fake = {"A1": {"six": {"point": 0.01}, "eight": {"point": 0.02}}}
    assert futility_opens_seeds(out=fake, part=1) is False
    fake["A1"]["eight"]["point"] = -0.001
    assert futility_opens_seeds(out=fake, part=1) is True
    assert futility_opens_seeds(
        out={"B1": {"six": {"point": -1.0}, "eight": {"point": 1.0}}}, part=2
    )
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("part1", "part2"))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--smoke", metavar="FOLD", help="load one fold at seed 0 and print row counts only"
    )
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.smoke:
        return smoke(fold=args.smoke)
    assert args.command is not None, "pass part1, part2, or --selftest"
    part = 1 if args.command == "part1" else 2
    out = read_part(part=part)
    out["futility_opens_seeds"] = futility_opens_seeds(out=out, part=part)
    score.write(name=f"futility_part{part}", obj=out, score_dir=score.score_dir_of(part=part))
    print(
        f"INTERIM part {part}: {LABELS[part]['first']} point is below zero in at least one group: {out['futility_opens_seeds']}"
    )
    for label in LABELS[part].values():
        for group in score.GROUPS_COUNTED:
            c = out[label][group]
            print(
                f"INTERIM {label} {out[label]['treatment']} minus {out[label]['reference']}, {group}: "
                f"point {c['point']:+.4f}, {c['level']}% interval {c['lower']:+.4f} to {c['upper']:+.4f}, "
                f"by fold {[round(x, 3) for x in c['monthly']]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
