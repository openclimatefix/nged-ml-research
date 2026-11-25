# ruff: noqa: E501
"""Experiment 717: the S1 secondary reading, `A1` on the sunny folds' solar block, INTERIM (CPU only).

Usage: run.sh e00717_s1_read.py --smoke FOLD   # load one fold at seed 0, print row counts
       run.sh e00717_s1_read.py read           # the reading at the seeds that exist (2 seeds)
       run.sh e00717_s1_read.py --selftest

S1 is one of 14 secondary readings. It is report-only, read at 95%, and changes no verdict. It reads
`A1` (`H` minus `M*`, treatment minus reference in CRPS points of capacity, negative favours `H`)
on the solar block of the folds 2026-04 to 2026-06, for the six signed-MW Primaries and for the
eight MVA-only Primaries separately. The reviewed `secondary_s1_s2` of `e00716_score`
decides whether S1 holds (the point is at most -0.05 and the interval lies below zero), on every
stored lead, as the final S1 does. The command also uses `load_all` and `pinned_seeds` without
change, and writes `score/s1_interim.json` of part 2. It uses one thread and no GPU. Memory: 12
forecast files (2 arms, 3 folds, 2 seeds), an estimated 2 to 3 GB; the 12 GB default of `run.sh` is
ample.
"""

import argparse
import sys
from datetime import UTC, datetime

import e00716_arms as arms  # ty: ignore[unresolved-import]
import e00716_score as score  # ty: ignore[unresolved-import]
import e00717_guard as guard
import polars as pl
from e00713_arms import m_star  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

guard.require_part1(score, arms)

NOTE = "S1 is one of 14 secondaries and changes no verdict"
"""The label printed and stored beside INTERIM and the 95% level."""


def read(*, n_seeds: int = 2) -> dict:
    """Return A1 on the solar block of the sunny folds per counted group at 95%."""
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "H"]
    seeds = score.pinned_seeds(n_seeds=n_seeds, names=names, months=score.SUNNY_FOLDS)
    per_seed, averaged, groups = score.load_all(
        names=names, months=score.SUNNY_FOLDS, seeds=dict.fromkeys(names, seeds)
    )
    res = score.secondary_s1_s2(
        per_seed=per_seed,
        averaged=averaged,
        groups=groups,
        pairs={"A1": ("H", star)},
        n_seeds=n_seeds,
    )
    out: dict = {
        "interim": True, "report_only": True, "note": NOTE, "reading": "S1 (A1)", "contrast": "H minus M*",
        "level": score.LEVEL_SECONDARY, "band": "every stored lead (band=None, as the final S1)",
        "folds": list(score.SUNNY_FOLDS), "seeds": list(seeds),
    }  # fmt: skip
    for group in score.GROUPS_COUNTED:
        c = res[f"A1:{group}"]
        out[group] = {k: c[k] for k in ("point", "lower", "upper", "holds")}
    return out


def smoke(*, fold: str) -> int:
    """Load one fold at seed 0 for `M*` and `H`, apply the solar-block slice, and print row counts only."""
    assert fold in score.SUNNY_FOLDS, f"{fold} is not a sunny fold {score.SUNNY_FOLDS}"
    star = m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    names = [star, "H"]
    _, averaged, groups = score.load_all(
        names=names, months=(fold,), seeds=dict.fromkeys(names, (0,))
    )
    sliced = score.apply_slice(averaged, solar=True, folds=(fold,))
    for name in names:
        print(
            f"smoke {name} {fold} seed 0: {averaged[name].height} rows, {sliced[name].height} in the solar block"
        )
    assert sliced[names[0]].height == sliced[names[1]].height, "the slice changed the alignment"
    print(f"smoke: {len(groups)} Primary groups")
    return 0


def check_holds_rule() -> None:
    """Call `secondary_s1_s2` on tiny synthetic frames with a stubbed `contrast`, to catch a reversed rule."""
    stamp = datetime(2026, 5, 3, 9, 0, tzinfo=UTC)
    frame = pl.DataFrame({"power_fcst_init_time": [stamp], "valid_time": [stamp]})
    frames = {"H": frame, "M*": frame}
    cases = [  # (point, upper, must hold)
        (-0.06, -0.01, True),
        (-0.05, -0.001, True),
        (-0.04, -0.01, False),
        (-0.06, 0.01, False),
        (0.06, 0.08, False),
    ]
    real = score.contrast
    try:
        for point, upper, expected in cases:
            score.contrast = lambda *, p=point, u=upper, **_kw: {
                "point": p,
                "lower": p - 0.1,
                "upper": u,
            }  # ty: ignore[invalid-assignment]
            out = score.secondary_s1_s2(
                per_seed=frames, averaged=frames, groups={}, pairs={"A1": ("H", "M*")}, n_seeds=2
            )
            for group in score.GROUPS_COUNTED:
                assert out[f"A1:{group}"]["holds"] is expected, (point, upper, expected)
    finally:
        score.contrast = real


def selftest() -> int:
    """Check the constants S1 reads, with no data."""
    assert set(score.SUNNY_FOLDS) == {"2026-04", "2026-05", "2026-06"}
    assert tuple(score.SOLAR_BLOCK_MINUTES) == (360, 870), score.SOLAR_BLOCK_MINUTES
    assert score.LEVEL_SECONDARY == 95.0
    check_holds_rule()
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
    score.write(name="s1_interim", obj=out, score_dir=score.score_dir_of(part=2))
    for group in score.GROUPS_COUNTED:
        c = out[group]
        print(
            f"INTERIM report-only S1 ({NOTE}; 95%) A1, {group}: point {c['point']:+.4f}, interval {c['lower']:+.4f} to {c['upper']:+.4f}, holds {c['holds']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
