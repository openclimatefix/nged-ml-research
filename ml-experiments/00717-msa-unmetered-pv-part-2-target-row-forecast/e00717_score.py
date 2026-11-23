"""Score part 2 of F7 (arms `HT`, `HT-S`, `T`) with part 1's scoring code.

The contrasts are B1 (`HT` minus `H`) and B2 (`HT` minus `HT-S`). The Z rule applies to part 1 only.
Run `run.sh e00717_score.py part2`. The scoring code lives in `e00716_score` (see `EXTRA_PYTHONPATH`
in `run.sh`).
"""

import argparse
import sys

import e00716_score as score


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
    parser.add_argument("command", nargs="?", choices=("part2",))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    return selftest() if args.selftest else score.cmd_part(part=2)


if __name__ == "__main__":
    sys.exit(main())
