"""Experiment 719: run experiment 718's `report` command for part 2 (the union of measures).

Usage: e00719_report.py COMMAND   (part 2 is filled in; `self-test` runs the self-test of experiment 718)
"""

import runpy
import sys

import e00719_part1  # noqa: F401  (puts experiment 718's folder on sys.path)

PART_ARGS = {"score": [], "contrasts": ["2"], "figures": [], "report": ["2"]}


def main() -> int:
    """Delegate to experiment 718's script, inserting part 2 where its command line takes a part."""
    args = sys.argv[1:]
    if args and args[0] != "self-test" and "report" in {"contrasts", "report"} and "2" not in args:
        args = (
            [args[0], *PART_ARGS["report"], *args[1:]]
            if "report" == "contrasts"
            else PART_ARGS["report"]
        )
    sys.argv = ["e00718_report.py", *args]
    runpy.run_module("e00718_report", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
