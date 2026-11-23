"""Print the tables of part 2's page from the score JSON files, using part 1's report code."""

import argparse
import sys

import e00716_report as report716  # ty: ignore[unresolved-import]
import e00717_guard as guard

guard.require_part1(report716)


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        assert (
            report716.report(part=2, data_dir=report716.ML_EXPERIMENTS_DATA_DIR / "does-not-exist")
            == 0
        )
        print("selftest passed")
        return 0
    return report716.report(part=2)


if __name__ == "__main__":
    sys.exit(main())
