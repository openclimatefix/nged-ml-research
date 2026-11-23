"""Draw part 2's charts with part 1's builders: the counted contrasts B1 and B2, and B1 by fold."""

import argparse
import json
import sys
from pathlib import Path

import e00716_arms as arms
import e00716_figures as figs
import e00716_score as score


def cmd_contrasts(*, out_dir: Path) -> int:
    """Draw the counted contrasts and the monthly differences of part 2."""
    part = json.loads((score.score_dir_of(part=2) / "part2.json").read_text())
    table = figs.interval_table(
        result=part,
        prefix="counted",
        labels={"first": "B1: HT minus H", "placebo": "B2: HT minus HT-S"},
    )
    figs.save(
        figs.contrast_chart(table=table, title="Part 2: the two counted contrasts at 98.75%"),
        path=out_dir / "contrasts.svg",
    )
    monthly = {g: part[f"counted:first:{g}"]["monthly"] for g in score.GROUPS_COUNTED}
    figs.save(
        figs.fold_chart(monthly=monthly, folds=arms.ALL_FOLDS, title="B1 by fold"),
        path=out_dir / "b1_by_fold.svg",
    )
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        assert callable(figs.contrast_chart)
        assert callable(figs.fold_chart)
        print("selftest passed")
        return 0
    return cmd_contrasts(out_dir=args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
