# ruff: noqa: E501
"""Experiment 715: print the result tables of the stored scores as Markdown, for the write-up.

Usage: e00715_report.py

Reads `score/headline.json` and `score/report_only.json` from the experiment's data folder and prints the
table of N1 and N2 (point, 98.75% interval, the three guards, the group's reading), the verdict, the seed
noise F4 actually has, and the report-only contrasts. It computes nothing new: every number is the
scorer's. A missing file is stated, not skipped.
"""

import json
import sys

import e00715_score as s715


def fmt(x: object) -> str:
    """Return a number to three decimals, or the text of anything else."""
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> int:
    """Print the tables."""
    head = s715.SCORE_DIR / "headline.json"
    if not head.exists():
        print(f"{head} is missing: run `e00715_score.py headline` first")
        return 1
    h = json.loads(head.read_text())
    print("| Contrast | Group | Point | Lower | Upper | t guard | Seed guard | Floor guard |")
    print("|---|---|---|---|---|---|---|---|")
    for name in ("N1", "N2"):
        for g in ("six", "eight"):
            c = h[name][g]
            print(
                f"| {name} | {g} | {fmt(c['point'])} | {fmt(c['lower'])} | {fmt(c['upper'])} | {c['t_guard']} | {c['seed_guard']} | {c['floor_guard']} |"
            )
    print()
    print("Group readings:", h["readings"], "; verdict:", h["verdict"])
    print(
        "Seed noise (monthly standard deviation of seed 0 minus seed 1):",
        json.dumps(h["seed_noise"]),
    )
    ro = s715.SCORE_DIR / "report_only.json"
    if ro.exists():
        r = json.loads(ro.read_text())
        print()
        print("| Report-only contrast | six | eight | other | all 16 |")
        print("|---|---|---|---|---|")
        for name, v in r.items():
            if isinstance(v, dict) and "six" in v:
                print(
                    f"| {name} | "
                    + " | ".join(fmt(v[g]["point"]) for g in ("six", "eight", "other", "all16"))
                    + " |"
                )
    else:
        print(f"{ro} is missing: run `e00715_score.py report-only`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
