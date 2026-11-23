# ruff: noqa: E501, PERF401
"""Experiment 716: print every number that `result.md` quotes, from the stored score and log files (WP7).

Usage: run.sh e00716_report.py [--part 1]
       run.sh e00716_report.py --selftest

The script reads `score/*.json` (written by `e00716_score.py` and `e00716_readouts.py`), the fit logs
under `work/logs/`, and the known-answer files, and prints one table per section of the page. It
computes no new score. A table whose input does not exist yet is reported as missing. Part 2's
`e00717_report.py` calls the same functions with `part=2`.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import e00716_arms as arms
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR


def read_json(*, path: Path) -> dict | None:
    """Return a JSON file, or None with a message when it does not exist."""
    if not path.exists():
        print(f"[missing] {path}")
        return None
    return json.loads(path.read_text())


def contrast_rows(*, results: dict, prefix: str) -> list[dict]:
    """Return one table row per contrast under `prefix`: point, interval, t interval, seeds, and the three guards."""
    rows = []
    for key, v in results.items():
        if key.startswith(prefix) and isinstance(v, dict) and "point" in v:
            rows.append({
                "contrast": key, "point": round(v["point"], 3), "lower": round(v["lower"], 3), "upper": round(v["upper"], 3),
                "t_lower": round(v["t_lower"], 3), "t_upper": round(v["t_upper"], 3), "seeds_same_sign": f"{v['seeds_same_sign']}/{v['n_seeds']}",
                "t_guard": v["t_guard"], "seed_guard": v["seed_guard"], "floor_guard": v["floor_guard"], "level": v["level"],
            })  # fmt: skip
    return rows


def show(*, title: str, rows: list[dict]) -> None:
    """Print a titled table."""
    print(f"\n## {title}")
    with pl.Config(tbl_rows=200, tbl_cols=30, fmt_str_lengths=70):
        print(pl.DataFrame(rows) if rows else "(no rows)")


def training_log_rows(*, part: int, data_dir: Path) -> list[dict]:
    """Return, per arm and stage, the mean stopping pass and the number of reruns at the 120-pass cap."""
    rows = []
    for path in sorted(
        (arms.root(data_dir=data_dir, arm="Z" if part == 1 else "T") / "work" / "logs").glob(
            "*/*_fit.json"
        )
    ):
        log = json.loads(path.read_text())
        for stage in log["stages"]:
            rows.append({"arm": log["arm"], "month": log["month"], "seed": log["seed"], "stage": stage["stage"], "passes": stage["passes"], "best_val": round(stage["best_val"], 4), "cap_hit": stage["cap_hit"], "reran": stage["rerun"] is not None})  # fmt: skip
    return rows


def report(*, part: int, data_dir: Path = ML_EXPERIMENTS_DATA_DIR) -> int:
    """Print every table of a part's page."""
    folder = arms.root(data_dir=data_dir, arm="Z" if part == 1 else "T")
    res = read_json(path=folder / "score" / f"part{part}.json")
    if res is not None:
        print(f"\n# Part {part}: verdict {res['verdict']!r}; closes the line: {res['closes_line']}")
        print(
            "readings:",
            res["readings"],
            "| placebo against its control:",
            res["placebo_control_readings"],
            "| harmful placebo:",
            res["harmful_placebo"],
        )
        show(title="Counted contrasts at 98.75%", rows=contrast_rows(results=res, prefix="counted"))
        show(
            title="Sensitivity at 99.375% (never changes a verdict)",
            rows=contrast_rows(results=res, prefix="sensitivity"),
        )
        print("\n## Guards of `supported`")
        print(json.dumps(res["guards"], indent=1, default=float))
        show(
            title="Secondary readings S1 and S2 (95%)",
            rows=contrast_rows(results=res["secondary_s1_s2"], prefix=""),
        )
        show(
            title="Secondary reading S4 (95%)",
            rows=contrast_rows(results=res["secondary_s4"], prefix=""),
        )
        show(
            title="Report-only cuts of the first contrast (95%)",
            rows=contrast_rows(results=res["report_only_first"], prefix=""),
        )
        print("\nby Primary:", res["report_only_first"].get("by_primary"))
        print(res["secondary_count_note"])
    if part == 1:
        z = read_json(path=folder / "score" / "zrule.json")
        if z is not None:
            print("\n## The rule on Z:", json.dumps(z, default=float))
        base = read_json(path=folder / "score" / "baseline.json")
        if base is not None:
            print("\n## Baseline (M* against C, 8 folds):", json.dumps(base, default=float)[:2000])
    show(title="Fit logs", rows=training_log_rows(part=part, data_dir=data_dir))
    gate = read_json(path=arms.gate_file(data_dir=data_dir))
    if gate is not None:
        print(
            "\n## Known answers:",
            json.dumps(
                {k: v for k, v in gate.items() if k.startswith(("KF", "early", "failed"))},
                default=float,
            )[:3000],
        )
    rung = read_json(path=arms.root(data_dir=data_dir, arm="Z") / "work" / "world_rung.json")
    if rung is not None:
        print(
            "\n## World rung:",
            rung["rung"],
            {
                k: {g: round(v["gaps"][g]["gap"], 4) for g in v["gaps"]}
                for k, v in rung["results"].items()
            },
        )
    return 0


def selftest() -> int:
    """Write a tiny fake score file and check that the report reads it."""
    base = Path(tempfile.mkdtemp(prefix="f7rep_", dir=Path.home()))
    try:
        folder = arms.root(data_dir=base, arm="Z")
        (folder / "score").mkdir(parents=True)
        c = {
            "point": -0.03,
            "lower": -0.06,
            "upper": 0.0,
            "t_lower": -0.07,
            "t_upper": 0.01,
            "seeds_same_sign": 5,
            "n_seeds": 5,
            "t_guard": False,
            "seed_guard": True,
            "floor_guard": False,
            "level": 98.75,
        }
        fake = {
            "verdict": "inconclusive", "closes_line": False, "readings": {"six": "cannot tell", "eight": "cannot tell"}, "placebo_control_readings": {"six": "cannot tell", "eight": "cannot tell"},
            "harmful_placebo": False, "counted:first:six": c, "sensitivity_99_375:first:six": c, "guards": {"all_hold": False}, "secondary_s1_s2": {"S1:six": c}, "secondary_s4": {"S4:six": c},
            "report_only_first": {"mae:six": c, "by_primary": {"5": 0.01}}, "secondary_count_note": "14 secondary readings",
        }  # fmt: skip
        (folder / "score" / "part1.json").write_text(json.dumps(fake))
        assert report(part=1, data_dir=base) == 0
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    return selftest() if args.selftest else report(part=args.part)


if __name__ == "__main__":
    sys.exit(main())
