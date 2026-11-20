# ruff: noqa: E501
"""Experiment 713: print every number that `result.md` quotes, from the stored score and log files.

Usage: e00713_report.py

The script reads `score/*.json` (written by `e00713_score.py` and `e00713_era.py`), the fit logs under
`work/logs/`, and `work/selection.json`, and prints one table per section of the page. It computes no
new score. A table whose input file does not exist yet is reported as missing. The figures of the page
(example forecasts, the read-out weights by age and day type, the event plots, and the load-shift
plots) are not drawn here; the implementation note lists them as open work.
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00713_arms as arms
import e00713_model as mdl
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT: Final[Path] = arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)


def read_json(*, name: str) -> dict | None:
    """Return a score file, or None with a message when it does not exist."""
    path = ROOT / "score" / f"{name}.json"
    if not path.exists():
        print(f"[missing] {path}")
        return None
    return json.loads(path.read_text())


def contrast_table(*, title: str, results: dict) -> None:
    """Print contrasts: point, interval, t interval, seeds with the same sign, and the three guards."""
    print(f"\n## {title}")
    rows = [
        {
            "contrast": key, "point": round(v["point"], 3), "lower": round(v["lower"], 3),
            "upper": round(v["upper"], 3), "t_lower": round(v["t_lower"], 3), "t_upper": round(v["t_upper"], 3),
            "seeds_same_sign": f"{v['seeds_same_sign']}/{v['n_seeds']}", "t_guard": v["t_guard"],
            "floor_guard": v["floor_guard"], "level": v["level"],
        }
        for key, v in results.items()
        if isinstance(v, dict) and "point" in v
    ]  # fmt: skip
    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        print(pl.DataFrame(rows))


def training_log_table() -> None:
    """Print, per arm, the mean stopping pass per stage and the count of reruns at the 120-pass cap."""
    rows = []
    for path in sorted((ROOT / "work" / "logs").glob("*/*_fit.json")):
        log = json.loads(path.read_text())
        for stage in log["stages"]:
            rows.append(  # noqa: PERF401
                {
                    "arm": log["arm"],
                    "stage": stage["stage"],
                    "passes": stage["passes"],
                    "cap_hit": stage["cap_hit"],
                    "rerun": stage["rerun"] is not None,
                }
            )
    print("\n## Training log: mean stopping pass per stage, reruns at the 120-pass cap")
    if not rows:
        print("[missing] no fit logs")
        return
    frame = pl.DataFrame(rows).group_by("arm", "stage").agg(
        pl.col("passes").mean().round(1).alias("mean_passes"), pl.col("rerun").sum().alias("reruns"),
        pl.col("cap_hit").sum().alias("still_at_cap"), pl.len().alias("fits"),
    ).sort("arm", "stage")  # fmt: skip
    with pl.Config(tbl_rows=100):
        print(frame)


def parameter_table() -> None:
    """Print the parameter count of each arm from the built module."""
    print("\n## Parameter counts (built modules)")
    for letter in ("M", "A", "K", "N", "E", "G"):
        print(letter, mdl.parameter_count(model=mdl.MsaNet(switches=mdl.ARM_SWITCHES[letter])))


def main() -> int:
    """Print all tables."""
    parameter_table()
    sel = ROOT / "work" / "selection.json"
    print("\n## Choice of M*:", sel.read_text() if sel.exists() else "[missing]")
    training_log_table()
    for name, title in (
        (
            "candidate",
            "KA8: the stored Candidate (CRPS, pooled / mean of months / seed diff mean / sd / half-hours)",
        ),
        ("screen", "Screen: sanity rule and stop rule (interim)"),
        ("headline", "Headline: H1 and H2 at 98.75%"),
        ("report_only", "Report-only readings at 95%"),
        ("era_r1", "Secondary R1: D8 alone"),
        ("era_r2_r3", "Secondary R2 and R3"),
    ):
        data = read_json(name=name)
        if data is None:
            continue
        if name in {"candidate", "era_r2_r3"}:
            print(f"\n## {title}\n{json.dumps(data, indent=1)}")
        else:
            contrast_table(title=title, results=data)
            for extra in ("readings", "h2_upper", "verdict", "rules", "rules_without_2025_11"):
                if extra in data:
                    print(extra, data[extra])
    return 0


if __name__ == "__main__":
    sys.exit(main())
