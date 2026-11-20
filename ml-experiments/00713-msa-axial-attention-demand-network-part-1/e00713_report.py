# ruff: noqa: E501
"""Experiment 713: print every number that `result.md` quotes, from the stored score and log files.

Usage: e00713_report.py

The script reads `score/*.json` (written by `e00713_score.py` and `e00713_era.py`), the fit logs under
`work/logs/`, and `work/selection.json`, and prints one table per section of the page. It computes no
new score. A table whose input file does not exist yet is reported as missing. The figures of the page
are drawn by `e00713_figures.py` and `e00713_attention.py`; the audit triggers are printed at the end.
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


def sign(x: float) -> int:
    """Return the sign of `x` as -1, 0, or 1."""
    return (x > 0) - (x < 0)


NESTED: Final = ((None, "A"), (None, "K"), ("K", "N"), (None, "N"))
"""(treatment, reference) pairs of the report-only table in which the reference is nested in the treatment."""


def audit_triggers() -> dict[str, str]:  # noqa: PLR0912
    """Return the seven audit triggers of the `ml-experiment` skill as fired / not fired / not computed, with the reason.

    A trigger that needs a number the stored files do not hold is reported as `not computed`, never as
    not fired.
    """
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    out: dict[str, str] = {}
    gate = ROOT / "work" / "known_answers_passed.json"
    screen, headline = read_json(name="screen"), read_json(name="headline")
    reasons = []
    if not gate.exists():
        reasons.append("the known-answer gate file is missing")
    if screen and screen["rules"]["sanity_stop_and_audit"]:
        reasons.append("the sanity rule of the screen fired (M* minus P positive in both groups)")
    if headline:
        reasons += [
            f"{k} H2 interval upper {v:.3f} is not below zero"
            for k, v in headline["h2_upper"].items()
            if v >= 0 and headline["verdict"] == "supported"
        ]
    out["1 control fails"] = (
        ("FIRED: " + "; ".join(reasons))
        if reasons
        else ("not fired" if headline else "not computed")
    )
    if headline:
        out["2 contradicts prediction"] = (
            f"verdict is {headline['verdict']!r}; FIRED unless the write-up records a tested explanation"
            if headline["verdict"] not in {"supported", "partly supported"}
            else "not fired"
        )
    else:
        out["2 contradicts prediction"] = "not computed"
    report = read_json(name="report_only")
    if report:
        flips = []
        for key, v in report.items():
            if key.startswith("crps:"):
                other = report.get("abs_err:" + key[len("crps:") :])
                if other and sign(v["point"]) != sign(other["point"]):
                    flips.append(key[len("crps:") :])
        out["3 CRPS and MAE disagree"] = ("FIRED: " + ", ".join(flips)) if flips else "not fired"
        nested = []
        for key, v in report.items():
            if not key.startswith("crps:"):
                continue
            pair = key.split(":")[1]
            for t, r in NESTED:
                if pair == f"{t or star}-{r}" and v["point"] > 0:
                    nested.append(
                        f"{pair}:{key.split(':')[2]} point {v['point']:.3f}"
                        + (" (clear)" if v["lower"] > 0 else "")
                    )
        out["5 nested arm beats container"] = (
            ("FIRED: " + "; ".join(nested)) if nested else "not fired"
        )
    else:
        out["3 CRPS and MAE disagree"] = out["5 nested arm beats container"] = "not computed"
    out["4 leave-one-out flips sign"] = (
        ("FIRED: " if not headline["guards"]["all_hold"] else "not fired: ")
        + json.dumps(
            {
                g: [
                    headline["guards"][g]["leave_one_primary_out"],
                    headline["guards"][g]["leave_one_month_out"],
                ]
                for g in ("six", "eight")
            }
        )
        if headline and "guards" in headline
        else "not computed"
    )
    out["6 interval width against a sister experiment"] = (
        "not computed: compare by hand with experiment 667's and 709's intervals on the same rows"
    )
    out["7 later design depends on the result"] = (
        "not computed: the maintainer decides, once the verdict is known"
    )
    return out


def median_minus_mean() -> None:
    """Print, per site, the mean of the median forecast minus the mean forecast of the selected arm (a stored `mean` column)."""
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    parts = []
    for month in arms.ALL_FOLDS:
        path = arms.fit_file(data_dir=ML_EXPERIMENTS_DATA_DIR, arm=star, month=month, seed=0)
        if not path.exists():
            print("[missing]", path)
            return
        frame = pl.read_parquet(path)
        if "mean" not in frame.columns:
            print(f"[no mean column] {path}: predict again to add it")
            return
        parts.append(frame.select("time_series_id", (pl.col("q50") - pl.col("mean")).alias("d")))
    table = (
        pl.concat(parts)
        .group_by("time_series_id")
        .agg((pl.col("d").mean() * 100).alias("median_minus_mean_pct"))
        .sort("time_series_id")
    )
    print("\n## Median minus mean forecast, per site (% of capacity), M* seed 0\n", table)


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
    print("\n## Audit triggers")
    for name, state in audit_triggers().items():
        print(f"- {name}: {state}")
    median_minus_mean()
    return 0


if __name__ == "__main__":
    sys.exit(main())
