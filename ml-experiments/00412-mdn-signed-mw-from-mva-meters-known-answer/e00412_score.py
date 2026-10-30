# ruff: noqa: E501
"""Score experiment 412's first test: the predicted sign against the known signed truth.

For each arm, setting, and hidden Primary it reports, on the headline slice (the real scaled CAMS
unit shape at least 0.3 and the CAMS clear-sky index at least 0.7):

- balanced accuracy, recall on reverse flow, and the false-reverse rate of the predicted sign
  (`P(X < 0) > 0.5`), where reverse flow is a negative signed truth;
- the same three numbers for the threshold rule (reverse when the unit shape exceeds a threshold
  fitted on the refit's training window) and for the hour-and-month baseline (reverse at the UTC
  summer-midday hours of April to September);
- the false-reverse rate on sunny summer-midday half-hours with no reverse flow (P4's number), and
  the mean pinball loss of the 11 signed quantiles over all test rows, in points of capacity.

All numbers from the first test are interim: two test months, one seed, no intervals.

```bash
PYTHONPATH=... uv run --no-sync python ml-experiments/00412-*/e00412_score.py first
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00412_inputs as inputs
import e00412_models as models
import numpy as np
import polars as pl

OUT_DIR: Final[Path] = inputs.WORK_DIR
SUNNY_K: Final[float] = inputs.SUNNY_K
THRESHOLDS: Final[np.ndarray] = np.arange(0.05, 1.0, 0.025)
LEVEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{int(100 * q):02d}" for q in models.LEVELS)


def balanced(*, truth_reverse: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return balanced accuracy, recall on reverse flow, and the false-reverse rate."""
    recall = float(predicted[truth_reverse].mean()) if truth_reverse.any() else float("nan")
    false = float(predicted[~truth_reverse].mean()) if (~truth_reverse).any() else float("nan")
    return {
        "balanced_accuracy": (recall + (1.0 - false)) / 2.0,
        "recall": recall,
        "false_reverse": false,
        "n_reverse": int(truth_reverse.sum()),
        "n_other": int((~truth_reverse).sum()),
    }


def threshold_fit(*, site: str, setting: inputs.Setting, month: str) -> float:
    """Return the unit-shape threshold with the best balanced accuracy on the refit's training window.

    Daylight half-hours (unit shape above 0.01) inside the 730-day window before the refit, with the
    signed truth of the setting; the truth is known only because the lab hid it.
    """
    refit = base.refit_of(month=month)
    arr = inputs.site_arrays(site=site, fold=True, setting=setting)
    lo = int((refit.timestamp() - base.WINDOW_DAYS * 86_400) * 1_000_000)
    hi = int((refit - regimes.REFIT_CUTOFF_AGE).timestamp() * 1_000_000)
    sel = (arr.valid > lo) & (arr.valid <= hi) & np.isfinite(arr.y_truth)
    valid, truth = arr.valid[sel], arr.y_truth[sel]
    _, first = np.unique(valid, return_index=True)
    valid, truth = valid[first], truth[first]
    u, _k = inputs.cams_for_rows(site=site, valid_us=valid)
    day = np.nan_to_num(u) > inputs.DAYLIGHT_U
    reverse = truth[day] < 0
    best, best_score = 0.5, -1.0
    for t in THRESHOLDS:
        score = balanced(truth_reverse=reverse, predicted=u[day] > t)["balanced_accuracy"]
        if score > best_score:
            best, best_score = float(t), score
    return best


def pinball(*, frame: pl.DataFrame) -> float:
    """Return the mean pinball loss of the 11 signed quantiles against the signed truth."""
    truth = frame["truth_points"].to_numpy()[:, None]
    q = frame.select(LEVEL_COLUMNS).to_numpy()
    tau = np.array(models.LEVELS)[None, :]
    diff = truth - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def score_file(*, path: Path, setting: inputs.Setting, month: str) -> list[dict]:
    """Score one forecast file, one row per hidden Primary and one pooled row."""
    frame = pl.read_parquet(path).filter(pl.col("truth_points").is_finite())
    out = []
    pooled = {"truth": [], "pred": [], "rule": [], "hm": []}
    for site in frame["site"].unique(maintain_order=True).to_list():
        part = frame.filter(pl.col("site") == site)
        u_gate, k = part["u_gate"].to_numpy(), part["k"].to_numpy()
        gate = (u_gate >= models.GATE_U) & (k >= SUNNY_K)
        truth = part["truth_points"].to_numpy() < 0.0
        pred = part["p_negative"].to_numpy() > 0.5
        threshold = threshold_fit(site=site, setting=setting, month=month)
        rule = u_gate > threshold
        midday = inputs.is_midday(valid_us=part["valid"].to_numpy())
        row = {
            "site": site,
            "month": month,
            "n_slice": int(gate.sum()),
            "model": balanced(truth_reverse=truth[gate], predicted=pred[gate]),
            "threshold_rule": {
                **balanced(truth_reverse=truth[gate], predicted=rule[gate]),
                "threshold": threshold,
            },
            "hour_month_baseline": balanced(truth_reverse=truth[gate], predicted=midday[gate]),
            "false_reverse_sunny_midday": float(pred[gate & midday & ~truth].mean())
            if (gate & midday & ~truth).any()
            else float("nan"),
            "pinball_points": pinball(frame=part),
        }
        out.append(row)
        pooled["truth"].append(truth[gate])
        pooled["pred"].append(pred[gate])
        pooled["rule"].append(rule[gate])
        pooled["hm"].append(midday[gate])
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    out.append(
        {
            "site": "pooled",
            "month": month,
            "model": balanced(truth_reverse=cat["truth"], predicted=cat["pred"]),
            "threshold_rule": balanced(truth_reverse=cat["truth"], predicted=cat["rule"]),
            "hour_month_baseline": balanced(truth_reverse=cat["truth"], predicted=cat["hm"]),
        }
    )
    return out


def parse_name(*, name: str) -> tuple[inputs.Setting, str, int]:
    """Parse `L2_Q2_2026-03_s0` into the setting, the month, and the seed."""
    level, q, month, seed = name.split("_")
    return inputs.Setting(float(level[1:]), float(q[1:])), month, int(seed[1:])


def run_first() -> None:
    """Score every first-test forecast file and print and store the numbers."""
    results = {}
    for path in sorted((OUT_DIR / "first").glob("*/*.parquet")):
        setting, month, seed = parse_name(name=path.stem)
        key = f"{path.parent.name} {setting.tag} {month} s{seed}"
        results[key] = score_file(path=path, setting=setting, month=month)
        for r in results[key]:
            m = r["model"]
            print(
                f"{key} {r['site']:>6}: BA {m['balanced_accuracy']:.3f} recall {m['recall']:.3f} "
                f"false {m['false_reverse']:.3f} | rule BA {r['threshold_rule']['balanced_accuracy']:.3f} "
                f"| hour-month BA {r['hour_month_baseline']['balanced_accuracy']:.3f}"
            )
    (OUT_DIR / "first_scores.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first",))
    parser.parse_args()
    run_first()


if __name__ == "__main__":
    main()
