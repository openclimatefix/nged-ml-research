# ruff: noqa: E501
"""Score experiment 417's fits: the predicted sign against the known signed truth.

Experiment 412's metrics, pooled over the two test months (point estimates, interim): on the
headline slice (real scaled CAMS unit shape at least 0.3 and clear-sky index at least 0.7) the
balanced accuracy, recall, and false-reverse rate of the predicted sign (`P(X < 0) > 0.5`); the
false-reverse rate on sunny summer middays with no injected and no real reverse flow; the pinball
loss of the 11 signed quantiles; and the coverage and pinball loss at the 0.01, 0.05, 0.95, and
0.99 quantiles. The pass bars are read from these numbers by `bars`.

```bash
. env.sh && uv run --no-sync python ml-experiments/00417-*/e00417_score.py run
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00412_score as s412  # ty: ignore[unresolved-import]
import e00417_inputs as inputs
import e00417_models as models
import numpy as np
import polars as pl

OUT_DIR: Final[Path] = inputs.WORK_DIR
SUNNY_K: Final[float] = inputs.i412.SUNNY_K
LEVEL_COLUMNS: Final[tuple[str, ...]] = s412.LEVEL_COLUMNS
TAILS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
BARS: Final[dict[str, float]] = {
    "pooled_ba": 0.85,
    "site_ba": 0.70,
    "false_reverse": 0.05,
    "placebo_false_reverse_d5": 0.01,
    "margin_over_shuffle": 0.10,
}


def tails(*, frame: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Return the coverage and pinball loss of the signed quantiles at the tail levels."""
    truth = frame["truth_points"].to_numpy()
    out = {}
    for tau in TAILS:
        q = frame[f"q{round(100 * tau):02d}"].to_numpy()
        diff = truth - q
        out[f"{tau:g}"] = {
            "coverage": float(np.mean(truth <= q)),
            "pinball": float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff))),
        }
    return out


def score_frame(*, frame: pl.DataFrame) -> list[dict]:
    """Score a forecast frame: one row per hidden Primary and one pooled row."""
    frame = frame.filter(pl.col("truth_points").is_finite())
    out, pooled = [], {"truth": [], "pred": [], "hm": []}
    for site in frame["site"].unique(maintain_order=True).to_list():
        part = frame.filter(pl.col("site") == site)
        u_gate, k = part["u_gate"].to_numpy(), part["k"].to_numpy()
        gate = (u_gate >= models.GATE_U) & (k >= SUNNY_K)
        truth = part["truth_points"].to_numpy() < 0.0
        pred = part["p_negative"].to_numpy() > 0.5
        midday = inputs.is_midday(valid_us=part["valid"].to_numpy())
        out.append(
            {
                "site": site,
                "n_slice": int(gate.sum()),
                "model": s412.balanced(truth_reverse=truth[gate], predicted=pred[gate]),
                "hour_month_baseline": s412.balanced(
                    truth_reverse=truth[gate], predicted=midday[gate]
                ),
                "false_reverse_sunny_midday": float(pred[gate & midday & ~truth].mean())
                if (gate & midday & ~truth).any()
                else float("nan"),
                "pinball_points": s412.pinball(frame=part),
                "tails": tails(frame=part),
                "share_rows_open": float(part["row_open"].cast(pl.Float64).mean()),  # ty: ignore[invalid-argument-type],
            }
        )
        pooled["truth"].append(truth[gate])
        pooled["pred"].append(pred[gate])
        pooled["hm"].append(midday[gate])
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    out.append(
        {
            "site": "pooled",
            "model": s412.balanced(truth_reverse=cat["truth"], predicted=cat["pred"]),
            "hour_month_baseline": s412.balanced(truth_reverse=cat["truth"], predicted=cat["hm"]),
        }
    )
    return out


def run() -> dict:
    """Score every arm and setting over both months, and read the bars."""
    results: dict = {}
    first = OUT_DIR / "first"
    for arm_dir in sorted(p for p in first.iterdir() if p.is_dir()):
        groups: dict[str, list[Path]] = {}
        for path in sorted(arm_dir.glob("*.parquet")):
            tag, _month, seed = path.stem.rsplit("_", 2)
            groups.setdefault(f"{tag}_{seed}", []).append(path)
        for key, paths in groups.items():
            frame = pl.concat([pl.read_parquet(p) for p in paths])
            infos = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
            entry = {
                "months": [p.stem.rsplit("_", 2)[1] for p in paths],
                "scores": score_frame(frame=frame),
                "capacity_points": [i["capacity_points"] for i in infos],
                "prior_gradient_share_end": [i["prior_gradient_share"]["end"] for i in infos],
                "gates": [i.get("gates") for i in infos],
                "labels": [i.get("labels") for i in infos],
                "dropped": [sum(i["dropped"].values()) for i in infos],
            }
            results[f"{arm_dir.name} {key}"] = entry
    (OUT_DIR / "scores.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    """Score and print the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.parse_args()
    for key, entry in run().items():
        for r in entry["scores"]:
            m = r["model"]
            extra = (
                f" false-sunny-midday {r['false_reverse_sunny_midday']:.3f} open {r['share_rows_open']:.2f}"
                if "false_reverse_sunny_midday" in r
                else ""
            )
            print(
                f"{key} {r['site']:>6}: BA {m['balanced_accuracy']:.3f} recall {m['recall']:.3f} "
                f"false {m['false_reverse']:.3f} n_rev {m['n_reverse']}{extra}"
            )


if __name__ == "__main__":
    main()
