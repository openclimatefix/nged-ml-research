# ruff: noqa: E501
"""Step 0: experiment 417's day-flip diagnostic (design I) rerun with a corrected broken-cloud threshold.

The audit of 2026-10-30 found that design I's broken-cloud rule (mean absolute second difference of
hourly `u` above 0.05) also classed clear days as broken cloud: at D5, 82% of days whose hourly
clear-sky index `k` stays at or above 0.8 through daylight exceeded 0.05. This script sets the
threshold to the 75th percentile of that roughness over the clear days of the six signed-MW
Primaries, writes it to disk before any diagnostic score is computed, then reruns experiment 417's
`run` with that threshold. No network is fitted. The result is reported with no bar.

Experiment 417's `run` writes `day_flip.json` to a diagnostics folder; this script points that
folder at experiment 420's own work folder first, so experiment 417's stored files are not touched.

```bash
. env.sh && uv run --no-sync python ml-experiments/00420-*/e00420_day_flip.py threshold
. env.sh && uv run --no-sync python ml-experiments/00420-*/e00420_day_flip.py run
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00417_day_flip as flip  # ty: ignore[unresolved-import]
import e00417_diagnostics as diag  # ty: ignore[unresolved-import]
import e00417_inputs as inputs  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00420-neural-mva-export-gated-signed-final" / "work" / "day_flip"
)
CLEAR_K: Final[float] = 0.8
"""A day is clear when the hourly clear-sky index is at least this in every daylight hour."""
PERCENTILE: Final[float] = 75.0


def clear_day_roughness() -> dict[str, np.ndarray]:
    """Return, per signed-MW Primary, the roughness of `u` on its clear eligible summer days.

    The roughness is the mean absolute second difference of hourly `u` over the day's daylight hours
    (`u >= 0.1`), as `score_days` computes it. `u` and `k` come from CAMS and do not depend on the
    injection, so the no-injection setting is read.
    """
    placebo = inputs.Setting(0.0, 2.0)
    out = {}
    for site in inputs.SIGNED_SIX:
        h = flip.hourly(t=diag.load_table(site=site, setting=placebo))
        values = []
        for day in np.unique(h.day):
            sel = np.flatnonzero(h.day == day)
            if not (flip.SUMMER[0] <= h.month[sel[0]] <= flip.SUMMER[1]):
                continue
            if h.u[sel].max() < flip.MIN_DAY_MAX_U:
                continue
            daylight = h.u[sel] >= flip.U_LOW
            if daylight.sum() <= 2 or (h.k[sel][daylight] < CLEAR_K).any():
                continue
            u_by_time = h.u[sel][daylight][np.argsort(h.hod[sel][daylight])]
            values.append(float(np.mean(np.abs(np.diff(u_by_time, n=2)))))
        out[site] = np.array(values)
    return out


def main() -> None:
    """Fix the threshold, or rerun the diagnostic with it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("threshold", "run"))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "threshold.json"
    if args.stage == "threshold":
        rough = clear_day_roughness()
        pooled = np.concatenate(list(rough.values()))
        result = {
            "threshold": float(np.percentile(pooled, PERCENTILE)),
            "percentile": PERCENTILE,
            "clear_k": CLEAR_K,
            "n_clear_days": {s: len(v) for s, v in rough.items()},
            "median_by_primary": {s: float(np.median(v)) for s, v in rough.items()},
            "share_above_old_threshold": {s: float((v > 0.05).mean()) for s, v in rough.items()},
        }
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    threshold = json.loads(path.read_text())["threshold"]
    flip.BROKEN_ROUGHNESS = threshold
    diag.OUT_DIR = OUT_DIR
    results = flip.run()
    print(f"broken-cloud threshold {threshold:.4f}")
    for name, e in results.items():
        print(
            f"== {name}: AUC {e['auc_positive_vs_negative_days']:.3f}; paired AUC "
            f"{e['paired_auc_pooled']:.3f}; recall {e['pooled_recall']:.2f} of "
            f"{e['n_positive_days']} positive days; AUC smooth {e['auc_smooth_days']:.3f} "
            f"(n_pos {e['n_positive_smooth']}), broken {e['auc_broken_days']:.3f} "
            f"(n_pos {e['n_positive_broken']})"
        )
        for s, p in e["per_primary"].items():
            print(
                f"  {s}: recall {p['recall_at_threshold']:.2f} (n_pos {p['n_positive']}), false flag "
                f"{p['false_flag_placebo']:.3f}, broken {p['n_broken_injected']} of "
                f"{p['n_eligible_injected']} injected days"
            )


if __name__ == "__main__":
    main()
