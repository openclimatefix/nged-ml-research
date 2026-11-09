# ruff: noqa: E501
"""Experiment 661, step 6: experiment 644's explicit-minus-free contrast at 3-hourly and 6-hourly ensemble leads.

Reads experiment 644's stored quantiles (arms `a` free and `b` calibrated explicit, real months, five
seeds) and splits the sunny-midday contrast by the ensemble's native step width: a power-forecast
lead up to 135 hours is an ensemble lead up to 144 hours (3-hourly steps), and a longer one sits on
the 6-hourly steps. The power-forecast origin is the ensemble initialisation plus 9 hours. Also gives
the contrast by lead day. A positive value favours the explicit network.

Usage: python e00661_step6.py
"""

import json
from typing import Final

import e00639_analyse as a639  # ty: ignore[unresolved-import]
import e00644_analyse as a644  # ty: ignore[unresolved-import]
import e00661_audit as audit
import numpy as np

N_CONTRASTS: Final = 12
"""This page's contrasts, for the Bonferroni level (step 6: two step widths; step 5: three cutoff-pooled contrasts; Part 3: three bands and a placebo)."""
LEVEL: Final = 1.0 - a639.ALPHA / N_CONTRASTS
ORIGIN_DELAY_H: Final = 9.0
SEEDS: Final = (0, 1, 2, 3, 4)


def main() -> None:
    """Score the split contrast and write it next to the audit output."""
    rows = a644.load(world="real")
    ca = a644.arm_crps(world="real", arm="a", rows=rows, seeds=SEEDS)
    cb = a644.arm_crps(world="real", arm="b", rows=rows, seeds=SEEDS)
    ens_lead = rows["lead"] + ORIGIN_DELAY_H
    base = rows["sunny"] & rows["in_window"]
    out: dict = {"level": LEVEL, "by_step_width": {}, "by_lead_day": {}}
    for name, mask in {
        "3-hourly (ensemble lead up to 144 h)": base & (ens_lead <= 144.0),
        "6-hourly (ensemble lead above 144 h)": base & (ens_lead > 144.0),
        "all leads": base,
    }.items():
        diff = (ca - cb).mean(axis=0)
        mean, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
        out["by_step_width"][name] = {
            "free_minus_explicit": mean,
            "lo": lo,
            "hi": hi,
            "n_rows": int(mask.sum()),
            "free_crps": float(ca[:, mask].mean()),
            "explicit_crps": float(cb[:, mask].mean()),
        }
        print(name, round(mean, 3), round(lo, 3), round(hi, 3), int(mask.sum()), flush=True)
    for day in range(1, 11):
        mask = base & (rows["lead_day"] == day)
        diff = (ca - cb).mean(axis=0)
        mean, lo, hi = a639.bootstrap(diff=diff[mask], day=rows["day"][mask], level=LEVEL)
        out["by_lead_day"][day] = {
            "free_minus_explicit": mean,
            "lo": lo,
            "hi": hi,
            "n_rows": int(mask.sum()),
        }
        print("day", day, round(mean, 3), round(lo, 3), round(hi, 3), int(mask.sum()), flush=True)
    audit.OUT.mkdir(parents=True, exist_ok=True)
    (audit.OUT / "step6.json").write_text(json.dumps(out, indent=1, default=float))
    np.set_printoptions(precision=3)


if __name__ == "__main__":
    main()
