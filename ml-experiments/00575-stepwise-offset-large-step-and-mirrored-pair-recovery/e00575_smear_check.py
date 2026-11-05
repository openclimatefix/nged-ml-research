# ruff: noqa: E501
"""Experiment 575: does the offset move by the injected amount where `extract` finds no step?

For each scored Kl event of 20% of capacity or more lasting 30 days or more in the latest
fold, compare the offset's mean level in the 5 days before the end with its mean level in the
20 days before the onset, in capacity fraction, with the injected size. Prints the median ratio
and the share within 25%, per arm.
"""

import json

import e00573_steps as s573  # ty: ignore[unresolved-import]
import e00575_run as run
import numpy as np

fold = s573.load_fold(month="2026-06", device="cpu")
print(
    "arm | n | median recovered/injected | share of events within 25% | share with abs(step) over 1 day <0.1% (smeared)"
)
for arm in ["S0", "S1@3000", "S1@1000", "S2@3000/300", "S2@1000/30"]:
    ratios, smear = [], []
    for seed in range(5):
        res = json.loads((run.SCORES_DIR / f"2026-06_Kl{seed}.json").read_text())[arm]
        path = np.load(run.WORK / "paths" / f"2026-06_Kl{seed}_{arm.replace('/', '_')}.npy")
        for e in res["events"]:
            if not e["scored"] or abs(e["size"]) < 0.2 or e["duration"] < 30:
                continue
            p, std = e["primary"], fold.std[e["primary"]]
            before = path[p, e["onset"] - 20 : e["onset"] - 2].mean() * std
            during = path[p, e["end"] - 6 : e["end"] - 1].mean() * std
            ratios.append((during - before) / e["size"])
            smear.append(
                float(np.abs(np.diff(path[p, e["onset"] - 3 : e["onset"] + 4]) * std).max() < 0.001)
            )
    r = np.array(ratios)
    print(
        arm,
        len(r),
        round(float(np.median(r)), 2),
        round(float((np.abs(r - 1) <= 0.25).mean()), 2),
        round(float(np.mean(smear)), 2),
    )
