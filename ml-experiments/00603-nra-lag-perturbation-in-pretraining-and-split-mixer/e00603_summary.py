# ruff: noqa: E501, ANN001, ANN201, ANN003, E402, PT018, B023, F821, ANN202, D103, SIM115, PTH123, PERF401, UP037, F841, RUF100
"""Experiment 603: print the headline table of the scored arms (CPU only, reads each arm's score folder).

Per arm and metering group, on injected 15 and 30 day events at 24-120 h for the lag network (arm V):
q05-q95 coverage and safe-side share by phase, the median's absolute error against the NRA (`ae50`,
the follow proxy: it grows when the median follows the shifted lags), band width; and on event-free
rows the coverage, width, and loss of the 90% band.

Usage: python e00603_summary.py SUFFIX ARM [ARM ...]   (reads `score_ARM` plus the suffix)
"""

import csv
import sys

from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = ML_EXPERIMENTS_DATA_DIR / "00603-nra-lag-perturbation-in-pretraining-and-split-mixer"
suffix, arms = sys.argv[1], sys.argv[2:]
PHASES = ("days4-end", "after0-7", "after7-14")
GROUPS = ("MVA-only eight", "signed-MW six")
rows: dict[str, list[dict]] = {}
free: dict[str, list[dict]] = {}
for a in arms:
    d = ROOT / f"score_{a}{suffix}"
    rows[a] = list(csv.DictReader(open(d / "bank_summary.csv")))
    free[a] = list(csv.DictReader(open(d / "event_free.csv")))
print(
    "injected 15 and 30 day events, 24-120 h, arm V: cover90 | safe-side nra_side95 | ae50 | width90"
)
for g in GROUPS:
    for ph in PHASES:
        out = []
        for a in arms:
            r = [
                x
                for x in rows[a]
                if x["group"] == g
                and x["length_class"] == "15 and 30 days"
                and x["phase"] == ph
                and x["band"] == "24-120 h"
                and x["arm"] == "V"
            ]
            if r:
                x = r[0]
                out.append(
                    f"{a}: {float(x['cover90']):.3f} | {float(x['nra_side95']):.3f} | {float(x['ae50']):.4f} | {float(x['width90']):.3f}"
                )
        print(f"{g:15s} {ph:10s}", "   ".join(out))
print("event-free, 24-120 h, arm V: cover90 | width90 | loss90")
for g in GROUPS:
    out = []
    for a in arms:
        r = [x for x in free[a] if x["group"] == g and x["band"] == "24-120 h" and x["arm"] == "V"]
        if r:
            x = r[0]
            out.append(
                f"{a}: {float(x['cover90']):.3f} | {float(x['width90']):.3f} | {float(x['loss90']):.4f}"
            )
    print(f"{g:15s}", "   ".join(out))
