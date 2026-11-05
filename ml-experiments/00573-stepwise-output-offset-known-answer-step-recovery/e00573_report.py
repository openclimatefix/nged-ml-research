# ruff: noqa: E501
"""Experiment 573: collate Step 1 into the K-S1, K-S2, and K-S3 verdicts.

Prints every number the page quotes.
"""

import json

import e00573_steps as s  # ty: ignore[unresolved-import]
import numpy as np

R = s.WORK / "step1_run_r2"
ARMS = ["S0", "S1@3000", "S1@1000", "S2@3000/300", "S2@3000/30", "S2@1000/300", "S2@1000/30"]


def load(month: str, world: str) -> dict:
    """Return one fold-world result."""
    return json.loads((R / f"{month}_{world}.json").read_text())


def main() -> None:
    """Print per-arm recall, precision, false-step rate, and ramp pass share per fold, then the verdicts."""
    for arm in ARMS:
        rec, prec, fa, ram, nfound = [], [], [], [], []
        for m in s.MONTHS:
            ks, k0, kr = load(m, "Ks")[arm], load(m, "K0")[arm], load(m, "Kr")[arm]
            rec.append(ks["recall_3pct"])
            prec.append(ks["precision"])
            fa.append(k0["injected_six"]["per_primary_year"])
            ram.append(kr["pass_share"])
            nfound.append((ks["n_truth_3pct"], ks["n_recovered_3pct"]))
        ks1 = sum(r >= 0.8 for r in rec) >= 4
        ks2 = sum(f <= 1.0 for f in fa) >= 4
        ks3 = float(np.nanmean(ram)) >= 2 / 3
        print(arm)
        print(
            "  recall(3%+) per fold", [round(r, 2) for r in rec], "n(truth,3%+ recovered)", nfound
        )
        print("  precision per fold", [round(p, 3) for p in prec])
        print("  K0 false steps >=1% per Primary-year (six) per fold", [round(f, 2) for f in fa])
        print(
            "  K0 untouched ten",
            [round(load(m, "K0")[arm]["untouched_ten"]["per_primary_year"], 2) for m in s.MONTHS],
        )
        print("  ramp pass share per fold", [round(x, 2) for x in ram])
        print(f"  K-S1 {ks1}  K-S2 {ks2}  K-S3 {ks3}")


if __name__ == "__main__":
    main()
