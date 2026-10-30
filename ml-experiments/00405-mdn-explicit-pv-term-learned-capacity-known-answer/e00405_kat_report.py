# ruff: noqa: E501
"""Read the known-answer fits of experiment 405 and apply the gates of the plan.

Recovery at a Primary and seed is the learned capacity in the as-of season-year cell, injected
minus control, divided by the injected increment X (50% of the studies page's 2025/26 capacity).
The divisor is the median recovery of the divisor farms A, B, and D at the same Primary. The
recovery gate is a divided recovery of 0.75 to 1.25 at 5 of the 6 Primaries (the seed-mean). The
placebo gate is the placebo recovery over the held-out farm's real-sky recovery, at most 0.20 in
median and maximum. The drift gate is reported, not gating (the plan explains why).

Only ratios leave the private folder; no farm series is printed.
"""

import json
from pathlib import Path
from typing import Final

import e00405_prep as prep  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import numpy as np

KAT_DIR: Final[Path] = pv.WORK_DIR / "kat"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
RECOVERY_LOW, RECOVERY_HIGH, RECOVERY_SITES_NEEDED = 0.75, 1.25, 5
PLACEBO_LIMIT: Final[float] = 0.20
SHARE: Final[float] = 0.5


def capacity(*, arm: str, condition: str, seed: int, site: str, cell: str = "as_of") -> float:
    """Return a fit's learned capacity (MW) at a Primary, in the as-of cell or by index."""
    d = json.loads((KAT_DIR / arm / f"{condition}_s{seed}.json").read_text())
    values = d["capacity_mw"][site]
    return float(values[d["as_of_year_cell"]] if cell == "as_of" else values[int(cell)])


def recovery(*, arm: str, condition: str, site: str, seed: int, cell: str = "as_of") -> float:
    """Return the raw recovery of an injected condition at a Primary and seed."""
    x = SHARE * prep.studies_capacity_2025(site=site)
    gain = capacity(arm=arm, condition=condition, seed=seed, site=site, cell=cell) - capacity(
        arm=arm, condition="control", seed=seed, site=site, cell=cell
    )
    return gain / x


def report(*, arm: str) -> dict:
    """Return the recovery table and gate results of an arm."""
    rows = {}
    for site in pv.SIGNED_SIX:
        held = np.array([recovery(arm=arm, condition="held", site=site, seed=s) for s in SEEDS])
        divisors = np.array(
            [
                np.median(
                    [recovery(arm=arm, condition=f"divisor_{f}", site=site, seed=s) for f in "ABD"]
                )
                for s in SEEDS
            ]
        )
        placebo = np.array(
            [recovery(arm=arm, condition="placebo", site=site, seed=s) for s in SEEDS]
        )
        divided = held / divisors
        rows[site] = {
            "raw_held": float(held.mean()),
            "divisor": float(divisors.mean()),
            "divided": float(divided.mean()),
            "divided_seed_sd": float(divided.std(ddof=1)),
            "placebo_over_held": float(placebo.mean() / held.mean()),
            "within_tolerance": bool(RECOVERY_LOW <= divided.mean() <= RECOVERY_HIGH),
        }
    n_ok = sum(r["within_tolerance"] for r in rows.values())
    ratios = np.array([r["placebo_over_held"] for r in rows.values()])
    return {
        "arm": arm,
        "sites": rows,
        "recovery_gate_sites": n_ok,
        "recovery_gate_pass": n_ok >= RECOVERY_SITES_NEEDED,
        "placebo_median": float(np.median(np.abs(ratios))),
        "placebo_max": float(np.max(np.abs(ratios))),
        "placebo_gate_pass": bool(
            np.max(np.abs(ratios)) <= PLACEBO_LIMIT and np.median(np.abs(ratios)) <= PLACEBO_LIMIT
        ),
    }


def main() -> None:
    """Print each arm's gates and write them under the private folder."""
    out = {}
    for arm in ("C2", "C4"):
        if not (KAT_DIR / arm / "placebo_s2.json").exists():
            print(f"{arm}: fits not finished")
            continue
        out[arm] = report(arm=arm)
        r = out[arm]
        print(
            f"{arm}: recovery gate {r['recovery_gate_sites']}/6 sites ({'pass' if r['recovery_gate_pass'] else 'FAIL'}); "
            f"placebo median {r['placebo_median']:.2f}, max {r['placebo_max']:.2f} ({'pass' if r['placebo_gate_pass'] else 'FAIL'})"
        )
        for site, v in r["sites"].items():
            print(
                f"  {site}: raw {v['raw_held']:.2f}, divisor {v['divisor']:.2f}, divided {v['divided']:.2f} (seed sd {v['divided_seed_sd']:.2f}), placebo/held {v['placebo_over_held']:.2f}"
            )
    (pv.WORK_DIR / "kat_gates.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
