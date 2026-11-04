# ruff: noqa: E501
"""Part 3 of experiment 522, by metering group: the CRPS change of every arm at the six and the eight.

Reads `work/part3_per_day_<network>.npz` (per Primary and origin day) and writes
`work/part3_groups.json`: the mean change over the 6 signed-MW and the 8 MVA-only Primaries that are
not report-only, with a 95% day-block bootstrap interval, in percentage points of effective capacity.
"""

import json
from typing import Final

import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np

BOOTSTRAPS: Final[int] = 2_000


def main() -> None:
    """Write the by-group changes and print them."""
    part3 = json.loads((common.WORK_DIR / "part3.json").read_text())
    sites = part3["sites"]
    signed = [sites.index(s) for s in ("D5", "D9", "D10", "D11", "D14", "D16")]
    eight = [i for i, s in enumerate(sites) if i not in signed and s not in ("D15", "D18")]
    rng = np.random.default_rng(3)
    out: dict[str, dict] = {}
    for name in ("L0PT_pretrained", "L0PT_finetuned", "A"):
        z = np.load(common.WORK_DIR / f"part3_per_day_{name}.npz")
        identity = z["identity"]
        n_days = identity.shape[1]
        resamples = rng.integers(0, n_days, size=(BOOTSTRAPS, n_days))
        out[name] = {}
        for arm in (k for k in z.files if k != "identity"):
            delta = (z[arm] - identity[None]).mean(axis=0)
            for label, idx in (("six", signed), ("eight", eight)):
                per_day = np.nanmean(delta[idx], axis=0)
                boot = [100 * np.nanmean(per_day[r]) for r in resamples]
                out[name][f"{arm}|{label}"] = [
                    round(100 * float(np.nanmean(per_day)), 3),
                    round(float(np.quantile(boot, 0.025)), 3),
                    round(float(np.quantile(boot, 0.975)), 3),
                ]
        out[name]["identity_six"] = round(
            100 * float(np.nanmean(np.nanmean(identity[signed], axis=1))), 3
        )
        out[name]["identity_eight"] = round(
            100 * float(np.nanmean(np.nanmean(identity[eight], axis=1))), 3
        )
    (common.WORK_DIR / "part3_groups.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
