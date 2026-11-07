# ruff: noqa: E501
"""Experiment 624, test 1: scaling audit of the injected farms A to F (CPU, no fitting).

Each farm's series is its output divided by the 99th percentile of its readings with the sun under 75
degrees of zenith, then multiplied by the injected megawatts, so the injected series reaches the
injected megawatts at its own 99th percentile by construction. The audit therefore compares the
farm's normalised series with the physics unit curve driven by each MW Primary's CAMS shortwave at
the same half-hours (the unit curve is scaled by experiment 405's `u99`): the mean ratio of the two
sets how much more or less energy a "4 MW" real farm injects than a "4 MW" physics curve, which sets
the recovery that the probe can reach even if it were exact.

Usage: python e00624_audit.py --cutoff 2026-06
"""

import argparse
import json

import e00405_prep as p5  # ty: ignore[unresolved-import]
import e00409_prep as p9  # ty: ignore[unresolved-import]
import e00623_common as k
import e00623_probe as p623  # ty: ignore[unresolved-import]
import numpy as np


class Std:
    """The frame's standardiser, in the form `raw_weather` reads."""

    def __init__(self, *, mean: np.ndarray, std: np.ndarray) -> None:
        """Hold the first seven weather columns' mean and standard deviation."""
        self.mean, self.std = mean, std


def main() -> None:
    """Write `work/audit_<cutoff>.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    args = parser.parse_args()
    frame = p623.load_frame(cutoff=args.cutoff)
    six = k.six_indices()
    out: dict = {"farm": {}, "farm_by_primary": {}}
    for farm in k.FARMS:
        _, v = p5.farm_norm_series(farm=farm)
        ok = np.isfinite(v)
        day = v[ok & (v > 0)]
        out["farm"][farm] = {
            "p99": round(float(np.percentile(day, 99)), 3), "p999": round(float(np.percentile(day, 99.9)), 4),
            "max": round(float(day.max()), 3), "share_above_1": round(float((day > 1.0).mean()), 4),
            "share_above_0.95": round(float((day > 0.95).mean()), 4),
        }  # fmt: skip
    for site, i in six.items():
        sel = frame["primary"] == i
        x, idx = frame["x"][sel], frame["target_idx"][sel]
        times = k.time_us(target_idx=idx)
        ghi, temp = k.raw_weather(
            x=x, standardiser=Std(mean=frame["std_mean"], std=frame["std_std"])
        )
        u = k.unit_curve(ghi=ghi, x=x, times_us=times, temp=temp)
        out["farm_by_primary"][site] = {}
        for farm in k.FARMS:
            series = p5.farm_norm_series(farm=farm)
            f = p9.series_at(times_us=times, series=series, shuffle_salt=None)
            ok = np.isfinite(f) & (u > 0.02)
            out["farm_by_primary"][site][farm] = {
                "mean_ratio_farm_over_unit_curve": round(float(f[ok].sum() / u[ok].sum()), 3),
                "slope_through_origin": round(float((f[ok] * u[ok]).sum() / (u[ok] ** 2).sum()), 3),
                "n": int(ok.sum()),
            }
    path = (
        k.WORK.parent.parent
        / "00624-pv-probe-contrastive-injection-arms"
        / "work"
        / f"audit_{args.cutoff}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
