"""Rule-only diagnostics for experiment 408: the signed-label rule on its own, with no model fit.

Fits the rule at the refit 2026-01-01 (training window only) and prints, per Primary:

- mode `abs` (real readings as apparent power): whether the gate opens, its two statistics, `C`,
  and the share of sunny summer-midday half-hours labelled reverse flow at tau 0.7 and 0.5;
- mode `inj:<level>`: the rule's recall on true reverse flow and its false-reverse rate over the
  daylight half-hours scored from that refit on.

```bash
E408_MODE=abs PYTHONPATH=... uv run --no-sync python e00408_rule_check.py --mode abs
```
"""

import argparse
import os
from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00408_common as c
import numpy as np
import polars as pl

REFIT = datetime(2026, 1, 1, tzinfo=UTC)
SITES = (*c.SIGNED_SITES, *c.MVA_ONLY)


def main() -> None:
    """Print the rule diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    if os.environ.get(c.MODE_ENV) != args.mode:
        msg = f"set {c.MODE_ENV}={args.mode}"
        raise SystemExit(msg)
    series, arrays = fd._series_and_arrays()
    grid = common.grid_times()
    hour, month = c.start_hour_and_month(grid=grid)
    truths = fd._CACHE["truth"]
    for site in SITES:
        row = next(series.filter(pl.col("site") == site).iter_rows(named=True))
        cams = c.cams_arrays(row=row)
        y = arrays[site].power.astype(float) * 100
        finite = np.isfinite(y) & np.isfinite(cams.u) & np.isfinite(cams.k)
        usable = finite & (grid <= c.latest_valid(refit=REFIT)).to_numpy()
        rule = c.fit_rule(y_points=y, u=cams.u, k=cams.k, hour=hour, month=month, usable=usable)
        parts = [
            (
                f"{site} gate_open={rule.gate_open} spearman={rule.spearman:.2f} "
                f"p1={rule.p1_points:.1f} C={rule.capacity:.1f}"
            )
        ]
        masks = {
            tau: c.reverse_mask(rule=rule, tau=tau, y_points=y, u=cams.u, k=cams.k, hour=hour)
            for tau in (0.7, 0.5)
        }
        if site in truths and args.mode.startswith("inj:"):
            truth = truths[site] * 100
            scored = finite & np.isfinite(truth) & (grid >= REFIT).to_numpy() & (cams.u > 0.01)
            true_reverse = truth < -1.0
            for tau, mask in masks.items():
                parts.append(
                    f"tau={tau}: recall={mask[scored & true_reverse].mean():.2f} "
                    f"false={mask[scored & ~true_reverse].mean():.3f}"
                )
        else:
            midday = (
                finite
                & c._summer_midday(hour=hour, month=month)
                & (cams.k >= c.GATE_K_MIN)
                & (grid <= c.latest_valid(refit=REFIT)).to_numpy()
            )
            for tau, mask in masks.items():
                parts.append(
                    f"tau={tau}: reverse share of sunny summer middays={mask[midday].mean():.3f}"
                )
        print("  ".join(parts))


if __name__ == "__main__":
    main()
