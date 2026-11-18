# ruff: noqa: E501
"""Experiment 700: time the monthly GAM refit (T-all) at the 2026-01-01 cut against the stored fit.

Run once per source, in the sandbox data root (see `e00700_sandbox.sh`):

    E00377_SOURCE=control PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python ml-experiments/00700-*/e00700_gam_refit.py --sites D5 D9

Writes `timing/gam_refit_<source>.json` under the experiment's private folder. The refit is the one
the production-style monthly cadence needs: the T-all fit of experiment 377 at the first of the
month, on the 24 months before the cut. The stored fit list of experiment 377 is read, never written.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_tall as tall  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

CUT = datetime(2026, 1, 1, tzinfo=UTC)
OUT = Path(os.environ["DATA_PATH_INTERNAL"]).resolve().parent / "timing"


def main() -> None:
    """Refit each site at the 2026-01 cut and report time and agreement with the stored fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    t0 = time.monotonic()
    cache0 = common.load_cache()
    load_s = time.monotonic() - t0
    rows = {}
    for site in args.sites:
        cache = common.variant_cache(cache=cache0, site=site)
        vintages = common.vintages_for(cache=cache)
        started, cpu = time.monotonic(), time.process_time()
        fit = tall.fit_tall(
            cache=cache, vintages=vintages, site=site, cut=CUT, penalty_weight=common.T_ALL_LAMBDA
        )
        wall, cpu_s = time.monotonic() - started, time.process_time() - cpu
        stored = next(f for f in tall.load_fits(cadence="monthly", site=site) if f.cut == CUT)
        diff = float(np.max(np.abs(np.asarray(fit.fit.theta) - np.asarray(stored.fit.theta))))
        rows[site] = {"wall_s": wall, "cpu_s": cpu_s, "max_abs_theta_diff_vs_stored": diff}
        print(site, rows[site], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gam_refit_{common.SOURCE}.json").write_text(
        json.dumps({"cache_load_s": load_s, "sites": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
