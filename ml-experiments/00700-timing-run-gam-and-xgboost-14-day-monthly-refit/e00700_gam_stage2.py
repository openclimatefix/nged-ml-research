"""Experiment 700: stage 2 of the GAM for origin month 2026-01, timed.

    E00377_SOURCE=control E00377_ARM=a0 \
    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python ml-experiments/00700-*/e00700_gam_stage2.py --sites D5

Stage 2 applies experiment 155's arm-d mean correction (winter night offset and recipe R) to the
stage 1 frame, then refits the spread table for the month on the corrected residuals of runs whose
origin and valid time both precede 2026-01-01 (`e00377s3_stack.with_fold_spread`), six lead bands up
to 336 hours. D8 uses the donor spread, as experiment 378 does. The month's rows replace the
rows of the sandbox copy of `stack_d_<site>.parquet`; the stored file is never touched. Writes
`timing/gam_stage2_<source>_<sites>.json`.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import e00377s2_common as common  # ty: ignore[unresolved-import]
import e00377s3_stack as stack  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

MONTH = datetime(2026, 1, 1, tzinfo=UTC)
NEXT = datetime(2026, 2, 1, tzinfo=UTC)
OUT = Path(os.environ["DATA_PATH_INTERNAL"]).resolve().parent / "timing"


def main() -> None:
    """Rebuild the month's stage 2 rows at each site and compare them with the stored rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    rows = {}
    for site in args.sites:
        t0, c0 = time.monotonic(), time.process_time()
        frame = stack.corrected_frame(site=site)
        if site == "D8":
            pools, arm, name = (
                [stack.corrected_frame(site=d) for d in common.DONORS],
                stack.DONOR_ARM,
                "stack_d_donor_D8",
            )
        else:
            pools, arm, name = [frame], stack.OWN_ARM, f"stack_d_{site}"
        month = stack.with_fold_spread(frame=frame, pools=pools, months=[MONTH], arm=arm)
        wall, cpu = time.monotonic() - t0, time.process_time() - c0
        assert month["lead_hours"].max() <= 339.0
        path = common.OUT_DIR / f"{name}.parquet"
        old = pl.read_parquet(path)
        stored = old.filter(
            (pl.col("power_fcst_init_time") >= MONTH) & (pl.col("power_fcst_init_time") < NEXT)
        )
        joined = month.join(stored, on=["run", "valid_time"], suffix="_old")
        diff = float(np.nanmax(np.abs(joined["q95"].to_numpy() - joined["q95_old"].to_numpy())))
        merged = pl.concat(
            [
                old.filter(
                    ~(
                        (pl.col("power_fcst_init_time") >= MONTH)
                        & (pl.col("power_fcst_init_time") < NEXT)
                    )
                ),
                month.select(old.columns),
            ]
        ).sort(["power_fcst_init_time", "valid_time"])
        merged.write_parquet(path)
        rows[site] = {
            "rows": month.height,
            "stored_rows": stored.height,
            "wall_s": wall,
            "cpu_s": cpu,
            "max_abs_q95_diff_vs_stored": diff,
        }
        print(site, rows[site], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gam_stage2_{common.WORK_DIR.name}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(rows, indent=2)
    )


if __name__ == "__main__":
    main()
