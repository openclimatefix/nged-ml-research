# ruff: noqa: I001
"""Experiment 700: forecast origin month 2026-01 with the GAM stage 1 (experiment 377) and time it.

    E00377_SOURCE=control PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
        uv run --no-sync python ml-experiments/00700-*/e00700_gam_stage1.py --sites D5

Each daily run of the month (00:00 UTC ENS run, native origin 09:00 UTC) is forecast 351 hours
ahead from the 2026-01-01 refit, with the correction filter and the spread table that experiment 377
uses. The weather is the run's own 51 ensemble members, so nothing after the origin enters. The
frame replaces the month's rows in the sandbox copy of experiment 377's `dp_preds` file (the
stored file is never touched) and is compared with the stored rows. Writes
`timing/gam_stage1_<source>.json`.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import e00377_common as common  # ty: ignore[unresolved-import]
import e00700_horizon  # noqa: F401  (sets the 678 half-hour window before e00377_run is imported)
import e00377_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 2, 1, tzinfo=UTC)
OUT = Path(os.environ["DATA_PATH_INTERNAL"]).resolve().parent / "timing"


def main() -> None:
    """Forecast the month at each site and compare it with the stored rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache = common.load_cache()
    rows = {}
    for site in args.sites:
        t0, c0 = time.monotonic(), time.process_time()
        context = run.load_site(site=site, cache=cache)
        forecaster = run.ArmForecaster(arm="a2", context=context)
        setup_s, setup_cpu = time.monotonic() - t0, time.process_time() - c0
        idx = [
            i
            for i, origin in enumerate(context.setup.origins)
            if START <= origin < END
            and common.period_of_run(run=context.setup.run_dates[i]) != "warm_up"
        ]
        # Weather at the origin: every run is the 00:00 UTC run, published before the 09:00 origin.
        assert all(
            context.setup.origins[i] - context.setup.run_dates[i] == common.PUBLICATION_DELAY
            for i in idx
        )
        t1, c1 = time.monotonic(), time.process_time()
        frame = forecaster.forecast(runs=idx)
        fc_s, fc_cpu = time.monotonic() - t1, time.process_time() - c1
        assert frame["lead_hours"].max() <= 339.0
        stored = run.out_path(arm="a2", period="scored", site=site)
        old = pl.read_parquet(stored)
        month = old.filter(
            (pl.col("power_fcst_init_time") >= START) & (pl.col("power_fcst_init_time") < END)
        )
        joined = frame.join(month, on=["run", "valid_time"], suffix="_old")
        diff = float(np.nanmax(np.abs(joined["q50"].to_numpy() - joined["q50_old"].to_numpy())))
        merged = pl.concat(
            [
                old.filter(
                    ~(
                        (pl.col("power_fcst_init_time") >= START)
                        & (pl.col("power_fcst_init_time") < END)
                    )
                ),
                frame.select(old.columns),
            ]
        ).sort(["power_fcst_init_time", "valid_time"])
        merged.write_parquet(stored)
        rows[site] = {
            "runs": len(idx),
            "rows": frame.height,
            "stored_rows": month.height,
            "setup_wall_s": setup_s,
            "setup_cpu_s": setup_cpu,
            "forecast_wall_s": fc_s,
            "forecast_cpu_s": fc_cpu,
            "max_lead_h": float(frame["lead_hours"].max()),
            "max_abs_q50_diff_vs_stored": diff,
        }
        print(site, rows[site], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gam_stage1_{common.SOURCE}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(rows, indent=2)
    )


if __name__ == "__main__":
    main()
