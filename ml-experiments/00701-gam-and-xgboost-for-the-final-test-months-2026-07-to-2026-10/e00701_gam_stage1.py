# ruff: noqa: E501, I001
"""Experiment 701, GAM side: stage 1 of the GAM (arm a2) for the origin months 2026-07 to 2026-10.

    E00377_SOURCE=control PYTHONPATH=<folders of 377, 700, 701> DATA_PATH_INTERNAL=<sandbox> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_stage1.py --sites D5 [--first 2026-07] [--last 2026-10] [--dry-run]

This is `e00700_gam_stage1.py` with the origin months as arguments. Each daily 00:00 UTC ENS run of
the months (native origin 09:00 UTC) is forecast 339 hours ahead (678 half-hours) from the monthly
T-all fit of the latest cut at or before the origin (`e00701_gam_refit.py` makes the cuts
2026-07-01 to 2026-10-01), with the filter correction and the tuning-month spread of experiment 377.
The weather is the run's own 51 ensemble members, so nothing after the origin enters.

The frame replaces the months' rows in the sandbox copy of `dp_preds/a2_scored_<site>.parquet`
(rows of other months are kept; rewriting is atomic). Rows of the months already stored are compared
with the stored rows (`max_abs_quantile_mean_actual_diff_vs_stored`, the largest difference over the 11 quantiles, the mean, and the actual; `null` for months with no stored rows).
`--dry-run` computes and compares but writes nothing, for the smoke test on a stored month.
Writes `timing/gam_stage1_<source>_<sites>.json`.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


QUANTILES = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")


def month_start(text: str) -> datetime:
    """Return the first instant of a `YYYY-MM` month."""
    return datetime.strptime(text, "%Y-%m").replace(tzinfo=UTC)


def main() -> None:
    """Forecast the months at each site and compare them with the stored rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--first", default="2026-07")
    parser.add_argument("--last", default="2026-10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first = month_start(args.first)
    last = month_start(args.last)
    end = last.replace(year=last.year + (last.month == 12), month=last.month % 12 + 1)
    torch.set_num_threads(1)
    cache = common.load_cache()
    rows = {}
    for site in args.sites:
        t0, c0 = time.monotonic(), time.process_time()
        context = run.load_site(site=site, cache=cache)
        forecaster = run.ArmForecaster(arm="a2", context=context)
        setup_s, setup_cpu = time.monotonic() - t0, time.process_time() - c0
        assert forecaster.fits[-1].cut >= last, (
            f"{site}: the fit list ends at {forecaster.fits[-1].cut}, run e00701_gam_refit.py first"
        )
        idx = [
            i
            for i, origin in enumerate(context.setup.origins)
            if first <= origin < end
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
        assert frame["power_fcst_init_time"].min() >= first
        assert frame["power_fcst_init_time"].max() < end
        stored = run.out_path(arm="a2", period="scored", site=site)
        old = pl.read_parquet(stored)
        in_months = (pl.col("power_fcst_init_time") >= first) & (
            pl.col("power_fcst_init_time") < end
        )
        month = old.filter(in_months)
        joined = frame.join(month, on=["run", "valid_time"], suffix="_old")
        diff = (
            max(
                float(np.nanmax(np.abs(joined[q].to_numpy() - joined[f"{q}_old"].to_numpy())))
                for q in (*QUANTILES, "mean", "actual")
            )
            if joined.height
            else None
        )
        if not args.dry_run:
            assert ext.SANDBOX in stored.parents, f"{stored} is outside the sandbox"
            merged = pl.concat([old.filter(~in_months), frame.select(old.columns)]).sort(
                ["power_fcst_init_time", "valid_time"]
            )
            temporary = stored.with_name(f"{stored.name}.tmp")
            merged.write_parquet(temporary)
            temporary.rename(stored)
        rows[site] = {
            "runs": len(idx),
            "rows": frame.height,
            "stored_rows": month.height,
            "joined_rows": joined.height,
            "setup_wall_s": setup_s,
            "setup_cpu_s": setup_cpu,
            "forecast_wall_s": fc_s,
            "forecast_cpu_s": fc_cpu,
            "max_lead_h": float(frame["lead_hours"].max()),
            "max_abs_quantile_mean_actual_diff_vs_stored": diff,
            "dry_run": args.dry_run,
        }
        print(site, rows[site], flush=True)
    ext.TIMING.mkdir(parents=True, exist_ok=True)
    tag = "dry_" if args.dry_run else ""
    (ext.TIMING / f"gam_stage1_{tag}{common.SOURCE}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(rows, indent=2)
    )


if __name__ == "__main__":
    main()
