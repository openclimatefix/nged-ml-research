# ruff: noqa: E501, I001
"""Experiment 701, GAM side: stage 2 of the GAM for the origin months 2026-07 to 2026-10.

    E00377_SOURCE=control E00377_ARM=a0 PYTHONPATH=<folders of 377, 700, 701> DATA_PATH_INTERNAL=<sandbox> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_stage2.py --sites D5 [--first 2026-07] [--last 2026-10] [--dry-run]

This is `e00700_gam_stage2.py` with the origin months as arguments. Stage 2 applies experiment
155's arm-d mean correction (winter night offset and recipe R) to the stage 1 frame of
`e00701_gam_stage1.py`, then refits the spread table for each month on the corrected residuals of
runs whose origin and valid time both precede the month's first instant, six lead bands up to 336
hours (`e00377s3_stack.with_fold_spread`). D8 uses the donor spread, as experiment 378 does. The
months' rows replace those of the sandbox copy of `stack_d_<site>.parquet` (other rows are kept; the
write is atomic); the stored file is never touched. Months with stored rows are compared with
them (`max_abs_q95_diff_vs_stored`; `null` for months with no stored rows). `--dry-run` writes
nothing. Run stage 1 for every Primary (D8 needs its six donors) before stage 2.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00377s2_common as common  # ty: ignore[unresolved-import]
import e00377s3_stack as stack  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def month_start(text: str) -> datetime:
    """Return the first instant of a `YYYY-MM` month."""
    return datetime.strptime(text, "%Y-%m").replace(tzinfo=UTC)


def next_month(moment: datetime) -> datetime:
    """Return the first instant of the month after `moment`'s month."""
    return moment.replace(year=moment.year + (moment.month == 12), month=moment.month % 12 + 1)


def main() -> None:
    """Rebuild the months' stage 2 rows at each site and compare them with the stored rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--first", default="2026-07")
    parser.add_argument("--last", default="2026-10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first, last = month_start(args.first), month_start(args.last)
    months = []
    month = first
    while month <= last:
        months.append(month)
        month = next_month(month)
    end = next_month(last)
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
        new = stack.with_fold_spread(frame=frame, pools=pools, months=months, arm=arm)
        wall, cpu = time.monotonic() - t0, time.process_time() - c0
        assert new["lead_hours"].max() <= 339.0
        assert new["power_fcst_init_time"].min() >= first
        assert new["power_fcst_init_time"].max() < end
        assert new["power_fcst_init_time"].dt.truncate("1mo").n_unique() == len(months), (
            f"{site}: a month has no stage 1 rows"
        )
        path = common.OUT_DIR / f"{name}.parquet"
        old = pl.read_parquet(path)
        in_months = (pl.col("power_fcst_init_time") >= first) & (
            pl.col("power_fcst_init_time") < end
        )
        stored = old.filter(in_months)
        joined = new.join(stored, on=["run", "valid_time"], suffix="_old")
        quantiles = list(common.QUANTILES)
        diff = (
            max(
                float(np.nanmax(np.abs(joined[q].to_numpy() - joined[f"{q}_old"].to_numpy())))
                for q in quantiles
            )
            if joined.height
            else None
        )
        if not args.dry_run:
            assert ext.SANDBOX in path.parents, f"{path} is outside the sandbox"
            merged = pl.concat([old.filter(~in_months), new.select(old.columns)]).sort(
                ["power_fcst_init_time", "valid_time"]
            )
            temporary = path.with_name(f"{path.name}.tmp")
            merged.write_parquet(temporary)
            temporary.rename(path)
        rows[site] = {
            "rows": new.height,
            "stored_rows": stored.height,
            "joined_rows": joined.height,
            "wall_s": wall,
            "cpu_s": cpu,
            "max_abs_quantile_diff_vs_stored": diff,
            "dry_run": args.dry_run,
        }
        print(site, rows[site], flush=True)
    ext.TIMING.mkdir(parents=True, exist_ok=True)
    tag = "dry_" if args.dry_run else ""
    (ext.TIMING / f"gam_stage2_{tag}{common.WORK_DIR.name}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(rows, indent=2)
    )


if __name__ == "__main__":
    main()
