# ruff: noqa: D205, D209, E501
"""Experiment 700: the member-level solar responses of origin month 2026-01, timed.

    E00377_SOURCE=control E00377_ARM=a0 \
    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:ml-experiments/00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1:ml-experiments/00184-dp-ens-daylight-calibration \
        uv run --no-sync python ml-experiments/00700-*/e00700_gam_members.py --sites D5

For each daily run of the month this passes the 51 ensemble members of the run through the site's
2026-01 T-all solar term (experiment 184's `compute_run`, as experiment 378 does) and stores the
response at the stage 1 rows of lead up to 168 hours (`low_`) and above 168 and up to 240 hours
(long). The month's runs replace the same runs in the sandbox copy of experiment 378's member files.
Writes `timing/gam_members_<source>_<sites>.json`.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import e00378_common as common  # ty: ignore[unresolved-import]
import e00378_members as mem  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 2, 1, tzinfo=UTC)
OUT = Path(os.environ["DATA_PATH_INTERNAL"]).resolve().parent / "timing"


def one(*, site: str, low: bool, context, fits) -> tuple[int, float, float]:  # noqa: ANN001
    """Compute and merge one band's responses; return the number of rows, the maximum difference
    from the stored responses, and the wall time."""
    t0 = time.monotonic()
    rows = mem.stage_one(site=site, period="scored").filter(
        (pl.col("power_fcst_init_time") >= START) & (pl.col("power_fcst_init_time") < END)
    )
    if low:
        rows = rows.filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
    else:
        rows = rows.filter(
            (pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
            & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        )
    pieces = []
    for run_index, part in rows.group_by("run", maintain_order=True):
        full = mem.m184.compute_run(
            cache=context.cache, setup=context.setup, fits=fits, index=int(run_index[0])
        )
        start = int(full["window_start"][0])
        position = (
            np.searchsorted(
                context.cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
            )
            - start
        )
        pieces.append(
            {
                "run": part["run"].to_numpy(),
                "valid_time": part["valid_time"].dt.replace_time_zone(None).to_numpy(),
                "delta_m": full["delta_m"][position].astype(np.float32),
                "n_members": full["n_members"][position],
            }
        )
    new = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    target = common.MEMBERS_DIR / f"{'low_' if low else ''}{site}_scored.npz"
    with np.load(target) as stored:
        old = {k: stored[k] for k in stored.files}
    runs = set(new["run"].tolist())
    keep = ~np.isin(old["run"], list(runs))
    in_old = ~keep
    # Compare with the stored rows of the same runs (same order of runs and valid times).
    same = (
        float(
            np.nanmax(
                np.abs(
                    old["delta_m"][in_old].astype(np.float64) - new["delta_m"].astype(np.float64)
                )
            )
        )
        if in_old.sum() == len(new["run"])
        else float("nan")
    )
    merged = {k: np.concatenate([old[k][keep], new[k]]) for k in new}
    np.savez(target.with_name(target.name + ".tmp.npz"), **merged)  # ty: ignore[invalid-argument-type]
    target.with_name(target.name + ".tmp.npz").rename(target)
    return len(new["run"]), same, time.monotonic() - t0


def main() -> None:
    """Compute the month at each site."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = {}
    for site in args.sites:
        context = mem.run377.load_site(site=site)
        fits = mem.tall377.load_fits(cadence="monthly", site=site)
        c0 = time.process_time()
        n_low, d_low, s_low = one(site=site, low=True, context=context, fits=fits)
        n_long, d_long, s_long = one(site=site, low=False, context=context, fits=fits)
        out[site] = {
            "rows_low": n_low,
            "rows_long": n_long,
            "max_abs_diff_low": d_low,
            "max_abs_diff_long": d_long,
            "wall_s_low": s_low,
            "wall_s_long": s_long,
            "cpu_s_total": time.process_time() - c0,
        }
        print(site, out[site], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gam_members_{mem.c377.SOURCE}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
