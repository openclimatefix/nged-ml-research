# ruff: noqa: D205, D209, E501, I001
"""Experiment 701, GAM side: the member-level solar responses for the origin months 2026-07 to 2026-10.

    E00377_SOURCE=control E00377_ARM=a0 \
    PYTHONPATH=<folders of 377, 378, 184, 700, 701> DATA_PATH_INTERNAL=<sandbox> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_members.py --sites D5 [--first 2026-07] [--last 2026-10] [--dry-run]

This is `e00700_gam_members.py` with the origin months as arguments. For each daily run of the
months this passes the 51 ensemble members of the run through the site's T-all solar term (experiment
184's `compute_run`, as experiment 378 does, with the monthly fits of the cut at or before the origin)
and stores the response at the stage 1 rows of lead up to 168 hours (`low_<site>_scored.npz`) and
above 168 and up to 240 hours (`<site>_scored.npz`). The months' runs replace the same runs in the
sandbox copy of experiment 378's member files, and the runs of other months are kept. Where the
months' runs are already stored, the script prints the largest difference from the stored responses
(`nan` where the stored file holds no such rows). `--dry-run` writes nothing. Run stage 1 and the
refit for the sites first.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00378_common as common  # ty: ignore[unresolved-import]
import e00378_members as mem  # ty: ignore[unresolved-import]
from e00377_run import SiteContext  # ty: ignore[unresolved-import]
from e00377_tall import TallFit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]


def month_start(text: str) -> datetime:
    """Return the first instant of a `YYYY-MM` month."""
    return datetime.strptime(text, "%Y-%m").replace(tzinfo=UTC)


def one(
    *,
    site: str,
    low: bool,
    context: SiteContext,
    fits: list[TallFit],
    first: datetime,
    end: datetime,
    dry_run: bool,
) -> tuple[int, float, float]:
    """Compute and merge one band's responses; return the number of rows, the maximum difference
    from the stored responses, and the wall time."""
    t0 = time.monotonic()
    rows = mem.stage_one(site=site, period="scored").filter(
        (pl.col("power_fcst_init_time") >= first) & (pl.col("power_fcst_init_time") < end)
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
        assert position.min() >= 0, f"{site}: a valid time precedes the run's window"
        assert position.max() < len(full["base"]), (
            f"{site}: a valid time lies outside the run's window"
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
    if not dry_run:
        assert ext.SANDBOX in target.parents, f"{target} is outside the sandbox"
        merged = {k: np.concatenate([old[k][keep], new[k]]) for k in new}
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez(temporary, **merged)  # ty: ignore[invalid-argument-type]
        temporary.rename(target)
    return len(new["run"]), same, time.monotonic() - t0


def main() -> None:
    """Compute the months at each site."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--first", default="2026-07")
    parser.add_argument("--last", default="2026-10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    first, last = month_start(args.first), month_start(args.last)
    end = last.replace(year=last.year + (last.month == 12), month=last.month % 12 + 1)
    torch.set_num_threads(1)
    out = {}
    for site in args.sites:
        context = mem.run377.load_site(site=site)
        fits = mem.tall377.load_fits(cadence="monthly", site=site)
        assert fits[-1].cut >= last, (
            f"{site}: the fit list ends at {fits[-1].cut}, run e00701_gam_refit.py first"
        )
        c0 = time.process_time()
        n_low, d_low, s_low = one(
            site=site,
            low=True,
            context=context,
            fits=fits,
            first=first,
            end=end,
            dry_run=args.dry_run,
        )
        n_long, d_long, s_long = one(
            site=site,
            low=False,
            context=context,
            fits=fits,
            first=first,
            end=end,
            dry_run=args.dry_run,
        )
        out[site] = {
            "rows_low": n_low,
            "rows_long": n_long,
            "max_abs_diff_low": d_low,
            "max_abs_diff_long": d_long,
            "wall_s_low": s_low,
            "wall_s_long": s_long,
            "cpu_s_total": time.process_time() - c0,
            "dry_run": args.dry_run,
        }
        print(site, out[site], flush=True)
    ext.TIMING.mkdir(parents=True, exist_ok=True)
    tag = "dry_" if args.dry_run else ""
    (ext.TIMING / f"gam_members_{tag}{mem.c377.SOURCE}_{'_'.join(args.sites)}.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
