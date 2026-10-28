# ruff: noqa: E501
"""Member-level solar responses from 12 to 240 hours at all Primaries, through experiment 377's code.

Experiment 184's `compute_run` passes the 51 ensemble members through a site's T-all solar term.
Experiment 319 ran it for six Primaries on experiment 317's chain. Here the same function runs
against experiment 377's setup, calibration, plant, and monthly T-all fits, by registering 377's
modules under 184's module names before 184's members module is imported. It stores `delta_m` at
the stage-1 rows of experiment 377 with a lead above 168 hours and up to 240 hours, and with
`--gate` it also stores rows of 168 hours or less for two runs, to compare with experiment 184's
stored `delta_m` at the signed-MW sites.

One invocation handles the sites of one experiment 377 source, because that source is read at
import time from `E00377_SOURCE`:

    PYTHONPATH=<folders of experiments 184 and 377> E00377_SOURCE=control OMP_NUM_THREADS=1 \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00378-*/e00378_members.py --sites D5 D9 [--gate]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import e00377_common as c377  # ty: ignore[unresolved-import]
import e00377_ens  # ty: ignore[unresolved-import]
import e00377_run as run377  # ty: ignore[unresolved-import]
import e00377_tall as tall377  # ty: ignore[unresolved-import]

sys.modules["e00184_common"] = c377
sys.modules["e00184_ens"] = e00377_ens
sys.modules["e00184_tall"] = tall377

import e00184_members as m184  # noqa: E402  # ty: ignore[unresolved-import]
import e00378_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

PERIODS = ("tuning", "scored")


def stage_one(*, site: str, period: str) -> pl.DataFrame:
    """Return experiment 377's stage-1 rows of a site and period (empty when the file is absent)."""
    path = (
        common.E377_WORK
        / common.SOURCE_OF_SITE[site]
        / "arms"
        / "a0"
        / "dp_preds"
        / f"a2_{period}_{site}.parquet"
    )
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def job(arguments: tuple[str, str, bool]) -> str:
    """Compute one site and period's member responses, or run the gate on two runs."""
    site, period, gate = arguments
    torch.set_num_threads(1)
    started = time.monotonic()
    context = run377.load_site(site=site)
    cache, setup = context.cache, context.setup
    fits = tall377.load_fits(cadence="monthly", site=site)
    rows = stage_one(site=site, period=period)
    if rows.is_empty():
        return f"{site} {period}: no stage-1 file"
    if gate:
        rows = rows.filter(pl.col("lead_hours") <= common.STORED_HORIZON_HOURS)
        rows = rows.filter(pl.col("run").is_in(rows["run"].unique().sort().to_list()[-2:]))
    else:
        rows = rows.filter(
            (pl.col("lead_hours") > common.STORED_HORIZON_HOURS)
            & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        )
    pieces = []
    for run_index, part in rows.group_by("run", maintain_order=True):
        full = m184.compute_run(cache=cache, setup=setup, fits=fits, index=int(run_index[0]))
        start = int(full["window_start"][0])
        position = (
            np.searchsorted(cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy())
            - start
        )
        if position.min() < 0 or position.max() >= len(full["base"]):
            msg = f"{site}: a valid time lies outside the run's window"
            raise ValueError(msg)
        pieces.append(
            {
                "run": part["run"].to_numpy(),
                "valid_time": part["valid_time"].dt.replace_time_zone(None).to_numpy(),
                "delta_m": full["delta_m"][position].astype(np.float32),
                "n_members": full["n_members"][position],
            }
        )
    out = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    tag = "gate_" if gate else ""
    target = common.MEMBERS_DIR / f"{tag}{site}_{period}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **out)  # ty: ignore[invalid-argument-type]
    temporary.rename(target)
    return f"{site} {period}: {len(out['run']):,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Compute every requested site and period in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    wrong = [s for s in args.sites if common.SOURCE_OF_SITE[s] != c377.SOURCE]
    if wrong:
        msg = f"{wrong} are not in experiment 377's source {c377.SOURCE!r}; set E00377_SOURCE"
        raise ValueError(msg)
    periods = ("scored",) if args.gate else PERIODS
    jobs = [(s, p, args.gate) for p in periods for s in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
