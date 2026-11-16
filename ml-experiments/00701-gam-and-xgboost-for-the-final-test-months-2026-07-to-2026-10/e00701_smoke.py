# ruff: noqa: E501
"""Experiment 701: smoke tests that the refit reproduces experiment 87's stored predictions.

Two tests, each one refit of one configuration for a few series and runs, written to
`work/smoke_anchor_preds` and compared with experiment 87's stored files:

- `monthly`: the n1h monthly refit of 2026-06-01 (500 rounds, as in the run) for series D2 (MVA
  units, no offset), D9 (an offset series, with and without the offset), and D5, forecasting the
  runs 2026-06-02 to 2026-06-04 from the frames that `e00701_build_inputs.py frames-check` rebuilt.
- `frozen`: the frozen model (ST), refitted once on rows to 2025-06-30 with the same settings and
  seed, forecasting the runs 2025-07-01 to 2025-07-03.

The difference is printed as fractions of capacity (the stored quantiles are fractions of the
series' effective capacity). The XGBoost GPU fit is not bit-reproducible, so a nonzero difference is
expected; experiment 700 found up to about 0.04 of capacity at the largest, and a mean near 0.0004.

    PYTHONPATH=<folders of experiments 87, 33, 48, 701> \
        uv run --no-sync python ml-experiments/00701-*/e00701_smoke.py monthly|frozen
"""

import argparse
import time
from datetime import UTC, datetime

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fit  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00701_common as mine
import numpy as np
import polars as pl

SMOKE_DIR = mine.WORK / "smoke_anchor_preds"
NTHREAD = 4
LEVELS = common.LEVEL_COLUMNS


def compare(*, config: str, refit: datetime, sites: list[str], runs: list[datetime]) -> None:
    """Print the largest and mean absolute differences from the stored files, per series and overall."""
    origins = [common.origin_of(run=run) for run in runs]
    gaps_all: list[np.ndarray] = []
    for site in sites:
        new = pl.read_parquet(
            fit.job_path(root=SMOKE_DIR, config=config, site=site, seed=0, refit=refit)
        )
        old = pl.read_parquet(
            fit.job_path(
                root=mine.E87_WORK / "anchor_preds", config=config, site=site, seed=0, refit=refit
            )
        ).filter(pl.col("power_fcst_init_time").is_in(origins))
        keys = list(common.KEYS)
        joined = new.join(old, on=keys, suffix="_old", validate="1:1")
        if joined.height != new.height or joined.height != old.height:
            msg = f"{config} {site}: {joined.height} joined rows, {new.height} new, {old.height} stored"
            raise ValueError(msg)
        gap = np.abs(
            joined.select(LEVELS).to_numpy().astype(np.float64)
            - joined.select([f"{q}_old" for q in LEVELS]).to_numpy().astype(np.float64)
        )
        gaps_all.append(gap)
        median = gap[:, common.MEDIAN]
        print(
            f"  {config} {site}: {joined.height:,} rows; all 11 levels: max {gap.max():.4f}, mean {gap.mean():.5f}; "
            f"median level: max {median.max():.4f}, mean {median.mean():.5f} (fractions of capacity)",
            flush=True,
        )
    stacked = np.concatenate(gaps_all)
    print(
        f"{config}: overall max {stacked.max():.4f}, mean {stacked.mean():.5f}; median level max "
        f"{stacked[:, common.MEDIAN].max():.4f}, mean {stacked[:, common.MEDIAN].mean():.5f}",
        flush=True,
    )


def main() -> None:
    """Run one smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", choices=("monthly", "frozen"))
    args = parser.parse_args()
    started = time.monotonic()
    if args.test == "monthly":
        refit = datetime(2026, 6, 1, tzinfo=UTC)
        runs = [datetime(2026, 6, day, tzinfo=UTC) for day in (2, 3, 4)]
        plain = ["D2", "D5", "D9"]
        for offset, sites in ((False, plain), (True, ["D9"])):
            for line in fit.run_job(
                config_name="n1h",
                refit=refit,
                origins=[common.origin_of(run=r) for r in runs],
                sites=sites,
                offset=offset,
                out_root=SMOKE_DIR,
                rounds=common.N_ROUNDS,
                nthread=NTHREAD,
                seed=0,
                workers=1,
                log_hours=False,
            ):
                print(line, flush=True)
        compare(config="n1h", refit=refit, sites=plain, runs=runs)
        compare(config="n1h_o", refit=refit, sites=["D9"], runs=runs)
        compare(config="n1h_oheld", refit=refit, sites=["D9"], runs=runs)
    else:
        refit = regimes.FROZEN_REFIT
        runs = [datetime(2025, 7, day, tzinfo=UTC) for day in (1, 2, 3)]
        plain = ["D2", "D9"]
        for offset, sites in ((False, plain), (True, ["D9"])):
            for line in fit.run_job(
                config_name="frozen",
                refit=refit,
                origins=[common.origin_of(run=r) for r in runs],
                sites=sites,
                offset=offset,
                out_root=SMOKE_DIR,
                rounds=common.N_ROUNDS,
                nthread=NTHREAD,
                seed=0,
                workers=1,
                log_hours=False,
            ):
                print(line, flush=True)
        compare(config="frozen", refit=refit, sites=plain, runs=runs)
        compare(config="frozen_o", refit=refit, sites=["D9"], runs=runs)
    print(f"{args.test}: {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
