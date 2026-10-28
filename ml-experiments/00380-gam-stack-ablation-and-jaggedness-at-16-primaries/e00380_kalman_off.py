"""Stage 1 of the experiment 377 stack with the Kalman filter off, at the named Primaries.

Reads experiment 377's stored monthly fits, ensemble stores, and feature cache (read only), and
forecasts every run with no filter correction (`trace=None`). The spread is fitted on the tuning
months from the same unfiltered forecasts. Output: `a2_<period>_<site>.parquet` under
`<DATA>/00380-.../work/kalman_off/<source>/dp_preds/`, in the layout of experiment 377's stage 1.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility \
    E00377_SOURCE=control E00377_ARM=a0 uv run --no-sync python \
        ml-experiments/00380-*/e00380_kalman_off.py --sites D5
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_ens as ens  # ty: ignore[unresolved-import]
import e00377_run as run  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00380-gam-stack-ablation-and-jaggedness-at-16-primaries"
    / "work"
    / "kalman_off"
    / common.SOURCE
    / "dp_preds"
)


class UnfilteredForecaster(run.ArmForecaster):
    """Experiment 377's stage 1 forecaster with no filter correction."""

    def __init__(self, *, arm: str, context: run.SiteContext) -> None:
        """Fit the spread on the tuning months of the unfiltered forecasts."""
        self.arm = arm
        self.context = context
        self.fits = run.load_arm_fits(arm=arm, site=context.site)
        self.run = SimpleNamespace(trace=None)
        first = self.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        self.tuning = [
            i
            for i in context.setup.indices(periods={"tuning"})
            if context.setup.origins[i] >= first
        ]
        if self.tuning:
            tuning_frame = self._frame(runs=self.tuning, spread=ens.ZERO_SPREAD)
        else:
            tuning_frame = pl.concat(
                [pl.read_parquet(OUT_DIR / f"a2_tuning_{donor}.parquet") for donor in common.DONORS]
            )
        self.spread = ens.fit_spread_from_frame(frame=tuning_frame)


def _job(site: str) -> str:
    torch.set_num_threads(1)
    started = time.monotonic()
    context = run.load_site(site=site)
    forecaster = UnfilteredForecaster(arm="a2", context=context)
    first = forecaster.fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    rows = 0
    for period in run.PERIODS:
        runs = [
            i for i in context.setup.indices(periods={period}) if context.setup.origins[i] >= first
        ]
        if not runs:
            continue
        frame = forecaster.forecast(runs=runs)
        target = OUT_DIR / f"a2_{period}_{site}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Forecast the named Primaries, donors first when D8 is among them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sites", nargs="+", required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
