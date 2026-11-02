"""Experiment 507: the ERA5 and CAMS lab lagless table, with the mixture and a scored-month module.

Experiment 449's stored ERA5 tables hold only the 5%, 50%, and 95% quantiles and only the
cross-fitted half-hours before each refit. The lab correction needs the 9-number mixture at every
row and a forecast for the scored month. This script reuses experiment 449's ERA5 inputs and
experiment 404's cross-fit blocks, and writes a table in experiment 404's layout (`source` is
`crossfit` or `further`), so experiment 404's and 472's readers work unchanged. Weather is ERA5 and
CAMS everywhere. Nothing at or after the refit enters any fit.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_lagless.py build --month 2026-01
```
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_lagless as lag  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00449_lagless as era5  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00507-history-encoder-first-test-oracle-repair-and-total-variation"
    / "work"
)  # fmt: skip
TABLE_DIR: Final[Path] = WORK_DIR / "lagless_era5_mixture"
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")


def table_path(*, month: str, seed: int) -> Path:
    """Return where a scored month's lab table is stored."""
    return TABLE_DIR / f"month={month}" / f"seed={seed}.parquet"


def lab_table(*, month: str, seed: int, device: str, clean: np.ndarray) -> pl.DataFrame:
    """Return the cross-fitted ERA5 table plus the `further` module's scored-month forecasts."""
    refit = prep.refit_of(month=month)
    end = int(common.grid_index(times=refit))
    start = int(common.grid_index(times=prep418.PRETRAIN_START))
    train = era5.build_pool(lo=start, hi=end, clean=clean)
    after = era5.build_pool(lo=end, hi=end + lag.FURTHER_STEPS, clean=clean)
    assert train.grid.max() < end <= after.grid.min(), "the table must stop before the refit"
    observed = np.isfinite(train.y)
    block = lag.block_of(grid=train.grid, lo=start, hi=end, folds=lag.FOLDS)
    weight = torch.ones(len(train.grid), dtype=torch.float32, device=device)

    def run(*, fit_mask: np.ndarray, pool: lag.Pool, rows: np.ndarray, source: str) -> pl.DataFrame:
        scaler = lag.Scaler(
            x=train.x[fit_mask], site=train.site[fit_mask], y=train.y[fit_mask], n_primaries=16
        )
        module = lag.fit_module(
            x=torch.tensor(scaler.inputs(x=train.x[fit_mask]), device=device),
            z=torch.tensor(
                scaler.target(y=train.y[fit_mask], site=train.site[fit_mask]), device=device
            ),
            site=torch.tensor(train.site[fit_mask], device=device),
            weight=weight[torch.tensor(fit_mask, device=device)],
            seed=seed,
        )
        raw, quant = lag.predict(
            module=module,
            x=torch.tensor(scaler.inputs(x=pool.x[rows]), device=device),
            site=torch.tensor(pool.site[rows], device=device),
        )
        site = pool.site[rows]
        q = quant * scaler.y_std[site][:, None] + scaler.y_mean[site][:, None]
        return pl.DataFrame(
            {
                "primary": site.astype(np.int16),
                "grid": pool.grid[rows].astype(np.int32),
                "source": np.full(len(site), source),
                **{f"raw_{i}": raw[:, i] for i in range(9)},
                "q05": q[:, 0].astype(np.float32),
                "q50": q[:, 1].astype(np.float32),
                "q95": q[:, 2].astype(np.float32),
                "y_mean": scaler.y_mean[site].astype(np.float32),
                "y_std": scaler.y_std[site].astype(np.float32),
            }
        )

    parts = [
        run(fit_mask=observed & (block != k), pool=train, rows=np.flatnonzero(block == k),
            source="crossfit")
        for k in range(lag.FOLDS)
    ]  # fmt: skip
    parts.append(
        run(fit_mask=observed, pool=after, rows=np.arange(len(after.grid)), source="further")
    )
    return pl.concat(parts)


def main() -> None:
    """Build the lab table of one or more scored months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--month", nargs="*", default=list(SCREEN_MONTHS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    stack, _ = data.load_stack(include_eval=False)
    for month in args.month:
        target = table_path(month=month, seed=args.seed)
        if target.exists():
            print(f"{month}: exists")
            continue
        started = time.monotonic()
        table = lab_table(month=month, seed=args.seed, device=args.device, clean=stack[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        table.write_parquet(temporary)
        temporary.rename(target)
        print(f"{month}: {table.height:,} rows, {time.monotonic() - started:.0f} s -> {target}",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
