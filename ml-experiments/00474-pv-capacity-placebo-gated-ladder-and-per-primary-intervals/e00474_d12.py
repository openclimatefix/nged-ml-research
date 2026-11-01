"""D12 (reverse flow) of experiment 474: month-block bootstrap of the fit through the meter.

D12 is the one MVA-only Primary where experiment 370's rule chooses the fit through the meter, with
reactive power zero. This script repeats that fit under month-block resamples (row weights) and
writes the capacity at the read-out season-year, in MW, for each resample.

    PYTHONPATH=<folders of experiments 110, 358, 366, 269 and 474> \
        uv run --no-sync python ml-experiments/00474-*/e00474_d12.py --resamples 20
"""

import argparse
import time
from dataclasses import replace

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common110  # ty: ignore[unresolved-import]
import e00269_fit as f269  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import e00474_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_gam import prepare_fit
from studies.dp_meter import fit_gam_through_meter


def main() -> None:
    """Write `work/d12_bootstrap.parquet`."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=20)
    arguments = parser.parse_args()
    cache = common110.load_cache()
    part_4 = arms.read_part_4_tuned()
    spec = replace(arms.base_specs(part_4=part_4)["b"], capacity="fitted")
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    cut = known.CUTS["scored"]
    _, index = prepare_fit(
        cache=cache,
        site="D12",
        spec=spec,
        cut=cut,
        window=window,
        season_years=common110.SEASON_YEARS,
    )
    months_of_rows = f269.row_months(cache=cache, rows=index)
    months = np.arange(months_of_rows.min(), months_of_rows.max() + 1)
    counts = f269.month_block_counts(months=months, n_resamples=arguments.resamples, seed=77)
    scale = cache.sites["D12"].table.effective_capacity_mw / 100.0
    rows = []
    for b in range(arguments.resamples + 1):
        started = time.time()
        weights = counts[b][months_of_rows - months[0]]
        meter = fit_gam_through_meter(
            cache=cache,
            site="D12",
            spec=spec,
            cut=cut,
            window=window,
            season_years=common110.SEASON_YEARS,
            qmode="zero",
            weights=weights,
        )
        capacity = float(meter.fit.capacity_points()[meter.fit.last_season_year]) * scale
        rows.append({"b": b, "capacity_mw": capacity})
        print(b, round(capacity, 3), f"{time.time() - started:.0f}s", flush=True)
        pl.DataFrame(rows).write_parquet(common.WORK_DIR / "d12_bootstrap.parquet")


if __name__ == "__main__":
    main()
