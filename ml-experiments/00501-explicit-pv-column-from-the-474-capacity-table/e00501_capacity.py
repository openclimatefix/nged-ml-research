"""Per-fold, out-of-fold capacity of the six signed-MW Primaries, by experiment 474's method.

For each fold the script sets experiment 269's cut to the fold's refit instant (00:00 UTC on the
first day of the test month), refits experiment 474's convex fit (rung R4, cloud terms, point fit
without the bootstrap) on rows before the cut, and reads the cumulative alternating-current
capacity at the season-year that contains the cut. The farm response tables that build rung R4's
shape are refitted on farm output before the same cut. The capacity is divided by the median
known-answer recovery that experiment 474's table stores for the Primary. `--check-full` refits
with experiment 269's own cut (2026-07-01) and compares the result with the stored `point_ac`.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
export PYTHONPATH=<folders of experiments 269, 129 and 474>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=10G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00501-*/e00501_capacity.py --sites D10
```
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00269_common as c269  # ty: ignore[unresolved-import]
import e00269_fit as f269  # ty: ignore[unresolved-import]
import e00474_common as common  # ty: ignore[unresolved-import]
import e00474_fit as fit  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, prepare_fit
from studies.pv_plant_fit import fit_response_table
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00501-explicit-pv-column-from-the-474-capacity-table" / "work"
)
FOLDS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SIGNED_SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
RUNG: Final[str] = "R4_pooled_table_era5_cams_mean"
TABLE_474: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
    / "nged_table.parquet"
)


def refit_of(*, month: str) -> datetime:
    """Return 00:00 UTC on the first day of the test month."""
    return datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)


def season_year_of(*, cut: datetime) -> int:
    """Return the July-named season-year that contains the cut's last training day."""
    return cut.year if cut.month >= 8 else cut.year - 1


def learned_response_before_cut(*, cache: FeatureCache, site: str, farm: str, cut: datetime):  # noqa: ANN201
    """Return experiment 269's learned response of a farm, fitted on half-hours before `cut`."""
    unit = c269.reference_unit_shape(cache=cache, site=site, reference=farm)
    ghi, elevation = c269.farm_sky(cache=cache, farm=farm)
    before = cache.end_times <= np.datetime64(cut.replace(tzinfo=None), "us")
    rows = (
        np.isfinite(unit)
        & np.isfinite(ghi)
        & (elevation > c269.RESPONSE_MIN_ELEVATION_DEG)
        & before
    )
    return fit_response_table(
        grid=c269.RESPONSE_GRID,
        irradiance_w_m2=ghi[rows],
        elevation_deg=elevation[rows],
        output=unit[rows],
        prior_output=np.zeros(int(rows.sum())),
        penalty=0.0,
    )


def shape_before_cut(*, cache: FeatureCache, site: str, cut: datetime) -> np.ndarray:
    """Return rung R4's unit shape with the farm response tables fitted before the cut."""
    features = cache.sites[site]
    era5 = common.era5_ghi(cache=cache, site=site)
    sky = (era5 + features.ghi_w_m2) / 2
    shapes = [
        learned_response_before_cut(cache=cache, site=site, farm=farm, cut=cut).predict(
            irradiance_w_m2=np.nan_to_num(sky), elevation_deg=features.elevation_deg
        )
        for farm in c269.ALL_FARMS
    ]
    shape = np.mean(shapes, axis=0)
    shape[features.elevation_deg <= c269.RESPONSE_MIN_ELEVATION_DEG] = 0.0
    shape[~np.isfinite(sky)] = np.nan
    return shape


def capacity_at(*, cache: FeatureCache, site: str, cut: datetime) -> tuple[float, int]:
    """Return the point-fit AC capacity in MW at the cut's season-year, and the rows trained on."""
    c269.CUT = cut
    reading = cache.sites[site].y
    shape = shape_before_cut(cache=cache, site=site, cut=cut)
    probe = common.with_shape(cache=cache, site=site, shape=shape, y=reading)
    _, index = prepare_fit(
        cache=probe,
        site=site,
        spec=GamSpec(ridge=c269.RIDGE, capacity="fitted"),
        cut=cut,
        window=c269.TRAINING_WINDOW,
        season_years=c269.SEASON_YEARS,
    )
    rows = f269.row_months(cache=probe, rows=index)
    months = np.arange(rows.min(), rows.max() + 1)
    counts = f269.month_block_counts(months=months, n_resamples=0, seed=4242)
    paths, index, _, _ = fit.solve_targets(
        cache=cache,
        site=site,
        shape=shape,
        y_rows=reading,
        targets=reading[None],
        weights_by_target=[counts],
        cloud_terms=True,
    )
    scale = cache.sites[site].table.effective_capacity_mw / 100.0
    k = min(season_year_of(cut=cut), c269.SEASON_YEARS[-1]) - c269.SEASON_YEARS[0]
    return float(paths[0][0, k] * scale), len(index)


def main() -> None:
    """Write `work/capacity_by_fold.parquet` (or `capacity_check_full.parquet`)."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SIGNED_SIX))
    parser.add_argument("--folds", nargs="+", default=list(FOLDS))
    parser.add_argument("--check-full", action="store_true")
    arguments = parser.parse_args()
    stored = pl.read_parquet(TABLE_474)
    cache = c269.load_cache()
    rows: list[dict] = []
    cuts = (
        {"full": datetime(2026, 7, 1, tzinfo=UTC)}
        if arguments.check_full
        else {m: refit_of(month=m) for m in arguments.folds}
    )
    for name, cut in cuts.items():
        for site in arguments.sites:
            point, n_rows = capacity_at(cache=cache, site=site, cut=cut)
            divisor = float(
                stored.filter(pl.col("site") == site, pl.col("season_year") == 2025)[
                    "recovery_ratio_median"
                ][0]
            )
            row = {
                "fold": name,
                "site": site,
                "raw_ac_mw": point,
                "divisor": divisor,
                "ac_mw": point / divisor,
                "n_rows": n_rows,
            }
            if arguments.check_full:
                row["stored_point_ac"] = float(
                    stored.filter(pl.col("site") == site, pl.col("season_year") == 2025)[
                        "point_ac"
                    ][0]
                )
            rows.append(row)
            print(row, flush=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    target = "capacity_check_full" if arguments.check_full else "capacity_by_fold"
    pl.DataFrame(rows).write_parquet(WORK_DIR / f"{target}_{'_'.join(arguments.sites)}.parquet")


if __name__ == "__main__":
    main()
