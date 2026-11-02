"""Fit experiment 479's anomaly regression: real capacities, known answer, and placebos.

For each Primary, the script writes `work/real_<site>.parquet` (capacity percentiles by
season-year with and without the cloud terms, and the irradiance-shuffle placebo) and, at the
signed-MW Primaries, `work/known_<site>.parquet` (recovery of injected farms in the easy world,
two cloud-load worlds, a load with no farm, and the profile-aware placebo).

    PYTHONPATH=<folders of 269, 129, 474> OMP_NUM_THREADS=1 \
        systemd-run --user --scope -p MemoryMax=6G -p OOMPolicy=continue \
        uv run --no-sync python e00479_fit.py --sites D10 D11
"""

import argparse
from typing import Final

import e00269_common as c269  # ty: ignore[unresolved-import]
import e00474_common as c474  # ty: ignore[unresolved-import]
import e00479_anomaly as anomaly
import e00479_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

PERCENTILES: Final[tuple[int, ...]] = (5, 10, 50, 90, 95)
CLOUD_LOAD_MW: Final[float] = 2.0
"""The injected capacity that sets the mean of the cloud-following load (experiment 474)."""


def cloud_load_mw(*, cache: FeatureCache, prepared: anomaly.Prepared, shape: str) -> np.ndarray:
    """Return the cloud-following load in megawatts on the grid.

    `linear` is experiment 129's load (proportional to the day's mean cloud response). `squared`
    follows the square of the day-mean cloud response, scaled to the same mean over daylight rows,
    so the day-mean cloud term of the estimator matches it only approximately.
    """
    points = c474.points_from_mw(cache=cache, site=prepared.site, mw=CLOUD_LOAD_MW)
    linear = c474.cloud_load(cache=cache, site=prepared.site, points=points)
    load = linear * prepared.eff_mw / 100.0
    if shape == "linear":
        return load
    day_mean = prepared.cloud_columns[:, 0]
    daylight = day_mean > 0
    squared = np.where(daylight, day_mean**2, 0.0)
    scale = load[daylight].mean() / squared[daylight].mean()
    return squared * scale


def percentile_rows(*, values: np.ndarray) -> dict[str, float]:
    """Return the point value (draw 0) and the percentiles over the bootstrap draws."""
    out = {"point": float(values[0])}
    for q in PERCENTILES:
        out[f"p{q:02d}"] = float(np.percentile(values[1:], q))
    return out


def real_rows(*, cache: FeatureCache, prepared: anomaly.Prepared, seed: int) -> list[dict]:
    """Return the real-data capacity rows and the irradiance-shuffle placebo rows."""
    rows: list[dict] = []
    signed = prepared.site in common.SIGNED_SITES
    for cloud in (False, True):
        design = anomaly.build_design(prepared=prepared, mask=prepared.usable, cloud_terms=cloud)
        gram, rhs = anomaly.monthly_normal_equations(design=design)
        counts = anomaly.month_counts(
            months=design.months, n_resamples=anomaly.N_BOOTSTRAP, seed=seed
        )
        paths = anomaly.capacity_paths(
            coef=anomaly.solve(gram=gram, rhs=rhs, counts=counts), design=design
        )
        shrunk = anomaly.shrink_rises_only(paths=paths)
        for j, year in enumerate(design.years):
            for prior, source in (("none", paths), ("rises_only", shrunk)):
                rows.append(
                    {
                        "site": prepared.site,
                        "cloud": cloud,
                        "kind": "real",
                        "prior": prior,
                        "season_year": int(year),
                        "shuffle": 0,
                        **percentile_rows(values=source[:, j, 0]),
                    }
                )
        if not signed:
            continue
        point = paths[0, :, 0]
        for k in range(common.N_SHUFFLES):
            shuffled = anomaly.shuffle_within_month(
                cache=cache, series=prepared.shape, seed=500 + 13 * k
            )
            shuffled = np.where(np.isfinite(prepared.shape), shuffled, np.nan)
            cloud_shuffled = np.stack(
                [
                    anomaly.shuffle_within_month(
                        cache=cache, series=prepared.cloud_columns[:, c], seed=500 + 13 * k
                    )
                    for c in range(prepared.cloud_columns.shape[1])
                ],
                axis=1,
            )
            d2 = anomaly.build_design(
                prepared=prepared,
                mask=prepared.usable,
                cloud_terms=cloud,
                shape_override=shuffled,
                cloud_override=np.nan_to_num(cloud_shuffled, nan=0.0),
            )
            g2, r2 = anomaly.monthly_normal_equations(design=d2)
            one = np.ones((1, len(d2.months)))
            p2 = anomaly.capacity_paths(coef=anomaly.solve(gram=g2, rhs=r2, counts=one), design=d2)
            for j, year in enumerate(d2.years):
                real_j = np.flatnonzero(design.years == year)
                if not len(real_j):
                    continue
                rows.append(
                    {
                        "site": prepared.site,
                        "cloud": cloud,
                        "kind": "irradiance_shuffle",
                        "prior": "none",
                        "season_year": int(year),
                        "shuffle": k,
                        "point": float(p2[0, j, 0]),
                        "real_point": float(point[real_j[0]]),
                    }
                )
    return rows


def known_rows(*, cache: FeatureCache, prepared: anomaly.Prepared, seed: int) -> list[dict]:
    """Return recovery rows for every injected farm, world, and estimator at a signed Primary."""
    rows: list[dict] = []
    loads = {
        s: cloud_load_mw(cache=cache, prepared=prepared, shape=s) for s in ("linear", "squared")
    }
    for farm in common.FARMS:
        norm = c269.farm_norm(cache=cache, farm=farm)
        mask = prepared.usable & np.isfinite(norm)
        profile = np.where(
            np.isfinite(norm), c474.day_of_year_profile(cache=cache, series=norm), np.nan
        )
        zero_norm = np.nan_to_num(norm, nan=0.0)
        specs: list[tuple[str, float, int, np.ndarray]] = []
        specs.extend(("easy", mw, 0, -mw * zero_norm) for mw in common.INJECTION_MW)
        mw0 = common.INJECTION_MW[0]
        specs.append(("cloud_linear", mw0, 0, -mw0 * zero_norm + loads["linear"]))
        specs.append(("cloud_squared", mw0, 0, -mw0 * zero_norm + loads["squared"]))
        specs.append(("load_only_linear", mw0, 0, loads["linear"]))
        specs.append(("load_only_squared", mw0, 0, loads["squared"]))
        specs.append(("profile_only", mw0, 0, -mw0 * np.nan_to_num(profile, nan=0.0)))
        for k in range(common.N_SHUFFLES):
            placebo = c474.shuffled_placebo(
                cache=cache, norm=norm, profile=profile, seed=1000 * k + 7
            )
            specs.append(("placebo", mw0, k, -mw0 * np.nan_to_num(placebo, nan=0.0)))
        extra = np.stack([s[3] for s in specs])
        for cloud in (False, True):
            design = anomaly.build_design(prepared=prepared, mask=mask, cloud_terms=cloud)
            col = anomaly.readout_column(design=design, season_year=anomaly.READ_OUT_SEASON_YEAR)
            if col < 0:
                print(f"{prepared.site} farm {farm} cloud={cloud}: no read-out column", flush=True)
                rows.append(
                    {"site": prepared.site, "farm": farm, "cloud": cloud, "kind": "missing"}
                )
                continue
            gram, rhs = anomaly.monthly_normal_equations(design=design, extra_targets=extra)
            counts = anomaly.month_counts(
                months=design.months,
                n_resamples=anomaly.N_BOOTSTRAP,
                seed=seed + 31 * common.FARMS.index(farm) + cloud,
            )
            paths = anomaly.capacity_paths(
                coef=anomaly.solve(gram=gram, rhs=rhs, counts=counts), design=design
            )
            factor = anomaly.recovery_factor(prepared=prepared, design=design, norm=norm)
            base = paths[:, col, 0]
            for t, (kind, mw, k, _series) in enumerate(specs, start=1):
                delta = paths[:, col, t] - base
                recovery = delta / mw * factor
                rows.append(
                    {
                        "site": prepared.site,
                        "farm": farm,
                        "cloud": cloud,
                        "kind": kind,
                        "mw": mw,
                        "shuffle": k,
                        "factor": factor,
                        "n_rows": len(design.rows),
                        **percentile_rows(values=recovery),
                    }
                )
    return rows


def main() -> None:
    """Write the per-Primary result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    parser.add_argument("--no-known", action="store_true")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = c269.load_cache()
    for site in arguments.sites:
        prepared = anomaly.prepare(cache=cache, site=site)
        seed = 4000 + 17 * common.ALL_SITES.index(site)
        pl.DataFrame(real_rows(cache=cache, prepared=prepared, seed=seed)).write_parquet(
            common.WORK_DIR / f"real_{site}.parquet"
        )
        print(site, "real done", flush=True)
        if site in common.SIGNED_SITES and not arguments.no_known:
            pl.DataFrame(known_rows(cache=cache, prepared=prepared, seed=seed)).write_parquet(
                common.WORK_DIR / f"known_{site}.parquet"
            )
            print(site, "known done", flush=True)


if __name__ == "__main__":
    main()
