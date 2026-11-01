"""Stage 1 of experiment 474: the per-Primary capacity table with intervals.

For each signed-MW Primary with a usable history, the script fits experiment 129's convex model to
the real reading with the chosen rung's solar shape, runs a month-block bootstrap, divides each
resample's capacity by a recovery drawn from that Primary's known-answer injections (experiment
474's ladder file), and writes the 10th, 50th and 90th percentiles of the alternating-current (AC)
and direct-current (DC) capacity per season-year. DC capacity is AC capacity times a DC-to-AC ratio
drawn from a triangular distribution on 1.1 to 1.5 centred at 1.25 (experiment 73: the ratio is not
identified, so it is an assumption). The eight MVA-only Primaries reuse experiment 370's stored
census fits.

    PYTHONPATH=<folders of experiments 269, 129 and 474> \
        uv run --no-sync python ml-experiments/00474-*/e00474_table.py --rung C0_pooled_table
"""

import argparse
from typing import Final

import e00269_common as c269  # ty: ignore[unresolved-import]
import e00269_fit as f269  # ty: ignore[unresolved-import]
import e00474_common as common
import e00474_fit as fit
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, prepare_fit
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

N_DRAWS: Final[int] = 200
MVA_ONLY: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
CENSUS_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00370-mva-fold-pv-capacity-and-eight-primaries" / "work" / "census"
)
STATUS: Final[dict[str, str]] = {
    "D5": "export limit: the clip-free solar term does not represent it",
    "D9": "pinned capacity too high in the fit through the meter",
    "D12": "reverse flow",
    "D15": "generator case; history only from 2025-03-01; report-only",
    "D16": "pinned capacity too high in the fit through the meter",
    "D18": "history only from 2025-03-01; report-only",
}


def divisor_draws(*, ladder: pl.DataFrame, rung: str, site: str) -> np.ndarray:
    """Return the known-answer capacity ratios of the rung: the Primary's own, else all."""
    real = ladder.filter(
        pl.col("rung") == rung,
        pl.col("kind") == "real",
        pl.col("stat") == "ratio",
        pl.col("b") == 0,
    )
    own = real.filter(pl.col("site") == site)["value"].to_numpy()
    return own if len(own) else real["value"].to_numpy()


def fitted_paths(
    *, cache: FeatureCache, site: str, shape: np.ndarray, n_resamples: int, cloud_terms: bool
) -> tuple[np.ndarray, int]:
    """Return bootstrap cumulative capacity paths (B+1, n_seasons) for the real reading."""
    reading = cache.sites[site].y
    probe = fit.common.with_shape(cache=cache, site=site, shape=shape, y=reading)
    _, index = prepare_fit(
        cache=probe,
        site=site,
        spec=GamSpec(ridge=c269.RIDGE, capacity="fitted"),
        cut=c269.CUT,
        window=c269.TRAINING_WINDOW,
        season_years=c269.SEASON_YEARS,
    )
    rows = f269.row_months(cache=probe, rows=index)
    months = np.arange(rows.min(), rows.max() + 1)
    counts = f269.month_block_counts(months=months, n_resamples=n_resamples, seed=4242)
    paths, index, _, _ = fit.solve_targets(
        cache=cache,
        site=site,
        shape=shape,
        y_rows=reading,
        targets=reading[None],
        weights_by_target=[counts],
        cloud_terms=cloud_terms,
    )
    return paths[0], len(index)


def main() -> None:
    """Write `work/table_<rung>.parquet`."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", default="C0_pooled_table")
    parser.add_argument("--cloud-terms", action="store_true")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=[
            "D5",
            "D9",
            "D10",
            "D11",
            "D14",
            "D16",
            "D18",
            "D2",
            "D3",
            "D4",
            "D8",
            "D19",
            "D21",
            "D22",
        ],
    )
    arguments = parser.parse_args()
    ladder = pl.concat(
        [pl.read_parquet(p) for p in sorted(common.WORK_DIR.glob("ladder_D*.parquet"))]
    )
    cache = c269.load_cache()
    rng = np.random.default_rng(99)
    rows: list[dict] = []
    for site in arguments.sites:
        table = cache.sites[site].table
        try:
            shape = common.rung_shape(cache=cache, site=site, rung=arguments.rung, farm=None)
            paths, n_rows = fitted_paths(
                cache=cache,
                site=site,
                shape=shape,
                n_resamples=200,
                cloud_terms=arguments.cloud_terms,
            )
        except (ValueError, IndexError) as error:
            rows.append({"site": site, "season_year": None, "status": f"fit failed: {error}"})
            continue
        ladder_rung = arguments.rung + ("+cloudterm" if arguments.cloud_terms else "")
        divisors = divisor_draws(ladder=ladder, rung=ladder_rung, site=site)
        scale = table.effective_capacity_mw / 100.0
        try:
            pinned_mw = (
                pinned_capacity(vintages=read_vintages(), table=table, cut=c269.CUT).held_points
                * scale
            )
        except ValueError, KeyError, IndexError:
            pinned_mw = None
        for k, year in enumerate(c269.SEASON_YEARS):
            ac = np.array(
                [paths[b % len(paths), k] * scale / rng.choice(divisors) for b in range(N_DRAWS)]
            )
            ratio = rng.triangular(
                common.DC_AC_LOW, common.DC_AC_CENTRE, common.DC_AC_HIGH, N_DRAWS
            )
            dc = ac * ratio
            rows.append(
                {
                    "site": site,
                    "season_year": year,
                    "ac_p10": float(np.percentile(ac, 10)),
                    "ac_p50": float(np.percentile(ac, 50)),
                    "ac_p90": float(np.percentile(ac, 90)),
                    "dc_p10": float(np.percentile(dc, 10)),
                    "dc_p50": float(np.percentile(dc, 50)),
                    "dc_p90": float(np.percentile(dc, 90)),
                    "point_ac": float(paths[0, k] * scale / np.median(divisors)),
                    "pinned_mw": pinned_mw,
                    "n_rows": n_rows,
                    "recovery_ratio_median": float(np.median(divisors)),
                    "rung": ladder_rung,
                    "status": STATUS.get(
                        site,
                        "MVA-only: fit to the apparent-power reading, reactive power assumed zero"
                        if site in MVA_ONLY
                        else "",
                    ),
                }
            )
        print(f"{site} done", flush=True)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    ladder_name = arguments.rung + ("_cloudterm" if arguments.cloud_terms else "")
    summary = pl.read_parquet(common.WORK_DIR / "rung_summary.parquet").select(
        "rung",
        "placebo_p95_max",
        coverage_easy=pl.col("coverage_boot_plus_recovery_spread"),
        coverage_cloud_load=pl.col("cloud_cov_boot_plus_spread"),
    )
    rows = (
        pl.DataFrame(rows, infer_schema_length=None).join(summary, on="rung", how="left").to_dicts()
    )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        common.WORK_DIR / f"table_{ladder_name}.parquet"
    )


if __name__ == "__main__":
    main()
