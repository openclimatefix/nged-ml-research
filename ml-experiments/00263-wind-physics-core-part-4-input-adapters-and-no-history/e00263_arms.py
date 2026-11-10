# ruff: noqa: E402
"""Fit the member arms of experiment 00263 and write their forecasts.

Arms: A0r (the experiment 33 recipe refitted with the 0.01 and 0.99 levels, ERA5-trained XGBoost
mixture over the 51 ENS members), A1 (A0r trained on the ENS control member's weather), B1 (the
physics core trained on ERA5), B2 (the core trained on the ENS control member), and B8 (B1 without
the idle layer). Each member's quantiles are mixed with equal weights and given the spread factor
experiment 33 fitted month by month on earlier cases.

Output: `work/arms/<arm>_mixture.parquet` and `work/arms/<arm>_spread.parquet`, keyed by case, with
`seed` and the 11 quantile columns. A physics arm has no seed, so its rows are repeated per seed.

    uv run --no-sync python ml-experiments/00263-*/e00263_arms.py [ARM ...]
"""

import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00263_adapters as adapters
import e00263_common as common
import e00263_core as core_module
import e00263_neural as neural
import e00263_step0 as step0
import numpy as np
import polars as pl

ARMS = (
    "B10",
    "B11",
    "B12",
    "B13",
    "B1_3m",
    "B6_3m",
    "B10_3m",
    "B1_lofo",
    "B6_lofo",
    "B12_lofo",
)
STOCHASTIC_ARMS = ("B10", "B11", "B12", "B13", "B6_3m", "B10_3m", "B6_lofo", "B12_lofo")
ADAPTER_PARTS = {
    "B10": adapters.Parts(input_adapter=True),
    "B11": adapters.Parts(input_adapter=True, calibrator=True),
    "B13": adapters.Parts(gate=True),
    "B10_3m": adapters.Parts(input_adapter=True),
}
HISTORY_DAYS_SHORT = 90
"""The `_3m` arms fit on each farm's first 90 days of training rows."""
OUT_DIR = common.WORK_DIR / "arms"
N_ROUNDS = 500


def core_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, with_idle_layer: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the physics core plus its residual quantiles."""
    target = train["target"].to_numpy()
    weather = step0.weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    expected = core_module.predict_expected_power(
        core=core, weather=step0.weather_arrays(frame=members)
    )
    return residuals.quantiles(expected=expected)


def pure_network_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, seed: int
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the network with no physics (B6)."""
    centre, scale = neural.raw_input_scaling(frame=train)
    network = neural.fit_free_network(
        inputs=neural.raw_inputs(frame=train, centre=centre, scale=scale),
        target=train["target"].to_numpy(),
        seed=seed,
    )
    return neural.free_quantiles(
        network=network, inputs=neural.raw_inputs(frame=members, centre=centre, scale=scale)
    )


def first_days(*, frame: pl.DataFrame, days: int) -> pl.DataFrame:
    """Return the rows of `frame` within `days` days of its first valid time."""
    start = frame["valid_time"].min()
    return frame.filter(pl.col("valid_time") < start + pl.duration(days=days))


def fit_arm(
    *,
    arm: str,
    era5_by_farm: dict[int, pl.DataFrame],
    farm_index: int,
    members: pl.DataFrame,
    seed: int,
) -> np.ndarray:
    """Return one farm's member quantiles for one arm and seed.

    The `_lofo` arms never see the farm's own rows: they fit on the other farms and apply to it.
    """
    own = era5_by_farm[farm_index]
    others = {index: frame for index, frame in era5_by_farm.items() if index != farm_index}
    if arm in ("B10", "B11", "B13"):
        return adapters.hybrid_member_quantiles(
            parts=ADAPTER_PARTS[arm], train=own, members=members, seed=seed, farm_index=None
        )
    if arm == "B12":
        return adapters.pooled_member_quantiles(
            parts=adapters.Parts(input_adapter=True, farm_label=True),
            train_by_farm=era5_by_farm,
            members=members,
            seed=seed,
            target_index=farm_index,
        )
    return fit_no_history_arm(arm=arm, own=own, others=others, members=members, seed=seed)


def fit_no_history_arm(
    *,
    arm: str,
    own: pl.DataFrame,
    others: dict[int, pl.DataFrame],
    members: pl.DataFrame,
    seed: int,
) -> np.ndarray:
    """Return one farm's member quantiles for a `_3m` (first 90 days) or `_lofo` arm."""
    short = first_days(frame=own, days=HISTORY_DAYS_SHORT)
    if arm == "B1_3m":
        return core_member_quantiles(train=short, members=members, with_idle_layer=False)
    if arm == "B6_3m":
        return pure_network_member_quantiles(train=short, members=members, seed=seed)
    if arm == "B10_3m":
        return adapters.hybrid_member_quantiles(
            parts=ADAPTER_PARTS[arm], train=short, members=members, seed=seed, farm_index=None
        )
    if arm == "B1_lofo":
        return adapters.prior_core_member_quantiles(train_by_farm=others, members=members)
    if arm == "B6_lofo":
        return adapters.pooled_network_member_quantiles(
            train_by_farm=others, members=members, seed=seed
        )
    if arm == "B12_lofo":
        return adapters.pooled_member_quantiles(
            parts=adapters.Parts(input_adapter=True, farm_label=True),
            train_by_farm=others,
            members=members,
            seed=seed,
            target_index=None,
        )
    msg = f"unknown arm {arm}"
    raise ValueError(msg)


def main() -> None:
    """Write each requested arm's mixture and spread-adjusted forecasts."""
    wanted = tuple(sys.argv[1:]) or ARMS
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    era5_train = common.training_table(sites=sites, era5=era5)
    cases = common.cases_table(sites=sites)
    members = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    era5_by_farm = {
        index: era5_train.filter(pl.col("time_series_id") == int(row["time_series_id"]))
        for index, row in enumerate(sites.iter_rows(named=True))
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in wanted:
        stochastic = arm in STOCHASTIC_ARMS
        pieces_mix, pieces_spread = [], []
        for seed in common.SEEDS if stochastic else common.SEEDS[:1]:
            mixtures, kept = [], []
            for farm_index, row in enumerate(sites.iter_rows(named=True)):
                series, farm = int(row["time_series_id"]), row["site"]
                farm_cases = cases.filter(pl.col("time_series_id") == series)
                farm_members = members.filter(pl.col("time_series_id") == series)
                one_per_case = farm_members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
                if (
                    farm_members.height != farm_cases.height * common.MEMBERS
                    or not one_per_case.equals(farm_cases.select(common.KEY_COLUMNS))
                ):
                    msg = f"{farm}: the members do not line up with the cases"
                    raise ValueError(msg)
                if (
                    farm_members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
                    != np.arange(common.MEMBERS)
                ).any():
                    msg = f"{farm}: a case does not hold members 0 to 50 in order"
                    raise ValueError(msg)
                quantiles = fit_arm(
                    arm=arm,
                    era5_by_farm=era5_by_farm,
                    farm_index=farm_index,
                    members=farm_members,
                    seed=seed,
                )
                mixtures.append(
                    common.mixture_quantiles(
                        member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11)
                    )
                )
                kept.append(farm_cases)
            ordered = pl.concat(kept)
            mixture = np.concatenate(mixtures)
            spread = common.apply_spread(cases=ordered, mixture=mixture.astype(np.float64))
            keys = ordered.select(*common.KEY_COLUMNS)
            for store, values in ((pieces_mix, mixture), (pieces_spread, spread)):
                frame = keys.hstack(
                    pl.DataFrame(values.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                )
                store.extend(
                    frame.with_columns(seed=pl.lit(s, dtype=pl.Int32))
                    for s in (common.SEEDS if not stochastic else (seed,))
                )
            print(f"{arm} seed {seed} done", flush=True)
        for name, pieces in (("mixture", pieces_mix), ("spread", pieces_spread)):
            target = OUT_DIR / f"{arm}_{name}.parquet"
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    main()
