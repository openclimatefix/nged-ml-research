"""Pass the daily ensemble members through wind default 2's frozen plant for the new runs.

For each farm and each daily run in `wind_members/` (2025-06-17 to 2026-06-30), the script runs the
51 members through experiment 347's Stage 3 plant (arm y2) and stores one row per case with the
six summaries that experiment 347's head reads (median, mean, standard deviation, 10th and 90th
percentile, saturation share), the observed power over the old capacity, and the idle-mask flag.
Experiment 347's stored hindcast summaries cover the runs before 2025-06-17, so the two parts join
into one daily series.

    PYTHONPATH=<347>:<230>:<303>:<310>:<326>:<336>:<259>:<615 folders> \
        uv run --no-sync python ml-experiments/00615-*/e00615_wind_inputs.py
"""

import os
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00303_common as c303  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_common as common347  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_plant as plant  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_s4_inputs as inputs347  # noqa: E402  # ty: ignore[unresolved-import]
import e00615_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SUMMARY_NAMES = ("sum_median", "sum_mean", "sum_sd", "sum_p10", "sum_p90", "sum_saturation")
MEMBERS = c303.MEMBERS
OUT_DIR = common.DATA_DIR / "wind_inputs"


def observed(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return power over the old capacity to 2026-07-01 as experiment 303 builds it for training."""
    c303.TRAIN_END = common.VALID_LIMIT
    return c303.observed_training_power(sites=sites).rename({"target": "actual_norm"})


def farm_cases(
    *, farm: str, series: int, parameters: plant.PlantParameters, prior: float
) -> pl.DataFrame:
    """Return one farm's cases with the plant's six summaries, from every member file."""
    pieces = []
    for path in sorted(common.MEMBERS_DIR.glob("init=*.parquet")):
        rows = pl.read_parquet(path).filter(pl.col("time_series_id") == series)
        if rows.height == 0:
            continue
        rows = rows.sort(*c303.KEY_COLUMNS, "ensemble_member")
        if rows.height % MEMBERS:
            msg = f"{path.name} {farm}: {rows.height} rows are not a whole number of cases"
            raise ValueError(msg)
        weather = inputs347.member_weather(rows=rows)
        power = inputs347.expected_in_chunks(
            parameters=parameters, weather=weather, prior_scale=prior
        )
        power = power.reshape(-1, MEMBERS)
        keys = rows.gather_every(MEMBERS).select(c303.KEY_COLUMNS)
        members = rows["ensemble_member"].to_numpy().reshape(-1, MEMBERS)
        if not (members == np.arange(MEMBERS)).all():
            msg = f"{path.name} {farm}: members are not 0 to 50 in every case"
            raise ValueError(msg)
        summary = inputs347.summarise(power=power, flat_top=parameters.scale)
        pieces.append(keys.with_columns(**{n: summary[:, i] for i, n in enumerate(SUMMARY_NAMES)}))
    return pl.concat(pieces).with_columns(site=pl.lit(farm))


def main() -> None:
    """Write `wind_inputs/cases_<farm>.parquet` for each farm named (all three by default)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    actual = observed(sites=sites)
    mask = c303.idle_mask().with_columns(masked=pl.lit(value=True))
    priors = common347.prior_scales()
    plants = common347.read_json(name="plant_stage3.json")
    for farm in sys.argv[1:] or list(common347.FARMS):
        started = time.time()
        series = int(sites.filter(pl.col("site") == farm)["time_series_id"][0])
        parameters = plant.PlantParameters.from_dict(plants[farm])
        cases = farm_cases(farm=farm, series=series, parameters=parameters, prior=priors[farm])
        cases = (
            cases.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
            .with_columns(
                masked=pl.col("masked").fill_null(value=False),
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0,
            )
            .sort(*c303.KEY_COLUMNS)
        )
        path = OUT_DIR / f"cases_{farm}.parquet"
        cases.write_parquet(path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        print(f"{farm}: {cases.height:,} cases {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
