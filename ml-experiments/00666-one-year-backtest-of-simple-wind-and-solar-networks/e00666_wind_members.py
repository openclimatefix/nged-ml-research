"""Rebuild the daily ensemble member rows at the wind farms for the runs 2026-06-16 to 2026-10-31.

Copied from experiment 259's `e00259_hindcast.build_run` (and experiment 615's driver), with the
valid-time limit moved to 2026-11-20 so every run keeps its full 15 days. Experiment 615 cut its
runs at 2026-07-01, which truncated the runs from 2026-06-16. The production feature engineer left
joins power onto the NWP in bulk mode, so rows without observed power are kept.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00666-*/e00666_wind_members.py
"""

import os
from datetime import datetime, timedelta
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common
import numpy as np
import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import Nwp
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer
from nged_data.storage import scan_cleaned_power
from studies.pv_dataset import wind_sites
from studies.sources import REPO_DATA_DIR

KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
WEATHER_COLUMNS: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
    "temperature_c",
)
MAX_LEAD: Final[timedelta] = timedelta(days=15)
ENGINEERED: Final[frozenset[str]] = frozenset(
    {
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        "temperature_2m",
        "nwp_lead_time_hours",
    }
)


def settings() -> Settings:
    """Return the settings with every data path under the shared data folder."""
    root = str(REPO_DATA_DIR)
    return Settings(data_path_internal=root, data_path_delivery=root, local_artifacts_path=root)


def build_run(
    *, settings: Settings, metadata: pt.DataFrame[TimeSeriesMetadata], init: datetime
) -> pl.DataFrame:
    """Return one run's member rows at the farms: the key columns, member, and the weather."""
    ids = metadata["time_series_id"].to_list()
    end = min(init + MAX_LEAD, common.VALID_LIMIT - timedelta(minutes=30))
    power = scan_cleaned_power(
        settings.cleaned_power_time_series_data_path, settings.storage_options
    ).filter(pl.col("time_series_id").is_in(ids), pl.col("time") >= init, pl.col("time") <= end)
    nwp = Nwp.scan_delta(settings.nwp_data_path, storage_options=settings.storage_options).filter(
        pl.col("init_time") == init,
        pl.col("valid_time") >= init,
        pl.col("valid_time") <= end,
        pl.col("h3_index").is_in(metadata["h3_res_5"].unique().to_list()),
    )
    frame = pl.DataFrame(
        TabularFeatureEngineer()
        .engineer(
            selected_features=set(ENGINEERED),
            power_time_series=pt.LazyFrame.from_existing(power).set_model(PowerTimeSeries),
            time_series_metadata=metadata,
            nwp=pt.LazyFrame.from_existing(nwp).set_model(Nwp),
        )
        .collect(engine="streaming")
    )
    if frame.height == 0:
        return frame
    bearing_10 = pl.col("wind_direction_10m") * np.pi / 180.0
    bearing_100 = pl.col("wind_direction_100m") * np.pi / 180.0
    return (
        frame.select(
            *KEY_COLUMNS,
            "ensemble_member",
            "wind_speed_10m",
            "wind_speed_100m",
            bearing_10.sin().alias("wind_dir_sin_10m"),
            bearing_10.cos().alias("wind_dir_cos_10m"),
            bearing_100.sin().alias("wind_dir_sin_100m"),
            bearing_100.cos().alias("wind_dir_cos_100m"),
            pl.col("temperature_2m").alias("temperature_c"),
        )
        .with_columns(pl.col(WEATHER_COLUMNS).cast(pl.Float32))
        .filter(pl.col("valid_time") < common.VALID_LIMIT)
        .sort(*KEY_COLUMNS, "ensemble_member")
    )


def main() -> None:
    """Write every missing run from 2026-06-16 to 2026-10-31, skipping finished runs."""
    common.MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    config = settings()
    ids = wind_sites()["time_series_id"].to_list()
    metadata = pt.DataFrame(
        pl.read_parquet(
            config.metadata_path, storage_options=typeddict_to_dict(config.storage_options)
        ).filter(pl.col("time_series_id").is_in(ids))
    ).set_model(TimeSeriesMetadata)
    init = common.FIRST_REBUILT_RUN
    while init <= common.LAST_RUN:
        path = common.MEMBERS_DIR / f"init={init:%Y-%m-%d}.parquet"
        if not path.exists():
            frame = build_run(settings=config, metadata=metadata, init=init)
            if frame.height:
                tmp = path.with_suffix(".tmp")
                frame.write_parquet(tmp)
                tmp.rename(path)
            print(f"{init:%Y-%m-%d}: {frame.height:,} rows", flush=True)
        init += timedelta(days=1)
    total = (common.LAST_RUN - common.FIRST_REBUILT_RUN).days + 1
    present = len(list(common.MEMBERS_DIR.glob("init=*.parquet")))
    print(f"{present} of {total} runs present", flush=True)


if __name__ == "__main__":
    main()
