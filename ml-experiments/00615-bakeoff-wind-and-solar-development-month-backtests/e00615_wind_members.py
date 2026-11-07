"""Build the daily ensemble member rows at the wind farms for 2025-06-17 to 2026-06-30.

Experiment 259's `build_run` turns each 00 UTC run into half-hourly member rows at the three farms'
cells. Experiment 259 stopped at 2025-06-16, so this script continues it with the same function and
the same columns, and keeps valid times before 2026-07-01 so that July 2026 stays unused.

    PYTHONPATH=ml-experiments/00259-wind-plant-plus-quantile-offset-head:ml-experiments/00615-* \
        uv run --no-sync python ml-experiments/00615-*/e00615_wind_members.py
"""

import os
from datetime import timedelta

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00259_common as common259  # ty: ignore[unresolved-import]
import e00259_hindcast as hind259  # ty: ignore[unresolved-import]
import e00615_common as common
import patito as pt
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata
from contracts.typing_utils import typeddict_to_dict

common259.TRAIN_END = common.VALID_LIMIT
hind259.common.TRAIN_END = common.VALID_LIMIT


def main() -> None:
    """Write every missing run, skipping finished runs."""
    common.MEMBERS_DIR.mkdir(parents=True, exist_ok=True)
    config = hind259.settings()
    sites = common259.site_table()
    ids = sites["time_series_id"].to_list()
    metadata = pt.DataFrame(
        pl.read_parquet(
            config.metadata_path, storage_options=typeddict_to_dict(config.storage_options)
        ).filter(pl.col("time_series_id").is_in(ids))
    ).set_model(TimeSeriesMetadata)
    init = common.FIRST_NEW_RUN
    present = 0
    while init <= common.LAST_RUN:
        path = common.MEMBERS_DIR / f"init={init:%Y-%m-%d}.parquet"
        if not path.exists():
            frame = hind259.build_run(settings=config, metadata=metadata, init=init)
            if frame.height:
                tmp = path.with_suffix(".tmp")
                frame.write_parquet(tmp)
                tmp.rename(path)
            print(f"{init:%Y-%m-%d}: {frame.height:,} rows", flush=True)
        present += path.exists()
        init += timedelta(days=1)
    total = (common.LAST_RUN - common.FIRST_NEW_RUN).days + 1
    print(f"{present} of {total} runs present", flush=True)


if __name__ == "__main__":
    main()
