"""Gate: compare the solar inputs of this experiment with experiments 430 and 659 to 1e-4.

This check script, not the pipeline, imports the earlier experiments' modules, because the
reference is whatever they computed. For every farm it builds experiment 659's `farm_band_frame`
over the six day bands (days 1 to 10, valid times before 2026-07-01) and joins it to this
experiment's file on `(init_time, time)`. Rows with lead 252 h or more do not exist in the
reference. It prints the largest absolute difference per column and the number of rows found on
one side only.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    PYTHONPATH=ml-experiments/00310-*:ml-experiments/00430-*:ml-experiments/00461-*:\
ml-experiments/00659-*:ml-experiments/00666-* OMP_NUM_THREADS=1 \
        uv run --no-sync python ml-experiments/00666-*/e00666_solar_gate.py
"""

import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00659_solar_fit as f659  # ty: ignore[unresolved-import]
import e00666_common as common
import polars as pl

TOLERANCE: Final[float] = 1e-4
COLUMNS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_sd",
    "temp_mean",
    "temp_sd",
    "wind_mean",
    "wind_sd",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "pl_mean",
    "pl_sd",
    "cell_mean",
    "cell_sd",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "lead_hours",
)


def main() -> None:
    """Print the gate's differences per farm and fail if any exceeds the tolerance."""
    worst: dict[str, float] = dict.fromkeys((*COLUMNS, "actual"), 0.0)
    for farm in c310.FARMS:
        reference = pl.concat([f659.farm_band_frame(farm=farm, band=b) for b in c310.BAND_ORDER])
        mine = pl.read_parquet(common.SOLAR_DIR / f"{farm}.parquet").filter(
            pl.col("daylight")
            & (pl.col("lead_hours") < 252.0)
            & (pl.col("time") < common.OLD_VALID_LIMIT)
        )
        joined = reference.join(mine, on=["init_time", "time"], how="inner", suffix="_new")
        only_reference = reference.join(mine, on=["init_time", "time"], how="anti").height
        mine_only = mine.join(reference, on=["init_time", "time"], how="anti")
        print(
            f"{farm}: reference {reference.height:,} rows, mine {mine.height:,}, joined "
            f"{joined.height:,}, only in reference {only_reference:,}, only in mine "
            f"{mine_only.height:,} ({mine_only['actual'].drop_nulls().len():,} with an actual)",
            flush=True,
        )
        for column in COLUMNS:
            difference = float(
                joined.select((pl.col(column) - pl.col(f"{column}_new")).abs().max()).item()
            )
            worst[column] = max(worst[column], difference)
        actual = joined.select((pl.col("output") - pl.col("actual")).abs().max()).item()
        worst["actual"] = max(worst["actual"], float(actual))
    for column, value in worst.items():
        print(f"max abs difference {column}: {value:.3e}")
    if max(worst.values()) > TOLERANCE:
        msg = "the solar inputs differ from experiments 430 and 659 by more than the tolerance"
        raise AssertionError(msg)


if __name__ == "__main__":
    main()
