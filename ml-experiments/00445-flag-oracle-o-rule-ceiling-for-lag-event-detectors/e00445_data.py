"""Dense lagless inputs and power arrays of experiment 404.

The lagless module forecasts every half-hour of the dense grid from the freshest published
ensemble summary: the run whose publication (run time plus 9 hours) is the latest at or before the
valid time, so leads from the run time run from 9.5 to 33.0 hours. Its inputs are the ensemble
mean and spread of the seven weather columns, the solar position, the calendar, and the holiday
columns of experiment 397, and no lag and no lead.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
    uv run --no-sync python ml-experiments/00445-*/e00445_data.py smoke
```
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

FIRST_VALID_INDEX: Final[int] = int((prep.FIRST_RUN - common.GRID_START) / common.STEP) + 19
"""First grid index with a published run (the first run's valid times begin 9.5 hours after it)."""
LAGLESS_GROUPS: Final[tuple[str, ...]] = ("weather", "solar", "calendar", "holidays")
"""The input groups of the lagless module; lags and lead are excluded."""


@dataclass(frozen=True)
class DenseInputs:
    """One Primary's lagless inputs on the dense grid.

    Attributes:
        first: Grid index of row 0.
        x: Raw (unstandardised) lagless inputs, float32, one row per grid half-hour from `first`.
        columns: The raw input matrix's column names.
        valid: 1 where the row has a freshest-run weather input (weather finite), else 0.
    """

    first: int
    x: np.ndarray
    columns: list[str]
    valid: np.ndarray


def lagless_columns() -> list[str]:
    """Return the names of the columns of `prep.feature_matrix` that the lagless module sees."""
    names, _, layout = prep.feature_names(scheme="summary")
    keep = layout.columns_of(groups=LAGLESS_GROUPS)
    return [names[i] for i in keep]


def freshest_run_rows(*, site_row: dict) -> pl.DataFrame:
    """Return a Primary's ensemble summaries keyed by valid time, from the freshest run.

    A summary's `power_fcst_init_time` is its run time plus the 9-hour publication delay. For each
    valid time the row comes from the latest such origin strictly before the valid time, which
    gives leads from 9.5 to 33.0 hours (from the run time).
    """
    summary = prep.load_summary(series_id=site_row["time_series_id"])
    published = summary.with_columns(
        (pl.col("valid_time").dt.offset_by("-570m").dt.truncate("1d").dt.offset_by("9h")).alias(
            "freshest_init"
        )
    )
    freshest = published.filter(pl.col("power_fcst_init_time") == pl.col("freshest_init"))
    return freshest.drop("freshest_init").sort("valid_time")


def dense_inputs(*, site_row: dict) -> DenseInputs:
    """Return a Primary's raw lagless inputs for every grid half-hour with a published run.

    Args:
        site_row: The Primary's row of `prep.primaries()`.

    Returns:
        The inputs from the first valid time of the first run to the last valid time of the
        summaries. A half-hour with no summary gets NaN weather and `valid == 0`.
    """
    rows = freshest_run_rows(site_row=site_row)
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    first, last = int(index.min()), int(index.max())
    grid = pl.DataFrame({"valid_time": common.grid_times().slice(first, last - first + 1)}).join(
        rows, on="valid_time", how="left", validate="1:1"
    )
    present = grid["mean_" + prep.WEATHER[0]].is_not_null().to_numpy()
    filled = grid.with_columns(
        [pl.col(c).fill_null(0.0) for c in grid.columns if c.startswith(("mean_", "std_"))]
    )
    frame = common.with_derived(
        frame=filled, latitude=site_row["latitude"], longitude=site_row["longitude"]
    ).with_columns(pl.lit(None, dtype=pl.Float32).alias(c) for c in common.LAG_COLUMNS)
    frame = frame.with_columns(lead_hours=pl.lit(24.0))
    raw = prep.feature_matrix(frame=frame, scheme="summary")
    names, _, layout = prep.feature_names(scheme="summary")
    keep = layout.columns_of(groups=LAGLESS_GROUPS)
    return DenseInputs(
        first=first,
        x=raw[:, keep],
        columns=[names[i] for i in keep],
        valid=present.astype(np.int8),
    )


def smoke() -> None:
    """Build one Primary's dense inputs and report their shape and coverage."""
    site_row = next(prep.primaries().iter_rows(named=True))
    dense = dense_inputs(site_row=site_row)
    span = len(dense.x)
    print(
        f"{site_row['site']}: {span:,} half-hours from grid index {dense.first}, "
        f"{int(dense.valid.sum()):,} with weather, {len(dense.columns)} columns"
    )
    assert dense.x.shape[1] == len(lagless_columns())
    assert timedelta(hours=9) == common.PUBLICATION_DELAY
    print("smoke passed")


def main() -> None:
    """Run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke",))
    parser.parse_args()
    smoke()


if __name__ == "__main__":
    main()
