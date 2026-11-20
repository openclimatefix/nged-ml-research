"""Origin-time assertions of experiment 711, and the example row that each target prints.

**Every row takes all of its weather, cloud included, from one run: the 00 UTC run issued at least
9 hours before the row's origin.** The Primary demand origin is the run plus 9 hours. The solar
origin is 12:00 UTC on the run's day. A failed assertion makes the experiment `invalid`.

`assert_origin_after_run` is a copy of experiment 707's function of the same name, because
`e00707_prep.py` imports `torch` and the tests of this folder run without it. The solar builder
has no row-level assertion of its own, so `assert_solar_rows` adds one.

```bash
# environment as in e00711_common.py
uv run --no-sync python ml-experiments/00711-*/e00711_origin.py primary-example
```
"""

import argparse
from datetime import UTC, datetime, timedelta
from typing import Final

import e00711_common as common
import numpy as np
import polars as pl

PRIMARY_EXAMPLE_RUN: Final[datetime] = datetime(2026, 1, 14, tzinfo=UTC)
PRIMARY_EXAMPLE_TARGET: Final[datetime] = datetime(2026, 1, 19, 17, 30, tzinfo=UTC)
PRIMARY_EXAMPLE_LEAD_HOURS: Final[float] = 128.5
SOLAR_EXAMPLE_RUN: Final[datetime] = datetime(2026, 4, 10, tzinfo=UTC)
SOLAR_EXAMPLE_TARGET: Final[datetime] = datetime(2026, 4, 13, 11, 30, tzinfo=UTC)
US_PER_HOUR: Final[int] = 3_600 * 1_000_000
DELAY_US: Final[int] = common.PUBLICATION_DELAY // timedelta(microseconds=1)


def assert_origin_after_run(*, nwp_init: np.ndarray, origin: np.ndarray) -> None:
    """Assert `nwp_init_time + 9 h <= origin` for every row (both in microseconds since the epoch).

    Args:
        nwp_init: The issue time of the run that each row reads.
        origin: The origin of each row.

    Raises:
        ValueError: If a row reads a run issued less than 9 hours before its origin.
    """
    late = nwp_init + DELAY_US > origin
    if late.any():
        msg = f"{int(late.sum())} rows read a run issued less than 9 hours before the origin"
        raise ValueError(msg)


def assert_primary_rows(*, frame: pl.DataFrame) -> None:
    """Assert the origin rule on Primary demand rows, cloud summaries included.

    Args:
        frame: Rows with `nwp_init_time` and `power_fcst_init_time` (microsecond datetimes).

    Raises:
        ValueError: If a row reads a run issued less than 9 hours before its origin, or the origin
            is not exactly the run plus 9 hours, or the run is not a 00:00 UTC run.
    """
    init = frame["nwp_init_time"].dt.epoch("us").to_numpy()
    origin = frame["power_fcst_init_time"].dt.epoch("us").to_numpy()
    assert_origin_after_run(nwp_init=init, origin=origin)
    if (origin - init != DELAY_US).any():
        msg = "a Primary demand origin is not exactly its run plus 9 hours"
        raise ValueError(msg)
    if (init % (24 * US_PER_HOUR) != 0).any():
        msg = "a Primary demand row reads a run that is not a 00:00 UTC run"
        raise ValueError(msg)


def assert_solar_rows(*, init_time: pl.Series, origin: pl.Series, valid_time: pl.Series) -> None:
    """Assert the origin rule on solar rows: `init_time + 9 h <= origin`, `init_time` is 00 UTC.

    The run of a row is the 00 UTC run of the origin's own date, and the origin is 12:00 UTC.

    Args:
        init_time: The run each row reads.
        origin: The origin of each row (12:00 UTC on the run's day).
        valid_time: The half-hour that each row forecasts.

    Raises:
        ValueError: If a row reads a run issued less than 9 hours before its origin, if the run is
            not 00:00 UTC of the origin's date, if the origin is not 12:00 UTC, or if the valid
            time is not after the origin.
    """
    assert_origin_after_run(
        nwp_init=init_time.dt.epoch("us").to_numpy(), origin=origin.dt.epoch("us").to_numpy()
    )
    if (init_time != origin.dt.truncate("1d")).any():
        msg = "a solar row's run is not 00:00 UTC of its origin's date"
        raise ValueError(msg)
    if (origin.dt.hour() != common.SOLAR_ORIGIN_HOUR).any() or (origin.dt.minute() != 0).any():
        msg = "a solar origin is not 12:00 UTC"
        raise ValueError(msg)
    if (valid_time <= origin).any():
        msg = "a solar row forecasts a half-hour that is not after its origin"
        raise ValueError(msg)


def primary_example(*, frame: pl.DataFrame) -> str:
    """Return the example row of Primary demand: its run, origin, target, lead, and cloud.

    The example is run 2026-01-14, origin 2026-01-14 09:00 UTC, target 2026-01-19 17:30 UTC (lead
    128.5 hours). The function asserts that the cloud comes from the run of 2026-01-14 and not from
    the run of 2026-01-19.

    Args:
        frame: Cloud summary rows of one series, from the cache file of the example run, with
            `nwp_init_time`, `power_fcst_init_time`, `valid_time`, `cloud_mean`, and `cloud_spread`.

    Returns:
        A printable line.

    Raises:
        ValueError: If the example row is missing, or reads a different run.
    """
    row = frame.filter(pl.col("valid_time") == PRIMARY_EXAMPLE_TARGET)
    if row.height != 1:
        msg = f"expected one row at {PRIMARY_EXAMPLE_TARGET}, found {row.height}"
        raise ValueError(msg)
    record = row.row(0, named=True)
    origin = common.origin_of(run=PRIMARY_EXAMPLE_RUN)
    if record["nwp_init_time"] != PRIMARY_EXAMPLE_RUN or record["power_fcst_init_time"] != origin:
        msg = "the example row does not read the run of 2026-01-14 with origin 09:00 UTC"
        raise ValueError(msg)
    lead = (record["valid_time"] - origin) / timedelta(hours=1)
    if lead != PRIMARY_EXAMPLE_LEAD_HOURS:
        msg = f"the example lead is {lead} hours, not {PRIMARY_EXAMPLE_LEAD_HOURS}"
        raise ValueError(msg)
    assert_primary_rows(frame=row)
    return (
        f"run {record['nwp_init_time']:%Y-%m-%d %H:%M} UTC, origin "
        f"{record['power_fcst_init_time']:%Y-%m-%d %H:%M} UTC, target "
        f"{record['valid_time']:%Y-%m-%d %H:%M} UTC (lead {lead} hours): cloud mean "
        f"{record['cloud_mean']:.3f}, spread {record['cloud_spread']:.3f}, from the run of "
        f"{record['nwp_init_time']:%Y-%m-%d} and never from the run of "
        f"{PRIMARY_EXAMPLE_TARGET:%Y-%m-%d}"
    )


def solar_example(*, frame: pl.DataFrame) -> str:
    """Return the example row of the solar farms: run 2026-04-10, origin 12:00, target 2026-04-13.

    Args:
        frame: One farm's rows with `init_time`, `time`, `cloud_mean`, and `cloud_spread`.

    Returns:
        A printable line.

    Raises:
        ValueError: If the example row is missing or fails the solar origin assertion.
    """
    row = frame.filter(
        pl.col("init_time") == SOLAR_EXAMPLE_RUN, pl.col("time") == SOLAR_EXAMPLE_TARGET
    )
    if row.height != 1:
        msg = f"expected one row at {SOLAR_EXAMPLE_TARGET}, found {row.height}"
        raise ValueError(msg)
    record = row.row(0, named=True)
    origin = common.solar_origin_of(run=record["init_time"])
    assert_solar_rows(
        init_time=row["init_time"],
        origin=pl.Series([origin], dtype=row["init_time"].dtype),
        valid_time=row["time"],
    )
    return (
        f"run {record['init_time']:%Y-%m-%d %H:%M} UTC, origin {origin:%Y-%m-%d %H:%M} UTC, "
        f"target {record['time']:%Y-%m-%d %H:%M} UTC: cloud mean {record['cloud_mean']:.3f}, "
        f"spread {record['cloud_spread']:.3f}, from the run of {record['init_time']:%Y-%m-%d}"
    )


def main() -> None:
    """Print the Primary demand example row from the cloud cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("primary-example",))
    parser.parse_args()
    frame = pl.read_parquet(common.cache_path(run=PRIMARY_EXAMPLE_RUN))
    series = frame["time_series_id"].min()
    print(primary_example(frame=frame.filter(pl.col("time_series_id") == series)))


if __name__ == "__main__":
    main()
