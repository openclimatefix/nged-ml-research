"""Raw-matrix builders of the solar arms of experiment 711: PLP rows with the cloud columns.

**The rows are experiment 666's PLP rows, with the arm's extra inputs appended.** The table of
`e00711_solar_inputs.py` holds each farm's half-hours with the 18 PLP inputs on daylight half-hours
and the four cloud columns on every half-hour with a stamp. This module selects a refit's training
and target rows, adds the arm's cloud columns (real, shuffled-run placebo, or time-shifted
placebo), and builds the raw matrix. It imports no `torch`.

**A masked row is NaN before standardisation, and 0 after.** A row whose run is in the 19-month gap
(or is run 2024-01-03) has `cloud_missing = true`. The standardiser of a refit uses the training
rows with `cloud_missing = false` only, and a masked row's cloud inputs equal the column's mean, so
the standardised input is exactly 0. The placebo copies the real missing pattern, and the code
asserts that the two flags are equal on every row.

**A refit at the first of month M** trains on daylight half-hours whose run is before the refit and
whose valid time is at least 1 day before it, and forecasts the daylight half-hours of the runs of
month M that are before 2026-07-01. The origin rule is asserted on every training and target row.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00711_arms as arms
import e00711_common as common
import e00711_origin as origin_rules
import e00711_placebo as placebo
import numpy as np
import polars as pl

SOLAR_DIR: Final[Path] = common.WORK_DIR / "solar_inputs"
SOLAR_PLACEBO_DIR: Final[Path] = common.PLACEBO_DIR / "solar"
BASE_COLUMNS: Final[tuple[str, ...]] = (
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
"""The 18 inputs of experiment 659's `PLP` arm."""
CLOUD_COLUMNS: Final[tuple[str, ...]] = (
    "cloud_mean",
    "cloud_spread",
    "cloud_clear_share",
    "cloud_overcast_share",
)
MAX_LEAD_HOURS: Final[float] = 348.0
VALIDATION_DAYS: Final[int] = 61
"""A refit stops early on the last 61 days of its training rows."""
REFIT_GAP_DAYS: Final[int] = 1
"""A training row's valid time is at least this many days before the refit."""
SOLAR_SERIES_KEY_BASE: Final[int] = 1000
"""A farm's placebo draw is seeded by this plus the farm's position in `pv_sites()`."""
ROW_FIRST_RUN: Final[datetime] = common.CACHE_FIRST_RUN + timedelta(days=14)
"""The first run of a row: its time-shifted donor is the first back-filled run."""
FIRST_HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
RUN_KEY: Final[tuple[str, str]] = ("init_time", "time")


def refit_of(*, month: str) -> datetime:
    """Return a fold's refit time: 00:00 UTC on the first day of the month."""
    return datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)


def next_month(*, refit: datetime) -> datetime:
    """Return the first day of the month after `refit`."""
    return (refit + timedelta(days=32)).replace(day=1)


def target_runs() -> list[datetime]:
    """Return every run that a solar row can read, from the first row run to 2026-06-30."""
    return common.runs_between(first=ROW_FIRST_RUN, last=common.SCORED_LAST_RUN)


def load_farm(*, site: str) -> pl.DataFrame:
    """Return one farm's table from `e00711_solar_inputs.py` (every half-hour of every run)."""
    path = SOLAR_DIR / f"{site}.parquet"
    if not path.exists():
        msg = f"{path} does not exist: run `e00711_solar_inputs.py`"
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


def daylight_rows(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return a farm's daylight half-hours with all 18 inputs and a lead below 348 hours.

    A daylight half-hour without its inputs is allowed only on run 2024-01-03, which is absent
    upstream. Any other run's daylight half-hour without a stamp is an error.

    Raises:
        ValueError: If a daylight half-hour of another run lacks an input.
    """
    daylight = table.filter(pl.col("daylight") & (pl.col("lead_hours") < MAX_LEAD_HOURS))
    incomplete = daylight.filter(pl.any_horizontal(pl.col(*BASE_COLUMNS).is_null()))
    allowed = incomplete.filter(pl.col("init_time").dt.date() == common.UPSTREAM_GAP_RUN.date())
    if incomplete.height != allowed.height:
        msg = f"{incomplete.height - allowed.height} daylight half-hours lack an input"
        raise ValueError(msg)
    return daylight.filter(~pl.any_horizontal(pl.col(*BASE_COLUMNS).is_null()))


def series_key(*, farm_index: int) -> int:
    """Return the integer that seeds a farm's placebo draw."""
    return SOLAR_SERIES_KEY_BASE + farm_index


def solar_donor_map(*, arm: arms.Arm, month: str, seed: int, farm_index: int) -> np.ndarray | None:
    """Return a farm's donor map for the arm, or `None` for a real arm.

    The shuffled-run map follows `e00711_placebo.shuffled_donor_map` over the solar runs, the
    cache's runs as donors, the fold's cut, the fitting seed, and the farm's key. The time-shifted
    map follows `shifted_donor_map`. The maps are stored under `work/placebo/solar/`.
    """
    if arm.placebo == "none":
        return None
    targets = target_runs()
    available = common.cache_runs()
    key = series_key(farm_index=farm_index)
    if arm.placebo == "shuffled":
        cut = refit_of(month=month)
        donors = placebo.shuffled_donor_map(
            targets=targets, available=available, cut=cut, seed=seed, series_key=key
        )
        placebo.assert_shuffled_rules(targets=targets, donors=donors, cut=cut)
        path = SOLAR_PLACEBO_DIR / "shuffled" / f"{month}_s{seed}" / f"{key}.npz"
    else:
        donors = placebo.shifted_donor_map(targets=targets, available=available)
        placebo.assert_shifted_rules(targets=targets, donors=donors)
        path = SOLAR_PLACEBO_DIR / "shifted" / f"{key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, donor=donors, target=np.array([placebo.day_number(run=r) for r in targets]))
    temporary.rename(path)
    return donors


def apply_solar_donors(
    *, rows: pl.DataFrame, full: pl.DataFrame, targets: list[datetime], donors: np.ndarray
) -> pl.DataFrame:
    """Return the rows with each run's cloud columns replaced by its donor run's at the same lead.

    A row whose run is unmapped (it carries no cloud) gets nulls. The donor's half-hour is read from
    `full`, the farm's table of every half-hour, so a half-hour that is not daylight in the donor
    run still has cloud.

    Args:
        rows: The farm's selected rows, with `init_time` and `lead_hours`.
        full: The farm's whole table, with `init_time`, `lead_hours`, and the cloud columns.
        targets: The runs of the map.
        donors: The donor day of each target (`placebo.UNMAPPED` for a run without cloud).

    Returns:
        The rows with the donor's cloud columns, in the rows' order.

    Raises:
        ValueError: If a mapped row finds no donor half-hour, or a run is absent from the map.
    """
    dtype = rows.schema["init_time"]
    link = pl.DataFrame(
        {
            "init_time": [t for t, d in zip(targets, donors, strict=True) if d != placebo.UNMAPPED],
            "donor_init_time": [
                placebo.run_of(day=int(d)) for d in donors if d != placebo.UNMAPPED
            ],
        },
        schema={"init_time": dtype, "donor_init_time": dtype},
    )
    donor_values = full.select(
        pl.col("init_time").alias("donor_init_time"), "lead_hours", *CLOUD_COLUMNS
    )
    joined = (
        rows.with_row_index("_row")
        .drop(*CLOUD_COLUMNS)
        .join(link, on="init_time", how="left")
        .join(donor_values, on=["donor_init_time", "lead_hours"], how="left")
        .sort("_row")
    )
    mapped_without_value = joined.filter(
        pl.col("donor_init_time").is_not_null() & pl.col(CLOUD_COLUMNS[0]).is_null()
    )
    if mapped_without_value.height:
        msg = f"{mapped_without_value.height} rows find no donor half-hour at their lead"
        raise ValueError(msg)
    if joined.height != rows.height:
        msg = "the donor join changed the number of rows"
        raise ValueError(msg)
    return joined.drop("_row", "donor_init_time")


def assert_same_flag(*, rows: pl.DataFrame) -> None:
    """Assert that a placebo row has null cloud if and only if the real row is `cloud_missing`.

    Raises:
        ValueError: If a row's placebo cloud is null where the real flag is false, or the reverse.
    """
    differs = rows.filter(pl.col(CLOUD_COLUMNS[0]).is_null() != pl.col("cloud_missing"))
    if differs.height:
        msg = f"{differs.height} rows differ in real and placebo missing pattern"
        raise ValueError(msg)


def with_arm_columns(
    *,
    rows: pl.DataFrame,
    full: pl.DataFrame,
    arm: arms.Arm,
    month: str,
    seed: int,
    farm_index: int,
) -> pl.DataFrame:
    """Return a farm's rows with the arm's cloud columns (real or placebo) and the real flag.

    Args:
        rows: The farm's selected rows from `daylight_rows`.
        full: The farm's whole table.
        arm: The arm.
        month: The fold's month, which sets the shuffled placebo's cut.
        seed: The fitting seed.
        farm_index: The farm's position in `pv_sites()`.

    Returns:
        The rows; for a placebo arm the cloud columns come from the donor run, and the placebo's
        null pattern is asserted equal to the real pattern first.
    """
    donors = solar_donor_map(arm=arm, month=month, seed=seed, farm_index=farm_index)
    if donors is None:
        return rows
    out = apply_solar_donors(rows=rows, full=full, targets=target_runs(), donors=donors)
    assert_same_flag(rows=out)
    return out


@dataclass(frozen=True)
class SolarRows:
    """The training and target rows of one refit, as frames with every column an arm may use."""

    train: pl.DataFrame
    target: pl.DataFrame
    farm_order: tuple[str, ...]


def assert_origin_rules(*, frame: pl.DataFrame) -> None:
    """Assert the origin rule on every row: the run is at least 9 hours old at the 12:00 origin."""
    origin = frame["init_time"] + timedelta(hours=common.SOLAR_ORIGIN_HOUR)
    origin_rules.assert_solar_rows(
        init_time=frame["init_time"], origin=origin, valid_time=frame["time"]
    )


def select_rows(*, frame: pl.DataFrame, refit: datetime, arm: arms.Arm) -> SolarRows:
    """Return the training and target rows of a refit from one frame of every farm's rows.

    Training rows have a finite actual, a run before the refit, and a valid time at least 1 day
    before it (the arm that drops the gap also drops the rows of the gap runs). Target rows are the
    runs of the refit's month, with valid times before 2026-07-01.

    Args:
        frame: The daylight rows of every farm, with the arm's cloud columns.
        refit: The first day of the fold's month.
        arm: The arm.

    Returns:
        The two frames, each asserted against the origin rule.

    Raises:
        ValueError: If a training row is valid at or after the refit, a target row is a gap run
            for an arm that drops the gap rows, or an origin rule fails.
    """
    cutoff = refit - timedelta(days=REFIT_GAP_DAYS)
    train = frame.filter(
        (pl.col("time") < cutoff) & (pl.col("init_time") < refit) & pl.col("actual").is_finite()
    )
    if arm.drop_gap:
        train = train.filter(~pl.col("cloud_missing"))
    target = frame.filter(
        (pl.col("init_time") >= refit)
        & (pl.col("init_time") < next_month(refit=refit))
        & (pl.col("time") < FIRST_HOLDOUT)
    )
    if train.height == 0 or target.height == 0:
        msg = f"refit {refit:%Y-%m}: {train.height} training rows, {target.height} target rows"
        raise ValueError(msg)
    if train.select(pl.col("time").max()).item() >= refit:
        msg = "a training row is valid at or after the refit"
        raise ValueError(msg)
    if arm.drop_gap and target["cloud_missing"].any():
        msg = "a scored row's run is in the gap, so the arm that drops gap rows cannot score it"
        raise ValueError(msg)
    assert_origin_rules(frame=train)
    assert_origin_rules(frame=target)
    return SolarRows(train=train, target=target, farm_order=tuple(sorted(frame["series"].unique())))


def input_columns(*, arm: arms.Arm) -> list[str]:
    """Return the arm's input columns: the 18 of PLP, the summaries, and the flag."""
    return [*BASE_COLUMNS, *arm.extra_columns]


def raw_matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the raw input matrix, float64, with NaN where a masked row has no cloud.

    A masked row's cloud columns are null in the frame and become NaN here. The flag is 0 or 1.
    """
    return frame.select([pl.col(c).cast(pl.Float64) for c in columns]).to_numpy()


def standardisation(
    *, x: np.ndarray, is_fit: np.ndarray, columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean and standard deviation of each column, and the matrix with masks filled.

    Each summary column's statistics use the `is_fit` rows that are not masked. A masked row's NaN
    is replaced by the column's mean, so its standardised input is 0. The flag has mean 0 and
    standard deviation 1.

    Args:
        x: The raw matrix from `raw_matrix`.
        is_fit: Which rows the network trains on.
        columns: The column names of `x`.

    Returns:
        The mean, the standard deviation, and the matrix with NaN replaced.

    Raises:
        ValueError: If a base column holds NaN, or a summary column has no unmasked fit row.
    """
    mean = np.zeros(x.shape[1])
    std = np.ones(x.shape[1])
    filled = x.copy()
    for j, name in enumerate(columns):
        if name == arms.FLAG_COLUMN:
            continue
        values = x[is_fit, j]
        if name in CLOUD_COLUMNS or name in arms.SUMMARY_COLUMNS["t3"]:
            usable = values[np.isfinite(values)]
            if usable.size == 0:
                msg = f"no unmasked fit row for the column {name}"
                raise ValueError(msg)
            mean[j] = usable.mean()
            std[j] = max(float(usable.std()), 1e-6)
            filled[~np.isfinite(filled[:, j]), j] = mean[j]
        else:
            if np.isnan(x[:, j]).any():
                msg = f"the base column {name} holds NaN"
                raise ValueError(msg)
            mean[j] = values.mean()
            std[j] = max(float(values.std()), 1e-6)
    return mean, std, filled


def prediction_frame(
    *, target: pl.DataFrame, fraction: np.ndarray, ratio: pl.Series, seed: int
) -> pl.DataFrame:
    """Return the target rows' quantiles as a fraction of each farm's stored effective capacity.

    A network predicts a fraction of the refit's capacity, and the stored effective capacity is
    `1 / ratio` of it, so the fraction of the stored capacity is the fraction times the ratio of the
    refit's capacity to the stored one.

    Args:
        target: The target rows with `time_series_id`, `init_time`, and `time`.
        fraction: Quantiles as a fraction of the refit's capacity, shape (rows, 11).
        ratio: Each row's new capacity over the stored effective capacity.
        seed: The fitting seed.

    Returns:
        A frame keyed by `time_series_id`, `power_fcst_init_time` (the 12:00 UTC origin), and
        `valid_time`, with the 11 quantiles and `seed`.
    """
    scaled = fraction * ratio.to_numpy()[:, None]
    quantiles = pl.DataFrame(scaled.astype(np.float32), schema=list(LEVEL_COLUMNS), orient="row")
    keys = target.select(
        pl.col("time_series_id").cast(pl.Int64),
        power_fcst_init_time=pl.col("init_time") + pl.duration(hours=common.SOLAR_ORIGIN_HOUR),
        valid_time=pl.col("time"),
        seed=pl.lit(seed).cast(pl.Int64),
    )
    return keys.hstack(quantiles)
