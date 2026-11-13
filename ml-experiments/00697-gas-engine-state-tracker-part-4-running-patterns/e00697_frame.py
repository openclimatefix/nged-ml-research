r"""Build experiment 697's private half-hour frames: one parquet per series.

    uv run --no-sync python $E/00697-*/e00697_frame.py

A frame has one row per half-hour, indexed by the half-hour **start**, with the local fields, the
labels, the truth where it is known, the reading, and every driver. The frames are private: a frame
of Gas A holds dated output, and a frame of D8 holds its dated tracked state, so none leaves the
data folder. The frames are:

- `frame_d8`: D8's three eras with experiment 694's smoothed (`ts`) and filtered (`tf`) M2 flags,
  experiment 695's stored mean probability (`p_any_a`, from the hindsight arm) and fixed-lag
  probability (`p_any_c`, 2025-11 to 2026-06), the fitted block size in MW, the panel's weather
  columns, and whether the half-hour lies in one of experiment 159's multi-day drops.
- `frame_record`: Gas A's record, with `r1` and `r2` (1 or more and 2 or more units, from `power`).
- `frame_<control>`: the stored runs of `e00697_controls.py`, one frame per control series and kind
  of truth, with a `seed` column, `truth1` and `truth2`, and the labels.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00695_common import (  # ty: ignore[unresolved-import]
    FADE_MONTHS,
    ENGINE_MONTHS,
    fold_refit,
    grid_index,
)
from e00697_common import (  # ty: ignore[unresolved-import]
    DAY,
    E695_DIR,
    FRAMES_DIR,
    PIPELINE_OFFSET,
    PROBABILITY_PATH,
    add_local_fields,
    add_segments,
    gas_day_expression,
    half_hour_starts,
    load_d8_eras,
    load_panel,
    panel_array,
)
from e00697_controls import Job, jobs, load_control  # ty: ignore[unresolved-import]
from e00697_drivers import (  # ty: ignore[unresolved-import]
    bank_holiday_dates,
    christmas_week_expression,
    day_ahead_prices,
    drops,
    gas_prices,
    national_wind,
    spark_spread_expression,
    system_prices,
)
from e00697_engine import record_frame, units_of  # ty: ignore[unresolved-import]

WEATHER_KEYS: Final[tuple[str, ...]] = ("temp", "wind100", "ghi", "expected")
"""D8's panel columns, carried by grid index: 2 m temperature, 100 m wind, irradiance, expected power."""
N_PRICE_PERIODS: Final[int] = DAY
"""A local day has a price rank only when all 48 settlement periods have a price."""


@dataclass(frozen=True)
class DriverTables:
    """The public driver tables, loaded once."""

    system: pl.DataFrame
    day_rank: pl.DataFrame
    wind: pl.DataFrame
    gas: pl.DataFrame
    day_ahead: pl.DataFrame
    holidays: frozenset


def load_drivers() -> DriverTables:
    """Load every public driver table."""
    system = system_prices()
    ranked = add_local_fields(frame=system).with_columns(
        n=pl.col("system_price").count().over("local_date"),
        rank=pl.col("system_price").rank(method="ordinal").over("local_date").cast(pl.Int16) - 1,
    )
    day_rank = ranked.select(
        "start",
        price_rank_in_day=pl.when((pl.col("n") == N_PRICE_PERIODS) & ~pl.col("clock_change_day"))
        .then(pl.col("rank"))
        .otherwise(None),
    )
    return DriverTables(
        system=system,
        day_rank=day_rank,
        wind=national_wind(),
        gas=gas_prices(),
        day_ahead=day_ahead_prices(),
        holidays=bank_holiday_dates(),
    )


def add_drivers(*, frame: pl.DataFrame, tables: DriverTables) -> pl.DataFrame:
    """Join the drivers to a frame that has `start` and the local fields.

    Every join is on a start: the half-hour start for the system price, the day rank, and the wind,
    the gas day (local start minus 5 hours) for the gas price, and the hour that contains the start
    for the day-ahead price. The order of the frame is kept.
    """
    out = (
        frame.join(tables.system, on="start", how="left", maintain_order="left")
        .join(tables.day_rank, on="start", how="left", maintain_order="left")
        .join(tables.wind, on="start", how="left", maintain_order="left")
        .with_columns(gas_day=gas_day_expression(), hour=pl.col("start").dt.truncate("1h"))
        .join(tables.gas, on="gas_day", how="left", maintain_order="left")
        .join(tables.day_ahead, on="hour", how="left", maintain_order="left")
        .drop("gas_day", "hour")
    )
    return out.with_columns(
        spark_spread=spark_spread_expression(),
        bank_holiday=pl.col("local_date").is_in(pl.Series(sorted(tables.holidays))),
        christmas_week=christmas_week_expression(),
        dearest_six=pl.col("price_rank_in_day") >= N_PRICE_PERIODS - 6,
    )


def fixed_lag_probability() -> np.ndarray:
    """Return experiment 695's fixed-lag P(1 or more units) on the pipeline grid, NaN where unstored."""
    out = np.full(len(np.load(PROBABILITY_PATH)["lmean"]), np.nan)
    months = (*ENGINE_MONTHS, *FADE_MONTHS)
    for i, month in enumerate(months):
        first = grid_index(when=fold_refit(month=month))
        last = (
            grid_index(when=fold_refit(month=months[i + 1])) if i + 1 < len(months) else first + 30 * DAY
        )
        with np.load(E695_DIR / "states" / "A1" / f"{month}.npz") as data:
            out[first:last] = 1.0 - data["pk"][first:last, 0].astype(float)
    return out


def d8_frame(*, tables: DriverTables) -> pl.DataFrame:
    """Return D8's frame: three eras, labels, probabilities, weather, and drivers."""
    panel = load_panel()
    pk_all = np.load(PROBABILITY_PATH)["pk"]
    p_any_a = 1.0 - pk_all[:, 0].astype(float)
    p_any_c = fixed_lag_probability()
    weather = {key: panel_array(panel=panel, label="D8", key=key) for key in WEATHER_KEYS}
    drop_table = drops()
    parts = []
    for era in load_d8_eras():
        n = len(era["y"])
        index = int(era["start"]) + np.arange(n)
        m2 = era["M2"]
        reading = np.asarray(era["y"], dtype=float)
        part = pl.DataFrame(
            {
                "index": index,
                "era": [era["label"]] * n,
                "reading": reading,
                "ts": np.asarray(m2["flag"], dtype=bool),
                "tf": np.asarray(m2["filt"], dtype=bool),
                "k_path": np.asarray(m2["k_path"], dtype=np.int64),
                "unit_mw": np.full(n, float(m2["u"])),
                "p_any_a": p_any_a[index + PIPELINE_OFFSET],
                "p_any_c": p_any_c[index + PIPELINE_OFFSET],
                **{key: weather[key][index] for key in WEATHER_KEYS},
            }
        ).with_columns(
            reading=pl.col("reading").fill_nan(None),
            observed=pl.col("reading").is_not_nan(),
            fitted_mw=pl.col("k_path") * pl.col("unit_mw"),
            p_any_c=pl.col("p_any_c").fill_nan(None),
            **{key: pl.col(key).fill_nan(None) for key in WEATHER_KEYS},
        )
        parts.append(part)
    frame = pl.concat(parts).with_columns(start=half_hour_starts(index=np.concatenate([p["index"].to_numpy() for p in parts])))
    frame = add_segments(frame=add_local_fields(frame=frame), group_columns=("era",))
    in_drop = np.zeros(len(panel_array(panel=panel, label="D8", key="power")), dtype=bool)
    for row in drop_table.iter_rows(named=True):
        in_drop[row["first_stamp"] : row["last_stamp"] + 1] = True
    frame = frame.with_columns(in_drop=pl.Series(in_drop[frame["index"].to_numpy()]))
    return add_drivers(frame=frame, tables=tables)


def record_with_drivers(*, tables: DriverTables) -> pl.DataFrame:
    """Return Gas A's frame with the drivers and the two running flags taken from `power`."""
    base = record_frame()
    units = units_of(g=np.clip(base["power"].to_numpy(), 0.0, None))
    frame = base.with_columns(
        r1=pl.Series(units >= 1),
        r2=pl.Series(units >= 2),
        era=pl.lit("record"),
        observed=pl.lit(value=True),
    )
    return add_drivers(frame=frame, tables=tables)


def control_frame(*, group: list[Job], tables: DriverTables, panel: dict[int, dict]) -> pl.DataFrame:
    """Return the frame of one control series and kind of truth, stacking its stored runs."""
    parts = []
    for job in group:
        run = load_control(job=job)
        index = np.arange(int(run["first"]), int(run["last"]))
        n = len(index)
        g = run["g"]
        part = pl.DataFrame(
            {
                "index": index,
                "era": [job.span] * n,
                "seed": [job.seed] * n,
                "reading": run["y"],
                "truth_mw": g,
                "truth1": units_of(g=g) >= 1,
                "truth2": units_of(g=g) >= 2,
                "ts": run["flag"],
                "tf": run["filtered"],
                "k_path": run["k_path"],
                "p_any": run["p_any"].astype(float),
                "unit_mw": np.full(n, float(run["unit"])),
            }
        )
        parts.append(part)
    frame = pl.concat(parts).with_columns(
        observed=pl.col("reading").is_not_nan(),
        reading=pl.col("reading").fill_nan(None),
        fitted_mw=pl.col("k_path") * pl.col("unit_mw"),
        start=half_hour_starts(index=np.concatenate([p["index"].to_numpy() for p in parts])),
    )
    if group[0].series == "D8":
        weather = {key: panel_array(panel=panel, label="D8", key=key) for key in WEATHER_KEYS}
        frame = frame.with_columns(
            **{key: pl.Series(weather[key][frame["index"].to_numpy()]).fill_nan(None) for key in WEATHER_KEYS}
        )
    frame = add_segments(frame=add_local_fields(frame=frame), group_columns=("seed", "era"))
    return add_drivers(frame=frame, tables=tables)


def control_groups() -> dict[str, list[Job]]:
    """Return the control jobs grouped by series and kind of truth: the name of a frame to its jobs."""
    groups: dict[str, list[Job]] = {}
    for job in jobs():
        key = f"{job.control}_{job.truth}" if job.control != "c2" else f"c2_{job.series.lower()}_{job.truth}"
        if job.control.startswith("c1s"):
            key = f"{job.control}_{job.truth}"
        groups.setdefault(key, []).append(job)
    return groups


def frame_path(*, name: str) -> str:
    """Return the path of a stored frame as text."""
    return str(FRAMES_DIR / f"frame_{name}.parquet")


def load_frame(*, name: str) -> pl.DataFrame:
    """Return a stored frame."""
    return pl.read_parquet(frame_path(name=name))


def main() -> None:
    """Build every frame and print its size and the share of each driver that is missing."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_drivers()
    panel = load_panel()
    frames = {"d8": d8_frame(tables=tables), "record": record_with_drivers(tables=tables)}
    for name, group in control_groups().items():
        frames[name] = control_frame(group=group, tables=tables, panel=panel)
    for name, frame in frames.items():
        frame.write_parquet(frame_path(name=name))
        missing = {
            c: round(float(frame[c].null_count() / frame.height), 3)
            for c in ("system_price", "gas_price_gbp_mwh", "day_ahead_price", "national_wind_mw")
        }
        print(f"{name}: {frame.height} half-hours, missing share {missing}")


if __name__ == "__main__":
    main()
