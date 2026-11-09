"""Score `wind.parquet` and `solar.parquet` against observed power, which is not in the files.

Observed power is the cleaned power of `studies.power.scan_power`, divided by the series' stored
effective capacity so that scores are in points of capacity (100 times a fraction). For wind, the
rows of experiment 210's idle mask (to 2026-09-13) count as zero, as in the fits' target. Rows with
no observed power are left out. The columns are:

- `score`: the 11-level quantile score, the mean pinball loss over the 11 levels, in points of
  capacity (the "CRPS" of experiments 615 and 659).
- `mae_median`: the absolute error of `q50`, in points of capacity.
- `inside_q05_q95`: the share of rows with the observation between `q05` and `q95`.
- `above_q99` and `below_q01`: the shares beyond the outer quantiles.

Bands are the hours from the origin (12:00 UTC) to the valid time: 0-1, 1-3, 3-7, 7-10, and 10-14
days. Tables are written to `scores/` in this experiment's data folder. Solar is scored on all
half-hours (night half-hours score zero) and on daylight half-hours only. A climatology for solar
(the quantiles of the farm's earlier power at the same calendar month and half-hour of day, taken
before each refit date) is scored on the same rows.

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    uv run --no-sync python ml-experiments/00666-*/e00666_score.py
"""

import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00666_common as common
import e00666_solar_fit as solar_fit
import e00666_wind_fit as wind_fit
import numpy as np
import polars as pl
from studies.power import scan_power
from studies.pv_dataset import pv_sites, wind_sites

LEVELS: Final = np.array(common.LEVELS)
BANDS: Final[dict[str, tuple[float, float]]] = {
    "0-1 d": (0.0, 1.0),
    "1-3 d": (1.0, 3.0),
    "3-7 d": (3.0, 7.0),
    "7-10 d": (7.0, 10.0),
    "7-14 d": (7.0, 14.0),
    "10-14 d": (10.0, 14.0),
}
SCORES_DIR: Final = common.DATA_DIR / "scores"


def observed(*, sites: pl.DataFrame, mask: bool) -> pl.DataFrame:
    """Return `time_series_id`, `valid_time`, and the observed power as a fraction of capacity."""
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", valid_time="time", power="power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .select(
            "time_series_id", "valid_time", actual=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )
    if mask:
        idle = (
            pl.read_parquet(wind_fit.MASK_PATH)
            .select("time_series_id", "valid_time")
            .with_columns(masked=pl.lit(value=True))
        )
        power = (
            power.join(idle, on=["time_series_id", "valid_time"], how="left")
            .with_columns(
                actual=pl.when(pl.col("masked").fill_null(value=False))
                .then(0.0)
                .otherwise(pl.col("actual"))
            )
            .drop("masked")
        )
    return power.filter(pl.col("actual").is_finite())


def losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the frame with the per-row score columns, in points of capacity."""
    y = frame["actual"].to_numpy().astype(np.float64)
    q = np.stack(
        [
            frame[name].to_numpy().astype(np.float64) / frame["capacity_mw"].to_numpy()
            for name in common.QUANTILES
        ],
        axis=1,
    )
    d = y[:, None] - q
    return frame.select("time_series_id", "month", "band").with_columns(
        overall=pl.lit("all"),
        score=pl.Series(100 * np.maximum(LEVELS * d, (LEVELS - 1) * d).mean(axis=1)),
        mae_median=pl.Series(100 * np.abs(d[:, 5])),
        inside_q05_q95=pl.Series(((y >= q[:, 1]) & (y <= q[:, 9])).astype(float)),
        above_q99=pl.Series((y > q[:, 10]).astype(float)),
        below_q01=pl.Series((y < q[:, 0]).astype(float)),
    )


def prepare(*, forecast: pl.DataFrame, sites: pl.DataFrame, mask: bool) -> pl.DataFrame:
    """Return the forecast rows with the observation, capacity, origin month, and lead band."""
    days = (
        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 1440.0
    ).alias("days")
    joined = (
        forecast.join(
            observed(sites=sites, mask=mask), on=["time_series_id", "valid_time"], how="inner"
        )
        .join(
            sites.select("time_series_id", capacity_mw="effective_capacity_mw").with_columns(
                pl.col("time_series_id").cast(pl.Int8)
            ),
            on="time_series_id",
        )
        .with_columns(days, month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"))
    )
    band = pl.lit(None, dtype=pl.String)
    for name, (low, high) in BANDS.items():
        band = (
            pl.when((pl.col("days") >= low) & (pl.col("days") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    return joined.with_columns(band=band)


def summarise(*, frame: pl.DataFrame, by: str) -> pl.DataFrame:
    """Return the mean of every score column by `by`, with the row count."""
    return (
        losses(frame=frame)
        .group_by(by)
        .agg(
            rows=pl.len(),
            **{
                c: pl.col(c).mean()
                for c in ("score", "mae_median", "inside_q05_q95", "above_q99", "below_q01")
            },
        )
        .sort(by)
    )


def climatology(*, sites: pl.DataFrame, forecast: pl.DataFrame) -> pl.DataFrame:
    """Return the rows with quantiles of earlier power at the same month and half-hour."""
    ids = sites["time_series_id"].to_list()
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(ids))
        .select("time_series_id", "time", "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .with_columns(
            pl.col("time_series_id").cast(pl.Int8),
            fraction=pl.col("power") / pl.col("effective_capacity_mw"),
        )
    )
    pieces = []
    for refit in solar_fit.REFITS:
        end = solar_fit.month_add(start=refit, months=solar_fit.REFIT_MONTHS)
        history = (
            power.filter(pl.col("time") < refit)
            .with_columns(
                month_of_year=pl.col("time").dt.month(),
                half_hour=pl.col("time").dt.hour() * 2 + pl.col("time").dt.minute() // 30,
            )
            .group_by("time_series_id", "month_of_year", "half_hour")
            .agg(
                **{
                    q: pl.col("fraction").quantile(level)
                    for q, level in zip(common.QUANTILES, common.LEVELS, strict=True)
                }
            )
        )
        rows = forecast.filter(
            (pl.col("nwp_init_time") >= refit) & (pl.col("nwp_init_time") < end)
        ).with_columns(
            month_of_year=pl.col("valid_time").dt.month(),
            half_hour=pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30,
        )
        rows = rows.drop(list(common.QUANTILES)).join(
            history, on=["time_series_id", "month_of_year", "half_hour"], how="left"
        )
        capacity = sites.select(pl.col("time_series_id").cast(pl.Int8), "effective_capacity_mw")
        rows = rows.join(capacity, on="time_series_id").with_columns(
            pl.col(common.QUANTILES) * pl.col("effective_capacity_mw")
        )
        pieces.append(rows.drop("month_of_year", "half_hour", "effective_capacity_mw"))
    return pl.concat(pieces)


def report(*, name: str, frame: pl.DataFrame) -> None:
    """Print and write the by-band and by-month tables of one scored frame."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for by in ("band", "month"):
        table = summarise(frame=frame, by=by)
        table.write_csv(SCORES_DIR / f"{name}_by_{by}.csv")
        print(f"\n{name} by {by}\n{table}", flush=True)
    overall = summarise(frame=frame, by="overall")
    overall.write_csv(SCORES_DIR / f"{name}_overall.csv")
    print(f"\n{name} overall\n{overall}", flush=True)


def main() -> None:
    """Score both files."""
    wind_sites_frame = wind_sites()
    wind = pl.read_parquet(common.OUTPUT_DIR / "wind.parquet")
    report(name="wind", frame=prepare(forecast=wind, sites=wind_sites_frame, mask=True))

    solar_sites = pv_sites()
    solar = pl.read_parquet(common.OUTPUT_DIR / "solar.parquet")
    report(
        name="solar_all_half_hours", frame=prepare(forecast=solar, sites=solar_sites, mask=False)
    )
    flags = pl.concat(
        [
            pl.read_parquet(common.SOLAR_DIR / f"{s}.parquet").select(
                pl.col("time_series_id").cast(pl.Int8),
                nwp_init_time="init_time",
                valid_time="time",
                daylight="daylight",
            )
            for s in solar_sites["site"].to_list()
        ]
    )
    solar_flagged = solar.join(
        flags, on=["time_series_id", "nwp_init_time", "valid_time"], how="left", validate="1:1"
    )
    daylight = solar_flagged.filter(pl.col("daylight")).drop("daylight")
    report(name="solar_daylight", frame=prepare(forecast=daylight, sites=solar_sites, mask=False))
    clim = climatology(sites=solar_sites, forecast=daylight)
    report(
        name="solar_daylight_climatology",
        frame=prepare(forecast=clim, sites=solar_sites, mask=False),
    )


if __name__ == "__main__":
    main()
