"""Experiment 453, steps 1 and 2: describe the upper tail and classify half-hours above p99.

Power is divided by the stored effective capacity (the 99th percentile of the whole cleaned record),
so 1.0 is the current p99. A half-hour above 1.0 is flagged: `spike` (above both neighbours by more
than `SPIKE_JUMP`), `stuck` (part of a run of `STUCK_RUN` or more equal readings), `step` (the
half-hour's calendar month is a high-level month: the 90th percentile of the series' output in the
month, among half-hours with strong wind or strong clear-sky irradiance, exceeds the median of all
such monthly values by more than `STEP_JUMP`), and `weather_off` (wind: ERA5 100 m wind speed,
where it exists, below the speed at which the farm's median output reaches 0.5 of capacity; solar: Haurwitz clear-sky irradiance below half of its 99th percentile over daylight).
It is `implausible` if any of the four flags holds, otherwise `plausible`. Output:
`work/tail_summary.parquet`, `work/high_halfhours.parquet`.

```bash
export DATA_PATH_INTERNAL=...; source ml-experiments/00451-*/env.sh
uv run --no-sync python ml-experiments/00453-*/e00453_tail.py
```
"""

from typing import Final

import e00453_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

THRESHOLDS: Final[dict[str, float]] = {"max": 1.0, "p99.99": 0.9999, "p99.9": 0.999, "p99": 0.99}
SPIKE_JUMP: Final[float] = 0.2
STUCK_RUN: Final[int] = 4
STEP_JUMP: Final[float] = 0.10
MIN_MONTH_ROWS: Final[int] = 20
STRONG_WIND_M_S: Final[float] = 14.0
STRONG_CLEAR_FRACTION: Final[float] = 0.8
RATED_LEVEL: Final[float] = 0.5
CLEAR_FRACTION: Final[float] = 0.5
NEIGHBOUR_LEVEL: Final[float] = 0.5
HALF_HOUR_US: Final[int] = 30 * 60 * 1_000_000


def prepare(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return one site's rows on a complete half-hourly grid (gaps become null), with `x`."""
    site = frame
    start, end = site["time"].min(), site["time"].max()
    grid = pl.DataFrame({"time": pl.datetime_range(start, end, "30m", eager=True)})
    return grid.join(site, on="time", how="left").with_columns(
        x=pl.col("power") / pl.col("cap"), site=pl.col("site").forward_fill().backward_fill()
    )


def flags(*, full: pl.DataFrame) -> pl.DataFrame:
    """Return `full` with the spike and stuck flags and the neighbouring values."""
    x = full["x"].to_numpy()
    power = full["power"].to_numpy()
    prev_x = np.concatenate([[np.nan], x[:-1]])
    next_x = np.concatenate([x[1:], [np.nan]])
    spike = ((x - prev_x) > SPIKE_JUMP) & ((x - next_x) > SPIKE_JUMP)
    # runs of equal consecutive readings (null breaks a run)
    same_prev = np.concatenate([[False], (power[1:] == power[:-1]) & ~np.isnan(power[1:])])
    run_id = np.cumsum(~same_prev)
    run_len = pl.DataFrame({"r": run_id}).with_columns(n=pl.len().over("r"))["n"].to_numpy()
    stuck = (run_len >= STUCK_RUN) & ~np.isnan(power)
    return full.with_columns(
        spike=pl.Series(np.nan_to_num(spike, nan=0.0).astype(bool)),
        stuck=pl.Series(stuck),
        prev_x=pl.Series(prev_x),
        next_x=pl.Series(next_x),
    )


def high_level_months(*, part: pl.DataFrame, tech: str) -> pl.DataFrame:
    """Return each month's strong-resource output level and whether it is a high-level month."""
    if tech == "wind":
        strong = part.filter(pl.col("wind") >= STRONG_WIND_M_S)
    else:
        daylight = part.filter(pl.col("clear") > 0)["clear"].to_numpy()
        strong = part.filter(pl.col("clear") >= STRONG_CLEAR_FRACTION * np.quantile(daylight, 0.99))
    monthly = (
        strong.with_columns(month=pl.col("time").dt.truncate("1mo"))
        .group_by("month")
        .agg(pl.col("x").quantile(0.9).alias("level"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_MONTH_ROWS)
    )
    median = float(monthly["level"].median())
    return monthly.with_columns(step=pl.col("level") > median + STEP_JUMP).select(
        "month", "level", "step"
    )


def weather_thresholds(*, series: pl.DataFrame) -> dict[str, float]:
    """Return each wind farm's wind speed at which the median output reaches `RATED_LEVEL`."""
    out: dict[str, float] = {}
    wind = series.filter((pl.col("tech") == "wind") & pl.col("wind").is_not_null())
    for site in wind["site"].unique().sort():
        part = wind.filter(pl.col("site") == site).with_columns(
            x=pl.col("power") / pl.col("cap"), speed=(pl.col("wind") * 2).round() / 2
        )
        med = part.group_by("speed").agg(pl.col("x").median().alias("m"), pl.len().alias("n"))
        ok = med.filter((pl.col("m") >= RATED_LEVEL) & (pl.col("n") >= 50)).sort("speed")
        out[site] = float(ok["speed"][0])
    return out


def run() -> None:
    """Write the tail summary and the classified high half-hours."""
    series = pl.read_parquet(data.SERIES_PATH)
    speed_needed = weather_thresholds(series=series)
    print("wind speed at which median output reaches 0.5 (m/s):", speed_needed)
    summaries, highs = [], []
    for site in series["site"].unique().sort():
        part = series.filter(pl.col("site") == site)
        tech = part["tech"][0]
        full = flags(full=prepare(frame=part)).with_columns(tech=pl.lit(tech))
        months = high_level_months(part=full.filter(pl.col("x").is_not_null()), tech=tech)
        months.with_columns(site=pl.lit(site)).write_parquet(
            data.WORK_DIR / f"months_{site.replace(' ', '_')}.parquet"
        )
        full = (
            full.with_columns(month=pl.col("time").dt.truncate("1mo"))
            .join(months.select("month", "step"), on="month", how="left")
            .with_columns(step=pl.col("step").fill_null(value=False))
        )
        valid = full.filter(pl.col("x").is_not_null())
        x = valid["x"].to_numpy()
        row = {"site": site, "tech": tech, "half_hours": len(x), "cap_mw": float(part["cap"][0])}
        row["max"] = float(x.max())
        for name, q in THRESHOLDS.items():
            if name != "max":
                row[name] = float(np.quantile(x, q))
        for name in THRESHOLDS:
            row[f"n_above_{name}"] = int((x > row[name] + 1e-9).sum())
        row["n_above_1"] = int((x > 1.0).sum())
        summaries.append(row)
        high = valid.filter(pl.col("x") > 1.0)
        if tech == "wind":
            weather_off = high["wind"].is_not_null() & (high["wind"] < speed_needed[site])
            high = high.with_columns(weather_off=weather_off)
        else:
            clear_p99 = np.quantile(valid.filter(pl.col("clear") > 0)["clear"].to_numpy(), 0.99)
            high = high.with_columns(weather_off=pl.col("clear") < CLEAR_FRACTION * clear_p99)
        highs.append(high)
    summary = pl.DataFrame(summaries)
    high = pl.concat(highs, how="diagonal_relaxed")
    # support from the other series of the same technology at the same time
    pivot = (
        series.with_columns(x=pl.col("power") / pl.col("cap"))
        .group_by("tech", "time")
        .agg(pl.col("x").max().alias("best"), pl.len().alias("n_series"))
    )
    high = high.join(pivot, on=["tech", "time"], how="left")
    high = high.with_columns(
        implausible=pl.col("spike") | pl.col("stuck") | pl.col("step") | pl.col("weather_off")
    ).select(
        "site", "tech", "time", "x", "prev_x", "next_x", "wind", "clear", "spike", "stuck", "step",
        "weather_off", "implausible",
    )  # fmt: skip
    data.WORK_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(data.WORK_DIR / "tail_summary.parquet")
    high.write_parquet(data.WORK_DIR / "high_halfhours.parquet")
    with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=250):
        print(summary)
        print(
            high.group_by("site")
            .agg(
                pl.len().alias("n"),
                pl.col("spike").sum().alias("spike"),
                pl.col("stuck").sum().alias("stuck"),
                pl.col("step").sum().alias("step"),
                pl.col("weather_off").sum().alias("weather_off"),
                pl.col("implausible").sum().alias("implausible"),
            )
            .sort("site")
        )


if __name__ == "__main__":
    run()
