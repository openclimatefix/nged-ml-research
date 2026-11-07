# ruff: noqa: E501
"""Write the development-month forecasts to the private export folder, one file per candidate and month.

source <env>; uv run --no-sync python ml-experiments/00615-*/e00615_export.py
"""

import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00615_common as common
import polars as pl

OUT: Final = common.E1 / "backtest_export_wind_solar_dev_2026-11-07"
README: Final[str] = """# Wind and solar development-month backtests (experiment 615)

Private: these files hold the time series of NGED metered generators. Do not publish them. Series
are labelled W1 to W3 (wind) and Solar A to Solar F (solar); no file holds a generator's name,
identifier, location, or capacity in megawatts.

One Parquet file per candidate and forecast-origin month, `<candidate>_<YYYY-MM>.parquet`. The five
development months are 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06. No forecast origin on or
after 2026-07-01 and no valid time on or after 2026-07-01 appears. The final test is not run.

Columns: `series`; `init_time` (the 00 UTC weather-model run the forecast is issued from);
`valid_time`; `lead_hours`; `actual` (the observed power); `q01` to `q99` (the 11 forecast
quantiles); `exact_zero` (true where `actual` is exactly 0). Every power is a fraction of the
scoring capacity (1.0 means 100%), so multiply by 100 for points of capacity. The scoring capacity
is the same for every candidate of a series: the series' older capacity for solar, and the 99.9th
percentile of power before 2025-10-01 for wind.

Candidates:

- `wind_default2`: experiment 347's physics plant and mixture-density head, refitted at day 0 and day
  14 of each month, seeds 0 and 1 averaged quantile by quantile, capacity at the 99.9th percentile of
  earlier power. Rows with a missing reading or dropped by the idle mask are absent.
- `wind_climatology`: per farm, month, and hour quantiles of power before 2025-10-01.
- `wind_default1_stored_tuesdays` and `wind_default2_stored_tuesdays`: experiment 303's and 347's
  stored forecasts, available on the Tuesday forecast origins only.
- `solar_a2_member_quarterly` and `solar_a2_member_annual`: experiment 430's and 493's stored
  networks (six seeds averaged), refitted every quarter and once a year.
- `solar_gam`, `solar_xgboost`: experiment 310's stored references (seed 0).
- `solar_climatology`, `solar_manual_heuristic`: experiment 310's stored baselines.

Solar rows are daylight half-hours at leads of 12 to 251.5 hours. Wind rows cover leads up to 15
days. Solar refits are quarterly (day 0 of the quarter) because those are the stored forecasts; they
were not refitted at day 14.
"""


def main() -> None:
    """Split every assembled table by origin month."""
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(common.TABLE_DIR.glob("*.parquet")):
        frame = (
            pl.read_parquet(path)
            .drop("tech")
            .with_columns(
                exact_zero=pl.col("actual") == 0.0, month=pl.col("init_time").dt.strftime("%Y-%m")
            )
        )
        for month in common.DEV_MONTHS:
            part = frame.filter(pl.col("month") == month).drop("month")
            if part.height:
                part.drop("candidate").write_parquet(OUT / f"{path.stem}_{month}.parquet")
    (OUT / "README.md").write_text(README)
    print(len(list(OUT.glob("*.parquet"))), "files in", OUT)


if __name__ == "__main__":
    main()
