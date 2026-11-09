"""Write experiment 659's forecasts to the private export folder, one file per arm and origin month.

uv run --no-sync python ml-experiments/00659-*/e00659_export.py
"""

import os
from typing import Final

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import e00659_common as common
import polars as pl

OUT: Final = common.DATA_DIR.parent / "backtest_export_simple_shared_wind_solar_2026-11-09"
README: Final[
    str
] = """# Simple shared wind and solar networks, development-month backtests (experiment 659)

Private: these files hold the time series of NGED metered generators. Do not publish them. Series
are labelled W1 to W3 (wind) and Solar A to Solar F (solar). No file holds a generator's name,
identifier, location, or capacity in megawatts.

One Parquet file per arm and forecast-origin month, `<arm>_<YYYY-MM>.parquet`. The five development
months are 2025-10, 2025-12, 2026-01, 2026-03, and 2026-06. No forecast origin or valid time on or
after 2026-07-01 appears. The final test was not read. The comparison candidates (wind default 2,
solar `a2_member`, and the climatology and reference forecasts) are in
`backtest_export_wind_solar_dev_2026-11-07/`, in the same layout and the same units.

In plain words: each arm is one small neural network per technology (wind, solar), shared by every
farm of that technology. It reads summaries across the 51 ensemble weather forecasts and outputs
a probability distribution, from which the 11 quantiles are read.

- `S` (simple): the average and spread of the raw weather variables only.
- `PLP` (physics-lite plus raw weather): `S` plus the average and spread of a fixed, hand-written
  power estimate computed from each of the 51 weather forecasts separately.
- `PL` (physics-lite only): the average and spread of the power estimate, with no raw weather.
- `S_cal`, `PLP_cal` (wind only): `S` and `PLP` plus hour-of-day and day-of-year columns.

The power estimate for wind is a power curve of the 100 m wind speed (cut-in 3 m/s, rated 12 m/s,
cut-out 25 m/s, cubic between cut-in and rated). For solar it is irradiance over 1000 W/m2, adjusted
by -0.4% per kelvin of cell temperature above 25 C, with the cell temperature from the Faiman model.

Columns: `series`; `init_time` (the 00 UTC weather run the forecast comes from); `valid_time`;
`lead_hours`; `actual`; `q01` to `q99` (the 11 forecast quantiles); `exact_zero` (true where
`actual` is exactly 0). Wind files also hold `actual_raw` (the reading as metered), `masked` (true
where the wind-farm idle mask flagged the half-hour), and `actual` equals `actual_raw` except that
masked half-hours are set to 0 ("cleaned zeros"). The idle mask is imperfect: it missed 51% to 69%
of idle periods out of sample (experiment 210), so some idle noise stays in `actual`. Every power is
a fraction of the scoring capacity (1.0 means 100%; multiply by 100 for points of capacity). The
scoring capacity is the same as in the 2026-11-07 export: for wind, the 99.9th percentile of power
before 2025-10-01; for solar, the older capacity. The networks themselves use a capacity that is the
99.9th percentile of power over the 730 days before each refit.

Refits: wind at day 0 and day 14 of each month; solar at the start of each quarter (2025-09,
2025-12, 2026-03, 2026-06). Quantiles are the average of 3 seeds, sorted across the 11 levels. Solar
rows are daylight half-hours at leads of 12 to 251.5 hours. Wind rows cover leads up to 15 days.
"""


def main() -> None:
    """Split every arm's table by origin month."""
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(common.TABLE_DIR.glob("*.parquet")):
        frame = (
            pl.read_parquet(path)
            .drop("tech", "candidate")
            .with_columns(
                exact_zero=pl.col("actual") == 0.0, month=pl.col("init_time").dt.strftime("%Y-%m")
            )
        )
        if "solar" in path.stem:
            frame = frame.drop("masked")
        for month in common.DEV_MONTHS:
            part = frame.filter(pl.col("month") == month).drop("month")
            if part.height:
                part.write_parquet(OUT / f"{path.stem}_{month}.parquet")
    (OUT / "README.md").write_text(README)
    print(len(list(OUT.glob("*.parquet"))), "files in", OUT)


if __name__ == "__main__":
    main()
