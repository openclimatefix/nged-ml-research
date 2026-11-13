"""Loaders for the public drivers of experiment 697: prices, gas, wind, holidays, and drops.

Every loader returns a table keyed on the **start** of its period, so a join with a power-derived
half-hour start needs no further shift. The system-price, wind, and gas loaders copy the private
`predictability_work/data.py` of the engine-state follow-on, with the changes the design asks for.
Nothing here reads a power stamp.
"""

from datetime import date
from typing import Final

import holidays
import polars as pl
from studies.sources import MARKET_DOWNLOADS_DIR, ML_EXPERIMENTS_DATA_DIR

SYSTEM_PRICE_DIRS: Final[tuple[str, ...]] = (
    "00186-*/public/system_prices",
    "00185-*/public/system_prices",
    "00177-*/public/system_prices",
)
"""Experiments 186, 185, and 177 each hold a span of Elexon's published system prices."""
WIND_DIR: Final[str] = "00185-*/public/wind_fuelhh"
GAS_PATH_GLOB: Final[str] = "00186-*/public/gas_sap.csv"
DAY_AHEAD_PATH: Final = MARKET_DOWNLOADS_DIR / "neso_n2ex_day_ahead" / "neso_n2ex_day_ahead.parquet"
DROPS_GLOB: Final[str] = "00159-*/drops.parquet"
EXPECTED_SYSTEM_PRICE_ROWS: Final[int] = 54096
GAS_BAD_VALUES: Final[tuple[float, ...]] = (72.8, 63.9, 0.51, 0.47)
"""Pence per kWh. Experiment 186 lists these four daily values as implausible."""
GAS_MIN_P_PER_KWH: Final[float] = 1.0
GAS_MAX_P_PER_KWH: Final[float] = 20.0
GAS_EFFICIENCY: Final[float] = 0.40
"""Electrical efficiency that converts a gas price per MWh of gas to a cost per MWh of power."""
DROPS_FROM: Final[date] = date(2023, 7, 1)
EXPECTED_DROPS: Final[int] = 28
CHRISTMAS_START: Final[tuple[int, int]] = (12, 24)
CHRISTMAS_END: Final[tuple[int, int]] = (1, 1)


def system_prices() -> pl.DataFrame:
    """Return Elexon's half-hourly system sell price at the start of each settlement period.

    Columns: `start` (UTC) and `system_price` (pounds per MWh). The three experiments' files are
    concatenated and a repeated start keeps one row.
    """
    files = [
        f
        for pattern in SYSTEM_PRICE_DIRS
        for directory in sorted(ML_EXPERIMENTS_DATA_DIR.glob(pattern))
        for f in sorted(directory.glob("*.parquet"))
    ]
    frame = pl.concat([pl.read_parquet(f).select("startTime", "systemSellPrice") for f in files])
    return (
        frame.select(
            start=pl.col("startTime").str.to_datetime(time_zone="UTC", time_unit="us"),
            system_price=pl.col("systemSellPrice").cast(pl.Float64),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def national_wind() -> pl.DataFrame:
    """Return national transmission wind generation (MW) at the start of each half-hour."""
    files = [f for d in sorted(ML_EXPERIMENTS_DATA_DIR.glob(WIND_DIR)) for f in sorted(d.glob("*.parquet"))]
    frame = pl.concat([pl.read_parquet(f) for f in files])
    return (
        frame.select(
            start=pl.col("startTime").str.to_datetime(time_zone="UTC", time_unit="us"),
            national_wind_mw=pl.col("generation").cast(pl.Float64),
        )
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def gas_prices() -> pl.DataFrame:
    """Return the National Gas system average price per gas day, in pounds per MWh of gas.

    Columns: `gas_day` (the date the gas day starts on) and `gas_price_gbp_mwh`. The four values
    that experiment 186 lists as implausible and any value outside 1 to 20 pence per kWh are
    dropped.
    """
    path = next(ML_EXPERIMENTS_DATA_DIR.glob(GAS_PATH_GLOB))
    raw = pl.read_csv(path).select(
        gas_day=pl.col("Applicable For").str.to_date("%d/%m/%Y"),
        pence_per_kwh=pl.col("Value").cast(pl.Float64, strict=False),
    )
    bad = pl.lit(value=False)
    for value in GAS_BAD_VALUES:
        bad = bad | ((pl.col("pence_per_kwh") - value).abs() < 1e-6)
    return (
        raw.filter(
            ~bad,
            pl.col("pence_per_kwh").is_between(GAS_MIN_P_PER_KWH, GAS_MAX_P_PER_KWH),
        )
        .unique(subset="gas_day", keep="first", maintain_order=True)
        .select("gas_day", gas_price_gbp_mwh=pl.col("pence_per_kwh") * 10.0)
        .sort("gas_day")
    )


def day_ahead_prices() -> pl.DataFrame:
    """Return NESO's N2EX day-ahead price (pounds per MWh) at the start of each delivery hour."""
    return pl.read_parquet(DAY_AHEAD_PATH).select(
        hour=pl.col("time").dt.cast_time_unit("us"),
        day_ahead_price=pl.col("price_gbp_per_mwh"),
    )


def bank_holiday_dates() -> frozenset[date]:
    """Return England's bank holidays, from the `holidays` package, for 2021 to 2027."""
    return frozenset(holidays.country_holidays("GB", subdiv="ENG", years=range(2021, 2028)))


def christmas_week_expression(*, local_date: str = "local_date") -> pl.Expr:
    """Return an expression that is true on 24 December to 1 January."""
    month, day = pl.col(local_date).dt.month(), pl.col(local_date).dt.day()
    return ((month == CHRISTMAS_START[0]) & (day >= CHRISTMAS_START[1])) | (
        (month == CHRISTMAS_END[0]) & (day <= CHRISTMAS_END[1])
    )


def drops() -> pl.DataFrame:
    """Return experiment 159's multi-day drops of D8 from 2023-07 on, with panel-grid stamps.

    Columns: `first_stamp` and `last_stamp` (panel-grid indices), `start_date`, `end_date`.
    """
    path = next(ML_EXPERIMENTS_DATA_DIR.glob(DROPS_GLOB))
    return pl.read_parquet(path).filter(pl.col("start_date") >= DROPS_FROM).select(
        "first_stamp", "last_stamp", "start_date", "end_date"
    )


def spark_spread_expression() -> pl.Expr:
    """Return system price minus the gas price per MWh of gas divided by the efficiency."""
    return pl.col("system_price") - pl.col("gas_price_gbp_mwh") / GAS_EFFICIENCY
