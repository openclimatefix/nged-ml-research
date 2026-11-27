# ruff: noqa: E501, PT018
"""Experiment 725: the live rule for the history weather, as index arithmetic with no data access.

A history token is one of the 14 x 48 half-hours of the matrix of experiment 713. Its 21 weather
columns fall into five groups, and the live rule fixes which source each group reads in each cell
(design section 2.2):

- columns 0 and 15 to 18 (shortwave irradiance and the four trailing-irradiance columns) read the CAMS
  analysis where the token's CAMS source stamp is at or before 00:00 UTC on day D - 2, otherwise the
  ensemble (ENS) value;
- columns 1 to 14 (the other six weather columns and the eight trailing-temperature columns) read the
  ERA5 analysis from age 7 on, otherwise the ENS value;
- columns 19 and 20 (the CAMS day summary) read the summary of the earlier of the day before the
  token's stamp day and day D - 3.

`cell_sources` computes the table on the grid. `reference_table` computes the same table a second time
from `datetime` arithmetic with no grid index (check KL2).
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Final, Literal
from zoneinfo import ZoneInfo

import numpy as np

STEPS_PER_DAY: Final = 48
N_ROWS: Final = 14
FIRST_AGE: Final = 2
N_WEATHER_COLUMNS: Final = 21
N_ENS_COLUMNS: Final = 19
"""Columns 0 to 18 of a token can read the ENS value: 7 weather columns and the 12 trailing columns."""
CAMS_COLUMNS: Final = (0, 15, 16, 17, 18)
ERA5_COLUMNS: Final = tuple(range(1, 15))
DAY_COLUMNS: Final = (19, 20)
NATIVE_OFFSET_STEPS: Final = 6
"""Grid steps from the native origin (09:00 UTC) to the contract origin (12:00 UTC)."""
ENS_LAG_HOURS: Final = 9
CAMS_LAG_HOURS: Final = 36
ERA5_LAG_HOURS: Final = 120
ERA5_FIRST_AGE: Final = 7
"""Age of the newest row whose every token and local-day extreme is at least 120 h older than 09:00 UTC."""
CAMS_LAST_DAY_BEFORE: Final = 3
"""The newest CAMS UTC day in hand at the native origin on day D is D - 3."""
LOCAL_ZONE: Final = ZoneInfo("Europe/London")
CHANGED_VALUES_PER_SAMPLE: Final = 3763
"""Weather values of a history matrix whose source the rule changes (design section 2.2)."""
REFERENCE_DAY: Final = 100
"""Grid day number used to evaluate the table; the table does not depend on the day."""

SRC_ERA5: Final = 0
SRC_CAMS: Final = 1
SRC_ENS: Final = 2
SRC_DAY: Final = 3
"""The CAMS day summary of the day before the token's stamp day (the analysis array, unchanged)."""
SRC_DAY_D3: Final = 4
"""The CAMS day summary of UTC day D - 3, which replaces a later day."""
RuleName = Literal["L", "Z", "E"]
RULES: Final = ("L", "Z", "E")


def ages() -> np.ndarray:
    """Return the ages 2 to 15 of the 14 history rows, newest first."""
    return np.arange(FIRST_AGE, FIRST_AGE + N_ROWS, dtype=np.int64)


def token_grid_index(*, day: int = REFERENCE_DAY) -> np.ndarray:
    """Return the (14, 48) grid indices of the history tokens of the contract origin of grid day `day`.

    The contract origin is 12:00 UTC, grid index `48 day + 24`, and the grid starts at 00:00 UTC.
    """
    c = STEPS_PER_DAY * day + STEPS_PER_DAY // 2
    return c - STEPS_PER_DAY * ages()[:, None] + np.arange(STEPS_PER_DAY)[None, :]


def cams_source_index(*, grid_index: np.ndarray) -> np.ndarray:
    """Return the grid index of the CAMS source stamp behind each grid half-hour.

    CAMS is hourly with period-end stamps, and the dense array copies each hourly value to both
    half-hours of its hour, so the half-hour stamped HH:30 holds the CAMS value stamped (HH + 1):00.
    The grid starts at 00:00 UTC, so an odd index is a half-hour stamped HH:30.
    """
    return grid_index + (grid_index % 2)


def cell_sources(*, rule: RuleName) -> np.ndarray:
    """Return the (14, 48, 21) int8 table of sources for `rule`.

    Rule `Z` reads the analysis in every cell, `L` is the live rule, and `E` reads the ENS value in
    columns 0 to 18 at every age (and the live rule's day summary in columns 19 and 20).
    """
    assert rule in RULES, rule
    day = REFERENCE_DAY
    g = token_grid_index(day=day)
    out = np.empty((N_ROWS, STEPS_PER_DAY, N_WEATHER_COLUMNS), dtype=np.int8)
    out[..., list(CAMS_COLUMNS)] = SRC_CAMS
    out[..., list(ERA5_COLUMNS)] = SRC_ERA5
    out[..., list(DAY_COLUMNS)] = SRC_DAY
    if rule == "Z":
        return out
    cams_ok = cams_source_index(grid_index=g) <= STEPS_PER_DAY * (day - 2)
    era5_ok = np.broadcast_to((ages() >= ERA5_FIRST_AGE)[:, None], g.shape)
    for col in CAMS_COLUMNS:
        out[..., col] = np.where(cams_ok, SRC_CAMS, SRC_ENS)
    for col in ERA5_COLUMNS:
        out[..., col] = np.where(era5_ok, SRC_ERA5, SRC_ENS)
    if rule == "E":
        out[..., :N_ENS_COLUMNS] = SRC_ENS
    later = (g // STEPS_PER_DAY - 1) > (day - CAMS_LAST_DAY_BEFORE)
    out[..., list(DAY_COLUMNS)] = np.where(later, SRC_DAY_D3, SRC_DAY)[..., None]
    return out


def changed_cells(*, rule: RuleName) -> np.ndarray:
    """Return the (14, 48, 21) bool mask of the values whose source differs from the analysis (rule `Z`)."""
    return cell_sources(rule=rule) != cell_sources(rule="Z")


def ens_mask(*, rule: RuleName) -> np.ndarray:
    """Return the (14, 48, 19) bool mask of the cells that read the ENS value."""
    return cell_sources(rule=rule)[..., :N_ENS_COLUMNS] == SRC_ENS


def summary_day_read(
    *, token_day: np.ndarray, contract_day: np.ndarray | int, live: bool
) -> np.ndarray:
    """Return the grid day number of the UTC day whose CAMS summary a token reads.

    Args:
        token_day: Grid day number of each token's stamp (`grid index // 48`).
        contract_day: Grid day number D of the contract origin.
        live: True for the live rule (the earlier of the day before the stamp day and D - 3); False
            for the analysis array (the day before the stamp day).

    Returns:
        The grid day number of the summary read, the shape of `token_day` broadcast with `contract_day`.
    """
    previous = np.asarray(token_day) - 1
    if not live:
        return previous + 0 * np.asarray(contract_day)
    return np.minimum(previous, np.asarray(contract_day) - CAMS_LAST_DAY_BEFORE)


# ---------------------------------------------------------------------------------- the reference


def _local_day_end(*, stamp: datetime) -> datetime:
    """Return the UTC instant of the next local (Europe/London) midnight after the half-hour that ends at `stamp`."""
    local = (stamp - timedelta(minutes=30)).astimezone(LOCAL_ZONE)
    midnight = datetime.combine(local.date() + timedelta(days=1), time(0), tzinfo=LOCAL_ZONE)
    return midnight.astimezone(UTC)


def reference_table(*, contract: datetime, rule: RuleName) -> np.ndarray:
    """Return the (14, 48, 21) table for the contract origin `contract` from `datetime` arithmetic alone.

    No grid index appears: each token's stamp is `contract - age days + column half-hours`, the native
    origin is `contract - 3 h`, a CAMS value is in hand when its hourly source stamp is at or before
    00:00 UTC on day D - 2, an ERA5 row is in hand when every token of the row and the end of every
    token's local day is at least 120 h before the native origin, and the day summary read is the
    earlier of the UTC day before the token's stamp day and D - 3.

    Args:
        contract: The 12:00 UTC contract origin as an aware UTC datetime.
        rule: `L`, `Z`, or `E`.

    Returns:
        The table with the codes of this module.
    """
    assert contract.tzinfo is not None and contract.hour == 12 and contract.minute == 0
    native = contract - timedelta(hours=3)
    day_d: date = contract.date()
    cams_limit = datetime.combine(day_d - timedelta(days=2), time(0), tzinfo=UTC)
    era5_limit = native - timedelta(hours=ERA5_LAG_HOURS)
    out = np.empty((N_ROWS, STEPS_PER_DAY, N_WEATHER_COLUMNS), dtype=np.int8)
    for slot, age in enumerate(range(FIRST_AGE, FIRST_AGE + N_ROWS)):
        stamps = [
            contract - timedelta(days=age) + timedelta(minutes=30 * j) for j in range(STEPS_PER_DAY)
        ]
        row_era5_in_hand = all(max(s, _local_day_end(stamp=s)) <= era5_limit for s in stamps)
        for j, stamp in enumerate(stamps):
            cams_stamp = stamp if stamp.minute == 0 else stamp + timedelta(minutes=30)
            cams_in_hand = cams_stamp <= cams_limit
            for col in CAMS_COLUMNS:
                out[slot, j, col] = SRC_CAMS if cams_in_hand and rule != "E" else SRC_ENS
                if rule == "Z":
                    out[slot, j, col] = SRC_CAMS
            for col in ERA5_COLUMNS:
                out[slot, j, col] = SRC_ERA5 if row_era5_in_hand and rule != "E" else SRC_ENS
                if rule == "Z":
                    out[slot, j, col] = SRC_ERA5
            read_day = stamp.date() - timedelta(days=1)
            later = read_day > day_d - timedelta(days=CAMS_LAST_DAY_BEFORE)
            for col in DAY_COLUMNS:
                out[slot, j, col] = SRC_DAY_D3 if later and rule != "Z" else SRC_DAY
    return out
