"""The three label-only checks of D8's tracker labels, by era, with and without the drops.

The checks read only the stored frame of D8 (`frame_d8.parquet` of experiment 697): the tracker's
smoothed labels `ts`, CAMS irradiance `ghi`, the spark spread, the bank-holiday flag, and the
multi-day drops. **None of them shows that solar generation is the cause of a change between eras.**

- `weekend_cheap`: the flagged share on weekends and bank holidays, 09:00 to 15:00 local, with a
  negative spark spread.
- `night`: the flagged share from 00:00 to 05:00 local.
- `bmd`: the flagged share in the brightest third of CAMS irradiance minus the dullest third,
  10:00 to 15:00 local, above 50 W/m2. Thirds are taken within an era, a local half-hour, a weekday
  class, and a month, among the rows left after the drops are removed (when they are).

Intervals resample whole ISO weeks (1,000 resamples), and the swing between the eras resamples the
two eras independently, as experiment 697's differences do.
"""

import zlib
from collections.abc import Callable
from typing import Final

import numpy as np
import polars as pl
from e00697_common import (  # ty: ignore[unresolved-import]
    Estimate,
    difference_estimate,
    group_within,
    ratio,
    share_columns,
    share_difference_statistic,
    week_estimate,
)
from e00698_common import (  # ty: ignore[unresolved-import]
    BOOTSTRAP_SEED,
    GHI_MIN,
    LABEL_ONLY_RESAMPLES,
    MIDDAY_FIRST,
    MIDDAY_LAST,
    NIGHT_LAST,
)

CHECKS: Final[tuple[str, ...]] = ("weekend_cheap", "night", "bmd")
CHECK_TITLES: Final[dict[str, str]] = {
    "weekend_cheap": (
        "Flagged share, weekends and bank holidays, 09:00 to 15:00, negative spark spread"
    ),
    "night": "Flagged share at night, 00:00 to 05:00",
    "bmd": "Bright-minus-dull flagged share, 10:00 to 15:00",
}
ERAS: Final[tuple[str, ...]] = ("net", "solar")
WEEKEND_CHEAP_FIRST: Final[int] = 18
WEEKEND_CHEAP_LAST: Final[int] = 29
"""Local half-hours 18 to 29 cover 09:00 to 15:00."""


def usable(*, frame: pl.DataFrame, drops: bool) -> pl.DataFrame:
    """Keep half-hours with a reading, off clock-change days, and (without `drops`) outside them."""
    rows = frame.filter(pl.col("observed"), ~pl.col("clock_change_day"))
    return rows if drops else rows.filter(~pl.col("in_drop"))


def seed_of(*key: str) -> int:
    """Return a seed that depends only on the text of `key`."""
    return BOOTSTRAP_SEED + zlib.crc32("|".join(key).encode()) % 1_000_003


def single_share(*, rows: pl.DataFrame, mask: np.ndarray, seed: int) -> Estimate | None:
    """Return the flagged share of the masked rows, with a week-block interval."""
    if not mask.any():
        return None
    sub = rows.filter(pl.Series(mask))
    state = sub["ts"].to_numpy().astype(float)
    return week_estimate(
        week=sub["week"].to_numpy(),
        columns=np.stack([state, np.ones(len(state))], axis=1),
        statistic=lambda s: ratio(numerator=s[..., 0], denominator=s[..., 1]),
        seed=seed,
        n_resamples=LABEL_ONLY_RESAMPLES,
    )


def bright_minus_dull(*, rows: pl.DataFrame, seed: int) -> Estimate | None:
    """Return the bright-minus-dull flagged share at 10:00 to 15:00, with a week-block interval."""
    midday = rows.filter(
        pl.col("local_half_hour").is_between(MIDDAY_FIRST, MIDDAY_LAST), pl.col("ghi") > GHI_MIN
    )
    if midday.height == 0:
        return None
    graded = midday.with_columns(
        third=group_within(value="ghi", n_groups=3, within=("local_half_hour", "weekend", "month"))
    )
    third = graded["third"].to_numpy()
    columns = share_columns(
        state=graded["ts"].to_numpy().astype(float), mask_a=third == 2, mask_b=third == 0
    )
    return week_estimate(
        week=graded["week"].to_numpy(),
        columns=columns,
        statistic=share_difference_statistic,
        seed=seed,
        n_resamples=LABEL_ONLY_RESAMPLES,
    )


def estimate_of(*, frame: pl.DataFrame, check: str, era: str, drops: bool) -> Estimate | None:
    """Return one check's estimate for one era, with or without the drops."""
    rows = usable(frame=frame, drops=drops).filter(pl.col("era") == era)
    seed = seed_of(check, era, str(drops))
    if check == "bmd":
        return bright_minus_dull(rows=rows, seed=seed)
    if check == "night":
        mask = (rows["local_half_hour"] <= NIGHT_LAST).to_numpy()
    else:
        mask = (
            (rows["weekend"] | rows["bank_holiday"]).to_numpy()
            & rows["local_half_hour"].is_between(WEEKEND_CHEAP_FIRST, WEEKEND_CHEAP_LAST).to_numpy()
            & (rows["spark_spread"] < 0).fill_null(value=False).to_numpy()
        )
    return single_share(rows=rows, mask=mask, seed=seed)


def label_only_checks(*, frame: pl.DataFrame, say: Callable[[str], None]) -> dict:
    """Compute and print the three checks by era, with and without drops, and the swing.

    Args:
        frame: The stored frame of D8.
        say: Prints and records a line.

    Returns:
        A nested dictionary `result[check][era][drops]` of `{point, lo, hi, n_weeks}`, with
        `result[check]["swing"][drops]` for the solar minus the net value.
    """
    result: dict[str, dict] = {}
    for check in CHECKS:
        result[check] = {era: {} for era in ERAS}
        say(f"  {CHECK_TITLES[check]}")
        for drops in (True, False):
            tag = "with drops" if drops else "without drops"
            parts = {
                era: estimate_of(frame=frame, check=check, era=era, drops=drops) for era in ERAS
            }
            for era, est in parts.items():
                if est is None:
                    say(f"    {era}, {tag}: no rows")
                    continue
                result[check][era][str(drops)] = est.to_dict()
                say(f"    {era}, {tag}: {est.point:+.3f} ({est.lo:+.3f} to {est.hi:+.3f})")
            net, solar = parts["net"], parts["solar"]
            if net is not None and solar is not None:
                swing = difference_estimate(first=solar, second=net)
                result[check].setdefault("swing", {})[str(drops)] = swing.to_dict()
                say(
                    f"    swing (solar minus net), {tag}: "
                    f"{swing.point:+.3f} ({swing.lo:+.3f} to {swing.hi:+.3f})"
                )
    return result
