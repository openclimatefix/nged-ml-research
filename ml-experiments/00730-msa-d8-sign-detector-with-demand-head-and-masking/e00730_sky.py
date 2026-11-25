"""Experiment 730: fill the short gaps of the metered farms' sky and power before banks are built.

Experiment 729's `farm_set` replaces a missing CAMS irradiance (ghi, dni, dhi) by zero after it has
accepted a farm-day with up to 6% of its sun-up half-hours missing, and the simulator then draws
zero solar output at those half-hours. The metered-farm path (source S1) does the same with a
missing power reading. The result is a one- to four-half-hour "notch" of zero output in the
middle of a clear day. Experiment 729's code is not edited. `fill_sky_gaps` repairs the
`Calendar` that this experiment builds its banks on:

- **Sun-up rule.** Only half-hours with a solar zenith below `SUN_UP_ZENITH_MAX_DEG` are gaps.
  Dawn, dusk, and night are never touched.
- **Sky gaps (source S2).** A run of at most `MAX_SKY_GAP_HH` half-hours with zero irradiance at
  sun-up is filled by linear interpolation of ghi over the cosine of the zenith, between the
  nearest positive half-hours on both sides of the same day, or held at the clearness of the one
  positive neighbour when the run touches the first or last sun-up half-hour. The direct and
  diffuse parts are re-derived from the filled ghi with the Erbs model. A longer run, or a run
  with no positive half-hour on either side, is not filled: the farm-day's sky becomes NaN and
  the farm-day leaves `sky_ok`, like a farm-day with no sky. A gap is never zero-filled.
- **Power gaps (source S1).** A run of at most `MAX_POWER_GAP_HH` missing half-hours at sun-up
  is filled the same way, over the power. A longer run
  removes the farm-day from `power_ok`. Genuine metered zeros are left as they are.
"""

import dataclasses
from typing import Final

import e00730_paths  # noqa: F401

# isort: split
import e00729_sim as sim729
import numpy as np

SUN_UP_ZENITH_MAX_DEG: Final = 80.0
"""A half-hour below this zenith is sun-up, so missing data there is a gap."""
MAX_SKY_GAP_HH: Final = 4
"""The longest run of missing irradiance, in half-hours, that is interpolated (2 hours)."""
MAX_POWER_GAP_HH: Final = 3
"""The longest run of missing metered power, in half-hours, that is interpolated."""
SKY_FILL_VERSION: Final = (
    f"fill-v1-zenith{SUN_UP_ZENITH_MAX_DEG:g}-sky{MAX_SKY_GAP_HH}-power{MAX_POWER_GAP_HH}"
)
"""The code version that the run marker carries; any change of a limit changes the marker."""
MIN_COS_ZENITH: Final = 1e-3
MAX_ZEROED_SHARE: Final = 0.01
"""Pre-registered bound on the share of zeroed sun-up half-hours inside a source-S2 solar day."""


def gap_runs(*, mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the `(start, stop)` of each run of True values in a one-dimensional mask."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def day_of_year(*, dates: np.ndarray) -> np.ndarray:
    """Return the day of the year (1 to 366) of each `datetime64[D]` date."""
    days = dates.astype("datetime64[D]")
    return (days - days.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1


def sun_up_zero_mask(*, farms: sim729.FarmSet) -> np.ndarray:
    """Return where irradiance is exactly zero although the sun is up, `(F, n, 48)`."""
    with np.errstate(invalid="ignore"):
        return (farms.ghi == 0.0) & (farms.zenith < SUN_UP_ZENITH_MAX_DEG)


def zeroed_sky_share(*, farms: sim729.FarmSet) -> float:
    """Return the share of sun-up half-hours of `sky_ok` farm-days whose irradiance is zero."""
    with np.errstate(invalid="ignore"):
        up = (farms.zenith < SUN_UP_ZENITH_MAX_DEG) & farms.sky_ok[:, :, None]
    return float((sun_up_zero_mask(farms=farms) & up).sum() / max(int(up.sum()), 1))


def fill_run(
    *, values: np.ndarray, cos_z: np.ndarray, start: int, stop: int, usable: np.ndarray
) -> np.ndarray | None:
    """Return the fill of `values[start:stop]`, or None when the run has no usable neighbour.

    The fill interpolates `values / cos_z` linearly between the neighbours on both sides when both
    are usable, and holds the value of the one usable neighbour otherwise (a gap that touches
    the first or last sun-up half-hour). It multiplies back by `cos_z`.
    """
    anchors = [i for i in (start - 1, stop) if 0 <= i < len(values) and usable[i]]
    if not anchors:
        return None
    ratio = np.interp(np.arange(start, stop), anchors, [values[i] / cos_z[i] for i in anchors])
    return ratio * cos_z[start:stop]


def _cos_zenith(*, zenith: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(np.radians(zenith)), MIN_COS_ZENITH, None)


def fill_sky_gaps(*, cal: sim729.Calendar) -> sim729.Calendar:
    """Return `cal` with the sun-up gaps of the farms' sky and power filled or excluded.

    Args:
        cal: The calendar of base days, whose `farms` hold zero where irradiance was missing.

    Returns:
        A calendar whose farm set has no zero irradiance at sun-up on a `sky_ok` farm-day, and no
        missing power inside a short sun-up gap of a `power_ok` farm-day.
    """
    import pvlib

    farms = cal.farms
    ghi, dni, dhi = farms.ghi.copy(), farms.dni.copy(), farms.dhi.copy()
    power = farms.power.copy()
    sky_ok, power_ok = farms.sky_ok.copy(), farms.power_ok.copy()
    doy = day_of_year(dates=cal.dates)
    zero = sun_up_zero_mask(farms=farms)
    n_farms = ghi.shape[0]
    for f in range(n_farms):
        for d in np.flatnonzero(sky_ok[f] & zero[f].any(axis=1)):
            row, cos_z, at = ghi[f, d], _cos_zenith(zenith=farms.zenith[f, d]), zero[f, d]
            usable = np.isfinite(row) & (row > 0.0)
            filled = row.copy()
            ok = True
            for a, b in gap_runs(mask=at):
                part = (
                    fill_run(values=row, cos_z=cos_z, start=a, stop=b, usable=usable)
                    if b - a <= MAX_SKY_GAP_HH
                    else None
                )
                if part is None:
                    ok = False
                else:
                    filled[a:b] = part
            if ok:
                split = pvlib.irradiance.erbs(
                    ghi=filled, zenith=farms.zenith[f, d], datetime_or_doy=np.full(48, doy[d])
                )
                ghi[f, d, at] = filled[at]
                dni[f, d, at] = np.asarray(split["dni"])[at]
                dhi[f, d, at] = np.asarray(split["dhi"])[at]
            else:
                sky_ok[f, d] = False
                ghi[f, d, at] = dni[f, d, at] = dhi[f, d, at] = np.nan
        with np.errstate(invalid="ignore"):
            missing = ~np.isfinite(farms.power[f]) & (farms.zenith[f] < SUN_UP_ZENITH_MAX_DEG)
        for d in np.flatnonzero(power_ok[f] & missing.any(axis=1)):
            row, cos_z = power[f, d], _cos_zenith(zenith=farms.zenith[f, d])
            usable = np.isfinite(row)
            for a, b in gap_runs(mask=missing[d]):
                part = (
                    fill_run(values=row, cos_z=cos_z, start=a, stop=b, usable=usable)
                    if b - a <= MAX_POWER_GAP_HH
                    else None
                )
                if part is None:
                    power_ok[f, d] = False
                else:
                    row[a:b] = part
    return dataclasses.replace(
        cal,
        farms=dataclasses.replace(
            farms, ghi=ghi, dni=dni, dhi=dhi, power=power, sky_ok=sky_ok, power_ok=power_ok
        ),
    )
