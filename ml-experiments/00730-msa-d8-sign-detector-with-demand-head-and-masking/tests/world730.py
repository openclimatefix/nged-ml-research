"""A tiny synthetic calendar for the tests: two months of base days with invented values."""

import e00729_data as data
import e00729_sim as sim
import numpy as np


def make_calendar(
    *, n_days: int = 62, n_farms: int = 3, seed: int = 0, bsp: bool = True
) -> sim.Calendar:
    """Return a calendar of `n_days` consecutive days from 2022-03-01 with invented values.

    Args:
        n_days: The number of base days (62 covers March and April).
        n_farms: The number of farms.
        seed: The seed of the invented values.
        bsp: Whether the BSP meter has data on every day.

    Returns:
        A calendar with a demand shape, a sun shape, and noisy farms.
    """
    rng = np.random.default_rng(seed)
    dates = np.datetime64("2022-03-01", "D") + np.arange(n_days)
    hh = np.arange(48)
    demand = 40.0 + 20.0 * np.sin(np.pi * (hh - 10) / 48.0) ** 2
    reading = demand[None, :] + rng.normal(0, 1.0, (n_days, 48))
    # Like the real meter: a day needs only 40 of its 48 half-hours, so a few readings are missing.
    for day in range(n_days):
        reading[day, rng.choice(48, size=2, replace=False)] = np.nan
    sun = np.clip(np.sin(np.pi * (hh - 12) / 24.0), 0.0, None)
    farm_power = np.clip(sun[None, None, :] * rng.uniform(0.3, 1.0, (n_farms, n_days, 1)), 0, None)
    ghi = 800.0 * sun[None, None, :] * rng.uniform(
        0.3, 1.0, (n_farms, n_days, 1)
    ) + 1e-3 * rng.random((n_farms, n_days, 48))
    sun_up = np.broadcast_to(sun > 0, (n_farms, n_days, 48))
    sky_ok = data.sky_present(ghi=np.where(sun_up, ghi, np.nan), sun_up=sun_up)
    power_ok = np.ones((n_farms, n_days), dtype=bool)
    power_ok[0, 5] = False  # one farm has no metered output on one day
    farm_power = np.where(power_ok[:, :, None], farm_power, np.nan)
    farms = sim.FarmSet(
        power=farm_power,
        power_ok=power_ok,
        sky_ok=sky_ok,
        ghi=ghi,
        dni=0.7 * ghi,
        dhi=0.3 * ghi,
        zenith=np.where(sun > 0, 60.0, 100.0)[None, None, :] * np.ones((n_farms, n_days, 1)),
        azimuth=np.full((n_farms, n_days, 48), 180.0),
    )
    # Like the real table: the clear-sky index is NaN wherever the clear-sky irradiance is tiny,
    # which includes the first and last half-hour with the sun up.
    within_day = 1.0 + 0.05 * np.sin(hh / 3.0)[None, :]  # the index varies inside each day
    csi = np.clip(rng.uniform(0.5, 1.0, (n_days, 1)) * within_day, 0, 1.2)
    up = sun > 0
    edge = up & ~np.roll(up, 1) | up & ~np.roll(up, -1)
    csi = np.where(up & ~edge, csi, np.nan)
    return sim.Calendar(
        dates=dates,
        reading=reading,
        bsp=rng.normal(0, 4.0, (n_days, 48)) if bsp else np.full((n_days, 48), np.nan),
        csi=csi,
        clear=1000.0 * sun[None, :] * np.ones((n_days, 1)),
        cosz=sun[None, :] * np.ones((n_days, 1)),
        t2m=10.0 + rng.normal(0, 3.0, (n_days, 48)),
        wind10=np.abs(5.0 + rng.normal(0, 2.0, (n_days, 48))),
        precip=np.clip(rng.normal(0.0, 0.3, (n_days, 48)), 0.0, None),
        d8_ghi=ghi[0] * 1.07 + 1.0,
        holidays=np.array(["2022-04-15", "2022-04-18"], dtype="datetime64[D]"),
        farms=farms,
    )
