"""A tiny synthetic calendar for the tests: two months of base days with invented values."""

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
    sun = np.clip(np.sin(np.pi * (hh - 12) / 24.0), 0.0, None)
    farm_power = np.clip(sun[None, None, :] * rng.uniform(0.3, 1.0, (n_farms, n_days, 1)), 0, None)
    ghi = 800.0 * sun[None, None, :] * rng.uniform(
        0.3, 1.0, (n_farms, n_days, 1)
    ) + 1e-3 * rng.random((n_farms, n_days, 48))
    farms = sim.FarmSet(
        power=farm_power,
        power_ok=np.ones((n_farms, n_days), dtype=bool),
        sky_ok=np.ones((n_farms, n_days), dtype=bool),
        ghi=ghi,
        dni=0.7 * ghi,
        dhi=0.3 * ghi,
        zenith=np.where(sun > 0, 60.0, 100.0)[None, None, :] * np.ones((n_farms, n_days, 1)),
        azimuth=np.full((n_farms, n_days, 48), 180.0),
    )
    return sim.Calendar(
        dates=dates,
        reading=reading,
        bsp=rng.normal(0, 4.0, (n_days, 48)) if bsp else np.full((n_days, 48), np.nan),
        csi=np.clip(rng.uniform(0.2, 1.0, (n_days, 1)) * np.ones((1, 48)), 0, 1.2),
        clear=1000.0 * sun[None, :] * np.ones((n_days, 1)),
        cosz=sun[None, :] * np.ones((n_days, 1)),
        t2m=10.0 + rng.normal(0, 3.0, (n_days, 48)),
        d8_ghi=ghi[0] * 1.07 + 1.0,
        holidays=np.array(["2022-04-15", "2022-04-18"], dtype="datetime64[D]"),
        farms=farms,
    )
