# ruff: noqa: E501
"""A random `SegmentArrays` for the tests of the synthetic world, the baselines, and the fits."""

import e00733_synth as sy
import numpy as np


def make_segment(
    *, n_days: int = 40, seed: int = 0, missing_days: tuple[int, ...] = ()
) -> sy.SegmentArrays:
    """Return a segment of `n_days` dense days with random arrays; days in `missing_days` are not base days."""
    rng = np.random.default_rng(seed)
    s = n_days * sy.STEPS
    present = np.ones(n_days, dtype=bool)
    present[list(missing_days)] = False
    slot_present = np.repeat(present, sy.STEPS)
    x_neg = (rng.random(s) < 0.2).astype(np.float32)
    x_neg[~slot_present] = np.nan
    reading = rng.normal(size=s).astype(np.float32)
    reading[~slot_present] = np.nan
    u_raw = np.stack([rng.random(s), rng.random(s) * 30.0, rng.random(s)], axis=1).astype(
        np.float32
    )
    day_start = 48 * (1000 + np.arange(n_days))
    init_us = 1_567_296_000_000_000 + day_start * 1_800_000_000
    seg = sy.SegmentArrays(
        present=present,
        covered=present.copy(),
        reading_z=reading,
        x_neg=x_neg,
        solar_points=(rng.random(s) * 20).astype(np.float32),
        u_raw=u_raw,
        u_ok=slot_present.copy(),
        hist_weather=rng.normal(size=(s, sy.N_WEATHER_COLUMNS)).astype(np.float32),
        hist_datetime=rng.normal(size=(s, sy.N_WEATHER_COLUMNS)).astype(np.float32),
        day_type=rng.integers(0, 8, s),
        local_hh=np.tile(np.arange(sy.STEPS), n_days),
        month=np.full(n_days, 6),
        top_weather=rng.normal(size=(n_days, 14, 48, sy.N_WEATHER_COLUMNS)).astype(np.float32),
        top_spread=rng.random((n_days, 14, 48, sy.N_SPREAD_COLUMNS)).astype(np.float32),
        top_datetime=rng.normal(size=(n_days, 14, 48, sy.N_WEATHER_COLUMNS)).astype(np.float32),
        top_day_type=rng.integers(0, 8, (n_days, 14, 48)),
        top_run_init_us=np.where(present, init_us, 0),
        top_run_avail_us=np.where(present, init_us + 9 * 3_600_000_000, 0),
        top_other_valid_us=np.where(present, init_us, 0),
        top_c_grid=day_start + 24,
        top_filled=np.broadcast_to(present[:, None, None], (n_days, 14, 48)).copy(),
    )
    seg.check_shapes()
    return seg
