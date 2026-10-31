# ruff: noqa: ANN201, PT018
"""Day windows and lags gathered on the GPU from dense power arrays (experiment 428).

All indices are positions on experiment 87's dense half-hourly grid. For a row with forecast origin
`o` and target `t`, the newest usable reading is `n = o - NEWEST_AGE` (3.5 hours before the origin).

- **D1** is the 48 half-hours `n - 47 .. n`.
- **D2** is the 48 half-hours `w - 24 .. w + 23`, where `w = t - 336 * ceil((t - n) / 336)` is the
  weekly lag of `common.origin_lags`; half-hours after `n` are unobserved.
- **D3** is D2 moved back by 336 half-hours.
- The three lags of arms R and R-inj are `common.origin_lags`: `n`, the daily lag, and `w`.

A window is returned as five channels: the power standardised by `power_mean` and `power_std` and
clipped at 5 standard deviations (0 where unobserved), the observed flag, sin and cos of the UTC
time of day, and the age before the origin in units of 336 hours.

```bash
uv run --no-sync python ml-experiments/00428-*/e00428_windows.py selftest
```
"""

import math
from typing import Final

import torch  # ty: ignore[unresolved-import]

NEWEST_AGE: Final[int] = 7
"""Half-hours between the origin and the newest usable reading (`common.NEWEST_AGE`)."""
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 336
WINDOW: Final[int] = 48
CLIP_SD: Final[float] = 5.0
AGE_UNIT: Final[float] = 2.0 * 336.0
"""Half-hours in 336 hours."""


def ceil_div(*, a: torch.Tensor, b: int) -> torch.Tensor:
    """Return `ceil(a / b)` for non-negative integer tensors."""
    return torch.div(a + b - 1, b, rounding_mode="floor")


def lag_indices(*, origin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the grid index of the three lags, shape (rows, 3): newest, daily, weekly."""
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * ceil_div(a=gap, b=STEPS_PER_DAY)
    week = target - STEPS_PER_WEEK * ceil_div(a=gap, b=STEPS_PER_WEEK)
    return torch.stack([newest, day, week], dim=1)


def window_indices(*, origin: torch.Tensor, target: torch.Tensor, days: int) -> torch.Tensor:
    """Return the grid index of every position of the first `days` windows: (rows, days, 48)."""
    newest = origin - NEWEST_AGE
    gap = target - newest
    week = target - STEPS_PER_WEEK * ceil_div(a=gap, b=STEPS_PER_WEEK)
    offsets = torch.arange(WINDOW, device=origin.device)
    starts = [newest - (WINDOW - 1), week - WINDOW // 2, week - WINDOW // 2 - STEPS_PER_WEEK]
    return torch.stack([s[:, None] + offsets[None, :] for s in starts[:days]], dim=1)


def take(*, power: torch.Tensor, world: torch.Tensor, primary: torch.Tensor, index: torch.Tensor):
    """Return `power[world, primary, index]` with NaN where the index is off the grid.

    Args:
        power: (worlds, primaries, grid) float32, NaN where unobserved.
        world: (rows,) int64.
        primary: (rows,) int64.
        index: (rows, ...) int64 grid indices.

    Returns:
        The gathered values, NaN where the index is off the grid.
    """
    size = power.shape[-1]
    inside = (index >= 0) & (index < size)
    safe = index.clamp(0, size - 1)
    shape = (-1,) + (1,) * (index.dim() - 1)
    values = power[world.reshape(shape), primary.reshape(shape), safe]
    return torch.where(inside, values, torch.full_like(values, float("nan")))


def origin_lags(
    *, power: torch.Tensor, world: torch.Tensor, primary: torch.Tensor,
    origin: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:  # fmt: skip
    """Return the three raw lags (rows, 3), NaN where unobserved (torch `common.origin_lags`)."""
    return take(
        power=power, world=world, primary=primary, index=lag_indices(origin=origin, target=target)
    )


def lag_ages(*, origin: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return each lag's distance before the target, in units of 336 hours: (rows, 3).

    This is experiment 397's `lag_ages_hours` divided by its `LEAD_SCALE_HOURS`.
    """
    indices = lag_indices(origin=origin, target=target)
    return (target[:, None] - indices).to(torch.float32) / AGE_UNIT


def build_windows(
    *, raw: torch.Tensor, index: torch.Tensor, origin: torch.Tensor, newest_limit: bool
) -> torch.Tensor:
    """Return the (rows, days, 5, 48) channels from raw windows and their grid indices.

    Args:
        raw: (rows, days, 48) raw power over capacity, NaN where unobserved.
        index: (rows, days, 48) grid indices.
        origin: (rows,) grid index of the origin.
        newest_limit: Mask positions after the newest usable reading (always True in production
            use; the argument exists so a self-test can show the mask matters).

    Returns:
        The (rows, days, 5, 48) channel tensor.
    """
    observed = torch.isfinite(raw)
    if newest_limit:
        observed = observed & (index <= (origin - NEWEST_AGE)[:, None, None])
    phase = 2.0 * math.pi * (index % STEPS_PER_DAY).to(torch.float32) / STEPS_PER_DAY
    age = (origin[:, None, None] - index).to(torch.float32) / AGE_UNIT
    power = torch.where(observed, raw, torch.zeros_like(raw))
    return torch.stack(
        [power, observed.to(torch.float32), torch.sin(phase), torch.cos(phase), age], dim=2
    )


def standardise_power(
    *, windows: torch.Tensor, power_mean: float, power_std: float
) -> torch.Tensor:
    """Return `windows` with channel 0 standardised and clipped, 0 where unobserved."""
    out = windows.clone()
    observed = windows[:, :, 1, :] > 0.5
    z = ((windows[:, :, 0, :] - power_mean) / power_std).clamp(-CLIP_SD, CLIP_SD)
    out[:, :, 0, :] = torch.where(observed, z, torch.zeros_like(z))
    return out


def lag_columns(
    *, lags: torch.Tensor, ages: torch.Tensor, power_mean: float, power_std: float
) -> torch.Tensor:
    """Return the nine lag columns of experiment 397 (values, missing flags, ages): (rows, 9)."""
    missing = ~torch.isfinite(lags)
    z = ((lags - power_mean) / power_std).clamp(-CLIP_SD, CLIP_SD)
    z = torch.where(missing, torch.zeros_like(z), z)
    return torch.cat([z, missing.to(torch.float32), ages], dim=1)


def selftest() -> None:
    """Check `lag_indices` against a plain-Python copy of `common.origin_lags` and the windows."""
    import numpy as np

    rng = np.random.default_rng(0)
    origin = rng.integers(10_000, 50_000, 5_000)
    target = origin + rng.integers(1, 480, 5_000)
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - 48 * np.ceil(gap / 48).astype(int)
    week = target - 336 * np.ceil(gap / 336).astype(int)
    expected = np.stack([newest, day, week], axis=1)
    got = lag_indices(origin=torch.from_numpy(origin), target=torch.from_numpy(target)).numpy()
    assert (got == expected).all(), "lag indices differ from common.origin_lags"
    assert (got[:, 1] <= newest).all() and (got[:, 2] <= newest).all()
    idx = window_indices(origin=torch.from_numpy(origin), target=torch.from_numpy(target), days=3)
    assert idx.shape == (5_000, 3, 48)
    assert (idx[:, 0, -1].numpy() == newest).all(), "D1 must end at the newest reading"
    assert (idx[:, 1, 24].numpy() == week).all(), "D2 must be centred on the weekly lag"
    assert (idx[:, 2, 24].numpy() == week - 336).all()
    grid = torch.arange(60_000, dtype=torch.float32)[None, None, :]
    raw = take(
        power=grid, world=torch.zeros(5_000, dtype=torch.long),
        primary=torch.zeros(5_000, dtype=torch.long), index=idx,
    )  # fmt: skip
    built = build_windows(raw=raw, index=idx, origin=torch.from_numpy(origin), newest_limit=True)
    last_used = (built[:, :, 1, :] > 0.5) * idx
    assert last_used.max() <= torch.from_numpy(newest).max(), "a window position is after n"
    after = idx > torch.from_numpy(newest)[:, None, None]
    assert (built[:, :, 1, :][after] == 0).all(), (
        "positions after the newest reading must be masked"
    )
    print("selftest passed")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["selftest"]:
        selftest()
