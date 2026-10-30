"""Injected load-conserving load-shift events and transient spikes for experiment 404.

Experiment 402 (draft PR 56) has no injected-event generator, so this module builds a small one.
It works on each Primary's dense half-hourly power array (fraction of effective capacity, NaN where
unobserved), the same arrays that experiment 87's lags are read from, so the lag columns, the
surprises, and the neighbour windows can all be recomputed from the injected arrays.

A load-shift event moves load from a donor Primary to its connected recipients, conserving the MW:
the donor's power changes by `sign * share * trailing_mean` over the event, and each recipient
changes by the opposite MW divided among the recipients. A share of the events has no observed
recipient. A transient spike is a short change at one Primary that is not a load shift. The depth
shares and durations are drawn by resampling the 112 usable rows of the corrected label table.

```bash
uv run --no-sync python ml-experiments/00404-*/e00404_inject.py selftest
```
"""

import argparse
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

STEP_HOURS: Final[float] = 0.5
TRAILING_STEPS: Final[int] = 30 * 48
"""The depth is a share of the donor's mean over the 30 days before the event begins."""
NO_RECIPIENT_SHARE: Final[float] = 0.3
"""Share of load shifts with no observed recipient (a stated guess, varied as a sensitivity)."""
SPIKE_SHARE: Final[float] = 0.3
"""Share of injected events that are transient spikes and not load shifts."""
SPIKE_STEPS: Final[tuple[int, int]] = (1, 12)
SPIKE_AMPLITUDE_SHARE: Final[tuple[float, float]] = (0.05, 0.5)
NONE, DONOR, RECIPIENT, SPIKE = 0, 1, 2, 3
"""Codes of the event mask."""


@dataclass(frozen=True)
class EventPool:
    """Depth shares and durations resampled from the corrected label table.

    Attributes:
        share: Signed depth of each usable event as a share of its first Primary's trailing mean.
        steps: Duration of each usable event in half-hours.
    """

    share: np.ndarray
    steps: np.ndarray


def event_pool(*, table: pl.DataFrame, trailing_mean_mw: dict[str, float]) -> EventPool:
    """Return the pool from the corrected label table and each first Primary's mean MW.

    Args:
        table: `sw_block_detector_label_table_corrected.csv` as a frame.
        trailing_mean_mw: The mean absolute power in MW of each Primary over the 30 days before
            its events, keyed by the `first_primary` label of the table. A label without a mean is
            skipped.
    """
    usable = table.filter(
        pl.col("corrected_depth_first_mw").is_not_null()
        & pl.col("duration_hours").is_not_null()
        & pl.col("first_primary").is_in(list(trailing_mean_mw))
    )
    share = np.array(
        [
            r["corrected_depth_first_mw"] / trailing_mean_mw[r["first_primary"]]
            for r in usable.iter_rows(named=True)
        ]
    )
    steps = np.maximum(1, np.rint(usable["duration_hours"].to_numpy() / STEP_HOURS)).astype(int)
    return EventPool(share=share, steps=steps)


def trailing_mean(*, power: np.ndarray, start: int) -> float:
    """Return the mean over the 30 days before `start`, ignoring NaN; NaN if none is observed."""
    window = power[max(0, start - TRAILING_STEPS) : start]
    return float(np.nanmean(window)) if np.isfinite(window).any() else float("nan")


@dataclass(frozen=True)
class Injected:
    """The result of one injection.

    Attributes:
        power: Injected power per site, fraction of capacity, same grid as the input.
        mask: Event code per site (`NONE`, `DONOR`, `RECIPIENT`, `SPIKE`), int8, same grid.
        events: One row per injected event and affected site, with the grid range and the change.
    """

    power: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    events: pl.DataFrame


def inject(  # noqa: PLR0915
    *,
    power: dict[str, np.ndarray],
    capacity_mw: dict[str, float],
    neighbours: dict[str, list[str]],
    pool: EventPool,
    first: int,
    last: int,
    coverage: float,
    seed: int,
    no_recipient_share: float = NO_RECIPIENT_SHARE,
) -> Injected:
    """Inject events into the grid range `[first, last)` until each site's coverage is reached.

    Events never extend past `last`, so a training window can be cut at a refit without leaking an
    event into later months. An event does not overlap another event at any site it touches.

    Args:
        power: Clean power per site, fraction of capacity.
        capacity_mw: Effective capacity per site in MW.
        neighbours: Connected Primaries per site (experiment 383's `neighbours_nged.json`).
        pool: Depth shares and durations to resample.
        first: First grid index that may be inside an event (after the 30-day trailing window).
        last: One past the last grid index that may be inside an event.
        coverage: Target share of `[first, last)` that is inside an event, per site.
        seed: Seed of the generator.
        no_recipient_share: Share of load shifts with no observed recipient.

    Returns:
        The injected arrays, the event mask, and the event table.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(power)
    out = {s: power[s].copy() for s in sites}
    mask = {s: np.zeros(power[s].shape, dtype=np.int8) for s in sites}
    covered = dict.fromkeys(sites, 0)
    span = last - first
    rows: list[dict] = []
    attempts = 0
    while min(covered.values()) < coverage * span and attempts < 200_000:
        attempts += 1
        donor = sites[int(rng.integers(len(sites)))]
        if covered[donor] >= coverage * span * 1.5:
            continue
        spike = rng.random() < SPIKE_SHARE
        if spike:
            n = int(rng.integers(SPIKE_STEPS[0], SPIKE_STEPS[1] + 1))
        else:
            n = int(pool.steps[int(rng.integers(len(pool.steps)))])
        n = min(n, span)
        start = int(rng.integers(first, last - n + 1))
        mean = trailing_mean(power=power[donor], start=start)
        if not np.isfinite(mean):
            continue
        touched = [donor]
        recipients: list[str] = []
        if not spike and rng.random() >= no_recipient_share:
            candidates = [r for r in neighbours.get(donor, []) if r in power]
            if candidates:
                k = int(rng.integers(1, min(2, len(candidates)) + 1))
                recipients = list(rng.choice(candidates, size=k, replace=False))
                touched += recipients
        if any((mask[s][start : start + n] != NONE).any() for s in touched):
            continue
        if spike:
            share = float(rng.uniform(*SPIKE_AMPLITUDE_SHARE)) * (1 if rng.random() < 0.5 else -1)
        else:
            share = float(pool.share[int(rng.integers(len(pool.share)))])
            share *= 1 if rng.random() < 0.5 else -1
        delta_donor = share * abs(mean)
        window = slice(start, start + n)
        out[donor][window] += delta_donor
        mask[donor][window] = SPIKE if spike else DONOR
        covered[donor] += n
        rows.append(
            {
                "site": donor, "role": "spike" if spike else "donor", "start": start,
                "stop": start + n, "delta_mw": delta_donor * capacity_mw[donor],
            }
        )  # fmt: skip
        if recipients:
            delta_mw = -delta_donor * capacity_mw[donor] / len(recipients)
            for r in recipients:
                out[r][window] += delta_mw / capacity_mw[r]
                mask[r][window] = RECIPIENT
                covered[r] += n
                rows.append(
                    {
                        "site": r, "role": "recipient", "start": start, "stop": start + n,
                        "delta_mw": delta_mw,
                    }
                )  # fmt: skip
    events = pl.DataFrame(
        rows,
        schema={
            "site": pl.String, "role": pl.String, "start": pl.Int64, "stop": pl.Int64,
            "delta_mw": pl.Float64,
        },
    )  # fmt: skip
    return Injected(power=out, mask=mask, events=events)


def share_of_rows_with_a_lag_inside(
    *, mask: np.ndarray, origins: np.ndarray, targets: np.ndarray, newest_age: int
) -> float:
    """Return the share of rows with at least one of the three lags inside an event.

    The lags are the experiment 87 definition: the newest usable observation at
    `origin - newest_age`, the latest earlier time with the target's half-hour of day, and the
    latest earlier time with its weekday and half-hour.

    Args:
        mask: One site's event mask on the grid.
        origins: Grid index of each row's forecast origin.
        targets: Grid index of each row's valid time.
        newest_age: Half-hours between the origin and the newest usable observation.
    """
    day, week = 48, 7 * 48
    newest = origins - newest_age
    gap = targets - newest
    lag_day = targets - day * np.ceil(gap / day).astype(int)
    lag_week = targets - week * np.ceil(gap / week).astype(int)
    inside = (mask[newest] != NONE) | (mask[lag_day] != NONE) | (mask[lag_week] != NONE)
    return float(inside.mean())


def selftest() -> None:
    """Check conservation, non-overlap, the cut at `last`, and the realised row share."""
    rng = np.random.default_rng(0)
    n = 48 * 365 * 2
    sites = [f"D{i}" for i in range(1, 7)]
    power = {s: (0.5 + 0.2 * np.sin(np.arange(n) / 24.0) + 0.01 * rng.normal(size=n)).astype(
        np.float32) for s in sites}  # fmt: skip
    cap = {s: 10.0 + 5 * i for i, s in enumerate(sites)}
    graph = {s: [t for t in sites if t != s][:3] for s in sites}
    pool = EventPool(share=np.array([-0.05, -0.1, 0.2, -0.3]), steps=np.array([5, 20, 96, 500]))
    first, last = TRAILING_STEPS, n - 48 * 30
    result = inject(
        power=power, capacity_mw=cap, neighbours=graph, pool=pool, first=first, last=last,
        coverage=0.10, seed=1,
    )  # fmt: skip
    assert all((result.mask[s][last:] == NONE).all() for s in sites), "an event passed `last`"
    assert all((result.mask[s][:first] == NONE).all() for s in sites), "an event started early"
    ev = result.events
    for _key, group in ev.group_by(["start", "stop"]):
        shifts = group.filter(pl.col("role") != "spike")
        if shifts.height > 1:
            assert abs(shifts["delta_mw"].sum()) < 1e-6, "a load shift does not conserve MW"
    changed = sum(
        int((result.power[s] != power[s]).sum()) == int((result.mask[s] != NONE).sum())
        for s in sites
    )
    assert changed == len(sites), "power changed outside the mask"
    origins = rng.integers(first + 48 * 30, last - 48 * 15, size=20_000)
    targets = origins + rng.integers(1, 48 * 10, size=origins.size)
    shares = [
        share_of_rows_with_a_lag_inside(
            mask=result.mask[s], origins=origins, targets=targets, newest_age=7
        )
        for s in sites
    ]
    print(f"events {ev.height}, rows with a lag inside an event: {np.round(shares, 3)}")
    print("selftest passed")


def main() -> None:
    """Run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest",))
    parser.parse_args()
    selftest()


if __name__ == "__main__":
    main()
