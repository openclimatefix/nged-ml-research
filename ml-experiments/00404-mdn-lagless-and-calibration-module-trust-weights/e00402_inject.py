# Copied unchanged from experiment 402 (PR 56, branch lag-gate-injected-load-shifts); replaces experiment 404's own generator.
"""Injected load-conserving load shifts and transient spikes for experiment 402.

The structure follows experiment 404's `e00404_inject.py` (nged-ml-research PR 58): an event pool
resampled from the corrected label table, load conserved between a donor and its connected
recipients, events cut at `last`, and no overlap. Four differences:

- The donor changes by a fraction `f` of its own series, so the daily shape survives. A recipient
  gains the donor's change in MW, shared equally. `f` is the table's signed depth divided by the
  mean of the donor over the 30 days before the event.
- A spike has a constant size of 4 to 20 times the Primary's stage-1 residual median absolute
  deviation (MW), either sign, for 1 to 12 half-hours.
- At a Primary with magnitude-only (MVA) readings, an event enters only a window in which every
  affected reading stays above `MAGNITUDE_FLOOR` of capacity, because subtracting from a magnitude
  is not load-conserving where the flow can reverse.
- Base periods may be restricted by the caller (`allowed`), for a label-free sensitivity run.

All arrays are fractions of effective capacity on experiment 87's dense half-hourly grid, NaN where
unobserved.

```bash
uv run --no-sync python ml-experiments/00402-*/e00402_inject.py selftest
```
"""

import argparse
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

STEP_HOURS: Final[float] = 0.5
TRAILING_STEPS: Final[int] = 30 * 48
NO_RECIPIENT_SHARE: Final[float] = 0.3
SPIKE_SHARE: Final[float] = 0.3
SPIKE_STEPS: Final[tuple[int, int]] = (1, 12)
SPIKE_MAD_MULTIPLE: Final[tuple[float, float]] = (4.0, 20.0)
MAGNITUDE_FLOOR: Final[float] = 0.2
NONE, DONOR, RECIPIENT, SPIKE = 0, 1, 2, 3
"""Codes of the event mask."""


@dataclass(frozen=True)
class EventPool:
    """Signed depth fractions and durations resampled from the corrected label table."""

    share: np.ndarray
    steps: np.ndarray


def event_pool(*, table: pl.DataFrame, trailing_mean_mw: dict[str, float]) -> EventPool:
    """Return the pool from the corrected label table and each first Primary's mean MW."""
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
    """Injected power per site, the event code per site and half-hour, and one row per change."""

    power: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    events: pl.DataFrame


def inject(  # noqa: PLR0915
    *,
    power: dict[str, np.ndarray],
    capacity_mw: dict[str, float],
    neighbours: dict[str, list[str]],
    magnitude_only: set[str],
    spike_mad_mw: dict[str, float],
    pool: EventPool,
    first: int,
    last: int,
    coverage: float,
    seed: int,
    no_recipient_share: float = NO_RECIPIENT_SHARE,
    allowed: dict[str, np.ndarray] | None = None,
) -> Injected:
    """Inject events into `[first, last)` until every site's share inside events is `coverage`.

    Args:
        power: Clean power per site, fraction of capacity.
        capacity_mw: Effective capacity per site in MW.
        neighbours: Connected Primaries per site.
        magnitude_only: Sites whose readings are magnitudes (the MVA-only Primaries).
        spike_mad_mw: Each site's stage-1 residual median absolute deviation, in MW.
        pool: Depth fractions and durations to resample.
        first: First grid index that may be inside an event (after the trailing window).
        last: One past the last grid index that may be inside an event.
        coverage: Target share of `[first, last)` inside an event, per site.
        seed: Seed of the generator.
        no_recipient_share: Share of load shifts with no observed recipient.
        allowed: Optional boolean grid per site; an event touching a False half-hour is skipped.
    """
    rng = np.random.default_rng(seed)
    sites = sorted(power)
    out = {s: power[s].copy() for s in sites}
    mask = {s: np.zeros(power[s].shape, dtype=np.int8) for s in sites}
    covered = dict.fromkeys(sites, 0)
    span = last - first
    rows: list[dict] = []
    attempts = 0
    while min(covered.values()) < coverage * span and attempts < 400_000:
        attempts += 1
        donor = sites[int(rng.integers(len(sites)))]
        if covered[donor] >= coverage * span * 1.5:
            continue
        spike = rng.random() < SPIKE_SHARE
        n = (
            int(rng.integers(SPIKE_STEPS[0], SPIKE_STEPS[1] + 1))
            if spike
            else int(pool.steps[int(rng.integers(len(pool.steps)))])
        )
        n = min(n, span)
        start = int(rng.integers(first, last - n + 1))
        window = slice(start, start + n)
        mean = trailing_mean(power=power[donor], start=start)
        if not np.isfinite(mean) or not np.isfinite(power[donor][window]).all():
            continue
        touched = [donor]
        recipients: list[str] = []
        if not spike and rng.random() >= no_recipient_share:
            candidates = [
                r
                for r in neighbours.get(donor, [])
                if r in power and np.isfinite(power[r][window]).all()
            ]
            if candidates:
                k = int(rng.integers(1, min(2, len(candidates)) + 1))
                recipients = list(rng.choice(candidates, size=k, replace=False))
                touched += recipients
        if any((mask[s][window] != NONE).any() for s in touched):
            continue
        if allowed is not None and any(not allowed[s][window].all() for s in touched):
            continue
        if any(
            s in magnitude_only and (power[s][window] <= MAGNITUDE_FLOOR).any() for s in touched
        ):
            continue
        sign = 1 if rng.random() < 0.5 else -1
        if spike:
            multiple = float(rng.uniform(*SPIKE_MAD_MULTIPLE))
            delta = sign * multiple * spike_mad_mw[donor] / capacity_mw[donor] * np.ones(n)
        else:
            delta = (
                sign * float(pool.share[int(rng.integers(len(pool.share)))]) * power[donor][window]
            )
        if donor in magnitude_only and (power[donor][window] + delta <= MAGNITUDE_FLOOR / 2).any():
            continue
        out[donor][window] += delta
        mask[donor][window] = SPIKE if spike else DONOR
        covered[donor] += n
        rows.append(
            {"site": donor, "role": "spike" if spike else "donor", "start": start,
             "stop": start + n, "mean_delta_mw": float(delta.mean() * capacity_mw[donor])}
        )  # fmt: skip
        for r in recipients:
            gain = -delta * capacity_mw[donor] / len(recipients) / capacity_mw[r]
            out[r][window] += gain
            mask[r][window] = RECIPIENT
            covered[r] += n
            rows.append(
                {"site": r, "role": "recipient", "start": start, "stop": start + n,
                 "mean_delta_mw": float(gain.mean() * capacity_mw[r])}
            )  # fmt: skip
    events = pl.DataFrame(
        rows,
        schema={"site": pl.String, "role": pl.String, "start": pl.Int64, "stop": pl.Int64,
                "mean_delta_mw": pl.Float64},
    )  # fmt: skip
    return Injected(power=out, mask=mask, events=events)


def selftest() -> None:
    """Check conservation, non-overlap, the cuts at `first` and `last`, and the magnitude floor."""
    rng = np.random.default_rng(0)
    n = 48 * 365 * 2
    sites = [f"D{i}" for i in range(1, 7)]
    power = {
        s: (0.5 + 0.2 * np.sin(np.arange(n) / 24.0) + 0.01 * rng.normal(size=n)).astype(np.float64)
        for s in sites
    }
    cap = {s: 10.0 + 5 * i for i, s in enumerate(sites)}
    graph = {s: [t for t in sites if t != s][:3] for s in sites}
    pool = EventPool(share=np.array([-0.05, -0.1, 0.2, -0.3]), steps=np.array([5, 20, 96, 500]))
    first, last = TRAILING_STEPS, n - 48 * 30
    result = inject(
        power=power, capacity_mw=cap, neighbours=graph, magnitude_only={"D5", "D6"},
        spike_mad_mw=dict.fromkeys(sites, 0.3), pool=pool, first=first, last=last,
        coverage=0.10, seed=1,
    )  # fmt: skip
    assert all((result.mask[s][last:] == NONE).all() for s in sites), "an event passed `last`"
    assert all((result.mask[s][:first] == NONE).all() for s in sites), "an event started early"
    mw_before = sum(power[s] * cap[s] for s in sites)
    mw_after = sum(result.power[s] * cap[s] for s in sites)
    spike_mw = np.zeros(n)
    for s in sites:
        spike_mw += np.where(result.mask[s] == SPIKE, (result.power[s] - power[s]) * cap[s], 0.0)
    unobserved_recipient = result.events.filter(pl.col("role") == "donor").height
    assert unobserved_recipient > 0
    assert np.allclose(
        mw_after - mw_before - spike_mw, _lost_to_unobserved(result, power, cap, sites, n)
    ), "total MW changed other than by spikes and unobserved recipients"
    for s in ("D5", "D6"):
        assert (result.power[s][result.mask[s] != NONE] > 0).all(), "magnitude went non-positive"
    changed = sum(
        int((result.power[s] != power[s]).sum()) <= int((result.mask[s] != NONE).sum())
        for s in sites
    )
    assert changed == len(sites), "power changed outside the mask"
    print("selftest passed:", result.events.group_by("role").len().sort("role").to_dicts())


def _lost_to_unobserved(
    result: Injected, power: dict, cap: dict, sites: list, n: int
) -> np.ndarray:
    """Return the MW that donors lost without an observed recipient (the expected imbalance)."""
    lost = np.zeros(n)
    donors = result.events.filter(pl.col("role") == "donor")
    recipients = result.events.filter(pl.col("role") == "recipient")
    for d in donors.iter_rows(named=True):
        got = recipients.filter((pl.col("start") == d["start"]) & (pl.col("stop") == d["stop"]))
        if got.is_empty():
            w = slice(d["start"], d["stop"])
            lost[w] += (result.power[d["site"]][w] - power[d["site"]][w]) * cap[d["site"]]
    return lost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["selftest"])
    parser.parse_args()
    selftest()
