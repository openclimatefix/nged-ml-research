"""Cascade and cold-spell layers of the worlds of experiment 734.

A cascade replaces the start (and in half of the chosen groups the end) of a positive event by 2 or
3 steps within 24 hours. A cold spell adds a raised-cosine share of the forecast demand to every
Primary for 4 to 14 days. Both layers are built from the table of generator 660 and from arrays,
and both are deterministic functions of the random generator they are given.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from e00653_inject import Truth  # ty: ignore[unresolved-import]

CASCADE_SHARE: Final[float] = 0.15
"""Share of the eligible groups that are cascaded."""
COLD_SPELLS_PER_WINTER: Final[int] = 3
DAY: Final[int] = 48
"""Half-hours per day."""
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = 78816
CASCADE_LABELS: Final[tuple[str, ...]] = ("edge_transfer", "shift")
MAX_OFFSET_HH: Final[int] = 47
"""Largest start offset of a later step, in half-hours after the first."""
MIN_CASCADE_DAYS: Final[int] = 2
"""A group whose event ends sooner than this after its start is not eligible for a cascade."""
SPELL_DAYS: Final[tuple[float, float]] = (4.0, 14.0)
SPELL_SHARE: Final[tuple[float, float]] = (0.03, 0.10)
SPELL_LOADING: Final[tuple[float, float]] = (0.5, 1.5)
N_ZERO_LOADING: Final[int] = 2
"""Primaries per world whose cold-spell loading is zero."""


def eligible_groups(*, events: pl.DataFrame) -> list[int]:
    """Groups whose donor row is a positive event that lasts long enough to hold a cascade."""
    out = []
    donors = events.filter(
        pl.col("label").is_in(CASCADE_LABELS) & pl.col("role").is_in(["donor", "solo"])
    )
    for r in donors.to_dicts():
        rows = events.filter(pl.col("group_id") == r["group_id"])
        ok = all(
            x["end"] is None or x["end"] - x["start"] >= MIN_CASCADE_DAYS * DAY
            for x in rows.to_dicts()
        )
        if ok:
            out.append(int(r["group_id"]))
    return sorted(set(out))


def cascade_events(*, events: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Replace 15% of the eligible groups by cascades of 2 or 3 steps.

    For each chosen group, a number of steps `k` in {2, 3} is drawn, then `k - 1` distinct start
    offsets from 1 to 47 half-hours (sorted, so the last step is within 24 hours of the first),
    and shares from a Dirichlet(1, ..., 1). Every row of the group (donor and recipients) becomes
    `k` rows with the same `group_id`, `end`, and other columns, the start shifted by the offsets,
    and `delta` multiplied by the share. In half of the chosen groups the end is cascaded the same
    way, with the offsets in reverse order. New rows carry `cascade_id` (the original `event_id`)
    and `cascade_end` (the original end); every other row has null in both columns.

    Args:
        events: The ground-truth table of an injected world.
        rng: The random generator.

    Returns:
        The table with the chosen groups cascaded, sorted by `event_id`.
    """
    groups = eligible_groups(events=events)
    n_pick = round(CASCADE_SHARE * len(groups))
    picked = set(rng.choice(groups, size=n_pick, replace=False).tolist()) if n_pick else set()
    base = events.with_columns(
        cascade_id=pl.lit(None, dtype=pl.Int64), cascade_end=pl.lit(None, dtype=pl.Int64)
    )
    rows = base.to_dicts()
    out = [r for r in rows if r["group_id"] not in picked]
    next_id = int(cast("int", events["event_id"].max() or 0)) + 1
    for gid in sorted(picked):
        k = int(rng.integers(2, 4))
        offsets = np.concatenate(
            [[0], np.sort(rng.choice(np.arange(1, MAX_OFFSET_HH + 1), size=k - 1, replace=False))]
        )
        shares = rng.dirichlet(np.ones(k))
        cascade_end = bool(rng.random() < 0.5)
        for r in (x for x in rows if x["group_id"] == gid):
            for j in range(k):
                new = dict(r)
                new["event_id"] = next_id
                next_id += 1
                new["start"] = int(r["start"] + offsets[j])
                if cascade_end and r["end"] is not None:
                    new["end"] = int(r["end"] + offsets[k - 1 - j])
                new["delta"] = float(r["delta"] * shares[j])
                new["cascade_id"] = int(r["event_id"])
                new["cascade_end"] = None if r["end"] is None else int(r["end"])
                out.append(new)
    return pl.DataFrame(out, schema=base.schema).sort("event_id")


def merged_cascades(*, events: pl.DataFrame) -> pl.DataFrame:
    """One row per cascade and Primary: earliest start, original end, and summed `delta`."""
    plain = events.filter(pl.col("cascade_id").is_null()).drop("cascade_id", "cascade_end")
    casc = events.filter(pl.col("cascade_id").is_not_null())
    if casc.height == 0:
        return plain
    merged = []
    for (_, _), g in casc.group_by(["cascade_id", "site"], maintain_order=True):
        first = g.sort("start").row(0, named=True)
        row = {k: v for k, v in first.items() if k not in ("cascade_id", "cascade_end")}
        row["start"] = int(g["start"].min())  # ty: ignore[invalid-argument-type]
        row["end"] = first["cascade_end"]
        row["delta"] = float(g["delta"].sum())
        row["event_id"] = int(first["cascade_id"])
        merged.append(row)
    return pl.concat([plain, pl.DataFrame(merged, schema=plain.schema)]).sort("event_id")


def cascade_truths(*, events: pl.DataFrame, lo: int, hi: int) -> list[Truth]:
    """Truth events of a cascade world: one per cascade and Primary, others through `truths`.

    A cascade has its earliest start, its original end, and the sum of its deltas as the size.
    Every other row goes through experiment 660's `truths` unchanged.

    Args:
        events: The cascaded table.
        lo: First half-hour at which a truth event may start.
        hi: One past the last half-hour at which a truth event may start.

    Returns:
        The truth events starting in `[lo, hi)`.
    """
    from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]

    return truths(events=merged_cascades(events=events), lo=lo, hi=hi)


def _winters() -> list[tuple[int, int]]:
    """Half-hour ranges of each November to February in the record."""
    out = []
    for year in range(2021, 2026):
        a = int((datetime(year, 11, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
        b = int((datetime(year + 1, 3, 1, tzinfo=UTC) - GRID_START).total_seconds() // 1800)
        a, b = max(a, 0), min(b, GRID_LENGTH)
        if b > a:
            out.append((a, b))
    return out


def cold_spells(
    *, rig: object, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], pl.DataFrame]:
    """Cold-spell additions to every Primary and the table of spells.

    Each winter (November to February, 2022 to mid-2026) has 3 spells whose start is uniform in the
    winter and whose length is uniform from 4 to 14 days, with a raised-cosine shape. Each spell
    has an amplitude share `a` uniform from 0.03 to 0.10. Each world draws a loading uniform from
    0.5 to 1.5 for every Primary and sets the loading to 0 at 2 Primaries drawn at random. The
    addition at a Primary is `a * loading * shape * forecast`, in megawatts.

    Args:
        rig: An object with `forecast` (site to half-hourly array) and `panel.sites`.
        rng: The random generator.

    Returns:
        The additions per Primary, each of length `GRID_LENGTH`, and a table of spells with their
        start, end, share, and per-site loading.
    """
    sites = list(rig.panel.sites)  # ty: ignore[unresolved-attribute]
    forecast = rig.forecast  # ty: ignore[unresolved-attribute]
    loading = {s: float(rng.uniform(*SPELL_LOADING)) for s in sites}
    for s in rng.choice(sites, size=N_ZERO_LOADING, replace=False):
        loading[str(s)] = 0.0
    add = {s: np.zeros(GRID_LENGTH) for s in sites}
    spells = []
    for lo, hi in _winters():
        for _ in range(COLD_SPELLS_PER_WINTER):
            start = int(rng.integers(lo, hi))
            length = int(rng.uniform(*SPELL_DAYS) * DAY)
            end = min(start + length, GRID_LENGTH)
            share = float(rng.uniform(*SPELL_SHARE))
            u = (np.arange(start, end) - start) / length
            shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
            for s in sites:
                add[s][start:end] += (
                    share * loading[s] * shape * np.nan_to_num(forecast[s][start:end])
                )
            spells.append(
                {
                    "start": start,
                    "end": end,
                    "share": share,
                    **{f"lambda_{s}": loading[s] for s in sites},
                }
            )
    return add, pl.DataFrame(spells)


def apply_additions(
    *, power: dict[str, np.ndarray], additions: dict[str, np.ndarray], is_mva: dict[str, bool]
) -> dict[str, np.ndarray]:
    """Add the layer to a world's power, as the generator does: `|power + add|` at an MVA Primary.

    A half-hour with no reading, or with no addition, is unchanged.
    """
    out = {}
    for s, base in power.items():
        moved = base + additions[s]
        if is_mva[s]:
            moved = np.abs(moved)
        out[s] = np.where((additions[s] != 0) & np.isfinite(base), moved, base)
    return out
