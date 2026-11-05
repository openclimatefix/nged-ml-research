# ruff: noqa: E501, PLR0912, PLR0915, ANN001, ANN201
r"""Synthetic load-shift worlds, the neighbour graph, and the auxiliary-flag labels of experiment 567.

The injection is experiment 402's `inject()` (copied here, not edited) with three added parameters:
`gain_range` (the share of the donor's loss that a recipient gains: 1.0 conserves load exactly, and
experiment 459 measured 60% to 100% in real events), `spike_share`, and `require_recipient` (held-out
worlds keep only events that have a recipient, so that every scored event carries a mirror).

A world is the clean power of the 16 Primaries (capacity fraction, NaN where flagged or cut) with
synthetic events added to it, plus the event code per half-hour. Training worlds only change the lags:
the training target is always the clean series.

```bash
PYTHONPATH=<every experiment folder> uv run --no-sync python ml-experiments/00567-*/e00567_world.py selftest
```
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00402_inject as inj  # ty: ignore[unresolved-import]
import e00402_worlds as worlds402  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00567-neighbour-lag-encoders-synthetic-load-shifts-in-pretraining-and-fine-tuning"
)
WORLD_DIR: Final[Path] = WORK / "worlds"
K_NEIGHBOURS: Final[int] = 4
"""Most neighbours a row reads."""
SLOTS: Final[int] = 2 + 2 * K_NEIGHBOURS
"""Lag slots of a row: own day and week, then each neighbour's day and week."""
RATES: Final[dict[str, float]] = {"r0": 0.0, "r3": 0.0117, "r7": 0.0235, "r21": 0.075}
"""Injection coverage (the generator's target share of half-hours inside events) for each rate label.

The labels name the share of Primary-days touched by a synthetic event in the 730 days before the
cutoff: none, 3.5%, 7%, and 21% (measured 6.0% to 7.0% at `r7`, see the write-up). The real labelled
events touch 3.6% to 4.3% of Primary-days in the same window, and 6.3% to 7.7% once padded by 2 days
on each side as experiment 510's training-row drop pads them. The coverage values were found by
bisection on a calibration world (fold 2026-01), as the share of Primary-days is not the share of
half-hours.
"""
GAIN_EXACT: Final[tuple[float, float]] = (1.0, 1.0)
GAIN_IMPERFECT: Final[tuple[float, float]] = (0.6, 1.0)
FLAG_MIN_FRACTION: Final[float] = 0.02
"""A lag is positive when the injected change is at least this share of the Primary's capacity."""
FRACTION_FLOOR: Final[float] = 0.05
"""Smallest level (capacity fraction) the signed-fraction target divides by."""
HELD_OUT_SHARE: Final[float] = 1 / 3
HELDOUT_SEED_OFFSET: Final[int] = 70_000
TRAIN_SEED_OFFSET: Final[int] = 100


def inject(
    *,
    power: dict[str, np.ndarray],
    capacity_mw: dict[str, float],
    neighbours: dict[str, list[str]],
    magnitude_only: set[str],
    spike_mad_mw: dict[str, float],
    pool: inj.EventPool,
    first: int,
    last: int,
    coverage: float,
    seed: int,
    no_recipient_share: float = inj.NO_RECIPIENT_SHARE,
    allowed: dict[str, np.ndarray] | None = None,
    gain_range: tuple[float, float] = GAIN_EXACT,
    spike_share: float = inj.SPIKE_SHARE,
    require_recipient: bool = False,
    active: set[str] | None = None,
) -> inj.Injected:
    """Experiment 402's injection with the three added parameters (see the module docstring)."""
    rng = np.random.default_rng(seed)
    sites = sorted(power)
    out = {s: power[s].copy() for s in sites}
    mask = {s: np.zeros(power[s].shape, dtype=np.int8) for s in sites}
    covered = dict.fromkeys(sites, 0)
    span = last - first
    rows: list[dict] = []
    attempts = 0
    target_sites = sorted(active) if active is not None else sites
    while min(covered[x] for x in target_sites) < coverage * span and attempts < 400_000:
        attempts += 1
        donor = sites[int(rng.integers(len(sites)))]
        if covered[donor] >= coverage * span * 1.5:
            continue
        spike = rng.random() < spike_share
        n = (
            int(rng.integers(inj.SPIKE_STEPS[0], inj.SPIKE_STEPS[1] + 1))
            if spike
            else int(pool.steps[int(rng.integers(len(pool.steps)))])
        )
        n = min(n, span)
        start = int(rng.integers(first, last - n + 1))
        window = slice(start, start + n)
        if not np.isfinite(power[donor][window]).all():
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
        if require_recipient and not recipients:
            continue
        if any((mask[s][window] != inj.NONE).any() for s in touched):
            continue
        if allowed is not None and any(not allowed[s][window].all() for s in touched):
            continue
        if any(
            s in magnitude_only and (power[s][window] <= inj.MAGNITUDE_FLOOR).any() for s in touched
        ):
            continue
        sign = 1 if rng.random() < 0.5 else -1
        if spike:
            multiple = float(rng.uniform(*inj.SPIKE_MAD_MULTIPLE))
            delta = sign * multiple * spike_mad_mw[donor] / capacity_mw[donor] * np.ones(n)
        else:
            delta = (
                sign * float(pool.share[int(rng.integers(len(pool.share)))]) * power[donor][window]
            )
        if (
            donor in magnitude_only
            and (power[donor][window] + delta <= inj.MAGNITUDE_FLOOR / 2).any()
        ):
            continue
        gain = float(rng.uniform(*gain_range)) if gain_range[0] < gain_range[1] else gain_range[0]
        gains = {
            r: -delta * gain * capacity_mw[donor] / len(recipients) / capacity_mw[r]
            for r in recipients
        }
        if any(
            (r in magnitude_only and (power[r][window] + g <= inj.MAGNITUDE_FLOOR / 2).any())
            or (power[r][window] + g < 0).any()
            or covered[r] >= coverage * span * 1.5
            for r, g in gains.items()
        ):
            continue
        out[donor][window] += delta
        mask[donor][window] = inj.SPIKE if spike else inj.DONOR
        covered[donor] += n
        rows.append(
            {"site": donor, "role": "spike" if spike else "donor", "start": start,
             "stop": start + n, "mean_delta_mw": float(delta.mean() * capacity_mw[donor])}
        )  # fmt: skip
        for r in recipients:
            out[r][window] += gains[r]
            mask[r][window] = inj.RECIPIENT
            covered[r] += n
            rows.append(
                {"site": r, "role": "recipient", "start": start, "stop": start + n,
                 "mean_delta_mw": float(gains[r].mean() * capacity_mw[r])}
            )  # fmt: skip
    short = [s for s in target_sites if covered[s] < coverage * span]
    if short:
        print(f"warning: sites below the coverage target after {attempts} attempts: {short}")
    events = pl.DataFrame(
        rows,
        schema={"site": pl.String, "role": pl.String, "start": pl.Int64, "stop": pl.Int64,
                "mean_delta_mw": pl.Float64},
    )  # fmt: skip
    return inj.Injected(power=out, mask=mask, events=events)


# --- the neighbour graph ------------------------------------------------------------------------


def neighbour_graph(*, ctx: worlds402.Context) -> dict[str, list[str]]:
    """Return each Primary's neighbours from the interconnections table, in a fixed (sorted) order."""
    return {s: sorted(ctx.neighbours[s], key=lambda n: int(n[1:])) for s in ctx.labels}


def held_out_pairs(*, graph: dict[str, list[str]], seed: int = 0) -> set[frozenset[str]]:
    """Return a third of the undirected neighbour pairs, chosen so that every Primary keeps a training neighbour when it has one."""
    pairs = sorted({frozenset((a, b)) for a, nbs in graph.items() for b in nbs}, key=sorted)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    left = {s: len(nbs) for s, nbs in graph.items()}
    chosen: set[frozenset[str]] = set()
    target = round(len(pairs) * HELD_OUT_SHARE)
    for i in order:
        pair = pairs[int(i)]
        a, b = sorted(pair)
        in_slots = b in graph[a][:K_NEIGHBOURS] and a in graph[b][:K_NEIGHBOURS]
        if in_slots and left[a] > 1 and left[b] > 1:
            chosen.add(pair)
            left[a] -= 1
            left[b] -= 1
        if len(chosen) >= target:
            break
    return chosen


def split_graph(
    *, graph: dict[str, list[str]], held: set[frozenset[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (training graph without the held-out pairs, graph of only the held-out pairs)."""
    train = {s: [n for n in nbs if frozenset((s, n)) not in held] for s, nbs in graph.items()}
    test = {s: [n for n in nbs if frozenset((s, n)) in held] for s, nbs in graph.items()}
    return train, test


def neighbour_index(*, graph: dict[str, list[str]], labels: list[str]) -> np.ndarray:
    """Return (primaries, K) neighbour indices in `labels` order, -1 where absent; at most K_NEIGHBOURS."""
    out = np.full((len(labels), K_NEIGHBOURS), -1, dtype=np.int64)
    for p, s in enumerate(labels):
        for j, n in enumerate(graph[s][:K_NEIGHBOURS]):
            out[p, j] = labels.index(n)
    return out


def placebo_index(*, graph: dict[str, list[str]], labels: list[str], seed: int = 0) -> np.ndarray:
    """Return (primaries, K) indices of random non-neighbour Primaries, as many as the Primary has neighbours."""
    rng = np.random.default_rng(seed)
    real = neighbour_index(graph=graph, labels=labels)
    out = np.full_like(real, -1)
    for p, s in enumerate(labels):
        others = [i for i, n in enumerate(labels) if n != s and n not in graph[s]]
        k = int((real[p] >= 0).sum())
        out[p, :k] = rng.choice(others, size=k, replace=False)
    return out


def event_pair_index(
    *, labels: list[str], table: pl.DataFrame, drop: set[frozenset[str]] | None = None
) -> np.ndarray:
    """Return (primaries, K) neighbour indices from the label table's `pair` column (arm N4).

    A pair names two Primaries (as `D5 / D8` style text); every pair among the 16 Primaries makes both
    members neighbours of one another, ordered by index.
    """
    pairs: set[frozenset[str]] = set()
    for text in table["pair"].drop_nulls().to_list():
        names = [
            t
            for t in str(text).replace("/", " ").replace("-", " ").replace(",", " ").split()
            if t in labels
        ]
        for a in names:
            for b in names:
                if a != b:
                    pairs.add(frozenset((a, b)))
    pairs -= drop or set()
    graph = {
        s: sorted({n for p in pairs if s in p for n in p if n != s}, key=lambda n: int(n[1:]))
        for s in labels
    }
    return neighbour_index(graph=graph, labels=labels)


# --- worlds -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldSpec:
    """Which world: its kind, rate, fold, and seed."""

    kind: str
    rate: str
    month: str
    seed: int
    gain: str = "mixed"

    def file(self) -> Path:
        """Return the world's file."""
        return WORLD_DIR / f"{self.kind}_{self.rate}_{self.gain}_{self.month}_s{self.seed}.npz"


def real_share_of_days(*, table: pl.DataFrame, n_primaries: int) -> float:
    """Return the share of Primary-days that a labelled event with a Primary named touches, over the table's span."""
    usable = table.filter(pl.col("first_primary").is_not_null())
    onset = usable["onset_utc"].str.to_datetime().dt.date().to_numpy().astype("datetime64[D]")
    end = usable["end_utc"].str.to_datetime().dt.date().to_numpy().astype("datetime64[D]")
    names = usable["first_primary"].to_list()
    first, last = onset.min(), end.max()
    days = int((last - first).astype(int)) + 1
    seen: dict[str, np.ndarray] = {}
    for name, a, b in zip(names, onset, end, strict=True):
        arr = seen.setdefault(name, np.zeros(days, dtype=bool))
        arr[int((a - first).astype(int)) : int((b - first).astype(int)) + 1] = True
    return float(sum(a.sum() for a in seen.values()) / (n_primaries * days))


def share_of_days_with_event(*, code: np.ndarray, first: int, last: int) -> float:
    """Return the share of Primary-days in [first, last) with at least one half-hour inside an event."""
    days = (last - first) // 48
    seg = code[:, first : first + days * 48].reshape(code.shape[0], days, 48)
    return float((seg != inj.NONE).any(axis=2).mean())


def build_world(
    *,
    spec: WorldSpec,
    clean: np.ndarray,
    ctx: worlds402.Context,
    pool: inj.EventPool,
    graph: dict[str, list[str]],
    last: int,
    first: int | None = None,
    allowed_extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (power, code), each (primaries, grid), for a world; cached on disk.

    `kind` is `train` (training graph, mixed exact and 60%-100% conservation, no spikes: a spike is a reading error, not a load transfer) or
    `exact` / `imperfect` (held-out pairs only, every event has a recipient, no spikes). Events end
    before `last`. The base is `clean` with real labelled events (padded) excluded.
    """
    path = spec.file()
    if path.exists():
        z = np.load(path)
        return z["power"], z["code"]
    labels = list(ctx.labels)
    assert labels == c.sites(), "context and row order differ"
    if spec.kind == "train":
        coverage = RATES[spec.rate]
        seed = TRAIN_SEED_OFFSET + 2 * spec.seed
        gain, spike, need = GAIN_EXACT, 0.0, False
        no_recip = inj.NO_RECIPIENT_SHARE
    else:
        coverage = 0.20
        seed = HELDOUT_SEED_OFFSET + spec.seed
        gain = GAIN_EXACT if spec.kind == "exact" else GAIN_IMPERFECT
        spike, need, no_recip = 0.0, True, 0.0
    lo = inj.TRAILING_STEPS if first is None else first
    if coverage == 0.0:
        power, code = clean.copy(), np.zeros(clean.shape, dtype=np.int8)
    else:
        allowed = {
            s: ~ctx.event_mask[s]
            & np.isfinite(clean[i])
            & (True if allowed_extra is None else allowed_extra[i])
            for i, s in enumerate(labels)
        }
        if spec.kind == "train":
            # Half of the events conserve exactly, half give the recipient 60% to 100% of the loss.
            # Two runs on disjoint halves of the grid would change coverage; instead alternate by seed parity per event
            # inside one run by drawing the gain range per call: run twice with half the coverage on the same mask.
            injected_a = inject(power={s: clean[i].astype(np.float64) for i, s in enumerate(labels)},
                                capacity_mw=ctx.capacity, neighbours=graph, magnitude_only=ctx.magnitude_only,
                                spike_mad_mw=ctx.stage1_mad_mw, pool=pool, first=lo, last=last,
                                coverage=coverage / 2, seed=seed, allowed=allowed, gain_range=GAIN_EXACT,
                                spike_share=spike)  # fmt: skip
            allowed_b = {s: allowed[s] & (injected_a.mask[s] == inj.NONE) for s in labels}
            injected = inject(power=injected_a.power, capacity_mw=ctx.capacity, neighbours=graph,
                              magnitude_only=ctx.magnitude_only, spike_mad_mw=ctx.stage1_mad_mw, pool=pool,
                              first=lo, last=last, coverage=coverage / 2, seed=seed + 1, allowed=allowed_b,
                              gain_range=GAIN_IMPERFECT, spike_share=spike)  # fmt: skip
            code = np.stack(
                [np.maximum(injected.mask[s], injected_a.mask[s]) for s in labels]
            ).astype(np.int8)
        else:
            injected = inject(power={s: clean[i].astype(np.float64) for i, s in enumerate(labels)},
                              capacity_mw=ctx.capacity, neighbours=graph, magnitude_only=ctx.magnitude_only,
                              spike_mad_mw=ctx.stage1_mad_mw, pool=pool, first=lo, last=last, coverage=coverage,
                              seed=seed, no_recipient_share=no_recip, allowed=allowed, gain_range=gain,
                              spike_share=spike, require_recipient=need,
                              active={x for x, nbs in graph.items() if nbs} if need else None)  # fmt: skip
            code = np.stack([injected.mask[s] for s in labels]).astype(np.int8)
        power = np.stack([injected.power[s] for s in labels]).astype(np.float32)
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, power=power.astype(np.float32), code=code)
    tmp.rename(path)
    return power.astype(np.float32), code


# --- auxiliary labels ---------------------------------------------------------------------------


def aux_labels(
    *,
    world: np.ndarray,
    clean: np.ndarray,
    code: np.ndarray,
    slot_primary: np.ndarray,
    slot_idx: np.ndarray,
    real_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flag, fraction) for lag slots: flag 1 positive, 0 negative, -1 excluded; fraction float16.

    `slot_primary` and `slot_idx` are (rows, slots) arrays (Primary index, lag grid index). A slot
    is excluded when the lag is missing (no index, or NaN in the world) or the injected change is
    below `FLAG_MIN_FRACTION` of capacity inside a synthetic event (too small to identify). The
    fraction is the injected change over the clean level, floored at `FRACTION_FLOOR`.
    """
    ok = (slot_primary >= 0) & (slot_idx >= 0)
    p = np.clip(slot_primary, 0, None)
    i = np.clip(slot_idx, 0, None)
    w, base, ev = world[p, i], clean[p, i], code[p, i]
    finite = ok & np.isfinite(w) & np.isfinite(base)
    delta = np.where(finite, w - base, 0.0)
    big = np.abs(delta) >= FLAG_MIN_FRACTION
    flag = np.where(~finite, -1, np.where(big, 1, np.where(ev != inj.NONE, -1, 0))).astype(np.int8)
    if (
        real_mask is not None
    ):  # a lag inside a real labelled event is neither clean nor a synthetic positive
        flag = np.where((flag == 0) & real_mask[p, i], -1, flag).astype(np.int8)
    frac = np.where(finite, delta / np.maximum(np.abs(base), FRACTION_FLOOR), 0.0)
    return flag, np.clip(frac, -1.0, 1.0).astype(np.float16)


def selftest() -> None:
    """Check the injection copy against experiment 402's and the graph helpers on a small synthetic grid."""
    rng = np.random.default_rng(0)
    n = 48 * 365 * 2
    sites = [f"D{i}" for i in range(1, 7)]
    power = {
        s: (0.5 + 0.2 * np.sin(np.arange(n) / 24.0) + 0.01 * rng.normal(size=n)) for s in sites
    }
    cap = {s: 10.0 + 5 * i for i, s in enumerate(sites)}
    graph = {s: [t for t in sites if t != s][:2] for s in sites}
    pool = inj.EventPool(share=np.array([-0.05, -0.1, 0.2, -0.3]), steps=np.array([5, 20, 96, 500]))
    first, last = inj.TRAILING_STEPS, n - 48 * 30
    kw = dict(power=power, capacity_mw=cap, neighbours=graph, magnitude_only={"D5", "D6"},
              spike_mad_mw=dict.fromkeys(sites, 0.3), pool=pool, first=first, last=last, coverage=0.10, seed=1)  # fmt: skip
    ref = inj.inject(**kw)
    mine = inject(**kw)
    assert all(np.array_equal(ref.power[s], mine.power[s]) for s in sites), (
        "copy differs from experiment 402"
    )
    strict = dict(kw, require_recipient=True, spike_share=0.0, no_recipient_share=0.0)
    for gain, conserved in ((GAIN_EXACT, True), (GAIN_IMPERFECT, False)):
        world = inject(**strict, gain_range=gain)
        net = sum((world.power[s] - power[s]) * cap[s] for s in sites)
        assert (np.abs(net).max() < 1e-9) == conserved, (gain, float(np.abs(net).max()))
    held = inject(**kw, require_recipient=True, spike_share=0.0, no_recipient_share=0.0)
    donors = held.events.filter(pl.col("role") == "donor").height
    recips = held.events.filter(pl.col("role") == "recipient").height
    assert donors > 0 and recips >= donors, (donors, recips)
    pairs = held_out_pairs(graph=graph)
    train_g, test_g = split_graph(graph=graph, held=pairs)
    assert all(len(v) >= 1 for v in train_g.values())
    assert {frozenset((a, b)) for a, v in test_g.items() for b in v} == pairs
    labels = sites
    idx = neighbour_index(graph=graph, labels=labels)
    assert idx.shape == (6, K_NEIGHBOURS) and (idx[:, 2:] == -1).all() and (idx[:, :2] >= 0).all()
    plc = placebo_index(graph=graph, labels=labels)
    for p, s in enumerate(labels):
        assert all(labels[j] not in graph[s] and labels[j] != s for j in plc[p] if j >= 0)
    clean_arr = np.stack([power[s] for s in sites])
    world = np.stack([mine.power[s] for s in sites])
    code = np.stack([mine.mask[s] for s in sites])
    sp = np.array([[0, 1], [2, 3]])
    si = np.array([[100, 100], [5000, 6000]])
    flag, frac = aux_labels(world=world, clean=clean_arr, code=code, slot_primary=sp, slot_idx=si)
    assert flag.shape == (2, 2) and frac.dtype == np.float16
    hit = np.argwhere(code != inj.NONE)
    s0, t0 = int(hit[0][0]), int(hit[0][1])
    f1, _ = aux_labels(
        world=world,
        clean=clean_arr,
        code=code,
        slot_primary=np.array([[s0]]),
        slot_idx=np.array([[t0]]),
    )
    assert f1[0, 0] in (1, -1), f1
    print(
        "selftest passed:",
        mine.events.group_by("role").len().sort("role").to_dicts(),
        "held-out pairs",
        len(pairs),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest",))
    parser.parse_args()
    selftest()
