"""Experiment 678: donor alarms of experiment 675's probes fed into experiment 670's recipient search.

Run (from the repository root; see `e00675_step0a.py` for the environment) on worlds `first_seed`
to `first_seed + n_worlds - 1`:

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00678-donor-posterior-hybrid-for-load-transfer-pairing:\
$P/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    uv run --no-sync python $P/00678-*/e00678_run.py <first_seed> <n_worlds>

Each hybrid is the table of E1 plus one added downward donor event per probe alarm, searched by
experiment 670's `search_recipients` unchanged. Truth never enters the construction: the donor's
start and end come from the arm-C residual around the alarm.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM
from e00653_core import decode_world, match
from e00653_segment import Event, classify
from e00653_slow import to_half_hours
from e00660_ev34_detector import truths  # ty: ignore[unresolved-import]
from e00660_ev34_detector import Rig
from e00665_score import cell_of  # ty: ignore[unresolved-import]
from e00670_run import CONFIGS, params653, record_end_of, rewire_low_overlap  # ty: ignore[unresolved-import]
from e00670_search import search_recipients, step_size
from e00675_pretrain import ROOT as PROBE_ROOT
from e00675_score import SCORED_H, episodes
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00678-donor-posterior-hybrid-for-load-transfer-pairing"
)
PROBE: Final[Path] = PROBE_ROOT / "probe"
ARMS: Final[tuple[tuple[str, int], ...]] = (
    ("F", 0),
    *((a, k) for a in ("R", "P-M", "P-C") for k in (0, 1, 2)),
)
BEFORE_HH: Final[int] = 48
"""Half-hours before the alarm hour at which the donor start search begins (24 hours)."""
AFTER_HH: Final[int] = 24
"""Half-hours after the alarm hour at which the donor start search stops (12 hours)."""
DUPLICATE_HH: Final[int] = 24
"""An alarm whose start lies this close to an E1 or earlier added downward donor adds nothing."""
MIN_STAY_HH: Final[int] = 96
"""Earliest end of an added donor, relative to its start (2 days)."""
LEVEL_WINDOW_HH: Final[int] = 48
"""Running-median window of the end search (1 day)."""
MAX_STAY_HH: Final[int] = 60 * 48
"""Longest stay searched for an end; a donor still shifted after 60 days has no end."""
PERMANENT_DAYS: Final[float] = 180.0
"""Days of record left after an open-ended donor for it to count as permanent (experiment 653)."""


def thresholds() -> dict[tuple[str, int], float]:
    """Each probe's alarm threshold, set in experiment 675 at E1-J's false-donor rate."""
    cells = json.loads((PROBE / "summary.json").read_text())["cells"]["long, deep"]
    return {(a, k): cells[f"{a}/{k}"]["threshold"] for a, k in ARMS}


def donor_end(*, xh: np.ndarray, start: int, step: float) -> int | None:
    """First half-hour from 2 days after `start` where the 1-day median has recovered half the step."""
    before = xh[max(0, start - 3 * DAY) : start]
    if np.isfinite(before).sum() < 48:
        return None
    target = np.nanmedian(before) + step / 2
    for s in range(start + MIN_STAY_HH, min(len(xh) - LEVEL_WINDOW_HH, start + MAX_STAY_HH), 12):
        block = xh[s : s + LEVEL_WINDOW_HH]
        if np.isfinite(block).sum() >= 12 and np.nanmedian(block) > target:
            return s
    return None


def added_donors(
    *,
    events: list[Event],
    xh: dict[str, np.ndarray],
    sigma: dict[str, float],
    score: np.ndarray,
    sites: list[str],
    thr: float,
    record_end: int,
) -> list[Event]:
    """One downward donor event per probe alarm that no E1 downward event already covers."""
    out: list[Event] = []
    for i, site in enumerate(sites):
        if site not in xh or not np.isfinite(sigma.get(site, float("nan"))):
            continue
        existing = [
            e.start for e in events if e.primary == site and e.amp_mw < 0 and e.units >= 4.0
        ]
        for a, _ in episodes(hot=score[i, SCORED_H:] > thr, score=score[i, SCORED_H:]):
            centre = 2 * (SCORED_H + a)
            best: tuple[int, float] | None = None
            for s in range(centre - BEFORE_HH, centre + AFTER_HH + 1):
                if s < 0 or s >= len(xh[site]):
                    continue
                d = step_size(xh=xh[site], start=s, end=None)
                if np.isfinite(d) and (best is None or d < best[1]):
                    best = (s, d)
            if best is None or best[1] >= 0:
                continue
            start, step = best
            if any(abs(start - x) <= DUPLICATE_HH for x in existing):
                continue
            end = donor_end(xh=xh[site], start=start, step=step)
            permanent = end is None and (record_end - start) / DAY >= PERMANENT_DAYS
            existing.append(start)
            out.append(
                Event(
                    primary=site,
                    start=start,
                    end=end,
                    amp_mw=step,
                    units=abs(step) / sigma[site],
                    confidence=0.5,
                    ramp_fraction=0.0,
                    kind="permanent" if permanent else ("open" if end is None else "temporary"),
                    cls="permanent_shift" if permanent else "shift_no_partner",
                )
            )
    return out


def score_rows(*, truth: list, events: list[Event], method: str, world: int) -> list[dict]:
    """Scored truth rows of one method in one world, with the world's decoded-transfer counts."""
    scored = match(truth=truth, events=events, world_end=GRID_LENGTH)
    n_edge = sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in events)
    n_right = sum(bool(r["right_class"]) for r in scored)
    return [
        {
            **r,
            "world": world,
            "method": method,
            "world_edge_events": n_edge,
            "world_right_pairs": n_right,
        }
        for r in scored
    ]


def main() -> None:
    """Run worlds `first_seed` to `first_seed + n_worlds - 1` and write the scored rows."""
    first, n = int(sys.argv[1]), int(sys.argv[2])
    rig = Rig()
    params = params653()
    thr = thresholds()
    sites = list(rig.p653.sites)
    rows: list[dict] = []
    for seed in range(first, first + n):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        residual = rig.residual_of(world=world)
        truth = truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH)
        events, slows = decode_world(
            panel=rig.p653, residual=residual, frozen=rig.frozen, arm="masked", params=params
        )
        xh = {s: residual[s] - to_half_hours(slow=slows[s]) for s in slows}
        sigma = rig.frozen.sigma
        record_end = record_end_of(residual=residual)
        nb_rewired, _ = rewire_low_overlap(rig=rig, rng=np.random.default_rng(seed))
        rewired_base = classify(
            events=copy.deepcopy(events),
            sigmas={k: (v if np.isfinite(v) else 1.0) for k, v in sigma.items()},
            neighbours=nb_rewired,
            p=params,
            record_end=record_end,
        )
        tables = {
            "E1": events,
            "E1-J": search_recipients(
                events=events, xh=xh, sigma=sigma, neighbours=rig.p653.neighbours
            ),
            "E1-J rewired": search_recipients(
                events=rewired_base, xh=xh, sigma=sigma, neighbours=nb_rewired
            ),
        }
        for (arm, k), t in thr.items():
            score = np.load(PROBE / f"score_{arm}_seed{k}_world{seed}.npy")
            for graph, base, nb in (
                ("", events, rig.p653.neighbours),
                (" rewired", rewired_base, nb_rewired),
            ):
                if graph and arm not in ("F", "R"):
                    continue
                donors = added_donors(
                    events=base, xh=xh, sigma=sigma, score=score, sites=sites, thr=t,
                    record_end=record_end,
                )  # fmt: skip
                tables[f"{arm}-hyb{graph}/{k}"] = search_recipients(
                    events=base + donors, xh=xh, sigma=sigma, neighbours=nb
                )
                tables[f"{arm}-donors{graph}/{k}"] = [
                    d for d in donors if d.primary in xh
                ]  # alarm-derived donors alone, for timing and counts
        for method, ev in tables.items():
            if "-donors" in method:
                continue
            rows.extend(score_rows(truth=truth, events=ev, method=method, world=seed))
        print(seed, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        cell=pl.Series(
            [
                cell_of(days=None if r["end"] < 0 else r["days"], units=r["units"])
                if r["units"] >= 2.0
                else "under 2"
                for r in rows
            ]
        )
    )
    df.write_parquet(OUT / f"rows_{first}_{n}.parquet")


if __name__ == "__main__":
    main()
