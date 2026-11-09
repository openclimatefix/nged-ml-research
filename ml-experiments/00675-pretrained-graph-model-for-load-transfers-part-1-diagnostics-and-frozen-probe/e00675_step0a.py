"""Step 0a of experiment 675: why E1-J leaves found recipient megawatts unpaired.

Re-runs E1 and E1-J (experiment 670's code, unchanged) on experiment 670's default worlds and sorts
each injected recipient end that E1-J found but did not pair into the first of six reasons that
applies. Run (from the repository root):

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    P=ml-experiments
    export PYTHONPATH=$P/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    uv run --no-sync python $P/00675-*/e00675_step0a.py <first_seed> <n_worlds>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM
from e00653_segment import Event
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00665_score import cell_of  # ty: ignore[unresolved-import]
from e00670_run import CONFIGS, decode_variants, params653  # ty: ignore[unresolved-import]
from e00670_search import SearchParams
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe"
REASONS: Final[tuple[str, ...]] = (
    "1 donor not found",
    "2 recipient start outside search window",
    "3 megawatt ratio outside share window",
    "4 linked to the wrong donor",
    "5 transfer split across 2 or 3 recipients",
    "6 other",
)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def match_events(*, truth: list, events: list[Event]) -> dict[int, Event | None]:
    """Experiment 653's matching rule, returning the matched event of each truth (by index)."""
    taken: set[int] = set()
    out: dict[int, Event | None] = {}
    order = sorted(range(len(truth)), key=lambda i: -abs(truth[i].amp_mw))
    for i in order:
        t = truth[i]
        t_end = GRID_LENGTH if t.end is None else t.end
        best, best_iou = None, 0.0
        for e in events:
            if id(e) in taken or e.primary != t.site or np.sign(e.amp_mw) != np.sign(t.amp_mw):
                continue
            e_end = GRID_LENGTH if e.end is None else e.end
            iou = _iou((t.start, t_end), (e.start, e_end))
            if abs(e.start - t.start) <= 24 and iou >= 0.5 and iou > best_iou:
                best, best_iou = e, iou
        if best is not None:
            taken.add(id(best))
        out[i] = best
    return out


def classify_world(*, truth: list, events: list[Event], world: int) -> list[dict]:
    """One row per found recipient end that E1-J did not pair, with its reason."""
    sp = SearchParams()
    m = match_events(truth=truth, events=events)
    rows = []
    for i, t in enumerate(truth):
        if t.kind != "edge" or t.amp_mw <= 0 or not t.partners:
            continue
        e_r = m[i]
        if e_r is None:
            continue
        paired = e_r.cls in ("edge_transfer", "permanent_transfer") and bool(
            set(t.partners) & set(e_r.partners)
        )
        if paired:
            continue
        donors = [(j, d) for j, d in enumerate(truth) if d.group == t.group and d.amp_mw < 0]
        if not donors:
            continue
        j, d = donors[0]
        e_d = m[j]
        if e_d is None:
            reason = REASONS[0]
        elif not sp.lag_lo <= e_r.start - e_d.start <= sp.lag_hi:
            reason = REASONS[1]
        elif not sp.share_low <= e_r.amp_mw / max(abs(e_d.amp_mw), 1e-9) <= sp.share_high:
            reason = REASONS[2]
        elif e_r.partners and e_d.primary not in e_r.partners:
            reason = REASONS[3]
        elif len(d.partners) >= 2:
            reason = REASONS[4]
        else:
            reason = REASONS[5]
        days = None if t.end is None else (t.end - t.start) / 48
        rows.append(
            {
                "world": world,
                "site": t.site,
                "amp_mw": t.amp_mw,
                "units": t.units,
                "cell": cell_of(days=days, units=t.units) if t.units >= 2.0 else "under 2",
                "reason": reason,
                "n_recipients": len(d.partners),
                "recipient_cls": e_r.cls,
                "lag_hh": None if e_d is None else e_r.start - e_d.start,
                "donor_cls": None if e_d is None else e_d.cls,
            }
        )
    return rows


def main() -> None:
    """Run the worlds `first_seed` to `first_seed + n_worlds - 1` and write the rows."""
    first, n = int(sys.argv[1]), int(sys.argv[2])
    rig = Rig()
    params = params653()
    rows: list[dict] = []
    for seed in range(first, first + n):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS["default"], seed=seed, start_times=rig.starts
        )
        residual = rig.residual_of(world=world)
        truth = truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH)
        ev = decode_variants(
            rig=rig,
            residual=residual,
            frozen=rig.frozen,
            params=params,
            rng=np.random.default_rng(seed),
        )["E1-J"]
        rows.extend(classify_world(truth=truth, events=ev, world=seed))
        print(seed, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT / f"step0a_rows_{first}_{n}.parquet")


if __name__ == "__main__":
    main()
