"""Run experiment 680: E1-J, E1-J-sym, and E1-J-sym on a rewired graph.

Run from the repository root:

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy
    export NGED_S3_BUCKET_SECRET=dummy
    P=ml-experiments
    export PYTHONPATH=$P/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python $P/00680-*/e00680_run.py injected <config> <first_seed> <n_worlds>
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python $P/00680-*/e00680_run.py real
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, SCORED_FROM, WORK
from e00653_core import Frozen, decode_world, freeze, match
from e00653_segment import Event, Params, classify
from e00653_slow import to_half_hours
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00670_run import (  # ty: ignore[unresolved-import]
    CELLS,
    CONFIGS,
    N_REWIRE_REAL,
    event_free,
    events_table,
    logged_shifts,
    matched,
    params653,
    record_end_of,
    rewire_low_overlap,
    verified_d5_d16,
)
from e00670_search import search_recipients
from e00680_search import search_both_ends
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00660_generator as gen  # ty: ignore[unresolved-import]  # isort: skip

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search"
)
METHODS: Final[tuple[str, str, str]] = ("E1-J", "E1-J-sym", "E1-J-sym rewired")


def decode_variants(
    *, rig: Rig, residual: dict, frozen: Frozen, params: Params, rng: np.random.Generator
) -> tuple[dict[str, list[Event]], float]:
    """Event tables of E1-J, E1-J-sym, and E1-J-sym on a rewired graph, and the share kept."""
    p653 = rig.p653
    events, slows = decode_world(
        panel=p653, residual=residual, frozen=frozen, arm="masked", params=params
    )
    xh = {s: residual[s] - to_half_hours(slow=slows[s]) for s in slows}
    sigma = frozen.sigma
    nb, kept = rewire_low_overlap(rig=rig, rng=rng)
    rewired = classify(
        events=copy.deepcopy(events),
        sigmas={k: (v if np.isfinite(v) else 1.0) for k, v in sigma.items()},
        neighbours=nb,
        p=params,
        record_end=record_end_of(residual=residual),
    )
    return {
        "E1-J": search_recipients(events=events, xh=xh, sigma=sigma, neighbours=p653.neighbours),
        "E1-J-sym": search_both_ends(events=events, xh=xh, sigma=sigma, neighbours=p653.neighbours),
        "E1-J-sym rewired": search_both_ends(events=rewired, xh=xh, sigma=sigma, neighbours=nb),
    }, kept


def run_injected(*, config_name: str, first_seed: int, n_worlds: int) -> None:
    """Score the three methods on injected worlds of one generator configuration."""
    rig = Rig()
    params = params653()
    rows = []
    for seed in range(first_seed, first_seed + n_worlds):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS[config_name], seed=seed, start_times=rig.starts
        )
        residual = rig.residual_of(world=world)
        truth = truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH)
        variants, _ = decode_variants(
            rig=rig,
            residual=residual,
            frozen=rig.frozen,
            params=params,
            rng=np.random.default_rng(seed),
        )
        for method, events in variants.items():
            scored = match(truth=truth, events=events, world_end=GRID_LENGTH)
            decoded = [e for e in events if e.cls.endswith("transfer") and e.start >= SCORED_FROM]
            n_right = sum(bool(r["right_class"]) for r in scored)
            rows.extend(
                {
                    **r,
                    "config": config_name,
                    "world": seed,
                    "method": method,
                    "world_edge_events": len(decoded),
                    "world_right_pairs": n_right,
                }
                for r in scored
            )
        print(config_name, seed, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT / f"injected_{config_name}_{first_seed}.parquet")


def run_real() -> None:
    """Decode the real record with each method and write the real-record tables."""
    rig = Rig()
    params = params653()
    resid = rig.real_residual
    frozen = freeze(panel=rig.p653, residual=resid)
    rng = np.random.default_rng(680)
    variants, _ = decode_variants(rig=rig, residual=resid, frozen=frozen, params=params, rng=rng)
    scored_years = sum(
        int(np.isfinite(rig.p653.residual[s][SCORED_FROM:GRID_LENGTH]).sum())
        for s in rig.p653.sites
    ) / (48 * 365.25)
    logged = logged_shifts(rig=rig, resid=resid)
    res: dict = {"scored_primary_years": scored_years, "logged_total": logged.height}
    OUT.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pl.DataFrame] = {}
    for m, ev in variants.items():
        t = events_table(events=ev)
        tables[m] = t
        t.write_parquet(OUT / f"real_events_{m.replace(' ', '_')}.parquet")
        edge = t.filter(pl.col("cls").is_in(["edge_transfer", "permanent_transfer"]))
        res[m] = {
            "events": t.height,
            "events_per_primary_year": t.height / scored_years,
            "edge_events": edge.height,
            "edge_events_by_cell": {c: edge.filter(pl.col("cell") == c).height for c in CELLS},
            "logged_matched_all": matched(table=t, logged=logged),
        }
        if m != "E1-J-sym rewired":
            res[m]["d5_d16"] = verified_d5_d16(events=ev)
    ev0, slows = decode_world(
        panel=rig.p653, residual=resid, frozen=frozen, arm="masked", params=params
    )
    xh = {s: resid[s] - to_half_hours(slow=slows[s]) for s in slows}
    sig = {k: (v if np.isfinite(v) else 1.0) for k, v in frozen.sigma.items()}
    counts, kept_all = [], []
    for _ in range(N_REWIRE_REAL):
        nb, kept = rewire_low_overlap(rig=rig, rng=rng)
        kept_all.append(kept)
        e2 = classify(
            events=copy.deepcopy(ev0),
            sigmas=sig,
            neighbours=nb,
            p=params,
            record_end=record_end_of(residual=resid),
        )
        e3 = search_both_ends(events=e2, xh=xh, sigma=frozen.sigma, neighbours=nb)
        counts.append(sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in e3))
    res["rewired_edge_events"] = counts
    res["rewired_share_of_real_links_kept"] = float(np.mean(kept_all))
    stored = pl.read_parquet(WORK / "real_events_masked.parquet").filter(
        pl.col("start") >= SCORED_FROM
    )
    res["event_free"] = event_free(rig=rig, tables=tables, scored_years=scored_years)
    res["stored_e1_real_events"] = stored.height
    (OUT / "real.json").write_text(json.dumps(res, indent=1, default=float))
    for m in METHODS:
        print(m, {k: v for k, v in res[m].items() if k != "d5_d16"})
    print("rewired median", float(np.median(counts)))


if __name__ == "__main__":
    if sys.argv[1] == "injected":
        run_injected(
            config_name=sys.argv[2], first_seed=int(sys.argv[3]), n_worlds=int(sys.argv[4])
        )
    else:
        run_real()
