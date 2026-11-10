"""Run experiment 682: E1-F (edge flows chosen by cost) and its placebo arms.

Run from the repository root:

    export DATA_PATH_INTERNAL=<path to the internal data folder>
    export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy
    export NGED_S3_BUCKET_SECRET=dummy
    P=ml-experiments
    export PYTHONPATH=$P/00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model:\
$P/00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search:\
$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    uv run --no-sync python $P/00682-*/e00682_run.py tune <config> <n_worlds>
    uv run --no-sync python $P/00682-*/e00682_run.py injected <config> <first_seed> <n> <costs>
    uv run --no-sync python $P/00682-*/e00682_run.py real <costs>

`<costs>` names a file `costs_<name>.json` written by stage `tune` (name `default` or
`constant_mw_square`). The scored worlds are experiment 680's, so E1-J and E1-J-sym are read
from the stored rows of experiments 670 and 680.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00653_common import GRID_LENGTH, GRID_START, SCORED_FROM, idx_of
from e00653_core import Frozen, decode_world, freeze, match
from e00653_segment import Event, Params
from e00653_slow import to_half_hours
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00665_score import with_cells  # ty: ignore[unresolved-import]
from e00670_run import (  # ty: ignore[unresolved-import]
    CELLS,
    CONFIGS,
    N_REWIRE_REAL,
    event_free,
    events_table,
    logged_shifts,
    matched,
    params653,
    rewire_low_overlap,
    verified_d5_d16,
)
from e00670_search import search_recipients
from e00682_flow import FlowCosts, candidates, reconstruct
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model"
)
TRAIN_END: Final[int] = idx_of(when=datetime(2023, 7, 1, tzinfo=UTC))
"""Validation events start from 2023-07-01; the world's readings stop at 2024-01-01."""
VAL_SEED0: Final[int] = 20_000
C_EDGE_GRID: Final[tuple[float, ...]] = (2.0, 4.0, 6.0, 8.0, 12.0)
DELTA_GRID: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0, 8.0)
WIDE_C_EDGE_GRID: Final[tuple[float, ...]] = (8.0, 16.0, 32.0)
WIDE_DELTA_GRID: Final[tuple[float, ...]] = (8.0, 16.0, 32.0, 64.0)
"""Post hoc grid, run only because the plan's grid met no pair under E1-J's false share."""
ARMS: Final[tuple[str, ...]] = ("E1-F", "E1-F rewired", "E1-F no edges", "E1-F all pairs")
DEEP: Final[tuple[str, str]] = ("long, deep", "medium, deep")


def validation_world(*, rig: Rig, config: gen.GeneratorConfig, seed: int) -> gen.InjectedWorld:
    """Events before 2024-01-01 injected into the panel with every later reading hidden."""
    idx = np.arange(rig.panel.length)
    hidden = replace(
        rig.panel,
        power={s: np.where(idx < SCORED_FROM, v, np.nan) for s, v in rig.panel.power.items()},
    )
    cutoff = datetime.fromtimestamp(GRID_START.timestamp() + 1800 * SCORED_FROM, tz=UTC)
    return gen.inject(
        panel=hidden, config=replace(config, cutoff=cutoff), seed=seed, start_times=rig.starts
    )


def graphs(*, rig: Rig, rng: np.random.Generator) -> dict[str, dict[str, list[str]]]:
    """Edge sets of the four arms: real, rewired, none, and every pair of Primaries."""
    sites = list(rig.p653.sites)
    nb, _ = rewire_low_overlap(rig=rig, rng=rng)
    return {
        "E1-F": rig.p653.neighbours,
        "E1-F rewired": nb,
        "E1-F no edges": {s: [] for s in sites},
        "E1-F all pairs": {s: [t for t in sites if t != s] for s in sites},
    }


def decode_inputs(
    *, rig: Rig, residual: dict, params: Params, frozen: Frozen | None = None
) -> tuple[list[Event], dict[str, np.ndarray]]:
    """E1's events and the residual minus its slow level."""
    events, slows = decode_world(
        panel=rig.p653, residual=residual, frozen=frozen or rig.frozen, arm="masked", params=params
    )
    return events, {s: residual[s] - to_half_hours(slow=slows[s]) for s in slows}


def match_validation(*, truth: list, events: list[Event]) -> list[dict]:
    """Score validation events whose true end may lie beyond the hidden readings.

    The overlap test uses each truth end clipped to the end of the readings. The reported
    length, and so the cell, keeps the true end.
    """
    clipped = [
        replace(t, end=None if t.end is not None and t.end > SCORED_FROM else t.end) for t in truth
    ]
    rows = match(truth=clipped, events=events, world_end=SCORED_FROM)
    true_days = {(t.site, t.start): t for t in truth}
    for row in rows:
        t = true_days[(row["site"], row["start"])]
        row["days"] = -1.0 if t.end is None else (t.end - t.start) / 48
    return rows


def sums(*, rows: list[dict], decoded: int) -> dict:
    """Deep-cell paired megawatts, their total, correct ends, and decoded transfers."""
    df = with_cells(df=pl.DataFrame(rows))
    d = df.filter(pl.col("cell").is_in(DEEP) & pl.col("edge_truth"))
    return {
        "paired": float((d["right_class"] * d["amp_mw"].abs()).sum()),
        "total": float(d["amp_mw"].abs().sum()),
        "right": sum(bool(x["right_class"]) for x in rows),
        "decoded": decoded,
    }


def tune(*, config_name: str, n_worlds: int, wide: bool = False) -> None:
    """Choose the change costs on validation worlds from the fixed grid (rule in the plan)."""
    rig = Rig()
    params = params653()
    grid = list(
        itertools.product(
            WIDE_C_EDGE_GRID if wide else C_EDGE_GRID, WIDE_DELTA_GRID if wide else DELTA_GRID
        )
    )
    ref_dec = ref_right = 0
    acc = {g: {"paired": 0.0, "total": 0.0, "right": 0, "decoded": 0} for g in grid}
    for k in range(n_worlds):
        world = validation_world(rig=rig, config=CONFIGS[config_name], seed=VAL_SEED0 + k)
        events, xh = decode_inputs(rig=rig, residual=rig.residual_of(world=world), params=params)
        truth = truths(events=world.events, lo=TRAIN_END, hi=SCORED_FROM)
        nb = rig.p653.neighbours
        cands = candidates(events=events, xh=xh, sigma=rig.frozen.sigma, neighbours=nb)
        j = search_recipients(events=events, xh=xh, sigma=rig.frozen.sigma, neighbours=nb)
        ref_dec += sum(e.cls.endswith("transfer") and e.start >= TRAIN_END for e in j)
        ref_right += sum(bool(r["right_class"]) for r in match_validation(truth=truth, events=j))
        for g in grid:
            ev = reconstruct(
                events=events,
                xh=xh,
                sigma=rig.frozen.sigma,
                neighbours=nb,
                costs=FlowCosts(*g),
                cands=cands,
            )
            rows = match_validation(truth=truth, events=ev)
            dec = sum(e.cls.endswith("transfer") and e.start >= TRAIN_END for e in ev)
            s = sums(rows=rows, decoded=dec)
            for key in ("paired", "total", "right", "decoded"):
                acc[g][key] += s[key]
        print(config_name, "validation world", k, flush=True)
    table = []
    for g, a in acc.items():
        paired = a["paired"] / a["total"] if a["total"] else float("nan")
        false = 1 - a["right"] / a["decoded"] if a["decoded"] else float("nan")
        table.append({"c_edge": g[0], "delta": g[1], "deep_paired": paired, "false_share": false})
    ref_false = 1 - ref_right / max(1, ref_dec)
    table = [t for t in table if np.isfinite(t["deep_paired"]) and np.isfinite(t["false_share"])]
    ok = [t for t in table if t["false_share"] <= ref_false]
    pool = ok or table
    best = max(pool, key=lambda t: (t["deep_paired"], t["delta"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"costs_{config_name}{'_wide' if wide else ''}.json").write_text(
        json.dumps(
            {"chosen": best, "fallback": not ok, "e1j_false_share": ref_false, "table": table},
            indent=1,
        )
    )
    print("chosen", best, "E1-J false share", ref_false)


def load_costs(*, name: str) -> FlowCosts:
    """The change costs chosen by `tune`."""
    c = json.loads((OUT / f"costs_{name}.json").read_text())["chosen"]
    return FlowCosts(c_edge=c["c_edge"], delta=c["delta"])


def run_injected(*, config_name: str, first_seed: int, n_worlds: int, costs_name: str) -> None:
    """Score the four arms on injected worlds of one generator configuration."""
    rig = Rig()
    params = params653()
    costs = load_costs(name=costs_name)
    rows = []
    for seed in range(first_seed, first_seed + n_worlds):
        world = gen.inject(
            panel=rig.panel, config=CONFIGS[config_name], seed=seed, start_times=rig.starts
        )
        events, xh = decode_inputs(rig=rig, residual=rig.residual_of(world=world), params=params)
        truth = truths(events=world.events, lo=SCORED_FROM, hi=GRID_LENGTH)
        for arm, nb in graphs(rig=rig, rng=np.random.default_rng(seed)).items():
            ev = reconstruct(
                events=events, xh=xh, sigma=rig.frozen.sigma, neighbours=nb, costs=costs
            )
            scored = match(truth=truth, events=ev, world_end=GRID_LENGTH)
            n_dec = sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in ev)
            n_right = sum(bool(r["right_class"]) for r in scored)
            rows.extend(
                {
                    **r,
                    "config": config_name,
                    "world": seed,
                    "method": arm,
                    "world_edge_events": n_dec,
                    "world_right_pairs": n_right,
                }
                for r in scored
            )
        print(config_name, seed, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{config_name}_{first_seed}_{costs_name}"
    pl.DataFrame(rows).write_parquet(OUT / f"injected_{tag}.parquet")


def run_real(*, costs_name: str) -> None:
    """Decode the real record with each arm."""
    rig = Rig()
    params = params653()
    costs = load_costs(name=costs_name)
    resid = rig.real_residual
    frozen = freeze(panel=rig.p653, residual=resid)
    events, xh = decode_inputs(rig=rig, residual=resid, params=params, frozen=frozen)
    rng = np.random.default_rng(682)
    scored_years = sum(
        int(np.isfinite(rig.p653.residual[s][SCORED_FROM:GRID_LENGTH]).sum())
        for s in rig.p653.sites
    ) / (48 * 365.25)
    logged = logged_shifts(rig=rig, resid=resid)
    res: dict = {"scored_primary_years": scored_years, "logged_total": logged.height}
    OUT.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pl.DataFrame] = {}
    for arm, nb in graphs(rig=rig, rng=rng).items():
        ev = reconstruct(events=events, xh=xh, sigma=frozen.sigma, neighbours=nb, costs=costs)
        t = events_table(events=ev)
        tables[arm] = t
        t.write_parquet(OUT / f"real_events_{arm.replace(' ', '_')}.parquet")
        edge = t.filter(pl.col("cls").is_in(["edge_transfer", "permanent_transfer"]))
        res[arm] = {
            "events": t.height,
            "edge_events": edge.height,
            "edge_per_primary_year": edge.height / scored_years,
            "edge_events_by_cell": {c: edge.filter(pl.col("cell") == c).height for c in CELLS},
            "logged_matched_all": matched(table=t, logged=logged),
        }
        if arm in ("E1-F", "E1-F all pairs"):
            res[arm]["d5_d16"] = verified_d5_d16(events=ev)
            res[arm]["edge_rows"] = edge.to_dicts()
    j = search_recipients(events=events, xh=xh, sigma=frozen.sigma, neighbours=rig.p653.neighbours)
    jt = events_table(events=j)
    jt.write_parquet(OUT / "real_events_E1-J.parquet")
    res["E1-J"] = {
        "edge_events": jt.filter(
            pl.col("cls").is_in(["edge_transfer", "permanent_transfer"])
        ).height
    }
    counts, kept_all = [], []
    for _ in range(N_REWIRE_REAL):
        nb, kept = rewire_low_overlap(rig=rig, rng=rng)
        kept_all.append(kept)
        ev = reconstruct(events=events, xh=xh, sigma=frozen.sigma, neighbours=nb, costs=costs)
        counts.append(sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in ev))
    res["rewired_edge_events"] = counts
    res["rewired_share_of_real_links_kept"] = float(np.mean(kept_all))
    res["event_free"] = event_free(rig=rig, tables=tables, scored_years=scored_years)
    (OUT / f"real_{costs_name}.json").write_text(json.dumps(res, indent=1, default=float))
    for arm in ARMS:
        print(arm, {k: v for k, v in res[arm].items() if k not in ("d5_d16", "edge_rows")})
    print("rewired median", float(np.median(counts)))


if __name__ == "__main__":
    if sys.argv[1] in ("tune", "tune_wide"):
        tune(config_name=sys.argv[2], n_worlds=int(sys.argv[3]), wide=sys.argv[1] == "tune_wide")
    elif sys.argv[1] == "injected":
        run_injected(
            config_name=sys.argv[2],
            first_seed=int(sys.argv[3]),
            n_worlds=int(sys.argv[4]),
            costs_name=sys.argv[5],
        )
    else:
        run_real(costs_name=sys.argv[2])
