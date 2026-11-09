"""Run experiment 670: E1, E1-J, and E1-J on a rewired graph, on injected and real data.

Run from the repository root with the data folder and the imported folders on the path:

    export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
    P=ml-experiments
    export PYTHONPATH=$P/00670-event-table-recipient-search-on-six-cell-grid:\
$P/00660-shared-generator-of-injected-load-transfer-events:\
$P/00653-edge-state-event-table-penalised-segmentation:\
$P/00665-event-table-recall-by-length-and-depth
    uv run --no-sync python $P/00670-*/e00670_run.py injected <config> <first_seed> <n_worlds>
    uv run --no-sync python $P/00670-*/e00670_run.py real

Stage `injected` writes one parquet of scored truth rows per config and first seed. Stage `real`
decodes the real record and writes `real.json` and one events parquet per method.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00660_data as data  # ty: ignore[unresolved-import]
import e00660_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00653_common import DAY, E580, GRID_LENGTH, SCORED_FROM, WORK
from e00653_core import Frozen, decode_world, freeze, match
from e00653_inject import OFFSET_2019
from e00653_run import rewire
from e00653_segment import Event, Params, classify
from e00653_slow import to_half_hours
from e00660_ev34_detector import Rig, truths  # ty: ignore[unresolved-import]
from e00665_score import cell_of, depth_units  # ty: ignore[unresolved-import]
from e00670_search import search_recipients
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00670-event-table-recipient-search-on-six-cell-grid"
METHODS: Final[tuple[str, str, str]] = ("E1", "E1-J", "E1-J rewired")
CONFIGS: Final[dict[str, gen.GeneratorConfig]] = {
    "default": gen.default_config(),
    "long_heavy": gen.long_heavy(),
    "short_heavy": gen.short_heavy(),
    "constant_mw_square": gen.constant_mw_square(),
}
N_REWIRE_REAL: Final[int] = 100
rewired_kept: list[float] = []
"""Share of each rewired graph's links that are also real links (the placebo's leakage)."""
CELLS: Final[tuple[str, ...]] = (
    "long, deep",
    "long, shallow",
    "medium, deep",
    "medium, shallow",
    "short, deep",
    "short, shallow",
    "under 2",
    "unmeasured",
)


def params653() -> Params:
    """Experiment 653's tuned penalties."""
    return replace(Params(), **json.loads((WORK / "params.json").read_text()))


def rewire_low_overlap(*, rig: Rig, rng: np.random.Generator) -> tuple[dict[str, list[str]], float]:
    """The rewiring, out of 30 draws, that keeps the fewest real links, and its kept share."""
    real = rig.p653.neighbours
    best: tuple[dict[str, list[str]], float] | None = None
    for _ in range(30):
        nb = rewire(edges=rig.p653.edges, rng=rng)
        nb = {s: nb.get(s, []) for s in rig.p653.sites}
        links = sum(len(v) for v in nb.values())
        kept = sum(b in real[a] for a in nb for b in nb[a]) / max(1, links)
        if best is None or kept < best[1]:
            best = (nb, kept)
    assert best is not None
    return best


def record_end_of(*, residual: dict[str, np.ndarray]) -> int:
    """One past the last half-hour with a reading at any Primary."""
    end = 0
    for v in residual.values():
        ok = np.flatnonzero(np.isfinite(v))
        end = max(end, int(ok[-1]) + 1 if len(ok) else 0)
    return end


def decode_variants(
    *,
    rig: Rig,
    residual: dict[str, np.ndarray],
    frozen: Frozen,
    params: Params,
    rng: np.random.Generator,
) -> dict[str, list[Event]]:
    """Event tables of E1, E1-J, and E1-J with the graph rewired (degree-preserving)."""
    p653 = rig.p653
    events, slows = decode_world(
        panel=p653, residual=residual, frozen=frozen, arm="masked", params=params
    )
    xh = {s: residual[s] - to_half_hours(slow=slows[s]) for s in slows}
    sigma = frozen.sigma
    nb, kept = rewire_low_overlap(rig=rig, rng=rng)
    rewired_kept.append(kept)
    rewired = classify(
        events=copy.deepcopy(events),
        sigmas={k: (v if np.isfinite(v) else 1.0) for k, v in sigma.items()},
        neighbours=nb,
        p=params,
        record_end=record_end_of(residual=residual),
    )
    return {
        "E1": events,
        "E1-J": search_recipients(events=events, xh=xh, sigma=sigma, neighbours=p653.neighbours),
        "E1-J rewired": search_recipients(events=rewired, xh=xh, sigma=sigma, neighbours=nb),
    }


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
        variants = decode_variants(
            rig=rig,
            residual=residual,
            frozen=rig.frozen,
            params=params,
            rng=np.random.default_rng(seed),
        )
        for method, events in variants.items():
            scored = match(truth=truth, events=events, world_end=GRID_LENGTH)
            n_edge = sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in events)
            n_right = sum(bool(r["right_class"]) for r in scored)
            rows.extend(
                {
                    **r,
                    "config": config_name,
                    "world": seed,
                    "method": method,
                    "world_edge_events": n_edge,
                    "world_right_pairs": n_right,
                }
                for r in scored
            )
        print(config_name, seed, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(OUT / f"injected_{config_name}_{first_seed}.parquet")


def events_table(*, events: list[Event]) -> pl.DataFrame:
    """Events of the scored years with their cell, by stored size and length."""
    rows = []
    for e in events:
        if e.start < SCORED_FROM:
            continue
        days = None if e.end is None else (e.end - e.start) / DAY
        rows.append(
            {
                "primary": e.primary,
                "start": e.start,
                "end": -1 if e.end is None else e.end,
                "amp_mw": e.amp_mw,
                "units": e.units,
                "cls": e.cls,
                "partners": ",".join(e.partners),
                "cell": cell_of(days=days, units=e.units) if e.units >= 2.0 else "under 2",
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "primary": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "amp_mw": pl.Float64,
            "units": pl.Float64,
            "cls": pl.String,
            "partners": pl.String,
            "cell": pl.String,
        },
    )


def logged_shifts(*, rig: Rig, resid: dict[str, np.ndarray]) -> pl.DataFrame:
    """NGED's logged shifts of more than 1 day in the scored years, with panel depth and cell."""
    sigma = json.loads((WORK / "noise_units.json").read_text())["sigma_block_mw"]
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet").filter(
        (pl.col("label_length_days") > 1)
        & pl.col("label").is_in(rig.p653.sites)
        & (pl.col("label_onset_idx") - OFFSET_2019 >= SCORED_FROM)
    )
    out = []
    for r in lab.iter_rows(named=True):
        on = r["label_onset_idx"] - OFFSET_2019
        end = r["label_end_idx"] - OFFSET_2019
        u = depth_units(resid=resid[r["label"]], sigma=sigma[r["label"]], onset=on, end=end)
        if np.isfinite(u) and u >= 2.0:
            cell = cell_of(days=r["label_length_days"], units=u)
        else:
            cell = "unmeasured" if not np.isfinite(u) else "under 2"
        out.append({"primary": r["label"], "onset": on, "cell": cell})
    return pl.DataFrame(out, schema={"primary": pl.String, "onset": pl.Int64, "cell": pl.String})


def matched(*, table: pl.DataFrame, logged: pl.DataFrame, cell: str | None = None) -> list[int]:
    """Logged shifts of a cell, and how many have an event start within 2 and within 24 hours."""
    sub = logged if cell is None else logged.filter(pl.col("cell") == cell)
    starts = {p: table.filter(pl.col("primary") == p)["start"].to_list() for p in sub["primary"]}
    rows = list(sub.iter_rows(named=True))
    strict = sum(any(abs(x - r["onset"]) <= 4 for x in starts[r["primary"]]) for r in rows)
    loose = sum(any(abs(x - r["onset"]) <= 48 for x in starts[r["primary"]]) for r in rows)
    return [sub.height, strict, loose]


def verified_d5_d16(*, events: list[Event]) -> dict:
    """The verified D5 to D16 transfer: the nearby events, and whether any pair links them."""
    v = pl.read_parquet(E580 / "events_verified.parquet").filter(
        (pl.col("primary_a") == "D5")
        & (pl.col("primary_b") == "D16")
        & pl.col("onset_a").str.starts_with("2025-12")
    )
    r = v.row(0, named=True)
    on_a, end_a = r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019
    on_b, end_b = r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019
    near_d5 = [e for e in events if e.primary == "D5" and abs(e.start - on_a) <= 20 * DAY]
    near_d16 = [e for e in events if e.primary == "D16" and abs(e.start - on_b) <= 20 * DAY]
    pairs = [
        {
            "d5_start_err_hh": a.start - on_a,
            "d5_end_err_hh": None if a.end is None else a.end - end_a,
            "d16_start_err_hh": b.start - on_b,
            "d16_end_err_hh": None if b.end is None else b.end - end_b,
            "d5_mw": a.amp_mw,
            "d16_mw": b.amp_mw,
        }
        for a in near_d5
        for b in near_d16
        if a.amp_mw < 0 < b.amp_mw and "D16" in a.partners and "D5" in b.partners
    ]
    return {
        "d5_events": [[e.start - on_a, e.amp_mw, e.cls, e.partners] for e in near_d5],
        "d16_events": [[e.start - on_b, e.amp_mw, e.cls, e.partners] for e in near_d16],
        "paired": pairs,
    }


def event_free(*, rig: Rig, tables: dict[str, pl.DataFrame], scored_years: float) -> dict:
    """Events per Primary-year in time over 14 days from any logged or verified event."""
    spans = []
    lab = pl.read_parquet(E580 / "events_labelled_status.parquet")
    spans += [
        (r["label"], r["label_onset_idx"] - OFFSET_2019, r["label_end_idx"] - OFFSET_2019)
        for r in lab.iter_rows(named=True)
    ]
    for r in pl.read_parquet(E580 / "events_verified.parquet").iter_rows(named=True):
        spans += [
            (r["primary_a"], r["onset_a_idx"] - OFFSET_2019, r["end_a_idx"] - OFFSET_2019),
            (r["primary_b"], r["onset_b_idx"] - OFFSET_2019, r["end_b_idx"] - OFFSET_2019),
        ]
    busy_by = {}
    free_hh = 0
    for site in rig.p653.sites:
        busy = np.zeros(GRID_LENGTH, bool)
        for s, a, b in spans:
            if s == site:
                busy[max(0, a - 14 * DAY) : max(0, b + 14 * DAY)] = True
        busy[:SCORED_FROM] = True
        busy_by[site] = busy
        free_hh += int((~busy & np.isfinite(rig.p653.residual[site])).sum())
    free_years = free_hh / (DAY * 365.25)
    out: dict = {"free_primary_years": free_years, "scored_primary_years": scored_years}
    for m, t in tables.items():
        n_all = n_edge = 0
        for r in t.iter_rows(named=True):
            end = GRID_LENGTH if r["end"] < 0 else r["end"]
            if not busy_by[r["primary"]][r["start"] : max(end, r["start"] + 1)].any():
                n_all += 1
                n_edge += r["cls"] in ("edge_transfer", "permanent_transfer")
        out[m] = {"events": n_all, "edge_events": n_edge, "edge_per_year": n_edge / free_years}
    return out


def run_real() -> None:
    """Decode the real record with each method and write the real-record tables."""
    rig = Rig()
    params = params653()
    resid = rig.real_residual
    frozen = freeze(panel=rig.p653, residual=resid)
    rng = np.random.default_rng(670)
    variants = decode_variants(rig=rig, residual=resid, frozen=frozen, params=params, rng=rng)
    scored_years = sum(
        int(np.isfinite(rig.p653.residual[s][SCORED_FROM:GRID_LENGTH]).sum())
        for s in rig.p653.sites
    ) / (DAY * 365.25)
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
            "events_by_cell": {c: t.filter(pl.col("cell") == c).height for c in CELLS},
            "logged_matched_all": matched(table=t, logged=logged),
            "logged_matched_by_cell": {c: matched(table=t, logged=logged, cell=c) for c in CELLS},
        }
        if m != "E1-J rewired":
            res[m]["d5_d16"] = verified_d5_d16(events=ev)
    ev0, slows = decode_world(
        panel=rig.p653, residual=resid, frozen=frozen, arm="masked", params=params
    )
    xh = {s: resid[s] - to_half_hours(slow=slows[s]) for s in slows}
    sig = {k: (v if np.isfinite(v) else 1.0) for k, v in frozen.sigma.items()}
    counts = []
    for _ in range(N_REWIRE_REAL):
        nb, kept = rewire_low_overlap(rig=rig, rng=rng)
        rewired_kept.append(kept)
        e2 = classify(
            events=copy.deepcopy(ev0),
            sigmas=sig,
            neighbours=nb,
            p=params,
            record_end=record_end_of(residual=resid),
        )
        e3 = search_recipients(events=e2, xh=xh, sigma=frozen.sigma, neighbours=nb)
        counts.append(sum(e.cls.endswith("transfer") and e.start >= SCORED_FROM for e in e3))
    res["rewired_edge_events"] = counts
    res["rewired_share_of_real_links_kept"] = float(np.mean(rewired_kept))
    stored = pl.read_parquet(WORK / "real_events_masked.parquet").filter(
        pl.col("start") >= SCORED_FROM
    )
    res["gate2_e1_real_events"] = [res["E1"]["events"], stored.height]
    res["event_free"] = event_free(rig=rig, tables=tables, scored_years=scored_years)
    res["real_graph_edge_events"] = res["E1-J"]["edge_events"]
    (OUT / "real.json").write_text(json.dumps(res, indent=1, default=float))
    for m in METHODS:
        print(m, {k: v for k, v in res[m].items() if k != "d5_d16"})
    print("rewired median", float(np.median(counts)), "real graph", res["real_graph_edge_events"])
    assert data.GRID_LENGTH == GRID_LENGTH


if __name__ == "__main__":
    if sys.argv[1] == "injected":
        run_injected(
            config_name=sys.argv[2], first_seed=int(sys.argv[3]), n_worlds=int(sys.argv[4])
        )
    else:
        run_real()
