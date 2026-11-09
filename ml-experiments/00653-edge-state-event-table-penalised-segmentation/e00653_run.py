"""Run experiment 653: tune on 2022 and 2023, score injected worlds, decode the real record.

    DATA_PATH_INTERNAL=... PYTHONPATH=. uv run --no-sync python e00653_run.py <stage>

Stages: `tune`, `inject`, `real`. Each writes tables under the experiment's data folder.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, replace

import numpy as np
import polars as pl
from e00653_common import DAY, GRID_LENGTH, SCORED_FROM, WORK, Panel
from e00653_core import ARMS, absorption, decode_world, events_frame, freeze, match
from e00653_inject import GenParams, inject
from e00653_segment import Params

TUNE_ON = (4.0, 6.0, 8.0)
TUNE_STAY = (0.02, 0.05, 0.1)
MAX_REAL_RATE = 6.0
"""Largest decoded-event rate per Primary-year on the real 2022 and 2023 residual."""
N_TUNE_WORLDS = 3
N_WORLDS = 6


def setup() -> tuple[Panel, dict[str, np.ndarray], object]:
    """Panel, residual with the common mode removed, and the frozen noise units and slow levels."""
    panel = Panel()
    res = panel.mask_common_mode(residual=panel.residual)
    return panel, res, freeze(panel=panel, residual=res)


def big(rows: pl.DataFrame) -> pl.DataFrame:
    """P1 subset: 4 noise units or more and 1 day or more (or no end)."""
    return rows.filter((pl.col("units") >= 4) & ((pl.col("days") >= 1) | (pl.col("days") < 0)))


def mw_recall(rows: pl.DataFrame, col: str) -> float:
    """MW-weighted share of rows with `col` true."""
    w = rows["amp_mw"].abs()
    return float((rows[col].cast(pl.Float64) * w).sum() / w.sum()) if rows.height else float("nan")


def tune() -> Params:
    """Pick penalties on the 2022 and 2023 worlds, label-free."""
    panel, res, fz = setup()
    n_years = SCORED_FROM / (365.25 * DAY)
    tune_res = {s: np.where(np.arange(GRID_LENGTH) < SCORED_FROM, v, np.nan) for s, v in res.items()}
    worlds = [inject(panel=panel, residual=tune_res, sigma=fz.sigma,
                     rng=np.random.default_rng(100 + k), lo=0, hi=SCORED_FROM,
                     kinds=("transfer",)) for k in range(N_TUNE_WORLDS)]
    out = []
    for on, stay in itertools.product(TUNE_ON, TUNE_STAY):
        p = Params(on_cost=on, off_cost=on, stay_cost=stay)
        ev, _ = decode_world(panel=panel, residual=tune_res, frozen=fz, arm="masked", params=p)
        rate = len(ev) / (len(panel.sites) * n_years)
        recalls = []
        for w, truth in worlds:
            e, _ = decode_world(panel=panel, residual=w, frozen=fz, arm="masked", params=p)
            rows = big(pl.DataFrame(match(truth=truth, events=e)).filter(
                pl.col("kind").is_in(["edge", "solo", "permanent_transfer", "permanent_shift"])))
            recalls.append(mw_recall(rows, "found"))
        out.append({"on": on, "stay": stay, "real_rate": rate, "recall": float(np.mean(recalls))})
        print(out[-1], flush=True)
    table = pl.DataFrame(out)
    WORK.mkdir(parents=True, exist_ok=True)
    table.write_parquet(WORK / "tuning.parquet")
    ok = table.filter(pl.col("real_rate") <= MAX_REAL_RATE)
    pick = (ok if ok.height else table).sort("recall", descending=True).row(0, named=True)
    params = Params(on_cost=pick["on"], off_cost=pick["on"], stay_cost=pick["stay"])
    (WORK / "params.json").write_text(json.dumps(asdict(params), indent=1))
    print("picked", params)
    return params


def load_params() -> Params:
    """The tuned penalties."""
    return Params(**json.loads((WORK / "params.json").read_text()))


def rewire(*, edges: list[tuple[str, str]], rng: np.random.Generator) -> dict[str, list[str]]:
    """Degree-preserving random rewiring of an edge list (simple graph), as neighbour lists."""
    e = [tuple(x) for x in edges]
    present = {frozenset(x) for x in e}
    for _ in range(10 * len(e)):
        i, j = rng.choice(len(e), size=2, replace=False)
        (a, b), (c, d) = e[i], e[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or frozenset((a, d)) in present or frozenset((c, b)) in present:
            continue
        present -= {frozenset((a, b)), frozenset((c, d))}
        present |= {frozenset((a, d)), frozenset((c, b))}
        e[i], e[j] = (a, d), (c, b)
    nb: dict[str, list[str]] = {}
    for a, b in e:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    return nb


def run_inject() -> None:
    """Score injected worlds in the scored years for every slow-level arm."""
    panel, res, fz = setup()
    p = load_params()
    specs: list[tuple[str, GenParams, tuple[str, ...], int]] = [
        ("default", GenParams(), ("transfer", "negatives"), N_WORLDS),
        ("constant", GenParams(shape_prop=0.0), ("transfer", "negatives"), 2),
        ("proportional", GenParams(shape_prop=1.0), ("transfer", "negatives"), 2),
        ("short", GenParams(duration_mix="short"), ("transfer", "negatives"), 2),
        ("long", GenParams(duration_mix="long"), ("transfer", "negatives"), 2),
        ("stress", GenParams(share=(0.2, 1.8), start_jitter_hh=24), ("transfer", "negatives"), 2),
        ("pv_growth", GenParams(), ("pv_growth",), 2),
        ("seasonal_bias", GenParams(), ("seasonal_bias",), 2),
    ]
    base_ev = {arm: decode_world(panel=panel, residual=res, frozen=fz, arm=arm, params=p)
               for arm in ARMS}
    rows_all, abs_all, fp_all = [], [], []
    for name, gen, kinds, n in specs:
        for k in range(n):
            world, truth = inject(panel=panel, residual=res, sigma=fz.sigma,
                                  rng=np.random.default_rng(1000 + 17 * k + len(name)),
                                  lo=SCORED_FROM, hi=GRID_LENGTH, gen=gen, kinds=kinds)
            for arm in ARMS:
                ev, slows = decode_world(panel=panel, residual=world, frozen=fz, arm=arm, params=p)
                scored = match(truth=truth, events=ev)
                ab = absorption(truth=truth, slow_world=slows, slow_base=base_ev[arm][1])
                for r, a in zip(scored, ab, strict=True):
                    rows_all.append({**r, "spec": name, "world": k, "arm": arm, "absorption": a})
                if name in ("pv_growth", "seasonal_bias"):
                    base_keys = {(e.primary, e.start // DAY) for e in base_ev[arm][0]}
                    new = [e for e in ev if (e.primary, e.start // DAY) not in base_keys
                           and e.start >= SCORED_FROM]
                    fp_all.append({"spec": name, "world": k, "arm": arm, "new_events": len(new)})
            print(name, k, flush=True)
    pl.DataFrame(rows_all).write_parquet(WORK / "injected_truth_scored.parquet")
    pl.DataFrame(fp_all).write_parquet(WORK / "growth_bias_false_events.parquet")


def run_real() -> None:
    """Decode the real record for each slow-level arm and the rewired-graph placebo."""
    panel, res, fz = setup()
    p = load_params()
    for arm in ARMS:
        ev, slows = decode_world(panel=panel, residual=res, frozen=fz, arm=arm, params=p)
        events_frame(events=ev).write_parquet(WORK / f"real_events_{arm}.parquet")
        np.savez(WORK / f"real_slow_{arm}.npz", **slows)
        if arm == "masked":
            rng = np.random.default_rng(7)
            counts = []
            for _ in range(100):
                nb = rewire(edges=panel.edges, rng=rng)
                e2, _ = decode_world(panel=panel, residual=res, frozen=fz, arm=arm, params=p,
                                     neighbours={s: nb.get(s, []) for s in panel.sites},
                                     refit_slow=False)
                counts.append(sum(e.cls == "edge_transfer" and e.start >= SCORED_FROM for e in e2))
            real = sum(e.cls == "edge_transfer" and e.start >= SCORED_FROM for e in ev)
            (WORK / "rewired.json").write_text(json.dumps({"real": real, "rewired": counts}))
            print("edge events real", real, "rewired median", np.median(counts))
    (WORK / "noise_units.json").write_text(json.dumps({"sigma_block_mw": fz.sigma,
                                                       "sigma_3day_mw": fz.sigma3}, indent=1))


if __name__ == "__main__":
    stage = sys.argv[1]
    {"tune": tune, "inject": run_inject, "real": run_real}[stage]()
    assert replace(Params(), on_cost=1).on_cost == 1
