"""Select the decoder on validation worlds and score one trained network on the scored worlds.

    uv run --no-sync python e00676_run.py <arm: G|G-R|G-0> <seed>   (PYTHONPATH as `e00676_train.py`)

The onset cost and link cut-off are chosen on the 20 validation worlds (events from 2023-07-01) by
the highest pooled paired share in the long and deep and the medium and deep cells, subject to 3 to
9 decoded edge events per Primary-year on the real record in 2023-07 to 2023-12; the grid is
widened once if no value meets the band. The chosen decoder is then applied once to the 40 scored
worlds (seeds 900 to 939, hours from 2024-01-01). Output: `scored/<arm>_seed<k>_rows.parquet`
(same columns as experiment 678's rows), `scored/<arm>_seed<k>_decoder.json`.
"""

# ruff: noqa: C409, C416, E501, E702

from __future__ import annotations

import itertools
import json
import sys
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00653_common import GRID_LENGTH, SCORED_FROM
from e00653_core import match
from e00653_inject import Truth
from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]
from e00665_score import cell_of  # ty: ignore[unresolved-import]
from e00672_data import TRAIN_END
from e00675_pretrain import ROOT as ROOT675
from e00675_pretrain import hourly_residual
from e00676_decode import decode, predict_world
from e00676_net import EdgeStateNet
from e00676_train import DEV, graph_of
from e00676_worlds import ROOT, VAL_SEEDS, WORLDS

GRID: Final = (tuple([2.0, 4.0, 8.0, 16.0]), tuple([0.15, 0.3, 0.5, 0.7]))
WIDER: Final = (tuple([2.0, 4.0, 8.0, 16.0, 32.0, 64.0]), tuple([0.15, 0.3, 0.5, 0.7, 0.85, 0.95]))
BAND: Final = (3.0, 9.0)
SCORED_H: Final[int] = SCORED_FROM // 2
TRAIN_H: Final[int] = TRAIN_END // 2
CELLS: Final = ("long, deep", "medium, deep")


def truths_of(df: pl.DataFrame) -> list[Truth]:
    """Truth objects from a cached truth table (partner amplitudes from the recipient rows)."""
    rows = df.to_dicts()
    out = []
    for r in rows:
        partners = [p for p in r["partners"].split(",") if p]
        pa = {}
        if r["amp_mw"] < 0:
            for p in partners:
                rec = next(
                    (
                        q
                        for q in rows
                        if q["group"] == r["group"] and q["site"] == p and q["amp_mw"] > 0
                    ),
                    None,
                )
                pa[p] = float(rec["amp_mw"]) if rec else 0.0
        out.append(Truth(site=r["site"], start=int(r["start"]), end=None if r["end"] < 0 else int(r["end"]), amp_mw=float(r["amp_mw"]), units=float(r["units"]), kind=r["kind"], group=int(r["group"]), partners=partners, partner_amp=pa))  # fmt: skip
    return out


def xh_of(*, resid: np.ndarray, sigma: np.ndarray, sites: list[str]) -> dict[str, np.ndarray]:
    """Half-hour residual in megawatts per Primary (each hour repeated twice)."""
    return {s: np.repeat(resid[i].astype(np.float64) * sigma[i], 2) for i, s in enumerate(sites)}


def paired_rows(
    *, truth: list[Truth], events: list, world: int, method: str, world_end: int
) -> list[dict]:
    """Scored truth rows with the world's decoded-transfer counts (as experiment 678)."""
    scored = match(truth=truth, events=events, world_end=world_end)
    n_edge = sum(e.cls.endswith("transfer") and e.start >= (TRAIN_END if world_end == SCORED_FROM else SCORED_FROM) for e in events)  # fmt: skip
    n_right = sum(bool(r["right_class"]) for r in scored)
    return [{**r, "world": world, "method": method, "world_edge_events": n_edge, "world_right_pairs": n_right, "lag_hh_err": np.nan} for r in scored]  # fmt: skip


def pooled_share(rows: list[dict]) -> float:
    """MW-weighted paired share over the long-and-deep and medium-and-deep edge ends."""
    num = den = 0.0
    for r in rows:
        if not r["edge_truth"] or r["units"] < 2.0:
            continue
        days = None if r["end"] < 0 else r["days"]
        if cell_of(days=days, units=r["units"]) in CELLS:
            w = abs(r["amp_mw"]); den += w; num += w * float(r["right_class"])  # fmt: skip
    return num / den if den else 0.0


def main() -> None:
    """Select the decoder on validation worlds, then score the scored worlds once."""
    arm, seed = sys.argv[1], int(sys.argv[2])
    rig = Rig()
    sites, mva, edges, adj = graph_of(rig=rig, arm=arm)
    net = EdgeStateNet(edges=edges, no_links=arm == "G-0").to(DEV)
    net.load_state_dict(torch.load(ROOT / "weights" / f"{arm}_seed{seed}.pt", map_location=DEV))
    kw = {"net": net, "mva": mva, "adj": adj, "device": DEV}
    # real record, 2023-07 to 2023-12
    x_all, sig_t, _, _ = hourly_residual(rig=rig)
    real = predict_world(resid=x_all.numpy(), sigma=sig_t.numpy(), hours=SCORED_H, **kw)
    sigma_mw = {s: float(sig_t[i]) for i, s in enumerate(sites)}
    # validation posteriors
    val = []
    for w in VAL_SEEDS:
        z = np.load(WORLDS / f"val_{w}.npz")
        post = predict_world(resid=z["resid"], sigma=z["sigma"], hours=SCORED_H, **kw)
        truth = truths_of(pl.read_parquet(WORLDS / f"val_{w}_truth.parquet"))
        val.append((w, post, xh_of(resid=z["resid"][:, :SCORED_H], sigma=z["sigma"], sites=sites), truth, dict(zip(sites, z["sigma"], strict=True))))  # fmt: skip
    xh_real = xh_of(resid=x_all.numpy()[:, :SCORED_H], sigma=sig_t.numpy(), sites=sites)
    assert SCORED_FROM % 2 == 0
    years = 15 * (SCORED_H - TRAIN_H) / (24 * 365.25)

    def evaluate(onset: float, cutoff: float) -> tuple[float, float]:
        rows = []
        for w, post, xh, truth, sg in val:
            ev = decode(post=post, edges=edges, sites=sites, xh=xh, sigma=sg, onset=onset, cutoff=cutoff, lo=TRAIN_H, hi=SCORED_H)  # fmt: skip
            rows += paired_rows(truth=truth, events=ev, world=w, method="v", world_end=SCORED_H * 2)
        ev_real = decode(post=real, edges=edges, sites=sites, xh=xh_real, sigma=sigma_mw, onset=onset, cutoff=cutoff, lo=TRAIN_H, hi=SCORED_H)  # fmt: skip
        rate = sum(e.amp_mw < 0 and e.cls.endswith("transfer") for e in ev_real) / years
        return pooled_share(rows), rate

    results, widened = {}, False
    for grid in (GRID, WIDER):
        for onset, cutoff in itertools.product(*grid):
            if (onset, cutoff) not in results:
                results[(onset, cutoff)] = evaluate(onset, cutoff)
        ok = {k: v for k, v in results.items() if BAND[0] <= v[1] <= BAND[1]}
        if ok:
            break
        widened = True
    pool = ok or {k: v for k, v in results.items()}
    key = max(pool, key=lambda k: pool[k][0]) if ok else min(results, key=lambda k: min(abs(results[k][1] - BAND[0]), abs(results[k][1] - BAND[1])))  # fmt: skip
    onset, cutoff = key
    print("chosen", key, results[key], "widened", widened, "in band", bool(ok), flush=True)
    out = ROOT / "scored"
    out.mkdir(exist_ok=True)
    (out / f"{arm}_seed{seed}_decoder.json").write_text(json.dumps({"onset": onset, "cutoff": cutoff, "val_paired": results[key][0], "real_rate": results[key][1], "widened": widened, "in_band": bool(ok), "grid": {f"{k[0]}/{k[1]}": v for k, v in results.items()}}, indent=1))  # fmt: skip
    rows = []
    for w in range(900, 940):
        z = np.load(ROOT675 / "worlds" / f"test_{w}.npz")
        post = predict_world(resid=z["resid"], sigma=z["sigma"], **kw)
        truth = truths_of(pl.read_parquet(ROOT675 / "worlds" / f"test_{w}_truth.parquet"))
        sg = dict(zip(sites, z["sigma"], strict=True))
        ev = decode(post=post, edges=edges, sites=sites, xh=xh_of(resid=z["resid"], sigma=z["sigma"], sites=sites), sigma=sg, onset=onset, cutoff=cutoff, lo=SCORED_H, hi=GRID_LENGTH // 2)  # fmt: skip
        rows += paired_rows(
            truth=truth, events=ev, world=w, method=f"{arm}/{seed}", world_end=GRID_LENGTH
        )
        print(w, len(rows), flush=True)
    df = pl.DataFrame(rows).with_columns(cell=pl.Series([cell_of(days=None if r["end"] < 0 else r["days"], units=r["units"]) if r["units"] >= 2.0 else "under 2" for r in rows]))  # fmt: skip
    df.write_parquet(out / f"{arm}_seed{seed}_rows.parquet")


if __name__ == "__main__":
    main()
