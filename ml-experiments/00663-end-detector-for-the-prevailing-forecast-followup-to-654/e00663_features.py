# ruff: noqa: D103, E501, PLR0915, PLR0912
"""Experiment 663, step 1: the detector's inputs for every origin of every fold (CPU).

Writes `det_inputs_v2/<fold month>_<data month>.npz` with, for the data month's real origins and bank
origins (the row order of `head_data/real_*.npz` and `bank_*.npz`, one row per `org`): `real_*`
and `bank_*` matrices for the variants `g` (connected neighbours), `rw0` to `rw2` (rewired
graphs), and `neg_<variant>` (the same graph with hard negatives added to the residuals).

Usage: python e00663_features.py [--folds ...] [--data-months ...]
"""

import argparse
import time

import e00578_common as k578  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00663_common as c663
import e00663_panel as pan
import numpy as np

OUT = c663.WORK / "det_inputs_v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--folds",
        nargs="+",
        default=list(c663.MONTHS),
        help="a month, or `final` (statistics from all five months, plus the window)",
    )
    ap.add_argument("--data-months", nargs="+", default=list(c663.MONTHS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nb_of, _cap, _mw = c650.world()
    nb_of = {p: sorted(set(v)) for p, v in nb_of.items()}
    for p, v in list(nb_of.items()):
        for q in v:
            nb_of.setdefault(q, [])
            if p not in nb_of[q]:
                nb_of[q] = sorted([*nb_of[q], p])
    graphs = {"g": nb_of}
    kept = []
    for k in range(c663.N_REWIRINGS):
        graphs[f"rw{k}"] = pan.rewire(nb_of=nb_of, seed=k)
        kept.append(pan.rewire.kept)  # ty: ignore[unresolved-attribute]
    print("edges kept by each rewiring:", kept, flush=True)
    panels = {m: pan.load_month_panel(month=m) for m in c663.MONTHS}
    clean = k578.clean()
    for fold in a.folds:
        stats = pan.fit_fold(panels=panels, train_months=[m for m in c663.MONTHS if m != fold])
        if fold == "final":
            wp = pan.load_window_panel()
            zw = pan.world_z(panel=wp, fold=stats, own=None)
            real = np.load(c663.WORK / "window" / "real_window.npz")
            res = {
                f"real_{name}": pan.features(
                    z=zw,
                    panel=wp,
                    origins=real["org_oi"].astype(np.int64),
                    prims=real["org_prim"].astype(int),
                    nb_of=g,
                )
                for name, g in graphs.items()
            }
            np.savez(OUT / "final_window.npz", **res)
            print("final_window.npz written", flush=True)
        for m in a.data_months:
            out = OUT / f"{fold}_{m}.npz"
            if out.exists():
                continue
            t0 = time.time()
            panel = panels[m]
            res: dict[str, np.ndarray] = {}
            real = np.load(c663.E647 / "head_data" / f"real_{m}.npz")
            bank = np.load(c650.WORK / "head_data" / f"bank_{m}.npz")
            negs, negc = pan.hard_negatives(
                shape=panel.obs.shape, nb_of=nb_of, seed=663 + c663.MONTHS.index(m)
            )
            z_real = pan.world_z(panel=panel, fold=stats, own=None)
            z_real_neg = pan.world_z(
                panel=panel, fold=stats, own=None, extra=negs, extra_common=negc[None, :]
            )
            po, oo = real["org_prim"].astype(int), real["org_oi"].astype(np.int64)
            for name, g in graphs.items():
                res[f"real_{name}"] = pan.features(
                    z=z_real, panel=panel, origins=oo, prims=po, nb_of=g
                )
                res[f"real_neg_{name}"] = pan.features(
                    z=z_real_neg, panel=panel, origins=oo, prims=po, nb_of=g
                )
            bp, bo, bq = (
                bank["org_prim"].astype(int),
                bank["org_oi"].astype(np.int64),
                bank["org_pass"].astype(int),
            )
            for name in graphs:
                for pre in ("bank_", "bank_neg_"):
                    res[f"{pre}{name}"] = np.zeros((len(bp), pan.N_FEATURES), np.float32)
            k1s = pan.bank_k1(month=m, panel=panel)
            for r in np.unique(bq):
                own, mirror, _ev = c650.shifts_for(clean=clean, month=m, r=int(r))
                z_b = pan.world_z(panel=panel, fold=stats, own=own, mirror=mirror, k1=k1s[int(r)])
                z_bn = pan.world_z(
                    panel=panel,
                    fold=stats,
                    own=own,
                    mirror=mirror,
                    k1=k1s[int(r)],
                    extra=negs,
                    extra_common=negc[None, :],
                )
                sel = np.flatnonzero(bq == r)
                for name, g in graphs.items():
                    res[f"bank_{name}"][sel] = pan.features(
                        z=z_b, panel=panel, origins=bo[sel], prims=bp[sel], nb_of=g
                    )
                    res[f"bank_neg_{name}"][sel] = pan.features(
                        z=z_bn, panel=panel, origins=bo[sel], prims=bp[sel], nb_of=g
                    )
            np.savez(out.with_suffix(".tmp.npz"), **res, kept=np.array(kept))
            out.with_suffix(".tmp.npz").rename(out)
            print(f"{out.name}: {time.time() - t0:.0f}s", flush=True)


main()
