# ruff: noqa: E501
"""Experiment 673: re-probe D12 (and D19, as a check) from experiment 662's stored 2026-06 twin network.

Experiment 671 stored the S(u) data of one probe cell for D2, D3, D4, D19, D21, D5, and D10, and no
stored output holds D12's. This script runs the same probe code on experiment 662's stored
uninjected network (condition `pf_twin`, seed 0, cutoff 2026-06), for D12 and D19, and writes the
probe cell nearest 12:00 UTC in July with the linear and fold fits. It trains no network.

Usage: ml-experiments/00671*/run.sh ../00673*/e00673_d12_probe.py (CPU, one thread)
"""

import json
from pathlib import Path
from typing import Final

import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import e00662_common as k  # ty: ignore[unresolved-import]
import e00662_probe as pr  # ty: ignore[unresolved-import]
import e00671_part2 as p2  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00673-pv-capacity-for-nged-per-primary-with-register-cross-check"
    / "work"
)
SITES: Final[tuple[str, ...]] = ("D12", "D19")
COND: Final = "pf_twin"
SEED: Final = 0


def main() -> None:
    """Probe the stored network and write one JSON of S(u) cell data per site."""
    torch.set_num_threads(1)
    WORK.mkdir(parents=True, exist_ok=True)
    frame = pr.load_frame(cutoff="2026-06")
    cap = k621.effective_capacity_mw()
    idx = k.index_of(sites=SITES)
    model = m.ModularMdn(n_primaries=16, stage="pre")
    model.load_state_dict(
        torch.load(k.weights_path(cutoff="2026-06", condition=COND, seed=SEED), map_location="cpu")
    )
    model.eval()
    out = {}
    for site, i in idx.items():
        pp = pr.PrimaryProbe(frame=frame, primary=i, site=site, cap=float(cap[i]))
        pp.set_arm(arm=k.parse(name=COND).arm)
        y = pp.forward(model=model, frame=frame)
        y_scale = float(
            np.median(
                [np.std(y[pp.cell == c]) for c in range(pp.n_cells) if (pp.cell == c).sum() > 3]
            )
        )
        sd = np.maximum(0.25 * pp.mu, 1e-3)
        l0 = pr.fit_linear(y=y, cell=pp.cell, regs=[pp.u_std, pp.temp], w=np.ones(len(y)))
        res = {"m_mw": pp.m_mw, "L0": l0}
        fits = {}
        for name, qf in p2.VARIANTS.items():
            fit = p2.multistart(pp=pp, y=y, mu=pp.mu, sd=sd, y_scale=y_scale, q_fixed=qf)
            fits[name] = fit
            res[f"FA_{name}"] = fit["c_total"]
        c = p2.pick_cell(pp=pp)
        sel = pp.cell == c
        u, yy = pp.u_std[sel], y[sel]
        curve = {
            "u": u.tolist(),
            "y": yy.tolist(),
            "cell_hh": int(pp.cell_hh[c]),
            "cell_month": int(pp.cell_month[c]),
            "a_lin": float(np.mean(yy + l0 * u)),
            "c_lin": l0,
            "u_basis": pp.u_basis[sel].tolist(),
            "dt": (pp.temp - pp.temp.mean())[sel].tolist(),
        }
        for name, fit in fits.items():
            curve[f"fold_{name}"] = {"a": float(fit["a"][c]), "theta": fit["theta"].tolist()}
        res["curve"] = curve
        out[site] = res
        print(site, {kk: round(v, 2) for kk, v in res.items() if isinstance(v, float)}, flush=True)
    (WORK / "d12_d19_probe_seed0.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
