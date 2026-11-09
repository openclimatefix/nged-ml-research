# ruff: noqa: E501, ANN001, ANN201, ANN202, RUF059, PLR0915
"""Experiment 671, Part 2: re-probe stored 2026-06 networks of experiment 662 with the reactive terms fixed.

For each Primary and seed it probes the stored network, then fits: L0 (linear), FA free (experiment
662's fold model with a night prior), FA-Z (q0 = q1 = 0 fixed), and FA-N (q0 = 0, q1 = 0.25 fixed,
the power-factor 0.97 share of the native worlds). Seed 0 also stores the S(u) data of one probe
cell per Primary with each fit's curve, for the S(u) figure.

Usage: python e00671_part2.py --seeds 0 1 --out-suffix a
"""

import argparse
import json
import sys

import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import e00662_common as k  # ty: ignore[unresolved-import]
import e00662_probe as pr  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00671_a1 import OUT

Q1_N = 0.25
NATIVE_AND_INJECTED = ("D2", "D3", "D4", "D19", "D21", "D5", "D10")
VARIANTS = {"free": None, "Z": (0.0, 0.0), "N": (0.0, Q1_N)}


def fold_fit_q(*, y, cell, u, dt, n_cells, w, c0, mu, sd, y_scale, q_fixed):
    """Experiment 662's `fold_fit` with (q0, q1) held at `q_fixed` (None keeps them free)."""
    if q_fixed is None:
        return pr.fold_fit(
            y=y, cell=cell, u=u, dt=dt, n_cells=n_cells, w=w, c0=c0, mu=mu, sd=sd, y_scale=y_scale
        )
    n, kk = u.shape
    cnt = np.bincount(cell, weights=w, minlength=n_cells)
    ybar = np.bincount(cell, weights=w * y, minlength=n_cells) / np.maximum(cnt, 1e-9)
    ubar = np.stack(
        [
            np.bincount(cell, weights=w * u[:, i], minlength=n_cells) / np.maximum(cnt, 1e-9)
            for i in range(kk)
        ],
        1,
    )
    cvec = np.full(kk, c0 / kk)
    theta = np.concatenate([cvec, [0.0, q_fixed[0], q_fixed[1]]])
    a = np.maximum(ybar + ubar @ cvec, 1e-3)
    free = np.arange(kk + 1)  # C_k and b only
    lam = 1e-2

    def residual(theta, a):
        cv, b, q0, q1 = theta[:kk], theta[kk], theta[kk + 1], theta[kk + 2]
        mm = a[cell] - u @ cv + b * dt
        qq = q0 + q1 * a[cell]
        s = np.sqrt(mm**2 + qq**2) + 1e-9
        return s - y, mm, qq, s

    def cost(r, a):
        return float((w * r**2).sum()) + float((((a - mu) / sd) ** 2).sum()) * y_scale**2

    r, mm, qq, s = residual(theta, a)
    cur = cost(r, a)
    for _ in range(60):
        dm, dq = mm / s, qq / s
        ja = dm + dq * theta[kk + 2]
        jt = np.concatenate([-u * dm[:, None], (dt * dm)[:, None]], axis=1)
        haa = np.bincount(cell, weights=w * ja * ja, minlength=n_cells) + y_scale**2 / sd**2
        ga = (
            np.bincount(cell, weights=w * ja * r, minlength=n_cells) + y_scale**2 * (a - mu) / sd**2
        )
        hat = np.stack(
            [
                np.bincount(cell, weights=w * ja * jt[:, i], minlength=n_cells)
                for i in range(len(free))
            ],
            1,
        )
        wj = jt * w[:, None]
        htt = jt.T @ wj
        gt = wj.T @ r
        improved = False
        for _try in range(8):
            haa_d = haa * (1.0 + lam) + 1e-12
            htt_d = htt + lam * np.diag(np.diag(htt)) + 1e-12 * np.eye(len(free))
            schur = htt_d - (hat.T * (1.0 / haa_d)) @ hat
            rhs = gt - (hat.T * (1.0 / haa_d)) @ ga
            try:
                dth = -np.linalg.solve(schur, rhs)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            da = -(ga + hat @ dth) / haa_d
            new_theta = theta.copy()
            new_theta[free] += dth
            new_theta[:kk] = np.maximum(new_theta[:kk], 0.0)
            new_a = np.maximum(a + da, 1e-6)
            r2, mm2, qq2, s2 = residual(new_theta, new_a)
            c2 = cost(r2, new_a)
            if c2 < cur:
                rel = (cur - c2) / max(cur, 1e-12)
                theta, a, r, mm, qq, s, cur = new_theta, new_a, r2, mm2, qq2, s2, c2
                lam = max(lam / 5.0, 1e-7)
                improved = True
                break
            lam *= 10
        if not improved or rel < 1e-7:
            break
    return {"theta": theta, "a": a, "sse": cur, "c_total": float(theta[:kk].sum()), "c": theta[:kk]}


def multistart(*, pp, y, mu, sd, y_scale, q_fixed):
    """Best fit over experiment 662's five starting capacities."""
    dt = pp.temp - pp.temp.mean()
    best = None
    for mult in k.START_MULTIPLES:
        fit = fold_fit_q(
            y=y,
            cell=pp.cell,
            u=pp.u_basis,
            dt=dt,
            n_cells=pp.n_cells,
            w=np.ones(len(y)),
            c0=mult * pp.m_mw,
            mu=mu,
            sd=sd,
            y_scale=y_scale,
            q_fixed=q_fixed,
        )
        if best is None or fit["sse"] < best["sse"]:
            best = fit
    return best


def pick_cell(*, pp) -> int:
    """Return the probe cell nearest 12:00 UTC (half-hour index 24) in July, else the nearest in the summer."""
    score = np.abs(pp.cell_hh - 24) + 5 * np.abs(pp.cell_month - 7)
    return int(np.argmin(score))


def main() -> None:
    """Probe and refit, writing one JSON per (condition, seed)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", default=["pf_twin", "pf_fleet_l05_r0"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    frame = pr.load_frame(cutoff="2026-06")
    cap = k621.effective_capacity_mw()
    idx = k.index_of(sites=NATIVE_AND_INJECTED)
    probes = {
        s: pr.PrimaryProbe(frame=frame, primary=i, site=s, cap=float(cap[i]))
        for s, i in idx.items()
    }
    out_dir = OUT / "part2"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in args.conditions:
        for seed in args.seeds:
            path = out_dir / f"{cond}_s{seed}.json"
            if path.exists():
                continue
            model = m.ModularMdn(n_primaries=16, stage="pre")
            model.load_state_dict(
                torch.load(
                    k.weights_path(cutoff="2026-06", condition=cond, seed=seed), map_location="cpu"
                )
            )
            model.eval()
            rec = {}
            for site, pp in probes.items():
                pp.set_arm(arm=k.parse(name=cond).arm)
                y = pp.forward(model=model, frame=frame)
                y_scale = float(
                    np.median(
                        [
                            np.std(y[pp.cell == c])
                            for c in range(pp.n_cells)
                            if (pp.cell == c).sum() > 3
                        ]
                    )
                )
                sd = np.maximum(0.25 * pp.mu, 1e-3)
                res = {
                    "m_mw": pp.m_mw,
                    "L0": pr.fit_linear(
                        y=y, cell=pp.cell, regs=[pp.u_std, pp.temp], w=np.ones(len(y))
                    ),
                }
                fits = {}
                for name, qf in VARIANTS.items():
                    fit = multistart(pp=pp, y=y, mu=pp.mu, sd=sd, y_scale=y_scale, q_fixed=qf)
                    fits[name] = fit
                    res[f"FA_{name}"] = fit["c_total"]
                    res[f"FA_{name}_q"] = [float(fit["theta"][-2]), float(fit["theta"][-1])]
                if seed == 0:
                    c = pick_cell(pp=pp)
                    sel = pp.cell == c
                    u, yy = pp.u_std[sel], y[sel]
                    a_lin = float(np.mean(yy + res["L0"] * u))
                    res["curve"] = {
                        "u": u.tolist(),
                        "y": yy.tolist(),
                        "cell_hh": int(pp.cell_hh[c]),
                        "cell_month": int(pp.cell_month[c]),
                        "a_lin": a_lin,
                        "c_lin": res["L0"],
                        "u_basis": pp.u_basis[sel].tolist(),
                        "dt": (pp.temp - pp.temp.mean())[sel].tolist(),
                    }
                    for name, fit in fits.items():
                        res["curve"][f"fold_{name}"] = {
                            "a": float(fit["a"][c]),
                            "theta": fit["theta"].tolist(),
                        }
                rec[site] = res
                print(
                    cond,
                    seed,
                    site,
                    {
                        kk: (round(v, 2) if isinstance(v, float) else v)
                        for kk, v in res.items()
                        if kk in ("L0", "FA_free", "FA_Z", "FA_N")
                    },
                    flush=True,
                )
            path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
    sys.exit(0)
