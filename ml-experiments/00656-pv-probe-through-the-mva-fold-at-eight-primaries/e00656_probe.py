# ruff: noqa: E501, ANN001, ANN202, PLR0915, PLW0603, RUF059
"""Experiment 656: the donor-day probe of a pretrained network with a fold model (CPU).

The probe is experiment 621's: hold every datetime input and the Primary at one probe date and
half-hour, feed the network the weather of donor half-hours (the same half-hour of the day within 21
days of the probe date's day of the year), and rescale donor shortwave by the donor's clear-sky
index to the probe date's sun. The probed reading `y` (MW; apparent power for the MVA arms) is then
regressed on the donor weather. Five regressions share the probe frames:

- `L0`: experiment 621's linear regression of `y` on the standard unit curve `u` and donor temperature.
- `L1`: `L0` on donor rows whose probed `y` is at least max(0.2 m, min(2 q_hat, 0.5 m)), with m the median
  sunny-midday reading (the cap stops a poorly identified q_hat from removing every row).
- `F4`: the fold model `y = sqrt((a_c - sum_k C_k u_k + b dT)^2 + (q0 + q1 a_c)^2)` with four unit curves.
- `F1`: the fold model with the standard unit curve only.
- `FA`: `F4` with a Gaussian prior on each cell's `a_c` from the Primary's night reading and the daytime-to-night
  demand ratio measured at the six signed-MW Primaries.

Usage: python e00656_probe.py --cutoff 2026-06 --conditions pf_twin --seeds 0 1 2
[--boot-conditions ...]
"""

import argparse
import time

import e00397_models as base  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00621_common as k621  # ty: ignore[unresolved-import]
import e00621_probe as p621  # ty: ignore[unresolved-import]
import e00656_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

p621.MINUTES = tuple(range(5 * 60, 19 * 60 + 1, 30))
p621.MIN_COS_ZENITH = 0.0976
"""Every half-hour with the sun 5.6 degrees up or more (sin 5.6 degrees = 0.0976)."""
WIDE_STARTS = False
"""Post hoc: also start the fold fit at 6 and 8 times the midday reading (set by `--wide-starts`; the pre-registered grid stops at 3)."""
N_DONORS = 40
"""Donor rows kept per probe cell (a fixed random subsample, for speed)."""
N_BOOT = 40
PROBED: tuple[str, ...] = (*k.SIX, *k.NATIVE, *k.FOLDERS)
ARMS: tuple[str, ...] = ("L0", "L1", "F4", "F1", "FA")


def load_frame(*, cutoff: str) -> dict[str, np.ndarray]:
    """Load a cutoff's probe frame."""
    with np.load(k.frame_path(cutoff=cutoff)) as z:
        return {name: z[name] for name in z.files}


class PrimaryProbe:
    """The probe inputs of one Primary: cells, donors, donor weather, and the unit curves."""

    def __init__(
        self, *, frame: dict[str, np.ndarray], primary: int, site: str, cap: float
    ) -> None:
        """Build the cells and donors, subsample donors per cell, and compute every regressor."""
        self.primary, self.site, self.cap = primary, site, cap
        raw = p621.cells_and_donors(frame=frame, primary=primary)
        rng = np.random.default_rng(656 + primary)
        keep = []
        for c in np.unique(raw["cell"]):
            rows = np.nonzero(raw["cell"] == c)[0]
            keep.append(rng.choice(rows, size=min(N_DONORS, len(rows)), replace=False))
        keep = np.sort(np.concatenate(keep))
        self.probe = {
            **raw,
            "base": raw["base"][keep],
            "donor": raw["donor"][keep],
            "cell": raw["cell"][keep],
        }
        _, self.cell = np.unique(self.probe["cell"], return_inverse=True)
        self.n_cells = int(self.cell.max()) + 1
        x, b, d = self.probe["x"], self.probe["base"], self.probe["donor"]
        s_mean, s_std = frame["std_mean"], frame["std_std"]
        ghi_d = np.clip(x[d, 0].astype(np.float64) * s_std[0] + s_mean[0], 0.0, None)
        self.temp = x[d, 2].astype(np.float64) * s_std[2] + s_mean[2]
        clear_d = k621.haurwitz_w_m2(cos_zenith=x[d, 14].astype(np.float64))
        clear_b = k621.haurwitz_w_m2(cos_zenith=x[b, 14].astype(np.float64))
        kt = np.clip(ghi_d / np.maximum(clear_d, 1.0), 0.0, p621.KT_MAX)
        self.ghi_in = kt * clear_b
        self.std_mean = s_mean
        self.std_std = s_std
        t_b = self.probe["t_us"][b]
        self.u_basis = k.basis_curves(ghi=self.ghi_in, x=x[b], times_us=t_b, temp=self.temp)
        lit = self.u_basis > 0.01
        self.basis_scale = np.array(
            [np.percentile(self.u_basis[lit[:, i], i], 99) for i in range(self.u_basis.shape[1])]
        )
        self.u_basis = self.u_basis / self.basis_scale
        self.u_std = k621.unit_curve(ghi=self.ghi_in, x=x[b], times_us=t_b, temp=self.temp)
        first = np.array([np.nonzero(self.cell == c)[0][0] for c in range(self.n_cells)])
        t_cell = t_b[first]
        self.cell_month = pl.Series(t_cell).cast(pl.Datetime("us")).dt.month().to_numpy()
        self.cell_hh = ((t_cell % k621.DAY_US) // 1_800_000_000).astype(int)
        donor_day = self.probe["days"][d]
        _, self.date_id = np.unique(donor_day, return_inverse=True)
        self.n_dates = int(self.date_id.max()) + 1
        # The Primary's median sunny-midday reading (MW), from the frame's real target.
        sel = frame["primary"] == primary
        xs = frame["x"][sel]
        ghi = np.clip(xs[:, 0].astype(np.float64) * s_std[0] + s_mean[0], 0.0, None)
        kt_all = ghi / np.maximum(k621.haurwitz_w_m2(cos_zenith=xs[:, 14].astype(np.float64)), 1.0)
        t = k621.time_us(target_idx=frame["target_idx"][sel])
        hour = (t // 3_600_000_000) % 24
        months = pl.Series(t).cast(pl.Datetime("us")).dt.month().to_numpy()
        ok = (kt_all >= k.SUNNY_KT) & (hour >= 10) & (hour <= 15) & (months >= 4) & (months <= 8)
        self.m_mw = float(np.median(frame["y"][sel][ok])) * cap
        night = float(frame["night_pts"][primary]) * cap
        order = np.argsort(frame["target_idx"][sel])
        y_sorted = frame["y"][sel][order].astype(np.float64) * cap
        first_row = np.array([np.nonzero(self.cell == c)[0][0] for c in range(self.n_cells)])
        self.cell_p_mw = y_sorted[self.probe["base"][first_row]]
        self.cell_umax = np.zeros(self.n_cells)
        np.maximum.at(self.cell_umax, self.cell, self.u_std)
        self.night_raw = night
        self.rho_cells = frame["rho"][self.cell_month, self.cell_hh]
        self.set_arm(arm="pf")

    def set_arm(self, *, arm: str) -> None:
        """Set the anchor's night reading for a reactive-power arm.

        The six Primaries read sqrt(P^2 + Q^2) in MVA worlds.
        """
        inflate = {
            "pf": 1.031,
            "floor": float(np.sqrt(1.0 + (0.10 * self.m_mw / max(self.night_raw, 1e-9)) ** 2)),
            "q0": 1.0,
            "sig": 1.0,
        }[arm]
        self.night_mw = self.night_raw * (inflate if self.site in k.SIX else 1.0)
        self.mu = self.rho_cells * self.night_mw

    def forward(self, *, model: m.ModularMdn, frame: dict[str, np.ndarray]) -> np.ndarray:
        """Return the probed reading (MW) of every probe row."""
        x, b, d = self.probe["x"], self.probe["base"], self.probe["donor"]
        inp = x[d, :35].copy()
        inp[:, 14:35] = x[b, 14:35]
        inp[:, 0] = np.clip((self.ghi_in - self.std_mean[0]) / self.std_std[0], -5.0, 5.0).astype(
            np.float32
        )
        with torch.no_grad():
            xt = torch.cat([torch.from_numpy(inp), torch.zeros(len(inp), 2)], dim=1)
            pt = torch.full((len(xt),), self.primary, dtype=torch.long)
            mean_z = []
            for i in range(0, len(xt), 65_536):
                out = model(xt[i : i + 65_536], pt[i : i + 65_536]).double()
                log_w, mean, _ = base.mixture_parts(out=out, components=m.COMPONENTS)
                mean_z.append((torch.softmax(log_w, dim=-1) * mean).sum(-1).numpy())
        z = np.concatenate(mean_z)
        return (
            z * frame["target_std"][self.primary] + frame["target_mean"][self.primary]
        ) * self.cap


# --- the regressions ------------------------------------------------------------------------


def fit_linear(*, y: np.ndarray, cell: np.ndarray, regs: list[np.ndarray], w: np.ndarray) -> float:
    """Return minus the slope on the first regressor, after weighted demeaning within each cell."""
    return p621.fit(y=y, cell=cell, regressors=regs, w=w)


def fold_fit(
    *, y, cell, u, dt, n_cells, w, c0, mu=None, sd=None, start=None, iters=60, y_scale=1.0,
) -> dict:  # fmt: skip
    """Fit the fold model by Levenberg-Marquardt with a Schur complement over the per-cell levels.

    `u` has shape (rows, K). Parameters: C_k >= 0 (K), b, q0 >= 0, q1 >= 0, and a_c >= 0 per cell.
    With `mu` and `sd` set, each a_c has a Gaussian prior (scaled by `y_scale`).
    """
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
    if start is None:
        cvec = np.full(kk, c0 / kk)
        theta = np.concatenate([cvec, [0.0, 0.05, 0.0]])
        a = np.maximum(ybar + ubar @ cvec, 1e-3)
    else:
        theta, a = start[0].copy(), start[1].copy()
    npar = kk + 3
    lam = 1e-2

    def residual(theta, a):
        cvec, b, q0, q1 = theta[:kk], theta[kk], theta[kk + 1], theta[kk + 2]
        mm = a[cell] - u @ cvec + b * dt
        qq = q0 + q1 * a[cell]
        s = np.sqrt(mm**2 + qq**2) + 1e-9
        return s - y, mm, qq, s

    def cost(r, a):
        c = float((w * r**2).sum())
        if mu is not None:
            c += float((((a - mu) / sd) ** 2).sum()) * y_scale**2
        return c

    r, mm, qq, s = residual(theta, a)
    cur = cost(r, a)
    for _ in range(iters):
        dm, dq = mm / s, qq / s
        ja = dm + dq * theta[kk + 2]
        jt = np.concatenate(
            [-u * dm[:, None], (dt * dm)[:, None], dq[:, None], (dq * a[cell])[:, None]], axis=1
        )
        haa = np.bincount(cell, weights=w * ja * ja, minlength=n_cells)
        ga = np.bincount(cell, weights=w * ja * r, minlength=n_cells)
        if mu is not None:
            haa = haa + y_scale**2 / sd**2  # ty: ignore[unsupported-operator]
            ga = ga + y_scale**2 * (a - mu) / sd**2  # ty: ignore[unsupported-operator]
        hat = np.stack(
            [np.bincount(cell, weights=w * ja * jt[:, i], minlength=n_cells) for i in range(npar)],
            1,
        )
        wj = jt * w[:, None]
        htt = jt.T @ wj
        gt = wj.T @ r
        improved = False
        for _try in range(8):
            haa_d = haa * (1.0 + lam) + 1e-12
            htt_d = htt + lam * np.diag(np.diag(htt)) + 1e-12 * np.eye(npar)
            schur = htt_d - (hat.T * (1.0 / haa_d)) @ hat
            rhs = gt - (hat.T * (1.0 / haa_d)) @ ga
            try:
                dth = -np.linalg.solve(schur, rhs)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            da = -(ga + hat @ dth) / haa_d
            new_theta = theta + dth
            new_theta[:kk] = np.maximum(new_theta[:kk], 0.0)
            new_theta[kk + 1 :] = np.maximum(new_theta[kk + 1 :], 0.0)
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


def fold_multistart(
    *, y, cell, u, dt, n_cells, w, m_mw, mu=None, sd=None, y_scale=1.0, starts=None
) -> dict:
    """Return the best fold fit over the starting capacities.

    The starts are 0.25, 1, and 3 times the midday reading, and 6 and 8 too when `WIDE_STARTS`.
    """
    best = None
    for mult in (0.25, 1.0, 3.0, 6.0, 8.0) if WIDE_STARTS else (0.25, 1.0, 3.0):
        fit = fold_fit(
            y=y,
            cell=cell,
            u=u,
            dt=dt,
            n_cells=n_cells,
            w=w,
            c0=mult * m_mw,
            mu=mu,
            sd=sd,
            y_scale=y_scale,
        )
        if best is None or fit["sse"] < best["sse"]:
            best = fit
    return best  # ty: ignore[invalid-return-type]


def curve_class(*, y: np.ndarray, cell: np.ndarray, u: np.ndarray) -> tuple[str, float, float]:
    """Return the class of S(u) (falling, rising, V, or mixed).

    Also returns the median slopes in the lower and upper half of u.
    """
    lo, hi = [], []
    for c in np.unique(cell):
        sel = cell == c
        if sel.sum() < 6:
            continue
        uu, yy = u[sel], y[sel]
        med = np.median(uu)
        for part, out in ((uu <= med, lo), (uu > med, hi)):
            if part.sum() >= 3 and np.ptp(uu[part]) > 1e-3:
                out.append(np.polyfit(uu[part], yy[part], 1)[0])
    s_lo, s_hi = float(np.median(lo)), float(np.median(hi))
    if s_lo < 0 and s_hi < 0:
        cls = "falling"
    elif s_lo > 0 and s_hi > 0:
        cls = "rising"
    elif s_lo < 0 < s_hi:
        cls = "V"
    else:
        cls = "mixed"
    return cls, s_lo, s_hi


def probe_network(*, pp: PrimaryProbe, y: np.ndarray, boot: bool, rng: np.random.Generator) -> dict:
    """Return every arm's capacity (MW), its bootstrap draws if asked, and the fold model's extra read-outs."""
    cell, kk = pp.cell, pp.u_basis.shape[1]
    one = np.ones(len(y))
    dt = pp.temp - pp.temp.mean()
    y_scale = float(
        np.median([np.std(y[cell == c]) for c in range(pp.n_cells) if (cell == c).sum() > 3])
    )
    u1 = pp.u_std[:, None]
    sd = np.maximum(0.25 * pp.mu, 1e-3)
    out: dict = {}

    def run(*, w: np.ndarray, warm: dict | None = None) -> dict:
        res = {}
        res["L0"] = fit_linear(y=y, cell=cell, regs=[pp.u_std, pp.temp], w=w)
        res["F4"] = (
            fold_multistart(
                y=y, cell=cell, u=pp.u_basis, dt=dt, n_cells=pp.n_cells, w=w, m_mw=pp.m_mw
            )
            if warm is None
            else fold_fit(
                y=y,
                cell=cell,
                u=pp.u_basis,
                dt=dt,
                n_cells=pp.n_cells,
                w=w,
                c0=0.0,
                start=(warm["F4"]["theta"], warm["F4"]["a"]),
                iters=12,
            )
        )
        res["F1"] = (
            fold_multistart(y=y, cell=cell, u=u1, dt=dt, n_cells=pp.n_cells, w=w, m_mw=pp.m_mw)
            if warm is None
            else fold_fit(
                y=y,
                cell=cell,
                u=u1,
                dt=dt,
                n_cells=pp.n_cells,
                w=w,
                c0=0.0,
                start=(warm["F1"]["theta"], warm["F1"]["a"]),
                iters=12,
            )
        )
        res["FA"] = (
            fold_multistart(
                y=y,
                cell=cell,
                u=pp.u_basis,
                dt=dt,
                n_cells=pp.n_cells,
                w=w,
                m_mw=pp.m_mw,
                mu=pp.mu,
                sd=sd,
                y_scale=y_scale,
            )
            if warm is None
            else fold_fit(
                y=y,
                cell=cell,
                u=pp.u_basis,
                dt=dt,
                n_cells=pp.n_cells,
                w=w,
                c0=0.0,
                mu=pp.mu,
                sd=sd,
                y_scale=y_scale,
                start=(warm["FA"]["theta"], warm["FA"]["a"]),
                iters=12,
            )
        )
        q_hat = float(
            res["F4"]["theta"][kk + 1] + res["F4"]["theta"][kk + 2] * np.median(res["F4"]["a"])
        )
        thr = max(0.2 * pp.m_mw, min(2.0 * q_hat, 0.5 * pp.m_mw))
        keep = (y >= thr) & (w > 0)
        res["L1"] = (
            fit_linear(y=y, cell=cell, regs=[pp.u_std, pp.temp], w=w * keep)
            if keep.sum() > 50
            else float("nan")
        )
        res["thr"] = thr
        return res

    point = run(w=one)
    umax = np.zeros(pp.n_cells)
    np.maximum.at(umax, cell, pp.u_basis.sum(axis=1))
    for arm in ARMS:
        v = point[arm]
        out[arm] = v["c_total"] if isinstance(v, dict) else float(v)
    for arm in ("F4", "F1", "FA"):
        fit = point[arm]
        cvec = fit["c"]
        top = (pp.u_basis @ cvec) if arm != "F1" else (pp.u_std * cvec[0])
        top_cell = np.zeros(pp.n_cells)
        np.maximum.at(top_cell, cell, top)
        out[f"{arm}_cross"] = (fit["a"] < top_cell).tolist()
        out[f"{arm}_q0"], out[f"{arm}_q1"] = (
            float(fit["theta"][len(cvec) + 1]),
            float(fit["theta"][len(cvec) + 2]),
        )
        out[f"{arm}_mix"] = (
            (fit["c"] / max(fit["c"].sum(), 1e-9)).tolist() if arm != "F1" else [1.0]
        )
    out["thr"], out["m_mw"], out["y_scale"] = point["thr"], pp.m_mw, y_scale
    cls, s_lo, s_hi = curve_class(y=y, cell=cell, u=pp.u_std)
    out["curve_class"], out["slope_lo"], out["slope_hi"] = cls, s_lo, s_hi
    out["n_rows"] = len(y)
    out["cell_p_mw"], out["cell_umax"] = pp.cell_p_mw.tolist(), pp.cell_umax.tolist()
    out["cell_month"], out["cell_hh"] = pp.cell_month.tolist(), pp.cell_hh.tolist()
    if boot:
        draws = {arm: np.empty(N_BOOT) for arm in ARMS}
        for i in range(N_BOOT):
            w = rng.poisson(1.0, size=pp.n_dates).astype(np.float64)[pp.date_id]
            res = run(w=w, warm=point)
            for arm in ARMS:
                v = res[arm]
                draws[arm][i] = v["c_total"] if isinstance(v, dict) else float(v)
        for arm in ARMS:
            out[f"{arm}_boot"] = draws[arm].tolist()
    return out


def main() -> None:
    """Probe every requested network and write one parquet of results per cutoff and condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--conditions", nargs="+", default=list(k.CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    parser.add_argument("--boot-conditions", nargs="*", default=[])
    parser.add_argument("--sites", nargs="+", default=list(PROBED))
    parser.add_argument(
        "--stored",
        action="store_true",
        help="Stage 0: probe experiment 621's stored `control` networks.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--wide-starts",
        action="store_true",
        help="Post hoc: wider fold-fit starts; writes to `probes_wide`.",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    global WIDE_STARTS
    WIDE_STARTS = args.wide_starts
    frame = load_frame(cutoff=args.cutoff)
    idx = k.index_of(sites=tuple(args.sites))
    cap = k621.effective_capacity_mw()
    probes = {
        site: PrimaryProbe(frame=frame, primary=i, site=site, cap=float(cap[i]))
        for site, i in idx.items()
    }
    for site, pp in probes.items():
        print(
            site, "rows", len(pp.cell), "cells", pp.n_cells, "m_mw", round(pp.m_mw, 2), flush=True
        )
    for name in args.conditions:
        out = (
            k.WORK
            / ("stage0" if args.stored else "probes_wide" if args.wide_starts else "probes")
            / args.cutoff
            / f"{name}.parquet"
        )
        if out.exists():
            continue
        boot = name in args.boot_conditions
        records = []
        started = time.monotonic()
        for seed in args.seeds:
            path = (
                k621.weights_path(cutoff=args.cutoff, condition="control", seed=seed)
                if args.stored
                else k.weights_path(cutoff=args.cutoff, condition=name, seed=seed)
            )
            assert path.exists(), path
            model = m.ModularMdn(n_primaries=16, stage="pre")
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            for site, pp in probes.items():
                pp.set_arm(arm="pf" if args.stored else k.parse(name=name).arm)
                y = pp.forward(model=model, frame=frame)
                rng = np.random.default_rng(1000 * seed + pp.primary)
                res = probe_network(pp=pp, y=y, boot=boot, rng=rng)
                records.append(
                    {"cutoff": args.cutoff, "condition": name, "seed": seed, "site": site, **res}
                )
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, f"{time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
