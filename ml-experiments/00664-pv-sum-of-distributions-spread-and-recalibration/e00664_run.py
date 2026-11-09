# ruff: noqa: E501, ANN001, PLR0915, PT018
"""Experiment 664: score the free network, its recalibration, and the sum-of-distributions arms.

No network is refitted. For each cutoff, month, and seed the script re-loads two stored fine-tuned
networks: experiment 644's free network (arm `a`, net demand) and experiment 655's member-average
cell-temperature network (arm `d` there, gross demand: the target plus capacity times the
member-average solar curve). It rebuilds experiment 655's fine-tuning and test rows with the same
code, and writes the 11 quantiles and the probability integral transform (PIT) of each arm for every
test row:

- `a` free network; `b` its recalibration;
- `c` sum of distributions, independent: a gross-demand draw from arm d's mixture minus the curve of
  a randomly drawn ensemble member, 2,000 draws per row;
- `cc` the same with the two draws coupled at the same quantile level (sorted draws paired);
- `c0` gross draw minus the member-average curve (a check of the sampling);
- `d`, `dcc` recalibrations of `c` and `cc`;
- `p` placebo with the capacity of another Primary; `s` placebo with the curves of another date.

A recalibration is fitted on held-out fine-tuning rows (`is_val`, six Primaries) only. In each cell
(lead band by forecast sunny call) it maps level `tau` to the `tau`-quantile of the cell's validation
PITs.

Usage: python e00664_run.py run --cutoff 2026-06 [--seeds 0 1 2 3 4] [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00638_common as k638  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00639_data as d639  # ty: ignore[unresolved-import]
import e00639_net as net  # ty: ignore[unresolved-import]
import e00644_fit as f644  # ty: ignore[unresolved-import]
import e00655_curves as cv  # ty: ignore[unresolved-import]
import e00655_extract as ex  # ty: ignore[unresolved-import]
import e00655_fit as f655  # ty: ignore[unresolved-import]
import e00664_members as mem
import numpy as np
import torch  # ty: ignore[unresolved-import]

DEVICE = "cuda"
WORK: Final[Path] = mem.WORK
N_DRAWS: Final = 2000
CHUNK: Final = 4096
VAL_ROWS: Final = 150_000
MIN_CELL_ROWS: Final = 500
LEVELS: Final = np.asarray(k639.LEVELS)
ARMS: Final = ("a", "b", "c", "cc", "c0", "d", "dcc", "p", "s")
"""Stored arms. `a`, `b` use the free network; the rest use arm d's gross-demand network."""
N_MEMBERS: Final = ex.N_MEMBERS


def out_path(*, cutoff: str, month: str, seed: int) -> Path:
    """Return where one (month, seed)'s quantiles and PITs are stored."""
    return WORK / "preds" / cutoff / f"{month}_s{seed}.npz"


def val_path(*, cutoff: str, month: str, seed: int) -> Path:
    """Return where one (month, seed)'s validation-row statistics are stored."""
    return WORK / "preds" / cutoff / f"{month}_s{seed}_val.json"


def cell_of(*, lead: np.ndarray, sunny_fc: np.ndarray) -> np.ndarray:
    """Return the recalibration cell: 2 * lead band (day 1-2, 3-5, 6-10) + forecast call sunny."""
    day = (lead // 24).astype(int) + 1
    band = np.where(day <= 2, 0, np.where(day <= 5, 1, 2))
    return 2 * band + sunny_fc.astype(int)


class Members:
    """The normalised member curves of one cutoff, looked up by Primary, origin, and valid time."""

    def __init__(self, *, cutoff: str) -> None:
        """Memory-map the six Primaries' tables."""
        self.tab = {
            p: np.load(mem.table_path(cutoff=cutoff, primary=p), mmap_mode="r")
            for p in k634.six_indices().values()
        }

    def lookup(
        self, *, primary: np.ndarray, origin_us: np.ndarray, valid_us: np.ndarray
    ) -> np.ndarray:
        """Return (rows, 51) normalised member curves; a missing member takes the row mean."""
        out = np.zeros((len(primary), N_MEMBERS), dtype=np.float32)
        run = (origin_us.astype(np.int64) - cv.FIRST_ORIGIN_US) // cv.DAY_US
        step = (valid_us.astype(np.int64) - origin_us.astype(np.int64)) // cv.STEP_US - 1
        for p, tab in self.tab.items():
            sel = np.flatnonzero(primary == p)
            if not len(sel):
                continue
            assert run[sel].min() >= 0 and run[sel].max() < tab.shape[0], p
            assert step[sel].min() >= 0 and step[sel].max() < ex.N_STEPS, p
            out[sel] = tab[run[sel], step[sel]]
        bad = ~np.isfinite(out)
        if bad.any():
            row_mean = np.nanmean(np.where(bad, np.nan, out), axis=1, keepdims=True)
            out = np.where(bad, np.nan_to_num(row_mean), out)
        return out


# --- mixture helpers (GPU) --------------------------------------------------------------------


def parts(*, model, x: np.ndarray, p: np.ndarray, sl: slice) -> tuple[torch.Tensor, ...]:
    """Return the mixture's log weights, means, and standard deviations of the rows in `sl`."""
    xb = torch.from_numpy(np.ascontiguousarray(x[sl])).to(DEVICE)
    pb = torch.from_numpy(p[sl]).to(DEVICE)
    with torch.no_grad():
        lw, mu, sd = base.mixture_parts(out=model(xb, pb).double(), components=m.COMPONENTS)
    return lw, mu, sd, pb


def mixture_pit(*, lw, mu, sd, z: torch.Tensor) -> torch.Tensor:
    """Return the exact CDF of each mixture at `z` (rows)."""
    zz = (z[:, None] - mu) / (sd * float(np.sqrt(2.0)))
    return (lw.exp() * 0.5 * (1.0 + torch.erf(zz))).sum(-1)


def mixture_quantile(*, lw, mu, sd, levels: torch.Tensor) -> torch.Tensor:
    """Return the mixture quantiles at per-row `levels` (rows, L), by bisection on the CDF."""
    w = lw.exp()
    lo = torch.full(levels.shape, -base.Z_LIMIT, dtype=mu.dtype, device=mu.device)
    hi = torch.full_like(lo, base.Z_LIMIT)
    for _ in range(base.BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        z = (mid[..., None] - mu[:, None, :]) / (sd[:, None, :] * float(np.sqrt(2.0)))
        cdf = (w[:, None, :] * 0.5 * (1.0 + torch.erf(z))).sum(-1)
        below = cdf < levels
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def sample_quantile(*, sorted_draws: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return quantiles of sorted draws (rows, N) at per-row levels (rows, L), by linear interpolation."""
    n = sorted_draws.shape[1]
    pos = levels.clamp(0.0, 1.0) * (n - 1)
    i0 = pos.floor().long().clamp(0, n - 1)
    i1 = (i0 + 1).clamp(max=n - 1)
    frac = pos - i0.to(pos.dtype)
    return sorted_draws.gather(1, i0) * (1 - frac) + sorted_draws.gather(1, i1) * frac


def draws_pit(*, net_draws: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the fraction of draws at or below `y` (rows)."""
    return (net_draws <= y[:, None]).float().mean(1)


class Maps:
    """Per-arm, per-cell recalibration: the sorted validation PITs of each cell."""

    def __init__(self) -> None:
        """Start empty."""
        self.pits: dict[tuple[str, int], np.ndarray] = {}
        self.fallback: dict[tuple[str, int], str] = {}

    def fit(self, *, arm: str, pit: np.ndarray, cell: np.ndarray) -> None:
        """Store each cell's sorted validation PITs; a cell with few rows borrows its lead band."""
        for k in range(6):
            sel = cell == k
            if sel.sum() >= MIN_CELL_ROWS:
                self.pits[(arm, k)] = np.sort(pit[sel])
                continue
            band = cell // 2 == k // 2
            if band.sum() >= MIN_CELL_ROWS:
                self.pits[(arm, k)] = np.sort(pit[band])
                self.fallback[(arm, k)] = "band"
            else:
                self.pits[(arm, k)] = np.sort(pit)
                self.fallback[(arm, k)] = "all"

    def levels(self, *, arm: str, cell: np.ndarray) -> np.ndarray:
        """Return, per row, the 11 levels at which the raw forecast is read to give the calibrated quantiles."""
        table = np.stack([np.quantile(self.pits[(arm, k)], LEVELS) for k in range(6)])
        return np.clip(table[cell], 1e-3, 1.0 - 1e-3)

    def calibrated_pit(self, *, arm: str, pit: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Return `G(pit)`, the empirical CDF of the cell's validation PITs at `pit`."""
        out = np.empty(len(pit))
        for k in range(6):
            sel = cell == k
            if sel.any():
                ref = self.pits[(arm, k)]
                out[sel] = np.searchsorted(ref, pit[sel], side="right") / (len(ref) + 1)
        return out


def run_pass(
    *, model_a, model_d, rows: dict, mean, std, seed: int, maps: Maps | None, arms: tuple[str, ...]
) -> dict[str, np.ndarray]:  # fmt: skip
    """Compute PITs (and, when `maps` is given, quantiles) of the arms for all rows of a block.

    `rows` holds `x`, `p`, `y` (points), `pv_mean` (points), `pv` (rows, 51; points), `pv_perm`
    (the curves scaled by another Primary's capacity), `pv_shuf` (another date's curves), and `cell`.
    """
    n = len(rows["y"])
    gen = torch.Generator(device=DEVICE).manual_seed(10_000 + seed)
    res: dict[str, list[np.ndarray]] = {}

    def put(name: str, arr: torch.Tensor) -> None:
        res.setdefault(name, []).append(arr.cpu().numpy())

    lv_t = torch.tensor(LEVELS, dtype=torch.float32, device=DEVICE)
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(lo + CHUNK, n))
        y = torch.from_numpy(rows["y"][sl]).to(DEVICE, torch.float32)
        cell = rows["cell"][sl]

        # Arm a (free network, net demand).
        if "a" in arms:
            lw, mu, sd, pb = parts(model=model_a, x=rows["x"], p=rows["p"], sl=sl)
            zy = ((y - mean[pb]) / std[pb]).double()
            pit_a = mixture_pit(lw=lw, mu=mu, sd=sd, z=zy)
            put("pit_a", pit_a)
            if maps is not None:
                q11 = mixture_quantile(
                    lw=lw, mu=mu, sd=sd, levels=lv_t.double()[None].expand(len(y), -1)
                )
                put("q_a", q11 * std[pb][:, None] + mean[pb][:, None])
                lev = torch.from_numpy(maps.levels(arm="a", cell=cell)).to(DEVICE)
                qb = mixture_quantile(lw=lw, mu=mu, sd=sd, levels=lev.double())
                put("q_b", qb * std[pb][:, None] + mean[pb][:, None])
                put(
                    "pit_b",
                    torch.from_numpy(
                        maps.calibrated_pit(arm="a", pit=pit_a.cpu().numpy(), cell=cell)
                    ),
                )
        # Arm d's gross-demand network: draws.
        lw, mu, sd, pb = parts(model=model_d, x=rows["x_d"], p=rows["p"], sl=sl)
        r = len(y)
        comp = torch.multinomial(lw.exp().float(), N_DRAWS, replacement=True, generator=gen)
        z = mu.float().gather(1, comp) + sd.float().gather(1, comp) * torch.randn(
            r, N_DRAWS, device=DEVICE, generator=gen
        )
        gross = z * std[pb][:, None] + mean[pb][:, None]
        member = torch.randint(0, N_MEMBERS, (r, N_DRAWS), device=DEVICE, generator=gen)
        pv = torch.from_numpy(rows["pv"][sl]).to(DEVICE)
        pv_draw = pv.gather(1, member)
        variants: dict[str, torch.Tensor] = {}
        if "c" in arms:
            variants["c"] = (gross - pv_draw).sort(dim=1).values
        if "cc" in arms:
            variants["cc"] = gross.sort(dim=1).values - pv_draw.sort(dim=1).values
        if "c0" in arms:
            variants["c0"] = (
                (gross - torch.from_numpy(rows["pv_mean"][sl]).to(DEVICE)[:, None])
                .sort(dim=1)
                .values
            )
        if "p" in arms:
            variants["p"] = (
                (gross - torch.from_numpy(rows["pv_perm"][sl]).to(DEVICE).gather(1, member))
                .sort(dim=1)
                .values
            )
        if "s" in arms:
            variants["s"] = (
                (gross - torch.from_numpy(rows["pv_shuf"][sl]).to(DEVICE).gather(1, member))
                .sort(dim=1)
                .values
            )
        for name, draws in variants.items():
            pit = draws_pit(net_draws=draws, y=y)
            put(f"pit_{name}", pit)
            if maps is not None:
                put(
                    f"q_{name}",
                    sample_quantile(sorted_draws=draws, levels=lv_t[None].expand(r, -1)),
                )
                if name in ("c", "cc"):
                    arm_cal = "d" if name == "c" else "dcc"
                    lev = torch.from_numpy(maps.levels(arm=name, cell=cell)).to(
                        DEVICE, torch.float32
                    )
                    put(f"q_{arm_cal}", sample_quantile(sorted_draws=draws, levels=lev))
                    put(
                        f"pit_{arm_cal}",
                        torch.from_numpy(
                            maps.calibrated_pit(arm=name, pit=pit.cpu().numpy(), cell=cell)
                        ),
                    )
    return {k: np.concatenate(v) for k, v in res.items()}


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE, torch.float32)


def run_month(*, cutoff: str, month: str, seeds: list[int], clean: np.ndarray, bank, members: Members, smoke: bool) -> None:  # fmt: skip
    """Run every seed of one cutoff's month."""
    six = k634.six_indices()
    six_idx = np.array(sorted(six.values()))
    refit = c.month_start(month=month)
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    mean = torch.from_numpy(mr.base.target_mean).to(DEVICE, torch.float32)
    std = torch.from_numpy(mr.base.target_std).to(DEVICE, torch.float32)
    sel = c.select_rows(rows=mr, cutoff=refit, half_life_days=f644.HALF_LIFE_DAYS)
    idx = sel.index
    inj = k638.Injector(primary=mr.primary[idx], target_idx=mr.target_idx[idx], x=mr.x[idx], standardiser=mr.base.standardiser)  # fmt: skip
    keep = ~inj.gap
    blk = d639._decorate(
        cutoff=cutoff, world="real", x=mr.x[idx], y=mr.y[idx].astype(np.float64), primary=mr.primary[idx],
        target_idx=mr.target_idx[idx], is_val=mr.is_val[idx], inj=inj,
    )  # fmt: skip
    tr = d639._select(block=blk, keep=keep)
    origin_tr = mr.origin_us[idx][keep]
    valid_tr = inj.t_us[keep]
    # Test rows exactly as experiment 655.
    ok = np.isin(test.primary, list(six.values())) & (test.lead >= f644.MIN_LEAD_H)
    y_test = clean[test.primary, test.target_idx]
    ok &= np.isfinite(y_test)
    tid = np.flatnonzero(ok)
    tinj = k638.Injector(primary=test.primary[tid], target_idx=test.target_idx[tid], x=test.x[tid], standardiser=mr.base.standardiser)  # fmt: skip
    tblk = d639._decorate(
        cutoff=cutoff, world="real", x=test.x[tid], y=y_test[tid].astype(np.float64),
        primary=test.primary[tid], target_idx=test.target_idx[tid],
        is_val=np.zeros(len(tid), dtype=bool), inj=tinj,
    )  # fmt: skip
    tkeep = ~tinj.gap
    donor_pre = d639._shuffle_donors(primary=test.primary[tid], t_us=tinj.t_us)
    origin_te_pre, valid_te_pre = test.origin[tid], tinj.t_us
    tblk = d639._select(block=tblk, keep=tkeep)
    tid = tid[tkeep]
    z644 = np.load(f644.rows_path(month=month, world="real"))
    assert np.array_equal(z644["primary"], tblk.primary) and np.array_equal(z644["t_us"], tblk.t_us)
    assert np.array_equal(z644["origin_us"], test.origin[tid]) and np.allclose(z644["y"], tblk.y)
    # Member curves of the test rows (normalised) and the member-average table of experiment 655 for a check.
    m_pre = members.lookup(primary=tinj.primary, origin_us=origin_te_pre, valid_us=valid_te_pre)
    table_d = bank.lookup(
        variant="d", primary=tinj.primary, origin_us=origin_te_pre, valid_us=valid_te_pre
    )
    live = table_d > 0
    err = np.abs(m_pre.mean(axis=1) - table_d)[live]
    assert np.quantile(err, 0.999) < 5e-3 and err.max() < 0.05, (
        "member mean differs from experiment 655's table",
        err.max(),
    )
    m_te = m_pre[tkeep]
    m_te_shuf = m_pre[donor_pre][tkeep]
    # Fine-tuning rows: held-out `is_val` rows of the six Primaries (a fixed random subsample).
    pick = np.flatnonzero(tr.is_val & np.isin(tr.primary, six_idx))
    rng = np.random.default_rng(664)
    if len(pick) > VAL_ROWS:
        pick = np.sort(rng.choice(pick, VAL_ROWS, replace=False))
    m_va = members.lookup(
        primary=tr.primary[pick], origin_us=origin_tr[pick], valid_us=valid_tr[pick]
    )
    lead_va = (valid_tr[pick] - origin_tr[pick]) / 3.6e9
    assert lead_va.min() >= 0.0 and lead_va.max() <= 360.0, (lead_va.min(), lead_va.max())
    lead_te = (tblk.t_us - test.origin[tid]) / 3.6e9
    assert np.allclose(lead_te, test.lead[tid], atol=1e-3), (
        "lead definition differs from experiment 644's"
    )
    sunny_te = (
        (tblk.kt >= k639.SUNNY_KT)
        & (tblk.hour >= k639.SUNNY_HOURS[0])
        & (tblk.hour <= k639.SUNNY_HOURS[1])
    )
    sunny_va = (
        (tr.kt[pick] >= k639.SUNNY_KT)
        & (tr.hour[pick] >= k639.SUNNY_HOURS[0])
        & (tr.hour[pick] <= k639.SUNNY_HOURS[1])
    )
    perm = np.zeros(16, dtype=np.int64)
    sx = sorted(six.values())
    for i, p in enumerate(sx):
        perm[p] = sx[(i + 1) % len(sx)]
    tensors = net.parameter_tensors(
        params=k639.capacity_parameters(cutoff=cutoff, arm="cal"), six=six, device=DEVICE
    )
    for seed in seeds:
        target = out_path(cutoff=cutoff, month=month, seed=seed)
        if target.exists() and not smoke:
            continue
        started = time.monotonic()
        torch.manual_seed(seed)
        model_a = m.ModularMdn(n_primaries=16, stage="ft").to(DEVICE)
        model_a.load_state_dict(torch.load(f644.fits_path(cutoff=cutoff, month=month, world="real", arm="a", seed=seed), map_location=DEVICE))  # fmt: skip
        model_d = m.ModularMdn(n_primaries=16, stage="ft").to(DEVICE)
        model_d.load_state_dict(torch.load(f655.fits_path(cutoff=cutoff, month=month, arm="d", seed=seed), map_location=DEVICE))  # fmt: skip
        model_a.eval()
        model_d.eval()
        delta = torch.tensor(json.loads(f655.fits_path(cutoff=cutoff, month=month, arm="d", seed=seed).with_suffix(".json").read_text())["delta_mw"], dtype=torch.float32, device=DEVICE)  # fmt: skip

        def pv_points(*, p, r4, after, inc, curves, p_cap=None) -> np.ndarray:
            cap = net.capacity_mw(t=tensors, delta=delta, p=torch.from_numpy(p if p_cap is None else p_cap).to(DEVICE), r4=_f32(r4), after=_f32(after))  # fmt: skip  # noqa: B023
            return (cap[:, None] * _f32(inc)[:, None] * _f32(curves)).cpu().numpy()

        pv_te = pv_points(p=tblk.primary, r4=tblk.r4, after=tblk.after, inc=tblk.inc, curves=m_te)
        pv_va = pv_points(
            p=tr.primary[pick], r4=tr.r4[pick], after=tr.after[pick], inc=tr.inc[pick], curves=m_va
        )
        rows_te = {
            "x": tblk.x, "x_d": tblk.x, "p": tblk.primary, "y": tblk.y, "pv": pv_te, "pv_mean": pv_te.mean(axis=1),
            "pv_perm": pv_points(p=tblk.primary, r4=tblk.r4, after=tblk.after, inc=tblk.inc, curves=m_te, p_cap=perm[tblk.primary]),
            "pv_shuf": pv_points(p=tblk.primary, r4=tblk.r4, after=tblk.after, inc=tblk.inc, curves=m_te_shuf),
            "cell": cell_of(lead=lead_te, sunny_fc=sunny_te),
        }  # fmt: skip
        rows_va = {
            "x": tr.x[pick], "x_d": tr.x[pick], "p": tr.primary[pick], "y": tr.y[pick], "pv": pv_va, "pv_mean": pv_va.mean(axis=1),
            "pv_perm": pv_va, "pv_shuf": pv_va, "cell": cell_of(lead=lead_va, sunny_fc=sunny_va),
        }  # fmt: skip
        # Pass 1: validation PITs -> recalibration maps.
        pit_va = run_pass(model_a=model_a, model_d=model_d, rows=rows_va, mean=mean, std=std, seed=seed, maps=None, arms=("a", "c", "cc"))  # fmt: skip
        maps = Maps()
        maps.fit(arm="a", pit=pit_va["pit_a"], cell=rows_va["cell"])
        maps.fit(arm="c", pit=pit_va["pit_c"], cell=rows_va["cell"])
        maps.fit(arm="cc", pit=pit_va["pit_cc"], cell=rows_va["cell"])
        # Correlation between the solar curve and the position of the realised gross demand in the network's gross distribution.
        gross_y = rows_va["y"] + rows_va["pv_mean"]
        with torch.no_grad():
            pit_gross = []
            for lo in range(0, len(gross_y), CHUNK):
                sl = slice(lo, min(lo + CHUNK, len(gross_y)))
                lw, mu, sd, pb = parts(model=model_d, x=rows_va["x_d"], p=rows_va["p"], sl=sl)
                zy = (
                    (torch.from_numpy(gross_y[sl]).to(DEVICE, torch.float32) - mean[pb]) / std[pb]
                ).double()
                pit_gross.append(mixture_pit(lw=lw, mu=mu, sd=sd, z=zy).cpu().numpy())
            pit_gross = np.concatenate(pit_gross)
        live = sunny_va & (rows_va["pv_mean"] > 0)
        stats = {
            "n_val_rows": len(pick),
            "n_val_sunny": int(live.sum()),
            "cells": {str(k): int((rows_va["cell"] == k).sum()) for k in range(6)},
            "fallback": {f"{a}:{k}": v for (a, k), v in maps.fallback.items()},
            "corr_pv_mean_vs_gross_pit_sunny": float(np.corrcoef(rows_va["pv_mean"][live], pit_gross[live])[0, 1]) if live.sum() > 10 else None,
            "corr_pv_member_spread_vs_abs_gross_pit_sunny": float(np.corrcoef(rows_va["pv"][live].std(axis=1), np.abs(pit_gross[live] - 0.5))[0, 1]) if live.sum() > 10 else None,
            "corr_pv_mean_vs_net_pit_a_sunny": float(np.corrcoef(rows_va["pv_mean"][live], pit_va["pit_a"][live])[0, 1]) if live.sum() > 10 else None,
            "pit_gross_mean_sunny": float(pit_gross[live].mean()) if live.any() else None,
            "levels_a_sunny_d1_2": maps.levels(arm="a", cell=np.array([1]))[0].tolist(),
            "levels_c_sunny_d1_2": maps.levels(arm="c", cell=np.array([1]))[0].tolist(),
        }  # fmt: skip
        # Pass 2: test rows.
        res = run_pass(
            model_a=model_a,
            model_d=model_d,
            rows=rows_te,
            mean=mean,
            std=std,
            seed=seed,
            maps=maps,
            arms=ARMS,
        )
        # Check: arm a's quantiles from the mixture equal the stored ones; c0 reproduces arm d's stored quantiles in the median.
        stored_a = np.load(
            f644.fits_path(
                cutoff=cutoff, month=month, world="real", arm="a", seed=seed
            ).with_suffix(".npz")
        )["q"]
        stored_d = np.load(
            f655.fits_path(cutoff=cutoff, month=month, arm="d", seed=seed).with_suffix(".npz")
        )["q"]
        stats["max_abs_q_a_vs_stored"] = float(np.abs(res["q_a"] - stored_a).max())
        stats["median_abs_q_c0_minus_stored_d"] = float(
            np.abs(res["q_c0"][:, 5] - stored_d[:, 5]).mean()
        )
        stats["max_abs_q_c0_minus_stored_d_median"] = float(
            np.abs(res["q_c0"][:, 5] - stored_d[:, 5]).max()
        )
        stats["seconds"] = round(time.monotonic() - started)
        print(
            f"{cutoff} {month} s{seed}: {json.dumps({k: v for k, v in stats.items() if k not in ('cells', 'fallback')})}",
            flush=True,
        )
        if smoke:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp.npz")
        np.savez(tmp, **{k: v.astype(np.float32) for k, v in res.items()})  # ty: ignore[invalid-argument-type]
        tmp.rename(target)
        val_path(cutoff=cutoff, month=month, seed=seed).write_text(json.dumps(stats))
        del model_a, model_d
        torch.cuda.empty_cache()


def main() -> None:
    """Run one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run",))
    parser.add_argument("--cutoff", required=True, choices=k639.CUTOFFS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k639.SEEDS))
    parser.add_argument("--smoke", action="store_true", help="Run one seed, store nothing.")
    a = parser.parse_args()
    torch.set_num_threads(2)
    clean = c.clean_stack()
    bank = f655.Bank(cutoff=a.cutoff)
    members = Members(cutoff=a.cutoff)
    for month in f644.MONTHS[a.cutoff]:
        run_month(cutoff=a.cutoff, month=month, seeds=a.seeds[:1] if a.smoke else a.seeds, clean=clean, bank=bank, members=members, smoke=a.smoke)  # fmt: skip
        if a.smoke:
            break
    print(
        json.dumps({"done": a.phase, "cutoff": a.cutoff, "at": datetime.now(UTC).isoformat()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
