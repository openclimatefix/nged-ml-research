# ruff: noqa: E501, ANN001, PLR0915
"""Experiment 661, Part 3: a bias adapter and a fleet-shape mixture.

Experiment 661, Part 3: a bias adapter and a fleet-shape mixture in front of experiment 644's
physics curve.

Arm A (`adapter`) is experiment 644's calibrated explicit network (arm b) with its fixed physics
curve replaced by a differentiable one:

1. A small adapter maps the ensemble inputs of each row (ensemble-mean clear-sky index, the ensemble
   standard deviation of shortwave as a share of clear-sky irradiance, lead in days, hour of day as
   sine and cosine, temperature) to a bounded correction of the clear-sky index, `kt' = kt + 0.25 *
   tanh(MLP)`, clipped to 0 to 1.4. The last layer starts at zero, so training starts at the
   mean-sky curve.
2. The corrected irradiance `kt' * clear_sky` goes through plane-of-array irradiance tabulated on a
   grid of clear-sky indices (pvlib Erbs split, three tilts, south facing), interpolated linearly.
3. A fleet-shape mixture per Primary weights 15 components (three tilts times five DC-to-AC ratios,
   1.0 to 1.7), each with the cell-temperature derating and a smooth clip at AC capacity (no hard
   clip). The weights are a softmax of free logits under a Dirichlet prior (concentration 2).

Arm A0 (`nested`) is the same curve with the adapter switched off (`kt' = kt`), which isolates the
mixture. The capacity timeline (calibrated, with the offset and its tight prior), the fine-tuning
rows, recency weights, the fresh mixer group, and the scoring are experiment 644's. Each fine-tune
starts from experiment 639's stored calibrated pretraining of the same cutoff and seed.

Usage: python e00661_adapter.py fit --cutoff 2026-06 --arms adapter nested --seeds 0 1 2 3 4 [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00634_common as k634  # ty: ignore[unresolved-import]
import e00638_common as k638  # ty: ignore[unresolved-import]
import e00639_common as k639  # ty: ignore[unresolved-import]
import e00639_data as d639  # ty: ignore[unresolved-import]
import e00639_net as net  # ty: ignore[unresolved-import]
import e00644_fit as fit  # ty: ignore[unresolved-import]
import e00661_degrade as g
import numpy as np
import pvlib
import torch  # ty: ignore[unresolved-import]
from studies.pv_physics import SunAndSky, plane_of_array_w_m2

DEVICE = "cuda"
WORK: Final[Path] = g.WORK / "adapter"
KT_GRID: Final[np.ndarray] = np.linspace(0.0, 1.5, 16)
"""Clear-sky indices at which plane-of-array irradiance is tabulated."""
TILTS_DEG: Final[tuple[float, ...]] = (20.0, 35.0, 50.0)
RATIOS: Final[tuple[float, ...]] = (1.0, 1.15, 1.3, 1.5, 1.7)
CORRECTION_BOUND: Final = 0.25
DIRICHLET_ALPHA: Final = 2.0
MIX_PENALTY_ROWS: Final = 1000.0
ADAPTER_LR: Final = 2e-3
SOFTCLIP_POWER: Final = 8.0
ARMS: Final[dict[str, bool]] = {"adapter": True, "nested": False}
"""Arm -> whether the adapter is on."""
INIT_COMPONENT: Final[tuple[int, int]] = (1, 1)
"""The (tilt, ratio) component experiment 644's fixed curve corresponds to: 35 degrees, 1.15."""
MIX_INIT_LOGIT: Final = 2.0


def poa_basis(*, cos_zenith: np.ndarray, azimuth_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Return plane-of-array irradiance (W/m2), shape (rows, tilts, KT_GRID), for south-facing fixed panels."""
    n = len(cos_zenith)
    clear = pv_clear(cos_zenith=cos_zenith)
    out = np.zeros((n, len(TILTS_DEG), len(KT_GRID)), dtype=np.float32)
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    day = cos_zenith > 0.0
    for j, kt in enumerate(KT_GRID):
        ghi = np.where(day, kt * clear, 0.0)
        split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=doy)
        sky = SunAndSky(
            half_hour_end_time=np.zeros(n, dtype="datetime64[us]"), ghi_w_m2=ghi,
            dni_w_m2=np.nan_to_num(np.asarray(split["dni"], dtype=np.float64)),
            dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"], dtype=np.float64)),
            zenith_deg=zenith, sun_azimuth_deg=azimuth_deg,
        )  # fmt: skip
        for i, tilt in enumerate(TILTS_DEG):
            out[:, i, j] = plane_of_array_w_m2(
                sky=sky, tilt_deg=tilt, azimuth_deg=180.0, tracker=False
            )
    return out


def pv_clear(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return clear-sky global horizontal irradiance (Haurwitz), the denominator of the clear-sky index."""
    return k634.haurwitz_w_m2(cos_zenith=cos_zenith)


class Rows:
    """Per-row inputs of the adapted curve for a set of fine-tuning or test rows (all on the GPU)."""

    def __init__(
        self,
        *,
        x: np.ndarray,
        primary: np.ndarray,
        target_idx: np.ndarray,
        lead: np.ndarray,
        standardiser,
        six: dict[str, int],
    ) -> None:
        """Build the adapter features and the tabulated irradiance for each row."""
        ghi, temp = k634.raw_weather(x=x, standardiser=standardiser)
        sd_col = 7  # std of the first weather column (shortwave), standardised like the mean
        sd = np.clip(
            x[:, sd_col].astype(np.float64) * standardiser.std[sd_col] + standardiser.mean[sd_col],
            0.0,
            None,
        )
        cz = x[:, 14].astype(np.float64)
        clear = pv_clear(cos_zenith=cz)
        sunlit = clear > 10.0
        kt = np.where(sunlit, ghi / np.maximum(clear, 1e-9), 0.0)
        kt_sd = np.where(sunlit, sd / np.maximum(clear, 1e-9), 0.0)
        t_us = k634.time_us(target_idx=target_idx)
        hour = ((t_us // 3_600_000_000) % 24).astype(np.float64)
        doy = ((t_us // 86_400_000_000) % 365.25).astype(np.float64) + 1.0
        key = primary.astype(np.int64) * 10**8 + target_idx.astype(np.int64)
        _uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        azimuth = (
            np.degrees(np.arctan2(x[first, 15].astype(np.float64), x[first, 16].astype(np.float64)))
            % 360.0
        )
        basis_u = poa_basis(cos_zenith=cz[first], azimuth_deg=azimuth, doy=doy[first])
        f32 = torch.float32
        feat = np.stack(
            [
                kt,
                kt_sd,
                lead / 240.0,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                temp / 20.0,
            ],
            axis=1,
        )
        self.feat = torch.from_numpy(feat).to(DEVICE, f32)
        self.kt = torch.from_numpy(kt).to(DEVICE, f32)
        self.clear = torch.from_numpy(clear).to(DEVICE, f32)
        self.temp = torch.from_numpy(temp).to(DEVICE, f32)
        self.basis_u = torch.from_numpy(basis_u).to(DEVICE, f32)
        self.inv = torch.from_numpy(inv.astype(np.int64)).to(DEVICE)
        self.primary = torch.from_numpy(primary).to(DEVICE)
        self.six = six

    def take(self, idx: torch.Tensor) -> Rows:
        """Return a shallow view of the rows at `idx` (indices on the GPU)."""
        view = object.__new__(Rows)
        for name in ("feat", "kt", "clear", "temp", "inv", "primary"):
            setattr(view, name, getattr(self, name)[idx])
        view.basis_u, view.six = self.basis_u, self.six
        return view


class AdaptedCurve(torch.nn.Module):
    """The differentiable physics curve: adapter, tabulated irradiance, fleet-shape mixture."""

    def __init__(self, *, adapter_on: bool, n_primaries: int = 16) -> None:
        """Initialise at experiment 644's curve: zero correction, weight concentrated on 35 degrees and 1.15."""
        super().__init__()
        self.adapter_on = adapter_on
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(6, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )
        torch.nn.init.zeros_(self.mlp[-1].weight)  # ty: ignore[invalid-argument-type]
        torch.nn.init.zeros_(self.mlp[-1].bias)  # ty: ignore[invalid-argument-type]
        logits = torch.zeros(n_primaries, len(TILTS_DEG) * len(RATIOS))
        logits[:, INIT_COMPONENT[0] * len(RATIOS) + INIT_COMPONENT[1]] = MIX_INIT_LOGIT
        self.logits = torch.nn.Parameter(logits)
        self.register_buffer("ratios", torch.tensor(RATIOS))
        self.register_buffer("grid", torch.tensor(KT_GRID, dtype=torch.float32))
        self.u99 = pv.u99()

    def weights(self) -> torch.Tensor:
        """Return the mixture weights per Primary (rows sum to one)."""
        return torch.softmax(self.logits, dim=1)

    def prior_penalty(self) -> torch.Tensor:
        """Return the negative log Dirichlet density of the mixture weights.

        Return the negative log Dirichlet density (up to a constant) of the weights of the six
        Primaries, per fine-tuning row.
        """
        w = self.weights()
        return -(DIRICHLET_ALPHA - 1.0) * torch.log(w + 1e-9).sum() / MIX_PENALTY_ROWS

    def corrected_kt(self, r: Rows) -> torch.Tensor:
        """Return the corrected clear-sky index of each row."""
        if not self.adapter_on:
            return r.kt
        return torch.clamp(
            r.kt + CORRECTION_BOUND * torch.tanh(self.mlp(r.feat).squeeze(1)), 0.0, KT_GRID[-1]
        )

    def forward(self, r: Rows) -> torch.Tensor:
        """Return the unit curve of each row.

        Return the unit curve (AC output per MW of AC capacity, divided by the 99th-percentile
        constant) of each row.
        """
        kt = self.corrected_kt(r)
        pos = kt / float(KT_GRID[1] - KT_GRID[0])
        i0 = torch.clamp(pos.floor().long(), 0, len(KT_GRID) - 2)
        frac = (pos - i0.float()).unsqueeze(1)
        table = r.basis_u[r.inv]  # (rows, tilts, grid)
        lo = torch.gather(table, 2, i0[:, None, None].expand(-1, len(TILTS_DEG), 1)).squeeze(2)
        hi = torch.gather(table, 2, (i0 + 1)[:, None, None].expand(-1, len(TILTS_DEG), 1)).squeeze(
            2
        )
        poa = (
            lo + (hi - lo) * frac
        ) / 1000.0  # (rows, tilts), kW/m2 = DC output per MW DC before derating
        cell = r.temp[:, None] + pv.CELL_RISE_K * poa
        dc = poa * torch.clamp(
            1.0 - pv.DERATING_PER_KELVIN * (cell - pv.REFERENCE_CELL_C), min=0.0
        )  # (rows, tilts)
        z = dc[:, :, None] * self.ratios[None, None, :]  # (rows, tilts, ratios), per MW AC  # ty: ignore[not-subscriptable]
        ac = z / (1.0 + z.clamp(min=0.0) ** SOFTCLIP_POWER) ** (1.0 / SOFTCLIP_POWER)
        w = self.weights()[r.primary].reshape(-1, len(TILTS_DEG), len(RATIOS))
        return (w * ac).sum(dim=(1, 2)) / self.u99


def _train(*, model, curve, tr: Rows, y, r4, after, p, w, is_val, mean, std, tensors, inc, seed: int, init_delta) -> tuple[int, float, torch.Tensor]:  # fmt: skip
    """Experiment 644's `_train` with the adapted curve in place of the fixed one."""
    torch.manual_seed(seed)
    delta = init_delta.clone().requires_grad_(True)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(DEVICE)]
    fresh = list(model.mixer_ft.parameters())
    fresh_ids = {id(q) for q in fresh}
    optimiser = torch.optim.AdamW(
        [
            {"params": [q for q in model.parameters() if id(q) not in fresh_ids], "lr": c.FINE_TUNE_LR, "weight_decay": c.WEIGHT_DECAY},
            {"params": fresh, "lr": fit.FRESH_LR, "weight_decay": c.WEIGHT_DECAY},
            {"params": [delta], "lr": c.FINE_TUNE_LR, "weight_decay": 0.0},
            {"params": list(curve.parameters()), "lr": ADAPTER_LR, "weight_decay": 0.0},
        ]
    )  # fmt: skip
    prior = delta.detach().clone()

    def nll_of(idx: torch.Tensor, ek, ed) -> torch.Tensor:
        cap = net.capacity_mw(
            t=tensors, delta=delta, p=p[idx], r4=r4[idx], after=after[idx], ek=ek, ed=ed
        )
        u = curve(tr.take(idx)) * inc[idx]
        z = (y[idx] + cap * u - mean[p[idx]]) / std[p[idx]]
        return base.mixture_nll(out=model(tr_x[idx], p[idx]), z=z, components=m.COMPONENTS)

    tr_x = tr.x_gpu  # ty: ignore[unresolved-attribute]
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        curve.train()
        ek = torch.randn(net.N_PRIMARIES, device=DEVICE) * tensors["sd_k"]
        ed = torch.randn(net.N_PRIMARIES, device=DEVICE) * tensors["sd_d"]
        order = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            penalty = (
                ((delta - prior) ** 2).sum()
                / (2 * k639.DELTA_PRIOR_MW**2)
                / k639.DELTA_PENALTY_ROWS
            )
            nll = nll_of(b, ek, ed)
            loss = (w[b] * nll).sum() / w[b].sum() + penalty + curve.prior_penalty()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        curve.eval()
        with torch.no_grad():
            val = float(
                torch.cat([nll_of(i, None, None) for i in val_idx.split(c.BATCH * 2)]).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = (
                {a: v.detach().clone() for a, v in model.state_dict().items()},
                {a: v.detach().clone() for a, v in curve.state_dict().items()},
                delta.detach().clone(),
            )
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state[0])
    curve.load_state_dict(best_state[1])
    model.eval()
    curve.eval()
    return epochs, best, best_state[2]


def run_month(
    *, cutoff: str, month: str, arms: list[str], seeds: list[int], clean: np.ndarray, smoke: bool
) -> None:
    """Fine-tune the adapter arms for one month of one cutoff and forecast it.

    Fine-tune the adapter arms for one month of one cutoff and forecast it (experiment 644's
    `run_month`, real world, arm b's start).
    """
    six = k634.six_indices()
    world = "real"
    refit = c.month_start(month=month)
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    mean = torch.from_numpy(mr.base.target_mean).to(DEVICE, torch.float32)
    std = torch.from_numpy(mr.base.target_std).to(DEVICE, torch.float32)
    sel = c.select_rows(rows=mr, cutoff=refit, half_life_days=fit.HALF_LIFE_DAYS)
    assert sel.report["rows_past_valid_cut"] == 0
    assert sel.report["rows_past_origin_cut"] == 0
    idx = sel.index[::20] if smoke else sel.index
    inj = k638.Injector(
        primary=mr.primary[idx],
        target_idx=mr.target_idx[idx],
        x=mr.x[idx],
        standardiser=mr.base.standardiser,
    )
    keep = ~inj.gap
    blk = d639._decorate(cutoff=cutoff, world=world, x=mr.x[idx], y=mr.y[idx].astype(np.float64), primary=mr.primary[idx], target_idx=mr.target_idx[idx], is_val=mr.is_val[idx], inj=inj)  # fmt: skip
    tr = d639._select(block=blk, keep=keep)
    lead_tr = mr.lead[idx][keep].astype(np.float64)
    w_all = (sel.weight[::20] if smoke else sel.weight)[keep]
    ok = np.isin(test.primary, list(six.values())) & (test.lead >= fit.MIN_LEAD_H)
    y_test = clean[test.primary, test.target_idx]
    ok &= np.isfinite(y_test)
    tid = np.flatnonzero(ok)
    tinj = k638.Injector(
        primary=test.primary[tid],
        target_idx=test.target_idx[tid],
        x=test.x[tid],
        standardiser=mr.base.standardiser,
    )
    tblk = d639._decorate(cutoff=cutoff, world=world, x=test.x[tid], y=y_test[tid].astype(np.float64), primary=test.primary[tid], target_idx=test.target_idx[tid], is_val=np.zeros(len(tid), dtype=bool), inj=tinj)  # fmt: skip
    tkeep = ~tinj.gap
    tblk = d639._select(block=tblk, keep=tkeep)
    tid = tid[tkeep]
    lead_te = test.lead[tid].astype(np.float64)
    std_ = mr.base.standardiser
    rows_tr = Rows(
        x=tr.x,
        primary=tr.primary,
        target_idx=tr.target_idx,
        lead=lead_tr,
        standardiser=std_,
        six=six,
    )
    rows_te = Rows(
        x=tblk.x,
        primary=tblk.primary,
        target_idx=tblk.target_idx,
        lead=lead_te,
        standardiser=std_,
        six=six,
    )
    rows_tr.x_gpu = torch.from_numpy(tr.x).to(DEVICE)  # ty: ignore[unresolved-attribute]
    f32 = fit._f32
    y, r4, after, p, w = (
        f32(tr.y),
        f32(tr.r4),
        f32(tr.after),
        torch.from_numpy(tr.primary).to(DEVICE),
        torch.from_numpy(w_all.astype(np.float32)).to(DEVICE),
    )
    is_val, inc = torch.from_numpy(tr.is_val).to(DEVICE), f32(tr.inc)
    params = k639.capacity_parameters(cutoff=cutoff, arm="cal")
    tensors = net.parameter_tensors(params=params, six=six, device=DEVICE)
    p_test = torch.from_numpy(tblk.primary).to(DEVICE)
    inc_te = f32(tblk.inc)
    for arm in arms:
        for seed in seeds:
            out = WORK / "fits" / cutoff / f"{month}_{arm}_s{seed}.pt"
            target = out.with_suffix(".npz")
            if target.exists() and not smoke:
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=16, stage="ft").to(DEVICE)
            pre = k639.explicit_weights_path(cutoff=cutoff, world=world, arm="cal", seed=seed)
            fit._load_for_finetune(model=model, path=pre)
            model.to(DEVICE)
            init = torch.tensor(
                json.loads(pre.with_suffix(".json").read_text())["delta_mw"],
                dtype=torch.float32,
                device=DEVICE,
            )
            curve = AdaptedCurve(adapter_on=ARMS[arm]).to(DEVICE)
            epochs, best, delta = _train(model=model, curve=curve, tr=rows_tr, y=y, r4=r4, after=after, p=p, w=w, is_val=is_val, mean=mean, std=std, tensors=tensors, inc=inc, seed=seed, init_delta=init)  # fmt: skip
            with torch.no_grad():
                cap = net.capacity_mw(
                    t=tensors, delta=delta, p=p_test, r4=f32(tblk.r4), after=f32(tblk.after)
                )
                u_te = (
                    torch.cat(
                        [
                            curve(rows_te.take(i))
                            for i in torch.arange(len(p_test), device=DEVICE).split(65_536)
                        ]
                    )
                    * inc_te
                )
            q = net.forecast(model=model, x=tblk.x, p=tblk.primary, mean=mean, std=std, subtract=(cap * u_te).cpu(), device=DEVICE)  # fmt: skip
            print(
                f"{'SMOKE ' if smoke else ''}{cutoff} {month} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(tr.x):,} rows",
                flush=True,
            )
            if smoke:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "curve": curve.state_dict()}, out.with_suffix(".tmp")
            )
            out.with_suffix(".tmp").rename(out)
            w_mix = curve.weights().detach().cpu().numpy()
            tmp = target.with_name(target.name + ".tmp.npz")
            np.savez(tmp, q=q.astype(np.float32), mix=w_mix, delta_mw=delta.cpu().numpy())
            tmp.rename(target)
            out.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "epochs": epochs,
                        "val_nll": best,
                        "seconds": round(time.monotonic() - started),
                        "rows": len(tr.x),
                    }
                )
            )
            del model, curve
    del rows_tr, rows_te
    torch.cuda.empty_cache()


def main() -> None:
    """Run the adapter fine-tunes of one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit",))
    parser.add_argument("--cutoff", required=True, choices=fit.CUTOFFS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    clean = c.clean_stack()
    for month in fit.MONTHS[a.cutoff]:
        run_month(
            cutoff=a.cutoff, month=month, arms=a.arms, seeds=a.seeds, clean=clean, smoke=a.smoke
        )
    print(
        json.dumps({"done": a.phase, "cutoff": a.cutoff, "at": datetime.now(UTC).isoformat()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
