# ruff: noqa: ANN001, ANN201, E402, D103, E501, ANN003, PLR0915, D102, D107, ANN201, ANN202, PT018, F841, PERF401, B007, RUF100
"""Experiment 654, step 1: fine-tune K1's network jointly with a prevailing head, and forecast
(GPU).

K1 is experiment 613's `k1_d5` network (experiment 591's A1 chain: shared lag encoder with lag
ID, D5 offset). This script replaces K1's daily fine-tune with one fine-tune that, as in K1,
starts from the fold's step-2 weights and trains on the rows before the cut (embargo 7 days,
newest 7 days at half weight), and adds three things:

1. The lags of the training rows carry injected shifts drawn by experiment 650's generator
   (training family, constant or ramped shape, receivers ignored), using a draw key that no
   scoring pass shares; the NRA target stays unshifted.
2. A prevailing head reads the lag-encoder outputs (detached in arm J1, not detached in arm J2),
   the Primary embedding, and the lead columns, and outputs a depth, a spread, an in-force logit,
   and ten daily hazards. The prevailing quantiles are `mixture_quantiles` of experiment 647 on
   the NRA quantiles, as in experiment 652.
3. The loss is the NRA mixture negative log-likelihood plus `LAM_PIN` times the prevailing
   pinball loss plus `LAM_BCE` times the binary cross-entropy of the shift-still-on probability.

Arms: `C0` (no head, no prevailing loss: K1 fine-tuned the same way, the matched-compute
control), `J1` (head reads detached encoder outputs, so the prevailing loss cannot change the NRA
output), and `J2` (the prevailing loss reaches the lag encoder).

Cuts: day 0 and day 14 of the scored month, windows [0, 14) and the rest of the month. Output per
(arm, month, cut, seed): `fits/<arm>_<month>_d<day>_s<seed>.npz` with, for the real, held-out,
and training-family row sets of experiment 650's head data, the row index, the NRA quantiles, the
prevailing quantiles (fractions of capacity), and the survival probability.

Usage: python e00654_fit.py ARM [--months ...] [--seeds ...] [--cuts 0 14]
"""

import argparse
import sys
import time
from datetime import timedelta

_p = argparse.ArgumentParser()
_p.add_argument("arm", choices=("C0", "J1", "J2"))
_p.add_argument("--months", nargs="+", default=["2025-10"])
_p.add_argument("--seeds", nargs="+", type=int, default=[0])
_p.add_argument("--cuts", nargs="+", type=int, default=[0, 14])
_p.add_argument("--max-epochs", type=int, default=None)
MINE = _p.parse_args()
assert max(MINE.months) <= "2026-06", "July 2026 onwards is never opened"
assert set(MINE.seeds) <= {0, 1}, "step-2 weights exist for seeds 0 and 1 only"
assert set(MINE.cuts) <= {0, 14}
sys.argv = [
    "e00591_arm.py",
    "A1",
    "fit",
    "--months",
    *MINE.months,
    "--seeds",
    *map(str, MINE.seeds),
]

import e00545_model as m545  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00591_arm as arm  # ty: ignore[unresolved-import]
import e00647_head as h647  # ty: ignore[unresolved-import]
import e00650_bank as gen  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00654_common as c654  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

e = arm.e
assert arm.LAMBDA == 0.0
LAM_PIN = 20.0
"""Weight of the prevailing pinball loss (capacity-fraction units, about 0.03) against the NLL (about 1)."""
LAM_BCE = 0.3
EMBARGO_DAYS, DOWN_WEIGHT, DOWN_WEIGHT_DAYS = 7, 0.5, 7
HISTORY_KEEP = 0.15
"""Share of (history month, Primary) cells that receive an injected shift in the training rows."""
HISTORY_OFFSET = 1_000_000
"""Added to the month index in the draw key, so no training draw equals a scoring draw."""
N_DAYS = 10
HAZARD_OFFSET = 3.0
NRA_LAYERS = 64
EXTRA: dict[int, dict[str, np.ndarray]] = {}
OUT = c654.WORK
e.WORK, e.W = OUT, OUT / "work"
e.EMBARGO_DAYS = EMBARGO_DAYS
_select_rows = e.c.select_rows


def select_rows(**kwargs):
    """Experiment 613's E3 rows: the newest 7 days left after the embargo get half weight."""
    refit = _select_rows(**kwargs)
    rows, cutoff = kwargs["rows"], kwargs["cutoff"]
    age_days = (e.c.us(t=cutoff) - rows.valid_us[refit.index]) / e.c.DAY_US
    window = (age_days >= EMBARGO_DAYS) & (age_days < EMBARGO_DAYS + DOWN_WEIGHT_DAYS)
    weight = np.where(window, refit.weight * DOWN_WEIGHT, refit.weight).astype(np.float32)
    return e.c.Refit(
        refit.index, refit.y, weight, refit.is_val, {**refit.report, "embargo_days": EMBARGO_DAYS}
    )


e.c.select_rows = select_rows


# ----------------------------------------------------------------------------- training rows


def history_shifts(*, clean: np.ndarray, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (own, on): injected shifts as shares of capacity and an in-force flag, shape (16,
    grid).

    One draw per (history month, Primary) from experiment 650's training family, kept with
    probability `HISTORY_KEEP`; constant or ramped shape (a proportional shape would read the
    event's own future power); a Primary with only non-negative power stays non-negative.
    """
    nb_of, _cap, is_mw = c650.world()
    n_prim, n = clean.shape
    own = np.zeros((n_prim, n), np.float32)
    on = np.zeros((n_prim, n), bool)
    first_month = first - 10 * 48
    ms = (first_month // (48 * 28)) * 48 * 28
    k = 0
    while ms < last:
        r = k % gen.N_TRAIN_PASSES
        rng = np.random.default_rng([654, ms, r])
        for p in range(n_prim):
            ev = gen.draw(
                month_idx=ms + HISTORY_OFFSET, r=r, primary=p, nb_of=nb_of, month_start_idx=ms,
                window_end_idx=10**9, is_mw=is_mw,
            )  # fmt: skip
            if rng.random() > HISTORY_KEEP or ev.end_idx <= ev.onset_idx:
                continue
            env = gen._profile(n=n, onset=ev.onset_idx, ramp=ev.ramp_half_hours, end=ev.end_idx)
            delta = (ev.sign * ev.depth * env).astype(np.float32)
            if not bool(is_mw[p]):
                delta = np.where(
                    np.isfinite(clean[p]), np.maximum(clean[p] + delta, 0.0) - clean[p], delta
                ).astype(np.float32)
            own[p] = np.clip(own[p] + delta, -0.5, 0.5)
            on[p, max(ev.onset_idx, 0) : max(ev.end_idx, 0)] = True
        ms += 48 * 28
        k += 1
    return own, on


_build_month_p5 = e.k.build_month


def build_month_e2e(*, month, clean):
    """Experiment 591's synthetic-lag builder, with this experiment's shifts and prevailing labels."""
    mr = _build_month_p5(month=month, clean=clean)
    origin_idx = (mr.origin_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
    first, last = int(mr.target_idx.min()), int(mr.target_idx.max())
    own, on = history_shifts(clean=clean, first=first, last=last)
    fixed = np.where(np.isfinite(clean), clean + own, clean).astype(clean.dtype)
    blk = e.block_of(
        clean=fixed, primary=mr.primary, target_idx=mr.target_idx, origin_idx=origin_idx
    )
    for old, new in ((38, 1), (39, 2), (41, 4), (42, 5), (43, 6), (44, 7)):
        assert np.array_equal(mr.x[:, old], blk[:, new]), (month, old)
    mr.x[:, 37:45] = blk
    prim, tgt = mr.primary, mr.target_idx
    EXTRA[id(mr)] = {
        "sh": own[prim, tgt].astype(np.float32),
        "on_o": on[prim, np.clip(origin_idx, 0, on.shape[1] - 1)],
        "on_v": on[prim, tgt],
        "lead_h": ((mr.valid_us - mr.origin_us) / 3.6e9).astype(np.float32),
    }
    print(
        f"{month}: share of rows with a shift in force at the valid time {EXTRA[id(mr)]['on_v'].mean():.3f}",
        flush=True,
    )
    return mr


class JointGpuMonth(e.LagGpuMonth):
    """The month's rows on the GPU; the fine-tuning tensors carry the prevailing labels."""

    def refit_tensors(self, *, refit, shift_z=None):
        t = super().refit_tensors(refit=refit, shift_z=shift_z)
        ex = EXTRA[id(self.mr)]
        for key, dtype in (
            ("sh", torch.float32),
            ("on_o", torch.float32),
            ("on_v", torch.float32),
            ("lead_h", torch.float32),
        ):
            t[key] = torch.from_numpy(ex[key][refit.index]).to(dtype).to(self.device)
        t["mu"] = torch.from_numpy(
            self.mr.base.target_mean[self.mr.primary[refit.index]].astype(np.float32)
        ).to(self.device)
        t["sd"] = torch.from_numpy(
            self.mr.base.target_std[self.mr.primary[refit.index]].astype(np.float32)
        ).to(self.device)
        return t


# ----------------------------------------------------------------------------- network


class Joint(nn.Module):
    """K1's network plus a prevailing head that reads the lag-encoder outputs."""

    def __init__(self, *, net: nn.Module, with_head: bool, detach: bool) -> None:
        super().__init__()
        self.net, self.with_head, self.detach = net, with_head, detach
        self._hl = None
        orig = net.lag_embedding

        def wrapped(**kw):
            h = orig(**kw)
            self._hl = h
            return h

        net.lag_embedding = wrapped
        if with_head:
            n_in = 2 * m560.LAG_OUT + m545.EMBEDDING + 2
            self.head = nn.Sequential(
                nn.Linear(n_in, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3 + N_DAYS),
            )
            with torch.no_grad():
                self.head[-1].weight.mul_(0.1)
                self.head[-1].bias.zero_()
                self.head[-1].bias[2] = -1.0

    def forward(self, x, p, lead_h=None, mu=None, sd=None):
        out = self.net(x, p)
        if not self.with_head or lead_h is None:
            return out, None, None
        hl = nn.functional.gelu(self._hl)
        emb = self.net.embedding(p)
        if self.detach:
            hl, emb = hl.detach(), emb.detach()
        o = self.head(torch.cat([hl, emb, x[:, e.m.LEAD_COLS]], dim=-1))
        d = h647.DEPTH_SCALE * torch.tanh(o[:, 0])
        sigma = h647.SIGMA_FLOOR + h647.SIGMA_SCALE * nn.functional.softplus(o[:, 1])
        haz = nn.functional.softplus(o[:, 3:] - HAZARD_OFFSET)
        days = torch.arange(N_DAYS, dtype=lead_h.dtype, device=lead_h.device)[None, :]
        share = torch.clamp(lead_h[:, None] / 24.0 - days, 0.0, 1.0)
        pr = torch.sigmoid(o[:, 2]) * torch.exp(-(haz * share).sum(dim=1))
        with torch.no_grad():
            log_w, mean, std = e.base.mixture_parts(
                out=out.detach().double(), components=e.m.COMPONENTS
            )
            qz = e.base.mixture_quantiles(log_w=log_w, mean=mean, std=std).float()
            qf = torch.sort(qz * sd[:, None] + mu[:, None], dim=1).values
        qprev = h647.mixture_quantiles(q=qf, d=d, sigma=sigma, p=pr)
        return out, qprev, pr

    def forecast(self, *, x, p, lead_h, mu, sd):
        """Return (NRA quantiles, prevailing quantiles, survival probability), capacity fractions."""
        self.eval()
        with torch.no_grad():
            out, qprev, pr = self(x, p, lead_h, mu, sd)
            log_w, mean, std = e.base.mixture_parts(out=out.double(), components=e.m.COMPONENTS)
            qz = e.base.mixture_quantiles(log_w=log_w, mean=mean, std=std).float()
            qn = torch.sort(qz * sd[:, None] + mu[:, None], dim=1).values
        return qn, qprev, pr


def batch_loss(*, model: Joint, t: dict, b) -> torch.Tensor:
    out, qprev, pr = model(t["x"][b], t["p"][b], t["lead_h"][b], t["mu"][b], t["sd"][b])
    w = t["w"][b]
    nll = e.row_nll(out=out, z=t["z"][b], p=t["p"][b])
    loss = (w * nll).sum() / w.sum()
    if model.with_head:
        y_prev = t["z"][b] * t["sd"][b] + t["mu"][b] + t["sh"][b]
        pin = h647.pinball(y=y_prev, q=qprev)
        label = t["on_o"][b] * t["on_v"][b]
        bce = nn.functional.binary_cross_entropy(
            torch.clamp(pr, 1e-4, 1 - 1e-4), label, reduction="none"
        )
        loss = loss + LAM_PIN * (w * pin).sum() / w.sum() + LAM_BCE * (w * bce).sum() / w.sum()
    return loss


def train_joint(*, model: Joint, t: dict, seed: int, device: str, max_epochs: int):
    """Experiment 574's loop (frozen weather and datetime encoders, AdamW) on the joint loss."""
    torch.manual_seed(seed)
    c = e.c
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    gen_ = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        val_idx = val_idx[
            torch.randperm(len(val_idx), generator=gen_)[: c.VALIDATION_ROWS].to(device)
        ]
    net = model.net
    enc = {id(q) for part in (net.weather, net.datetime) for q in part.parameters()}
    for part in (net.weather, net.datetime):
        for q in part.parameters():
            q.requires_grad_(False)
    params = [q for q in model.parameters() if id(q) not in enc]
    assert all(
        not q.requires_grad for part in (net.weather, net.datetime) for q in part.parameters()
    )
    opt = torch.optim.AdamW(params, lr=e.SPEC.lr, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        net.weather.eval()
        net.datetime.eval()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for s in range(0, len(order), c.BATCH):
            loss = batch_loss(model=model, t=t, b=order[s : s + c.BATCH])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), c.GRAD_CLIP)
            if model.with_head:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            parts = val_idx.split(c.BATCH * 2)
            val = float(
                sum(float(batch_loss(model=model, t=t, b=i)) * len(i) for i in parts) / len(val_idx)
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


# ----------------------------------------------------------------------------- forecasting


def key_of(prim, oi, ti):
    return (prim.astype(np.int64) * 4_000_000 + oi.astype(np.int64)) * 1024 + (ti - oi).astype(
        np.int64
    )


def forecast_rows(
    *, model: Joint, x: np.ndarray, prim: np.ndarray, lead_h: np.ndarray, mr, device: str
):
    if len(x) == 0:
        return np.zeros((0, 11), np.float32), np.zeros((0, 11), np.float32), np.zeros(0, np.float32)
    nq, pq, pp = [], [], []
    mu_all = torch.from_numpy(mr.base.target_mean.astype(np.float32)).to(device)
    sd_all = torch.from_numpy(mr.base.target_std.astype(np.float32)).to(device)
    for a in range(0, len(x), 65_536):
        sl = slice(a, a + 65_536)
        xb = torch.from_numpy(x[sl]).to(device)
        pb = torch.from_numpy(prim[sl]).to(device)
        n, q, p = model.forecast(
            x=xb,
            p=pb,
            lead_h=torch.from_numpy(lead_h[sl].astype(np.float32)).to(device),
            mu=mu_all[pb],
            sd=sd_all[pb],
        )
        nq.append(n.cpu().numpy())
        pq.append(q.cpu().numpy() if q is not None else np.zeros_like(n.cpu().numpy()))
        pp.append(p.cpu().numpy() if p is not None else np.zeros(len(n), np.float32))
    return np.concatenate(nq), np.concatenate(pq), np.concatenate(pp)


def main() -> None:
    device = "cuda"
    e.k.build_month = build_month_e2e
    e.k.GpuMonth = JointGpuMonth
    clean = e.c.clean_stack()
    max_epochs = MINE.max_epochs or e.SPEC.max_epochs
    (OUT / "fits").mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for month in MINE.months:
        mr = e.k.build_month(month=month, clean=clean)
        test = e.data.build_test(month=month, standardiser=mr.base.standardiser)
        g = e.k.GpuMonth(mr=mr, device=device)
        e.CTX.update(on=False, variants={}, seed=0)
        start = e.c.month_start(month=month)
        start_idx = (e.c.us(t=start) - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
        src = f650.sources(month=month, bank="new")
        sets = {
            "real": (src["real"], np.flatnonzero(src["real"]["usable"]), None),
            "held": (
                src["bank"],
                np.flatnonzero(src["bank"]["usable"] & (src["bank"]["pass_"] >= 200)),
                "bank",
            ),
            "fam": (
                src["bank"],
                np.flatnonzero(src["bank"]["usable"] & (src["bank"]["pass_"] < 200)),
                "bank",
            ),
        }
        tkey = key_of(test.primary, test.origin_idx, test.target_idx)
        order = np.argsort(tkey)
        tkey_sorted = tkey[order]
        lead_test = ((test.valid - test.origin) / 3.6e9).astype(np.float32)
        for seed in MINE.seeds:
            for day in MINE.cuts:
                out = OUT / "fits" / f"{MINE.arm}_{month}_d{day}_s{seed}.npz"
                if out.exists():
                    continue
                t0 = time.monotonic()
                cut = start + timedelta(days=day)
                refit = e.c.select_rows(rows=mr, cutoff=cut, half_life_days=e.SPEC.half_life)
                assert (
                    refit.report["rows_past_valid_cut"] == 0
                    and refit.report["rows_past_origin_cut"] == 0
                )
                refit = e.embargoed(mr=mr, refit=refit, cut=cut)
                e.CTX.update(seed=seed, cut=cut)
                tensors = g.refit_tensors(refit=refit)
                net = e.new_model(n_primaries=len(mr.base.target_mean)).to(device)
                net.load_state_dict(
                    torch.load(e.step2_file(month=month, seed=seed), map_location="cpu")
                )
                model = Joint(net=net, with_head=MINE.arm != "C0", detach=MINE.arm == "J1").to(
                    device
                )
                epochs, best = train_joint(
                    model=model, t=tensors, seed=seed, device=device, max_epochs=max_epochs
                )
                torch.save(
                    model.state_dict(), OUT / "models" / f"{MINE.arm}_{month}_d{day}_s{seed}.pt"
                )
                t_train = time.monotonic() - t0
                lo_us = e.c.us(t=cut)
                hi_us = (
                    e.c.us(t=start + timedelta(days=14))
                    if day == 0
                    else e.c.us(t=e.c.month_stop(month=month))
                )
                win = (test.origin >= lo_us) & (test.origin < hi_us)
                res: dict[str, np.ndarray] = {}
                for name, (z, idx, kind) in sets.items():
                    in_win = idx[
                        (
                            z["oi"][idx].astype(np.int64)
                            >= (lo_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
                        )
                        & (
                            z["oi"][idx].astype(np.int64)
                            < (hi_us - e.c.prep422.GRID_START_US) // e.c.prep422.STEP_US
                        )
                    ]
                    bkey = key_of(z["prim"][in_win], z["oi"][in_win], z["vi"][in_win])
                    pos = np.searchsorted(tkey_sorted, bkey)
                    pos = np.clip(pos, 0, len(tkey_sorted) - 1)
                    found = tkey_sorted[pos] == bkey
                    rows = order[pos[found]]
                    assert found.mean() > 0.99, (name, month, float(found.mean()))
                    in_win = in_win[found]
                    assert win[rows].all()
                    nq = np.zeros((len(in_win), 11), np.float32)
                    pq = np.zeros_like(nq)
                    pp = np.zeros(len(in_win), np.float32)
                    if kind is None:
                        nq, pq, pp = forecast_rows(
                            model=model,
                            x=test.x[rows],
                            prim=test.primary[rows],
                            lead_h=lead_test[rows],
                            mr=mr,
                            device=device,
                        )
                    else:
                        passes = z["pass_"][in_win]
                        for r in np.unique(passes):
                            m = passes == r
                            own, _mirror, _ev = c650.shifts_for(clean=clean, month=month, r=int(r))
                            fixed = np.where(np.isfinite(clean), clean + own, clean)
                            rr = rows[m]
                            blk = e.block_of(
                                clean=fixed,
                                primary=test.primary[rr],
                                target_idx=test.target_idx[rr],
                                origin_idx=test.origin_idx[rr],
                            )
                            x = test.x[rr].copy()
                            x[:, 37:45] = blk
                            assert np.array_equal(x[:, 43:45], test.x[rr][:, 43:45]), (
                                "contamination moved a lag time"
                            )
                            n_, q_, p_ = forecast_rows(
                                model=model,
                                x=x,
                                prim=test.primary[rr],
                                lead_h=lead_test[rr],
                                mr=mr,
                                device=device,
                            )
                            nq[m], pq[m], pp[m] = n_, q_, p_
                    res[f"{name}_idx"] = in_win.astype(np.int32)
                    res[f"{name}_nra"] = nq
                    res[f"{name}_q"] = pq
                    res[f"{name}_p"] = pp
                tmp = out.with_suffix(".tmp.npz")
                np.savez(
                    tmp,
                    **res,
                    epochs=epochs,
                    best=best,
                    train_seconds=t_train,
                    n_train=int((~tensors["v"]).sum()),
                    seconds=time.monotonic() - t0,
                )  # ty: ignore[invalid-argument-type]
                tmp.rename(out)
                print(
                    f"{out.name}: {epochs} epochs, val {best:.5f}, train {t_train:.0f}s, total {time.monotonic() - t0:.0f}s",
                    flush=True,
                )
                del model, net, tensors
                torch.cuda.empty_cache()


main()
