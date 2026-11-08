# ruff: noqa: D103, E501, ANN001, ANN201
"""Experiment 647, stage 3: the encoder arms E1 to E3 (GPU).

All arms use the survival-mixture head of H1 and add K1's lag-encoder embedding `hl` (the day-lag and
week-lag outputs, 64 numbers) and a refit indicator (0: the d0 model's window, 1: the d14 model's) to
the head's inputs. One head (and, for E2 and E3, one encoder) is fitted per K1 seed; the departure
outputs of the K1 seeds are then averaged (arm names with a trailing `a`: `E1a`, `E2a`, `E3a`).

- `E1`: the embedding comes from K1's frozen lag encoder. The row's own month, refit, and K1 seed
  supply the network.
- `E2`: the head has its own copy of the lag encoder (lag MLP and lag-ID embedding trainable; the
  weather and date-time embeddings it reads, and the Primary embedding, stay frozen as in K1's
  fine-tune), initialised from the scored month's K1 model of the row's refit, trained on the
  departure loss only. The NRA input of the head is the stored K1 forecast.
- `E3`: as E2, plus an NRA branch: a copy of K1's mixer and the month's output-offset knot (initialised from the scored
  month's K1 model) that reads the detached embedding of the E2 encoder and trains on the unshifted
  observed target only (the mixture negative log-likelihood, as K1's fine-tune). The head reads this
  branch's quantiles as its NRA input. The encoder moves under the departure loss only.
- `E3p`: E3 with the head's target replaced by the unshifted observed target (the placebo).

Training rows, early stopping, optimiser, and sample sizes are those of stage 2 (`e00647_fit`); the
lag-encoder copies use a learning rate one tenth of the head's (3e-4).

Usage: python e00647_enc.py ARM MONTH VARIANT [--seeds 0 1]
"""

import argparse
import copy
import time
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import e00574_model as mod  # ty: ignore[unresolved-import]
import e00591_model as m591  # ty: ignore[unresolved-import]
import e00647_common as c647  # ty: ignore[unresolved-import]
import e00647_fit as fit  # ty: ignore[unresolved-import]
import e00647_head as h  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from torch import nn  # ty: ignore[unresolved-import]

DEV: Final = "cuda"
ENC_LR: Final[float] = 3e-4
CACHE: Final = c647.WORK / "cache"
ROOT: Final = ML_EXPERIMENTS_DATA_DIR
MONTHS: Final = c647.MONTHS
_C: dict = {}
_META: dict = {}
_K1: dict = {}


def cache(month: str) -> dict:
    if month not in _C:
        _C[month] = {k: dict(np.load(CACHE / f"{month}_{k}.npz")) for k in ("real", "bank")}
    return _C[month]


def meta(month: str) -> dict:
    if month not in _META:
        _META[month] = dict(np.load(CACHE / f"{month}_meta.npz"))
    return _META[month]


def weights(*, month: str, seed: int, day: int) -> str:
    folder = "00608-nra-combined-networks-per-class-and-leave-one-out" if seed < 2 else "00613-nra-roster-contrasts-with-intervals-five-seeds"
    return str(ROOT / folder / "arms/k1_d5/work/models/P5" / f"{month}_d{day}_s{seed}.pt")


def k1_model(*, month: str, seed: int, refit: int) -> nn.Module:
    key = (month, seed, refit)
    if key not in _K1:
        mt = meta(month)
        mdl = m591.AuxLagMdn(n_primaries=len(mt["target_mean"]), stage="ft", n_knots=int(mt["n_knots"]), n_b1=int(mt["n_b1"]), kind="shared")
        mdl.load_state_dict(torch.load(weights(month=month, seed=seed, day=14 if refit else 0), map_location="cpu"))
        mdl.eval().to(DEV)
        for p in mdl.parameters():
            p.requires_grad_(False)
        _K1[key] = mdl
    return _K1[key]


def lag_hl(*, mdl, x, prim, tab) -> torch.Tensor:
    """The network's lag encoder (activated) rebuilt from cached inputs; identical to `lag_embedding` (checked in the capture)."""
    emb = mdl.embedding(prim)
    outs = []
    for j, cols in enumerate((m560.DAY, m560.WEEK)):
        row = tab[:, j].float()
        pad = torch.zeros(len(row), mdl.n_b1, device=row.device)
        hw = nn.functional.gelu(mdl.weather(torch.cat([row[:, :7], emb, pad], dim=-1)))
        hd = nn.functional.gelu(mdl.datetime(torch.cat([row[:, 7:], emb], dim=-1)))
        parts = [x[:, cols], hw, hd, emb, mdl.lag_id(torch.full((len(x),), j, dtype=torch.long, device=x.device))]
        outs.append(mdl.lag[0](torch.cat(parts, dim=-1)))
    return nn.functional.gelu(torch.cat(outs, dim=-1))


class LagPath(nn.Module):
    """A trainable copy of K1's lag encoder; the embeddings and the weather and date-time networks it reads are frozen."""

    def __init__(self, k1: nn.Module) -> None:
        super().__init__()
        self.embedding, self.weather, self.datetime = (copy.deepcopy(k1.embedding), copy.deepcopy(k1.weather), copy.deepcopy(k1.datetime))
        self.lag = nn.ModuleList([copy.deepcopy(k1.lag[0])])
        self.lag_id = copy.deepcopy(k1.lag_id)
        self.n_b1 = k1.n_b1
        for mm in (self.embedding, self.weather, self.datetime):
            for p in mm.parameters():
                p.requires_grad_(False)
        for mm in (self.lag, self.lag_id):
            for p in mm.parameters():
                p.requires_grad_(True)

    def train(self, mode: bool = True):  # noqa: ANN201
        super().train(mode)
        for mm in (self.embedding, self.weather, self.datetime):
            mm.eval()
        return self

    def forward(self, x, prim, tab):
        return lag_hl(mdl=self, x=x, prim=prim, tab=tab)


def nra_forward(*, mdl, x, prim, hl) -> torch.Tensor:
    """K1's forward with the embedding supplied (the stage-ft network without a detector)."""
    emb = mdl.embedding(prim)
    weather_in = [x[:, m559.WEATHER_COLS], emb]
    if mdl.n_b1:
        weather_in.append(x[:, mod.b1_cols(n_b1=mdl.n_b1)])
    hw = nn.functional.gelu(mdl.weather(torch.cat(weather_in, dim=-1)))
    hd = nn.functional.gelu(mdl.datetime(torch.cat([x[:, m559.DATETIME_COLS], emb], dim=-1)))
    core = torch.cat([hw, hd, hl, emb], dim=-1)
    out = mdl.mixer_ft(torch.cat([core, x[:, m559.SPREAD_COLS], x[:, m559.LEAD_COLS]], dim=-1))
    off = mdl.offset(x[:, mod.TAU], prim)
    k = m559.COMPONENTS
    return torch.cat([out[:, :k], out[:, k : 2 * k] + off, out[:, 2 * k :]], dim=-1)


class NraBranch(nn.Module):
    """A copy of K1's mixer and offset (the rest frozen), reading a supplied embedding."""

    def __init__(self, k1: nn.Module) -> None:
        super().__init__()
        self.k1 = copy.deepcopy(k1)
        for p in self.k1.parameters():
            p.requires_grad_(False)
        for p in (*self.k1.mixer_ft.parameters(), self.k1.knot_new):
            p.requires_grad_(True)

    def train(self, mode: bool = True):  # noqa: ANN201
        super().train(mode)
        for mm in (self.k1.embedding, self.k1.weather, self.k1.datetime):
            mm.eval()
        return self

    def forward(self, x, prim, hl):
        return nra_forward(mdl=self.k1, x=x, prim=prim, hl=hl)


class Rows:
    """Rows of a set of (month, kind, index) parts on the GPU: head inputs, K1 inputs, targets."""

    def __init__(self, *, parts: list, stats: tuple | None, arm_stats: tuple | None = None) -> None:
        pool = fit.Pool(parts=parts)
        x_org, age_org, self.stats = fit.make_inputs(raw=pool.raw, arm=fit.ARMS["H1"], stats=stats, seed=0)
        self.n = len(pool.y)
        xs, ps, rs, tb, yobs, mu, sd, mo = [], [], [], [], [], [], [], []
        self.parts = parts
        for month, kind, idx in parts:
            cc = cache(month)[kind]
            xs.append(cc["x"][idx])
            ps.append(cc["prim"][idx].astype(np.int64))
            rs.append(cc["refit"][idx].astype(np.int64))
            tb.append(cc["tab"][idx])
            z = fit.load(month=month)[kind]
            yobs.append(z["y"][idx].astype(np.float32))
            mt = meta(month)
            mu.append(mt["target_mean"][cc["prim"][idx]].astype(np.float32))
            sd.append(mt["target_std"][cc["prim"][idx]].astype(np.float32))
            mo.append(np.full(len(idx), MONTHS.index(month), np.int64))
        t = lambda a: torch.from_numpy(np.concatenate(a)).to(DEV)  # noqa: E731
        self.x, self.prim, self.refit, self.tab = t(xs), t(ps), t(rs), t(tb)
        self.yobs, self.mu, self.sd, self.month = t(yobs), t(mu), t(sd), t(mo)
        o = torch.from_numpy(pool.org).to(DEV)
        self.xorg = torch.from_numpy(x_org).to(DEV)[o]
        self.age = torch.from_numpy(age_org).to(DEV)[o]
        self.q, self.y, self.lead = (torch.from_numpy(a).to(DEV) for a in (pool.q, pool.y, pool.lead))


def row_inputs(*, q: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
    return torch.stack([lead / 240.0, (q[:, 5] - q[:, 1]) * 10.0, (q[:, 9] - q[:, 5]) * 10.0], dim=1)


class Net(nn.Module):
    """Head, encoders, and (E3) NRA branches for one scored month and K1 seed."""

    def __init__(self, *, arm: str, month: str, seed: int, hl_stats: tuple | None, n_in: int) -> None:
        super().__init__()
        self.arm, self.month, self.seed = arm, month, seed
        self.head = h.Head(n_in=n_in, kind="mix").to(DEV)
        self.hl_stats = hl_stats
        self.paths = nn.ModuleList()
        self.branches = nn.ModuleList()
        if arm in {"E2", "E3", "E3p"}:
            self.paths = nn.ModuleList([LagPath(k1_model(month=month, seed=seed, refit=r)) for r in (0, 1)]).to(DEV)
        if arm in {"E3", "E3p"}:
            self.branches = nn.ModuleList([NraBranch(k1_model(month=month, seed=seed, refit=r)) for r in (0, 1)]).to(DEV)

    def frozen_hl(self, rows: Rows, sel: torch.Tensor) -> torch.Tensor:
        """E1: the embedding of each row's own month, refit, and seed (no gradient)."""
        out = torch.empty(len(sel), 64, device=DEV)
        mo, rf = rows.month[sel], rows.refit[sel]
        with torch.no_grad():
            for mi, month in enumerate(MONTHS):
                for r in (0, 1):
                    w = (mo == mi) & (rf == r)
                    if w.any():
                        s = sel[w]
                        out[w] = lag_hl(mdl=k1_model(month=month, seed=self.seed, refit=r), x=rows.x[s], prim=rows.prim[s], tab=rows.tab[s])
        return out

    def hl(self, rows: Rows, sel: torch.Tensor) -> torch.Tensor:
        if self.arm == "E1":
            return self.frozen_hl(rows, sel)
        rf = rows.refit[sel]
        outs = [self.paths[r](rows.x[sel], rows.prim[sel], rows.tab[sel]) for r in (0, 1)]
        return torch.where((rf == 1)[:, None], outs[1], outs[0])

    def nra(self, rows: Rows, sel: torch.Tensor, hl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """E3: (raw outputs, quantiles in fraction of capacity) of the NRA branch reading the detached embedding."""
        rf = rows.refit[sel]
        hd = hl.detach()
        outs = [self.branches[r](rows.x[sel], rows.prim[sel], hd) for r in (0, 1)]
        out = torch.where((rf == 1)[:, None], outs[1], outs[0])
        with torch.no_grad():
            log_w, mean, std = base.mixture_parts(out=out.double(), components=m559.COMPONENTS)
            q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std).float()
            q = torch.sort(q * rows.sd[sel][:, None] + rows.mu[sel][:, None], dim=1).values
        return out, q

    def forward(self, rows: Rows, sel: torch.Tensor, *, short: float = 0.5):
        hl = self.hl(rows, sel)
        if self.arm in {"E3", "E3p"}:
            nra_out, q = self.nra(rows, sel, hl)
        else:
            nra_out, q = None, rows.q[sel]
        lead = rows.lead[sel]
        hln = (hl - self.hl_stats[0]) / self.hl_stats[1]
        x = torch.cat([rows.xorg[sel], row_inputs(q=q, lead=lead), hln, rows.refit[sel][:, None].float()], dim=1)
        qq, d, s = self.head(x=x, q=q, age=rows.age[sel], lead_h=lead, short_weight=short)
        return qq, d, s, nra_out, q


def initial_hl_stats(*, arm: str, rows: Rows, month: str, seed: int) -> tuple:
    net = Net(arm="E1" if arm == "E1" else "E2", month=month, seed=seed, hl_stats=(0.0, 1.0), n_in=1)
    outs = []
    with torch.no_grad():
        for a in range(0, rows.n, 50_000):
            sel = torch.arange(a, min(a + 50_000, rows.n), device=DEV)
            outs.append(net.hl(rows, sel))
    hl = torch.cat(outs)
    return hl.mean(0), hl.std(0).clamp(min=1e-3)


def target_of(*, arm: str, rows: Rows, sel: torch.Tensor) -> torch.Tensor:
    return rows.yobs[sel] if arm == "E3p" else rows.y[sel]


def losses(*, net: Net, rows: Rows, sel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    qq, _d, _s, nra_out, _q = net(rows, sel)
    head_loss = h.pinball(y=target_of(arm=net.arm, rows=rows, sel=sel), q=qq).mean()
    nra_loss = None
    if nra_out is not None:
        z = (rows.yobs[sel] - rows.mu[sel]) / rows.sd[sel]
        nra_loss = base.mixture_nll(out=nra_out, z=z, components=m559.COMPONENTS).mean()
    return head_loss, nra_loss


def train(*, net: Net, tr: Rows, va: Rows, seed: int) -> tuple[list, float]:
    groups = [{"params": list(net.head.parameters()), "lr": fit.LR, "weight_decay": fit.WD}]
    enc = [p for pth in net.paths for p in pth.parameters() if p.requires_grad]
    if enc:
        groups.append({"params": enc, "lr": ENC_LR, "weight_decay": 0.0})
    br = [p for b in net.branches for p in b.parameters() if p.requires_grad]
    if br:
        groups.append({"params": br, "lr": ENC_LR, "weight_decay": 0.0})
    opt = torch.optim.Adam(groups)
    best, best_state, bad, log = float("inf"), None, 0, []
    for epoch in range(fit.EPOCHS):
        net.train()
        perm = torch.from_numpy(np.random.default_rng(seed * 100 + epoch).permutation(tr.n)).to(DEV)
        for a in range(0, tr.n, fit.BATCH):
            sel = perm[a : a + fit.BATCH]
            hl_, nl = losses(net=net, rows=tr, sel=sel)
            loss = hl_ + (nl if nl is not None else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        v, vn = 0.0, 0.0
        with torch.no_grad():
            for a in range(0, va.n, 20_000):
                sel = torch.arange(a, min(a + 20_000, va.n), device=DEV)
                hl_, nl = losses(net=net, rows=va, sel=sel)
                v += float(hl_) * len(sel)
                vn += float(nl) * len(sel) if nl is not None else 0.0
        v /= va.n
        log.append((v, vn / va.n))
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= fit.PATIENCE:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    return log, best


def predict(*, net: Net, rows: Rows) -> dict[str, np.ndarray]:
    qs, ds, ss, nq = [], [], [], []
    with torch.no_grad():
        for a in range(0, rows.n, 50_000):
            sel = torch.arange(a, min(a + 50_000, rows.n), device=DEV)
            qq, d, s, nra_out, q = net(rows, sel)
            qs.append(qq.cpu().numpy())
            ds.append(d.cpu().numpy())
            ss.append(s.cpu().numpy())
            nq.append(q.cpu().numpy())
    return {"q": np.concatenate(qs), "d": np.concatenate(ds), "sigma": np.concatenate(ss), "q_nra": np.concatenate(nq)}


def training_parts(*, month: str, variant: str, rng: np.random.Generator) -> tuple[list, list]:
    """The same row samples as stage 2's `fit_one`."""
    import e00647_data as dat  # ty: ignore[unresolved-import]

    others = [m for m in MONTHS if m != month]
    if variant == "lomo":
        tr = [(m, k, i) for m in others[1:] for k, i in fit.sample(month=m, n_real=fit.N_REAL, n_bank=fit.N_BANK, rng=rng)]
        va = [(others[0], k, i) for k, i in fit.sample(month=others[0], n_real=fit.N_VAL, n_bank=fit.N_VAL_BANK, rng=rng)]
        return tr, va
    earlier = [m for m in MONTHS if m < month]
    assert earlier, "no earlier month to train on"
    tr, va = [], []
    for m in earlier:
        d = fit.load(month=m)
        ms = dat.month_start_idx(month=m)
        for k, n_tr, n_va in (("real", fit.N_REAL, fit.N_VAL), ("bank", fit.N_BANK, fit.N_VAL_BANK)):
            dom = (d[k]["oi"] - ms) // 48
            if variant == "fc4":  # every fourth origin day held out for the early stop, so both refit windows are trained
                for sel_, n, dst in ((dom % 4 != 3, n_tr * 3, tr), (dom % 4 == 3, n_va, va)):
                    idx = np.flatnonzero(d[k]["train_ok"] & sel_)
                    dst.append((m, k, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))))
                continue
            for lo, hi, n, dst in ((0, 14, n_tr, tr), (21, 99, n_va, va)):
                idx = np.flatnonzero(d[k]["train_ok"] & (dom >= lo) & (dom < hi))
                dst.append((m, k, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))))
    return tr, va


def run_one(*, arm: str, month: str, variant: str, seed: int) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    tp, vp = training_parts(month=month, variant=variant, rng=rng)
    tr = Rows(parts=tp, stats=None)
    va = Rows(parts=vp, stats=tr.stats)
    hl_stats = initial_hl_stats(arm=arm, rows=tr, month=month, seed=seed)
    n_in = tr.xorg.shape[1] + 3 + 64 + 1
    torch.manual_seed(seed)
    net = Net(arm=arm, month=month, seed=seed, hl_stats=hl_stats, n_in=n_in)
    t_fit = time.time()
    log, best = train(net=net, tr=tr, va=va, seed=seed)
    t_fit = time.time() - t_fit
    res: dict = {}
    for kind in ("real", "bank"):
        z = fit.load(month=month)[kind]
        idx = np.flatnonzero(z["usable"])
        te = Rows(parts=[(month, kind, idx)], stats=tr.stats)
        out = predict(net=net, rows=te)
        out["idx"] = idx.astype(np.int32)
        out["y"] = te.y.cpu().numpy()
        out["yobs"] = te.yobs.cpu().numpy()
        res[kind] = out
        del te
    print(f"  {arm} {month} {variant} s{seed}: {len(log)} epochs, best val {best:.5f}, fit {t_fit:.0f}s, total {time.time() - t0:.0f}s", flush=True)
    return {"res": res, "log": np.array(log), "n_train": tr.n, "n_val": va.n, "fit_seconds": t_fit, "seconds": time.time() - t0}


def write(*, name: str, out: dict, q_key: str, d: dict, extra: dict) -> None:
    out_dir = fit.data_dir() / "fits"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrs: dict = {}
    for kind in ("real", "bank"):
        r = out[kind]
        y = r["y"] if q_key == "q" else r["yobs"]
        m = fit.metrics(y=y, q=r[q_key])
        arrs[f"{kind}_idx"] = r["idx"]
        arrs[f"{kind}_d"] = d[kind]["d"].astype(np.float32)
        arrs[f"{kind}_sigma"] = d[kind]["sigma"].astype(np.float32)
        for k, v in m.items():
            arrs[f"{kind}_{k}"] = v
    np.savez(out_dir / f"{name}.npz", **arrs, **extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=("E1", "E2", "E3", "E3p"))
    ap.add_argument("month")
    ap.add_argument("variant", choices=("lomo", "fc", "fc4"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    a = ap.parse_args()
    t_start = time.time()
    print(f"START {a.arm} {a.month} {a.variant} seeds {a.seeds} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    per_seed = {}
    for seed in a.seeds:
        per_seed[seed] = run_one(arm=a.arm, month=a.month, variant=a.variant, seed=seed)
        r = per_seed[seed]
        write(
            name=f"{a.arm}_{a.month}_{a.variant}_s{seed}", out=r["res"], q_key="q", d=r["res"],
            extra={"val_log": r["log"], "n_train": r["n_train"], "n_val": r["n_val"], "seconds": r["seconds"], "fit_seconds": r["fit_seconds"]},
        )
        if a.arm in {"E3", "E3p"}:
            write(
                name=f"{a.arm}nra_{a.month}_{a.variant}_k{seed}",
                out={k: {**v, "q": v["q_nra"]} for k, v in r["res"].items()}, q_key="q_nra",
                d={k: {"d": np.zeros_like(v["d"]), "sigma": np.zeros_like(v["sigma"])} for k, v in r["res"].items()}, extra={},
            )
    # average the per-seed outputs: quantiles and departures
    avg: dict = {}
    for kind in ("real", "bank"):
        rs = [per_seed[s]["res"][kind] for s in a.seeds]
        avg[kind] = {k: np.mean([r[k] for r in rs], axis=0) if k in {"q", "d", "sigma", "q_nra"} else rs[0][k] for k in rs[0]}
    write(name=f"{a.arm}a_{a.month}_{a.variant}_s0", out=avg, q_key="q", d=avg, extra={"seeds": np.array(a.seeds)})
    if a.arm in {"E3", "E3p"}:
        write(
            name=f"{a.arm}nraa_{a.month}_{a.variant}_k0", out={k: {**v, "q": v["q_nra"]} for k, v in avg.items()}, q_key="q_nra",
            d={k: {"d": np.zeros_like(v["d"]), "sigma": np.zeros_like(v["sigma"])} for k, v in avg.items()}, extra={},
        )
        np.savez(fit.data_dir() / "fits" / f"{a.arm}nrarows_{a.month}_{a.variant}.npz", **{f"{k}_{s}": per_seed[s]["res"][k]["q_nra"] for k in ("real", "bank") for s in a.seeds})
    print(f"DONE {a.arm} {a.month} {a.variant} in {time.time() - t_start:.0f}s at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)


if __name__ == "__main__":
    main()
