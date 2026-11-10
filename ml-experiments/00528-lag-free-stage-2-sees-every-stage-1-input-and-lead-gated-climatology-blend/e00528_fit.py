# ruff: noqa: ANN001, ANN201, ANN204, D101, D102, D103, D107, E501, N802, PLR0915
"""Experiment 528, `e00528_fit.py`: kept for the record.

Experiment 528 (interim, unaudited, not for merge): stage 2 sees every stage-1 input.

S2-X : stage 2 = zero-initialised residual MLP (2 x 64) on [9 stage-1 outputs, all 37 input columns
       (28 weather/solar/calendar/holiday means and geometry, 7 spreads, lead, lead_sqrt), group flag,
       Primary embedding(8)]; NLL; trained on 524's cross-fitted stage-1 outputs under 510's row rule.
S2-G : stage-1 mixture (3 components) blended with a calendar-only mixture (3 components, a pooled
       MDN on the calendar, holiday and solar-geometry columns only) by a gate g(lead, 7 spreads);
       mixture weights `g * w_weather` and `(1 - g) * w_calendar`;
       gate trained by NLL, both mixtures frozen.
S2-XG: the same gate over the S2-X output.
"""

import argparse
import time

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00524_stage2 as s2  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ROOT = (
    ML_EXPERIMENTS_DATA_DIR
    / "00528-lag-free-stage-2-sees-every-stage-1-input-and-lead-gated-climatology-blend"
)
W = ROOT / "work"
XC = W / "xcache"
K = c.COMPONENTS
DEV = "cuda"
CAL = list(range(14, 35))  # cos_zenith .. nearest_inside: no weather, no spread, no lead
STD = list(range(7, 14))
LEAD_I = 35
HID = 64


class FormX(torch.nn.Module):
    def __init__(self, n_x: int, n_primaries: int):
        super().__init__()
        self.emb = torch.nn.Embedding(n_primaries, 8)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * K + n_x + 1 + 8, HID),
            torch.nn.ReLU(),
            torch.nn.Linear(HID, HID),
            torch.nn.ReLU(),
            torch.nn.Linear(HID, 3 * K),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, out, side):  # side = [x(37), group, primary]
        p = side[:, -1].long()
        return out + self.net(torch.cat([out, side[:, :-1], self.emb(p)], dim=1))


class Gate(torch.nn.Module):
    """g = sigmoid(MLP(lead / 240, 7 spreads)); both mixtures frozen."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.constant_(self.net[-1].bias, 2.0)

    def weight(self, g_in):
        return torch.sigmoid(self.net(g_in)).squeeze(1)


def blend(*, a, b, g):
    """Return the 6-component raw output of the gated blend of two 3-component raw outputs."""
    la = torch.log_softmax(a[:, :K], dim=1) + torch.log(g.clamp_min(1e-6))[:, None]
    lb = torch.log_softmax(b[:, :K], dim=1) + torch.log((1 - g).clamp_min(1e-6))[:, None]
    return torch.cat(
        [la, lb, a[:, K : 2 * K], b[:, K : 2 * K], a[:, 2 * K :], b[:, 2 * K :]], dim=1
    )


def train_gen(*, model, out, side, z, w, is_val, seed, lr=3e-3):
    torch.manual_seed(seed)
    tr, va = torch.nonzero(~is_val).squeeze(1), torch.nonzero(is_val).squeeze(1)
    g = torch.Generator().manual_seed(seed)
    va = (
        va[torch.randperm(len(va), generator=g)[: c.VALIDATION_ROWS].to(va.device)]
        if len(va) > c.VALIDATION_ROWS
        else va
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=c.WEIGHT_DECAY)
    best, st, stale = float("inf"), None, 0
    for _ in range(s2.STAGE2_EPOCHS):
        model.train()
        order = tr[torch.randperm(len(tr), device=tr.device)][: int(c.ROW_FRACTION * len(tr))]
        for a in range(0, len(order), s2.BATCH):
            b = order[a : a + s2.BATCH]
            nll = base.mixture_nll(out=model(out[b], side[b]), z=z[b], components=model.k)
            v = (w[b] * nll).sum() / w[b].sum()
            opt.zero_grad()
            v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(out[i], side[i]), z=z[i], components=model.k)
                        for i in va.split(65536)
                    ]
                ).mean()
            )
        if v < best - 1e-5:
            best, stale, st = v, 0, {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            stale += 1
            if stale >= s2.STAGE2_PATIENCE:
                break
    model.load_state_dict(st)
    model.eval()
    return model


class GateModel(torch.nn.Module):
    """Wraps the gate and the two frozen raw outputs: forward(idx-free) takes (a_out, side=[b_out, g_in])."""

    k = 2 * K

    def __init__(self):
        super().__init__()
        self.gate = Gate()

    def forward(self, a, side):
        return blend(a=a, b=side[:, : 3 * K], g=self.gate.weight(side[:, 3 * K :]))


class XModel(torch.nn.Module):
    k = K

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, out, side):
        return self.inner(out, side)


def quantile_frame(*, out, d, select, comps):
    idx = np.nonzero(select)[0]
    parts = []
    with torch.no_grad():
        for a in range(0, len(idx), 65_536):
            sl = idx[a : a + 65_536]
            lw, mean, std = base.mixture_parts(
                out=torch.from_numpy(out[sl]).double(), components=comps
            )
            q = base.mixture_quantiles(log_w=lw, mean=mean, std=std)
            p = d["primary_test"][sl]
            q = (
                q * torch.from_numpy(d["target_std"][p])[:, None]
                + torch.from_numpy(d["target_mean"][p])[:, None]
            )
            parts.append(torch.sort(q, dim=1).values.float().numpy())
    q = np.concatenate(parts)
    table = c.prep.primaries().select("primary", "time_series_id")
    frame = (
        pl.DataFrame(
            {
                "primary": d["primary_test"][idx],
                "power_fcst_init_time": d["origin_test"][idx],
                "valid_time": d["valid_test"][idx],
            }
        )
        .with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        .join(table, on="primary", how="left", maintain_order="left")
        .drop("primary")
    )
    levels = pl.DataFrame(q, schema=list(c.common.LEVEL_COLUMNS), orient="row")
    return frame.select(
        pl.col("time_series_id").cast(pl.Int32), "power_fcst_init_time", "valid_time"
    ).hstack(levels)


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def run(*, months, seeds, arms):
    torch.set_num_threads(2)
    for month in months:
        xc = np.load(XC / f"{month}.npz")
        for seed in seeds:
            todo = [
                a for a in arms if not (W / "seed_preds" / a / f"{month}_s{seed}.parquet").exists()
            ]
            if not todo:
                continue
            d = dict(np.load(s2.CACHE / f"{month}_s{seed}.npz"))
            n_pr = len(d["target_mean"])
            grp_tr, grp_te = s2.group_flag()[d["primary_train"]], s2.group_flag()[d["primary_test"]]
            xtr, xte = xc["x_train"], xc["x_test"]
            weeks = {a: [] for a in todo}
            for week, cut in enumerate(c.cutoffs(month=month)):
                t0 = time.monotonic()
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (d["origin_test"] >= lo) & (d["origin_test"] < hi)
                cu = c.us(t=cut)
                keep = (
                    (d["valid_train"] <= cu - c.CUTOFF_AGE_US)
                    & (d["origin_train"] <= cu)
                    & (d["valid_train"] > cu - c.WINDOW_DAYS * c.DAY_US)
                )
                age = (cu - d["valid_train"][keep]) / c.DAY_US
                w = T((0.5 ** (age / c.HALF_LIFE_90)).astype(np.float32))
                oday = ((d["origin_train"][keep] - c.us(t=c.prep.FIRST_RUN)) // c.DAY_US).astype(
                    np.int64
                )
                is_val = T(oday % 5 == 4)
                z = T(d["z_train"][keep])
                out0 = T(d["out_train"][keep])
                te = np.nonzero(sel)[0]
                out0_te = T(d["out_test"][te])
                # stage-2 X head
                xs_tr = T(
                    np.column_stack([xtr[keep], grp_tr[keep], d["primary_train"][keep]]).astype(
                        np.float32
                    )
                )
                xs_te = T(
                    np.column_stack([xte[te], grp_te[te], d["primary_test"][te]]).astype(np.float32)
                )
                need_x = any(a in todo for a in ("S2-X", "S2-XG"))
                if need_x:
                    xm = XModel(FormX(xtr.shape[1], n_pr)).to(DEV)
                    xm = train_gen(
                        model=xm, out=out0, side=xs_tr, z=z, w=w, is_val=is_val, seed=seed
                    )
                    with torch.no_grad():
                        outx = torch.cat(
                            [
                                xm(out0[i], xs_tr[i])
                                for i in torch.arange(len(out0), device=DEV).split(262144)
                            ]
                        )
                        outx_te = xm(out0_te, xs_te)
                    del xm
                # calendar-only mixture
                if any(a in todo for a in ("S2-G", "S2-XG")):
                    cal = c.new_model(outlier=False, n_inputs=len(CAL), n_primaries=n_pr).to(DEV)
                    pt = T(d["primary_train"][keep])
                    xk = np.ascontiguousarray(xtr[keep][:, CAL])
                    xkt = T(xk)
                    c.train(
                        model=cal,
                        x=xkt,
                        z=z,
                        w=w,
                        p=pt,
                        is_val=is_val,
                        seed=seed,
                        lr=c.PRETRAIN_LR,
                        outlier=False,
                        prior=False,
                        device=DEV,
                    )
                    cal.eval()
                    with torch.no_grad():
                        clim = torch.cat(
                            [
                                cal(xkt[i], pt[i])
                                for i in torch.arange(len(z), device=DEV).split(262144)
                            ]
                        )
                        clim_te = cal(T(xte[te][:, CAL]), T(d["primary_test"][te]))
                    del cal
                    gin_tr = torch.cat(
                        [out0.new_tensor(d["aux_train"][keep][:, :1]), T(xtr[keep][:, STD])], dim=1
                    )
                    gin_te = torch.cat(
                        [out0_te.new_tensor(d["aux_test"][te][:, :1]), T(xte[te][:, STD])], dim=1
                    )
                full = np.zeros((len(d["out_test"]), 3 * K), dtype=np.float32)
                for arm in todo:
                    comps = K
                    if arm == "S2-X":
                        full[te] = outx_te.cpu().numpy()
                        res = full
                    elif arm in ("S2-G", "S2-XG"):
                        a_tr, a_te = (out0, out0_te) if arm == "S2-G" else (outx, outx_te)
                        gm = GateModel().to(DEV)
                        gm = train_gen(
                            model=gm,
                            out=a_tr,
                            side=torch.cat([clim, gin_tr], 1),
                            z=z,
                            w=w,
                            is_val=is_val,
                            seed=seed,
                            lr=1e-2,
                        )
                        with torch.no_grad():
                            side_te = torch.cat([clim_te, gin_te], 1)
                            o = gm(a_te, side_te)
                            g = gm.gate.weight(gin_te).cpu().numpy()
                        res = np.zeros((len(d["out_test"]), 18), dtype=np.float32)
                        res[te] = o.cpu().numpy()
                        comps = 2 * K
                        lb = (d["lead_test"][te] // 30).astype(int).clip(0, 7)
                        rec = (
                            pl.DataFrame(
                                {
                                    "arm": arm,
                                    "month": month,
                                    "seed": seed,
                                    "week": week + 1,
                                    "lead_bin_30h": lb,
                                    "g": g,
                                }
                            )
                            .group_by("arm", "month", "seed", "week", "lead_bin_30h")
                            .agg(
                                pl.col("g").mean().alias("mean_gate_weather_weight"),
                                pl.len().alias("rows"),
                            )
                        )
                        (W / "gate").mkdir(parents=True, exist_ok=True)
                        rec.write_parquet(W / "gate" / f"{arm}_{month}_s{seed}_w{week + 1}.parquet")
                    weeks[arm].append(quantile_frame(out=res, d=d, select=sel, comps=comps))
                print(
                    f"{month} s{seed} w{week + 1} {todo}: {int(keep.sum()):,} rows {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
                torch.cuda.empty_cache()
            for arm, fr in weeks.items():
                tgt = W / "seed_preds" / arm / f"{month}_s{seed}.parquet"
                tgt.parent.mkdir(parents=True, exist_ok=True)
                fr_all = pl.concat(fr)
                fr_all.write_parquet(tgt.with_suffix(".tmp"))
                tgt.with_suffix(".tmp").rename(tgt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=list(c.MONTHS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=["S2-X", "S2-G", "S2-XG"])
    a = ap.parse_args()
    run(months=a.months, seeds=a.seeds, arms=a.arms)
