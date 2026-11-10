# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, E501, F841, PLR0915, PT018, RUF059
r"""GPU arms of the 524 remaining arms (interim, unaudited, not for merge).

`e2e`: S0-E2E. Stage 0 (form 2) is first fitted on the ENS-to-ERA5/CAMS pairs (ERA5 <= cutoff - 5 d,
CAMS <= cutoff - 1 d), then trained end to end through the frozen month's stage 1 against the power
negative log-likelihood on the cutoff's ENS rows (valid time <= cutoff - 3.5 h), stage 1 untouched.
`stress`: S2-J-0.4x leak stress. Refit week 3 with the last 7 days of targets shifted by +-3 standard
deviations of the Primary's 7-day mean (experiment 510's world), control and both signs, one seed;
the leak is sign x 100 x the change in the median forecast of the following week (points of capacity).
Also stores the control S2-J weights.
`s2r`: S2-R. Stage 1 refitted at each weekly cutoff from the month's pretrained weights on ERA5 and CAMS
rows up to cutoff - 5 d with a 90-day recency weight, then frozen; stage 2 (form B, NLL) is trained on
cross-fitted outputs (6 contiguous blocks of the history, each block-model refitted without its block).
"""

import argparse
import time
from datetime import timedelta

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00524_joint as joint
import e00524_rest_s0 as s0
import e00524_rest_score as rs
import e00524_stage1 as s1
import e00524_stage2 as s2
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

K = c.COMPONENTS
DEV = "cuda"
SEED_PREDS = rs.SEED_PREDS
MODELS = rs.WORK / "models"
E2E_LR = 1e-3


def write_arm(*, arm, month, seed, frame, suffix=""):
    target = SEED_PREDS / arm / f"{month}_s{seed}{suffix}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)


def month_setup(*, month, clean):
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    columns = list(mr.base.columns)
    return mr, test, columns


# ---- S0-E2E -----------------------------------------------------------------------------------


class E2E(torch.nn.Module):
    """Stage 0 (trainable) in front of frozen stage 1. Input columns: stage-1 inputs, then the 6 side inputs."""

    def __init__(self, *, stage0, stage1, n, mean_pos):
        super().__init__()
        self.stage0, self.stage1, self.n = stage0, stage1, n
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.register_buffer("pos", torch.tensor(mean_pos))

    def train(self, mode=True):
        super().train(mode)
        self.stage1.eval()
        return self

    def forward(self, x, p):
        xk, side = x[:, : self.n], x[:, self.n + 0 : self.n + 6][:, [0, 3, 4]]
        x7 = self.stage0(xk[:, self.pos], side)
        xk = xk.clone()
        xk[:, self.pos] = x7
        return self.stage1(xk, p)


def train_e2e(*, model, x, z, w, p, is_val, seed):
    torch.manual_seed(seed)
    tr, va = torch.nonzero(~is_val).squeeze(1), torch.nonzero(is_val).squeeze(1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if len(va) > c.VALIDATION_ROWS:
        va = va[torch.randperm(len(va), generator=g)[: c.VALIDATION_ROWS].to(DEV)]
    opt = torch.optim.AdamW(model.stage0.parameters(), lr=E2E_LR, weight_decay=c.WEIGHT_DECAY)
    best, state, stale, epochs = float("inf"), None, 0, 0
    with torch.no_grad():
        best = float(
            torch.cat(
                [
                    base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=K)
                    for i in va.split(c.BATCH * 2)
                ]
            ).mean()
        )
    state = {a: v.detach().clone() for a, v in model.state_dict().items()}
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = tr[torch.randperm(len(tr), device=DEV)][: int(c.ROW_FRACTION * len(tr))]
        for s in range(0, len(order), c.BATCH):
            b = order[s : s + c.BATCH]
            loss = (w[b] * base.mixture_nll(out=model(x[b], p[b]), z=z[b], components=K)).sum() / w[
                b
            ].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(
                torch.cat(
                    [
                        base.mixture_nll(out=model(x[i], p[i]), z=z[i], components=K)
                        for i in va.split(c.BATCH * 2)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if v < best - 1e-5:
            best, stale = v, 0
            state = {a: t.detach().clone() for a, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return epochs, best


def predict(*, model, x_test, test_primary, sel, d):
    out = np.zeros((len(test_primary), 3 * K), dtype=np.float32)
    rows = np.nonzero(sel)[0]
    with torch.no_grad():
        for a in range(0, len(rows), 65_536):
            sl = rows[a : a + 65_536]
            out[sl] = (
                model(x_test[sl].to(DEV), torch.from_numpy(test_primary[sl]).to(DEV)).cpu().numpy()
            )
    return s2.quantile_frame(out=out, d=d, select=sel)


def e2e(*, months, seeds):
    clean = c.clean_stack()
    for month in months:
        todo = [s for s in seeds if not (SEED_PREDS / "S0-E2E" / f"{month}_s{s}.parquet").exists()]
        if not todo:
            continue
        mr, test, columns = month_setup(month=month, clean=clean)
        n_primaries = len(mr.base.target_mean)
        aux_all = s2.aux_inputs(x=mr.x, columns=columns, lead=mr.lead, primary=mr.primary)
        aux_test = s2.aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        t_all, _ = s0.dense_targets(
            valid_us=mr.valid_us,
            primary=mr.primary,
            columns=columns,
            standardiser=mr.base.standardiser,
        )
        origin_day_all = ((mr.origin_us - c.us(t=c.prep.FIRST_RUN)) // c.DAY_US).astype(np.int64)
        for seed in todo:
            _, keep = s2.load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            mean_pos = [keep.index(columns.index(f"mean_{w}")) for w in s0.WEATHER]
            xk = np.ascontiguousarray(mr.x[:, keep])
            x_all = np.concatenate([xk, aux_all], axis=1)
            x_test = torch.from_numpy(
                np.concatenate([np.ascontiguousarray(test.x[:, keep]), aux_test], axis=1)
            )
            d = {
                "primary_test": test.primary,
                "origin_test": test.origin,
                "valid_test": test.valid,
                "target_mean": mr.base.target_mean,
                "target_std": mr.base.target_std,
            }
            frames = []
            for week, cut in enumerate(c.cutoffs(month=month)):
                t0 = time.monotonic()
                cu = c.us(t=cut)
                # pair fit on the pair-eligible rows (cutoff - 5 d / - 1 d)
                m = s0.pair_mask(t=t_all, valid_us=mr.valid_us, origin_us=mr.origin_us, cu=cu)
                ok = np.nonzero(m.any(1))[0]
                age = (cu - mr.valid_us[ok]) / c.DAY_US
                wp = (0.5 ** (age / s2.HALF_LIFE)).astype(np.float32)
                x7 = np.ascontiguousarray(xk[ok][:, mean_pos])
                st0 = s0.train_form2(
                    x=x7,
                    side=s0.side_of(aux_all[ok]),
                    t=t_all[ok],
                    m=m[ok],
                    w=wp,
                    origin_day=origin_day_all[ok],
                    seed=seed,
                    device=DEV,
                )
                refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=c.HALF_LIFE_90)
                assert (
                    refit.report["rows_past_valid_cut"] == 0
                    and refit.report["rows_past_origin_cut"] == 0
                )
                idx = refit.index
                prim = mr.primary[idx]
                z = (refit.y - mr.base.target_mean[prim]) / mr.base.target_std[prim]
                t = {
                    "x": torch.from_numpy(x_all[idx]).to(DEV),
                    "z": torch.from_numpy(z.astype(np.float32)).to(DEV),
                    "w": torch.from_numpy(refit.weight).to(DEV),
                    "p": torch.from_numpy(prim).to(DEV),
                    "v": torch.from_numpy(refit.is_val).to(DEV),
                }
                stage1, _ = s2.load_stage1(
                    path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
                )
                model = E2E(stage0=st0, stage1=stage1, n=len(keep), mean_pos=mean_pos).to(DEV)
                epochs, best = train_e2e(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed
                )
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (test.origin >= lo) & (test.origin < hi)
                frames.append(
                    predict(model=model, x_test=x_test, test_primary=test.primary, sel=sel, d=d)
                )
                (MODELS / "S0-E2E").mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.stage0.state_dict(), MODELS / "S0-E2E" / f"{month}_s{seed}_w{week}.pt"
                )
                print(
                    f"S0-E2E {month} s{seed} w{week + 1}: {epochs} epochs, val NLL {best:.4f}, {len(idx):,} rows, {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
                del t, model
                torch.cuda.empty_cache()
            write_arm(arm="S0-E2E", month=month, seed=seed, frame=pl.concat(frames))
        del mr, test


# ---- leak stress ------------------------------------------------------------------------------


def stress(*, months, seed, ratio):
    clean = c.clean_stack()
    rec = []
    for month in months:
        out_csv = rs.OUT / f"leak_stress_{month}.csv"
        if out_csv.exists():
            continue
        mr, test, columns = month_setup(month=month, clean=clean)
        n_primaries = len(mr.base.target_mean)
        aux_all = s2.aux_inputs(x=mr.x, columns=columns, lead=mr.lead, primary=mr.primary)
        aux_test = s2.aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        _, keep = s2.load_stage1(path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries)
        x_all = np.concatenate([np.ascontiguousarray(mr.x[:, keep]), aux_all], axis=1)
        x_test = torch.from_numpy(
            np.concatenate([np.ascontiguousarray(test.x[:, keep]), aux_test], axis=1)
        )
        d = {
            "primary_test": test.primary,
            "origin_test": test.origin,
            "valid_test": test.valid,
            "target_mean": mr.base.target_mean,
            "target_std": mr.base.target_std,
        }
        week = c.STRESS_WEEK
        cut = c.cutoffs(month=month)[week]
        shift = c.shift_sizes(clean=clean, cutoff=cut)
        lo, hi = fit510.window_of(month=month, week=week)
        sel = (test.origin >= lo) & (test.origin < hi)
        frames = {}
        for name, sign in (("control", 0), ("p", 1), ("n", -1)):
            t0 = time.monotonic()
            refit = c.select_rows(
                rows=mr,
                cutoff=cut,
                half_life_days=c.HALF_LIFE_90,
                shift=shift if sign else None,
                sign=sign,
                clean=clean,
            )
            idx = refit.index
            prim = mr.primary[idx]
            z = (refit.y - mr.base.target_mean[prim]) / mr.base.target_std[prim]
            t = {
                "x": torch.from_numpy(x_all[idx]).to(DEV),
                "z": torch.from_numpy(z.astype(np.float32)).to(DEV),
                "w": torch.from_numpy(refit.weight).to(DEV),
                "p": torch.from_numpy(prim).to(DEV),
                "v": torch.from_numpy(refit.is_val).to(DEV),
            }
            stage1, _ = s2.load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            model = joint.Joint(stage1=stage1, n_stage1_inputs=len(keep)).to(DEV)
            epochs, best = joint.train_joint(
                model=model,
                x=t["x"],
                z=t["z"],
                w=t["w"],
                p=t["p"],
                is_val=t["v"],
                seed=seed,
                stage1_lr=ratio * c.PRETRAIN_LR,
                device=DEV,
            )
            frames[name] = predict(
                model=model, x_test=x_test, test_primary=test.primary, sel=sel, d=d
            )
            if name == "control":
                (MODELS / f"S2-J-{ratio:g}x").mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), MODELS / f"S2-J-{ratio:g}x" / f"{month}_s{seed}_w{week}.pt"
                )
            print(
                f"stress {month} {name}: {epochs} epochs, injected rows {refit.report['injected_rows']:,}, {time.monotonic() - t0:.0f} s",
                flush=True,
            )
            del t, model
            torch.cuda.empty_cache()
        keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
        ctrl = frames["control"].select(*keys, c=pl.col("q50"))
        parts = []
        for name, sign in (("p", 1), ("n", -1)):
            j = ctrl.join(frames[name].select(*keys, s=pl.col("q50")), on=keys).with_columns(
                lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                shift=sign * 100.0 * (pl.col("s") - pl.col("c")),
            )
            parts.append(j)
        row = {"month": month, "ratio": ratio}
        for label, cond in {
            "all": pl.lit(True),
            "day1": pl.col("lead") <= 24,
            "days2_3": (pl.col("lead") > 24) & (pl.col("lead") <= 72),
            "days4_10": pl.col("lead") > 72,
        }.items():
            row[label] = float(np.mean([p_.filter(cond)["shift"].mean() for p_ in parts]))
        pl.DataFrame([row]).write_csv(out_csv)
        print(row, flush=True)
        del mr, test


# ---- S2-R -------------------------------------------------------------------------------------


def refit_stage1(*, pre, weights, month, seed, n_primaries, keep, z, held_block, edges, target_idx):
    """Fine-tune the month's pretrained stage 1 on the (block-left-out) ERA5 and CAMS rows; return the network."""
    model, _ = s2.load_stage1(path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries)
    model = model.to(DEV)
    use = np.ones(len(pre.y), dtype=bool)
    if held_block is not None:
        lo, hi = edges[held_block] - s1.PAD_STEPS, edges[held_block + 1] + s1.PAD_STEPS
        use = ~((target_idx >= lo) & (target_idx < hi))
    i = np.nonzero(use)[0]
    x = torch.from_numpy(np.ascontiguousarray(pre.x[i][:, keep])).to(DEV)
    zt = torch.from_numpy(z[i]).to(DEV)
    w = torch.from_numpy(weights[i]).to(DEV)
    p = torch.from_numpy(pre.primary[i]).to(DEV)
    v = torch.from_numpy(pre.is_val[i]).to(DEV)
    epochs, best = c.train(
        model=model,
        x=x,
        z=zt,
        w=w,
        p=p,
        is_val=v,
        seed=seed,
        lr=c.FINE_TUNE_LR,
        outlier=False,
        prior=False,
        device=DEV,
    )
    model = model.cpu().eval()
    del x, zt, w, p, v
    torch.cuda.empty_cache()
    return model, epochs


def s2r(*, months, seeds):
    torch.set_num_threads(2)
    clean = c.clean_stack()
    for month in months:
        todo = [s for s in seeds if not (SEED_PREDS / "S2-R" / f"{month}_s{s}.parquet").exists()]
        if not todo:
            continue
        mr, test, columns = month_setup(month=month, clean=clean)
        n_primaries = len(mr.base.target_mean)
        window_start = c.us(t=c.month_start(month=month)) - c.WINDOW_DAYS * c.DAY_US
        rows = np.nonzero(mr.valid_us > window_start)[0]
        aux_tr = s2.aux_inputs(
            x=mr.x[rows], columns=columns, lead=mr.lead[rows], primary=mr.primary[rows]
        )
        aux_test = s2.aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        d0 = dict(np.load(s2.CACHE / f"{month}_s0.npz"))
        assert np.array_equal(d0["valid_train"], mr.valid_us[rows])
        z_tr = d0["z_train"]
        valid_tr, origin_tr, prim_tr = mr.valid_us[rows], mr.origin_us[rows], mr.primary[rows]
        origin_day = ((origin_tr - c.us(t=c.prep.FIRST_RUN)) // c.DAY_US).astype(np.int64)
        d = {
            "primary_test": test.primary,
            "origin_test": test.origin,
            "valid_test": test.valid,
            "target_mean": d0["target_mean"],
            "target_std": d0["target_std"],
        }
        _, keep = s2.load_stage1(path=s1.STAGE1 / f"{month}_s{todo[0]}.pt", n_primaries=n_primaries)
        xk_tr = np.ascontiguousarray(mr.x[rows][:, keep])
        xk_te = np.ascontiguousarray(test.x[:, keep])
        del mr
        for seed in todo:
            frames = []
            for week, cut in enumerate(c.cutoffs(month=month)):
                t0 = time.monotonic()
                cu = c.us(t=cut)
                pre = r448.pretrain_rows(
                    rows=_base_rows(month, clean),
                    refit=cut - timedelta(days=5) + timedelta(hours=3.5),
                    clean=clean,
                )
                age = (
                    cu
                    - (
                        c.prep422.GRID_START_US
                        + pre.target_idx.astype(np.int64) * c.prep422.STEP_US
                    )
                ) / c.DAY_US
                weights = (0.5 ** (age / c.HALF_LIFE_90)).astype(np.float32)
                z = ((pre.y - pre.target_mean[pre.primary]) / pre.target_std[pre.primary]).astype(
                    np.float32
                )
                valid_pre = (
                    c.prep422.GRID_START_US + pre.target_idx.astype(np.int64) * c.prep422.STEP_US
                )
                assert valid_pre.max() <= cu - 5 * c.DAY_US, "pretraining rows past cutoff - 5 d"
                # 6 contiguous blocks of the history up to the cutoff (grid indices)
                edges = np.linspace(
                    pre.target_idx.min(),
                    (cu - c.prep422.GRID_START_US) // c.prep422.STEP_US + 1,
                    s1.BLOCKS + 1,
                ).astype(np.int64)
                models = []
                for b in [*range(s1.BLOCKS), None]:
                    m_, ep = refit_stage1(
                        pre=pre,
                        weights=weights,
                        month=month,
                        seed=seed,
                        n_primaries=n_primaries,
                        keep=keep,
                        z=z,
                        held_block=b,
                        edges=edges,
                        target_idx=pre.target_idx,
                    )
                    models.append(m_)
                full = models[-1]
                # cross-fitted stage-1 outputs for the cutoff's ENS rows
                keep_rows = (
                    (valid_tr <= cu - c.CUTOFF_AGE_US)
                    & (origin_tr <= cu)
                    & (valid_tr > cu - c.WINDOW_DAYS * c.DAY_US)
                )
                ri = np.nonzero(keep_rows)[0]
                idxg = (valid_tr[ri] - c.prep422.GRID_START_US) // c.prep422.STEP_US
                blk = np.clip(np.searchsorted(edges, idxg, side="right") - 1, 0, s1.BLOCKS - 1)
                out_tr = np.zeros((len(ri), 3 * K), dtype=np.float32)
                for b in range(s1.BLOCKS):
                    sel_b = np.nonzero(blk == b)[0]
                    for a in range(0, len(sel_b), 1_000_000):
                        sl = sel_b[a : a + 1_000_000]
                        out_tr[sl] = s2.raw_outputs(
                            model=models[b], x=xk_tr[ri[sl]], primary=prim_tr[ri[sl]]
                        )
                age_e = (cu - valid_tr[ri]) / c.DAY_US
                w2 = (0.5 ** (age_e / s2.HALF_LIFE)).astype(np.float32)
                stage2 = s2.train_stage2(
                    form="B",
                    loss="nll",
                    out=out_tr,
                    aux=aux_tr[ri],
                    z=z_tr[ri],
                    weight=w2,
                    origin_day=origin_day[ri],
                    seed=seed,
                )
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (test.origin >= lo) & (test.origin < hi)
                idx = np.nonzero(sel)[0]
                out = np.zeros((len(test.primary), 3 * K), dtype=np.float32)
                raw = s2.raw_outputs(model=full, x=xk_te[idx], primary=test.primary[idx])
                with torch.no_grad():
                    out[idx] = stage2(
                        torch.from_numpy(raw), torch.from_numpy(aux_test[idx])
                    ).numpy()
                frames.append(s2.quantile_frame(out=out, d=d, select=sel))
                print(
                    f"S2-R {month} s{seed} w{week + 1}: pretrain rows {len(pre.y):,}, {time.monotonic() - t0:.0f} s",
                    flush=True,
                )
                del pre, models
            write_arm(arm="S2-R", month=month, seed=seed, frame=pl.concat(frames))


_BASE: dict = {}


def _base_rows(month, clean):
    if month not in _BASE:
        _BASE.clear()
        _BASE[month] = data.build_fold(refit=c.month_start(month=month), clean=clean)
    return _BASE[month]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("e2e", "stress", "s2r"))
    p.add_argument("--months", nargs="+", default=list(c.MONTHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--ratio", type=float, default=0.4)
    a = p.parse_args()
    if a.command == "e2e":
        e2e(months=a.months, seeds=a.seeds)
    elif a.command == "stress":
        stress(months=a.months, seed=a.seeds[0], ratio=a.ratio)
    else:
        s2r(months=a.months, seeds=a.seeds)


if __name__ == "__main__":
    main()
