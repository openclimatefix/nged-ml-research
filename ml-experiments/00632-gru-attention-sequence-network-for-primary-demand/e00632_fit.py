# ruff: noqa: E501, D103, PLR0915
"""Experiment 632: pre-train Seq (or an ablation) on ERA5 and CAMS weather, one fit per fold and seed.

Usage: python e00632_fit.py ARM MONTH [MONTH ...] [--seeds 0 1 2] [--max-passes 60] [--frac 1.0]
                                                  [--tag NAME] [--hist 336] [--fut 336]

ARM is one of `seq`, `seq_nb`, `no_attn`, `no_gru`. Predictions go to `predictions/<tag>/` and the
fitted state dicts to `models/<tag>/`, both outside git; `tag` defaults to ARM.
"""

import argparse
import json
import math
import time

import e00632_data as d
import e00632_fold as fold_mod
import e00632_model as m
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00397_models import LEVELS, mixture_parts, mixture_quantiles  # ty: ignore[unresolved-import]

DEV = "cuda"
LR, WD, WARMUP, CLIP, BATCH, PATIENCE = 5e-4, 1e-2, 300, 1.0, 256, 8
P_ALL_MASK, P_RECENT_MASK, P_NB_DROP = 0.25, 0.05, 0.5
CUTOFF_AGE_STEPS = 7  # 3.5 h


def origins_for(
    *, fold, month: str, hist: int, fut: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (train_origins, val_origins, test_origins, target_end) as grid indices; origins are 09:00 UTC."""
    refit = fold_mod.month_start(month=month)
    target_end = fold_mod.idx_of(t=refit) - CUTOFF_AGE_STEPS
    day = 48
    last = target_end - day * 1
    first_origin = 335 + 1
    all_o = np.arange(first_origin - first_origin % day + 18 + day, last + 1, day)
    all_o = all_o[all_o <= target_end - 48]
    # Roster-style validation: spread-out held-out weeks with a purge around each, so training
    # runs right up to the refit and the newest D5 knot sees targets (as the comparators' does).
    held, purged = d.prep422.purge_masks(origin_day=all_o // day)
    val = all_o[held]
    train = all_o[~held & ~purged]
    if train.max() < target_end - 48 - 30 * day:
        raise ValueError("training origins stop long before the refit")
    return train, val, fold_mod.test_origins(month=month), target_end


def block_dropout(*, b, gen: torch.Generator, use_nb: bool) -> None:
    """Mask all power on about 25% of sequences and the newest 24 h of power-bearing tokens on about 5%."""
    n = len(b.origin)
    r = torch.rand(n, generator=gen, device=b.power.device)
    allm = r < P_ALL_MASK
    recent = (r >= P_ALL_MASK) & (r < P_ALL_MASK + P_RECENT_MASK)
    steps = b.hist_idx - (b.origin[:, None] - 48)  # <= 0 for power-bearing tokens
    newest = (steps > -48) & (steps <= 0)
    mask = allm[:, None] | (recent[:, None] & newest)
    b.power = b.power.masked_fill(mask, 0.0)
    b.power_missing = torch.where(mask, torch.ones_like(b.power_missing), b.power_missing)
    if use_nb:
        drop = torch.rand(n, generator=gen, device=b.power.device) < P_NB_DROP
        b.nb_mean = b.nb_mean.masked_fill(drop[:, None], 0.0)
        b.nb_share = b.nb_share.masked_fill(drop[:, None], 0.0)


def run_val(*, net, w, prim, orig, target_end) -> float:
    net.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(orig), BATCH):
            b = w.cut(
                primary=prim[a : a + BATCH], origin=orig[a : a + BATCH], target_end=target_end
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(b)
            s, n = m.masked_nll(out=out, target=b.target)
            tot += float(s)
            cnt += int(n)
    return tot / max(cnt, 1)


def predict(
    *, net, w, s, cap, month: str, origins: np.ndarray, hist: int, fut: int
) -> pl.DataFrame:
    net.eval()
    n_p = len(s.target_mean)
    prim = torch.arange(n_p, device=DEV).repeat_interleave(len(origins))
    orig = torch.from_numpy(np.tile(origins, n_p)).to(DEV)
    qs = []
    with torch.no_grad():
        for a in range(0, len(orig), BATCH):
            b = w.cut(primary=prim[a : a + BATCH], origin=orig[a : a + BATCH])
            d.check_cheap(batch=b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(b)
            log_w, mean, std = mixture_parts(out=out.reshape(-1, 9), components=3)
            q = torch.sort(mixture_quantiles(log_w=log_w, mean=mean, std=std), dim=1).values
            qs.append(q.reshape(len(b.origin), fut, 11).cpu().numpy())
    q = np.concatenate(qs)  # (n, fut, 11)
    p_np = prim.cpu().numpy()
    q = q * s.target_std[p_np, None, None] + s.target_mean[p_np, None, None]
    q = (q * cap[p_np, None, None]).reshape(-1, 11)
    o_np = orig.cpu().numpy()
    origin_idx = np.repeat(o_np, fut)
    target_idx = origin_idx + np.tile(np.arange(1, fut + 1), len(o_np))
    ids = d.prep.primaries().sort("primary")["time_series_id"].to_numpy()[np.repeat(p_np, fut)]
    to_t = lambda idx: pl.from_epoch(
        pl.Series(d.prep422.GRID_START_US + idx * d.prep422.STEP_US), time_unit="us"
    ).dt.replace_time_zone("UTC")
    return pl.DataFrame(
        {"time_series_id": ids, "power_fcst_init_time": to_t(origin_idx), "valid_time": to_t(target_idx),
         "lead_hours": ((target_idx - origin_idx) * 0.5).astype(np.float32),
         **{f"q{round(t * 100):02d}": q[:, i] for i, t in enumerate(LEVELS)}}
    ).sort("time_series_id", "power_fcst_init_time", "valid_time")  # fmt: skip


def fit_one(
    *,
    arm: str,
    tag: str,
    month: str,
    seed: int,
    max_passes: int,
    frac: float,
    hist: int,
    fut: int,
    ctx: dict,
    shifts: bool = False,
) -> dict:
    s, cap = ctx["s"][month], ctx["cap"]
    fold = fold_mod.fold_for_seq(s=s)
    refit = fold_mod.month_start(month=month)
    train_o, val_o, test_o, target_end = origins_for(fold=fold, month=month, hist=hist, fut=fut)
    n_p = len(s.target_mean)
    z = d.standardised_power(power=ctx["clean"], fold=fold)
    nb_mean, nb_share = d.neighbour_summary(z=z, of=ctx["of"])
    z_in = None
    if shifts:
        # lag-only synthetic load-shifts (experiment 581's recipe) on the history power; targets stay clean
        import e00581_bank as bank  # ty: ignore[unresolved-import]

        fixed, n_events = bank.synthetic_clean(
            clean=ctx["clean"], first=0, last=target_end, seed=int(month.replace("-", ""))
        )
        z_in = d.standardised_power(power=fixed, fold=fold)
        nb_in, nb_share_in = d.neighbour_summary(z=z_in, of=ctx["of"])
        print(f"{month}: {n_events} synthetic events", flush=True)
    st = np.load(fold_mod.P2_DIR / "stats" / "B1" / f"{month}.npz")
    b1g, cams = d.b1_scale(
        grid=ctx["b1_grid"], cams=ctx["b1_cams"], mean=st["mean"], std=st["std"], device=DEV
    )
    x = d.standardised_inputs(raw=ctx["raw"], fold=fold, device=DEV, b1_grid=b1g)
    w_clean = d.Windows(
        x=x, z=z, nb_mean=nb_mean, nb_share=nb_share, cams=cams, tau=ctx["tau"],
        n_hist=hist, n_fut=fut, device=DEV,
    )  # fmt: skip
    w = w_clean
    if shifts:
        w = d.Windows(
            x=x, z=z, nb_mean=nb_in, nb_share=nb_share_in, cams=cams, tau=ctx["tau"],
            n_hist=hist, n_fut=fut, device=DEV, z_in=z_in,
        )  # fmt: skip
    n_knots = int((refit.year - 2019) * 12 + refit.month - 9) + 1
    # a window must not read targets at or after target_end, and the newest training origin keeps >= 1 target
    pairs = lambda o: (
        torch.arange(n_p, device=DEV).repeat_interleave(len(o)),
        torch.from_numpy(np.tile(o, n_p)).to(DEV),
    )
    tp, to = pairs(train_o)
    vp, vo = pairs(val_o)
    # drop sequences with no finite target in the unmasked part
    fin = torch.isfinite(w.z)
    csum = torch.cumsum(fin.int(), dim=1)
    lo, hi = to + 1, torch.clamp(to + fut, max=target_end - 1)
    has = (csum[tp, hi] - csum[tp, lo - 1]) > 0
    tp, to = tp[has], to[has]
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    net = m.Seq(n_primaries=n_p, n_knots=n_knots, n_hist=hist, n_fut=fut, arm=arm).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    per_pass = max(1, int(len(to) * frac) // BATCH)
    total = max_passes * per_pass
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: (
            min(1.0, (i + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(i, total) / total))
        ),
    )
    best, best_state, bad, passes = float("inf"), None, 0, 0
    t0 = time.monotonic()
    poison_msg = ""
    for passes in range(1, max_passes + 1):
        net.train()
        perm = torch.randperm(len(to), device=DEV, generator=gen)[: per_pass * BATCH]
        run = 0.0
        for a in range(0, len(perm), BATCH):
            idx = perm[a : a + BATCH]
            b = w.cut(primary=tp[idx], origin=to[idx], target_end=target_end)
            d.check_cheap(batch=b)
            if not poison_msg:
                poison_msg = d.check_poisoned(batch=b, windows=w)
            block_dropout(b=b, gen=gen, use_nb=net.use_nb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(b)
            sm, cnt = m.masked_nll(out=out, target=b.target)
            loss = sm / cnt.clamp(min=1) + net.penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            sched.step()
            run += float(loss)
        v = run_val(net=net, w=w, prim=vp, orig=vo, target_end=target_end)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        print(
            f"  {tag} {month} s{seed} pass {passes}: train {run / (len(perm) // BATCH):.4f} val {v:.4f} best {best:.4f} ({time.monotonic() - t0:.0f} s)",
            flush=True,
        )
        if bad >= PATIENCE:
            break
    assert best_state is not None
    net.load_state_dict(best_state)
    # knots: nothing at or after the cutoff was fitted (targets there are masked), so the newest knot is read flat
    mdir = d.ROOT / "models" / tag
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, mdir / f"{month}_s{seed}.pt")
    # test predictions, with the full no-lookahead check on every test origin
    tprim = torch.arange(n_p, device=DEV).repeat_interleave(len(test_o))
    torig = torch.from_numpy(np.tile(test_o, n_p)).to(DEV)
    for a in range(0, len(torig), BATCH):
        d.check_poisoned(
            batch=w_clean.cut(primary=tprim[a : a + BATCH], origin=torig[a : a + BATCH]),
            windows=w_clean,
            limit=BATCH,
        )
    frame = predict(net=net, w=w_clean, s=s, cap=cap, month=month, origins=test_o, hist=hist, fut=fut)
    out = d.ROOT / "predictions" / tag / f"{month}_s{seed}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(out)
    info = {
        "tag": tag, "month": month, "seed": seed, "passes": passes, "best_val_nll": best, "seconds": time.monotonic() - t0,
        "train_sequences": len(to), "val_sequences": len(vo), "test_origins": len(test_o), "target_end_idx": target_end,
        "dropped_half_hours_train_region": int(np.isnan(ctx["clean"][:, : target_end]).sum()), "poison_check": poison_msg,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }  # fmt: skip
    (d.ROOT / "logs").mkdir(exist_ok=True)
    with (d.ROOT / "logs" / f"{tag}.jsonl").open("a") as f:
        f.write(json.dumps(info) + "\n")
    print(info, flush=True)
    del x, w, w_clean
    torch.cuda.empty_cache()
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("months", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-passes", type=int, default=60)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--synthetic-shifts", action="store_true")
    ap.add_argument("--hist", type=int, default=336)
    ap.add_argument("--fut", type=int, default=336)
    a = ap.parse_args()
    tag = a.tag or a.arm
    torch.cuda.set_per_process_memory_fraction(8 / 48)
    clean = d.clean_power()
    b1g, b1c = d.dense_b1()
    ctx = {
        "raw": d.dense_raw(), "clean": clean, "b1_grid": b1g, "b1_cams": b1c, "tau": d.tau_grid(),
        "of": d.neighbour_lists(), "cap": fold_mod.capacity_of_primary(),
        "s": {mo: fold_mod.fold_stats(month=mo, clean=clean) for mo in a.months},
    }  # fmt: skip
    for month in a.months:
        for seed in a.seeds:
            if (d.ROOT / "predictions" / tag / f"{month}_s{seed}.parquet").exists():
                continue
            fit_one(
                arm=a.arm,
                tag=tag,
                month=month,
                seed=seed,
                max_passes=a.max_passes,
                frac=a.frac,
                hist=a.hist,
                fut=a.fut,
                ctx=ctx,
                shifts=a.synthetic_shifts,
            )


if __name__ == "__main__":
    main()
