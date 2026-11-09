# ruff: noqa: D107, PT018, PLC3002, ANN001, D103, E501, PLR0915
"""Experiment 667, prevailing rows, step 3: fit the head of fold month M on the banks of earlier months (CPU).

Fold M's head is trained on the real rows and the injected bank rows of the fold months BEFORE M
only, and only on rows whose valid time is before the first half-hour of M (so no target at or
after the fold start is read). Early stopping uses origin days 21 onward of each training month; the
origin days 0 to 17 train (experiment 650's split). The loss, optimiser, batch size, epochs, and
patience are those of experiments 647 and 652, and the head is experiment 652's `R-haz` with 14 daily
hazards (`p667_head`), on the mixture `m50` (shifts short and long in equal shares).

Folds with fewer than `MIN_BANKS` earlier banks get `B-haz` instead: the Candidate's quantiles plus
its own recent residual (experiment 647's `res`) times the conditional survival of the best curve
fitted to the generator's synthetic shift durations. No observed power enters its fit.

Output `fits/<arm>_<month>_s<seed>.npz` (arm `Rhaz` or `Bhaz`): for the month's real rows, the
quantiles `real_q` (sorted) and survival probability `real_p`; for its bank rows, the survival
probability `bank_p`. The head's weights are in `fits/Rhaz_<month>_s<seed>.pt`.

Usage: python p667_fit.py MONTH [--seeds 0 1] [--months-all 2025-11 ...]
"""

import argparse
import time
from typing import Final

import e00647_data as d647  # ty: ignore[unresolved-import]
import e00647_fit as f647  # ty: ignore[unresolved-import]
import e00647_head as h647  # ty: ignore[unresolved-import]
import e00650_bank as gen  # ty: ignore[unresolved-import]
import e00650_common as c650  # ty: ignore[unresolved-import]
import e00652_score as s652  # ty: ignore[unresolved-import]
import numpy as np
import p667_data as dat
import p667_head as head667
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

MONTHS: Final = (
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10",
)  # fmt: skip
MIN_BANKS: Final[int] = 2
LAMBDA: Final[float] = 0.05
N_REAL: Final[int] = 60_000
N_BANK: Final[int] = 6_000
N_VAL: Final[int] = 20_000
N_VAL_BANK: Final[int] = 2_000
SHORT_SHARE: Final[float] = 0.5
GENERATOR_SHORT_SHARE: Final[float] = 0.5
SHORT_MAX_DAYS: Final[int] = 14
OUT: Final = dat.ROOT / "prevailing" / "fits"
_CACHE: dict = {}


def load(*, month: str, kind: str) -> dict:
    """Return the arrays of one head-data file with the raw origin-level inputs (as experiment 650's `load`)."""
    key = (month, kind)
    if key not in _CACHE:
        z = dict(np.load(dat.HEAD_DATA / f"{kind}_{month}.npz"))
        miss = list(d647.FEATURE_NAMES).index("gap_day_missing")
        age = z["age"].astype(np.float32)
        raw = np.concatenate(
            [z["feat"], np.log1p(age)[:, None], (age / 30.0)[:, None], z["gate"].astype(np.float32)[:, None],
             np.clip(z["lag_age"] / 48.0, 0.0, 10.0)[:, None]], axis=1,
        ).astype(np.float32)  # fmt: skip
        z["raw"] = np.nan_to_num(raw)
        z["usable"] = (z["feat"][z["org"], miss] == 0.0) & np.isfinite(z["y"])
        if kind == "real":
            z["target"] = z["y"]
            z["train_ok"] = z["usable"] & ~z["verified"] & ~z["gate"][z["org"]]
        else:
            z["target"] = (z["y"] + z["shift"]).astype(np.float32)
            z["train_ok"] = z["usable"]
        _CACHE[key] = z
    return _CACHE[key]


def fold_start_idx(*, month: str) -> int:
    return c650.month_start_idx(month=month)


def row_state(*, z: dict, month: str) -> dict[str, np.ndarray]:
    """Return per-row shift state of a bank file, as `e00652_common.row_state` (events from this folder's table)."""
    ev = pl.read_parquet(dat.HEAD_DATA / f"events_{month}.parquet")
    onset = {
        (int(r["pass"]), int(r["primary"])): int(r["onset_idx"]) for r in ev.iter_rows(named=True)
    }
    end = {(int(r["pass"]), int(r["primary"])): int(r["end_idx"]) for r in ev.iter_rows(named=True)}
    pa, pr = z["pass_"].astype(int), z["prim"].astype(int)
    on = np.array([onset[(p, q)] for p, q in zip(pa, pr, strict=True)])
    en = np.array([end[(p, q)] for p, q in zip(pa, pr, strict=True)])
    oi, vi = z["oi"].astype(np.int64), z["vi"].astype(np.int64)
    in_force = (oi >= on) & (oi < en)
    nb_of, _cap, is_mw = c650.world()
    ms = c650.month_start_idx(month=month)
    days = {
        (r, p): (lambda ev_: (ev_.end_idx - ev_.onset_idx) // 48)(
            gen.draw(
                month_idx=ms,
                r=r,
                primary=p,
                nb_of=nb_of,
                month_start_idx=ms,
                window_end_idx=10**9,
                is_mw=is_mw,
            )
        )
        for r, p in set(zip(pa.tolist(), pr.tolist(), strict=True))
    }
    short = np.array([days[(p, q)] <= SHORT_MAX_DAYS for p, q in zip(pa, pr, strict=True)])
    _u, inv, cnt = np.unique(pa * 100 + pr, return_inverse=True, return_counts=True)
    event_w = (cnt.mean() / cnt[inv]).astype(np.float32)
    censored = en >= dat.YEAR_END_IDX
    return {
        "in_force": in_force, "still_on": in_force & (vi < en), "short": short,
        "event_w": event_w, "censored": censored,
    }  # fmt: skip


def pick(*, z: dict, month: str, valid_before: int, lo: int, hi: int, n: int, rng) -> np.ndarray:
    dom = (z["oi"] - fold_start_idx(month=month)) // 48
    ok = z["train_ok"] & (dom >= lo) & (dom < hi) & (z["vi"] < valid_before)
    idx = np.flatnonzero(ok)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))


class Pool:
    """Rows of several (array dict, row index) parts with the origin-level raw inputs stacked."""

    def __init__(self, *, parts: list[tuple[dict, np.ndarray]]) -> None:
        cols: dict[str, list] = {k: [] for k in ("q", "y", "lead", "org")}
        raws, off = [], 0
        for z, idx in parts:
            cols["q"].append(z["q"][idx])
            cols["y"].append(z["target"][idx])
            cols["lead"].append(z["lead"][idx])
            uniq, inv = np.unique(z["org"][idx], return_inverse=True)
            raws.append(z["raw"][uniq])
            cols["org"].append(inv + off)
            off += len(uniq)
        self.q = np.concatenate(cols["q"]).astype(np.float32)
        self.y = np.concatenate(cols["y"]).astype(np.float32)
        self.lead = np.concatenate(cols["lead"]).astype(np.float32)
        self.org = np.concatenate(cols["org"]).astype(np.int64)
        self.raw = np.concatenate(raws).astype(np.float32)


def class_weight(*, short: np.ndarray) -> np.ndarray:
    s = SHORT_SHARE
    return np.where(short, s / GENERATOR_SHORT_SHARE, (1 - s) / (1 - GENERATOR_SHORT_SHARE))


def extras(*, z: dict, idx: np.ndarray, kind: str, month: str) -> tuple[np.ndarray, ...]:
    """Return (weight, label, in-force, Primary) of the selected rows; real rows have unit weight and no label."""
    n = len(idx)
    if kind == "real":
        return (np.ones(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), z["prim"][idx].astype(np.int64))  # fmt: skip
    if "short" not in z:
        z.update(row_state(z=z, month=month))
    w = (class_weight(short=z["short"][idx]) * z["event_w"][idx]).astype(np.float32)
    return (w, z["still_on"][idx].astype(np.float32), (z["in_force"][idx] & ~z["censored"][idx]).astype(np.float32), z["prim"][idx].astype(np.int64))  # fmt: skip


def predict_rows(
    *, head, z: dict, stats, seed: int, with_q: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return the head's (quantiles or None, survival probability) on every row of `z`, in 50,000-row batches."""
    idx = np.arange(len(z["lead"]))
    pool = Pool(parts=[({**z, "target": np.nan_to_num(z["target"])}, idx)])
    x, age, _ = f647.make_inputs(
        raw=pool.raw, arm={"kind": "mix", "inputs": "all", "short": 0.5}, stats=stats, seed=seed + 3
    )
    qs, ps = [], []
    with torch.no_grad():
        for a in range(0, len(idx), 50_000):
            ix = np.arange(a, min(a + 50_000, len(idx)))
            xb, q, ag, lead, _y = f647.batch_inputs(x_org=x, age_org=age, pool=pool, ix=ix)
            qq, p = head(x=xb, q=q, age=ag, lead_h=lead)
            if with_q:
                qs.append(qq.numpy())
            ps.append(p.numpy())
    return (np.concatenate(qs).astype(np.float32) if with_q else None), np.concatenate(ps).astype(
        np.float32
    )


def fit_rhaz(*, month: str, seed: int, train_months: list[str]) -> str:
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"Rhaz_{month}_s{seed}.npz"
    t0 = time.time()
    before = fold_start_idx(month=month)
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    tr, va, ex_tr, ex_va = [], [], [], []
    for tm in train_months:
        for kind, n_tr, n_va in (("real", N_REAL, N_VAL), ("bank", N_BANK, N_VAL_BANK)):
            z = load(month=tm, kind=kind)
            i_tr = pick(z=z, month=tm, valid_before=before, lo=0, hi=18, n=n_tr, rng=rng)
            i_va = pick(z=z, month=tm, valid_before=before, lo=21, hi=99, n=n_va, rng=rng)
            tr.append((z, i_tr))
            va.append((z, i_va))
            ex_tr.append(extras(z=z, idx=i_tr, kind=kind, month=tm))
            ex_va.append(extras(z=z, idx=i_va, kind=kind, month=tm))
    assert all(int(z["vi"][i].max()) < before for z, i in tr + va if len(i)), (
        "a training row at or after the fold start"
    )
    train, val = Pool(parts=tr), Pool(parts=va)

    def stack(ex: list) -> tuple:
        return tuple(np.concatenate([e[k] for e in ex]) for k in range(4))

    w_tr, lab_tr, inf_tr, _ = stack(ex_tr)
    w_va, lab_va, inf_va, _ = stack(ex_va)
    arm = {"kind": "mix", "inputs": "all", "short": 0.5}
    x_tr, age_tr, stats = f647.make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = f647.make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    torch.manual_seed(seed)
    head = head667.HazardHead(n_in=x_tr.shape[1] + 3)
    opt = torch.optim.Adam(head.parameters(), lr=f647.LR, weight_decay=f647.WD)

    def loss_of(*, pool, x_org, age_org, ix, w, lab, inf) -> torch.Tensor:
        x, q, age, lead, y = f647.batch_inputs(x_org=x_org, age_org=age_org, pool=pool, ix=ix)
        qq, p = head(x=x, q=q, age=age, lead_h=lead)
        wt = torch.from_numpy(w[ix])
        pin = (h647.pinball(y=y, q=qq) * wt).sum() / wt.sum()
        m = torch.from_numpy(inf[ix]) * (age >= h647.AGE_FLOOR_DAYS).to(torch.float32)
        bce = nn.functional.binary_cross_entropy(
            torch.clamp(p, 1e-4, 1 - 1e-4), torch.from_numpy(lab[ix]), reduction="none"
        )
        bce = (bce * m * wt).sum() / torch.clamp((m * wt).sum(), min=1e-6)
        return pin + LAMBDA * bce

    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(f647.EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, f647.BATCH):
            ix = perm[a : a + f647.BATCH]
            loss = loss_of(
                pool=train, x_org=x_tr, age_org=age_tr, ix=ix, w=w_tr, lab=lab_tr, inf=inf_tr
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = 0.0
            for a in range(0, len(val.y), 20_000):
                ix = np.arange(a, min(a + 20_000, len(val.y)))
                vl += float(
                    loss_of(
                        pool=val, x_org=x_va, age_org=age_va, ix=ix, w=w_va, lab=lab_va, inf=inf_va
                    )
                ) * len(ix)
            vl /= len(val.y)
        log.append(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= f647.PATIENCE:
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    torch.save(
        {"state": best_state, "stats": stats, "train_months": train_months},
        OUT / f"Rhaz_{month}_s{seed}.pt",
    )
    real, bank = load(month=month, kind="real"), load(month=month, kind="bank")
    real_q, real_p = predict_rows(head=head, z=real, stats=stats, seed=seed, with_q=True)
    _none, bank_p = predict_rows(head=head, z=bank, stats=stats, seed=seed, with_q=False)
    assert np.isfinite(real_q).all() and np.isfinite(real_p).all()  # ty: ignore[no-matching-overload]
    np.savez(
        out,
        real_q=real_q,  # ty: ignore[invalid-argument-type]
        real_p=real_p,
        bank_p=bank_p,
        val_log=np.array(log),
        n_train=n,
        best=best,
        seconds=time.time() - t0,
        train_months=np.array(train_months),
    )  # ty: ignore[invalid-argument-type]
    return f"{out.name}: trained on {train_months}, {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def generator_durations(*, month: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (durations in days, class weights) of the generator's draws for `month`, all 16 Primaries and 12 passes.

    The draws are synthetic and read no observed power.
    """
    nb_of, _cap, is_mw = c650.world()
    ms = c650.month_start_idx(month=month)
    d = np.array(
        [
            (lambda e: (e.end_idx - e.onset_idx) / 48.0)(
                gen.draw(
                    month_idx=ms,
                    r=r,
                    primary=p,
                    nb_of=nb_of,
                    month_start_idx=ms,
                    window_end_idx=10**9,
                    is_mw=is_mw,
                )
            )
            for r in range(gen.N_TRAIN_PASSES)
            for p in range(16)
        ]
    )
    return d, class_weight(short=d <= SHORT_MAX_DAYS)


def fit_bhaz(*, month: str) -> str:
    """Write the no-fit hazard baseline for a fold with too few earlier banks (no seed: nothing is trained)."""
    from scipy import optimize

    OUT.mkdir(parents=True, exist_ok=True)
    d, w = generator_durations(month=month)
    best = None
    for name, dist in s652.DISTS.items():
        n_par = 0 if name == "expon" else 1

        def nll(theta, dist=dist, n_par=n_par) -> float:
            shape = np.exp(theta[:n_par]) if n_par else ()
            return -float((w * dist.logpdf(d, *shape, 0.0, np.exp(theta[n_par]))).sum())

        r = optimize.minimize(
            nll, x0=np.r_[np.zeros(n_par), np.log(d.mean())], method="Nelder-Mead"
        )
        par = (*(np.exp(r.x[:n_par])), 0.0, float(np.exp(r.x[n_par])))
        if best is None or r.fun < best[2]:
            best = (name, par, float(r.fun))
    assert best is not None
    real = load(month=month, kind="real")
    age = real["age"][real["org"]].astype(np.float32)
    p_b = s652.cond_survival(curve=best, age_days=age, lead_h=real["lead"])
    res = np.nan_to_num(
        real["res"]
    )  # no residual (fewer than one day of earlier forecasts): no shift
    q = np.sort(real["q"] + (res * p_b)[:, None], axis=1).astype(np.float32)
    np.savez(
        OUT / f"Bhaz_{month}.npz",
        real_q=q,
        real_p=p_b.astype(np.float32),
        curve=np.array([best[0]]),
        params=np.array(best[1], dtype=float),
    )  # ty: ignore[invalid-argument-type]
    return f"Bhaz_{month}.npz: curve {best[0]} {np.round(best[1], 3).tolist()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("month")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--force-bhaz", action="store_true")
    a = ap.parse_args()
    earlier = [m for m in MONTHS if m < a.month]
    have = [m for m in earlier if (dat.HEAD_DATA / f"bank_{m}.npz").exists()]
    assert have == earlier, f"banks missing for {sorted(set(earlier) - set(have))}"
    if len(earlier) < MIN_BANKS or a.force_bhaz:
        print(fit_bhaz(month=a.month), flush=True)
    else:
        for seed in a.seeds:
            print(fit_rhaz(month=a.month, seed=seed, train_months=earlier), flush=True)


if __name__ == "__main__":
    main()
