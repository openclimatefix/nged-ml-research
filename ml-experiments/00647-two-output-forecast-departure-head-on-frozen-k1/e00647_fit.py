# ruff: noqa: D103, D107, E501, PLR0912, PLR0915
"""Experiment 647, stage 2: fit the departure heads on the first month pair and write per-row scores (CPU).

Arms (`ARMS`): H1 (survival mixture), H2 (free quantile head), H1-p and H2-p (noise placebos: every
head input is replaced by Gaussian noise with the column's training mean and variance, including the
age that sets the survival probability), AG (age, gate, and freshest-lag age only), AS (H1 with the
age block shuffled across origins), and H1-short / H1-long (the duration-prior sensitivity runs).

Variants: `lomo` trains on the other development months and holds out the scored month (one of them,
the next month in the list, is the early-stopping validation month); `fc` (forward-chained) trains on
earlier months only. Experiment 647's first month pair is 2025-10 and 2025-12: 2025-10 has no earlier
month, so `fc` runs on 2025-12 only (trained on 2025-10, with every fourth origin day held out for the
early stop).

Hyperparameters are the plan's: Adam, learning rate 0.003, weight decay 0.0001, batch 4096, up to 30
epochs, early stop (patience 4) on the validation loss. The training set is a fixed random sample of
each training month (`N_REAL` real rows and `N_BANK` bank rows, about 10% bank so that the share
of departures is near deployment), a stated deviation made for run time.

Output per fit, `fits/<arm>_<month>_<variant>_s<seed>.npz`: for every scored real and bank row, the
CRPS, coverage flags, band width, tail pinball losses, q05, q50, q95, the depth, and the spread; the
model weights; and the training log. Nothing here writes to K1's rows.

Usage: python e00647_fit.py [--arms H1 ...] [--months 2025-10 2025-12] [--variants lomo fc]
[--seeds 0 1] [--workers 12]
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Final

import numpy as np

NAME: Final = "00647-two-output-forecast-departure-head-on-frozen-k1"
MONTHS: Final = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
N_REAL: Final[int] = 60_000
N_BANK: Final[int] = 6_000
N_VAL: Final[int] = 20_000
N_VAL_BANK: Final[int] = 2_000
BATCH: Final[int] = 4096
EPOCHS: Final[int] = 30
PATIENCE: Final[int] = 4
LR: Final[float] = 0.003
WD: Final[float] = 0.0001
ARMS: Final = {
    "H1": {"kind": "mix", "inputs": "all", "short": 0.5},
    "H2": {"kind": "free", "inputs": "all", "short": 0.5},
    "H1p": {"kind": "mix", "inputs": "noise", "short": 0.5},
    "H2p": {"kind": "free", "inputs": "noise", "short": 0.5},
    "AG": {"kind": "mix", "inputs": "agegate", "short": 0.5},
    "AS": {"kind": "mix", "inputs": "shuffle", "short": 0.5},
    "H1short": {"kind": "mix", "inputs": "all", "short": 0.8},
    "H1long": {"kind": "mix", "inputs": "all", "short": 0.2},
}
_CACHE: dict = {}


def data_dir() -> Path:
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    return ML_EXPERIMENTS_DATA_DIR / NAME


def load(*, month: str) -> dict:
    """Return the month's real and bank arrays, with the origin-level inputs as one raw matrix per kind."""
    if month not in _CACHE:
        out = {}
        for kind in ("real", "bank"):
            z = dict(np.load(data_dir() / "head_data" / f"{kind}_{month}.npz"))
            names = None
            if names is None:
                import e00647_data as dat  # ty: ignore[unresolved-import]

                names = list(dat.FEATURE_NAMES)
            miss = names.index("gap_day_missing")
            age = z["age"].astype(np.float32)
            raw = np.concatenate(
                [z["feat"], np.log1p(age)[:, None], (age / 30.0)[:, None], z["gate"].astype(np.float32)[:, None],
                 np.clip(z["lag_age"] / 48.0, 0.0, 10.0)[:, None]], axis=1,
            ).astype(np.float32)  # fmt: skip
            z["raw"] = raw
            z["usable"] = (z["feat"][z["org"], miss] == 0.0) & np.isfinite(z["y"])
            if kind == "real":
                z["target"] = z["y"]
                z["train_ok"] = z["usable"] & ~z["flagged"] & ~z["verified"] & ~z["gate"][z["org"]]
            else:
                z["target"] = (z["y"] + z["shift"]).astype(np.float32)
                z["train_ok"] = z["usable"]
            out[kind] = z
        _CACHE[month] = out
    return _CACHE[month]


N_FEAT: Final[int] = 58
AGE_BLOCK: Final = slice(N_FEAT, N_FEAT + 2)
AGEGATE_COLS: Final = list(range(N_FEAT, N_FEAT + 4))


def sample(
    *,
    month: str,
    n_real: int,
    n_bank: int,
    rng: np.random.Generator,
    mask_day: tuple[int, int] | None = None,
    day_range: tuple[int, int] | None = None,
) -> list:
    """Return [(kind, row indices)] of a random sample of the month's trainable rows."""
    d = load(month=month)
    out = []
    for kind, n in (("real", n_real), ("bank", n_bank)):
        ok = d[kind]["train_ok"].copy()
        if mask_day is not None:
            day = d[kind]["oi"] // 48
            ok &= (day % mask_day[0] == mask_day[1]) if mask_day[1] >= 0 else ok
        if day_range is not None:  # origin day of the month in [lo, hi)
            import e00647_data as dat  # ty: ignore[unresolved-import]

            dom = (d[kind]["oi"] - dat.month_start_idx(month=month)) // 48
            ok &= (dom >= day_range[0]) & (dom < day_range[1])
        idx = np.flatnonzero(ok)
        out.append((kind, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))))
    return out


class Pool:
    """Rows from several months and kinds, with their origin-level raw inputs stacked."""

    def __init__(self, *, parts: list[tuple[str, str, np.ndarray]]) -> None:
        d = {m: load(month=m) for m in {p[0] for p in parts}}
        cols: dict[str, list] = {k: [] for k in ("q", "y", "lead", "org", "raw_age")}
        raws: list[np.ndarray] = []
        offset = 0
        self.tags: list[tuple[str, str, int]] = []
        for month, kind, idx in parts:
            z = d[month][kind]
            cols["q"].append(z["q"][idx])
            cols["y"].append(z["target"][idx])
            cols["lead"].append(z["lead"][idx])
            uniq, inv = np.unique(z["org"][idx], return_inverse=True)
            raws.append(z["raw"][uniq])
            cols["org"].append(inv + offset)
            offset += len(uniq)
        self.q = np.concatenate(cols["q"]).astype(np.float32)
        self.y = np.concatenate(cols["y"]).astype(np.float32)
        self.lead = np.concatenate(cols["lead"]).astype(np.float32)
        self.org = np.concatenate(cols["org"]).astype(np.int64)
        self.raw = np.concatenate(raws).astype(np.float32)  # (n_org, 62)


def make_inputs(
    *, raw: np.ndarray, arm: dict, stats: tuple | None, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return (standardised inputs, age in days, (mean, std)) for the arm; `stats` fixes the standardisation."""
    rng = np.random.default_rng(seed + 1000)
    r = raw.copy()
    if arm["inputs"] == "noise":
        mu, sd = r.mean(0), r.std(0)
        r = (mu + sd * rng.standard_normal(r.shape)).astype(np.float32)
        r[:, N_FEAT] = np.log1p(np.clip(np.expm1(r[:, N_FEAT]), 0.0, 120.0))
    if arm["inputs"] == "shuffle":
        perm = rng.permutation(len(r))
        r[:, AGE_BLOCK] = r[perm][:, AGE_BLOCK]
    age = np.expm1(np.clip(r[:, N_FEAT], 0.0, None)).astype(np.float32)
    x = r[:, AGEGATE_COLS] if arm["inputs"] == "agegate" else r
    if stats is None:
        stats = (x.mean(0), np.maximum(x.std(0), 1e-3))
    x = np.clip((x - stats[0]) / stats[1], -10.0, 10.0).astype(np.float32)
    return x, age, stats


def row_inputs(*, q: np.ndarray, lead: np.ndarray) -> np.ndarray:
    return np.stack(
        [lead / 240.0, (q[:, 5] - q[:, 1]) * 10.0, (q[:, 9] - q[:, 5]) * 10.0], axis=1
    ).astype(np.float32)


def batch_inputs(*, x_org, age_org, pool: Pool, ix):  # noqa: ANN001, ANN201
    import torch  # ty: ignore[unresolved-import]

    o = pool.org[ix]
    rows = row_inputs(q=pool.q[ix], lead=pool.lead[ix])
    return (
        torch.from_numpy(np.concatenate([x_org[o], rows], axis=1)),
        torch.from_numpy(pool.q[ix]),
        torch.from_numpy(age_org[o]),
        torch.from_numpy(pool.lead[ix]),
        torch.from_numpy(pool.y[ix]),
    )


def metrics(*, y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Per-row scores in points of capacity (x100) and flags."""
    tau = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99], np.float32)
    e = y[:, None] - q
    pin = np.maximum(tau * e, (tau - 1.0) * e) * 100.0
    # mean of the forecast: the integral of the piecewise-linear quantile function, flat beyond q01 and q99
    area = (
        q[:, 0] * 0.01
        + q[:, -1] * 0.01
        + (0.5 * (q[:, 1:] + q[:, :-1]) * np.diff(tau)[None, :]).sum(1)
    )
    return {
        "low01": (y < q[:, 0]).astype(np.uint8), "high99": (y > q[:, 10]).astype(np.uint8),
        "medmean": ((q[:, 5] - area) * 100.0).astype(np.float32),
        "crps": (2.0 * pin.mean(1)).astype(np.float32),
        "low": (y < q[:, 1]).astype(np.uint8), "high": (y > q[:, 9]).astype(np.uint8),
        "width": ((q[:, 9] - q[:, 1]) * 100.0).astype(np.float16),
        "pin01": pin[:, 0].astype(np.float16), "pin05": pin[:, 1].astype(np.float16),
        "pin95": pin[:, 9].astype(np.float16), "pin99": pin[:, 10].astype(np.float16),
        "q05": q[:, 1], "q50": q[:, 5], "q95": q[:, 9],
    }  # fmt: skip


def fit_one(job: tuple[str, str, str, int]) -> str:
    import e00647_head as h  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm_name, month, variant, seed = job
    arm = ARMS[arm_name]
    out_dir = data_dir() / "fits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{arm_name}_{month}_{variant}_s{seed}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    others = [m for m in MONTHS if m != month]
    if variant == "lomo":
        val_month = others[0]
        train_months = others[1:]
        tr = [
            (m, k, i)
            for m in train_months
            for k, i in sample(month=m, n_real=N_REAL, n_bank=N_BANK, rng=rng)
        ]
        va = [
            (val_month, k, i)
            for k, i in sample(month=val_month, n_real=N_VAL, n_bank=N_VAL_BANK, rng=rng)
        ]
    else:
        earlier = [m for m in MONTHS if m < month]
        assert earlier, "no earlier month to train on"
        tr, va = [], []
        import e00647_data as dat  # ty: ignore[unresolved-import]

        for m in earlier:
            d = load(month=m)
            ms = dat.month_start_idx(month=m)
            for k, n_tr, n_va in (("real", N_REAL, N_VAL), ("bank", N_BANK, N_VAL_BANK)):
                dom = (d[k]["oi"] - ms) // 48
                # contiguous blocks: train on days 0-13, leave days 14-20 out as a gap, validate on days 21 onward
                if (
                    variant == "fcb"
                ):  # train on valid-time days 0-17, a gap of 10 days, validate from day 28
                    vday = (d[k]["vi"] - ms) // 48
                    for sel_, n, dst in (
                        ((dom <= 17) & (vday <= 17), n_tr * 2, tr),
                        ((dom >= 28) & (vday >= 28), n_va, va),
                    ):
                        idx = np.flatnonzero(d[k]["train_ok"] & sel_)
                        dst.append(
                            (m, k, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)))
                        )
                    continue
                for lo, hi, n, dst in ((0, 14, n_tr, tr), (21, 99, n_va, va)):
                    idx = np.flatnonzero(d[k]["train_ok"] & (dom >= lo) & (dom < hi))
                    dst.append(
                        (m, k, np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)))
                    )
    train, val = Pool(parts=tr), Pool(parts=va)
    x_tr, age_tr, stats = make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    if arm["inputs"] == "noise":  # the validation noise uses the training mean and variance
        raw_stats = (train.raw.mean(0), np.maximum(train.raw.std(0), 1e-3))
        rv = np.random.default_rng(seed + 2000).standard_normal(val.raw.shape).astype(np.float32)
        r = (raw_stats[0] + raw_stats[1] * rv).astype(np.float32)
        r[:, N_FEAT] = np.log1p(np.clip(np.expm1(r[:, N_FEAT]), 0.0, 120.0))
        age_va = np.expm1(np.clip(r[:, N_FEAT], 0.0, None)).astype(np.float32)
        x_va = np.clip((r - stats[0]) / stats[1], -10.0, 10.0).astype(np.float32)
    torch.manual_seed(seed)
    n_in = x_tr.shape[1] + 3
    head = h.Head(n_in=n_in, kind=arm["kind"])  # ty: ignore[invalid-argument-type]
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, BATCH):
            ix = perm[a : a + BATCH]
            x, q, age, lead, y = batch_inputs(x_org=x_tr, age_org=age_tr, pool=train, ix=ix)
            qq, _, _ = head(x=x, q=q, age=age, lead_h=lead, short_weight=arm["short"])
            loss = h.pinball(y=y, q=qq).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = 0.0
            for a in range(0, len(val.y), 20000):
                ix = np.arange(a, min(a + 20000, len(val.y)))
                x, q, age, lead, y = batch_inputs(x_org=x_va, age_org=age_va, pool=val, ix=ix)
                qq, _, _ = head(x=x, q=q, age=age, lead_h=lead, short_weight=arm["short"])
                vl += float(h.pinball(y=y, q=qq).sum())
            vl /= len(val.y)
        log.append(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    res: dict[str, np.ndarray] = {}
    for kind in ("real", "bank"):
        z = load(month=month)[kind]
        idx = np.flatnonzero(z["usable"])
        pool = Pool(parts=[(month, kind, idx)])
        x_te, age_te, _ = make_inputs(raw=pool.raw, arm=arm, stats=stats, seed=seed + 3)
        if arm["inputs"] == "noise":
            rv = (
                np.random.default_rng(seed + 3000)
                .standard_normal(pool.raw.shape)
                .astype(np.float32)
            )
            r = (train.raw.mean(0) + np.maximum(train.raw.std(0), 1e-3) * rv).astype(np.float32)
            r[:, N_FEAT] = np.log1p(np.clip(np.expm1(r[:, N_FEAT]), 0.0, 120.0))
            age_te = np.expm1(np.clip(r[:, N_FEAT], 0.0, None)).astype(np.float32)
            x_te = np.clip((r - stats[0]) / stats[1], -10.0, 10.0).astype(np.float32)
        qs, ds, ss = [], [], []
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                ix = np.arange(a, min(a + 50000, len(idx)))
                x, q, age, lead, y = batch_inputs(x_org=x_te, age_org=age_te, pool=pool, ix=ix)
                qq, d, s = head(x=x, q=q, age=age, lead_h=lead, short_weight=arm["short"])
                qs.append(qq.numpy())
                ds.append(d.numpy())
                ss.append(s.numpy())
        q_all = np.concatenate(qs)
        m = metrics(y=pool.y, q=q_all)
        res[f"{kind}_idx"] = idx.astype(np.int32)
        res[f"{kind}_d"] = np.concatenate(ds).astype(np.float32)
        res[f"{kind}_sigma"] = np.concatenate(ss).astype(np.float32)
        for k, v in m.items():
            res[f"{kind}_{k}"] = v
    np.savez(
        out,
        **res,  # ty: ignore[invalid-argument-type]
        val_log=np.array(log),
        n_train=n,
        n_val=len(val.y),
        best=best,
        seconds=time.time() - t0,
    )
    torch.save(head.state_dict(), out_dir / f"{out.stem}.pt")
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=["2025-10", "2025-12"])
    ap.add_argument("--variants", nargs="+", default=["lomo", "fc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [
        (arm, m, v, s)
        for v in a.variants for m in a.months for arm in a.arms for s in a.seeds
        if not (v.startswith("fc") and m == MONTHS[0])
    ]  # fmt: skip
    print(f"{len(jobs)} fits", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
