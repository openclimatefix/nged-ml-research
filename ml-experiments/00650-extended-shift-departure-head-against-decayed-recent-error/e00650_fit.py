# ruff: noqa: E501, ANN001, D103, D107, PLR0915
"""Experiment 650, step 4: fit the departure heads on the first month pair and write per-row scores (CPU).

Arms: `Rnew` (experiment 647's H1 recipe on real rows plus the extended-shift bank), `Rold` (the
same recipe on real rows plus experiment 647's constant-shift bank, refitted here because the stored
weights of experiment 647 were deleted to free disk space), and `Placebo` (Rnew with the shifts
shuffled across the bank rows of the training fold). Same head, loss, optimiser, batch, epochs, and
early-stop patience as experiment 647's `e00647_fit.py` (imported for the head and inputs); 15,000
bank rows instead of 6,000, because one training month supplies the bank rows that four months
supplied there.

Month pair: the scored month is trained on the other month of the pair (the bank passes of the
scored month never enter training). Rows of the training month split by origin day: days 0 to 17
train, days 21 onward early-stop (a gap of 3 days), for real and bank rows alike. The early-stop
month is never the scored month.

Output per fit, `fits/<arm>_<month>_s<seed>.npz`: per row of the scored month's real rows, extended
held-out rows (passes 200 onward), and training-family rows (passes 0 to 11): the head's q05, q50,
q95, CRPS, coverage flags, tail pinball losses, and the row index.

Usage: python e00650_fit.py [--arms Rnew Rold Placebo] [--months 2025-10 2025-12]
[--seeds 0 1] [--workers 6]
"""

import argparse
import multiprocessing as mp
import time
from typing import Final

import e00650_common as c650
import numpy as np

PAIR: Final = ("2025-10", "2025-12")
MONTHS: Final = c650.MONTHS
N_BANK_PER_MONTH: Final[int] = 6_000

N_REAL: Final[int] = 60_000
N_BANK: Final[int] = 15_000
N_VAL: Final[int] = 20_000
N_VAL_BANK: Final[int] = 4_000
ARMS: Final = {"Rnew": "new", "Rold": "old", "Placebo": "new"}
_CACHE: dict = {}
D647 = c650.WORK.parent / "00647-two-output-forecast-departure-head-on-frozen-k1" / "head_data"


def load(*, path, kind: str) -> dict:
    """Return the arrays of one head-data file with the origin-level raw inputs, as experiment 647's `load`."""
    if path not in _CACHE:
        import e00647_data as dat  # ty: ignore[unresolved-import]

        z = dict(np.load(path))
        miss = list(dat.FEATURE_NAMES).index("gap_day_missing")
        age = z["age"].astype(np.float32)
        z["raw"] = np.concatenate([z["feat"], np.log1p(age)[:, None], (age / 30.0)[:, None], z["gate"].astype(np.float32)[:, None], np.clip(z["lag_age"] / 48.0, 0.0, 10.0)[:, None]], axis=1).astype(np.float32)  # fmt: skip
        z["usable"] = (z["feat"][z["org"], miss] == 0.0) & np.isfinite(z["y"])
        if kind == "real":
            z["target"] = z["y"]
            z["train_ok"] = z["usable"] & ~z["flagged"] & ~z["verified"] & ~z["gate"][z["org"]]
        else:
            z["target"] = (z["y"] + z["shift"]).astype(np.float32)
            z["train_ok"] = z["usable"]
        _CACHE[path] = z
    return _CACHE[path]


def sources(*, month: str, bank: str) -> dict:
    """Return the real and bank arrays of `month`; `bank` is `new` (this experiment) or `old` (experiment 647)."""
    bank_path = (
        c650.WORK / "head_data" / f"bank_{month}.npz"
        if bank == "new"
        else D647 / f"bank_{month}.npz"
    )
    return {
        "real": load(path=D647 / f"real_{month}.npz", kind="real"),
        "bank": load(path=bank_path, kind="bank"),
    }


class Pool:
    """Rows of several (array dict, row index) parts with the origin-level raw inputs stacked."""

    def __init__(self, *, parts: list[tuple[dict, np.ndarray, np.ndarray | None]]) -> None:
        cols: dict[str, list] = {k: [] for k in ("q", "y", "lead", "org")}
        raws, off = [], 0
        for z, idx, y_override in parts:
            cols["q"].append(z["q"][idx])
            cols["y"].append(z["target"][idx] if y_override is None else y_override)
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


def pick(
    *, z: dict, kind: str, month: str, lo: int, hi: int, n: int, rng, train_family: bool
) -> np.ndarray:
    ms = c650.month_start_idx(month=month)
    dom = (z["oi"] - ms) // 48
    ok = z["train_ok"] & (dom >= lo) & (dom < hi)
    if kind == "bank" and "pass_" in z and train_family:
        ok &= z["pass_"] < 200
    idx = np.flatnonzero(ok)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))


def fit_one(job: tuple[str, str, int, str]) -> str:
    import e00647_fit as f647  # ty: ignore[unresolved-import]
    import e00647_head as h  # ty: ignore[unresolved-import]
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm_name, month, seed, scope = job
    train_set = PAIR if scope == "pair" else MONTHS
    out_dir = c650.WORK / ("fits" if scope == "pair" else "fits5")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{arm_name}_{month}_s{seed}.npz"
    if out.exists():
        return f"{out.name}: exists"
    t0 = time.time()
    train_months = [m for m in train_set if m != month]
    rng = np.random.default_rng(1000 * seed + int(month.replace("-", "")) % 997)
    tr, va = [], []
    for train_month in train_months:
        src = sources(month=train_month, bank=ARMS[arm_name])
        n_bank = N_BANK if scope == "pair" else N_BANK_PER_MONTH
        n_val_bank = N_VAL_BANK if scope == "pair" else N_VAL_BANK // 2
        for kind, n_tr, n_va in (("real", N_REAL, N_VAL), ("bank", n_bank, n_val_bank)):
            z = src[kind]
            i_tr = pick(
                z=z, kind=kind, month=train_month, lo=0, hi=18, n=n_tr, rng=rng, train_family=True
            )
            i_va = pick(
                z=z, kind=kind, month=train_month, lo=21, hi=99, n=n_va, rng=rng, train_family=True
            )
            y_tr = y_va = None
            if (
                arm_name == "Placebo" and kind == "bank"
            ):  # shuffle the shifts across the bank rows of the fold
                sh = z["shift"].copy()
                sh[i_tr] = sh[i_tr][rng.permutation(len(i_tr))]
                sh[i_va] = sh[i_va][rng.permutation(len(i_va))]
                y_tr, y_va = (
                    (z["y"][i_tr] + sh[i_tr]).astype(np.float32),
                    (z["y"][i_va] + sh[i_va]).astype(np.float32),
                )
            tr.append((z, i_tr, y_tr))
            va.append((z, i_va, y_va))
    train, val = Pool(parts=tr), Pool(parts=va)
    arm = {"kind": "mix", "inputs": "all", "short": 0.5}
    x_tr, age_tr, stats = f647.make_inputs(raw=train.raw, arm=arm, stats=None, seed=seed)
    x_va, age_va, _ = f647.make_inputs(raw=val.raw, arm=arm, stats=stats, seed=seed + 1)
    torch.manual_seed(seed)
    head = h.Head(n_in=x_tr.shape[1] + 3, kind="mix")
    opt = torch.optim.Adam(head.parameters(), lr=f647.LR, weight_decay=f647.WD)
    best, best_state, bad, log = np.inf, None, 0, []
    n = len(train.y)
    for epoch in range(f647.EPOCHS):
        head.train()
        perm = np.random.default_rng(seed * 100 + epoch).permutation(n)
        for a in range(0, n, f647.BATCH):
            ix = perm[a : a + f647.BATCH]
            x, q, age, lead, y = f647.batch_inputs(x_org=x_tr, age_org=age_tr, pool=train, ix=ix)
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
                x, q, age, lead, y = f647.batch_inputs(x_org=x_va, age_org=age_va, pool=val, ix=ix)
                qq, _, _ = head(x=x, q=q, age=age, lead_h=lead, short_weight=arm["short"])
                vl += float(h.pinball(y=y, q=qq).sum())
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
    # score the scored month: real rows, the extended held-out passes, and the extended training-family passes
    test = sources(month=month, bank="new")
    res: dict[str, np.ndarray] = {}
    sets = {
        "real": (test["real"], np.flatnonzero(test["real"]["usable"])),
        "held": (
            test["bank"],
            np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] >= 200)),
        ),
        "fam": (
            test["bank"],
            np.flatnonzero(test["bank"]["usable"] & (test["bank"]["pass_"] < 200)),
        ),
    }
    for name, (z, idx) in sets.items():
        pool = Pool(parts=[(z, idx, None)])
        x_te, age_te, _ = f647.make_inputs(raw=pool.raw, arm=arm, stats=stats, seed=seed + 3)
        qs = []
        with torch.no_grad():
            for a in range(0, len(idx), 50000):
                ix = np.arange(a, min(a + 50000, len(idx)))
                x, q, age, lead, y = f647.batch_inputs(x_org=x_te, age_org=age_te, pool=pool, ix=ix)
                qq, _, _ = head(x=x, q=q, age=age, lead_h=lead, short_weight=arm["short"])
                qs.append(qq.numpy())
        q_all = np.concatenate(qs).astype(np.float32)
        res[f"{name}_idx"] = idx.astype(np.int32)
        res[f"{name}_q"] = q_all
    np.savez(
        out,
        **res,  # ty: ignore[invalid-argument-type]
        val_log=np.array(log),
        n_train=n,
        n_val=len(val.y),
        best=best,
        seconds=time.time() - t0,
    )  # ty: ignore[invalid-argument-type]
    torch.save(head.state_dict(), out_dir / f"{out.stem}.pt")
    return f"{out.name}: {len(log)} epochs, val {best:.5f}, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--months", nargs="+", default=list(PAIR))
    ap.add_argument("--scope", choices=("pair", "all"), default="pair")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(arm, m, s, a.scope) for m in a.months for arm in a.arms for s in a.seeds]
    print(f"{len(jobs)} fits", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for msg in pool.imap_unordered(fit_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
