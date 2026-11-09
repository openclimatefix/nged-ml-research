# ruff: noqa: E501, ANN001, PLR0912, PLR0915
"""Experiment 655: fine-tune the member-level explicit solar networks and forecast the test months.

The recipe is experiment 644's (`e00644_fit`), run on its real-world rows: a network starts from
experiment 639's pretrained weights, is fine-tuned on the 24 months of ensemble rows before the
month with 90-day recency weights, and forecasts every run of the month at lead up to 240 hours.
Only the explicit curve `u` differs (see `e00655_curves`). Arms:

- `k` (c0) mean-sky curve, air-temperature derate; `c` member-mean curve, air derate;
- `d` member-mean curve, cell temperature; `e` as `d` plus the member standard deviation of the
  curve as one extra mixer input; `p` capacity-permuted placebo and `s` shuffled-day placebo, both
  on the `d` curve.
Arms `a` (free) and `b` (experiment 644's explicit network) are read from experiment 644's fits.

Usage: python e00655_fit.py fit --cutoff 2026-06 [--arms k c d e p s] [--seeds 0 1] [--smoke]
"""

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
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
import e00655_curves as cv
import e00655_extract as ex
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn

NAME: Final = ex.NAME
WORK: Final[Path] = ex.WORK
DEVICE = "cuda"
CUTOFFS: Final = k639.CUTOFFS
MONTHS: Final = f644.MONTHS
ARMS: Final[dict[str, dict]] = {
    "k": {"variant": "c0", "pre": "cal", "shuffle": False},
    "c": {"variant": "c", "pre": "cal", "shuffle": False},
    "d": {"variant": "d", "pre": "cal", "shuffle": False},
    "e": {"variant": "d", "pre": "cal", "shuffle": False, "spread": True},
    "p": {"variant": "d", "pre": "perm", "shuffle": False},
    "s": {"variant": "d", "pre": "shuf", "shuffle": True},
}  # fmt: skip
"""Arm letter -> curve variant, experiment 639's pretrained arm name, and options."""
WINDOW_DAYS: Final = 730


class PvSpreadMdn(m.ModularMdn):
    """Experiment 554's modular network whose fine-tuning mixer takes one extra input column (the last of `x`)."""

    def __init__(self, *, n_primaries: int) -> None:
        """Build the network with a fine-tuning mixer one input wider."""
        super().__init__(n_primaries=n_primaries, stage="ft")
        wide = 2 * m.ENCODER_OUT + m.EMBEDDING
        self.mixer_ft = base.mlp(
            n_in=wide + 7 + 2 + 1, hidden=m.MIXER_HIDDEN, layers=2, n_out=3 * m.COMPONENTS, dropout=m.DROPOUT
        )
        base.init_head(head=self.mixer_ft, components=m.COMPONENTS)

    def forward(self, x: torch.Tensor, primary: torch.Tensor | None = None) -> torch.Tensor:
        """Return the mixture's raw outputs; `x[:, -1]` is the standardised spread of the curve."""
        assert primary is not None
        emb = self.embedding(primary)
        hw = nn.functional.gelu(self.weather(torch.cat([x[:, m.WEATHER_COLS], emb], dim=-1)))
        hd = nn.functional.gelu(self.datetime(torch.cat([x[:, m.DATETIME_COLS], emb], dim=-1)))
        core = torch.cat([hw, hd, emb], dim=-1)
        return self.mixer_ft(torch.cat([core, x[:, m.SPREAD_COLS], x[:, m.LEAD_COLS], x[:, -1:]], dim=-1))


class Bank:
    """The curve tables of one cutoff, looked up by Primary, origin, and valid time."""

    def __init__(self, *, cutoff: str) -> None:
        """Load the tables and compute the per-variant normalising constants (99th percentile)."""
        self.tables: dict[int, dict[str, np.ndarray]] = {}
        for p in k634.six_indices().values():
            z = np.load(cv.tables_path(cutoff=cutoff, primary=p))
            self.tables[p] = {k: z[k] for k in z.files}
        start = datetime.fromisoformat(f"{cutoff}-01").replace(tzinfo=UTC)
        first = (start - timedelta(days=WINDOW_DAYS) - ex.ORIGIN_DELAY - ex.FIRST_RUN).days
        last = (start - ex.ORIGIN_DELAY - ex.FIRST_RUN).days
        lo, hi = max(first, 0), last
        self.norm: dict[str, float] = {}
        for v in ("c0", "c", "d"):
            vals = np.concatenate([t[v][lo:hi].ravel() for t in self.tables.values()])
            vals = vals[vals > 0.0]
            self.norm[v] = float(np.quantile(vals, 0.99))
        self.norm["sd"] = self.norm["d"]

    def lookup(
        self, *, variant: str, primary: np.ndarray, origin_us: np.ndarray, valid_us: np.ndarray
    ) -> np.ndarray:
        """Return the normalised curve (or its spread) of each row; zero outside the six Primaries."""
        out = np.zeros(len(primary), dtype=np.float64)
        run = (origin_us.astype(np.int64) - cv.FIRST_ORIGIN_US) // cv.DAY_US
        step = (valid_us.astype(np.int64) - origin_us.astype(np.int64)) // cv.STEP_US - 1
        for p, tab in self.tables.items():
            sel = primary == p
            if not sel.any():
                continue
            r, j = run[sel], step[sel]
            assert r.min() >= 0 and r.max() < tab[variant].shape[0], (p, r.min(), r.max())
            assert j.min() >= 0 and j.max() < ex.N_STEPS, (p, j.min(), j.max())
            out[sel] = tab[variant][r, j] / self.norm[variant]
        return out


def fits_path(*, cutoff: str, month: str, arm: str, seed: int) -> Path:
    """Return where a fine-tuned network's weights are stored (its quantiles sit beside it, `.npz`)."""
    return WORK / "fits" / cutoff / f"{month}_real_{arm}_s{seed}.pt"


def _f32(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE, torch.float32)


def run_month(
    *, cutoff: str, month: str, arms: list[str], seeds: list[int], clean: np.ndarray, bank: Bank, smoke: bool,
) -> None:  # fmt: skip
    """Fine-tune and forecast one month of one cutoff's test window (real world)."""
    six = k634.six_indices()
    refit = c.month_start(month=month)
    mr = c.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    mean = torch.from_numpy(mr.base.target_mean).to(DEVICE, torch.float32)
    std = torch.from_numpy(mr.base.target_std).to(DEVICE, torch.float32)
    sel = c.select_rows(rows=mr, cutoff=refit, half_life_days=f644.HALF_LIFE_DAYS)
    assert sel.report["rows_past_valid_cut"] == 0 and sel.report["rows_past_origin_cut"] == 0
    idx = sel.index
    if smoke:
        idx = idx[::20]
    inj = k638.Injector(
        primary=mr.primary[idx], target_idx=mr.target_idx[idx], x=mr.x[idx], standardiser=mr.base.standardiser
    )
    keep = ~inj.gap
    y = mr.y[idx].astype(np.float64)
    blk = d639._decorate(
        cutoff=cutoff, world="real", x=mr.x[idx], y=y, primary=mr.primary[idx],
        target_idx=mr.target_idx[idx], is_val=mr.is_val[idx], inj=inj,
    )  # fmt: skip
    tr = d639._select(block=blk, keep=keep)
    w_all = (sel.weight[::20] if smoke else sel.weight)[keep]
    donor = d639._shuffle_donors(primary=mr.primary[idx], t_us=inj.t_us)
    origin_tr, valid_tr = mr.origin_us[idx], inj.t_us
    # Test rows: as experiment 644 (the same selection), with the curve looked up by origin and valid time.
    ok = np.isin(test.primary, list(six.values())) & (test.lead >= f644.MIN_LEAD_H)
    y_test = clean[test.primary, test.target_idx]
    ok &= np.isfinite(y_test)
    tid = np.flatnonzero(ok)
    tinj = k638.Injector(
        primary=test.primary[tid], target_idx=test.target_idx[tid], x=test.x[tid], standardiser=mr.base.standardiser
    )
    tblk = d639._decorate(
        cutoff=cutoff, world="real", x=test.x[tid], y=y_test[tid].astype(np.float64),
        primary=test.primary[tid], target_idx=test.target_idx[tid],
        is_val=np.zeros(len(tid), dtype=bool), inj=tinj,
    )  # fmt: skip
    tkeep = ~tinj.gap
    tdonor = d639._shuffle_donors(primary=test.primary[tid], t_us=tinj.t_us)
    origin_te, valid_te = test.origin[tid], tinj.t_us
    tblk = d639._select(block=tblk, keep=tkeep)
    tid = tid[tkeep]
    if not smoke:  # the rows must be experiment 644's
        z = np.load(f644.rows_path(month=month, world="real"))
        assert np.array_equal(z["primary"], tblk.primary) and np.array_equal(z["t_us"], tblk.t_us)
        assert np.array_equal(z["origin_us"], test.origin[tid]) and np.allclose(z["y"], tblk.y)
    curves_tr = {
        v: bank.lookup(variant=v, primary=mr.primary[idx], origin_us=origin_tr, valid_us=valid_tr)
        for v in cv.VARIANTS
    }
    pre_primary = test.primary[np.flatnonzero(ok)]
    pre_origin, pre_valid = origin_te, valid_te
    curves_te_pre = {
        v: bank.lookup(variant=v, primary=pre_primary, origin_us=pre_origin, valid_us=pre_valid)
        for v in cv.VARIANTS
    }
    if not smoke:
        np.savez(
            WORK / "curves" / f"test_curves_{cutoff}_{month}.npz",
            **{v: curves_te_pre[v][tkeep] for v in cv.VARIANTS}, primary=tblk.primary, t_us=tblk.t_us,
        )  # fmt: skip
    inc_tr, inc_te = tr.inc, tblk.inc
    g = {
        "x": torch.from_numpy(tr.x).to(DEVICE), "p": torch.from_numpy(tr.primary).to(DEVICE),
        "is_val": torch.from_numpy(tr.is_val).to(DEVICE), "y": _f32(tr.y), "r4": _f32(tr.r4),
        "after": _f32(tr.after), "w": torch.from_numpy(w_all.astype(np.float32)).to(DEVICE),
    }  # fmt: skip
    for arm in arms:
        spec = ARMS[arm]
        v = spec["variant"]
        u_tr_full = curves_tr[v]
        u_tr = (u_tr_full[donor] if spec["shuffle"] else u_tr_full)[keep]
        u_te_full = curves_te_pre[v]
        u_te = (u_te_full[tdonor] if spec["shuffle"] else u_te_full)[tkeep]
        x_tr, x_te = g["x"], tblk.x
        if spec.get("spread"):
            sd_tr = curves_tr["sd"][keep]
            live = (inc_tr > 0) & (u_tr > 0)
            mu, sg = float(sd_tr[live].mean()), float(sd_tr[live].std() + 1e-9)
            col_tr = ((sd_tr - mu) / sg).astype(np.float32)[:, None]
            col_te = ((curves_te_pre["sd"][tkeep] - mu) / sg).astype(np.float32)[:, None]
            x_tr = torch.cat([g["x"], torch.from_numpy(col_tr).to(DEVICE)], dim=1)
            x_te = np.concatenate([tblk.x, col_te], axis=1)
        pre = spec["pre"]
        params = k639.capacity_parameters(cutoff=cutoff, arm=pre)
        tensors = net.parameter_tensors(params=params, six=six, device=DEVICE)
        u_inc_tr = _f32(u_tr * inc_tr)
        u_inc_te = _f32(u_te * inc_te)
        for seed in seeds:
            out = fits_path(cutoff=cutoff, month=month, arm=arm, seed=seed)
            target = out.with_suffix(".npz")
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = (PvSpreadMdn(n_primaries=16) if spec.get("spread") else m.ModularMdn(n_primaries=16, stage="ft")).to(DEVICE)
            pre_path = k639.explicit_weights_path(cutoff=cutoff, world="real", arm=pre, seed=seed)
            f644._load_for_finetune(model=model, path=pre_path)
            model.to(DEVICE)
            js = pre_path.with_suffix(".json")
            init = torch.tensor(json.loads(js.read_text())["delta_mw"], dtype=torch.float32, device=DEVICE) if js.exists() else None
            epochs, best, delta = f644._train(
                model=model, x=x_tr, y=g["y"], u_inc=u_inc_tr, r4=g["r4"], after=g["after"], p=g["p"], w=g["w"],
                is_val=g["is_val"], mean=mean, std=std, tensors=tensors, seed=seed, init_delta=init,
            )  # fmt: skip
            p_test = torch.from_numpy(tblk.primary).to(DEVICE)
            cap = net.capacity_mw(t=tensors, delta=delta, p=p_test, r4=_f32(tblk.r4), after=_f32(tblk.after))
            q = net.forecast(model=model, x=x_te, p=tblk.primary, mean=mean, std=std, subtract=(cap * u_inc_te).cpu(), device=DEVICE)  # fmt: skip
            if smoke:
                print(f"SMOKE {cutoff} {month} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(tr.x):,} rows", flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out.with_suffix(".tmp"))
            out.with_suffix(".tmp").rename(out)
            tmp = target.with_name(target.name + ".tmp.npz")
            np.savez(tmp, q=q.astype(np.float32))
            tmp.rename(target)
            out.with_suffix(".json").write_text(
                json.dumps({"epochs": epochs, "val_nll": best, "seconds": round(time.monotonic() - started), "rows": len(tr.x), "delta_mw": delta.cpu().tolist(), "norm": bank.norm})
            )  # fmt: skip
            print(f"{cutoff} {month} {arm} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {len(tr.x):,} rows", flush=True)
            del model
    del g
    torch.cuda.empty_cache()


def main() -> None:
    """Run the fine-tunes of one cutoff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit",))
    parser.add_argument("--cutoff", required=True, choices=CUTOFFS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k639.SEEDS))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    torch.set_num_threads(2)
    clean = c.clean_stack()
    bank = Bank(cutoff=a.cutoff)
    for month in MONTHS[a.cutoff]:
        run_month(cutoff=a.cutoff, month=month, arms=a.arms, seeds=a.seeds, clean=clean, bank=bank, smoke=a.smoke)
    print(json.dumps({"done": a.phase, "cutoff": a.cutoff, "at": datetime.now(UTC).isoformat()}), flush=True)


if __name__ == "__main__":
    main()
