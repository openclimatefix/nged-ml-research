# ruff: noqa: E501, E402, ANN001, ANN201, PLR0915, PT018
r"""Experiment 566: the coarse-refit arms C0 to C4 and the real-event leak harness.

Rows, cutoffs, recency weights, early stopping, lag columns, block dropout, the expected-input table
and the weather and lag-time assertions are experiment 560's (imported, which also installs 560's
patches on experiment 511's harness). The real-event harness is experiment 556's `real_events`,
ported to the lag-encoder network. Step 2 (the three-step schedule) is experiment 559's recipe.

Arms (all fine-tuned at one cutoff per fold, the month's first, and scored over the whole month):

- `C0`: experiment 545's lag-free M0 (experiment 554's pretrained weights, fresh mixer);
- `C1`: LAG-v3 (experiment 560; its day-0 weights are reused);
- `C2`: C1 with the three-step schedule (fine-tune from the step-2 weights);
- `C3`: C1 with a 3-day embargo on the newest fine-tuning targets;
- `C4`: C2 with the embargo.

Commands: `step2`, `coarse`, `events`. Run with the PYTHONPATH of every experiment folder, e.g.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00566-*/e00566_fit.py coarse --arms C0 C2
```
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_argv = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=("step2", "coarse", "events"))
parser.add_argument("--arms", nargs="+", default=["C0", "C1", "C2", "C3", "C4"])
parser.add_argument("--months", nargs="+", default=None)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--limit", type=int, default=None, help="events: stop after this many events")
parser.add_argument(
    "--min-age",
    type=float,
    default=1.0,
    help="events: days between the event's end and the refit cutoff",
)
ARGS = parser.parse_args(_argv[1:])
sys.argv = [_argv[0], "fit", "--arm", "v3"]  # experiment 560 parses its own arguments at import

import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00559_model as m559  # ty: ignore[unresolved-import]
import e00560_fit as b  # ty: ignore[unresolved-import]
import e00560_model as m560  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

sys.argv = _argv
DD = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = (
    DD / "00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak"
)
W: Final[Path] = WORK / "work"
W560: Final[Path] = b.W
W554: Final[Path] = DD / "00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/work"
NRA_SHARES: Final[Path] = (
    DD
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails/work/nra_correction_share.npz"
)
"""Experiment 435's NRA correction (share of capacity, 16 x grid) at depth factor 1.0."""
FRESH_LR: Final[float] = 5e-4
EMBARGO_DAYS: Final[int] = 3
HALF_LIFE: Final[float] = 90.0
UNIFORM_HALF_LIFE: Final[float] = 1e12
WINDOW_DAYS: Final[int] = 7
DAY_US: Final[int] = c.DAY_US
MONTHS: Final[list[str]] = ARGS.months or list(c.MONTHS)


@dataclass(frozen=True)
class Arm:
    """How an arm fits: its network, where fine-tuning starts, and whether the newest targets are embargoed."""

    tag: str
    lag: bool
    step2: bool
    embargo: bool


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("C0", lag=False, step2=False, embargo=False),
        Arm("C1", lag=True, step2=False, embargo=False),
        Arm("C2", lag=True, step2=True, embargo=False),
        Arm("C3", lag=True, step2=False, embargo=True),
        Arm("C4", lag=True, step2=True, embargo=True),
    )
}
SIBLINGS: Final[tuple[str, ...]] = b.SIBLINGS


def new_net(*, arm: Arm, n_primaries: int, stage: str = "ft") -> torch.nn.Module:
    """Return an untrained network of the arm's kind (the lag network has the 45 input columns)."""
    if arm.lag:
        return b.new_model(
            outlier=False, n_inputs=m560.N_INPUTS, n_primaries=n_primaries, stage=stage
        )
    return m559.ModularMdn(n_primaries=n_primaries, stage=stage)


def pretrained_path(*, arm: Arm, month: str, seed: int) -> Path:
    """Return the arm's pretrained weights (experiment 560's LAG-v3, or experiment 554's M0), read only."""
    p = (
        W560 / "pretrained" / "LAG-v3" if arm.lag else W554 / "pretrained" / "M0"
    ) / f"{month}_s{seed}.pt"
    assert p.exists(), p
    return p


def step2_path(*, month: str, seed: int) -> Path:
    """Return the saved step-2 weights of the lag network."""
    return W / "step2" / f"{month}_s{seed}.pt"


def model_path(*, tag: str, month: str, key: str, seed: int) -> Path:
    """Return a saved fine-tuned model."""
    return W / "models" / tag / f"{month}_{key}_s{seed}.pt"


def train(*, model, x, z, w, p, is_val, seed: int, lr: float, fresh_lr: float | None, device: str):
    """Experiment 560's loop, with the fresh mixer's own rate only when `fresh_lr` is given.

    Fine-tuning from step-2 weights keeps the mixer at the ordinary rate, as experiment 559's A1 does.
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > c.VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[: c.VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    mixer = list(model.mixer_ft.parameters()) if fresh_lr is not None else []
    mixer_ids = {id(q) for q in mixer}
    groups = [{"params": [q for q in model.parameters() if id(q) not in mixer_ids], "lr": lr}]
    if mixer:
        groups.append({"params": mixer, "lr": fresh_lr})
    optimiser = torch.optim.AdamW(groups, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(c.ROW_FRACTION * len(order))]
        for start in range(0, len(order), c.BATCH):
            bi = order[start : start + c.BATCH]
            nll = b.row_nll(model=model, x=x[bi], p=p[bi], z=z[bi])
            loss = (w[bi] * nll).sum() / w[bi].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        b.row_nll(model=model, x=x[i], p=p[i], z=z[i])
                        for i in val_idx.split(c.BATCH * 2)
                    ]
                ).mean()
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


class Cell:
    """One fold: its rows (with lag columns and all of experiment 560's assertions), its GPU copy, and its test rows."""

    def __init__(self, *, month: str, device: str) -> None:
        """Build the month once."""
        self.month, self.device = month, device
        self.clean = c.clean_stack()
        self.mr = k.build_month(month=month, clean=self.clean)
        self.g = k.GpuMonth(mr=self.mr, device=device)
        self.test = b.data.build_test(month=month, standardiser=self.mr.base.standardiser)
        self.last_day = (k.cutoffs(month=month, cadence="d")[-1] - c.month_start(month=month)).days
        self.shares = np.load(NRA_SHARES)["f1.0"]
        self.mu_t = torch.from_numpy(self.mr.base.target_mean).to(device)
        self.sd_t = torch.from_numpy(self.mr.base.target_std).to(device)
        self.n_primaries = len(self.mr.base.target_mean)

    def select(
        self, *, arm: Arm, cut: datetime, half_life: float = HALF_LIFE, embargo: bool | None = None
    ) -> c.Refit:
        """Return the cutoff's rows; the embargo drops targets in the newest 3 days before the cutoff."""
        r = c.select_rows(rows=self.mr, cutoff=cut, half_life_days=half_life)
        assert r.report["rows_past_valid_cut"] == 0 and r.report["rows_past_origin_cut"] == 0
        if arm.embargo if embargo is None else embargo:
            keep = self.mr.valid_us[r.index] <= c.us(t=cut) - EMBARGO_DAYS * DAY_US
            r = c.Refit(r.index[keep], r.y[keep], r.weight[keep], r.is_val[keep], r.report)
        return r

    def fit(
        self, *, arm: Arm, cut: datetime, seed: int, shift: np.ndarray | None = None
    ) -> torch.nn.Module:
        """Fine-tune the arm at `cut`; `shift` (capacity fraction per selected row) corrects the targets."""
        refit = self.select(arm=arm, cut=cut)
        shift_z = None
        if shift is not None:
            shift_z = shift / self.mr.base.target_std[self.mr.primary[refit.index]]
        b.CTX.update(on=False, shift=None, sign=None, block=None, seed=seed, cut=cut)
        t = self.g.refit_tensors(refit=refit, shift_z=shift_z)
        model = new_net(arm=arm, n_primaries=self.n_primaries).to(self.device)
        if arm.step2:
            model.load_state_dict(
                torch.load(step2_path(month=self.month, seed=seed), map_location="cpu")
            )
            fresh = None
        else:
            b.load_pretrained(
                model=model, path=pretrained_path(arm=arm, month=self.month, seed=seed)
            )
            fresh = b.MIXER_LR
        model.to(self.device)
        train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
              lr=c.FINE_TUNE_LR, fresh_lr=fresh, device=self.device)  # fmt: skip
        del t
        return model

    def day(self, *, cut: datetime) -> int:
        """Return the cutoff's day of the month."""
        return (cut - c.month_start(month=self.month)).days

    def load(self, *, arm: Arm, path: Path) -> torch.nn.Module:
        """Load a saved fine-tuned model."""
        model = new_net(arm=arm, n_primaries=self.n_primaries)
        model.load_state_dict(torch.load(path, map_location=self.device))
        return model.to(self.device).eval()

    def control(self, *, arm: Arm, cut: datetime, seed: int) -> torch.nn.Module:
        """Return the uncorrected fit at a cutoff: the stored one where the recipe matches, else fitted once and saved."""
        day = self.day(cut=cut)
        if arm.tag == "C1":
            stored = W560 / "models" / "LAG-v3" / f"{self.month}_d{day}_s{seed}.pt"
            assert stored.exists(), stored
            return self.load(arm=arm, path=stored)
        path = model_path(tag=arm.tag, month=self.month, key=f"control_d{day}", seed=seed)
        if path.exists():
            return self.load(arm=arm, path=path)
        model = self.fit(arm=arm, cut=cut, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
        return model.eval()

    def median_rows(
        self, *, model: torch.nn.Module, x: torch.Tensor, p: torch.Tensor
    ) -> np.ndarray:
        """Return the model's median (capacity fraction) for rows on the GPU."""
        model.table = b.CTX.get("table_ft") if hasattr(model, "table") else None
        out = []
        for a in range(0, len(x), 262_144):
            q = c.serving_quantiles(
                model=model, x=x[a : a + 262_144], p=p[a : a + 262_144], outlier=False
            )
            q = torch.sort(q, dim=1).values[:, 5]
            pp = p[a : a + 262_144]
            out.append((q * self.sd_t[pp] + self.mu_t[pp]).float().cpu().numpy())
        return np.concatenate(out)

    def window_rows(self, *, cut: datetime) -> np.ndarray:
        """Return the indices of test rows whose origin falls in the 7 days from `cut`."""
        lo, hi = c.us(t=cut), c.us(t=cut) + WINDOW_DAYS * DAY_US
        return np.nonzero((self.test.origin >= lo) & (self.test.origin < hi))[0]

    def q50(self, *, model: torch.nn.Module, rows: np.ndarray) -> np.ndarray:
        """Return the median forecast (capacity fraction) of test rows."""
        x = torch.from_numpy(self.test.x[rows]).to(self.device)
        p = torch.from_numpy(self.test.primary[rows]).to(self.device)
        return self.median_rows(model=model, x=x, p=p)


def step2(*, device: str) -> None:
    """Train and save the lag network's step-2 weights: encoders and a fresh mixer on every row to the first cutoff.

    Experiment 559's recipe: uniform weights, early stopping on the usual weeks, fresh mixer, no
    recency weighting, rows up to the fold's first cutoff minus 3.5 hours.
    """
    arm = ARMS["C2"]
    for month in MONTHS:
        todo = [s for s in ARGS.seeds if not step2_path(month=month, seed=s).exists()]
        if not todo:
            continue
        cell = Cell(month=month, device=device)
        cut = c.month_start(month=month)
        refit = cell.select(arm=arm, cut=cut, half_life=UNIFORM_HALF_LIFE, embargo=False)
        for seed in todo:
            b.CTX.update(on=False, shift=None, sign=None, block=None, seed=seed, cut=cut)
            t = cell.g.refit_tensors(refit=refit)
            assert float(t["w"].min()) > 0.999, "step 2 must have no recency weighting"
            started = time.monotonic()
            model = new_net(arm=arm, n_primaries=cell.n_primaries).to(device)
            b.load_pretrained(model=model, path=pretrained_path(arm=arm, month=month, seed=seed))
            model.to(device)
            epochs, best = train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"],
                                 seed=seed, lr=c.FINE_TUNE_LR, fresh_lr=FRESH_LR, device=device)  # fmt: skip
            target = step2_path(month=month, seed=seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
            c.write_json(obj={"epochs": epochs, "val_nll": best, "rows": len(refit.index), **refit.report},
                         path=W / "logs" / "step2" / f"{month}_s{seed}.json")  # fmt: skip
            print(f"step 2 {month} s{seed}: {len(refit.index):,} rows, {epochs} epochs, val NLL {best:.4f}, "
                  f"{time.monotonic() - started:.0f} s; latest valid "
                  f"{datetime.fromtimestamp(cell.mr.valid_us[refit.index].max() / 1e6, UTC)}", flush=True)  # fmt: skip
            del t, model
        del cell
        torch.cuda.empty_cache()


def coarse(*, device: str) -> None:
    """Fit each arm once per fold at the month's first cutoff and forecast the whole month, with removed and aged variants."""
    for month in MONTHS:
        cell = Cell(month=month, device=device)
        cut = c.month_start(month=month)
        lo, hi = c.us(t=cut), c.us(t=c.month_stop(month=month))
        for tag in ARGS.arms:
            arm = ARMS[tag]
            for seed in ARGS.seeds:
                out = W / "seed_preds" / tag / f"{month}_s{seed}.parquet"
                if out.exists():
                    continue
                started = time.monotonic()
                mfile = model_path(tag=tag, month=month, key="coarse", seed=seed)
                if tag == "C1":
                    model = cell.control(arm=arm, cut=cut, seed=seed)  # experiment 560's day-0 fit
                elif mfile.exists():
                    model = cell.load(arm=arm, path=mfile)
                else:
                    model = cell.fit(arm=arm, cut=cut, seed=seed)
                    mfile.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), mfile.with_suffix(".tmp"))
                    mfile.with_suffix(".tmp").rename(mfile)
                b.CTX["seed"] = seed
                if arm.lag:
                    frame = b.window_frame(
                        model=model,
                        test=cell.test,
                        mr=cell.mr,
                        lo=lo,
                        hi=hi,
                        flag=False,
                        device=device,
                    )
                    extra = dict(b.CTX["extra"])
                else:
                    frame = b._window_frame(
                        model=model,
                        test=cell.test,
                        mr=cell.mr,
                        lo=lo,
                        hi=hi,
                        flag=False,
                        device=device,
                    )
                    extra = {}
                assert frame is not None
                for name, fr in extra.items():
                    b._atomic_frame(frame=fr, target=W / "seed_preds" / f"{tag}-{name}" / out.name)
                b._atomic_frame(frame=frame, target=out)
                print(
                    f"{tag} {month} s{seed}: {frame.height} rows, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
                del model
        del cell
        torch.cuda.empty_cache()
    print("CHECKS", {a: v for a, v in b.CHECK.items() if a != "months"}, flush=True)
    c.write_json(obj=b.CHECK, path=WORK / f"checks_coarse_{'-'.join(ARGS.arms)}.json")


def real_events(*, device: str) -> None:
    """Experiment 556's real-event harness: fit on metered targets and on NRA-corrected targets, per event.

    The cutoff is the first daily cutoff `--min-age` days or more after the event's end (experiment
    556 used 3 days; 1 day lets a 3-day embargo drop the event's newest rows). The event ends before
    the refit cut minus 3.5 hours (asserted). The correction is experiment 435's labelled step (depth
    factor 1.0), subtracted from the training targets of that one event only; the lag inputs stay as
    observed. The absorbed share is the event's sign times (metered median minus corrected median)
    over the 7 days after the cutoff at the event's Primary, divided by the labelled depth. The
    step-2 weights are not refitted per event, so an event that ends before the fold's first cutoff
    sits in step 2 in both fits; `in_step2` marks those events.
    """
    events = c.labelled_events()
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    out = W / "scores" / "real_events.parquet"
    rows_out: list[dict] = pl.read_parquet(out).to_dicts() if out.exists() else []
    seen = {
        (r["arm"], r["month"], r["primary"], r["onset"], r["seed"], r["min_age"]) for r in rows_out
    }
    count = 0
    seed = ARGS.seeds[0]
    for month in MONTHS:
        cell = Cell(month=month, device=device)
        cuts = k.cutoffs(month=month, cadence="d")
        first_cut = c.month_start(month=month)
        wanted: dict[datetime, list[tuple[int, int, int]]] = {}
        for p, spans in events.items():
            for onset, end in spans:
                end_us = grid + end * step
                depth = float(np.abs(cell.shares[p, onset : end + 1]).max())
                if depth <= 0:
                    continue
                ok = [x for x in cuts if c.us(t=x) >= end_us + ARGS.min_age * DAY_US]
                if (
                    ok
                    and (ok[0] - first_cut).days <= cell.last_day
                    and c.us(t=ok[0]) - end_us <= (14 + cell.last_day) * DAY_US
                ):
                    wanted.setdefault(ok[0], []).append((p, onset, end))
        print(
            f"{month}: {sum(len(v) for v in wanted.values())} events at {len(wanted)} cutoffs",
            flush=True,
        )
        for tag in ARGS.arms:
            arm = ARMS[tag]
            for cut, evs in sorted(wanted.items()):
                rows = cell.window_rows(cut=cut)
                todo = [e for e in evs if (tag, month, e[0], e[1], seed, ARGS.min_age) not in seen]
                if len(rows) == 0 or not todo:
                    continue
                model_m = cell.control(arm=arm, cut=cut, seed=seed)
                q_m = cell.q50(model=model_m, rows=rows).astype(np.float64)
                del model_m
                prim_rows = cell.test.primary[rows]
                refit = cell.select(arm=arm, cut=cut)
                prim = cell.mr.primary[refit.index]
                tgt = cell.mr.target_idx[refit.index]
                for p, onset, end in todo:
                    end_us = grid + end * step
                    assert end_us <= c.us(t=cut) - c.CUTOFF_AGE_US, (
                        "an event ends after the refit cut minus 3.5 h"
                    )
                    corrected = -cell.shares[prim, np.clip(tgt, 0, cell.shares.shape[1] - 1)]
                    corrected = np.where(
                        (prim == p) & (tgt >= onset) & (tgt <= end), corrected, 0.0
                    )
                    model_c = cell.fit(
                        arm=arm, cut=cut, seed=seed, shift=corrected.astype(np.float32)
                    )
                    diff = q_m - cell.q50(model=model_c, rows=rows).astype(np.float64)
                    sign = 1 if float(cell.shares[p, onset : end + 1].sum()) > 0 else -1
                    depth = float(np.abs(cell.shares[p, onset : end + 1]).max())
                    sel = prim_rows == p
                    rows_out.append({
                        "arm": tag, "month": month, "seed": seed, "min_age": ARGS.min_age, "cut": str(cut.date()),
                        "primary": p, "onset": onset, "end": end, "depth_points": 100 * depth,
                        "age_days": (c.us(t=cut) - end_us) / c.DAY_US, "n_events_in_fit": len(evs),
                        "n_rows_corrected": int((corrected != 0).sum()),
                        "in_step2": bool(end_us <= c.us(t=first_cut) - c.CUTOFF_AGE_US),
                        "absorbed_share": float(sign * diff[sel].mean() / depth) if sel.any() else None,
                    })  # fmt: skip
                    del model_c
                    count += 1
                    if count % 10 == 0:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        pl.DataFrame(rows_out).write_parquet(out)
                        print("events done", count, flush=True)
                    if ARGS.limit and count >= ARGS.limit:
                        break
                if ARGS.limit and count >= ARGS.limit:
                    break
        del cell
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows_out).write_parquet(out)
    print("real events", len(rows_out), flush=True)


if __name__ == "__main__":
    if ARGS.command == "step2":
        step2(device="cuda")
    elif ARGS.command == "coarse":
        coarse(device="cuda")
    else:
        real_events(device="cuda")
