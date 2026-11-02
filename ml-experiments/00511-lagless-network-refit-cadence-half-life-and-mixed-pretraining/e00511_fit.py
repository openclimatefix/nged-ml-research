# ruff: noqa: E501
r"""Fit and forecast the arms of experiment 511 (refit cadence, recency half-life, mixed pretraining).

Commands:

- `leak`: build each month's rows, select the rows of every refit of every arm, print the leak check
  (rows past the valid-time cut and past the origin cut must be 0, scored windows must tile the month
  and start at their own cutoff). Run it before any fit.
- `time`: fit one daily cutoff and print the feeder time (row selection and tensor build) and the
  training time separately.
- `pretrain-mix`: the mixed ERA5, CAMS, and ENS pretraining of L-mix (5 folds x 2 seeds).
- `fit --arm A`: fine-tune and forecast one arm on the months, seeds, and (with `--stress`) the
  injected-shift cutoffs given. A fit whose cutoff coincides with one of experiment 510's weekly
  fits of the same half-life, seed, and pretrained weights loads 510's saved weights instead of
  refitting; the forecast of the arm's own window is always produced here.
- `assemble`: concatenate each arm's window forecasts into one table per arm, month, and seed
  (experiment 425's layout under `seed_preds/<arm>/`).
- `stale`: inference only. Score each origin with the model refit `k` days earlier (k = 0 to 7).

```bash
source <PYTHONPATH of every folder under ml-experiments/>; export DATA_PATH_INTERNAL=<private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00511-*/e00511_fit.py leak
```
"""

import argparse
import dataclasses
import itertools
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

W: Final[Path] = k.WORK_DIR
E510: Final[Path] = k.E510_WORK
STRESS_BLOCK_DAYS: Final[int] = 7
STRESS_END_DAY: Final[int] = 14
HALF_BATCH: Final[int] = 4_096


@dataclass(frozen=True)
class Arm:
    """How an arm refits: cadence, recency half-life, pretraining, and warm start."""

    tag: str
    cadence: str
    half_life: float
    pretrained: str = "L0PT"
    warm: bool = False

    @property
    def flag(self) -> bool:
        """Whether the network has the source-flag input (only L-mix)."""
        return self.pretrained == "MIX"


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("L-3d", "t", 90.0),
        Arm("L-1d", "d", 90.0),
        Arm("L-Ww-30", "w", 30.0),
        Arm("L-Ww-60", "w", 60.0),
        Arm("L-mix", "w", 90.0, pretrained="MIX"),
        Arm("L-1d-warm", "d", 90.0, warm=True),
        Arm("L-mix-1d", "d", 90.0, pretrained="MIX"),
    )
}


def pretrained_file(*, arm: Arm, month: str, seed: int) -> Path:
    """Return the pretrained weights an arm starts from."""
    if arm.pretrained == "MIX":
        return W / "pretrained" / "MIX" / f"{month}_s{seed}.pt"
    return f510.pretrained_path(v=f510.VARIANTS["hl90"], month=month, seed=seed)


def day_of(*, month: str, cut: datetime) -> int:
    """Return a cutoff's offset in whole days from the month's first day."""
    return (cut - c.month_start(month=month)).days


def model_file(*, arm: Arm, month: str, day: int, seed: int, stress: str = "") -> Path:
    """Return the saved weights of a fit (this experiment's, or 510's when the fit coincides)."""
    own = W / "models" / arm.tag / f"{month}_d{day}_s{seed}{stress}.pt"
    reusable = (
        not stress
        and not arm.warm
        and arm.pretrained == "L0PT"
        and arm.half_life == c.HALF_LIFE_90
        and day in k.WEEKLY_OFFSETS
    )
    old = (
        E510
        / "models"
        / "hl90"
        / f"{month}_w{k.WEEKLY_OFFSETS.index(day) if day in k.WEEKLY_OFFSETS else 0}_s{seed}.pt"
    )
    if reusable and not own.exists() and old.exists():
        return old
    return own


def n_inputs(*, arm: Arm, mr: c.MonthRows) -> int:
    """Return the network's input width."""
    return mr.x.shape[1] + (1 if arm.flag else 0)


def train_one_epoch(
    *, model: torch.nn.Module, t: dict[str, torch.Tensor], seed: int, device: str
) -> float:
    """Warm-start step: one epoch (20% of the training rows) at the fine-tune rate, fresh optimiser, no early stopping."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    order = train_idx[torch.randperm(len(train_idx), device=device)]
    order = order[: int(c.ROW_FRACTION * len(order))]
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=c.FINE_TUNE_LR, weight_decay=c.WEIGHT_DECAY
    )
    model.train()
    last = float("nan")
    for start in range(0, len(order), c.BATCH):
        b = order[start : start + c.BATCH]
        nll = c.row_nll(model=model, x=t["x"][b], p=t["p"][b], z=t["z"][b], outlier=False)
        loss = (t["w"][b] * nll).sum() / t["w"][b].sum()
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
        optimiser.step()
        last = float(loss)
    model.eval()
    return last


def flagged_test(*, arm: Arm, test: data.TestRows) -> data.TestRows:
    """Return the test rows with the source-flag column (1) appended for L-mix arms, built once per month."""
    if not arm.flag:
        return test
    ones = np.ones((len(test.x), 1), np.float32)
    return dataclasses.replace(test, x=np.concatenate([test.x, ones], axis=1))


def window_frame(
    *,
    model: torch.nn.Module,
    test: data.TestRows,
    mr: c.MonthRows,
    lo: int,
    hi: int,
    flag: bool,
    device: str,
) -> pl.DataFrame | None:
    """Return the forecast of origins in [lo, hi) from a model, or None when no test origin falls there."""
    sel = (test.origin >= lo) & (test.origin < hi)
    _ = flag  # the source-flag column is already in `test.x` (see `flagged_test`)
    if not ((test.origin >= lo) & (test.origin < hi)).any():
        return None
    return c.forecast_frame(
        model=model, outlier=False, test=test, select=sel, rows=mr.base, widen=1.0, device=device
    )


def atomic_frame(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)


def leak_check() -> None:
    """Print, per month and cadence, the cut checks and the window tiling; fail on any violation."""
    clean = c.clean_stack()
    bad = 0
    for month in c.MONTHS:
        mr = k.build_month(month=month, clean=clean)
        print(
            f"{month}: {len(mr.y):,} rows ({mr.n_base:,} in experiment 428's fold), {mr.x.shape[1]} inputs",
            flush=True,
        )
        for cadence in k.CADENCES:
            cuts = k.cutoffs(month=month, cadence=cadence)
            starts = [k.window_of(month=month, cadence=cadence, index=i) for i in range(len(cuts))]
            tiles = all(a[1] == b[0] for a, b in itertools.pairwise(starts)) and (
                starts[0][0] == c.us(t=c.month_start(month=month))
                and starts[-1][1] == c.us(t=c.month_stop(month=month))
            )
            own_start = all(starts[i][0] == c.us(t=cuts[i]) for i in range(len(cuts)))
            assert tiles, f"{month} {cadence}: windows do not tile the month"
            assert own_start, f"{month} {cadence}: a window starts before its cutoff"
            worst = {"rows_past_valid_cut": 0, "rows_past_origin_cut": 0}
            for cut in cuts:
                for hl in (30.0, 60.0, 90.0):
                    r = c.select_rows(rows=mr, cutoff=cut, half_life_days=hl)
                    for key, value in worst.items():
                        worst[key] = max(value, r.report[key])
                    independent = int(
                        (mr.valid_us[r.index] > c.us(t=cut) - c.CUTOFF_AGE_US).sum()
                        + (mr.origin_us[r.index] > c.us(t=cut)).sum()
                    )
                    bad += (
                        r.report["rows_past_valid_cut"]
                        + r.report["rows_past_origin_cut"]
                        + independent
                    )
            print(
                f"  cadence {cadence}: {len(cuts)} cutoffs, windows tile the month, no window starts before its cutoff, worst past-cut counts {worst}",
                flush=True,
            )
        del mr
    assert bad == 0, f"{bad} rows past a cut"
    print("leak check passed: 0 rows past any cut")


def fine_tune(
    *, arm: Arm, g: k.GpuMonth, month: str, cut: datetime, seed: int, device: str,
    prev_state: dict | None, shift: np.ndarray | None = None, sign: int = 0,
    block: tuple[int, int] | None = None,
) -> tuple[torch.nn.Module, dict]:  # fmt: skip
    """Return a fine-tuned model and its log for one cutoff (restart from pretrained, or warm start)."""
    mr = g.mr
    started = time.monotonic()
    refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=arm.half_life)
    shift_z = None
    injected = 0
    if shift is not None and block is not None:
        inj = (
            (mr.valid_us[refit.index] >= block[0])
            & (mr.valid_us[refit.index] < block[1])
            & ~refit.is_val
        )
        prim = mr.primary[refit.index]
        shift_z = np.where(inj, sign * shift[prim] / mr.base.target_std[prim], 0.0)
        injected = int(inj.sum())
    t = g.refit_tensors(refit=refit, shift_z=shift_z)
    feeder = time.monotonic() - started
    model = c.new_model(
        outlier=False, n_inputs=n_inputs(arm=arm, mr=mr), n_primaries=len(mr.base.target_mean)
    ).to(device)
    day = day_of(month=month, cut=cut)
    t0 = time.monotonic()
    if arm.warm and prev_state is not None:
        model.load_state_dict(prev_state)
        epochs, best = 1, train_one_epoch(model=model, t=t, seed=seed, device=device)
    else:
        c.load_pretrained(model=model, path=pretrained_file(arm=arm, month=month, seed=seed))
        model.to(device)
        epochs, best = c.train(
            model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
            lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device,
        )  # fmt: skip
    log = {
        "name": f"{arm.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
        "feeder_seconds": round(feeder, 2), "train_seconds": round(time.monotonic() - t0, 2),
        "injected_rows": injected, **refit.report,
    }  # fmt: skip
    del t
    return model, log


def fit_arm(  # noqa: PLR0912, PLR0915
    *, arm: Arm, months: list[str], seeds: list[int], stress: bool, limit: int | None, device: str
) -> None:
    """Fine-tune and forecast an arm's own scored window at each cutoff (or its stress cutoffs)."""
    if stress and arm.warm:
        raise ValueError("stress runs are defined for restart arms only")
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = flagged_test(
            arm=arm, test=data.build_test(month=month, standardiser=mr.base.standardiser)
        )
        g = k.GpuMonth(mr=mr, device=device, flag=1.0 if arm.flag else None)
        cuts = k.cutoffs(month=month, cadence=arm.cadence)
        start_day = c.month_start(month=month)
        if stress:
            end = start_day + timedelta(days=STRESS_END_DAY)
            todo = [
                i
                for i, x in enumerate(cuts)
                if end <= x < end + timedelta(days=7 if arm.cadence == "d" else 1)
            ]
            shift = c.shift_sizes(clean=clean, cutoff=end)
            block = (c.us(t=end - timedelta(days=STRESS_BLOCK_DAYS)), c.us(t=end) - c.CUTOFF_AGE_US)
        else:
            todo = list(range(len(cuts)))[:limit]
            shift, block = None, None
        for seed in seeds:
            prev_state = None
            for i in todo:
                cut = cuts[i]
                day = day_of(month=month, cut=cut)
                for sign in (+1, -1) if stress else (0,):
                    sfx = f"_stress{'p' if sign > 0 else 'n'}" if stress else ""
                    out = W / "fits" / arm.tag / f"{month}_d{day}_s{seed}{sfx}.parquet"
                    mfile = model_file(arm=arm, month=month, day=day, seed=seed, stress=sfx)
                    if (out.exists() or out.with_suffix(".empty").exists()) and mfile.exists():
                        if arm.warm:
                            prev_state = torch.load(mfile, map_location=device)
                        continue
                    started = time.monotonic()
                    if mfile.exists():
                        model = c.new_model(
                            outlier=False,
                            n_inputs=n_inputs(arm=arm, mr=mr),
                            n_primaries=len(mr.base.target_mean),
                        ).to(device)
                        model.load_state_dict(torch.load(mfile, map_location=device))
                        model.eval()
                        log: dict = {
                            "name": f"{arm.tag}/{month}_d{day}_s{seed}",
                            "reused": str(mfile),
                        }
                    else:
                        model, log = fine_tune(
                            arm=arm, g=g, month=month, cut=cut, seed=seed, device=device,
                            prev_state=prev_state, shift=shift, sign=sign, block=block,
                        )  # fmt: skip
                        mfile.parent.mkdir(parents=True, exist_ok=True)
                        tmp_model = mfile.with_suffix(".tmp")
                        torch.save(model.state_dict(), tmp_model)
                        tmp_model.rename(mfile)
                    lo, hi = k.window_of(month=month, cadence=arm.cadence, index=i)
                    frame = window_frame(
                        model=model, test=test, mr=mr, lo=lo, hi=hi, flag=arm.flag, device=device
                    )
                    if frame is None:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.with_suffix(".empty").touch()
                    else:
                        atomic_frame(frame=frame, target=out)
                    log["seconds"] = round(time.monotonic() - started, 2)
                    c.write_json(
                        obj=log, path=W / "logs" / arm.tag / f"{month}_d{day}_s{seed}{sfx}.json"
                    )
                    print(
                        f"{log['name']}{sfx}: {log.get('epochs', 'reused')} epochs, {log['seconds']} s (feeder {log.get('feeder_seconds', 0)} s)",
                        flush=True,
                    )
                    if arm.warm:
                        prev_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
                    del model
        del g, mr, test
        torch.cuda.empty_cache()


def time_one(*, device: str) -> None:
    """Fit the 2026-01 day-10 daily cutoff (seed 0, not saved) and print the feeder and training seconds."""
    clean = c.clean_stack()
    t0 = time.monotonic()
    mr = k.build_month(month="2026-01", clean=clean)
    build = time.monotonic() - t0
    g = k.GpuMonth(mr=mr, device=device)
    upload = time.monotonic() - t0 - build
    arm = ARMS["L-1d"]
    for day in (10, 11, 12):
        cut = c.month_start(month="2026-01") + timedelta(days=day)
        model, log = fine_tune(
            arm=arm, g=g, month="2026-01", cut=cut, seed=0, device=device, prev_state=None
        )
        print(
            f"day {day}: {log['rows']:,} rows, {log['epochs']} epochs, feeder {log['feeder_seconds']} s, training {log['train_seconds']} s"
        )
        del model
    print(f"month build {build:.0f} s, upload {upload:.0f} s")


def pretrain_mix(*, device: str) -> None:
    """Pretrain L-mix: ERA5 and CAMS rows (flag 0) and the fold's ENS rows (flag 1), batches drawn 50/50."""
    clean = c.clean_stack()
    for month in c.MONTHS:
        todo = [s for s in (0, 1) if not (W / "pretrained" / "MIX" / f"{month}_s{s}.pt").exists()]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
        cut_valid, cut_origin = c.us(t=refit_t) - c.CUTOFF_AGE_US, c.us(t=refit_t)
        ens_valid = grid + rows.target_idx.astype(np.int64) * step
        ens_origin = grid + rows.origin_idx.astype(np.int64) * step
        era_valid = grid + pre.target_idx.astype(np.int64) * step
        past = int(
            (ens_valid > cut_valid).sum()
            + (ens_origin > cut_origin).sum()
            + (era_valid > cut_valid).sum()
            + (grid + pre.origin_idx.astype(np.int64) * step > cut_origin).sum()
        )
        print(
            f"{month}: ERA5/CAMS rows {len(pre.x):,} (flag 0), ENS rows {len(rows.x):,} (flag 1); rows past the valid or origin cut: {past}; latest ENS valid {datetime.fromtimestamp(ens_valid.max() / 1e6, UTC).isoformat()}; refit {refit_t.isoformat()}",
            flush=True,
        )
        assert past == 0, "pretraining rows past the cut"
        prim_e = pre.primary
        z_e = (pre.y - rows.target_mean[prim_e]) / rows.target_std[prim_e]
        z_n = (rows.y - rows.target_mean[rows.primary]) / rows.target_std[rows.primary]
        ones = np.ones((len(rows.x), 1), np.float32)
        t_e = {
            "x": torch.from_numpy(np.concatenate([pre.x, np.zeros((len(pre.x), 1), np.float32)], axis=1)).to(device),
            "z": torch.from_numpy(z_e.astype(np.float32)).to(device), "p": torch.from_numpy(prim_e).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }  # fmt: skip
        t_n = {
            "x": torch.from_numpy(np.concatenate([rows.x, ones], axis=1)).to(device),
            "z": torch.from_numpy(z_n.astype(np.float32)).to(device), "p": torch.from_numpy(rows.primary).to(device),
            "v": torch.from_numpy(rows.is_val).to(device),
        }  # fmt: skip
        for seed in todo:
            started = time.monotonic()
            torch.manual_seed(seed)
            model = c.new_model(
                outlier=False, n_inputs=t_e["x"].shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            epochs, best = train_mix(model=model, era=t_e, ens=t_n, seed=seed, device=device)
            target = W / "pretrained" / "MIX" / f"{month}_s{seed}.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  pretrain MIX {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del t_e, t_n, pre, rows
        torch.cuda.empty_cache()


def train_mix(
    *, model: torch.nn.Module, era: dict, ens: dict, seed: int, device: str
) -> tuple[int, float]:
    """Pretrain with half of every batch from each source; validate on the mean of the two sources' held-out NLL.

    One epoch draws `ROW_FRACTION` of the ERA5 and CAMS training rows from each source (so the fewer
    ENS rows are seen several times per epoch).
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr_e, tr_n = (torch.nonzero(~d["v"]).squeeze(1) for d in (era, ens))
    va = []
    for d in (era, ens):
        v = torch.nonzero(d["v"]).squeeze(1)
        if len(v) > c.VALIDATION_ROWS // 2:
            v = v[torch.randperm(len(v), generator=gen)[: c.VALIDATION_ROWS // 2].to(device)]
        va.append(v)
    steps = max(1, int(c.ROW_FRACTION * len(tr_e)) // HALF_BATCH)
    optimiser = torch.optim.AdamW(model.parameters(), lr=c.PRETRAIN_LR, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        for _ in range(steps):
            be = tr_e[torch.randint(len(tr_e), (HALF_BATCH,), device=device)]
            bn = tr_n[torch.randint(len(tr_n), (HALF_BATCH,), device=device)]
            x = torch.cat([era["x"][be], ens["x"][bn]])
            p = torch.cat([era["p"][be], ens["p"][bn]])
            z = torch.cat([era["z"][be], ens["z"][bn]])
            loss = c.row_nll(model=model, x=x, p=p, z=z, outlier=False).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            scores = [
                float(
                    torch.cat(
                        [
                            c.row_nll(
                                model=model, x=d["x"][ch], p=d["p"][ch], z=d["z"][ch], outlier=False
                            )
                            for ch in v.split(c.BATCH * 2)
                        ]
                    ).mean()
                )
                for d, v in zip((era, ens), va, strict=True)
            ]
        val = sum(scores) / 2
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


def assemble() -> None:
    """Concatenate each arm's window forecasts into one table per arm, month, and seed."""
    for arm in ARMS.values():
        for month in c.MONTHS:
            cuts = k.cutoffs(month=month, cadence=arm.cadence)
            for seed in (0, 1):
                parts = [
                    W
                    / "fits"
                    / arm.tag
                    / f"{month}_d{day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in cuts
                ]
                if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                    frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                    target = W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                    atomic_frame(frame=frame, target=target)
                    print("wrote", arm.tag, target.name, frame.height, flush=True)


def stale(*, arm: Arm, months: list[str], seeds: list[int], max_age: int, device: str) -> None:
    """Score each origin with the model refit `age` days earlier (age 0 to `max_age`), inference only."""
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = flagged_test(
            arm=arm, test=data.build_test(month=month, standardiser=mr.base.standardiser)
        )
        cuts = k.cutoffs(month=month, cadence=arm.cadence)
        for seed in seeds:
            models: dict[int, torch.nn.Module] = {}
            for i, cut in enumerate(cuts):
                day = day_of(month=month, cut=cut)
                f = model_file(arm=arm, month=month, day=day, seed=seed)
                if not f.exists():
                    break
                m = c.new_model(
                    outlier=False,
                    n_inputs=n_inputs(arm=arm, mr=mr),
                    n_primaries=len(mr.base.target_mean),
                ).to(device)
                m.load_state_dict(torch.load(f, map_location=device))
                m.eval()
                models[i] = m
            if len(models) != len(cuts):
                continue
            for age in range(max_age + 1):
                out = W / "stale" / arm.tag / f"{month}_a{age}_s{seed}.parquet"
                if out.exists():
                    continue
                parts = []
                start_us, stop_us = (
                    c.us(t=c.month_start(month=month)),
                    c.us(t=c.month_stop(month=month)),
                )
                for lo in range(start_us, stop_us, c.DAY_US):
                    # Each origin day is scored by the model with the latest cutoff at least `age` days earlier.
                    j = max(
                        (q for q, cu in enumerate(cuts) if c.us(t=cu) <= lo - age * c.DAY_US),
                        default=None,
                    )
                    if j is None:
                        continue
                    day_frame = window_frame(
                        model=models[j], test=test, mr=mr, lo=lo,
                        hi=min(lo + c.DAY_US, stop_us), flag=arm.flag, device=device,
                    )  # fmt: skip
                    if day_frame is not None:
                        parts.append(day_frame)
                if parts:
                    atomic_frame(frame=pl.concat(parts), target=out)
                    print(
                        "stale",
                        arm.tag,
                        month,
                        f"age {age}",
                        f"s{seed}",
                        sum(p.height for p in parts),
                        flush=True,
                    )
            del models
        del mr, test
        torch.cuda.empty_cache()


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("leak", "time", "pretrain-mix", "fit", "assemble", "stale")
    )
    parser.add_argument("--arm", choices=list(ARMS), default="L-1d")
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--stress",
        action="store_true",
        help="fit the cutoffs after the injected 7-day block, both signs",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="fit only the first N cutoffs of each month"
    )
    parser.add_argument("--max-age", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _ = math
    if args.command == "leak":
        leak_check()
    elif args.command == "time":
        time_one(device=args.device)
    elif args.command == "pretrain-mix":
        pretrain_mix(device=args.device)
    elif args.command == "fit":
        fit_arm(
            arm=ARMS[args.arm],
            months=args.months,
            seeds=args.seeds,
            stress=args.stress,
            limit=args.limit,
            device=args.device,
        )
    elif args.command == "assemble":
        assemble()
    else:
        stale(
            arm=ARMS[args.arm],
            months=args.months,
            seeds=args.seeds,
            max_age=args.max_age,
            device=args.device,
        )


if __name__ == "__main__":
    main()
