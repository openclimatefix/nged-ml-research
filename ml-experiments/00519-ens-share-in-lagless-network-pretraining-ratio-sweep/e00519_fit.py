# ruff: noqa: E501
r"""Pretrain and fine-tune the arms of experiment 519 (ENS share in the lag-free network's pretraining).

Experiment 511's harness (`e00511_fit`, `e00511_common`) and experiment 510's (`e00510_common`)
supply the network, the loop, the row selection, the weekly cutoffs, and the forecast frame. This
module adds the pretraining at an arbitrary ENS share of the batch, with or without the source-flag
column, and the two-stage control.

Commands:

- `pretrain`: the pretraining variants (5 folds x 2 seeds, skipping files that exist).
- `fit --arm A`: the weekly fine-tunes and forecasts of one arm.
- `assemble`: concatenate each arm's window forecasts into one table per arm, month, and seed.
- `time`: pretrain variant E25 and fit arm E25 for one month and seed 0; print the seconds and peak memory.

```bash
export PYTHONPATH=<every folder under ml-experiments/, joined by colons>
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00519-*/e00519_fit.py time
```
"""

import argparse
import dataclasses
import resource
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as f510  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00519-ens-share-in-lagless-network-pretraining-ratio-sweep" / "work"
)
W511: Final[Path] = k.WORK_DIR
TOTAL_BATCH: Final[int] = 2 * f511.HALF_BATCH
STAGE_TWO_LR_FACTOR: Final[float] = 0.1
SEEDS: Final[tuple[int, ...]] = (0, 1)


@dataclass(frozen=True)
class Variant:
    """A pretraining: the ENS share of every batch and whether the network has the flag column."""

    key: str
    share: float
    flagged: bool


VARIANTS: Final[dict[str, Variant]] = {
    v.key: v
    for v in (
        Variant("F0", 0.0, True),
        Variant("E10", 0.10, True),
        Variant("E25", 0.25, True),
        Variant("E75", 0.75, True),
        Variant("E100", 1.0, True),
        Variant("NF25", 0.25, False),
        Variant("NF50", 0.50, False),
    )
}
TWO_STAGE: Final[str] = "TS"


@dataclass(frozen=True)
class Arm:
    """A fine-tuned arm: where its pretrained weights come from and the flag value at fine-tune and scoring."""

    tag: str
    pretrained: str
    flag: float | None
    """Flag value at fine-tuning and scoring (None: no flag column)."""


ARMS: Final[dict[str, Arm]] = {
    a.tag: a
    for a in (
        Arm("F0", "F0", 1.0),
        Arm("E10", "E10", 1.0),
        Arm("E25", "E25", 1.0),
        Arm("E75", "E75", 1.0),
        Arm("E100", "E100", 1.0),
        Arm("NF25", "NF25", None),
        Arm("NF50", "NF50", None),
        Arm("Z25", "E25", 0.0),
        Arm("Z50", "MIX", 0.0),
        Arm("TS", TWO_STAGE, None),
    )
}


def pretrained_file(*, key: str, month: str, seed: int) -> Path:
    """Return a pretrained weights file (experiment 511's for the 50% share, this experiment's otherwise)."""
    if key == "MIX":
        return W511 / "pretrained" / "MIX" / f"{month}_s{seed}.pt"
    return W / "pretrained" / key / f"{month}_s{seed}.pt"


def atomic_save(*, state: dict, target: Path) -> None:
    """Save a state dict atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.rename(target)


def held_out_nll(
    *, model: torch.nn.Module, d: dict, idx: torch.Tensor, flag: float | None
) -> float:
    """Return the mean NLL of a source's held-out rows."""
    model.eval()
    with torch.no_grad():
        return float(
            torch.cat(
                [
                    c.row_nll(
                        model=model,
                        x=with_flag(x=d["x"][ch], flag=flag),
                        p=d["p"][ch],
                        z=d["z"][ch],
                        outlier=False,
                    )
                    for ch in idx.split(c.BATCH * 2)
                ]
            ).mean()
        )


def with_flag(*, x: torch.Tensor, flag: float | None) -> torch.Tensor:
    """Append a constant flag column to a batch (None: leave the batch as it is)."""
    if flag is None:
        return x
    return torch.cat([x, torch.full((len(x), 1), flag, dtype=x.dtype, device=x.device)], dim=1)


def train_share(
    *, model: torch.nn.Module, era: dict, ens: dict, v: Variant, seed: int, device: str
) -> tuple[int, float]:
    """Pretrain with `v.share` of every 8,192-row batch drawn from the ENS rows.

    One epoch takes as many steps as experiment 511's (20% of the ERA5 and CAMS training rows at 4,096
    rows per step), at every share. Early stopping uses the held-out NLL of the two sources averaged with
    the batch shares as weights, so 50% equals experiment 511's mean and 100% uses the ENS term alone.
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_ens = round(v.share * TOTAL_BATCH)
    n_era = TOTAL_BATCH - n_ens
    tr_e, tr_n = (torch.nonzero(~d["v"]).squeeze(1) for d in (era, ens))
    va = []
    for d in (era, ens):
        idx = torch.nonzero(d["v"]).squeeze(1)
        if len(idx) > c.VALIDATION_ROWS // 2:
            idx = idx[torch.randperm(len(idx), generator=gen)[: c.VALIDATION_ROWS // 2].to(device)]
        va.append(idx)
    steps = max(1, int(c.ROW_FRACTION * len(tr_e)) // f511.HALF_BATCH)
    optimiser = torch.optim.AdamW(model.parameters(), lr=c.PRETRAIN_LR, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(c.MAX_EPOCHS):
        model.train()
        for _ in range(steps):
            parts = []
            if n_era:
                be = tr_e[torch.randint(len(tr_e), (n_era,), device=device)]
                parts.append(
                    (
                        with_flag(x=era["x"][be], flag=0.0 if v.flagged else None),
                        era["p"][be],
                        era["z"][be],
                    )
                )
            if n_ens:
                bn = tr_n[torch.randint(len(tr_n), (n_ens,), device=device)]
                parts.append(
                    (
                        with_flag(x=ens["x"][bn], flag=1.0 if v.flagged else None),
                        ens["p"][bn],
                        ens["z"][bn],
                    )
                )
            x, p, z = (torch.cat(t) for t in zip(*parts, strict=True))
            loss = c.row_nll(model=model, x=x, p=p, z=z, outlier=False).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            optimiser.step()
        score = 0.0
        for d, idx, weight, flag in (
            (era, va[0], 1.0 - v.share, 0.0 if v.flagged else None),
            (ens, va[1], v.share, 1.0 if v.flagged else None),
        ):
            if weight > 0:
                score += weight * held_out_nll(model=model, d=d, idx=idx, flag=flag)
        epochs = epoch + 1
        if score < best - 1e-5:
            best, stale = score, 0
            best_state = {a: t.detach().clone() for a, t in model.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def stage_two(*, model: torch.nn.Module, ens: dict, seed: int, device: str) -> tuple[float, float]:
    """One epoch on ENS rows only at 0.1 x the pretraining rate; keep the weights only if the held-out ENS NLL fell.

    Returns the held-out ENS NLL before and after stage 2 (the kept state is the lower).
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tr = torch.nonzero(~ens["v"]).squeeze(1)
    va = torch.nonzero(ens["v"]).squeeze(1)
    if len(va) > c.VALIDATION_ROWS // 2:
        va = va[torch.randperm(len(va), generator=gen)[: c.VALIDATION_ROWS // 2].to(device)]
    before = held_out_nll(model=model, d=ens, idx=va, flag=None)
    saved = {a: t.detach().clone() for a, t in model.state_dict().items()}
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=STAGE_TWO_LR_FACTOR * c.PRETRAIN_LR, weight_decay=c.WEIGHT_DECAY
    )
    order = tr[torch.randperm(len(tr), device=device)][: int(c.ROW_FRACTION * len(tr))]
    model.train()
    for start in range(0, len(order), c.BATCH):
        b = order[start : start + c.BATCH]
        loss = c.row_nll(
            model=model, x=ens["x"][b], p=ens["p"][b], z=ens["z"][b], outlier=False
        ).mean()
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
        optimiser.step()
    after = held_out_nll(model=model, d=ens, idx=va, flag=None)
    if after >= before:
        model.load_state_dict(saved)
    model.eval()
    return before, after


def to_gpu(
    *,
    x: np.ndarray,
    y: np.ndarray,
    primary: np.ndarray,
    is_val: np.ndarray,
    rows: data.Rows,
    device: str,
) -> dict:
    """Return a source's inputs (no flag column), standardised targets, Primary indices, and held-out mask on the GPU."""
    z = (y - rows.target_mean[primary]) / rows.target_std[primary]
    return {
        "x": torch.from_numpy(x).to(device), "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "p": torch.from_numpy(primary).to(device), "v": torch.from_numpy(is_val).to(device),
    }  # fmt: skip


def pretrain(*, months: list[str], keys: list[str], seeds: list[int], device: str) -> None:
    """Pretrain every missing variant (and the two-stage weights) for the months and seeds."""
    clean = c.clean_stack()
    for month in months:
        todo = [
            (key, s)
            for key in keys
            for s in seeds
            if not pretrained_file(key=key, month=month, seed=s).exists()
        ]
        if not todo:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=clean)
        grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
        cut_valid, cut_origin = c.us(t=refit_t) - c.CUTOFF_AGE_US, c.us(t=refit_t)
        past = int(
            (grid + rows.target_idx.astype(np.int64) * step > cut_valid).sum()
            + (grid + rows.origin_idx.astype(np.int64) * step > cut_origin).sum()
            + (grid + pre.target_idx.astype(np.int64) * step > cut_valid).sum()
            + (grid + pre.origin_idx.astype(np.int64) * step > cut_origin).sum()
        )
        print(
            f"{month}: ERA5/CAMS rows {len(pre.x):,}, ENS rows {len(rows.x):,}; rows past the valid or origin cut: {past}; refit {refit_t.isoformat()}",
            flush=True,
        )
        assert past == 0, "pretraining rows past the cut"
        era = to_gpu(
            x=pre.x, y=pre.y, primary=pre.primary, is_val=pre.is_val, rows=rows, device=device
        )
        ens = to_gpu(
            x=rows.x, y=rows.y, primary=rows.primary, is_val=rows.is_val, rows=rows, device=device
        )
        n_primaries = len(rows.target_mean)
        for key, seed in todo:
            started = time.monotonic()
            if key == TWO_STAGE:
                model = c.new_model(
                    outlier=False, n_inputs=era["x"].shape[1], n_primaries=n_primaries
                ).to(device)
                c.load_pretrained(
                    model=model,
                    path=f510.pretrained_path(v=f510.VARIANTS["hl90"], month=month, seed=seed),
                )
                model.to(device)
                before, after = stage_two(model=model, ens=ens, seed=seed, device=device)
                note = f"stage 2 held-out ENS NLL {before:.4f} -> {after:.4f} ({'kept' if after < before else 'rejected'})"
            else:
                v = VARIANTS[key]
                width = era["x"].shape[1] + (1 if v.flagged else 0)
                torch.manual_seed(seed)
                model = c.new_model(outlier=False, n_inputs=width, n_primaries=n_primaries).to(
                    device
                )
                epochs, best = train_share(
                    model=model, era=era, ens=ens, v=v, seed=seed, device=device
                )
                note = f"{epochs} epochs, weighted val NLL {best:.4f}"
            atomic_save(
                state=model.state_dict(), target=pretrained_file(key=key, month=month, seed=seed)
            )
            print(
                f"  pretrain {key} {month} s{seed}: {note}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del model
        del era, ens, pre, rows
        torch.cuda.empty_cache()


def n_inputs(*, arm: Arm, mr: c.MonthRows) -> int:
    """Return the network's input width."""
    return mr.x.shape[1] + (0 if arm.flag is None else 1)


def fit_arm(*, arm: Arm, months: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune and forecast the arm's weekly windows (restart from the month's pretrained weights)."""
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        if arm.flag is not None:
            col = np.full((len(test.x), 1), arm.flag, np.float32)
            test = dataclasses.replace(test, x=np.concatenate([test.x, col], axis=1))
        g = k.GpuMonth(mr=mr, device=device, flag=arm.flag)
        cuts = k.cutoffs(month=month, cadence="w")
        for seed in seeds:
            for i, cut in enumerate(cuts):
                day = f511.day_of(month=month, cut=cut)
                out = W / "fits" / arm.tag / f"{month}_d{day}_s{seed}.parquet"
                if out.exists() or out.with_suffix(".empty").exists():
                    continue
                started = time.monotonic()
                refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=c.HALF_LIFE_90)
                t = g.refit_tensors(refit=refit)
                feeder = time.monotonic() - started
                model = c.new_model(
                    outlier=False,
                    n_inputs=n_inputs(arm=arm, mr=mr),
                    n_primaries=len(mr.base.target_mean),
                ).to(device)
                c.load_pretrained(
                    model=model, path=pretrained_file(key=arm.pretrained, month=month, seed=seed)
                )
                model.to(device)
                t0 = time.monotonic()
                epochs, best = c.train(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device,
                )  # fmt: skip
                train_s = time.monotonic() - t0
                atomic_save(
                    state=model.state_dict(),
                    target=W / "models" / arm.tag / f"{month}_d{day}_s{seed}.pt",
                )
                lo, hi = k.window_of(month=month, cadence="w", index=i)
                frame = f511.window_frame(
                    model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device=device
                )
                if frame is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.with_suffix(".empty").touch()
                else:
                    f511.atomic_frame(frame=frame, target=out)
                log = {
                    "name": f"{arm.tag}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
                    "feeder_seconds": round(feeder, 2), "train_seconds": round(train_s, 2),
                    "seconds": round(time.monotonic() - started, 2), **refit.report,
                }  # fmt: skip
                c.write_json(obj=log, path=W / "logs" / arm.tag / f"{month}_d{day}_s{seed}.json")
                print(
                    f"{log['name']}: {epochs} epochs, {log['seconds']} s (feeder {log['feeder_seconds']} s)",
                    flush=True,
                )
                del model, t
        del g, mr, test
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each arm's window forecasts into one table per arm, month, and seed."""
    for arm in ARMS.values():
        for month in c.MONTHS:
            for seed in SEEDS:
                parts = [
                    W
                    / "fits"
                    / arm.tag
                    / f"{month}_d{f511.day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in k.cutoffs(month=month, cadence="w")
                ]
                if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                    frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                    target = W / "seed_preds" / arm.tag / f"{month}_s{seed}.parquet"
                    f511.atomic_frame(frame=frame, target=target)
                    print("wrote", arm.tag, target.name, frame.height, flush=True)


def peak_rss_gb() -> float:
    """Return this process's peak resident set size in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def time_one(*, device: str) -> None:
    """Pretrain E25 and fit arm E25 for 2026-01, seed 0; print seconds, peak host memory, and peak GPU memory."""
    t0 = time.monotonic()
    pretrain(months=["2026-01"], keys=["E25"], seeds=[0], device=device)
    t1 = time.monotonic()
    fit_arm(arm=ARMS["E25"], months=["2026-01"], seeds=[0], device=device)
    t2 = time.monotonic()
    print(
        f"timing: pretrain stage (incl. data build) {t1 - t0:.0f} s; five weekly fine-tunes (incl. month build) {t2 - t1:.0f} s; "
        f"peak host RSS {peak_rss_gb():.2f} GB; peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB; "
        f"finished {datetime.now(UTC).isoformat()}",
        flush=True,
    )


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pretrain", "fit", "assemble", "time"))
    parser.add_argument("--arm", choices=list(ARMS), default="E25")
    parser.add_argument("--keys", nargs="+", default=[*VARIANTS, TWO_STAGE])
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "pretrain":
        pretrain(months=args.months, keys=args.keys, seeds=args.seeds, device=args.device)
    elif args.command == "fit":
        fit_arm(arm=ARMS[args.arm], months=args.months, seeds=args.seeds, device=args.device)
    elif args.command == "assemble":
        assemble()
    else:
        time_one(device=args.device)
    print(f"peak host RSS {peak_rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
