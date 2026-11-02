# ruff: noqa: E501
r"""Fit and forecast the arms of experiment 510 (lag-free demand network, weekly or monthly refits).

Commands:

- `leak`: build each month's rows, select the rows of every refit for every arm, and print the leak
  check (rows past the valid-time and origin cut must be 0). Run it before any fit.
- `pretrain`: the fold 2025-12 pretraining (3 components, seeds 0 and 1) and the 4-component
  pretraining of L-Ww-O (seed 0, all five folds), with experiment 448's pretraining rows and loop.
- `fit --variant V`: fine-tune and forecast one variant on the months and seeds given. Variants:
  `hl182` (experiment 425's 182.6-day weight: L-M uses its first refit for the whole month, L-W
  uses all five weekly refits), `hl90` (L-Ww), `hl90D` (L-Ww-D), `hl90O` (L-Ww-O, seed 0).
  `--stress` fits only refit week 3 on a 7-day injected shift, both signs.
- `assemble`: concatenate the per-refit forecasts into one table per arm, month, and seed in
  experiment 425's seed-prediction layout under `seed_preds/<arm>/`.

```bash
source <PYTHONPATH of every folder under ml-experiments/>; export DATA_PATH_INTERNAL=<private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00510-*/e00510_fit.py leak
```
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

W: Final[Path] = c.WORK_DIR


@dataclass(frozen=True)
class Variant:
    """How a variant fine-tunes: recency half-life, label drop, and the outlier component."""

    tag: str
    half_life: float
    drop: bool = False
    outlier: bool = False


VARIANTS: Final[dict[str, Variant]] = {
    "hl182": Variant("hl182", c.HALF_LIFE_425),
    "hl90": Variant("hl90", c.HALF_LIFE_90),
    "hl90D": Variant("hl90D", c.HALF_LIFE_90, drop=True),
    "hl90O": Variant("hl90O", c.HALF_LIFE_90, outlier=True),
}
ARM_OF: Final[dict[str, str]] = {
    "hl182": "L-W", "hl90": "L-Ww", "hl90D": "L-Ww-D", "hl90O": "L-Ww-O",
}  # fmt: skip


def pretrained_path(*, v: Variant, month: str, seed: int) -> Path:
    """Return the pretrained weights a variant starts from."""
    if v.outlier:
        return W / "pretrained" / "L0PT4" / f"{month}_s{seed}.pt"
    own = W / "pretrained" / "L0PT" / f"{month}_s{seed}.pt"
    return own if own.exists() else c.E448_PRETRAINED / f"{month}_s{seed}.pt"


def window_of(*, month: str, week: int) -> tuple[int, int]:
    """Return the scored origin window [start, stop) in microseconds of refit week `week`."""
    cuts = c.cutoffs(month=month)
    stop = cuts[week + 1] if week + 1 < len(cuts) else c.month_stop(month=month)
    return c.us(t=cuts[week]), c.us(t=stop)


def tensors(*, mr: c.MonthRows, refit: c.Refit, device: str) -> dict[str, torch.Tensor]:
    """Return the standardised target, inputs, weights, and flags of a refit on the device."""
    idx = refit.index
    prim = mr.primary[idx]
    z = (refit.y - mr.base.target_mean[prim]) / mr.base.target_std[prim]
    return {
        "x": torch.from_numpy(mr.x[idx]).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(refit.weight).to(device),
        "p": torch.from_numpy(prim).to(device),
        "v": torch.from_numpy(refit.is_val).to(device),
    }


def leak_check() -> None:
    """Print, for every month, refit, and variant, the rows selected and the cut check."""
    clean = c.clean_stack()
    events = c.labelled_events()
    bad = 0
    for month in c.MONTHS:
        mr = c.build_month(month=month, clean=clean)
        print(
            f"{month}: {len(mr.y):,} rows ({mr.n_base:,} in experiment 428's fold), {mr.x.shape[1]} inputs",
            flush=True,
        )
        for k, cut in enumerate(c.cutoffs(month=month)):
            for v in VARIANTS.values():
                drop = c.event_drop_mask(rows=mr, cutoff=cut, events=events) if v.drop else None
                r = c.select_rows(rows=mr, cutoff=cut, half_life_days=v.half_life, drop=drop).report
                sel = c.select_rows(
                    rows=mr, cutoff=cut, half_life_days=v.half_life, drop=drop
                ).index
                independent = int(
                    (mr.valid_us[sel] > c.us(t=cut) - c.CUTOFF_AGE_US).sum()
                    + (mr.origin_us[sel] > c.us(t=cut)).sum()
                )
                bad += r["rows_past_valid_cut"] + r["rows_past_origin_cut"] + independent
                print(f"  w{k + 1} {v.tag}: {r}", flush=True)
        del mr
    assert bad == 0, f"{bad} rows past a cut"
    print("leak check passed: 0 rows past any cut")


def pretrain(*, device: str) -> None:
    """Pretrain fold 2025-12 (3 components, seeds 0 and 1) and the 4-component networks (seed 0)."""
    data_clean = c.clean_stack()
    for month in c.MONTHS:
        todo3 = [
            s
            for s in (0, 1)
            if not (W / "pretrained" / "L0PT" / f"{month}_s{s}.pt").exists()
            and not (c.E448_PRETRAINED / f"{month}_s{s}.pt").exists()
        ]
        todo4 = [0] if not (W / "pretrained" / "L0PT4" / f"{month}_s0.pt").exists() else []
        if not todo3 and not todo4:
            continue
        refit_t = c.month_start(month=month)
        rows = data.build_fold(refit=refit_t, clean=data_clean)
        pre = r448.pretrain_rows(rows=rows, refit=refit_t, clean=data_clean)
        latest = datetime.fromtimestamp(
            (pre.target_idx.max() * c.prep422.STEP_US + c.prep422.GRID_START_US) / 1e6, UTC
        )
        print(
            f"{month}: pretraining rows {len(pre.x):,}; latest valid {latest.isoformat()} (refit {refit_t.isoformat()})",
            flush=True,
        )
        prim = pre.primary
        z = (pre.y - rows.target_mean[prim]) / rows.target_std[prim]
        t = {
            "x": torch.from_numpy(pre.x).to(device),
            "z": torch.from_numpy(z.astype(np.float32)).to(device),
            "w": torch.ones(len(pre.y), device=device),
            "p": torch.from_numpy(prim).to(device),
            "v": torch.from_numpy(pre.is_val).to(device),
        }
        for outlier, seeds, folder in ((False, todo3, "L0PT"), (True, todo4, "L0PT4")):
            for seed in seeds:
                started = time.monotonic()
                torch.manual_seed(seed)
                model = c.new_model(
                    outlier=outlier, n_inputs=pre.x.shape[1], n_primaries=len(rows.target_mean)
                ).to(device)
                epochs, best = c.train(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    lr=c.PRETRAIN_LR, outlier=outlier, prior=False, device=device,
                )  # fmt: skip
                target = W / "pretrained" / folder / f"{month}_s{seed}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                torch.save(model.state_dict(), tmp)
                tmp.rename(target)
                print(
                    f"  pretrain {folder} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                    flush=True,
                )
        del t, pre, rows
        torch.cuda.empty_cache()


def fit_variant(
    *, v: Variant, months: list[str], seeds: list[int], weeks: list[int], stress: bool, device: str
) -> None:
    """Fine-tune and forecast a variant for each month, refit week, and seed."""
    clean = c.clean_stack()
    events = c.labelled_events() if v.drop else {}
    for month in months:
        mr = c.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        cuts = c.cutoffs(month=month)
        for week in weeks:
            cut = cuts[week]
            drop = c.event_drop_mask(rows=mr, cutoff=cut, events=events) if v.drop else None
            shift = c.shift_sizes(clean=clean, cutoff=cut) if stress else None
            for seed in seeds:
                for sign in (+1, -1) if stress else (0,):
                    name = f"{v.tag}/{month}_w{week}_s{seed}" + (
                        f"_stress{'p' if sign > 0 else 'n'}" if stress else ""
                    )
                    out = W / "fits" / f"{name}.parquet"
                    if out.exists():
                        continue
                    started = time.monotonic()
                    refit = c.select_rows(
                        rows=mr,
                        cutoff=cut,
                        half_life_days=v.half_life,
                        drop=drop,
                        shift=shift,
                        sign=sign,
                    )
                    t = tensors(mr=mr, refit=refit, device=device)
                    model = c.new_model(
                        outlier=v.outlier,
                        n_inputs=mr.x.shape[1],
                        n_primaries=len(mr.base.target_mean),
                    ).to(device)
                    c.load_pretrained(
                        model=model, path=pretrained_path(v=v, month=month, seed=seed)
                    )
                    model.to(device)
                    epochs, best = c.train(
                        model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                        lr=c.FINE_TUNE_LR, outlier=v.outlier, prior=v.outlier, device=device,
                    )  # fmt: skip
                    lo, hi = window_of(month=month, week=week)
                    windows = {"week": (lo, hi)}
                    if week == 0 and v.tag == "hl182":
                        windows["month"] = (lo, c.us(t=c.month_stop(month=month)))
                    for label, (a, b) in windows.items():
                        sel = (test.origin >= a) & (test.origin < b)
                        frame = c.forecast_frame(
                            model=model,
                            outlier=v.outlier,
                            test=test,
                            select=sel,
                            rows=mr.base,
                            widen=1.0,
                            device=device,
                        )
                        target = (
                            out if label == "week" else out.with_name(out.stem + "_month.parquet")
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tmp = target.with_suffix(".tmp")
                        frame.write_parquet(tmp)
                        tmp.rename(target)
                    state = W / "models" / f"{name}.pt"
                    state.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), state)
                    c.write_json(
                        obj={
                            "name": name,
                            "epochs": epochs,
                            "val_nll": best,
                            "seconds": round(time.monotonic() - started),
                            **refit.report,
                        },
                        path=W / "logs" / f"{name}.json",
                    )
                    print(
                        f"{name}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s, {refit.report['rows']:,} rows",
                        flush=True,
                    )
                    del t, model
                    torch.cuda.empty_cache()
        del mr, test


def assemble() -> None:
    """Concatenate refit forecasts into one table per arm, month, and seed (experiment 425's layout)."""
    for tag, v in VARIANTS.items():
        for month in c.MONTHS:
            for seed in (0, 1):
                parts = [
                    W / "fits" / f"{tag}/{month}_w{k}_s{seed}.parquet"
                    for k in range(len(c.REFIT_DAYS))
                ]
                if all(p.exists() for p in parts):
                    frame = pl.concat([pl.read_parquet(p) for p in parts])
                    target = W / "seed_preds" / ARM_OF[tag] / f"{month}_s{seed}.parquet"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_parquet(target)
                    print("wrote", target.name, ARM_OF[tag], frame.height, flush=True)
                full = W / "fits" / f"{tag}/{month}_w0_s{seed}_month.parquet"
                if tag == "hl182" and full.exists():
                    target = W / "seed_preds" / "L-M" / f"{month}_s{seed}.parquet"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pl.read_parquet(full).write_parquet(target)
                    print("wrote", target.name, "L-M", flush=True)
        del v


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("leak", "pretrain", "fit", "assemble"))
    parser.add_argument("--variant", choices=list(VARIANTS), default="hl182")
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--weeks", nargs="+", type=int, default=list(range(len(c.REFIT_DAYS))))
    parser.add_argument(
        "--stress",
        action="store_true",
        help="fit refit week 3 on a 7-day injected shift, both signs",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _ = common
    if args.command == "leak":
        leak_check()
    elif args.command == "pretrain":
        pretrain(device=args.device)
    elif args.command == "fit":
        weeks = [c.STRESS_WEEK] if args.stress else args.weeks
        fit_variant(
            v=VARIANTS[args.variant],
            months=args.months,
            seeds=args.seeds,
            weeks=weeks,
            stress=args.stress,
            device=args.device,
        )
    else:
        assemble()


if __name__ == "__main__":
    main()
