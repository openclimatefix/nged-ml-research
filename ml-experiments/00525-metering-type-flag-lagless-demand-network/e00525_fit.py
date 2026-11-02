# ruff: noqa: E501
r"""Pretrain and fine-tune the metering-flag arms of experiment 525; measure mass below zero.

The harness is experiment 519's (`e00519_fit`), experiment 511's (`e00511_common`, `e00511_fit`),
and
experiment 510's (`e00510_common`): ERA5-and-CAMS pretraining at share 0 through `train_share`,
then weekly
fine-tunes from the month's pretrained weights. This module appends one per-Primary 0/1 column
(1 for apparent power in MVA) to the 37 lag-free inputs.

Arms: `M` (the real metering flag) and `P` (a fixed random 8/6 assignment of the 16 Primaries, report-only).
The constant-flag control C is experiment 519's stored `F0z` (flag 0 everywhere), reused
unrefitted. L-Ww is
experiment 510's.

Commands:

- `check`: input checks (the flag lookup against `sites()`, M's forced-zero input against 519's
  `with_flag`).
- `check-c`: run 519's stored F0z weights through this module's flagged test rows (CPU) and compare
  with 519's stored forecast.
- `pretrain`: pretrain M and P (5 folds x 2 seeds, skipping files that exist). Needs the GPU go.
- `fit --arm A`: the weekly fine-tunes and forecasts of one arm. Needs the GPU go.
- `assemble`: concatenate each arm's window forecasts into one table per arm, month, and seed.
- `mass --arm A`: the mixture's mass below zero per row for an arm (`LWW` and `C` read stored
  weights; CPU).

```bash
export PYTHONPATH=<every folder under ml-experiments/, joined by colons>
export DATA_PATH_INTERNAL=<private data folder>
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00525-*/e00525_fit.py check
```
"""

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_run as r448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import e00519_fit as f519  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

W: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00525-metering-type-flag-lagless-demand-network" / "work"
)
SEEDS: Final[tuple[int, ...]] = (0, 1)
ARMS: Final[tuple[str, ...]] = ("M", "P")
PERMUTATION_SEED: Final[int] = 525
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
MASS_ARMS: Final[tuple[str, ...]] = ("LWW", "C", "M", "P")
MASS_PRIMARIES: Final[tuple[str, ...]] = ("D8", "D12", "D22")
SHARE_ZERO: Final = f519.Variant("FLAGGED", 0.0, False)
"""Pretraining at ERA5-and-CAMS share 0; the flag column is already in `x`, so `flagged` is False."""


def lookup(*, arm: str) -> np.ndarray:
    """Return the 16-long flag per Primary index (M: 1 for MVA; P: permuted; C: zeros)."""
    table = prep.primaries().sort("primary")
    assert table["primary"].to_list() == list(range(16)), "Primary indices are not 0..15"
    assert table["site"].to_list() == data.sites(), "Primary table and sites() disagree"
    assert table["units"].is_in(["MW", "MVA"]).all(), "a Primary has no metering unit"
    flags = (table["units"] == "MVA").cast(pl.Float32).to_numpy()
    if arm == "M":
        return flags
    if arm == "P":
        permuted = np.random.default_rng(PERMUTATION_SEED).permutation(flags)
        assert not np.array_equal(permuted, flags), "the permuted flag equals the real flag"
        return permuted
    if arm == "C":
        return np.zeros(16, np.float32)
    raise ValueError(arm)


def add_flag(*, x: np.ndarray, primary: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return `x` with the per-Primary flag appended as the last column."""
    return np.concatenate([x, table[primary][:, None].astype(x.dtype)], axis=1)


def pretrained_file(*, arm: str, month: str, seed: int) -> Path:
    """Return an arm's pretrained weights file."""
    return W / "pretrained" / arm / f"{month}_s{seed}.pt"


def check() -> None:
    """Fail unless the flag lookup is right and a forced-zero input equals 519's flag 0."""
    m, p = lookup(arm="M"), lookup(arm="P")
    table = prep.primaries().sort("primary")
    print(dict(zip(table["site"], m.astype(int), strict=True)), flush=True)
    assert int(m.sum()) == 8, "expected 8 MVA Primaries"
    assert int(p.sum()) == 8, "the permuted flag must keep 8 ones"
    mva = set(table.filter(pl.col("units") == "MVA")["site"])
    assert mva == set(EIGHT), f"the MVA set is {sorted(mva)}"
    assert {"D15", "D18"}.isdisjoint(mva), "D15 and D18 must be signed MW"
    x = np.random.default_rng(0).normal(size=(50, 37)).astype(np.float32)
    primary = np.arange(50) % 16
    forced = add_flag(x=x, primary=primary, table=lookup(arm="C"))
    reference = f519.with_flag(x=torch.from_numpy(x), flag=0.0).numpy()
    assert np.array_equal(forced, reference), "forced-zero input differs from 519's with_flag(0.0)"
    print("check ok: 8 MVA Primaries, forced-zero input equals 519's constant flag 0", flush=True)


def check_c(*, month: str = "2026-01", seed: int = 0, week: int = 2) -> None:
    """Run C's stored F0z weights through this module's flagged test rows and compare with 519's stored forecast."""
    clean = c.clean_stack()
    mr, test = flagged_month(month=month, arm="C", clean=clean)
    cut = k.cutoffs(month=month, cadence="w")[week]
    day = f511.day_of(month=month, cut=cut)
    path, width = stored_model(arm="C", month=month, day=day, week=week, seed=seed)
    model = c.new_model(outlier=False, n_inputs=width, n_primaries=len(mr.base.target_mean))
    c.load_pretrained(model=model, path=path)
    model.eval()
    lo, hi = k.window_of(month=month, cadence="w", index=week)
    ours = f511.window_frame(model=model, test=test, mr=mr, lo=lo, hi=hi, flag=False, device="cpu")
    stored = pl.read_parquet(f519.W / "fits" / "F0z" / f"{month}_d{day}_s{seed}.parquet")
    assert ours is not None, "no rows in the window"
    assert ours.height == stored.height, f"row counts differ: {ours.height} vs {stored.height}"
    cols = [col for col in ours.columns if col.startswith("q")]
    keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    joined = ours.join(stored, on=keys, suffix="_s", how="inner")
    assert joined.height == ours.height, "keys differ from 519's stored forecast"
    worst = max(float((joined[col] - joined[f"{col}_s"]).abs().max()) for col in cols)
    print(f"check-c {month} d{day} s{seed}: {ours.height} rows, max |diff| {worst:.2e}", flush=True)
    assert worst < 1e-4, "M's code path with the lookup at 0 does not reproduce F0z"


def pretrain(*, months: list[str], arms: list[str], seeds: list[int], device: str) -> None:
    """Pretrain every missing arm for the months and seeds (experiment 519's `train_share` at share 0)."""
    clean = c.clean_stack()
    for month in months:
        todo = [
            (arm, s)
            for arm in arms
            for s in seeds
            if not pretrained_file(arm=arm, month=month, seed=s).exists()
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
        print(f"{month}: ERA5/CAMS rows {len(pre.x):,}; rows past the cut: {past}", flush=True)
        assert past == 0, "pretraining rows past the cut"
        for arm, seed in todo:
            table = lookup(arm=arm)
            era = f519.to_gpu(
                x=add_flag(x=pre.x, primary=pre.primary, table=table), y=pre.y,
                primary=pre.primary, is_val=pre.is_val, rows=rows, device=device,
            )  # fmt: skip
            ens = f519.to_gpu(
                x=add_flag(x=rows.x, primary=rows.primary, table=table), y=rows.y,
                primary=rows.primary, is_val=rows.is_val, rows=rows, device=device,
            )  # fmt: skip
            started = time.monotonic()
            torch.manual_seed(seed)
            model = c.new_model(
                outlier=False, n_inputs=era["x"].shape[1], n_primaries=len(rows.target_mean)
            ).to(device)
            epochs, best = f519.train_share(
                model=model, era=era, ens=ens, v=SHARE_ZERO, seed=seed, device=device
            )
            f519.atomic_save(
                state=model.state_dict(), target=pretrained_file(arm=arm, month=month, seed=seed)
            )
            print(
                f"  pretrain {arm} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, "
                f"{time.monotonic() - started:.0f} s",
                flush=True,
            )
            del model, era, ens
            torch.cuda.empty_cache()


def flagged_month(*, month: str, arm: str, clean: np.ndarray) -> tuple[c.MonthRows, data.TestRows]:
    """Return the month's rows and test rows with the arm's flag column appended."""
    mr = k.build_month(month=month, clean=clean)
    test = data.build_test(month=month, standardiser=mr.base.standardiser)
    table = lookup(arm=arm)
    mr = dataclasses.replace(mr, x=add_flag(x=mr.x, primary=mr.primary, table=table))
    test = dataclasses.replace(test, x=add_flag(x=test.x, primary=test.primary, table=table))
    return mr, test


def fit_arm(*, arm: str, months: list[str], seeds: list[int], device: str) -> None:
    """Fine-tune and forecast the arm's weekly windows (restart from the month's pretrained weights)."""
    clean = c.clean_stack()
    for month in months:
        mr, test = flagged_month(month=month, arm=arm, clean=clean)
        g = k.GpuMonth(mr=mr, device=device)
        cuts = k.cutoffs(month=month, cadence="w")
        for seed in seeds:
            for i, cut in enumerate(cuts):
                day = f511.day_of(month=month, cut=cut)
                out = W / "fits" / arm / f"{month}_d{day}_s{seed}.parquet"
                if out.exists() or out.with_suffix(".empty").exists():
                    continue
                started = time.monotonic()
                refit = c.select_rows(rows=mr, cutoff=cut, half_life_days=c.HALF_LIFE_90)
                assert refit.report["rows_past_valid_cut"] == 0, "rows past the valid-time cut"
                assert refit.report["rows_past_origin_cut"] == 0, "rows past the origin cut"
                t = g.refit_tensors(refit=refit)
                model = c.new_model(
                    outlier=False, n_inputs=mr.x.shape[1], n_primaries=len(mr.base.target_mean)
                ).to(device)
                c.load_pretrained(
                    model=model, path=pretrained_file(arm=arm, month=month, seed=seed)
                )
                model.to(device)
                epochs, best = c.train(
                    model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed,
                    lr=c.FINE_TUNE_LR, outlier=False, prior=False, device=device,
                )  # fmt: skip
                f519.atomic_save(
                    state=model.state_dict(),
                    target=W / "models" / arm / f"{month}_d{day}_s{seed}.pt",
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
                    "name": f"{arm}/{month}_d{day}_s{seed}", "epochs": epochs, "val_nll": best,
                    "seconds": round(time.monotonic() - started, 2), **refit.report,
                }  # fmt: skip
                c.write_json(obj=log, path=W / "logs" / arm / f"{month}_d{day}_s{seed}.json")
                print(f"{log['name']}: {epochs} epochs, {log['seconds']} s", flush=True)
                del model, t
        del g, mr, test
        torch.cuda.empty_cache()


def assemble() -> None:
    """Concatenate each arm's window forecasts into one table per arm, month, and seed."""
    for arm in ARMS:
        for month in c.MONTHS:
            for seed in SEEDS:
                parts = [
                    W
                    / "fits"
                    / arm
                    / f"{month}_d{f511.day_of(month=month, cut=cut)}_s{seed}.parquet"
                    for cut in k.cutoffs(month=month, cadence="w")
                ]
                if all(p.exists() or p.with_suffix(".empty").exists() for p in parts):
                    frame = pl.concat([pl.read_parquet(p) for p in parts if p.exists()])
                    target = W / "seed_preds" / arm / f"{month}_s{seed}.parquet"
                    f511.atomic_frame(frame=frame, target=target)
                    print("wrote", arm, target.name, frame.height, flush=True)


def mixture_mass_below_zero(*, out: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """Return the three main components' probability mass at or below 0 MW.

    `out` is the network output, `z0` is `-target_mean / target_std` of each row's Primary (0 MW in
    standardised units). The mass is sum_k softmax(log_w)_k * Phi((z0 - mean_k) / std_k); the outlier
    component is not used.
    """
    log_w, mean, std = base.mixture_parts(
        out=out.double()[..., : 3 * c.COMPONENTS], components=c.COMPONENTS
    )
    w = torch.softmax(log_w, dim=-1)
    return (w * torch.special.ndtr((z0[:, None].double() - mean) / std)).sum(dim=-1)


def stored_model(*, arm: str, month: str, day: int, week: int, seed: int) -> tuple[Path, int]:
    """Return the arm's stored fine-tuned weights file and input width."""
    if arm == "LWW":
        return c.WORK_DIR / "models" / "hl90" / f"{month}_w{week}_s{seed}.pt", 37
    if arm == "C":
        return f519.W / "models" / "F0z" / f"{month}_d{day}_s{seed}.pt", 38
    return W / "models" / arm / f"{month}_d{day}_s{seed}.pt", 38


def mass(*, arm: str, months: list[str], seeds: list[int], device: str) -> None:
    """Write the mixture's mass below zero for every test row of the arm's weekly windows."""
    clean = c.clean_stack()
    for month in months:
        mr = k.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        if arm != "LWW":
            test = dataclasses.replace(
                test, x=add_flag(x=test.x, primary=test.primary, table=lookup(arm=arm))
            )
        z0_all = -mr.base.target_mean / mr.base.target_std
        cuts = k.cutoffs(month=month, cadence="w")
        for seed in seeds:
            target = W / "mass" / arm / f"{month}_s{seed}.parquet"
            if target.exists():
                continue
            parts = []
            for i, cut in enumerate(cuts):
                lo, hi = k.window_of(month=month, cadence="w", index=i)
                sel = np.nonzero((test.origin >= lo) & (test.origin < hi))[0]
                if len(sel) == 0:
                    continue
                path, width = stored_model(
                    arm=arm, month=month, day=f511.day_of(month=month, cut=cut), week=i, seed=seed
                )
                model = c.new_model(
                    outlier=False, n_inputs=width, n_primaries=len(mr.base.target_mean)
                ).to(device)
                c.load_pretrained(model=model, path=path)
                model.eval()
                for a in range(0, len(sel), 65_536):
                    sl = sel[a : a + 65_536]
                    x = torch.from_numpy(test.x[sl]).to(device)
                    p = torch.from_numpy(test.primary[sl]).to(device)
                    with torch.no_grad():
                        m = mixture_mass_below_zero(
                            out=model(x, p),
                            z0=torch.from_numpy(z0_all).to(device)[p],
                        )
                    parts.append(
                        pl.DataFrame(
                            {
                                "primary": test.primary[sl],
                                "origin_us": test.origin[sl],
                                "valid_us": test.valid[sl],
                                "mass": m.float().cpu().numpy(),
                            }
                        )
                    )
                del model
            frame = pl.concat(parts)
            f511.atomic_frame(frame=frame, target=target)
            print("mass", arm, month, f"s{seed}", frame.height, flush=True)


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("check", "check-c", "pretrain", "fit", "assemble", "mass")
    )
    parser.add_argument("--arm", default="M")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "check-c":
        check_c()
    elif args.command == "pretrain":
        pretrain(months=args.months, arms=args.arms, seeds=args.seeds, device=args.device)
    elif args.command == "fit":
        fit_arm(arm=args.arm, months=args.months, seeds=args.seeds, device=args.device)
    elif args.command == "assemble":
        assemble()
    else:
        mass(arm=args.arm, months=args.months, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
