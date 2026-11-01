# ruff: noqa: E501, PLR0912, PLR0915, PT018
"""Pretrain the S1 and lag-only demand networks for 150 epochs, fine-tune, and forecast.

Reuses experiment 428's rows, fit loop, and forecast code, and experiment 449's pretraining rows
and ERA5 lagless tables (read, never written; the row builders are copied from experiment 449's
`e00449_run.py`, branch `exp-pretrain-with-lags-s1`, because experiment 449 is not on `main`). Two
new arms, both pooled over the 16 Primaries and pretrained on ERA5-and-CAMS rows with the three
origin lags live:

- `AS1L`: experiment 449's `PT1` recipe (lags and the surprise at each lag live in pretraining,
  fine-tuning at learning rate 2e-4) with the pretraining cap raised from 60 to 150 epochs.
- `AL`: the same loop and cap without the surprise inputs (the lag channels only).

Fine-tuning keeps the 60-epoch cap and patience 8 of both arms. `avg5` writes the five-seed
quantile average of an arm, as experiment 425 scores its arm A.

```bash
source /tmp/e452_env.sh
systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00452-*/e00452_run.py fit --arms AS1L --seeds 0 --months 2026-06
  uv run --no-sync python ml-experiments/00452-*/e00452_run.py predict --arms AS1L --seeds 0
  uv run --no-sync python ml-experiments/00452-*/e00452_run.py avg --arms AS1L AL PT1
  uv run --no-sync python ml-experiments/00452-*/e00452_run.py smoke
```
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00452_lagless as lagless  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WORK_DIR: Final[Path] = lagless.WORK_DIR
MODEL_ROOTS: Final[tuple[Path, ...]] = (
    WORK_DIR / "models",
    lagless.SOURCE_449 / "models",
)
"""Where a fitted network is looked for: this experiment, then experiment 449 (`PT1`)."""
PRETRAIN_MAX_EPOCHS: Final[int] = 150
"""Pretraining cap, in epochs of `fit.ROW_FRACTION` of the rows (experiment 449's cap was 60)."""
FINE_TUNE_LR: Final[float] = 2e-4
"""Experiment 425's fine-tuning learning rate."""
SEEDS5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
NEW_ARMS: Final[tuple[str, ...]] = ("AS1L", "AL")
AVG_ARMS: Final[tuple[str, ...]] = ("AS1L", "AL", "PT1")

fit.ARMS["PT1"] = fit.Arm("PT1", "lags", 0, True, surprise=True, history="s1")
fit.ARMS["AS1L"] = fit.Arm("AS1L", "lags", 0, True, surprise=True, history="s1")
fit.ARMS["AL"] = fit.Arm("AL", "lags", 0, True)


def install() -> None:
    """Redirect experiment 428's modules to this experiment's folders."""
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"


FINE_TUNE_MAX_EPOCHS: Final[int] = fit.MAX_EPOCHS
"""Experiment 428's cap, kept for fine-tuning."""


def locate_predictions(*, arm: str) -> Path:
    """Return the folder holding an arm's per-seed forecasts (experiment 449's for `PT1`)."""
    own = WORK_DIR / "predictions" / arm
    return own if own.exists() else lagless.SOURCE_449 / "predictions" / arm


def spread_table(*, rows: data.Rows) -> tuple[np.ndarray, np.ndarray]:
    """Return the median standardised `std_*` input of the fold's training rows by lead band."""
    keep_cols, _, _ = data.non_lag_layout()
    position = {c: i for i, c in enumerate(keep_cols)}
    _, std_full = prep418.column_indices()
    std_cols = np.array([position[c] for c in std_full])
    train = ~rows.is_val
    band = np.minimum(
        (rows.lead[train] // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1
    )
    x = rows.x[train]
    table = np.zeros((prep418.N_BANDS, len(std_cols)), dtype=np.float32)
    for b in range(prep418.N_BANDS):
        sel = band == b
        if sel.any():
            table[b] = np.median(x[sel][:, std_cols], axis=0)
    return table, std_cols


def pretrain_rows(*, rows: data.Rows, refit, clean: np.ndarray) -> data.Rows:  # noqa: ANN001
    """Return ERA5-and-CAMS pretraining rows on the fold's S1 input scale.

    A row is kept if its valid time is at least `REFIT_CUTOFF_AGE` before the refit, its origin is at
    or before the refit, its target half-hour is on the grid and is cleaned power (flagged and
    start-cut half-hours are NaN in `clean`), and it is not in the purged weeks. Held-out weeks are
    the same absolute weeks as the fold's.
    """
    keep_cols, _, _ = data.non_lag_layout()
    table, std_cols = spread_table(rows=rows)
    cutoff_valid = data._us(t=refit - data.regimes.REFIT_CUTOFF_AGE)
    cutoff_origin = data._us(t=refit)
    grid = data.grid_length()
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        keep = (z["valid"] <= cutoff_valid) & (z["origin"] <= cutoff_origin)
        origin_day = ((z["origin"] - data._us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(
            np.int64
        )
        held, purged = prep422.purge_masks(origin_day=origin_day)
        keep &= ~purged
        target_idx = ((z["valid"] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        origin_idx = ((z["origin"] - prep422.GRID_START_US) // prep422.STEP_US).astype(np.int64)
        keep &= (target_idx >= 0) & (target_idx < grid)
        y = np.full(len(keep), np.nan, dtype=np.float64)
        y[keep] = clean[row["primary"]][target_idx[keep]]
        keep &= np.isfinite(y)
        if not keep.any():
            continue
        x = np.empty((int(keep.sum()), len(keep_cols)), dtype=np.float32)
        raw = z["x"][keep][:, keep_cols]
        lead = z["lead"][keep]
        band = np.minimum((lead // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1)
        for a in range(0, len(x), 500_000):
            b = min(a + 500_000, len(x))
            chunk = rows.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            x[a:b] = chunk
        parts.append(
            {
                "x": x, "y": y[keep].astype(np.float32), "lead": lead,
                "primary": np.full(len(x), row["primary"], dtype=np.int64),
                "origin_idx": origin_idx[keep], "target_idx": target_idx[keep],
                "is_val": held[keep],
            }
        )  # fmt: skip
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    return data.Rows(
        x=cat["x"], y=cat["y"], primary=cat["primary"], origin_idx=cat["origin_idx"],
        target_idx=cat["target_idx"], lead=cat["lead"],
        weight=np.ones(len(cat["y"]), dtype=np.float32), is_val=cat["is_val"],
        standardiser=rows.standardiser, target_mean=rows.target_mean, target_std=rows.target_std,
        power_mean=rows.power_mean, power_std=rows.power_std, columns=rows.columns,
        dropped=0,
    )  # fmt: skip


def atomic_save(*, state: dict, target: Path) -> None:
    """Write a state dict to a temporary file, then rename it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(state, temporary)
    temporary.rename(target)


def tensors(*, rows: data.Rows, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the standardised target, the weight, and the held-out flag on `device`."""
    z = torch.from_numpy(
        ((rows.y - rows.target_mean[rows.primary]) / rows.target_std[rows.primary]).astype(
            np.float32
        )
    ).to(device)
    return z, torch.from_numpy(rows.weight).to(device), torch.from_numpy(rows.is_val).to(device)


def locate(*, arm: str, month: str, seed: int) -> Path:
    """Return the folder of a fitted network, searched in `MODEL_ROOTS`."""
    for root in MODEL_ROOTS:
        folder = root / arm / month / f"s{seed}"
        if (folder / "meta.json").exists():
            return folder
    raise FileNotFoundError(f"no fitted network for {arm} {month} s{seed}")


def load_from(*, folder: Path, arm_name: str, device: str) -> pr.Loaded:
    """Load a saved network from `folder`."""
    meta = json.loads((folder / "meta.json").read_text())
    z = np.load(folder / "standardiser.npz")
    arm = fit.ARMS[arm_name]
    model = fit.make_model(
        arm=arm, n_inputs=meta["n_inputs"], n_primaries=len(z["target_mean"])
    ).to(device)
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    model.eval()
    return pr.Loaded(
        arm=arm, model=model,
        standardiser=prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]),
        target_mean=z["target_mean"], target_std=z["target_std"],
        power_mean=meta["power_mean"], power_std=meta["power_std"], seed=meta["seed"],
    )  # fmt: skip


def make_inputs(
    *, rows: data.Rows, power: torch.Tensor, surprise: tuple[torch.Tensor, torch.Tensor] | None,
    device: str,
) -> fit.Inputs:  # fmt: skip
    """Return the inputs of a row set, with the fold's target statistics."""
    inputs = fit.Inputs(
        power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
        target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
        device=device, surprise=surprise,
    )  # fmt: skip
    inputs.target_stats = fit.target_stats(
        mean=rows.target_mean, std=rows.target_std, device=device
    )
    return inputs


def fit_arm(
    *, arm_name: str, month: str, seed: int, power: torch.Tensor, stack: np.ndarray, device: str,
    cache: dict,
) -> str:  # fmt: skip
    """Pretrain for `PRETRAIN_MAX_EPOCHS`, fine-tune on ensemble rows, and save in experiment 428's layout."""
    target = fit.MODELS_DIR / arm_name / month / f"s{seed}"
    if (target / "meta.json").exists():
        return f"{arm_name} {month} s{seed}: exists"
    arm = fit.ARMS[arm_name]
    started = time.monotonic()
    refit = prep.refit_of(month=month)
    if month not in cache:
        cache.clear()
        cache[month] = data.build_fold(refit=refit, clean=stack[0])
    rows: data.Rows = cache[month]
    ensemble_surprise = fit.surprise_for(arm=arm, month=month, seed=seed, device=device)
    pre_path = WORK_DIR / "pretrained" / arm_name / f"{month}_s{seed}.pt"
    torch.manual_seed(seed)
    model = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)).to(
        device
    )
    pre_epochs = pre_best = pre_rows = 0
    if pre_path.exists():
        model.load_state_dict(torch.load(pre_path, map_location=device))
    else:
        era5_surprise = None
        if arm.surprise:
            assert ensemble_surprise is not None
            era5_surprise = lagless.surprise_tables(month=month, seed=seed, device=device)
            assert not torch.equal(
                torch.nan_to_num(era5_surprise[0]), torch.nan_to_num(ensemble_surprise[0])
            ), "the pretraining and fine-tuning surprise tables must differ"
        else:
            assert ensemble_surprise is None
        if ("pre", month) not in cache:
            cache[("pre", month)] = pretrain_rows(rows=rows, refit=refit, clean=stack[0])
        pre = cache[("pre", month)]
        pre_rows = len(pre.x)
        pre_inputs = make_inputs(rows=pre, power=power, surprise=era5_surprise, device=device)
        z, w, is_val = tensors(rows=pre, device=device)
        fit.MAX_EPOCHS = PRETRAIN_MAX_EPOCHS
        try:
            pre_epochs, pre_best = fit.train(
                arm=arm, model=model, inputs=pre_inputs, z=z, w=w, is_val=is_val, seed=seed,
                device=device,
            )  # fmt: skip
        finally:
            fit.MAX_EPOCHS = FINE_TUNE_MAX_EPOCHS
        atomic_save(state=model.state_dict(), target=pre_path)
        del pre_inputs, z, w, is_val
        torch.cuda.empty_cache()
    inputs = make_inputs(rows=rows, power=power, surprise=ensemble_surprise, device=device)
    z, w, is_val = tensors(rows=rows, device=device)
    original = fit.SETTING
    fit.SETTING = replace(original, lr=FINE_TUNE_LR)
    try:
        epochs, best = fit.train(
            arm=arm, model=model, inputs=inputs, z=z, w=w, is_val=is_val, seed=seed, device=device
        )
    finally:
        fit.SETTING = original
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    np.savez(
        target / "standardiser.npz",
        mean=rows.standardiser.mean, std=rows.standardiser.std, mask=rows.standardiser.mask,
        target_mean=rows.target_mean, target_std=rows.target_std,
    )  # fmt: skip
    (target / "meta.json").write_text(
        json.dumps(
            {
                "arm": arm_name, "month": month, "seed": seed, "epochs": epochs, "val_nll": best,
                "pretrain_epochs": pre_epochs, "pretrain_val_nll": pre_best, "pretrain_rows": pre_rows,
                "power_mean": rows.power_mean, "power_std": rows.power_std,
                "n_rows": len(rows.x), "unusable_targets": rows.dropped,
                "n_inputs": int(rows.x.shape[1]),
                "parameters": fit.models.count_parameters(model=model),
                "seconds": round(time.monotonic() - started),
            }
        )
    )  # fmt: skip
    return (
        f"{arm_name} {month} s{seed}: pretrain {pre_epochs} epochs ({pre_rows:,} rows), "
        f"fine-tune {epochs} epochs, val nll {best:.4f}, {time.monotonic() - started:.0f} s"
    )


def quantile_average(*, arm_name: str, source_dir: Path, month: str) -> list[str]:
    """Write the five-seed quantile average of an arm under `<arm>-avg5`, one frame per world.

    The quantile columns are averaged over seeds 0 to 4 (experiment 425's seed average); the keys,
    the actual column, and the flag columns are identical across seeds and are checked to be. The
    same frame is written to each seed file, as the pools of experiment 449 are.
    """
    lines = []
    for w in pr.EVAL_WORLDS:
        targets = [pr.PRED_DIR / f"{arm_name}-avg5" / f"{month}_s{s}_w{w}.parquet" for s in SEEDS5]
        if all(t.exists() for t in targets):
            continue
        frames = [pl.read_parquet(source_dir / f"{month}_s{s}_w{w}.parquet") for s in SEEDS5]
        keys = ["primary", "origin_us", "valid_us", "lead"]
        for frame in frames[1:]:
            assert frame.select(keys).equals(frames[0].select(keys)), "seeds cover different rows"
            assert frame["actual"].equals(frames[0]["actual"]), "seeds disagree on the actual"
        levels = list(pr.LEVEL_NAMES)
        mean = frames[0].with_columns(
            [
                pl.mean_horizontal(*[f[level] for f in frames]).cast(pl.Float32).alias(level)
                for level in levels
            ]
        )
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            mean.write_parquet(temporary)
            temporary.rename(target)
        lines.append(f"{arm_name}-avg5 {month} w{w}: {mean.height:,} rows")
    return lines


def smoke(*, device: str) -> None:
    """Check the pretraining view of both arms: live lags, the surprise only for `AS1L`."""
    stack, _ = data.load_stack(include_eval=False)
    power = torch.from_numpy(stack).to(device)
    month = "2026-01"
    refit = prep.refit_of(month=month)
    rows = data.build_fold(refit=refit, clean=stack[0])
    pre = pretrain_rows(rows=rows, refit=refit, clean=stack[0])
    assert pre.x.shape[1] == rows.x.shape[1] and np.isfinite(pre.x).all()
    assert pre.is_val.any() and (~pre.is_val).any()
    assert pre.target_idx.max() < data.grid_length()
    assert pre.origin_idx.max() <= int(prep422.common.grid_index(times=refit))
    era5 = lagless.surprise_tables(month=month, seed=0, device=device)
    sample = np.sort(np.random.default_rng(0).choice(len(pre.x), 20_000, replace=False))
    small = data.Rows(
        **{
            **pre.__dict__,
            "x": pre.x[sample],
            "primary": pre.primary[sample],
            "origin_idx": pre.origin_idx[sample],
            "target_idx": pre.target_idx[sample],
        }
    )  # ty: ignore[invalid-argument-type]
    idx = torch.arange(20_000, device=device)
    world = torch.zeros_like(idx)
    for name in NEW_ARMS:
        arm = fit.ARMS[name]
        model = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean))
        print(name, "parameters", fit.models.count_parameters(model=model))
        inputs = make_inputs(
            rows=small, power=power, surprise=era5 if arm.surprise else None, device=device
        )
        x_in, _, _ = inputs.batch(arm=arm, rows=idx, world=world)
        assert torch.isfinite(x_in).all(), name
        block = x_in[:, rows.x.shape[1] :]
        print(
            name, "extra columns", block.shape[1], "lag missing share", float(block[:, 3:6].mean())
        )
        assert float(block[:, 3:6].mean()) < 0.05
        if arm.surprise:
            assert float((block[:, 9:12] != 0).float().mean()) > 0.5
    print("smoke test passed")


def main() -> None:
    """Fit, forecast, average, or run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "avg", "smoke"))
    parser.add_argument("--arms", nargs="+", default=list(NEW_ARMS))
    parser.add_argument("--months", nargs="+", default=[*fit.SCREEN_MONTHS, fit.FIFTH_MONTH])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS5))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install()
    if args.command == "smoke":
        smoke(device=args.device)
        return
    if args.command == "avg":
        for month in args.months:
            for name in args.arms:
                source = locate_predictions(arm=name)
                for line in quantile_average(arm_name=name, source_dir=source, month=month):
                    print(line, flush=True)
        return
    if args.command == "fit":
        stack, _ = data.load_stack(include_eval=False)
        power = torch.from_numpy(stack).to(args.device)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    print(fit_arm(arm_name=name, month=month, seed=seed, power=power, stack=stack,
                                  device=args.device, cache=cache), flush=True)  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                fit.MODELS_DIR = locate(arm=name, month=month, seed=seed).parents[2]
                for line in pr.predict_month(arm_name=name, month=month, seed=seed, device=args.device,
                                             power=power, flags=flags, y_clean=power_np[0]):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
