# ruff: noqa: E501, PLR0912, PT018
"""Fit and forecast the ladder arms of experiment 448 on the S1 network.

Reuses experiment 428's rows, fit loop, and forecast code (and experiment 439's S1 seeds 3 to 5)
and redirects the output folders to this experiment's own work folder. New here:

- `S1PT`, `S1PT-noemb`, `RinjPT`: S1 (or S1 without the Primary embedding, or R-inj) pretrained on
  ERA5-and-CAMS rows with every recent-power input masked as unobserved (the lags are NaN, the
  surprise 0), then fine-tuned on ensemble rows with the inputs live. The pretraining rows are
  experiment 418's, taken through the fold's own standardiser, with the `std_*` columns set to the
  fold's median ensemble spread by lead band as in experiment 425.
- `L0` for seeds 1 to 4 (experiment 428 stored seed 0 only).
- Combination forecasts (`combo`): an equal-weight pool of seeds, and the fixed 50/50 B-mix of a
  lagged pool and a lagless pool, formed on the capacity scale as in experiment 428's `bmix`.

```bash
export DATA_PATH_INTERNAL=<path to the private data folder>
source <paths>; PYTHONPATH=$PP:ml-experiments/00448-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00448-*/e00448_run.py fit --arms S1PT --seeds 0
  uv run --no-sync python ml-experiments/00448-*/e00448_run.py predict --arms S1PT --seeds 0
  uv run --no-sync python ml-experiments/00448-*/e00448_run.py combo
  uv run --no-sync python ml-experiments/00448-*/e00448_run.py smoke
```
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00448-s1-network-assembly-ladder-with-pretraining-b-mix-and-seed-pool"
    / "work"
)
MODEL_ROOTS: Final[tuple[Path, ...]] = (
    WORK_DIR / "models",
    ML_EXPERIMENTS_DATA_DIR
    / "00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five/work/models",
    data.WORK_DIR / "models",
)
"""Where a fitted network is looked for: this experiment, then 439 (S1 seeds 3 to 5), then 428."""
FINE_TUNE_LR: Final[float] = 2e-4
"""Experiment 425's fine-tuning learning rate."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
PT_ARMS: Final[tuple[str, ...]] = ("S1PT", "S1PT-noemb", "RinjPT")
PRETRAIN_EPOCH_ROWS: Final[float] = 0.2
"""The pretraining pass draws this share of the rows per epoch, as the fit loop does."""

fit.ARMS["S1PT"] = fit.Arm("S1PT", "lags", 0, True, surprise=True, history="s1")
fit.ARMS["S1PT-noemb"] = fit.Arm("S1PT-noemb", "lags", 0, True, surprise=True, history="s1")
fit.ARMS["RinjPT"] = fit.Arm("RinjPT", "lags", 0, True)
_ORIGINAL_MAKE_MODEL = fit.make_model


def _make_model(*, arm: fit.Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
    """Return the arm's network; an arm named `*-noemb` has no Primary embedding."""
    return _ORIGINAL_MAKE_MODEL(
        arm=arm, n_inputs=n_inputs, n_primaries=0 if arm.name.endswith("noemb") else n_primaries
    )


def install() -> None:
    """Redirect experiment 428's modules to this experiment's folders."""
    fit.MODELS_DIR = WORK_DIR / "models"
    fit.make_model = _make_model  # ty: ignore[invalid-assignment]
    pr.PRED_DIR = WORK_DIR / "predictions"


class MaskedInputs(fit.Inputs):
    """Inputs whose recent-power channels are always masked (the pretraining view)."""

    def batch(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> tuple:
        """Return `(x_in, primary, windows)` with every lag NaN and the surprise 0."""
        return self.assemble(arm=arm, raw=self.raw(arm=arm, rows=rows, world=world), masked=True)


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


def fit_pt(
    *, arm_name: str, month: str, seed: int, power: torch.Tensor, stack: np.ndarray, device: str,
    cache: dict,
) -> str:  # fmt: skip
    """Pretrain on ERA5-and-CAMS rows, fine-tune on ensemble rows, and save in experiment 428's layout."""
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
    surprise = fit.surprise_for(arm=arm, month=month, seed=seed, device=device)
    n_primaries = len(rows.target_mean)
    pre_path = WORK_DIR / "pretrained" / arm_name / f"{month}_s{seed}.pt"
    torch.manual_seed(seed)
    model = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=n_primaries).to(device)
    pre_epochs = pre_best = pre_rows = 0
    if pre_path.exists():
        model.load_state_dict(torch.load(pre_path, map_location=device))
    else:
        pre = pretrain_rows(rows=rows, refit=refit, clean=stack[0])
        pre_rows = len(pre.x)
        pre_inputs = MaskedInputs(
            power=power, x=pre.x, primary=pre.primary, origin_idx=pre.origin_idx,
            target_idx=pre.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
            device=device, surprise=surprise,
        )  # fmt: skip
        pre_inputs.target_stats = fit.target_stats(
            mean=rows.target_mean, std=rows.target_std, device=device
        )
        z, w, is_val = tensors(rows=pre, device=device)
        pre_epochs, pre_best = fit.train(
            arm=arm,
            model=model,
            inputs=pre_inputs,
            z=z,
            w=w,
            is_val=is_val,
            seed=seed,
            device=device,
        )
        atomic_save(state=model.state_dict(), target=pre_path)
        del pre, pre_inputs, z, w, is_val
        torch.cuda.empty_cache()
    inputs = fit.Inputs(
        power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
        target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
        device=device, surprise=surprise,
    )  # fmt: skip
    inputs.target_stats = fit.target_stats(
        mean=rows.target_mean, std=rows.target_std, device=device
    )
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
            }
        )
    )  # fmt: skip
    return (
        f"{arm_name} {month} s{seed}: pretrain {pre_epochs} epochs ({pre_rows:,} rows), "
        f"fine-tune {epochs} epochs, val nll {best:.4f}, {time.monotonic() - started:.0f} s"
    )


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


# A combination: (lagged members, lagless members, weight of the lagged half). Members are
# (arm, seed) pairs. A weight of 1.0 means a plain pool with no lagless head.
SEEDS5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


def combos() -> dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]], float]]:
    """Return the combination arms (pools and B-mixes). Single-seed B-mix arms are per seed."""
    out: dict[str, tuple[list, list, float]] = {}
    for name, arm in (
        ("S1", "S1"),
        ("S1PT", "S1PT"),
        ("S1PT-noemb", "S1PT-noemb"),
        ("RinjPT", "RinjPT"),
    ):
        out[f"{name}-pool5"] = ([(arm, s) for s in SEEDS5], [], 1.0)
    out["S1-pool3"] = ([("S1", s) for s in (0, 1, 2)], [], 1.0)
    out["S1PT-pool3"] = ([("S1PT", s) for s in (0, 1, 2)], [], 1.0)
    out["S1-pool5-Bmix"] = ([("S1", s) for s in SEEDS5], [("L0", s) for s in SEEDS5], 0.5)
    out["S1PT-pool5-Bmix"] = ([("S1PT", s) for s in SEEDS5], [("L0", s) for s in SEEDS5], 0.5)
    return out


def single_bmix(*, arm: str, seed: int) -> tuple[list, list, float]:
    """Return the single-seed B-mix spec of a lagged arm and the same seed's L0."""
    return ([(arm, seed)], [("L0", seed)], 0.5)


def forecast_combo(
    *, month: str, name: str, spec: tuple[list, list, float], seed_files: tuple[int, ...],
    device: str, power: torch.Tensor, flags: pr.Flags, y_clean: np.ndarray,
) -> list[str]:  # fmt: skip
    """Forecast an equal-weight pool of lagged members, optionally blended 50/50 with a lagless pool.

    The mixtures are pooled on the capacity scale: each member's components are converted from z
    space with its own network's target statistics, member log weights are shifted by
    `-log(members)`, and a group's log weights are shifted by the log of the group weight. The same
    frame is written to each file in `seed_files`.
    """
    lagged, lagless, weight = spec
    targets = {
        w: pr.PRED_DIR / name / f"{month}_s{s}_w{w}.parquet"
        for s in seed_files
        for w in pr.EVAL_WORLDS
    }
    if all(t.exists() for t in targets.values()):
        return [f"{name} {month}: exists"]
    members = [(a, s, weight / len(lagged)) for a, s in lagged]
    members += [(a, s, (1.0 - weight) / len(lagless)) for a, s in lagless]
    loaded = [
        (load_from(folder=locate(arm=a, month=month, seed=s), arm_name=a, device=device), s, g)
        for a, s, g in members
    ]
    first = loaded[0][0]
    test = data.build_test(month=month, standardiser=first.standardiser)
    for member, _, _ in loaded:
        assert np.array_equal(member.standardiser.mean, first.standardiser.mean), (
            "pooled networks must share the fold's standardiser"
        )
    inputs = [
        fit.Inputs(
            power=power, x=test.x, primary=test.primary, origin_idx=test.origin_idx,
            target_idx=test.target_idx, power_mean=m.power_mean, power_std=m.power_std,
            device=device, surprise=fit.surprise_for(arm=m.arm, month=month, seed=s, device=device),
        )
        for m, s, _ in loaded
    ]  # fmt: skip
    for inp, (m, _, _) in zip(inputs, loaded, strict=True):
        inp.target_stats = fit.target_stats(mean=m.target_mean, std=m.target_std, device=device)
    lines = []
    for w in pr.EVAL_WORLDS:
        if all(
            pr.PRED_DIR.joinpath(name, f"{month}_s{s}_w{w}.parquet").exists() for s in seed_files
        ):
            continue
        columns: dict[str, list[np.ndarray]] = {n: [] for n in pr.LEVEL_NAMES}
        flag_columns: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(test.x), pr.CHUNK):
            rows = torch.arange(start, min(start + pr.CHUNK, len(test.x)), device=device)
            world = torch.full((len(rows),), w, dtype=torch.long, device=device)
            pieces = []
            for (m, _, g), inp in zip(loaded, inputs, strict=True):
                res = pr.forecast_rows(loaded=m, inputs=inp, rows=rows, world=world)
                primary = inp.primary[rows]
                mu = torch.from_numpy(m.target_mean).to(device)[primary][:, None]
                sd = torch.from_numpy(m.target_std).to(device)[primary][:, None]
                pieces.append(
                    (res["log_w"] + float(np.log(g)), res["mean"] * sd + mu, res["std"] * sd)
                )
            with torch.no_grad():
                q = base.mixture_quantiles(
                    log_w=torch.cat([p[0] for p in pieces], dim=1),
                    mean=torch.cat([p[1] for p in pieces], dim=1),
                    std=torch.cat([p[2] for p in pieces], dim=1),
                )
                q = torch.sort(q, dim=1).values
            for i, level in enumerate(pr.LEVEL_NAMES):
                columns[level].append(q[:, i].float().cpu().numpy())
            for key, value in flags.of(inputs=inputs[0], rows=rows, world=world).items():
                flag_columns.setdefault(key, []).append(value.cpu().numpy())
        frame = pl.DataFrame(
            {
                "primary": test.primary, "origin_us": test.origin, "valid_us": test.valid,
                "lead": test.lead, "actual": y_clean[test.primary, test.target_idx],
                **{k: np.concatenate(v) for k, v in columns.items()},
                **{k: np.concatenate(v) for k, v in flag_columns.items()},
            }
        )  # fmt: skip
        for s in seed_files:
            target = pr.PRED_DIR / name / f"{month}_s{s}_w{w}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            frame.write_parquet(temporary)
            temporary.rename(target)
        lines.append(f"{name} {month} w{w}: {frame.height:,} rows")
    return lines


def smoke(*, device: str) -> None:
    """Check pretraining rows, masking, and the parameter count before any fit is trusted."""
    stack, _ = data.load_stack(include_eval=False)
    power = torch.from_numpy(stack).to(device)
    month = "2026-01"
    refit = prep.refit_of(month=month)
    rows = data.build_fold(refit=refit, clean=stack[0])
    pre = pretrain_rows(rows=rows, refit=refit, clean=stack[0])
    assert pre.x.shape[1] == rows.x.shape[1], "pretraining and fine-tuning rows differ in width"
    assert np.isfinite(pre.x).all() and np.isfinite(pre.y).all()
    assert pre.is_val.any() and (~pre.is_val).any()
    _, ends = pre.target_idx.min(), pre.target_idx.max()
    assert ends < data.grid_length()
    print(
        f"pretraining rows {len(pre.x):,}, fine-tuning rows {len(rows.x):,}, width {pre.x.shape[1]}"
    )
    for name in PT_ARMS:
        arm = fit.ARMS[name]
        surprise = fit.surprise_for(arm=arm, month=month, seed=0, device=device)
        sample = slice(0, 4096)
        inputs = MaskedInputs(
            power=power, x=pre.x[sample], primary=pre.primary[sample], origin_idx=pre.origin_idx[sample],
            target_idx=pre.target_idx[sample], power_mean=rows.power_mean, power_std=rows.power_std,
            device=device, surprise=surprise,
        )  # fmt: skip
        idx = torch.arange(4096, device=device)
        x_in, _, _ = inputs.batch(arm=arm, rows=idx, world=torch.zeros_like(idx))
        assert torch.isfinite(x_in).all(), f"{name}: non-finite masked input"
        width = rows.x.shape[1] + 9 + (3 if arm.history == "s1" else 0)
        assert x_in.shape[1] == width, (name, x_in.shape, width)
        # Masked lags: the missing flags (columns 3 to 5 after x) are all 1 and the values 0.
        lag_block = x_in[:, rows.x.shape[1] :]
        assert (lag_block[:, :3] == 0).all() and (lag_block[:, 3:6] == 1).all(), name
        if arm.history == "s1":
            assert (lag_block[:, 9:12] == 0).all(), "masked surprise must be 0"
        net = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean))
        print(name, "parameters", fit.models.count_parameters(model=net))
    stored = json.loads((locate(arm="S1", month=month, seed=0) / "meta.json").read_text())
    ours = fit.make_model(
        arm=fit.ARMS["S1PT"], n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)
    )
    assert fit.models.count_parameters(model=ours) == stored["parameters"] == 8585
    print("smoke test passed")


def main() -> None:
    """Fit, forecast, or run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "combo", "smoke"))
    parser.add_argument("--arms", nargs="+", default=["S1PT"])
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    install()
    if args.command == "smoke":
        smoke(device=args.device)
        return
    if args.command == "fit":
        stack, _ = data.load_stack(include_eval=False)
        power = torch.from_numpy(stack).to(args.device)
        cache: dict = {}
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    if name in PT_ARMS:
                        print(fit_pt(arm_name=name, month=month, seed=seed, power=power, stack=stack,
                                     device=args.device, cache=cache), flush=True)  # fmt: skip
                    else:
                        print(fit.fit_one(arm_name=name, month=month, seed=seed, power=power, stack=stack,
                                          device=args.device, rows_cache=cache), flush=True)  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    if args.command == "predict":
        for month in args.months:
            for seed in args.seeds:
                for name in args.arms:
                    # The loader reads this experiment's folder first, then 439's and 428's.
                    fit.MODELS_DIR = locate(arm=name, month=month, seed=seed).parents[2]
                    for line in pr.predict_month(arm_name=name, month=month, seed=seed, device=args.device,
                                                 power=power, flags=flags, y_clean=power_np[0]):  # fmt: skip
                        print(line, flush=True)
        return
    specs = combos()
    for month in args.months:
        for name, spec in specs.items():
            if args.arms != ["S1PT"] and name not in args.arms:
                continue
            for line in forecast_combo(month=month, name=name, spec=spec, seed_files=SEEDS5, device=args.device,
                                       power=power, flags=flags, y_clean=power_np[0]):  # fmt: skip
                print(line, flush=True)
        for arm in ("S1", "S1PT"):
            name = f"{arm}-Bmix"
            if args.arms != ["S1PT"] and name not in args.arms:
                continue
            for seed in SEEDS5:
                for line in forecast_combo(month=month, name=name, spec=single_bmix(arm=arm, seed=seed),
                                           seed_files=(seed,), device=args.device, power=power, flags=flags,
                                           y_clean=power_np[0]):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
