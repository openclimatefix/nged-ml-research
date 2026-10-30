"""Fit and forecast the library networks of experiment 413.

Stages (each writes only under this experiment's private data folder):

- `time`: time one D2 fit on the GPU, and one epoch of a fixed 200,000-row subsample on the GPU and
  on 1 CPU thread. Nothing is forecast or scored.
- `screen`: fit the named arms on the fold, save the networks and the learned curves, and forecast
  the test month.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00413-differentiable-physics-demand-network-part-1 \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00413-*/e00413_fit.py time --device cuda
```
"""

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as base_fit  # ty: ignore[unresolved-import]
import e00397_models as base_models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00413_models as models  # ty: ignore[unresolved-import]
import e00413_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FOLD_MONTH: Final[str] = "2026-01"
SETTING: Final[base_fit.Setting] = base_fit.setting_by_name(name="g01")
"""Experiment 397's plain-network setting (2 layers of 64, 3 components), used by every arm."""
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00413-differentiable-physics-demand-network-part-1" / "work"
)
TIMING_ROWS: Final[int] = 200_000
TEMPERATURE_GRID_C: Final[np.ndarray] = np.linspace(-10.0, 35.0, 91)


def make_model(*, arm: str, fold: prep.Fold, seed: int) -> torch.nn.Module:
    """Return an untrained network for an arm, with a fixed reference sample of training rows."""
    n_primaries = len(fold.target_mean)
    torch.manual_seed(seed)
    if arm == "D0":
        return models.PlainWrapper(
            n_base=len(prep.NAMES),
            n_primaries=n_primaries,
            hidden=SETTING.hidden,
            layers=SETTING.layers,
            components=SETTING.components,
        )
    train_rows = np.nonzero(~fold.is_val)[0]
    chosen = np.random.default_rng(413 + seed).choice(
        train_rows, size=min(models.REFERENCE_ROWS, len(train_rows)), replace=False
    )
    return models.PhysicsMdn(
        arm=arm,
        columns=fold.columns,
        n_primaries=n_primaries,
        reference_x=torch.from_numpy(fold.x[np.sort(chosen)]),
        hidden=SETTING.hidden,
        layers=SETTING.layers,
        components=SETTING.components,
    )


def fit_arm(
    *,
    arm: str,
    fold: prep.Fold,
    seed: int,
    device: str,
    row_limit: int | None = None,
    max_epochs: int | None = None,
) -> tuple[base_fit.Fitted, float]:
    """Fit an arm on a fold; return the fitted network and the seconds the training took."""
    z_all = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    select = (
        np.sort(np.random.default_rng(seed).choice(len(fold.x), row_limit, replace=False))
        if row_limit
        else np.arange(len(fold.x))
    )
    tensors = {
        "x": torch.from_numpy(fold.x[select]).to(device),
        "z": torch.from_numpy(z_all[select].astype(np.float32)).to(device),
        "w": torch.from_numpy(fold.weight[select]).to(device),
        "p": torch.from_numpy(fold.primary[select]).to(device),
        "v": torch.from_numpy(fold.is_val[select]).to(device),
    }
    model = make_model(arm=arm, fold=fold, seed=seed).to(device)
    previous = base_fit.MAX_EPOCHS
    if max_epochs is not None:
        base_fit.MAX_EPOCHS = max_epochs
    started = time.monotonic()
    try:
        model, epochs, val = base_fit.train_network(
            model=model,
            setting=SETTING,
            x=tensors["x"],
            z=tensors["z"],
            w=tensors["w"],
            primary=tensors["p"],
            is_val=tensors["v"],
            seed=seed,
        )
    finally:
        base_fit.MAX_EPOCHS = previous
    seconds = time.monotonic() - started
    return base_fit.Fitted(
        model=model, setting=SETTING, arm=arm, seed=seed, epochs=epochs, val_nll=val
    ), seconds


def forecast_month(
    *, fitted: base_fit.Fitted, fold: prep.Fold, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's quantiles (fraction of capacity) for every Primary and run."""
    pieces = []
    for origin in base.test_origins(month=month):
        for row, keys, x in prep.test_inputs(origin=origin, standardiser=fold.standardiser):
            p = int(row["primary"])
            q, _, _ = base_fit.predict_z_quantiles(
                fitted=fitted, x=x, primary=p, members=1, device=device
            )
            q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


def export_interpretation(*, fitted: base_fit.Fitted, fold: prep.Fold, target: Path) -> None:
    """Write the learned shape and response curves and the Primary mixing weights of an arm."""
    model = fitted.model
    if not isinstance(model, models.PhysicsMdn):
        return
    target.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cols = fold.columns
    ordinary = torch.median(model.reference_x, dim=0).values
    half_hours = np.arange(48) / 2.0
    grid = ordinary.repeat(48 * 7, 1)
    hour = np.tile(half_hours, 7)
    day = np.repeat(np.arange(1, 8), 48)
    for name, value in (
        ("sin_hour", np.sin(2 * np.pi * hour / 24)),
        ("cos_hour", np.cos(2 * np.pi * hour / 24)),
        ("sin_hour_2", np.sin(4 * np.pi * hour / 24)),
        ("cos_hour_2", np.cos(4 * np.pi * hour / 24)),
        *[(f"dow_{d}", (day == d).astype(np.float64)) for d in range(1, 8)],
    ):
        grid[:, prep.col(name=name)] = torch.from_numpy(value.astype(np.float32)).to(device)
    with torch.no_grad():
        shape, _ = model.shapes(grid.to(device))
        tables: dict[str, np.ndarray] = {
            "hour": hour,
            "day_of_week": day,
            "shapes": shape.cpu().numpy(),
        }
        index = torch.arange(len(model.embedding.weight), device=device)
        mix = model.embedding(index)
        tables["shape_weights"] = torch.softmax(model.to_shape_weights(mix), -1).cpu().numpy()
        tables["amplitudes"] = (
            torch.nn.functional.softplus(model.to_amplitude(mix) + 0.5413).cpu().numpy()
        )
        tables["level"] = model.level(index).squeeze(-1).cpu().numpy()
        tables["yearly_coefficients"] = model.yearly_coefficients.cpu().numpy()
        if model.with_responses:
            mean_t, std_t = (
                fold.standardiser.mean[cols.temperature],
                fold.standardiser.std[cols.temperature],
            )
            standardised = np.clip((TEMPERATURE_GRID_C - mean_t) / std_t, -5.0, 5.0)
            probe = ordinary.repeat(len(standardised), 1)
            probe[:, cols.temperature] = torch.from_numpy(standardised.astype(np.float32)).to(
                device
            )
            probe[:, cols.cos_zenith] = 0.0
            probe[:, cols.shortwave] = cols.shortwave_zero
            response = model.responses(probe.to(device))
            tables["temperature_c"] = TEMPERATURE_GRID_C
            tables["responses_of_temperature"] = response.cpu().numpy()
            tables["response_weights"] = (
                torch.softmax(model.to_response_weights(mix), -1).cpu().numpy()
            )
            tables["gate_coefficients"] = (
                (models.GATE_LIMIT * torch.tanh(model.to_gate(mix))).cpu().numpy()
            )
    np.savez(target / "interpretation.npz", **tables)  # ty: ignore[invalid-argument-type]


def monotone_below_reference(*, interpretation: Path) -> dict[str, bool]:
    """Return, for each Primary index, whether its mixed temperature response is non-increasing.

    The check covers temperatures below 15.5 degrees Celsius and mixes only the 3 heating curves
    and the cooling curve by the Primary's response weights.
    """
    z = np.load(interpretation)
    if "responses_of_temperature" not in z.files:
        return {}
    below = z["temperature_c"] < prep.REFERENCE_TEMPERATURE_C
    curves = z["responses_of_temperature"][below][:, :4]
    out = {}
    for p, weights in enumerate(z["response_weights"][:, :4]):
        mixed = curves @ weights
        out[str(p)] = bool((np.diff(mixed) <= 1e-6).all())
    return out


def save_fit(*, fitted: base_fit.Fitted, fold: prep.Fold, target: Path) -> None:
    """Save the network, the standardiser, the target statistics, and the setting."""
    target.mkdir(parents=True, exist_ok=True)
    torch.save(fitted.model.state_dict(), target / "model.pt")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "arm": fitted.arm,
                "setting": asdict(fitted.setting),
                "seed": fitted.seed,
                "epochs": fitted.epochs,
                "val_nll": fitted.val_nll,
                "target_mean": fold.target_mean.tolist(),
                "target_std": fold.target_std.tolist(),
                "dropped_target_rows": fold.dropped,
            }
        )
    )
    np.savez(
        target / "standardiser.npz",
        mean=fold.standardiser.mean,
        std=fold.standardiser.std,
        mask=fold.standardiser.mask,
    )


def peak_gpu_gb() -> float:
    """Return the peak GPU memory this process has allocated, in gigabytes."""
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def peak_rss_gb() -> float:
    """Return this process's peak resident memory, in gigabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_time(*, device: str) -> None:
    """Time one D2 fit and one subsample epoch on the GPU and on 1 CPU thread."""
    started = time.monotonic()
    fold = prep.build_fold(refit=base.refit_of(month=FOLD_MONTH))
    print(
        f"fold {FOLD_MONTH}: {len(fold.x):,} rows ({int(fold.is_val.sum()):,} early-stopping), "
        f"built in {time.monotonic() - started:.0f} s, peak RSS {peak_rss_gb():.1f} GB",
        flush=True,
    )
    print(f"rows dropped for a missing target, by site: {fold.dropped}", flush=True)
    fitted, seconds = fit_arm(arm="D2", fold=fold, seed=0, device=device)
    parameters = base_models.count_parameters(model=fitted.model)
    print(
        f"TIME D2 full fit on {device}: {seconds:.0f} s, {fitted.epochs} epochs "
        f"({seconds / fitted.epochs:.0f} s per epoch), {parameters:,} parameters, "
        f"peak GPU {peak_gpu_gb():.2f} GB, peak RSS {peak_rss_gb():.1f} GB",
        flush=True,
    )
    save_fit(fitted=fitted, fold=fold, target=OUT_DIR / "timing" / "D2")
    export_interpretation(fitted=fitted, fold=fold, target=OUT_DIR / "timing" / "D2")
    print(
        "monotone below the reference, by Primary:",
        monotone_below_reference(interpretation=OUT_DIR / "timing" / "D2" / "interpretation.npz"),
        flush=True,
    )
    for place in (device, "cpu"):
        torch.set_num_threads(1)
        _, epoch_seconds = fit_arm(
            arm="D2", fold=fold, seed=0, device=place, row_limit=TIMING_ROWS, max_epochs=1
        )
        print(f"TIME D2 one epoch of {TIMING_ROWS:,} rows on {place}: {epoch_seconds:.1f} s")


def run_screen(*, arms: list[str], seeds: list[int], device: str) -> None:
    """Fit each arm on the fold, save the network and curves, and forecast the test month."""
    fold = prep.build_fold(refit=base.refit_of(month=FOLD_MONTH))
    print(f"fold {FOLD_MONTH}: {len(fold.x):,} rows; dropped targets: {fold.dropped}", flush=True)
    for arm in arms:
        for seed in seeds:
            target = OUT_DIR / "seed_preds" / arm / f"{FOLD_MONTH}_s{seed}.parquet"
            if target.exists():
                continue
            fitted, seconds = fit_arm(arm=arm, fold=fold, seed=seed, device=device)
            folder = OUT_DIR / "models" / arm / FOLD_MONTH / f"s{seed}"
            save_fit(fitted=fitted, fold=fold, target=folder)
            export_interpretation(fitted=fitted, fold=fold, target=folder)
            common.atomic_write(
                frame=forecast_month(fitted=fitted, fold=fold, month=FOLD_MONTH, device=device),
                target=target,
            )
            print(
                f"{arm} {FOLD_MONTH} seed {seed}: {seconds:.0f} s, {fitted.epochs} epochs, "
                f"validation NLL {fitted.val_nll:.4f}, peak GPU {peak_gpu_gb():.2f} GB, "
                f"peak RSS {peak_rss_gb():.1f} GB",
                flush=True,
            )


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("time", "screen"))
    parser.add_argument("--arms", nargs="+", default=list(models.ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "time":
        run_time(device=args.device)
    else:
        run_screen(arms=args.arms, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
