# ruff: noqa: E501, PLR0912, PT018
"""Pretrain the S1 network with the lags and surprise inputs live, fine-tune it, and forecast.

Reuses experiment 428's rows, fit loop, and forecast code. The pretraining rows, spread table, and
helpers are copied from experiment 448's `e00448_run.py` (branch `exp-best-mdn-assembly`, commit
ca44c703), because experiment 448's folder is not on `main`. New here, four arms, all S1 networks
pretrained on ERA5-and-CAMS rows and then fine-tuned on ensemble rows with the inputs live:

- `PT1`: pretraining sees the three origin lags and the surprise at each lag. The surprise comes from
  `e00449_lagless` (a cross-fitted lagless forecast on ERA5 and CAMS weather), not from experiment
  404's ensemble-based table. Fine-tuning at learning rate 2e-4.
- `PT1-full`: the `PT1` pretrained network, fine-tuned at the full learning rate.
- `PT0`: experiment 448's masked-lag pretraining (refitted here only for the fifth fold).
- `PT1b`: lags live and surprise zero in pretraining (diagnostic).
- `PT2`: as `PT1`, with each lag token masked with probability `LAG_DROPOUT` while training.

```bash
source /tmp/e449_env.sh
systemd-run --user --scope -q -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00449-*/e00449_run.py fit --arms PT1 PT1-full --seeds 0
  uv run --no-sync python ml-experiments/00449-*/e00449_run.py predict --arms PT1 --seeds 0
  uv run --no-sync python ml-experiments/00449-*/e00449_run.py smoke
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
import e00449_lagless as lagless  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = lagless.WORK_DIR
MODEL_ROOTS: Final[tuple[Path, ...]] = (
    WORK_DIR / "models",
    ML_EXPERIMENTS_DATA_DIR
    / "00439-surprise-at-the-lags-confirmation-on-seeds-three-to-five/work/models",
    data.WORK_DIR / "models",
)
"""Where a fitted network is looked for: this experiment, then 439 (S1 seeds 3 to 5), then 428."""
FINE_TUNE_LR: Final[float] = 2e-4
"""Experiment 425's fine-tuning learning rate."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
LAG_DROPOUT: Final[float] = 0.3
"""Probability that `PT2` masks one lag token (its value, flag, and surprise; the age column stays) in pretraining."""
PT_ARMS: Final[tuple[str, ...]] = ("PT0", "PT1", "PT1-full", "PT1b", "PT2")
PRETRAIN_OF: Final[dict[str, str]] = {
    "PT0": "PT0",
    "PT1": "PT1",
    "PT1-full": "PT1",
    "PT1b": "PT1b",
    "PT2": "PT2",
}
"""The pretrained network each arm starts from (`PT1` and `PT1-full` share one)."""

for _name in PT_ARMS:
    fit.ARMS[_name] = fit.Arm(_name, "lags", 0, True, surprise=True, history="s1")


def install() -> None:
    """Redirect experiment 428's modules to this experiment's folders."""
    fit.MODELS_DIR = WORK_DIR / "models"
    pr.PRED_DIR = WORK_DIR / "predictions"


class MaskedInputs(fit.Inputs):
    """Inputs whose recent-power channels are always masked (experiment 448's pretraining view)."""

    def batch(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> tuple:
        """Return `(x_in, primary, windows)` with every lag NaN and the surprise 0."""
        return self.assemble(arm=arm, raw=self.raw(arm=arm, rows=rows, world=world), masked=True)


class NoSurpriseInputs(fit.Inputs):
    """Lags live, surprise always zero (the `PT1b` pretraining view)."""

    def lag_surprise(self, *, raw: dict, lags: torch.Tensor) -> torch.Tensor:
        """Return a zero surprise for every lag."""
        return torch.zeros_like(lags)


class DropoutInputs(fit.Inputs):
    """Inputs that mask each lag token with probability `LAG_DROPOUT` while the model trains."""

    model: torch.nn.Module | None = None

    def batch(self, *, arm: fit.Arm, rows: torch.Tensor, world: torch.Tensor) -> tuple:
        """Return `(x_in, primary, windows)`; in training mode some lags are set to NaN."""
        raw = self.raw(arm=arm, rows=rows, world=world)
        if self.model is not None and self.model.training:
            drop = torch.rand(raw["lags"].shape, device=raw["lags"].device) < LAG_DROPOUT
            raw["lags"] = torch.where(drop, torch.full_like(raw["lags"], float("nan")), raw["lags"])
        return self.assemble(arm=arm, raw=raw)


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


def pretrain_inputs(
    *, kind: str, pre: data.Rows, rows: data.Rows, power: torch.Tensor,
    surprise: tuple[torch.Tensor, torch.Tensor], model: torch.nn.Module, device: str,
) -> fit.Inputs:  # fmt: skip
    """Return the pretraining inputs of a pretraining kind (`PT1`, `PT1b`, or `PT2`)."""
    cls = {"PT0": MaskedInputs, "PT1": fit.Inputs, "PT1b": NoSurpriseInputs, "PT2": DropoutInputs}[
        kind
    ]
    inputs = cls(
        power=power, x=pre.x, primary=pre.primary, origin_idx=pre.origin_idx,
        target_idx=pre.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
        device=device, surprise=surprise,
    )  # fmt: skip
    if isinstance(inputs, DropoutInputs):
        inputs.model = model
    inputs.target_stats = fit.target_stats(
        mean=rows.target_mean, std=rows.target_std, device=device
    )
    return inputs


def fit_pt(
    *, arm_name: str, month: str, seed: int, power: torch.Tensor, stack: np.ndarray, device: str,
    cache: dict,
) -> str:  # fmt: skip
    """Pretrain with the lags live, fine-tune on ensemble rows, and save in experiment 428's layout."""
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
    assert ensemble_surprise is not None
    kind = PRETRAIN_OF[arm_name]
    pre_path = WORK_DIR / "pretrained" / kind / f"{month}_s{seed}.pt"
    n_primaries = len(rows.target_mean)
    torch.manual_seed(seed)
    model = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=n_primaries).to(device)
    pre_epochs = pre_best = pre_rows = 0
    if pre_path.exists():
        model.load_state_dict(torch.load(pre_path, map_location=device))
    else:
        era5_surprise = lagless.surprise_tables(month=month, seed=seed, device=device)
        assert not torch.equal(
            torch.nan_to_num(era5_surprise[0]), torch.nan_to_num(ensemble_surprise[0])
        ), "the pretraining and fine-tuning surprise tables must differ"
        if ("pre", month) not in cache:
            cache[("pre", month)] = pretrain_rows(rows=rows, refit=refit, clean=stack[0])
        pre = cache[("pre", month)]
        pre_rows = len(pre.x)
        pre_inputs = pretrain_inputs(
            kind=kind, pre=pre, rows=rows, power=power, surprise=era5_surprise, model=model,
            device=device,
        )  # fmt: skip
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
        del pre_inputs, z, w, is_val
        torch.cuda.empty_cache()
    inputs = fit.Inputs(
        power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
        target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
        device=device, surprise=ensemble_surprise,
    )  # fmt: skip
    inputs.target_stats = fit.target_stats(
        mean=rows.target_mean, std=rows.target_std, device=device
    )
    z, w, is_val = tensors(rows=rows, device=device)
    original = fit.SETTING
    fit.SETTING = original if arm_name.endswith("full") else replace(original, lr=FINE_TUNE_LR)
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


SEEDS5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)


def combos() -> dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]], float]]:
    """Return the pools: equal-weight mixtures of seeds, no lagless head (weight 1.0)."""
    return {
        "PT1-pool3": ([("PT1", s) for s in (0, 1, 2)], [], 1.0),
        "PT1-pool5": ([("PT1", s) for s in SEEDS5], [], 1.0),
    }


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
        assert np.array_equal(member.standardiser.mean, first.standardiser.mean) and np.array_equal(
            member.standardiser.std, first.standardiser.std
        ), "pooled networks must share the fold's standardiser"
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
    """Check the pretraining views: live lags, zero or live surprise, dropout, and the width."""
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
    idx = torch.arange(20_000, device=device)
    world = torch.zeros_like(idx)
    for name in PT_ARMS:
        arm = fit.ARMS[name]
        model = fit.make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean))
        assert fit.models.count_parameters(model=model) == 8585
        model.train()
        inputs = pretrain_inputs(
            kind=PRETRAIN_OF[name], pre=data.Rows(**{**pre.__dict__, "x": pre.x[sample], "primary": pre.primary[sample], "origin_idx": pre.origin_idx[sample], "target_idx": pre.target_idx[sample]}),
            rows=rows, power=power, surprise=era5, model=model.to(device), device=device,
        )  # fmt: skip
        x_in, _, _ = inputs.batch(arm=arm, rows=idx, world=world)
        assert torch.isfinite(x_in).all(), name
        block = x_in[:, rows.x.shape[1] :]
        missing = block[:, 3:6]
        surprise = block[:, 9:12]
        print(
            name, "lag missing share", float(missing.mean()), "nonzero surprise share",
            float((surprise != 0).float().mean()),
        )  # fmt: skip
        if name in ("PT1", "PT1-full"):
            assert float(missing.mean()) < 0.05 and float((surprise != 0).float().mean()) > 0.5
        if name == "PT1b":
            assert (surprise == 0).all() and float(missing.mean()) < 0.05
        if name == "PT2":
            assert 0.2 < float(missing.mean()) < 0.4
            assert ((surprise == 0) | (missing == 0)).all(), "a masked lag must have zero surprise"
    print("smoke test passed")


def main() -> None:
    """Fit, forecast, or run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "predict", "combo", "smoke"))
    parser.add_argument("--arms", nargs="+", default=list(PT_ARMS))
    parser.add_argument("--months", nargs="+", default=[*fit.SCREEN_MONTHS, fit.FIFTH_MONTH])
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
                    print(fit_pt(arm_name=name, month=month, seed=seed, power=power, stack=stack,
                                 device=args.device, cache=cache), flush=True)  # fmt: skip
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    if args.command == "combo":
        for month in args.months:
            for name, spec in combos().items():
                for line in forecast_combo(month=month, name=name, spec=spec, seed_files=SEEDS5, device=args.device,
                                           power=power, flags=flags, y_clean=power_np[0]):  # fmt: skip
                    print(line, flush=True)
        return
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                fit.MODELS_DIR = locate(arm=name, month=month, seed=seed).parents[2]
                for line in pr.predict_month(arm_name=name, month=month, seed=seed, device=args.device,
                                             power=power, flags=flags, y_clean=power_np[0]):  # fmt: skip
                    print(line, flush=True)


if __name__ == "__main__":
    main()
