# ruff: noqa: E501
r"""Fine-tune experiment 425's arm A with ERA5-and-CAMS replay rows mixed into the batches.

Every arm starts from the stored arm A pretrained weights of experiment 425
(`work/pretrained/A/<month>_s<seed>.pt`) and fine-tunes on the ensemble (ENS) rows of
`e00425_prep.build_ens_fold` at experiment 425's learning rate of 2e-4. The loop is experiment
397's `train_network` with one addition: each step also draws replay rows from the fold's
ERA5-and-CAMS pretraining rows (`e00425_prep.fold_pretrain`, held-out weeks excluded).

- `R0`: no replay. This is experiment 425's stored arm A fit, which is not refitted; `--arms R0`
  exists only for the one reproduction check (one month, one seed).
- `R25`: replay rows are 25% of each batch's loss weight (and 1.33 times the ENS rows per step).
- `R50`: replay rows are 50% of each batch's loss weight (and 2 times the ENS rows per step).
- `Rcold`: as `R25`, but replay rows are drawn only from cold-night ERA5-and-CAMS rows (sun more
  than 5 degrees below the horizon and 2 m temperature of 3 degrees Celsius or below, the
  protocol's `cold_nights` definition).

Epoch, steps, and loss: an epoch is one pass over the ENS training rows in batches of 16,384 ENS
rows, exactly as `R0`. Each step adds `round(16384 * r / (1 - r))` replay rows (5,461 for r = 0.25,
16,384 for r = 0.5), drawn uniformly with replacement from the replay pool. The step loss is
`(1 - r) * ENS loss + r * replay loss`, where the ENS loss is the recency-weighted mean negative
log-likelihood of `R0` (weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`) and the replay loss is
the unweighted mean (pretraining used equal weights). With r = 0 the step is `R0`'s. Early stopping
(patience 6, at most 40 epochs, best-epoch restore) reads the unweighted mean negative
log-likelihood of the ENS held-out weeks only, as in `R0`; replay rows never enter the validation.

Memory plan: the fold's ENS rows (4.9 million x 46 float32, 0.9 GB) and the replay pool (7.9
million x 46 float32, 1.5 GB) both sit on the GPU for the whole month, about 3 GB with the
targets and indices. The host builds the arrays once per month (peak resident memory about 6 GB)
and frees them after the copy.

```bash
source .e_replay_env.sh   # DATA_PATH_INTERNAL, dummy S3 variables, and PYTHONPATH (run from the repo root)
systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00487-*/e00487_fit.py --arms R25 R50 Rcold
```
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = (
    "00487-era5-and-cams-replay-rows-during-demand-network-fine-tuning"
)
WORK_DIR: Path = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME / "work"
"""Replaced by `work/smoke` under `--smoke`."""
E425_WORK: Final[Path] = prep.WORK_DIR / "work"
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The five folds of experiment 449 (and experiment 470's scored months)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
COLD_SUN_BELOW_HORIZON_DEG: Final[float] = 5.0
COLD_MAX_TEMPERATURE_C: Final[float] = 3.0


@dataclass(frozen=True)
class Arm:
    """One arm: the share of the step loss given to replay rows, and which replay rows."""

    name: str
    replay_share: float
    cold_only: bool = False

    @property
    def rows_per_ens_row(self) -> float:
        """Return replay rows drawn per ENS row in a batch (0 for no replay)."""
        return self.replay_share / (1.0 - self.replay_share)


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("R0", 0.0),
        Arm("R25", 0.25),
        Arm("R50", 0.5),
        Arm("Rcold", 0.25, cold_only=True),
    )
}
"""`R0` is experiment 425's stored fit; it is refitted only for the one reproduction check."""


def cold_night_mask(*, x: np.ndarray, fold: prep.EnsFold) -> np.ndarray:
    """Return which standardised rows are cold nights (the protocol's `cold_nights` slice).

    The temperature and the cosine of the solar zenith angle are inverted from the fold's
    standardiser (columns that it does not standardise have mean 0 and standard deviation 1). The
    sun is more than 5 degrees below the horizon when the cosine of the zenith angle is below
    `-sin(5 degrees)`. The temperature is the weather source's own 2 m temperature (Celsius).
    """
    names, _, _ = base.feature_names(scheme="summary")
    st = fold.standardiser
    temperature_i, cos_zenith_i = names.index("mean_temperature_2m"), names.index("cos_zenith")
    temperature = x[:, temperature_i] * st.std[temperature_i] + st.mean[temperature_i]
    cos_zenith = x[:, cos_zenith_i] * st.std[cos_zenith_i] + st.mean[cos_zenith_i]
    night = cos_zenith < -np.sin(np.radians(COLD_SUN_BELOW_HORIZON_DEG))
    return night & (temperature <= COLD_MAX_TEMPERATURE_C)


@dataclass
class Replay:
    """The replay pool on the device: standardised rows, standardised targets, and Primaries."""

    x: torch.Tensor
    z: torch.Tensor
    p: torch.Tensor
    all_idx: torch.Tensor
    cold_idx: torch.Tensor


def build_replay(*, fold: prep.EnsFold, pre: prep418.PretrainSet, device: str) -> Replay:
    """Return the replay pool (held-out weeks excluded) on the device."""
    train = ~pre.is_val
    z = (pre.y - fold.target_mean[pre.primary]) / fold.target_std[pre.primary]
    cold = cold_night_mask(x=pre.x, fold=fold)
    names, _, _ = base.feature_names(scheme="summary")
    i = names.index("mean_temperature_2m")
    celsius = pre.x[::100, i] * fold.standardiser.std[i] + fold.standardiser.mean[i]
    print(
        f"  replay 2 m temperature (min, median, max): {celsius.min():.1f}, "
        f"{np.median(celsius):.1f}, {celsius.max():.1f}",
        flush=True,
    )
    return Replay(
        x=torch.from_numpy(pre.x).to(device),
        z=torch.from_numpy(z.astype(np.float32)).to(device),
        p=torch.from_numpy(pre.primary).to(device),
        all_idx=torch.from_numpy(np.nonzero(train)[0]).to(device),
        cold_idx=torch.from_numpy(np.nonzero(train & cold)[0]).to(device),
    )


def train_with_replay(
    *,
    model: torch.nn.Module,
    setting: fit397.Setting,
    t: dict[str, torch.Tensor],
    replay: Replay | None,
    arm: Arm,
    seed: int,
) -> tuple[int, float]:
    """Train in place with experiment 397's loop plus replay rows; return epochs and best NLL."""
    torch.manual_seed(seed)
    generator = torch.Generator(device=t["x"].device)
    generator.manual_seed(seed)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    pool = None
    if arm.replay_share > 0.0:
        assert replay is not None
        pool = replay.cold_idx if arm.cold_only else replay.all_idx
        assert len(pool) > 0, "empty replay pool"
    n_replay = round(fit397.BATCH * arm.rows_per_ens_row)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(fit397.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=t["x"].device)]
        for start in range(0, len(order), fit397.BATCH):
            b = order[start : start + fit397.BATCH]
            nll = models.mixture_nll(
                out=fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]),
                z=t["z"][b],
                components=setting.components,
            )
            loss = (t["w"][b] * nll).sum() / t["w"][b].sum()
            if pool is not None:
                assert replay is not None
                r = pool[
                    torch.randint(len(pool), (n_replay,), device=pool.device, generator=generator)
                ]
                rnll = models.mixture_nll(
                    out=fit397.forward(model=model, x=replay.x[r], primary=replay.p[r]),
                    z=replay.z[r],
                    components=setting.components,
                )
                loss = (1.0 - arm.replay_share) * loss + arm.replay_share * rnll.mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit397.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        models.mixture_nll(
                            out=fit397.forward(model=model, x=t["x"][b], primary=t["p"][b]),
                            z=t["z"][b],
                            components=setting.components,
                        )
                        for b in val_idx.split(fit397.BATCH * 4)
                    ]
                ).mean()
            )
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= fit397.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def peak_gpu_gb() -> float:
    """Return the peak GPU memory allocated in gigabytes (0 on a CPU run)."""
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def seed_pred_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return WORK_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def run(*, months: list[str], arms: list[Arm], seeds: list[int], device: str, smoke: bool) -> None:
    """Fine-tune each arm on each fold and seed from the stored pretrained weights."""
    a425 = fit425.ARMS["A"]
    for month in months:
        todo = [
            (arm, seed)
            for arm in arms
            for seed in seeds
            if not seed_pred_path(arm=arm.name, month=month, seed=seed).exists()
        ]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        started = time.monotonic()
        fold = prep.build_ens_fold(refit=refit)
        needs_replay = any(arm.replay_share > 0.0 for arm, _ in todo)
        replay = None
        if needs_replay:
            pre, dropped = prep.fold_pretrain(fold=fold, refit=refit)
            replay = build_replay(fold=fold, pre=pre, device=device)
            print(
                f"fold {month}: {len(fold.x):,} ENS rows, {len(pre.x):,} replay rows "
                f"({len(replay.all_idx):,} train, {len(replay.cold_idx):,} cold nights); "
                f"dropped {dropped.as_dict()}; built in {time.monotonic() - started:.0f} s",
                flush=True,
            )
            del pre
        t = fit425.tensors(
            x=fold.x,
            y=fold.y,
            primary=fold.primary,
            weight=fold.weight,
            is_val=fold.is_val,
            fold=fold,
            device=device,
        )
        log_path = WORK_DIR / "logs" / f"{month}.json"
        log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
        log |= {
            "month": month,
            "fine_tune_rows": len(fold.x),
            "fine_tune_val_rows": int(fold.is_val.sum()),
            "fine_tune_rows_dropped": fold.dropped.as_dict(),
        }
        if replay is not None:
            log |= {
                "replay_train_rows": len(replay.all_idx),
                "replay_cold_night_rows": len(replay.cold_idx),
            }
        for arm, seed in todo:
            fit_started = time.monotonic()
            torch.manual_seed(seed)
            model = fit425.new_model(arm=a425, fold=fold).to(device)
            model.load_state_dict(
                torch.load(
                    E425_WORK / "pretrained" / "A" / f"{month}_s{seed}.pt", map_location=device
                )
            )
            setting = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
            epochs, best = train_with_replay(
                model=model, setting=setting, t=t, replay=replay, arm=arm, seed=seed
            )
            fitted = fit397.Fitted(
                model=model,
                setting=setting,
                arm="plain_pooled",
                seed=seed,
                epochs=epochs,
                val_nll=best,
            )
            fit_seconds = time.monotonic() - fit_started
            if not smoke:
                fit397.save_fits(
                    fits={None: fitted},
                    fold=fold,
                    target=WORK_DIR / "models" / arm.name / month / f"s{seed}",
                )
            q, _ = fit397.forecast_month(
                fits={None: fitted}, fold=fold, scheme="summary", month=month, device=device
            )
            common.atomic_write(
                frame=q, target=seed_pred_path(arm=arm.name, month=month, seed=seed)
            )
            seconds = round(time.monotonic() - fit_started)
            log[f"{arm.name}_s{seed}"] = {
                "epochs": epochs,
                "val_nll": best,
                "fit_seconds": round(fit_seconds),
                "seconds": seconds,
                "peak_gpu_gb": round(peak_gpu_gb(), 2),
            }
            print(
                f"{arm.name} {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, "
                f"fit {fit_seconds:.0f} s, with forecast {seconds} s, "
                f"peak GPU {peak_gpu_gb():.2f} GB",
                flush=True,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = log_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(log, indent=1))
            tmp.replace(log_path)
        del t, replay
        torch.cuda.empty_cache()


def main() -> None:
    """Fit the named arms on the named folds and seeds."""
    global WORK_DIR  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS), choices=list(MONTHS))
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=["R25", "R50", "Rcold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke", action="store_true", help="outputs under work/smoke; timing only"
    )
    args = parser.parse_args()
    if args.smoke:
        WORK_DIR = WORK_DIR / "smoke"
    run(
        months=args.months,
        arms=[ARMS[a] for a in args.arms],
        seeds=args.seeds,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
