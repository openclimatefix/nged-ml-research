"""Fit and forecast the pooling controls of experiment 406: no embedding, and short histories.

Every arm is experiment 397's plain network (setting `g09`, summary scheme) on its five screen folds
with three seeds. Three changes separate this experiment from experiment 397:

- D15 and D18 lose every training row before 2025-03-01, and every row whose oldest lag reaches
  back before that date (the maintainer's rule of 2026-10-30, set by experiment 399);
- `--history-months` keeps only each Primary's last `H` months of training rows before the refit;
- the arm `pooled_noemb` is one pooled plain network with no Primary embedding.

Arms: `per_primary` (one plain network per Primary), `pooled_emb` (one pooled plain network with an
8-wide Primary embedding), `pooled_noemb`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00406-*/e00406_fit.py --arms pooled_noemb pooled_emb \
    per_primary --history-months 0 --device cuda
```
"""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_fit as fit  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SETTING: Final[fit.Setting] = fit.setting_by_name(name="g09")
SCHEME: Final[str] = "summary"
ARMS: Final[tuple[str, ...]] = ("per_primary", "pooled_emb", "pooled_noemb")
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
DAYS_PER_MONTH: Final[float] = 30.44
US_PER_DAY: Final[int] = 86_400 * 1_000_000
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00406-pooled-mixture-density-network-without-embedding-and-with-short-history"
    / "work"
)


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def build_train(*, refit: datetime, history_months: int) -> prep.FoldTrain:
    """Return a fold's standardised training rows, as experiment 397's `build_train` but cut.

    `history_months` of 0 keeps the whole 24-month window. The start cut drops a D15 or D18 row if
    its valid time, or the time of its oldest lag (the weekly lag), is before 2025-03-01.
    """
    _, mask, layout = prep.feature_names(scheme=SCHEME)
    window_days = prep.WINDOW_DAYS if history_months == 0 else history_months * DAYS_PER_MONTH
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / SCHEME / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit) - int(window_days * US_PER_DAY))
        )
        if row["site"] in SHORT_HISTORY:
            week_lag_hours = prep.lag_ages_hours(lead_hours=z["lead"])[:, 2]
            week_lag_us = (week_lag_hours * 3_600 * 1_000_000).astype(np.int64)
            keep &= (z["valid"] - week_lag_us) >= _us(t=CUT)
        if not keep.any():
            continue
        parts.append(
            {
                "x": z["x"][keep],
                "y": z["y"][keep],
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "lead": z["lead"][keep],
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=prep.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = prep.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(prep.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    x_kept = standardiser.apply(x=cat["x"][keep])
    del cat["x"]
    return prep.FoldTrain(
        x=x_kept,
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin_day=origin_day[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
    )


def fit_pooled_without_embedding(
    *, fold: prep.FoldTrain, seed: int, device: str
) -> dict[int | None, fit.Fitted]:
    """Fit one plain network on every Primary's rows, with no Primary input of any kind."""
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    model = fit.make_model(
        arm="plain_own",
        setting=SETTING,
        layout=fold.layout,
        n_primaries=len(fold.target_mean),
        n_inputs=fold.x.shape[1],
    ).to(device)
    model, epochs, val = fit.train_network(
        model=model,
        setting=SETTING,
        x=torch.from_numpy(fold.x).to(device),
        z=torch.from_numpy(z.astype(np.float32)).to(device),
        w=torch.from_numpy(fold.weight).to(device),
        primary=torch.from_numpy(fold.primary).to(device),
        is_val=torch.from_numpy(fold.is_val).to(device),
        seed=seed,
    )
    return {
        None: fit.Fitted(
            model=model, setting=SETTING, arm="pooled_noemb", seed=seed, epochs=epochs, val_nll=val
        )
    }


def tag_of(*, arm: str, history_months: int) -> str:
    """Return the folder name of an arm at a history length (0 is the full window)."""
    return arm if history_months == 0 else f"{arm}_h{history_months}"


def run(
    *, arms: list[str], history_months: int, seeds: list[int], months: list[str], device: str
) -> None:
    """Fit each arm on each fold and seed, save the networks, forecast the test month."""
    for month in months:
        fold = build_train(refit=prep.refit_of(month=month), history_months=history_months)
        counts = {int(p): int((fold.primary == p).sum()) for p in np.unique(fold.primary)}
        print(
            f"fold {month} history {history_months}: {len(fold.x):,} rows; per Primary {counts}",
            flush=True,
        )
        for arm in arms:
            tag = tag_of(arm=arm, history_months=history_months)
            for seed in seeds:
                target = OUT_DIR / "seed_preds" / tag / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                started = time.monotonic()
                if arm == "pooled_noemb":
                    fits = fit_pooled_without_embedding(fold=fold, seed=seed, device=device)
                else:
                    fits = fit.fit_arm(
                        arm="plain_own" if arm == "per_primary" else "plain_pooled",
                        setting=SETTING,
                        fold=fold,
                        seed=seed,
                        device=device,
                    )
                if arm == "per_primary" and len(fits) != len(counts):
                    msg = (
                        f"{tag} {month}: networks for {sorted(fits)} but rows for {sorted(counts)}"
                    )
                    raise RuntimeError(msg)
                fit.save_fits(
                    fits=fits, fold=fold, target=OUT_DIR / "models" / tag / month / f"s{seed}"
                )
                q, _ = fit.forecast_month(
                    fits=fits, fold=fold, scheme=SCHEME, month=month, device=device
                )
                common.atomic_write(frame=q, target=target)
                print(f"{tag} {month} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run the named arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--history-months", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        arms=args.arms,
        history_months=args.history_months,
        seeds=args.seeds,
        months=args.months,
        device=args.device,
    )


if __name__ == "__main__":
    main()
