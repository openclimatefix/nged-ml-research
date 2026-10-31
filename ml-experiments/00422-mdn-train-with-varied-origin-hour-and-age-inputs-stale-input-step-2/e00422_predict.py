"""Forecast the fitted networks of experiment 422 under the stale-input arms of experiment 410.

Reads the networks that `e00422_fit.py` saved (every training arm, fold, and seed) and forecasts
every test origin under experiment 410's arms (FRESH, W24, W48, P6, P12, P24, DROP, W24_P12,
W24_P24) and three more that put the newest reading at the same hour of day as P6, P12, and P24 but
a day further back (P30, P36, P48), and P10, which lands off the 3-hour training grid. It reuses
experiment 410's input builder, so the lags, the lag ages, and the cleaned and start-cut power are
the same. Networks trained with the age inputs receive the age of the weather forecast (9 hours,
plus 24 per day of staleness) and the age of the newest reading (3.5 hours plus the delay).

Writes `preds/<training arm>/<test arm>/<month>_s<seed>.parquet`, the layout that experiment 410's
scorer reads.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00410-mdn-stale-weather-and-delayed-power-rescore:\
ml-experiments/00422-mdn-train-with-varied-origin-hour-and-age-inputs-stale-input-step-2 \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00422-*/e00422_predict.py --device cuda
```
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00410_stale_predict as stale  # ty: ignore[unresolved-import]
import e00422_fit as fit422  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

Arm = stale.Arm
TEST_ARMS: Final[tuple[Arm, ...]] = (
    Arm("FRESH"),
    Arm("W24", weather_days=1),
    Arm("W48", weather_days=2),
    Arm("P6", power_hours=6),
    Arm("P10", power_hours=10),
    Arm("P12", power_hours=12),
    Arm("P24", power_hours=24),
    Arm("P30", power_hours=30),
    Arm("P36", power_hours=36),
    Arm("P48", power_hours=48),
    Arm("DROP", drop_lags=True),
    Arm("W24_P12", weather_days=1, power_hours=12),
    Arm("W24_P24", weather_days=1, power_hours=24),
)
OUT_DIR: Final[Path] = fit422.WORK_DIR


def load_model(
    *, arm_name: str, month: str, seed: int, device: str
) -> tuple[fit397.Fitted, prep.Standardiser, np.ndarray, np.ndarray, bool]:
    """Return a saved network, its standardiser, target mean and std, and whether it has ages."""
    folder = fit422.MODELS_DIR / arm_name / month / f"s{seed}"
    meta = json.loads((folder / "meta.json").read_text())
    setting = fit397.Setting(**meta["setting"])
    arm = prep422.arm_by_name(name=arm_name)
    names, _, layout = prep422.feature_names(age_inputs=arm.age_inputs)
    model = fit397.make_model(
        arm="plain_pooled",
        setting=setting,
        layout=layout,
        n_primaries=len(meta["target_mean"]),
        n_inputs=len(names),
    )
    model.load_state_dict(torch.load(folder / "pooled.pt", map_location=device))
    model.to(device).eval()
    fitted = fit397.Fitted(
        model=model,
        setting=setting,
        arm="plain_pooled",
        seed=seed,
        epochs=meta["epochs"]["None"],
        val_nll=meta["val_nll"]["None"],
    )
    with np.load(folder / "standardiser.npz") as stored:
        standardiser = prep.Standardiser(
            mean=stored["mean"].copy(), std=stored["std"].copy(), mask=stored["mask"].copy()
        )
    return (
        fitted,
        standardiser,
        np.array(meta["target_mean"]),
        np.array(meta["target_std"]),
        arm.age_inputs,
    )


def with_ages(*, raw: np.ndarray, arm: Arm) -> np.ndarray:
    """Append the weather age and the newest-reading age (each over 48 hours) to the raw inputs."""
    weather = (prep422.PUBLICATION_HOURS + 24.0 * arm.weather_days) / prep422.AGE_SCALE_HOURS
    power = (prep422.NEWEST_AGE_HOURS + arm.power_hours) / prep422.AGE_SCALE_HOURS
    ages = np.tile(np.array([weather, power], dtype=np.float32), (len(raw), 1))
    return np.concatenate([raw, ages], axis=1)


def run_month(
    *,
    month: str,
    train_arms: list[str],
    test_arms: list[Arm],
    seeds: list[int],
    device: str,
    smoke: bool,
) -> None:
    """Forecast one test month for every training arm, seed, and test arm."""
    series, arrays = fd._series_and_arrays()
    del series
    clean = prep422.cleaned_arrays()
    primaries = prep.primaries()
    loaded = {
        (a, s): load_model(arm_name=a, month=month, seed=s, device=device)
        for a in train_arms
        for s in seeds
    }
    pieces: dict[tuple[str, str, int], list[pl.DataFrame]] = {}
    origins = prep.test_origins(month=month)
    for origin in origins[:2] if smoke else origins:
        run = origin - common.PUBLICATION_DELAY
        frame = pl.read_parquet(common.demand_frame_path(run=run))
        summary_all = pl.read_parquet(prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet")
        for row in primaries.iter_rows(named=True):
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[row["site"]],
                a7=False,
            )
            prepared = (
                prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
                .join(
                    summary_all.filter(pl.col("time_series_id") == row["time_series_id"]).drop(
                        "time_series_id"
                    ),
                    on=["power_fcst_init_time", "valid_time"],
                    how="inner",
                    validate="1:1",
                    maintain_order="left",
                )
                .sort("valid_time")
            )
            p = int(row["primary"])
            for arm in test_arms:
                made = stale.arm_inputs(
                    arm=arm, prepared=prepared, run=run, row=row, power=clean[row["site"]]
                )
                if made is None:
                    continue
                keys, raw = made
                raw_aged = with_ages(raw=raw, arm=arm)
                for (train_arm, seed), (
                    fitted,
                    standardiser,
                    t_mean,
                    t_std,
                    aged,
                ) in loaded.items():
                    x = standardiser.apply(x=raw_aged if aged else raw)
                    q, _, _ = fit397.predict_z_quantiles(
                        fitted=fitted, x=x, primary=p, members=1, device=device
                    )
                    q = np.sort(q * t_std[p] + t_mean[p], axis=1).astype(np.float32)
                    pieces.setdefault((train_arm, arm.name, seed), []).append(
                        keys.drop("lead_hours").hstack(
                            pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                        )
                    )
    for (train_arm, test_arm, seed), parts in pieces.items():
        target = (
            OUT_DIR
            / "preds"
            / train_arm
            / test_arm
            / f"{month}_s{seed}{'_smoke' if smoke else ''}.parquet"
        )
        common.atomic_write(frame=pl.concat(parts), target=target)


def main() -> None:
    """Forecast the requested months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--train-arms", nargs="+", default=[a.name for a in prep422.TRAIN_ARMS])
    parser.add_argument("--test-arms", nargs="+", default=[a.name for a in TEST_ARMS])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit422.SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="two origins, one month")
    args = parser.parse_args()
    test_arms = [a for a in TEST_ARMS if a.name in args.test_arms]
    for month in args.months[:1] if args.smoke else args.months:
        started = time.monotonic()
        run_month(
            month=month,
            train_arms=args.train_arms,
            test_arms=test_arms,
            seeds=args.seeds,
            device=args.device,
            smoke=args.smoke,
        )
        print(f"{month}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
