# ruff: noqa: E501
"""Fit and forecast the input-port arms of experiment 419.

Every arm is experiment 407's `ctrl` network (one pooled network with an 8-wide Primary embedding,
3 Gaussian components, experiment 397's rows, summary scheme, setting `g09`) fitted on the five
screen folds with three seeds. D15 and D18 lose every training row before 2025-03-01 and every row
whose weekly lag reaches back before it; rows whose target or lag half-hour is a flagged exact zero
are dropped (both exactly as `e00407_fit.build_train`). Arms:

- `ctrl_s3`: `ctrl` again, seeds 3 to 5 (the reference `ctrl` of experiment 407 used seeds 0 to 2);
- `sol`: adds the solar summary (`shortwave_day_mean`, `shortwave_prev_day_mean`);
- `hl91r`: `ctrl` with a row-weight half-life of 0.25 years, seeds 3 to 5;
- `sol_hl91`: `sol` with a half-life of 0.25 years;
- `bundle`: `sol` plus the sunny flag and `sunny_times_shortwave`, half-life 1 year.

The half-life is 0.5 years unless stated. The added columns are appended to experiment 397's
feature matrix and standardised on each fold's training rows, with the other weather inputs. Each
fit writes the raw quantiles only; the scorer widens them by the fixed factor 1.15.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00419-*/e00419_fit.py --arms sol ctrl_s3 --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00407_fit as f407  # ty: ignore[unresolved-import]
import e00419_features as feats  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00419-pooled-mixture-density-network-solar-summary-sunny-flag-and-half-life-input-ports"
    / "work"
)
HALF_LIFE_QUARTER_YEAR_DAYS: Final[float] = 0.25 * 365.25
HALF_LIFE_ONE_YEAR_DAYS: Final[float] = 365.25
FIRST_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SECOND_SEEDS: Final[tuple[int, ...]] = (3, 4, 5)
SHORTWAVE_NAME: Final[str] = "mean_shortwave"
COS_ZENITH_NAME: Final[str] = "cos_zenith"


@dataclass(frozen=True)
class ArmSpec:
    """One arm: the added column groups, the row-weight half-life, and the default seeds."""

    name: str
    groups: tuple[str, ...] = ()
    half_life_days: float = common.SHORT_HALF_LIFE_DAYS
    seeds: tuple[int, ...] = FIRST_SEEDS

    def network(self) -> f407.Arm:
        """Return experiment 407's arm for this spec: 3 Gaussian components, one half-life."""
        return f407.Arm(self.name, half_life_days=self.half_life_days)


ARMS: Final[dict[str, ArmSpec]] = {
    a.name: a
    for a in (
        ArmSpec("ctrl_s3", seeds=SECOND_SEEDS),
        ArmSpec("sol", groups=("solar",)),
        ArmSpec("hl91r", half_life_days=HALF_LIFE_QUARTER_YEAR_DAYS, seeds=SECOND_SEEDS),
        ArmSpec("sol_hl91", groups=("solar",), half_life_days=HALF_LIFE_QUARTER_YEAR_DAYS),
        ArmSpec("bundle", groups=("solar", "sunny"), half_life_days=HALF_LIFE_ONE_YEAR_DAYS),
    )
}

_DAILY: dict[str, pl.DataFrame] = {}


def daily_of(*, site: str, series_id: int) -> pl.DataFrame:
    """Return a Primary's per-run daily shortwave means, from every step of every run's summary."""
    if site not in _DAILY:
        summary = prep.load_summary(series_id=series_id)
        _DAILY[site] = feats.daily_means(
            origin_us=summary["power_fcst_init_time"].dt.epoch("us").to_numpy(),
            valid_us=summary["valid_time"].dt.epoch("us").to_numpy(),
            shortwave=summary[SHORTWAVE_NAME].to_numpy(),
        )
    return _DAILY[site]


def extras_of(
    *,
    groups: tuple[str, ...],
    base: np.ndarray,
    origin_us: np.ndarray,
    valid_us: np.ndarray,
    daily: pl.DataFrame,
) -> np.ndarray:
    """Return the arm's added columns for rows with raw base inputs `base` (rows, n)."""
    names, _, _ = prep.feature_names(scheme=f407.SCHEME)
    return feats.extra_matrix(
        groups=groups,
        origin_us=origin_us,
        valid_us=valid_us,
        shortwave=base[:, names.index(SHORTWAVE_NAME)],
        cos_zenith=base[:, names.index(COS_ZENITH_NAME)],
        daily=daily,
    )


def build_train(*, refit: datetime, spec: ArmSpec) -> tuple[prep.FoldTrain, int]:
    """Return a fold's standardised training rows and the number of rows dropped as flagged.

    As `e00407_fit.build_train`, with the spec's added columns appended to the raw inputs before
    the standardiser is fitted on the fold's training rows.
    """
    flagged = f407.flagged_times()
    dropped = 0
    _, mask, layout = prep.feature_names(scheme=f407.SCHEME)
    mask = np.concatenate([mask, np.ones(len(feats.extra_column_names(groups=spec.groups)), bool)])
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / f407.SCHEME / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= f407._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= f407._us(t=refit))
            & (z["valid"] > f407._us(t=refit) - prep.WINDOW_DAYS * f407.US_PER_DAY)
        )
        if row["site"] in f407.SHORT_HISTORY:
            week_lag_hours = prep.lag_ages_hours(lead_hours=z["lead"])[:, 2]
            week_lag_us = (week_lag_hours * 3_600 * 1_000_000).astype(np.int64)
            keep &= (z["valid"] - week_lag_us) >= f407._us(t=f407.CUT)
        bad = flagged.get(int(row["time_series_id"]))
        if bad is not None:
            lag_us = (prep.lag_ages_hours(lead_hours=z["lead"]) * 3_600 * 1_000_000).astype(
                np.int64
            )
            hit = np.isin(z["valid"], bad)
            for j in range(lag_us.shape[1]):
                hit |= np.isin(z["valid"] - lag_us[:, j], bad)
            dropped += int((keep & hit).sum())
            keep &= ~hit
        if not keep.any():
            continue
        base = z["x"][keep]
        if spec.groups:
            extra = extras_of(
                groups=spec.groups,
                base=base,
                origin_us=z["origin"][keep],
                valid_us=z["valid"][keep],
                daily=daily_of(site=row["site"], series_id=int(row["time_series_id"])),
            )
            base = np.concatenate([base, extra], axis=1)
        parts.append(
            {
                "x": base,
                "y": z["y"][keep],
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "lead": z["lead"][keep],
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
            }
        )
    print(
        f"  rows dropped for a flagged exact-zero target or lag half-hour: {dropped:,}", flush=True
    )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - f407._us(t=prep.FIRST_RUN)) // f407.US_PER_DAY).astype(np.int64)
    held, purged = prep.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (f407._us(t=refit) - cat["valid"]) / f407.US_PER_DAY
    weight = 0.5 ** (age_days / spec.half_life_days)
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
    fold = prep.FoldTrain(
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
    return fold, dropped


def run_inputs(
    *, origin: datetime, standardiser: prep.Standardiser, groups: tuple[str, ...]
) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return each Primary's (series row, keys, standardised inputs) for one forecast run.

    As `e00397_prep.run_inputs` (summary scheme), with the added columns computed from this run's
    own summary rows and appended before standardising.
    """
    import e00087_fit_demand as fd  # ty: ignore[unresolved-import]

    _series, arrays = fd._series_and_arrays()
    run = origin - common.PUBLICATION_DELAY
    frame = pl.read_parquet(common.demand_frame_path(run=run))
    summary_all = pl.read_parquet(prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet")
    out = []
    for row in prep.primaries().iter_rows(named=True):
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[row["site"]],
            a7=False,
        )
        summary = summary_all.filter(pl.col("time_series_id") == row["time_series_id"])
        first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
        prepared = first.join(
            summary.drop("time_series_id"),
            on=["power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
            maintain_order="left",
        )
        keys = prepared.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        x = prep.feature_matrix(frame=prepared, scheme=f407.SCHEME)
        if groups:
            origin_us = prepared["power_fcst_init_time"].dt.epoch("us").to_numpy()
            valid_us = prepared["valid_time"].dt.epoch("us").to_numpy()
            daily = feats.daily_means(
                origin_us=origin_us,
                valid_us=valid_us,
                shortwave=prepared[SHORTWAVE_NAME].to_numpy(),
            )
            extra = extras_of(
                groups=groups, base=x, origin_us=origin_us, valid_us=valid_us, daily=daily
            )
            x = np.concatenate([x, extra], axis=1)
        out.append((row, keys, standardiser.apply(x=x)))
    return out


def forecast_month(
    *, model: torch.nn.Module, spec: ArmSpec, fold: prep.FoldTrain, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity)."""
    arm = spec.network()
    raw = []
    for origin in prep.test_origins(month=month):
        for row, keys, x in run_inputs(
            origin=origin, standardiser=fold.standardiser, groups=spec.groups
        ):
            p = int(row["primary"])
            q = f407.predict_rows(
                model=model, arm=arm, x=x, primary=np.full(len(x), p, dtype=np.int64), device=device
            )
            q = q * fold.target_std[p] + fold.target_mean[p]
            raw.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(
                        np.sort(q, axis=1).astype(np.float32),
                        schema=list(common.LEVEL_COLUMNS),
                        orient="row",
                    )
                )
            )
    return pl.concat(raw)


def record_dropped(*, month: str, dropped: int) -> None:
    """Write a fold's flagged-row count to `dropped_rows.json` (the scorer prints it)."""
    path = OUT_DIR / "dropped_rows.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    table[month] = dropped
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1))


def run(*, arms: list[str], seeds: list[int] | None, months: list[str], device: str) -> None:
    """Fit each arm on each fold and seed, save the networks, and forecast the month."""
    ordered = sorted(arms, key=lambda n: (ARMS[n].half_life_days, ARMS[n].groups))
    for month in months:
        fold: prep.FoldTrain | None = None
        fold_key: tuple[float, tuple[str, ...]] | None = None
        for name in ordered:
            spec = ARMS[name]
            wanted = seeds if seeds is not None else list(spec.seeds)
            unregistered = sorted(set(wanted) - set(spec.seeds))
            if unregistered:
                msg = f"arm {name} is registered for seeds {list(spec.seeds)}, not {unregistered}"
                raise ValueError(msg)
            todo = [
                s
                for s in wanted
                if not (OUT_DIR / "seed_preds" / name / f"{month}_s{s}.parquet").exists()
            ]
            if not todo:
                continue
            key = (spec.half_life_days, spec.groups)
            if fold is None or fold_key != key:
                fold = None
                fold, dropped = build_train(refit=prep.refit_of(month=month), spec=spec)
                record_dropped(month=month, dropped=dropped)
                fold_key = key
                print(
                    f"fold {month} {name}: {len(fold.x):,} rows, {fold.x.shape[1]} inputs",
                    flush=True,
                )
            for seed in todo:
                started = time.monotonic()
                model, epochs, val = f407.fit_one(
                    arm=spec.network(), fold=fold, seed=seed, device=device
                )
                folder = OUT_DIR / "models" / name / month
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), folder / f"s{seed}.pt")
                (folder / f"s{seed}.json").write_text(
                    json.dumps(
                        {
                            "arm": name,
                            "epochs": epochs,
                            "heldout_loss": val,
                            "half_life_days": spec.half_life_days,
                            "groups": list(spec.groups),
                            "extra_columns": feats.extra_column_names(groups=spec.groups),
                            "standardiser_mean": fold.standardiser.mean.tolist(),
                            "standardiser_std": fold.standardiser.std.tolist(),
                            "target_mean": fold.target_mean.tolist(),
                            "target_std": fold.target_std.tolist(),
                        }
                    )
                )
                raw = forecast_month(model=model, spec=spec, fold=fold, month=month, device=device)
                common.atomic_write(
                    frame=raw, target=OUT_DIR / "seed_preds" / name / f"{month}_s{seed}.parquet"
                )
                print(f"{name} {month} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run the named arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="default: each arm's own seeds"
    )
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(arms=args.arms, seeds=args.seeds, months=args.months, device=args.device)


if __name__ == "__main__":
    main()
