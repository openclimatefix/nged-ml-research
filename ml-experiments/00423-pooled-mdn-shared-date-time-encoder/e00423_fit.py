# ruff: noqa: E501
"""Fit and forecast the calendar-encoder arms of experiment 423.

Every arm is experiment 407's `ctrl` network (one pooled network with an 8-wide Primary embedding,
3 Gaussian components, experiment 397's rows, summary scheme, setting `g09`, row-weight half-life
0.5 years) fitted on the screen folds. D15 and D18 lose every training row before 2025-03-01 and
every row whose weekly lag reaches back before it; rows whose target or lag half-hour is a flagged
exact zero are dropped (both exactly as `e00419_fit.build_train`). The 18 hand-made calendar
columns of experiment 397 (its `calendar` and `holidays` groups) are replaced by a calendar encoder
that reads integer calendar indices (`e00423_features`); `covid` always stays. Arms:

- `R0`: no encoder, all 18 columns, through this code path (the check that the stored reference of
  experiments 407 and 419 is reproduced); seed 0, folds 2025-12 and 2026-04;
- `E0`: half-hour (8-wide) and weekday (4-wide) only; seeds 0 to 2;
- `E1n`: E0, plus `bank_holiday`, `nearest_days`, and `nearest_inside` as direct inputs; seeds 0 to 5;
- `E1`: E1n, plus day of month, day of year, and the holiday name (4-wide each); seeds 0 to 5;
- `E2`: E0, plus day of month and day of year, no holiday information; seeds 0 to 5;
- `E3`: one E1 encoder per Primary; seeds 0 to 5; run only when asked for by name.

Each fit writes the raw quantiles only; the scorer widens them by the fixed factor 1.15. The
month 2026-04 (Good Friday and Easter Monday) is an optional fold: name it in `--months`. The
script checks first that its raw inputs exist and skips the month, with a message, if they do not.
`--count` prints the parameter counts and exits.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00423-pooled-mdn-shared-date-time-encoder \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00423-*/e00423_fit.py --arms E1 E1n E2 --device cuda
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
import e00423_features as feats  # ty: ignore[unresolved-import]
import e00423_models as models  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00423-pooled-mdn-shared-date-time-encoder" / "work"
)
SEEDS_0_TO_2: Final[tuple[int, ...]] = (0, 1, 2)
SEEDS_0_TO_5: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
EVENT_MONTH: Final[str] = "2026-04"
"""The optional fold that holds Good Friday (2026-04-03) and Easter Monday (2026-04-06)."""
R0_MONTHS: Final[tuple[str, ...]] = ("2025-12", EVENT_MONTH)


@dataclass(frozen=True)
class ArmSpec:
    """One arm: its encoder (None for R0), its default seeds, and the folds it is restricted to."""

    name: str
    encoder: str | None
    seeds: tuple[int, ...]
    months: tuple[str, ...] | None = None
    on_request: bool = False

    @property
    def encoder_spec(self) -> models.EncoderSpec | None:
        """Return the arm's encoder specification, or None for R0."""
        return None if self.encoder is None else models.ENCODERS[self.encoder]

    @property
    def fold_key(self) -> tuple[bool, tuple[str, ...]]:
        """Return what the fold's inputs depend on: the kept holiday columns and the indices."""
        enc = self.encoder_spec
        return (True, ()) if enc is None else (enc.keep_holiday_inputs, enc.index_columns)

    def network(self) -> f407.Arm:
        """Return experiment 407's arm for this spec: 3 Gaussian components, one half-life."""
        return f407.Arm(self.name)


ARMS: Final[dict[str, ArmSpec]] = {
    a.name: a
    for a in (
        ArmSpec("R0", encoder=None, seeds=(0,), months=R0_MONTHS),
        ArmSpec("E0", encoder="E0", seeds=SEEDS_0_TO_2),
        ArmSpec("E1n", encoder="E1n", seeds=SEEDS_0_TO_5),
        ArmSpec("E1", encoder="E1", seeds=SEEDS_0_TO_5),
        ArmSpec("E2", encoder="E2", seeds=SEEDS_0_TO_5),
        ArmSpec("E3", encoder="E3", seeds=SEEDS_0_TO_5, on_request=True),
    )
}


def input_columns(*, spec: ArmSpec) -> tuple[list[int], list[str]]:
    """Return the matrix columns an arm keeps and the names of the 18-column group it replaces.

    Raises:
        ValueError: If the layout's `calendar` and `holidays` groups are not the 18 hand-made
            columns of `e00423_features.HAND_MADE_COLUMNS`.
    """
    names, _, layout = prep.feature_names(scheme=f407.SCHEME)
    group = [names[i] for g in ("calendar", "holidays") for i in layout.columns_of(groups=[g])]
    if sorted(group) != sorted(feats.HAND_MADE_COLUMNS) or len(group) != 18:
        msg = f"the calendar and holiday groups are not the 18 hand-made columns: {group}"
        raise ValueError(msg)
    enc = spec.encoder_spec
    if enc is None:
        return list(range(len(names))), []
    return feats.kept_columns(names=names, keep_holiday_inputs=enc.keep_holiday_inputs), group


def index_matrix(*, spec: ArmSpec, valid_us: np.ndarray) -> np.ndarray:
    """Return the arm's calendar-index columns (rows, n) as float32, or (rows, 0) for R0."""
    enc = spec.encoder_spec
    if enc is None:
        return np.zeros((len(valid_us), 0), np.float32)
    all_indices = feats.calendar_indices(valid_us=valid_us)
    columns = [feats.INDEX_COLUMNS.index(c) for c in enc.index_columns]
    return all_indices[:, columns].astype(np.float32)


def parameter_counts() -> None:
    """Print the parameter counts of every encoder and of every arm's network."""
    names, _, _ = prep.feature_names(scheme=f407.SCHEME)
    print(f"input matrix: {len(names)} columns; hand-made calendar columns: 18")
    for spec in ARMS.values():
        columns, _ = input_columns(spec=spec)
        enc = spec.encoder_spec
        if enc is None:
            model = f407.fam.PooledNetwork(
                n_inputs=len(columns),
                family="gauss",
                hidden=f407.SETTING.hidden,
                layers=f407.SETTING.layers,
                dropout=f407.SETTING.dropout,
                n_primaries=len(prep.primaries()),
            )
            print(
                f"{spec.name}: {len(columns)} dense columns, network {_total(model):,} parameters"
            )
            continue
        net = models.EncodedPooledNetwork(
            n_dense=len(columns),
            spec=enc,
            hidden=f407.SETTING.hidden,
            layers=f407.SETTING.layers,
            dropout=f407.SETTING.dropout,
            n_primaries=len(prep.primaries()),
        )
        print(
            f"{spec.name}: {len(columns)} dense columns, indices {list(enc.index_columns)}, "
            f"{net.encoder_parameter_counts()}, network {_total(net):,} parameters"
        )


def _total(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def check_month_inputs(*, month: str) -> list[str]:
    """Return what is missing for a fold's raw training inputs and test-origin inputs.

    Checks that every Primary's raw matrix covers the training window up to the refit, and that
    every test origin of the month has its summary table and its stored demand frame.
    """
    missing: list[str] = []
    refit = prep.refit_of(month=month)
    last_needed = f407._us(t=refit - regimes.REFIT_CUTOFF_AGE) - f407.US_PER_DAY
    for row in prep.primaries().iter_rows(named=True):
        path = prep.RAW_DIR / f407.SCHEME / f"{row['site']}.npz"
        if not path.exists():
            missing.append(f"raw inputs {path}")
        elif int(np.load(path)["valid"].max()) < last_needed:
            missing.append(f"raw inputs {path} end before the refit")
    origins = prep.test_origins(month=month)
    if not origins:
        missing.append(f"no test origins in {month}")
    for origin in origins:
        run = origin - common.PUBLICATION_DELAY
        paths = (
            prep.SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet",
            common.demand_frame_path(run=run),
        )
        missing.extend(f"test-origin input {p}" for p in paths if not p.exists())
    return missing


def build_train(*, refit: datetime, spec: ArmSpec) -> tuple[prep.FoldTrain, int]:
    """Return a fold's training rows and the number of rows dropped as flagged exact zeros.

    As `e00419_fit.build_train` without added columns: the standardiser is fitted on all of
    experiment 397's columns (so every kept column is standardised exactly as in the reference),
    then the arm's columns are kept and its calendar indices, unstandardised, are appended.
    """
    flagged = f407.flagged_times()
    dropped = 0
    _, mask, layout = prep.feature_names(scheme=f407.SCHEME)
    columns, _ = input_columns(spec=spec)
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
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(prep.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    x_dense = standardiser.apply(x=cat["x"][keep])[:, columns]
    x_kept = np.concatenate([x_dense, index_matrix(spec=spec, valid_us=cat["valid"][keep])], axis=1)
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


def fit_one(
    *, spec: ArmSpec, fold: prep.FoldTrain, seed: int, device: str
) -> tuple[torch.nn.Module, int, float]:
    """Fit an arm's network on a fold and seed; return it with its epochs and held-out loss."""
    enc = spec.encoder_spec
    if enc is None:
        return f407.fit_one(arm=spec.network(), fold=fold, seed=seed, device=device)
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    torch.manual_seed(seed)
    model = models.EncodedPooledNetwork(
        n_dense=fold.x.shape[1] - len(enc.index_columns),
        spec=enc,
        hidden=f407.SETTING.hidden,
        layers=f407.SETTING.layers,
        dropout=f407.SETTING.dropout,
        n_primaries=len(fold.target_mean),
    ).to(device)
    return f407.train_network(
        model=model,  # ty: ignore[invalid-argument-type]
        arm=spec.network(),
        x=torch.from_numpy(fold.x).to(device),
        z=torch.from_numpy(z.astype(np.float32)).to(device),
        w=torch.from_numpy(fold.weight).to(device),
        primary=torch.from_numpy(fold.primary).to(device),
        is_val=torch.from_numpy(fold.is_val).to(device),
        seed=seed,
    )


def forecast_month(
    *, model: torch.nn.Module, spec: ArmSpec, fold: prep.FoldTrain, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity)."""
    arm = spec.network()
    columns, _ = input_columns(spec=spec)
    raw = []
    for origin in prep.test_origins(month=month):
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme=f407.SCHEME, standardiser=fold.standardiser
        ):
            if len(x) != len(keys):
                msg = f"{row['site']} {origin}: {len(x)} input rows for {len(keys)} keys"
                raise ValueError(msg)
            indices = index_matrix(spec=spec, valid_us=keys["valid_time"].dt.epoch("us").to_numpy())
            inputs = np.concatenate([x[:, columns], indices], axis=1)
            p = int(row["primary"])
            q = f407.predict_rows(
                model=model,
                arm=arm,
                x=inputs,
                primary=np.full(len(inputs), p, dtype=np.int64),
                device=device,
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


def usable_months(*, months: list[str]) -> list[str]:
    """Return the months whose raw inputs exist, printing what is missing for the others."""
    usable = []
    for month in months:
        missing = check_month_inputs(month=month)
        if missing:
            shown = "; ".join(missing[:5])
            print(
                f"fold {month} SKIPPED: {len(missing)} raw inputs missing, e.g. {shown}", flush=True
            )
        else:
            print(f"fold {month}: raw inputs present", flush=True)
            usable.append(month)
    return usable


def run(*, arms: list[str], seeds: list[int] | None, months: list[str], device: str) -> None:
    """Fit each arm on each fold and seed, save the networks, and forecast the month."""
    ordered = sorted(arms, key=lambda n: (ARMS[n].fold_key[0], ARMS[n].fold_key[1]))
    for month in usable_months(months=months):
        fold: prep.FoldTrain | None = None
        fold_key: tuple[bool, tuple[str, ...]] | None = None
        for name in ordered:
            spec = ARMS[name]
            if spec.months is not None and month not in spec.months:
                continue
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
            if fold is None or fold_key != spec.fold_key:
                fold = None
                fold, dropped = build_train(refit=prep.refit_of(month=month), spec=spec)
                record_dropped(month=month, dropped=dropped)
                fold_key = spec.fold_key
                print(
                    f"fold {month} {name}: {len(fold.x):,} rows, {fold.x.shape[1]} inputs",
                    flush=True,
                )
            for seed in todo:
                started = time.monotonic()
                model, epochs, val = fit_one(spec=spec, fold=fold, seed=seed, device=device)
                columns, _ = input_columns(spec=spec)
                names, _, _ = prep.feature_names(scheme=f407.SCHEME)
                folder = OUT_DIR / "models" / name / month
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), folder / f"s{seed}.pt")
                (folder / f"s{seed}.json").write_text(
                    json.dumps(
                        {
                            "arm": name,
                            "epochs": epochs,
                            "heldout_loss": val,
                            "half_life_days": common.SHORT_HALF_LIFE_DAYS,
                            "dense_columns": [names[i] for i in columns],
                            "index_columns": []
                            if spec.encoder_spec is None
                            else list(spec.encoder_spec.index_columns),
                            "parameters": _total(model),
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
    default_arms = [n for n, a in ARMS.items() if not a.on_request]
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=default_arms)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="default: each arm's own seeds"
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(prep.SCREEN_MONTHS),
        help=f"default: the five screen folds; add {EVENT_MONTH} for the Easter fold",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", action="store_true", help="print parameter counts and exit")
    args = parser.parse_args()
    parameter_counts()
    if args.count:
        return
    run(arms=args.arms, seeds=args.seeds, months=args.months, device=args.device)


if __name__ == "__main__":
    main()
