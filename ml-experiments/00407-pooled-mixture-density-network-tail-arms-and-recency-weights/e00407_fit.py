"""Fit and forecast the tail arms and recency-weight arms of experiment 407.

Every arm is one pooled network with an 8-wide Primary embedding on experiment 397's rows (summary
scheme, setting `g09`), fitted on the five screen folds with three seeds. D15 and D18 lose every
training row before 2025-03-01, and every row whose weekly lag reaches back before it. Arms:

- `ctrl`: the pooled plain network of experiment 397 (3 Gaussian components, half-life 0.5 years);
- `k5`: 5 Gaussian components;
- `skew`: 3 split-normal components (each skewed);
- `qhead`: 11 monotone quantiles trained with the pinball loss;
- `hl365` and `hl91`: `ctrl` with a row-weight half-life of 1 year and of 0.25 years;
- `a0`: experiment 397's group-attention network with uniform weights (arm A0), to redo experiment
  403's contrast with the cut.

Each fit writes two forecasts: the raw quantiles, and the quantiles widened per lead band. The
widening is fitted inside the refit, on the rows the network did not train on (one week in every
12, which also stops the training): for each lead band and each outer level, the factor that makes
the share of those rows below the widened quantile equal the level.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00407-*/e00407_fit.py --arms ctrl k5 --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_fit as fit  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00407_models as fam  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SETTING: Final[fit.Setting] = fit.setting_by_name(name="g09")
SCHEME: Final[str] = "summary"
SHORT_HISTORY: Final[tuple[str, ...]] = ("D15", "D18")
CUT: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
US_PER_DAY: Final[int] = 86_400 * 1_000_000
OUTER: Final[tuple[tuple[int, float], ...]] = ((0, 0.01), (1, 0.05), (9, 0.95), (10, 0.99))
"""Index in the 11 levels and nominal level of the four widened quantiles."""
BAND_EDGES_HOURS: Final[tuple[float, ...]] = (0.0, 24.0, 72.0, 168.0, 240.0, 337.0)
"""Lead bands of the widening: day 1, days 2 to 3, days 4 to 7, days 8 to 10, days 11 to 14."""
FACTOR_RANGE: Final[tuple[float, float]] = (0.5, 4.0)
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
"""The half-hours that the cleaned power table flags as `substation_zero` (experiment 398)."""
OUT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00407-pooled-mixture-density-network-tail-arms-and-recency-weights"
    / "work"
)


@dataclass(frozen=True)
class Arm:
    """One arm: an output family, a component count, and a row-weight half-life."""

    name: str
    family: str = "gauss"
    components: int = 3
    half_life_days: float = common.SHORT_HALF_LIFE_DAYS
    attention: bool = False


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("ctrl"),
        Arm("k5", components=5),
        Arm("skew", family="split"),
        Arm("qhead", family="quantile"),
        Arm("hl365", half_life_days=365.0),
        Arm("hl91", half_life_days=91.0),
        Arm("a0", attention=True),
    )
}


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def flagged_times() -> dict[int, np.ndarray]:
    """Return each series' flagged exact-zero half-hours, as microseconds since the epoch."""
    flagged = pl.read_parquet(FLAGGED_KEYS)
    return {
        int(i): np.sort(g["valid_time"].dt.epoch("us").to_numpy())
        for (i,), g in flagged.group_by("time_series_id")
    }


def build_train(*, refit: datetime, half_life_days: float) -> prep.FoldTrain:
    """Return a fold's standardised training rows, as experiment 397's `build_train` but cut.

    The start cut drops a D15 or D18 row if its valid time, or the time of its oldest lag (the
    weekly lag), is before 2025-03-01. It also drops a row if its own half-hour or the half-hour of
    any of its three lags is a flagged exact zero. Row weights halve every `half_life_days`.
    """
    flagged = flagged_times()
    dropped = 0
    _, mask, layout = prep.feature_names(scheme=SCHEME)
    parts = []
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / SCHEME / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit) - prep.WINDOW_DAYS * US_PER_DAY)
        )
        if row["site"] in SHORT_HISTORY:
            week_lag_hours = prep.lag_ages_hours(lead_hours=z["lead"])[:, 2]
            week_lag_us = (week_lag_hours * 3_600 * 1_000_000).astype(np.int64)
            keep &= (z["valid"] - week_lag_us) >= _us(t=CUT)
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
    origin_day = ((cat["origin"] - _us(t=prep.FIRST_RUN)) // US_PER_DAY).astype(np.int64)
    held, purged = prep.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = prep.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / US_PER_DAY
    weight = 0.5 ** (age_days / half_life_days)
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


def train_network(
    *,
    model: fam.PooledNetwork,
    arm: Arm,
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    primary: torch.Tensor,
    is_val: torch.Tensor,
    seed: int,
) -> tuple[fam.PooledNetwork, int, float]:
    """Train with the arm's weighted loss and early stopping; return the best network."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=SETTING.lr, weight_decay=SETTING.weight_decay
    )

    def row_loss(b: torch.Tensor) -> torch.Tensor:
        return fam.loss(
            family=arm.family,
            components=arm.components,
            out=model(x[b], primary[b]),
            z=z[b],
        )

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(fit.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=x.device)]
        for start in range(0, len(order), fit.BATCH):
            b = order[start : start + fit.BATCH]
            loss = (w[b] * row_loss(b)).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fit.GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([row_loss(b) for b in val_idx.split(fit.BATCH * 4)]).mean())
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= fit.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, epochs, best


def predict_z(
    *, model: fam.PooledNetwork, arm: Arm, x: np.ndarray, primary: np.ndarray, device: str
) -> np.ndarray:
    """Return the 11 standardised quantiles (rows, 11) of rows of inputs."""
    out = []
    with torch.no_grad():
        for start in range(0, len(x), fit.SCORE_CHUNK_ROWS * 8):
            xb = torch.from_numpy(x[start : start + fit.SCORE_CHUNK_ROWS * 8]).to(device)
            pb = torch.from_numpy(primary[start : start + fit.SCORE_CHUNK_ROWS * 8]).to(device)
            q = fam.quantiles(family=arm.family, components=arm.components, out=model(xb, pb))
            out.append(q.cpu().numpy())
    return np.sort(np.concatenate(out), axis=1)


def band_factors(*, q: np.ndarray, z: np.ndarray, lead_hours: np.ndarray) -> np.ndarray:
    """Return the widening factors, shape (bands, 4), fitted on held-out rows.

    For the upper level `a` (0.95 or 0.99) the factor of a band is the `a`-quantile of
    `(z - q50) / (q_a - q50)` over the band's rows, which makes the share of rows below the widened
    quantile equal `a`; the lower levels use the mirrored ratio and the level `1 - a`.
    """
    factors = np.ones((len(BAND_EDGES_HOURS) - 1, len(OUTER)))
    band = np.digitize(lead_hours, BAND_EDGES_HOURS[1:-1])
    median = q[:, fam.MEDIAN_INDEX]
    for b in range(factors.shape[0]):
        rows = band == b
        if not rows.any():
            continue
        for j, (index, level) in enumerate(OUTER):
            upper = level > 0.5
            gap = (q[rows, index] - median[rows]) * (1.0 if upper else -1.0)
            move = (z[rows] - median[rows]) * (1.0 if upper else -1.0)
            usable = gap > 1e-6
            if usable.sum() < 100:
                continue
            target = level if upper else 1.0 - level
            factors[b, j] = np.clip(np.quantile(move[usable] / gap[usable], target), *FACTOR_RANGE)
    return factors


def widen(*, q: np.ndarray, lead_hours: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Return quantiles (rows, 11) with the four outer levels widened about the median."""
    out = q.copy()
    band = np.digitize(lead_hours, BAND_EDGES_HOURS[1:-1])
    median = q[:, fam.MEDIAN_INDEX]
    for j, (index, _level) in enumerate(OUTER):
        out[:, index] = median + factors[band, j] * (q[:, index] - median)
    return np.sort(out, axis=1)


def fit_one(
    *, arm: Arm, fold: prep.FoldTrain, seed: int, device: str
) -> tuple[torch.nn.Module, int, float]:
    """Fit a network of an arm on a fold and seed; return it with its epochs and held-out loss."""
    if arm.attention:
        fits = fit.fit_arm(arm="A0", setting=SETTING, fold=fold, seed=seed, device=device)
        only = fits[None]
        return only.model, only.epochs, only.val_nll
    z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    torch.manual_seed(seed)
    model = fam.PooledNetwork(
        n_inputs=fold.x.shape[1],
        family=arm.family,
        components=arm.components,
        hidden=SETTING.hidden,
        layers=SETTING.layers,
        dropout=SETTING.dropout,
        n_primaries=len(fold.target_mean),
    ).to(device)
    return train_network(
        model=model,
        arm=arm,
        x=torch.from_numpy(fold.x).to(device),
        z=torch.from_numpy(z.astype(np.float32)).to(device),
        w=torch.from_numpy(fold.weight).to(device),
        primary=torch.from_numpy(fold.primary).to(device),
        is_val=torch.from_numpy(fold.is_val).to(device),
        seed=seed,
    )


def predict_rows(
    *, model: torch.nn.Module, arm: Arm, x: np.ndarray, primary: np.ndarray, device: str
) -> np.ndarray:
    """Return the 11 standardised quantiles of rows, for any arm."""
    if not arm.attention:
        assert isinstance(model, fam.PooledNetwork)
        return predict_z(model=model, arm=arm, x=x, primary=primary, device=device)
    held = fit.Fitted(model=model, setting=SETTING, arm="A0", seed=0, epochs=0, val_nll=0.0)
    pieces = []
    for p in np.unique(primary):
        rows = primary == p
        q, _, _ = fit.predict_z_quantiles(
            fitted=held, x=x[rows], primary=int(p), members=1, device=device
        )
        pieces.append((np.flatnonzero(rows), q))
    out = np.zeros((len(x), len(fam.LEVELS)))
    for index, q in pieces:
        out[index] = q
    return np.sort(out, axis=1)


def forecast_month(
    *,
    model: torch.nn.Module,
    arm: Arm,
    fold: prep.FoldTrain,
    factors: np.ndarray,
    month: str,
    device: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the test month's raw and band-widened quantiles (fraction of capacity)."""
    raw, wide = [], []
    for origin in prep.test_origins(month=month):
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme=SCHEME, standardiser=fold.standardiser
        ):
            p = int(row["primary"])
            q = predict_rows(
                model=model, arm=arm, x=x, primary=np.full(len(x), p, dtype=np.int64), device=device
            )
            q = q * fold.target_std[p] + fold.target_mean[p]
            lead = keys["lead_hours"].to_numpy()
            scaled = widen(
                q=(q - fold.target_mean[p]) / fold.target_std[p], lead_hours=lead, factors=factors
            )
            scaled = scaled * fold.target_std[p] + fold.target_mean[p]
            base = keys.drop("lead_hours")
            for target, values in ((raw, q), (wide, scaled)):
                target.append(
                    base.hstack(
                        pl.DataFrame(
                            np.sort(values, axis=1).astype(np.float32),
                            schema=list(common.LEVEL_COLUMNS),
                            orient="row",
                        )
                    )
                )
    return pl.concat(raw), pl.concat(wide)


def run(*, arms: list[str], seeds: list[int], months: list[str], device: str) -> None:
    """Fit each arm on each fold and seed, save the networks and factors, forecast the month."""
    ordered = sorted(arms, key=lambda name: ARMS[name].half_life_days)
    for month in months:
        fold: prep.FoldTrain | None = None
        fold_half_life = -1.0
        for name in ordered:
            arm = ARMS[name]
            todo = [
                s
                for s in seeds
                if not (OUT_DIR / "seed_preds" / name / f"{month}_s{s}.parquet").exists()
            ]
            if not todo:
                continue
            if fold is None or fold_half_life != arm.half_life_days:
                fold = None
                fold = build_train(
                    refit=prep.refit_of(month=month), half_life_days=arm.half_life_days
                )
                fold_half_life = arm.half_life_days
                print(
                    f"fold {month} half-life {fold_half_life:.0f} d: {len(fold.x):,} rows",
                    flush=True,
                )
            for seed in todo:
                target = OUT_DIR / "seed_preds" / name / f"{month}_s{seed}.parquet"
                started = time.monotonic()
                model, epochs, val = fit_one(arm=arm, fold=fold, seed=seed, device=device)
                held = np.flatnonzero(fold.is_val)
                q_held = predict_rows(
                    model=model, arm=arm, x=fold.x[held], primary=fold.primary[held], device=device
                )
                z_held = (fold.y[held] - fold.target_mean[fold.primary[held]]) / fold.target_std[
                    fold.primary[held]
                ]
                factors = band_factors(q=q_held, z=z_held, lead_hours=fold.lead[held])
                folder = OUT_DIR / "models" / name / month
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), folder / f"s{seed}.pt")
                (folder / f"s{seed}.json").write_text(
                    json.dumps(
                        {
                            "arm": name,
                            "epochs": epochs,
                            "heldout_loss": val,
                            "factors": factors.tolist(),
                            "target_mean": fold.target_mean.tolist(),
                            "target_std": fold.target_std.tolist(),
                        }
                    )
                )
                raw, wide = forecast_month(
                    model=model, arm=arm, fold=fold, factors=factors, month=month, device=device
                )
                common.atomic_write(
                    frame=wide,
                    target=OUT_DIR / "seed_preds" / f"{name}_bw" / f"{month}_s{seed}.parquet",
                )
                common.atomic_write(frame=raw, target=target)
                print(f"{name} {month} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Run the named arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(arms=args.arms, seeds=args.seeds, months=args.months, device=args.device)


if __name__ == "__main__":
    main()
