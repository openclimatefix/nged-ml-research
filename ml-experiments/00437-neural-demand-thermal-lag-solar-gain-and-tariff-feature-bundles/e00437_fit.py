# ruff: noqa: E501
r"""Fit the feature-bundle arms of experiment 437 on the four screen folds.

Every arm is experiment 425's reference network `R` (the pooled plain network with the 8-wide
Primary embedding, setting `g09`, ensemble mean and spread inputs, trained on ensemble rows only)
with extra input columns appended after experiment 397's 46. The rows, the D15 and D18 start cut,
the flagged exact-zero drop, the held-out weeks, the row weights, the standardiser, and the
per-Primary target statistics are experiment 425's (`e00425_prep.build_ens_fold`); the only change
is that the extra columns of `e00437_features` are joined on `(origin, valid time)` before the
standardiser is fitted. `check` proves that with no extra columns the fold equals experiment 425's.

An arm name is its components joined by `+` (`T+S`). Components: the bundles `T`, `S`, `C`, `F`,
and the halves used to take a winning bundle apart (`T_means`, `T_rest`, `S_means`, `S_rest`,
`C_tariff`, `C_school`, `C_national`, `F_temperature`, `F_shortwave`; see `COMPONENTS`).

Each fit writes the raw quantiles only (fraction of capacity); the scorer widens them by 1.15. The
fitted weights go under `work/models/` of this experiment's data folder.

```bash
source <(printf 'export PYTHONPATH=%s\n' \
  ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack:\
ml-experiments/00011-school-term-and-return-to-work-columns-stage1a-demand:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00407-pooled-mixture-density-network-tail-arms-and-recency-weights:\
ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens:\
ml-experiments/00425-best-mdn-architecture-assembly-with-ablations:\
ml-experiments/00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles)
export DATA_PATH_INTERNAL=<path to the private data folder>
uv run --no-sync python ml-experiments/00437-*/e00437_features.py build   # once, CPU, 30 s
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00437-*/e00437_fit.py fit --arms T S C F --device cuda
```
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00437_features as feats  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

SCREEN_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03")
"""The four folds of experiment 397's screen (autumn, Christmas, winter, spring)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
OUT_DIR: Path = feats.WORK_DIR / "work"
REFERENCE: Final[str] = "R"
"""Experiment 425's reference arm; its stored forecasts are the control."""
COMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    "T": feats.BUNDLES["T"],
    "S": feats.BUNDLES["S"],
    "C": feats.BUNDLES["C"],
    "C_tariff": ("economy_7", "economy_7_cold"),
    "C_school": tuple(c for c in feats.BUNDLES["C"] if c.startswith("school_")),
    "C_national": tuple(c for c in feats.BUNDLES["C"] if "_prev_day" in c),
    "F": feats.BUNDLES["F"],
    "T_means": ("t_mean_3h", "t_mean_6h", "t_mean_24h"),
    "T_rest": tuple(c for c in feats.BUNDLES["T"] if not c.startswith("t_mean_")),
    "S_means": ("sw_mean_3h", "sw_mean_6h", "sw_mean_24h"),
    "S_rest": ("csi_change_3h", "cams_prev_day_mean", "cams_prev_day_csi"),
    "F_temperature": ("f_t_mean_6h", "f_t_mean_24h", "f_cover_6h", "f_cover_24h"),
    "F_shortwave": ("f_sw_mean_6h", "f_sw_mean_24h", "f_cover_6h", "f_cover_24h"),
}
"""Components of an arm: the four bundles and the halves a winning bundle is taken apart into."""


PRETRAINED_SUFFIX: Final[str] = "_pre"
"""An arm named `T_pre` is `T` pretrained on ERA5-and-CAMS rows and fine-tuned on ensemble rows."""


def is_pretrained(*, arm: str) -> bool:
    """Return whether the arm name asks for the pretrained form."""
    return arm.endswith(PRETRAINED_SUFFIX)


def columns_of(*, arm: str) -> tuple[str, ...]:
    """Return the extra columns of an arm named by `+`-joined components, in order, without repeats.

    Raises:
        ValueError: If a component is unknown.
    """
    arm = arm.removesuffix(PRETRAINED_SUFFIX)
    unknown = [c for c in arm.split("+") if c not in COMPONENTS]
    if unknown:
        msg = f"unknown components {unknown}; known: {sorted(COMPONENTS)}"
        raise ValueError(msg)
    seen: dict[str, None] = {}
    for part in arm.split("+"):
        seen |= dict.fromkeys(COMPONENTS[part])
    return tuple(seen)


def seed_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where an arm's seed-level raw quantile forecast is stored."""
    return OUT_DIR / "seed_preds" / arm / f"{month}_s{seed}.parquet"


def build_fold(*, refit: datetime, columns: tuple[str, ...]) -> prep425.EnsFold:
    """Return experiment 425's fine-tuning fold with the extra columns appended.

    As `e00425_prep.build_ens_fold` with `cut=True`; the extra columns are joined from
    `e00437_features` on `(origin, valid time)` for the rows the cuts keep, and standardised with
    the other weather inputs on the fold's training rows. Without columns the result is
    experiment 425's fold.
    """
    names, mask, layout = base.feature_names(scheme="summary")
    del names
    mask = np.concatenate([mask, np.ones(len(columns), dtype=bool)])
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    parts = []
    for row in base.primaries().iter_rows(named=True):
        z = np.load(base.RAW_DIR / "summary" / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= prep425._us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= prep425._us(t=refit))
            & (z["valid"] > prep425._us(t=refit - timedelta(days=base.WINDOW_DAYS)))
        )
        keep, counts = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if not keep.any():
            continue
        x = z["x"][keep]
        if columns:
            table = pl.read_parquet(
                feats.table_path(site=row["site"]), columns=["origin_us", "valid_us", *columns]
            )
            extra = pl.DataFrame(
                {"origin_us": z["origin"][keep], "valid_us": z["valid"][keep]}
            ).join(
                table,
                on=["origin_us", "valid_us"],
                how="left",
                maintain_order="left",
                validate="m:1",
            )
            if extra.select(columns).null_count().sum_horizontal().item():
                msg = f"{row['site']}: a kept row has no value for an extra column"
                raise ValueError(msg)
            x = np.concatenate([x, extra.select(columns).to_numpy().astype(np.float32)], axis=1)
        parts.append(
            {"x": x}
            | {k: z[k][keep] for k in ("y", "origin", "valid", "lead")}
            | {"primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64)}
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - prep425._us(t=base.FIRST_RUN)) // prep425.US_PER_DAY).astype(
        np.int64
    )
    held, purged = base.held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = base.Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (prep425._us(t=refit) - cat["valid"]) / prep425.US_PER_DAY
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(base.primaries())
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    return prep425.EnsFold(
        x=standardiser.apply(x=cat["x"][keep]),
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
        valid=cat["valid"][keep],
        dropped=dropped,
    )


def extra_inputs(
    *, tables: dict[str, pl.DataFrame], site: str, keys: pl.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    """Return the raw extra columns of a run's forecast rows, looked up on `(origin, valid time)`."""
    wanted = keys.select(
        origin_us=pl.col("power_fcst_init_time").dt.epoch("us"),
        valid_us=pl.col("valid_time").dt.epoch("us"),
    )
    joined = wanted.join(
        tables[site],
        on=["origin_us", "valid_us"],
        how="left",
        maintain_order="left",
        validate="m:1",
    )
    if joined.select(columns).null_count().sum_horizontal().item():
        msg = f"{site}: a forecast row has no value for an extra column"
        raise ValueError(msg)
    return joined.select(columns).to_numpy().astype(np.float32)


def forecast_month(
    *,
    fitted: fit397.Fitted,
    fold: prep425.EnsFold,
    columns: tuple[str, ...],
    month: str,
    device: str,
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity), one run at a time.

    The base inputs come from `e00397_prep.run_inputs` with the first 46 standardiser columns; the
    extra columns are looked up for the same rows and standardised with the fold's statistics.
    """
    std = fold.standardiser
    n_base = len(std.mean) - len(columns)
    base_std = base.Standardiser(
        mean=std.mean[:n_base], std=std.std[:n_base], mask=std.mask[:n_base]
    )
    extra_std = base.Standardiser(
        mean=std.mean[n_base:], std=std.std[n_base:], mask=std.mask[n_base:]
    )
    tables = (
        {
            row["site"]: pl.read_parquet(
                feats.table_path(site=row["site"]), columns=["origin_us", "valid_us", *columns]
            )
            for row in base.primaries().iter_rows(named=True)
        }
        if columns
        else {}
    )
    pieces = []
    for origin in base.test_origins(month=month):
        for row, keys, x_base in base.run_inputs(
            origin=origin, scheme="summary", standardiser=base_std
        ):
            p = int(row["primary"])
            x = x_base
            if columns:
                extra = extra_inputs(tables=tables, site=row["site"], keys=keys, columns=columns)
                x = np.concatenate([x_base, extra_std.apply(x=extra)], axis=1)
            q, _, _ = fit397.predict_z_quantiles(
                fitted=fitted, x=x, primary=p, members=1, device=device
            )
            q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


_DENSE: dict[str, dict[str, np.ndarray]] = {}


def dense_of(*, row: dict) -> dict[str, np.ndarray]:
    """Return a Primary's dense ERA5-and-CAMS column arrays, built once per process."""
    if row["site"] not in _DENSE:
        _DENSE[row["site"]] = feats.dense_arrays(row=row, local_date=feats.local_date_codes())
    return _DENSE[row["site"]]


def fold_pretrain(
    *, fold: prep425.EnsFold, refit: datetime, columns: tuple[str, ...]
) -> tuple[prep418.PretrainSet, prep425.DroppedRows]:
    """Return a fold's ERA5-and-CAMS pretraining rows with the extra columns, and the cut counts.

    As `e00425_prep.fold_pretrain` (same row selection, the same two cuts, the same held-out weeks,
    the `std_*` columns set to the fold's median ensemble spread by lead band), with the extra
    columns of `e00437_features.pretrain_extras` appended to each row before the fold's standardiser
    is applied. The pretraining rows are valid before `refit - 3.5 h` and have origins at or
    before `refit`, so a network pretrained here has seen nothing the fold's test month holds, and
    it cannot be shared across folds.
    """
    table, _ = prep418.spread_tables(fold=fold)
    _, std_cols = prep418.column_indices()
    cutoff_valid = prep425._us(t=refit - regimes.REFIT_CUTOFF_AGE)
    cutoff_origin = prep425._us(t=refit)
    flagged = fit407.flagged_times()
    dropped = prep425.DroppedRows()
    kept = []
    total = 0
    for row in base.primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        origin = z["origin"]
        keep = (z["valid"] <= cutoff_valid) & (origin <= cutoff_origin)
        day = ((origin - prep425._us(t=base.FIRST_RUN)) // prep425.US_PER_DAY).astype(np.int64)
        held, purged = base.held_out_weeks(origin_day=day)
        keep &= ~purged
        keep, counts = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        dropped += counts
        if keep.any():
            kept.append((row, keep, held))
            total += int(keep.sum())
    out = prep418.PretrainSet(
        x=np.empty((total, fold.x.shape[1]), dtype=np.float32),
        y=np.empty(total, dtype=np.float32),
        primary=np.empty(total, dtype=np.int64),
        is_val=np.empty(total, dtype=bool),
        lead=np.empty(total, dtype=np.float32),
    )
    start = 0
    for row, keep, held in kept:
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        n = int(keep.sum())
        raw = np.concatenate(
            [
                z["x"][keep],
                feats.pretrain_extras(
                    row=row,
                    dense=dense_of(row=row),
                    columns=columns,
                    valid_us=z["valid"][keep],
                    origin_us=z["origin"][keep],
                    max_steps=prep418.MAX_STEPS,
                ),
            ],
            axis=1,
        )
        lead = z["lead"][keep]
        band = np.minimum((lead // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1)
        for a in range(0, n, prep418.CHUNK):
            b = min(a + prep418.CHUNK, n)
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            out.x[start + a : start + b] = chunk
        out.y[start : start + n] = z["y"][keep]
        out.primary[start : start + n] = row["primary"]
        out.is_val[start : start + n] = held[keep]
        out.lead[start : start + n] = lead
        start += n
    return out, dropped


def pretrained_path(*, arm: str, month: str, seed: int) -> Path:
    """Return where a pretrained network's weights are stored."""
    return OUT_DIR / "pretrained" / arm / f"{month}_s{seed}.pt"


def pretrain_arm(
    *,
    arm: str,
    fold: prep425.EnsFold,
    month: str,
    seeds: list[int],
    columns: tuple[str, ...],
    device: str,
    log: dict,
) -> None:
    """Pretrain and store one network per seed for a pretrained arm (equal row weights)."""
    todo = [s for s in seeds if not pretrained_path(arm=arm, month=month, seed=s).exists()]
    if not todo:
        return
    pre, dropped = fold_pretrain(fold=fold, refit=base.refit_of(month=month), columns=columns)
    log[f"{arm}_pretrain_rows"] = {
        "rows": len(pre.x),
        "val_rows": int(pre.is_val.sum()),
        "dropped": dropped.as_dict(),
    }
    print(
        f"  pretraining rows {len(pre.x):,} ({int(pre.is_val.sum()):,} held out); dropped {dropped.as_dict()}",
        flush=True,
    )
    t = fit425.tensors(
        x=pre.x,
        y=pre.y,
        primary=pre.primary,
        weight=np.ones(len(pre.y), dtype=np.float32),
        is_val=pre.is_val,
        fold=fold,
        device=device,
    )
    del pre
    for seed in todo:
        started = time.monotonic()
        torch.manual_seed(seed)
        model = fit425.new_model(arm=fit425.ARMS[REFERENCE], fold=fold).to(device)
        epochs, best = fit425.train(model=model, setting=fit425.SETTING, t=t, seed=seed)
        fit425.atomic_save(
            state=model.state_dict(), target=pretrained_path(arm=arm, month=month, seed=seed)
        )
        seconds = round(time.monotonic() - started)
        log[f"{arm}_pretrain_s{seed}"] = {"epochs": epochs, "val_nll": best, "seconds": seconds}
        print(
            f"  pretrain {arm} {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, {seconds} s",
            flush=True,
        )
    del t
    torch.cuda.empty_cache()


def run_arms(*, months: list[str], arms: list[str], seeds: list[int], device: str) -> None:
    """Fit each arm on each fold and seed, save the networks, and store the forecasts."""
    for month in months:
        for arm in arms:
            todo = [s for s in seeds if not seed_path(arm=arm, month=month, seed=s).exists()]
            if not todo:
                continue
            columns = columns_of(arm=arm)
            fold = build_fold(refit=base.refit_of(month=month), columns=columns)
            print(
                f"fold {month} arm {arm}: {len(fold.x):,} rows, {fold.x.shape[1]} inputs; dropped {fold.dropped.as_dict()}",
                flush=True,
            )
            log_path = OUT_DIR / "logs" / f"{month}.json"
            log = json.loads(log_path.read_text()) if log_path.exists() else {}
            log[f"{arm}_rows"] = {
                "rows": len(fold.x),
                "inputs": int(fold.x.shape[1]),
                "dropped": fold.dropped.as_dict(),
            }
            pretrained = is_pretrained(arm=arm)
            if pretrained:
                pretrain_arm(
                    arm=arm,
                    fold=fold,
                    month=month,
                    seeds=todo,
                    columns=columns,
                    device=device,
                    log=log,
                )
            setting = (
                replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR) if pretrained else fit425.SETTING
            )
            t = fit425.tensors(
                x=fold.x,
                y=fold.y,
                primary=fold.primary,
                weight=fold.weight,
                is_val=fold.is_val,
                fold=fold,
                device=device,
            )
            for seed in todo:
                started = time.monotonic()
                torch.manual_seed(seed)
                model = fit425.new_model(arm=fit425.ARMS[REFERENCE], fold=fold).to(device)
                if pretrained:
                    model.load_state_dict(
                        torch.load(pretrained_path(arm=arm, month=month, seed=seed))
                    )
                epochs, best = fit425.train(model=model, setting=setting, t=t, seed=seed)
                fitted = fit397.Fitted(
                    model=model,
                    setting=setting,
                    arm="plain_pooled",
                    seed=seed,
                    epochs=epochs,
                    val_nll=best,
                )
                fit397.save_fits(
                    fits={None: fitted},
                    fold=fold,
                    target=OUT_DIR / "models" / arm / month / f"s{seed}",
                )
                common.atomic_write(
                    frame=forecast_month(
                        fitted=fitted, fold=fold, columns=columns, month=month, device=device
                    ),
                    target=seed_path(arm=arm, month=month, seed=seed),
                )
                seconds = round(time.monotonic() - started)
                log[f"{arm}_s{seed}"] = {"epochs": epochs, "val_nll": best, "seconds": seconds}
                print(
                    f"{arm} {month} seed {seed}: {epochs} epochs, val NLL {best:.4f}, {seconds} s",
                    flush=True,
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = log_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(log, indent=1))
                temporary.replace(log_path)
            del t
            torch.cuda.empty_cache()


def check(*, month: str) -> None:
    """Prove that a fold without extra columns equals experiment 425's, and print the extra columns."""
    refit = base.refit_of(month=month)
    mine = build_fold(refit=refit, columns=())
    reference = prep425.build_ens_fold(refit=refit)
    assert mine.x.shape == reference.x.shape, (mine.x.shape, reference.x.shape)
    assert np.array_equal(mine.x, reference.x), "inputs differ from experiment 425's fold"
    assert np.array_equal(mine.y, reference.y)
    assert np.array_equal(mine.is_val, reference.is_val)
    assert mine.dropped == reference.dropped
    print(
        f"fold {month} without extra columns equals experiment 425's: {mine.x.shape}, dropped {mine.dropped.as_dict()}"
    )
    wide = build_fold(refit=refit, columns=feats.INPUT_COLUMNS)
    assert np.array_equal(wide.x[:, : mine.x.shape[1]], mine.x), "the 46 base inputs changed"
    print(f"all {len(feats.INPUT_COLUMNS)} extra columns appended: {wide.x.shape}")
    for i, name in enumerate(feats.INPUT_COLUMNS):
        col = wide.x[:, mine.x.shape[1] + i]
        print(
            f"  {name:20s} standardised mean {col.mean():+.3f} sd {col.std():.3f} min {col.min():+.2f} max {col.max():+.2f}"
        )


def check_pretraining(*, month: str) -> None:
    """Print each extra column's standardised mean and spread in the pretraining rows and the ensemble rows.

    A column whose pretraining distribution is far from its ensemble distribution would pretrain
    the network on inputs it never sees again; the weather columns are expected to agree to within
    a few tenths of a standard deviation, as the pretraining rows use ERA5 and CAMS.
    """
    refit = base.refit_of(month=month)
    columns = tuple(c for c in feats.INPUT_COLUMNS if c not in feats.BUNDLES["C"])
    fold = build_fold(refit=refit, columns=columns)
    pre, _ = fold_pretrain(fold=fold, refit=refit, columns=columns)
    offset = fold.x.shape[1] - len(columns)
    print(f"fold {month}: {len(pre.x):,} pretraining rows, {len(fold.x):,} ensemble rows")
    for i, name in enumerate(columns):
        a, b = fold.x[:, offset + i], pre.x[:, offset + i]
        print(
            f"  {name:20s} ensemble mean {a.mean():+.3f} sd {a.std():.3f} | pretraining mean {b.mean():+.3f} sd {b.std():.3f}"
        )


def main() -> None:
    """Run the named command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "check", "check-pretraining"))
    parser.add_argument("--months", nargs="+", default=list(SCREEN_MONTHS))
    parser.add_argument(
        "--arms", nargs="+", default=["T", "S", "C", "C_tariff", "F", "T_pre", "S_pre", "F_pre"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    args = parser.parse_args()
    if args.command == "check":
        check(month=args.months[0])
        return
    if args.command == "check-pretraining":
        check_pretraining(month=args.months[0])
        return
    if args.smoke:
        global OUT_DIR  # noqa: PLW0603
        OUT_DIR = feats.WORK_DIR / "work" / "smoke"
        fit397.MAX_EPOCHS = 1
    for arm in args.arms:
        if is_pretrained(arm=arm) and set(columns_of(arm=arm)) & set(feats.BUNDLES["C"]):
            parser.error(f"{arm}: the pretraining rows have no C columns")
    run_arms(months=args.months, arms=args.arms, seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
