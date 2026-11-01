# ruff: noqa: E501, PLR0915, PLW2901, PT018, RUF046
r"""Fine-tune experiment 425's arm A with the Christmas-period regime arms, and forecast.

Every arm starts from experiment 425's stored pretrained weights of arm A (`pretrained/A/{month}_s{seed}.pt`)
and fine-tunes at a learning rate of 2e-4 on experiment 425's fine-tuning rows. The first layer is
widened by three zero-initialised input columns (`e00485_regime.N_EXTRA`) in every arm, so the
network is the same shape throughout; an arm that does not use a column leaves it at zero.

- `H0r`: no new input, no weight, no replay, stored pretraining;
- `H1`: day-type columns;
- `H2`: window weight 5 and 25% ERA5 replay;
- `H3`: lag substitution and its flag;
- `H4`: H1 plus H2 plus H3;
- `P`: stored fine-tune recipe from a new pretraining that weights the window rows 5.

The stored reference H0 is experiment 425's own fits; `H0r` refits one (month, seed) to show the
widened network reproduces them within noise. Row weights multiply the recency weight
`0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`. Early stopping sees the held-out weeks' unweighted
negative log-likelihood, never replay rows. Replay rows have the fold's mean recency weight (so their loss share is about 25%) and are drawn with replacement.

```bash
source .e485env.sh
systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00485-*/e00485_fit.py --arms H1 --months 2026-06 --seeds 0
```
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as base  # ty: ignore[unresolved-import]
import e00407_fit as fit407  # ty: ignore[unresolved-import]
import e00418_prep as prep418  # ty: ignore[unresolved-import]
import e00422_prep as prep422  # ty: ignore[unresolved-import]
import e00425_fit as fit425  # ty: ignore[unresolved-import]
import e00425_prep as prep425  # ty: ignore[unresolved-import]
import e00485_regime as regime  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR_NAME: Final[str] = "00485-christmas-period-demand-forecast-regime-arms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_DIR_NAME
"""This experiment's private data folder."""
OUT_DIR: Path = WORK_DIR / "work"
MONTHS5: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
SEEDS3: Final[tuple[int, ...]] = (0, 1, 2)
STORED_PRETRAINED: Final[Path] = prep425.WORK_DIR / "work" / "pretrained" / "A"


@dataclass(frozen=True)
class Arm:
    """What an arm switches on."""

    name: str
    day_type: bool = False
    substitute: bool = False
    weight: bool = False
    replay: bool = False
    christmas_pretrain: bool = False


ARMS: Final[dict[str, Arm]] = {
    a.name: a
    for a in (
        Arm("H0r"),
        Arm("H1", day_type=True),
        Arm("H2", weight=True, replay=True),
        Arm("H3", substitute=True),
        Arm("H4", day_type=True, substitute=True, weight=True, replay=True),
        Arm("P", christmas_pretrain=True),
    )
}


def clean_stack() -> np.ndarray:
    """Return the cleaned power over capacity of the 16 Primaries, shape (primaries, grid)."""
    clean = prep422.cleaned_arrays()
    table = base.primaries().sort("primary")
    return np.stack([clean[r["site"]] for r in table.iter_rows(named=True)])


def widen_state(*, state: dict, n_in: int) -> dict:
    """Return a pretrained state with `N_EXTRA` zero columns inserted before the embedding."""
    out = dict(state)
    w = state["net.0.weight"]
    zeros = torch.zeros(w.shape[0], regime.N_EXTRA, dtype=w.dtype, device=w.device)
    out["net.0.weight"] = torch.cat([w[:, :n_in], zeros, w[:, n_in:]], dim=1)
    return out


def new_model(*, fold: prep425.EnsFold, device: str) -> torch.nn.Module:
    """Return an untrained pooled plain network with the widened input."""
    return fit397.make_model(
        arm="plain_pooled",
        setting=fit425.SETTING,
        layout=fold.layout,
        n_primaries=len(fold.target_mean),
        n_inputs=fold.x.shape[1] + regime.N_EXTRA,
    ).to(device)


def pretrain_rows(
    *, fold: prep425.EnsFold, refit: datetime, window_only: bool
) -> tuple[prep418.PretrainSet, np.ndarray, np.ndarray]:
    """Return ERA5-and-CAMS pretraining rows (as experiment 425) with their valid times.

    With `window_only`, only rows whose valid time is in the Christmas window are kept (the replay
    set, held-out weeks excluded). Returns the rows, the valid times (microseconds), and the
    `lead` already inside the rows.
    """
    table, _ = prep418.spread_tables(fold=fold)
    _, std_cols = prep418.column_indices()
    cutoff_valid, cutoff_origin = (
        prep425._us(t=refit - prep425.regimes.REFIT_CUTOFF_AGE),
        prep425._us(t=refit),
    )
    flagged = fit407.flagged_times()
    files, total = [], 0
    for row in base.primaries().iter_rows(named=True):
        z = np.load(prep418.RAW_DIR / f"{row['site']}.npz")
        origin = z["origin"]
        keep = (z["valid"] <= cutoff_valid) & (origin <= cutoff_origin)
        day = ((origin - prep425._us(t=base.FIRST_RUN)) // prep425.US_PER_DAY).astype(np.int64)
        held, purged = base.held_out_weeks(origin_day=day)
        keep &= ~purged
        keep, _ = prep425.apply_cuts(
            keep=keep,
            valid=z["valid"],
            lead=z["lead"],
            site=row["site"],
            series_id=int(row["time_series_id"]),
            flagged=flagged,
        )
        if window_only:
            keep &= regime.in_window(valid_us=z["valid"]) & ~held
        if keep.any():
            files.append((row["primary"], row["site"], keep, held))
            total += int(keep.sum())
    out = prep418.PretrainSet(
        x=np.empty((total, fold.x.shape[1]), dtype=np.float32),
        y=np.empty(total, dtype=np.float32),
        primary=np.empty(total, dtype=np.int64),
        is_val=np.empty(total, dtype=bool),
        lead=np.empty(total, dtype=np.float32),
    )
    valid_all = np.empty(total, dtype=np.int64)
    start = 0
    for primary, site, keep, held in files:
        z = np.load(prep418.RAW_DIR / f"{site}.npz")
        n = int(keep.sum())
        raw, lead = z["x"][keep], z["lead"][keep]
        band = np.minimum((lead // prep418.LEAD_BAND_HOURS).astype(int), prep418.N_BANDS - 1)
        for a in range(0, n, prep418.CHUNK):
            b = min(a + prep418.CHUNK, n)
            chunk = fold.standardiser.apply(x=raw[a:b])
            chunk[:, std_cols] = table[band[a:b]]
            out.x[start + a : start + b] = chunk
        out.y[start : start + n] = z["y"][keep]
        out.primary[start : start + n] = primary
        out.is_val[start : start + n] = held[keep]
        out.lead[start : start + n] = lead
        valid_all[start : start + n] = z["valid"][keep]
        start += n
    return out, valid_all, out.lead


def train_ext(
    *,
    model: torch.nn.Module,
    setting: fit397.Setting,
    t: dict[str, torch.Tensor],
    replay: dict[str, torch.Tensor] | None,
    seed: int,
) -> tuple[int, float]:
    """Experiment 397's training loop with an optional replay share of each batch.

    With `replay`, each batch holds `(1 - REPLAY_FRACTION)` of its rows from the fine-tuning rows and
    the rest drawn with replacement from the replay rows; an epoch is one pass over the
    fine-tuning rows. Validation is the held-out weeks of the fine-tuning rows, unweighted.
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    n_replay = int(round(fit397.BATCH * regime.REPLAY_FRACTION)) if replay is not None else 0
    n_main = fit397.BATCH - n_replay
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(fit397.MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=t["x"].device)]
        for start in range(0, len(order), n_main):
            b = order[start : start + n_main]
            x, z, w, p = t["x"][b], t["z"][b], t["w"][b], t["p"][b]
            if replay is not None:
                r = torch.randint(0, len(replay["x"]), (n_replay,), device=t["x"].device)
                x = torch.cat([x, replay["x"][r]])
                z = torch.cat([z, replay["z"][r]])
                w = torch.cat([w, replay["w"][r]])
                p = torch.cat([p, replay["p"][r]])
            nll = models.mixture_nll(
                out=fit397.forward(model=model, x=x, primary=p), z=z, components=setting.components
            )
            loss = (w * nll).sum() / w.sum()
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


def forecast_month(
    *,
    fitted: fit397.Fitted,
    fold: prep425.EnsFold,
    month: str,
    arm: Arm,
    clean: np.ndarray,
    maps: list,
    device: str,
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity) for a fitted network."""
    pieces = []
    for origin in base.test_origins(month=month):
        for row, keys, x in base.run_inputs(
            origin=origin, scheme="summary", standardiser=fold.standardiser
        ):
            p = int(row["primary"])
            valid_us = keys["valid_time"].dt.epoch("us").to_numpy().astype(np.int64)
            x = regime.extra_columns(
                x=x,
                valid_us=valid_us,
                lead=keys["lead_hours"].to_numpy().astype(np.float64),
                primary=np.full(len(x), p, dtype=np.int64),
                clean=clean,
                maps=maps,
                standardiser=fold.standardiser,
                day_type=arm.day_type,
                substitute=arm.substitute,
            )
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


def tensor_dict(
    *, x: np.ndarray, z: np.ndarray, w: np.ndarray, p: np.ndarray, v: np.ndarray, device: str
) -> dict[str, torch.Tensor]:
    """Return arrays as device tensors."""
    return {
        "x": torch.from_numpy(x).to(device),
        "z": torch.from_numpy(z.astype(np.float32)).to(device),
        "w": torch.from_numpy(w.astype(np.float32)).to(device),
        "p": torch.from_numpy(p).to(device),
        "v": torch.from_numpy(v).to(device),
    }


def pretrain_christmas(
    *, fold: prep425.EnsFold, refit: datetime, month: str, seeds: list[int], device: str
) -> None:
    """Pretrain arm P's networks: experiment 425's pretraining with window rows weighted 5."""
    todo = [s for s in seeds if not (OUT_DIR / "pretrained" / "P" / f"{month}_s{s}.pt").exists()]
    if not todo:
        return
    pre, valid, _ = pretrain_rows(fold=fold, refit=refit, window_only=False)
    weight = regime.window_weight(valid_us=valid)
    print(
        f"  P pretraining rows {len(pre.x):,}; window rows {int((weight > 1).sum()):,}", flush=True
    )
    z = (pre.y - fold.target_mean[pre.primary]) / fold.target_std[pre.primary]
    t = tensor_dict(x=pre.x, z=z, w=weight, p=pre.primary, v=pre.is_val, device=device)
    del pre
    for seed in todo:
        started = time.monotonic()
        torch.manual_seed(seed)
        model = fit397.make_model(
            arm="plain_pooled",
            setting=fit425.SETTING,
            layout=fold.layout,
            n_primaries=len(fold.target_mean),
            n_inputs=fold.x.shape[1],
        ).to(device)
        epochs, best = fit425.train(model=model, setting=fit425.SETTING, t=t, seed=seed)
        fit425.atomic_save(
            state=model.state_dict(), target=OUT_DIR / "pretrained" / "P" / f"{month}_s{seed}.pt"
        )
        seconds = time.monotonic() - started
        common.log_gpu_seconds(job=f"e485_pretrain_P_{month}_s{seed}", seconds=seconds, workers=1)
        print(
            f"  pretrain P {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {seconds:.0f} s",
            flush=True,
        )
    del t
    torch.cuda.empty_cache()


def run(*, months: list[str], arms: list[Arm], seeds: list[int], device: str) -> None:
    """Fit every requested (month, arm, seed) not yet stored and write its forecast."""
    clean = clean_stack()
    for month in months:
        todo = [
            (a, s)
            for a in arms
            for s in seeds
            if not (OUT_DIR / "seed_preds" / a.name / f"{month}_s{s}.parquet").exists()
        ]
        if not todo:
            continue
        refit = base.refit_of(month=month)
        fold = prep425.build_ens_fold(refit=refit, cut=True)
        maps = regime.class_maps(clean=clean, refit=refit)
        log_path = OUT_DIR / "logs" / f"{month}.json"
        log: dict = json.loads(log_path.read_text()) if log_path.exists() else {}
        log |= {
            "month": month,
            "fine_tune_rows": len(fold.x),
            "rows_dropped": fold.dropped.as_dict(),
            "class_maps": {
                str(p): {f"{m:02d}-{d:02d}": c for (m, d), c in sorted(mp.items())}
                for p, mp in enumerate(maps)
            },
        }
        print(f"fold {month}: {len(fold.x):,} rows; dropped {fold.dropped.as_dict()}", flush=True)
        if any(a.christmas_pretrain for a, _ in todo):
            pretrain_christmas(
                fold=fold,
                refit=refit,
                month=month,
                seeds=sorted({s for a, s in todo if a.christmas_pretrain}),
                device=device,
            )
        z = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
        replay_pre = replay_valid = None
        if any(a.replay for a, _ in todo):
            replay_pre, replay_valid, _ = pretrain_rows(fold=fold, refit=refit, window_only=True)
            print(f"  replay rows {len(replay_pre.x):,}", flush=True)
            log["replay_rows"] = len(replay_pre.x)
        cache: dict[tuple[bool, bool, bool], dict[str, torch.Tensor]] = {}
        replay_cache: dict[tuple[bool, bool, bool], dict[str, torch.Tensor]] = {}
        for arm, seed in todo:
            started = time.monotonic()
            key = (arm.day_type, arm.substitute, arm.weight)
            if key not in cache:
                for old in list(cache):
                    del cache[old]
                torch.cuda.empty_cache()
                x = regime.extra_columns(
                    x=fold.x,
                    valid_us=fold.valid,
                    lead=fold.lead,
                    primary=fold.primary,
                    clean=clean,
                    maps=maps,
                    standardiser=fold.standardiser,
                    day_type=arm.day_type,
                    substitute=arm.substitute,
                )
                w = fold.weight * (regime.window_weight(valid_us=fold.valid) if arm.weight else 1.0)
                cache[key] = tensor_dict(
                    x=x, z=z, w=w, p=fold.primary, v=fold.is_val, device=device
                )
                log[f"substituted_rows_{arm.name}"] = int(x[:, -1].sum())
                del x
            replay = None
            if arm.replay:
                assert replay_pre is not None and replay_valid is not None
                rx = regime.extra_columns(
                    x=replay_pre.x,
                    valid_us=replay_valid,
                    lead=replay_pre.lead,
                    primary=replay_pre.primary,
                    clean=clean,
                    maps=maps,
                    standardiser=fold.standardiser,
                    day_type=arm.day_type,
                    substitute=arm.substitute,
                )
                rz = (replay_pre.y - fold.target_mean[replay_pre.primary]) / fold.target_std[
                    replay_pre.primary
                ]
                replay = tensor_dict(
                    x=rx,
                    z=rz,
                    w=np.full(len(rx), float(fold.weight.mean())),
                    p=replay_pre.primary,
                    v=np.zeros(len(rx), dtype=bool),
                    device=device,
                )
            torch.manual_seed(seed)
            model = new_model(fold=fold, device=device)
            source = (OUT_DIR / "pretrained" / "P") if arm.christmas_pretrain else STORED_PRETRAINED
            state = torch.load(source / f"{month}_s{seed}.pt")
            model.load_state_dict(widen_state(state=state, n_in=fold.x.shape[1]))
            setting = replace(fit425.SETTING, lr=fit425.FINE_TUNE_LR)
            epochs, best = train_ext(
                model=model, setting=setting, t=cache[key], replay=replay, seed=seed
            )
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
                target=OUT_DIR / "models" / arm.name / month / f"s{seed}",
            )
            q = forecast_month(
                fitted=fitted,
                fold=fold,
                month=month,
                arm=arm,
                clean=clean,
                maps=maps,
                device=device,
            )
            common.atomic_write(
                frame=q, target=OUT_DIR / "seed_preds" / arm.name / f"{month}_s{seed}.parquet"
            )
            seconds = time.monotonic() - started
            common.log_gpu_seconds(
                job=f"e485_{arm.name}_{month}_s{seed}", seconds=seconds, workers=1
            )
            log[f"{arm.name}_s{seed}"] = {
                "epochs": epochs,
                "val_nll": best,
                "seconds": round(seconds),
            }
            print(
                f"{arm.name} {month} s{seed}: {epochs} epochs, val NLL {best:.4f}, {seconds:.0f} s",
                flush=True,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = log_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(log, indent=1))
            tmp.replace(log_path)
            del replay
        del cache, replay_cache, replay_pre
        torch.cuda.empty_cache()


def main() -> None:
    """Fit the named arms on the named folds and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS5))
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=[a for a in ARMS if a != "H0r"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS3))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="one epoch, outputs under work/smoke")
    args = parser.parse_args()
    if args.smoke:
        global OUT_DIR  # noqa: PLW0603
        OUT_DIR = WORK_DIR / "work" / "smoke"
        fit397.MAX_EPOCHS = 1
    run(months=args.months, arms=[ARMS[a] for a in args.arms], seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
