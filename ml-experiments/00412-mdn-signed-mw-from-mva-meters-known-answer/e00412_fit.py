# ruff: noqa: E501
"""Fit the structural signed mixture density network of experiment 412 and forecast the test month.

Stages (each writes only under this experiment's private data folder):

- `cams`: write each Primary's CAMS `u` and `k` and the pooled `u99` constant.
- `time`: fit one network on the 2026-06 fold and print the time, epochs, and GPU memory. It writes
  no forecast and reads no score.
- `first`: the first test (rotation `r1`, level 2.0): fit the named arms on the two test months and
  write the test-month forecasts of the folded pair.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand:\
ml-experiments/00405-mdn-explicit-pv-term-learned-capacity-known-answer:\
ml-experiments/00412-mdn-signed-mw-from-mva-meters-known-answer \
DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=1 \
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00412-*/e00412_fit.py time --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00405_pv as pv  # ty: ignore[unresolved-import]
import e00412_inputs as inputs
import e00412_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = 16_384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
GRAD_CLIP: Final[float] = 1.0
LR: Final[float] = 5e-4
LR_CAPACITY: Final[float] = 5e-3
WEIGHT_DECAY: Final[float] = 1e-3
SCORE_CHUNK_ROWS: Final[int] = 8_192
MAX_LEAD_HOURS: Final[float] = 240.0
TEST_MONTHS: Final[tuple[str, ...]] = ("2026-03", "2026-06")
OUT_DIR: Final[Path] = inputs.WORK_DIR
LEVEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{int(100 * q):02d}" for q in models.LEVELS)


@dataclass(frozen=True)
class Arm:
    """What an arm changes."""

    shuffle_u: bool = False
    u_forced_zero: bool = False
    start_scale: float = 1.0
    """The starting capacity as a multiple of the prior centre."""


ARMS: Final[dict[str, Arm]] = {
    "S1c": Arm(),
    "S1c-shuffle": Arm(shuffle_u=True),
    "S1c-u0": Arm(u_forced_zero=True),
    "S1c-start-low": Arm(start_scale=0.1 / 3.0),
    "S1c-start-high": Arm(start_scale=3.0),
}


def prior_centre_points(*, refit: datetime) -> float:
    """Return the median studies-page capacity of the signed-MW six, in points of capacity.

    Each Primary's capacity is that of the latest season-year that ended before the refit, divided
    by the Primary's effective capacity (the page gives none for the MVA-only eight).
    """
    table = base.primaries()
    cell = int(np.clip(pv.latest_complete_label(refit=refit) - 2023, 0, 2))
    values = []
    for row in table.iter_rows(named=True):
        if row["site"] not in inputs.SIGNED_SIX:
            continue
        centre, _sd = pv.studies_prior(site=row["site"], n_cells=3, refit=refit)
        values.append(100.0 * centre[cell] / float(row["effective_capacity_mw"]))
    return float(np.median(values))


def make_model(*, arm: Arm, fold: inputs.Fold, refit: datetime, seed: int) -> models.SignedMdn:
    """Return an untrained network."""
    torch.manual_seed(seed)
    centre = prior_centre_points(refit=refit)
    n = len(fold.target_mean)
    return models.SignedMdn(
        n_inputs=fold.x.shape[1],
        n_primaries=n,
        target_mean=fold.target_mean,
        target_std=fold.target_std,
        folded=fold.folded,
        capacity_start=np.full(n, centre * arm.start_scale),
        prior_centre_points=centre,
        u_forced_zero=arm.u_forced_zero,
    )


def train(
    *,
    model: models.SignedMdn,
    fold: inputs.Fold,
    seed: int,
    device: str,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[int, float]:
    """Train with the weighted folded negative log-likelihood and early stopping on the held weeks."""
    torch.manual_seed(seed)
    t = {
        "x": torch.from_numpy(fold.x).to(device),
        "r": torch.from_numpy(fold.y * 100.0).to(device),
        "w": torch.from_numpy(fold.weight).to(device),
        "p": torch.from_numpy(fold.primary).to(device),
        "u": torch.from_numpy(fold.u).to(device),
        "g": torch.from_numpy(fold.gate_u).to(device),
        "v": torch.from_numpy(fold.is_val).to(device),
    }
    model.to(device)
    train_idx = torch.nonzero(~t["v"]).squeeze(1)
    val_idx = torch.nonzero(t["v"]).squeeze(1)
    n_eff = float(t["w"][train_idx].sum())
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if n != "log_c"],
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }
    ]
    if not model.u_forced_zero:
        groups.append({"params": [model.log_c], "lr": LR_CAPACITY, "weight_decay": 0.0})
    optimiser = torch.optim.AdamW(groups)

    def batch_nll(b: torch.Tensor) -> torch.Tensor:
        log_w, mean, std = model.parts(t["x"][b], t["p"][b], t["u"][b])
        return models.nll(
            log_w=log_w,
            mean=mean,
            std=std,
            r_points=t["r"][b],
            folded=model.folded[t["p"][b]],
            gate_u=t["g"][b],
        )

    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = batch_nll(b)
            loss = (t["w"][b] * nll).sum() / t["w"][b].sum() + model.penalty() / n_eff
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([batch_nll(b) for b in val_idx.split(BATCH * 4)]).mean())
        epochs = epoch + 1
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def forecast_pair(
    *, model: models.SignedMdn, fold: inputs.Fold, arm: Arm, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's signed forecast at the folded pair, with the truth and the sky.

    Columns: `site`, `origin`, `valid`, `lead`, `u`, `u_gate` (the real scaled unit shape), `k`,
    `reading_points`, `truth_points` (signed, farm subtracted), `p_negative` (`P(X < 0)`), and the
    11 signed quantiles of `X` in points. Test rows are the runs whose origin falls in `month`,
    with a lead of at most 240 hours.
    """
    origins = {int(o.timestamp() * 1_000_000) for o in base.test_origins(month=month)}
    pieces = []
    for site in fold.pair:
        arr = inputs.site_arrays(site=site, fold=True, setting=fold.setting)
        sel = np.isin(arr.origin, list(origins)) & (arr.lead <= MAX_LEAD_HOURS) & np.isfinite(arr.y)
        valid = arr.valid[sel]
        u_gate, k = inputs.cams_for_rows(site=site, valid_us=valid)
        u_in = inputs.shuffled_u(site=site, valid_us=valid) if arm.shuffle_u else u_gate
        x = fold.standardiser.apply(x=arr.x[sel])[:, fold.keep]
        p = int(base.primaries().filter(pl.col("site") == site)["primary"][0])
        p_neg, qs = [], []
        with torch.no_grad():
            for s in range(0, len(x), SCORE_CHUNK_ROWS):
                xb = torch.from_numpy(x[s : s + SCORE_CHUNK_ROWS]).to(device)
                ub = torch.from_numpy(np.nan_to_num(u_in[s : s + SCORE_CHUNK_ROWS])).to(device)
                pb = torch.full((len(xb),), p, dtype=torch.int64, device=device)
                log_w, mean, std = (a.double() for a in model.parts(xb, pb, ub))
                p_neg.append(
                    models.mixture_cdf(
                        x=torch.zeros(len(xb), 1, device=device, dtype=torch.float64),
                        log_w=log_w,
                        mean=mean,
                        std=std,
                    )[:, 0]
                    .cpu()
                    .numpy()
                )
                qs.append(models.quantiles(log_w=log_w, mean=mean, std=std).cpu().numpy())
        q = np.sort(np.concatenate(qs), axis=1)
        frame = pl.DataFrame(
            {
                "site": [site] * int(sel.sum()),
                "origin": arr.origin[sel],
                "valid": valid,
                "lead": arr.lead[sel],
                "u": np.nan_to_num(u_in),
                "u_gate": np.nan_to_num(u_gate),
                "k": np.nan_to_num(k),
                "reading_points": arr.y[sel] * 100.0,
                "truth_points": arr.y_truth[sel] * 100.0,
                "p_negative": np.concatenate(p_neg).astype(np.float32),
            }
        ).hstack(pl.DataFrame(q.astype(np.float32), schema=list(LEVEL_COLUMNS), orient="row"))
        pieces.append(frame)
    return pl.concat(pieces)


def refit_time(*, month: str) -> datetime:
    """Return the refit time of a test month."""
    return base.refit_of(month=month)


def fit_one(
    *,
    arm_name: str,
    pair: tuple[str, ...],
    setting: inputs.Setting,
    month: str,
    seed: int,
    device: str,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[models.SignedMdn, inputs.Fold, int, float, float]:
    """Fit one arm at one setting on one refit; return the network, fold, epochs, loss, seconds."""
    arm = ARMS[arm_name]
    started = time.monotonic()
    refit = refit_time(month=month)
    fold = inputs.build_fold(refit=refit, pair=pair, setting=setting, shuffle_u=arm.shuffle_u)
    model = make_model(arm=arm, fold=fold, refit=refit, seed=seed)
    epochs, val = train(model=model, fold=fold, seed=seed, device=device, max_epochs=max_epochs)
    return model, fold, epochs, val, time.monotonic() - started


def run_time(*, device: str) -> None:
    """Fit S1c once on the 2026-06 fold and print the timings (no forecast, no score)."""
    started = time.monotonic()
    setting = inputs.Setting(level=2.0, q_points=2.0)
    arm = ARMS["S1c"]
    refit = refit_time(month="2026-06")
    fold = inputs.build_fold(refit=refit, pair=inputs.ROTATIONS["r1"], setting=setting)
    built = time.monotonic() - started
    print(
        f"fold: {len(fold.x):,} rows x {fold.x.shape[1]} inputs, built in {built:.0f} s; "
        f"dropped by the injection union: { {k: v for k, v in fold.n_dropped.items() if v} }",
        flush=True,
    )
    model = make_model(arm=arm, fold=fold, refit=refit, seed=0)
    fit_started = time.monotonic()
    epochs, val = train(model=model, fold=fold, seed=0, device=device)
    fit_seconds = time.monotonic() - fit_started
    cap = model.capacity().detach().cpu().numpy()
    print(
        f"S1c: {epochs} epochs, val nll {val:.4f}, fit {fit_seconds:.0f} s "
        f"({fit_seconds / epochs:.1f} s per epoch), total {time.monotonic() - started:.0f} s",
        flush=True,
    )
    print(f"capacity (points) by Primary: {np.round(cap, 2).tolist()}")
    print(f"gpu max memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def run_first(*, arms: list[str], seed: int, device: str, months: list[str]) -> None:
    """The first test: rotation r1, level 2.0, reactive floor 2 and 4 points, plus the placebo."""
    pair = inputs.ROTATIONS["r1"]
    settings = {
        "S1c": [inputs.Setting(2.0, 2.0), inputs.Setting(2.0, 4.0), inputs.Setting(0.0, 2.0)],
        "S1c-shuffle": [inputs.Setting(2.0, 2.0), inputs.Setting(2.0, 4.0)],
        "S1c-u0": [inputs.Setting(2.0, 2.0)],
    }
    for month in months:
        for arm_name in arms:
            for setting in settings[arm_name]:
                target = OUT_DIR / "first" / arm_name / f"{setting.tag}_{month}_s{seed}.parquet"
                if target.exists():
                    continue
                model, fold, epochs, val, seconds = fit_one(
                    arm_name=arm_name,
                    pair=pair,
                    setting=setting,
                    month=month,
                    seed=seed,
                    device=device,
                )
                frame = forecast_pair(
                    model=model, fold=fold, arm=ARMS[arm_name], month=month, device=device
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".tmp")
                frame.write_parquet(tmp)
                tmp.rename(target)
                cap = model.capacity().detach().cpu().numpy().tolist()
                state = OUT_DIR / "models" / arm_name / f"{setting.tag}_{month}_s{seed}.pt"
                state.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), state)
                target.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "epochs": epochs,
                            "val_nll": val,
                            "seconds": seconds,
                            "capacity_points": cap,
                            "dropped": fold.n_dropped,
                        }
                    )
                )
                print(
                    f"{arm_name} {setting.tag} {month} s{seed}: {epochs} epochs, val {val:.4f}, {seconds:.0f} s",
                    flush=True,
                )


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cams", "time", "first"))
    parser.add_argument("--arms", nargs="+", default=["S1c"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--months", nargs="+", default=list(TEST_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "cams":
        print(inputs.write_cams())
    elif args.stage == "time":
        run_time(device=args.device)
    else:
        run_first(arms=args.arms, seed=args.seed, device=args.device, months=args.months)


if __name__ == "__main__":
    main()
