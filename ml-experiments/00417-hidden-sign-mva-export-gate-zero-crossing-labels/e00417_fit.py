# ruff: noqa: E501
"""Fit the arms of experiment 417 on rotation r1 and forecast the test months.

Arms (each fitted on the CAMS-and-ERA5 inputs of `e00417_inputs`, monthly refit, one seed):

- `ref`: experiment 412's S1c (the folded likelihood opens where `u >= 0.3`).
- `A`: S1c with the Primary-level gate; a Primary whose gate is shut trains and forecasts with
  `X >= 0` at every row.
- `AH`: `A` with the high-irradiance row gate H on the Primaries whose gate is open.
- `C`: S1c with the zero-crossing labels as signed targets on labelled rows.

Each arm runs at injected level 2.0 (Q = 2 points), the no-injection placebo, and the shuffled sky,
in both test months, and at level 2.0 at power factor 0.97. The prior penalty's gradient share on
`log_c` is printed and stored for every fit.

```bash
. env.sh   # PYTHONPATH and DATA_PATH_INTERNAL; see the plan's Compute section
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00417-*/e00417_fit.py run --arms ref --device cuda
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_prep as base  # ty: ignore[unresolved-import]
import e00412_fit as f412  # ty: ignore[unresolved-import]
import e00417_gates as gates
import e00417_inputs as inputs
import e00417_models as models
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = f412.BATCH
MAX_EPOCHS: Final[int] = f412.MAX_EPOCHS
PATIENCE: Final[int] = f412.PATIENCE
GRAD_CLIP: Final[float] = f412.GRAD_CLIP
LR: Final[float] = f412.LR
LR_CAPACITY: Final[float] = f412.LR_CAPACITY
WEIGHT_DECAY: Final[float] = f412.WEIGHT_DECAY
SCORE_CHUNK_ROWS: Final[int] = f412.SCORE_CHUNK_ROWS
MAX_LEAD_HOURS: Final[float] = f412.MAX_LEAD_HOURS
TEST_MONTHS: Final[tuple[str, ...]] = f412.TEST_MONTHS
OUT_DIR: Final[Path] = inputs.WORK_DIR
LEVEL_COLUMNS: Final[tuple[str, ...]] = f412.LEVEL_COLUMNS


@dataclass(frozen=True)
class Arm:
    """What an arm changes."""

    mode: str = "none"
    """`none`, `primary` (design A), or `primary+hi` (A with H)."""
    labels: bool = False
    shuffle_u: bool = False


ARMS: Final[dict[str, Arm]] = {
    "ref": Arm(),
    "A": Arm(mode="primary"),
    "AH": Arm(mode="primary+hi"),
    "C": Arm(labels=True),
}
for _name in list(ARMS):
    ARMS[f"{_name}-shuffle"] = Arm(mode=ARMS[_name].mode, labels=ARMS[_name].labels, shuffle_u=True)

SETTINGS: Final[dict[str, inputs.Setting]] = {
    "injected": inputs.Setting(2.0, 2.0),
    "placebo": inputs.Setting(0.0, 2.0),
    "pf": inputs.Setting(2.0, 2.0, 0.97),
}
ARM_RUNS: Final[dict[str, tuple[str, ...]]] = {
    "": ("injected", "placebo", "pf"),
    "-shuffle": ("injected",),
}


@dataclass
class Prepared:
    """A fold with the masks the arm needs."""

    fold: inputs.Fold
    open_row: np.ndarray
    label: np.ndarray
    gates: dict
    thresholds: np.ndarray | None
    info: dict


def prepare(*, arm: Arm, fold: inputs.Fold, refit) -> Prepared:  # noqa: ANN001
    """Compute an arm's gates, thresholds, row mask, and labels from the fold's training rows."""
    gate_map = gates.primary_gates(fold=fold) if arm.mode != "none" else {}
    thresholds = (
        gates.high_irradiance_thresholds(fold=fold, refit=refit, pair=fold.pair)
        if arm.mode == "primary+hi"
        else None
    )
    open_row = gates.open_mask(
        fold=fold,
        primary=fold.primary,
        gate_u=fold.gate_u,
        valid_us=fold.valid,
        mode=arm.mode,
        gates=gate_map,
        thresholds=thresholds,
    )
    info: dict = {
        "gates": {
            s: {"open": g.open, "spearman": g.spearman, "p1_points": g.p1_points, "n": g.n_rows}
            for s, g in gate_map.items()
        }
    }
    if arm.labels:
        label, label_info = gates.labels_for_fold(fold=fold, refit=refit)
        info["labels"] = label_info
    else:
        label = np.zeros(len(fold.y), dtype=np.int8)
    return Prepared(
        fold=fold, open_row=open_row, label=label, gates=gate_map, thresholds=thresholds, info=info
    )


def train(  # noqa: PLR0915
    *, model: models.SignedMdn, prep: Prepared, seed: int, device: str
) -> tuple[int, float, dict[str, dict[str, float]]]:
    """Train with the weighted masked likelihood and early stopping on the held weeks."""
    fold = prep.fold
    torch.manual_seed(seed)
    t = {
        "x": torch.from_numpy(fold.x).to(device),
        "r": torch.from_numpy(fold.y * 100.0).to(device),
        "w": torch.from_numpy(fold.weight).to(device),
        "p": torch.from_numpy(fold.primary).to(device),
        "u": torch.from_numpy(fold.u).to(device),
        "o": torch.from_numpy(prep.open_row).to(device),
        "l": torch.from_numpy(prep.label.astype(np.int8)).to(device),
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
    if model.log_c.requires_grad:
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
            open_mask=t["o"][b],
            label=t["l"][b],
        )

    diag_rng = torch.Generator(device=device).manual_seed(seed)
    diag_idx = train_idx[torch.randperm(len(train_idx), device=device, generator=diag_rng)[:BATCH]]
    prior_divisor = 1.0 if model.prior_undivided else n_eff

    def prior_share() -> dict[str, float]:
        """Return the prior's share of the gradient on `log_c` (all Primaries, and D5 alone)."""
        b = diag_idx
        nll = (t["w"][b] * batch_nll(b)).sum() / t["w"][b].sum()
        g_nll = torch.autograd.grad(nll, model.log_c)[0]
        g_prior = torch.autograd.grad(model.penalty() / prior_divisor, model.log_c)[0]
        d5 = int(base.primaries().filter(pl.col("site") == "D5")["primary"][0])
        total = g_nll + g_prior
        return {
            "all": float(g_prior.norm() / total.norm()),
            "D5": float(g_prior[d5].abs() / total[d5].abs()),
            "prior_norm": float(g_prior.norm()),
            "nll_norm": float(g_nll.norm()),
        }

    model.eval()
    share_start = prior_share()
    share_epoch1: dict[str, float] = {}
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            loss = (t["w"][b] * batch_nll(b)).sum() / t["w"][
                b
            ].sum() + model.penalty() / prior_divisor
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.cat([batch_nll(b) for b in val_idx.split(BATCH * 4)]).mean())
        epochs = epoch + 1
        if epochs == 1:
            share_epoch1 = prior_share()
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
    return epochs, best, {"start": share_start, "epoch1": share_epoch1, "end": prior_share()}


def forecast_pair(
    *, model: models.SignedMdn, prep: Prepared, arm: Arm, month: str, device: str
) -> pl.DataFrame:
    """Return the test month's signed forecast at the folded pair, with the truth and the sky."""
    fold = prep.fold
    origins = {int(o.timestamp() * 1_000_000) for o in base.test_origins(month=month)}
    names = base.primaries()["site"].to_list()
    pieces = []
    for site in fold.pair:
        arr = inputs.site_arrays(site=site, fold=True, setting=fold.setting)
        sel = np.isin(arr.origin, list(origins)) & (arr.lead <= MAX_LEAD_HOURS) & np.isfinite(arr.y)
        valid = arr.valid[sel]
        u_gate, k = inputs.cams_for_rows(site=site, valid_us=valid)
        u_in = inputs.shuffled_u(site=site, valid_us=valid) if arm.shuffle_u else u_gate
        x = inputs.model_inputs(
            x_raw=arr.x[sel],
            valid_us=valid,
            site=site,
            standardiser=fold.standardiser,
            scaler=fold.scaler,
            cols=fold.cols,
        )
        p = names.index(site)
        row_open = gates.open_mask(
            fold=fold,
            primary=np.full(len(valid), p),
            gate_u=np.nan_to_num(u_gate),
            valid_us=valid,
            mode=arm.mode,
            gates=prep.gates,
            thresholds=prep.thresholds,
        )
        p_neg, qs = [], []
        with torch.no_grad():
            for s in range(0, len(x), SCORE_CHUNK_ROWS):
                xb = torch.from_numpy(x[s : s + SCORE_CHUNK_ROWS]).to(device)
                ub = torch.from_numpy(np.nan_to_num(u_in[s : s + SCORE_CHUNK_ROWS])).to(device)
                ob = torch.from_numpy(row_open[s : s + SCORE_CHUNK_ROWS]).to(device)
                pb = torch.full((len(xb),), p, dtype=torch.int64, device=device)
                log_w, mean, std = (a.double() for a in model.parts(xb, pb, ub))
                p_neg.append(
                    models.p_negative(log_w=log_w, mean=mean, std=std, open_mask=ob).cpu().numpy()
                )
                qs.append(
                    models.signed_quantiles(log_w=log_w, mean=mean, std=std, open_mask=ob)
                    .cpu()
                    .numpy()
                )
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
                "row_open": row_open,
            }
        ).hstack(pl.DataFrame(q.astype(np.float32), schema=list(LEVEL_COLUMNS), orient="row"))
        pieces.append(frame)
    return pl.concat(pieces)


def fit_one(
    *,
    arm_name: str,
    setting: inputs.Setting,
    month: str,
    seed: int,
    device: str,
    pair: tuple[str, ...],
) -> tuple[models.SignedMdn, Prepared, int, float, float, dict]:
    """Fit one arm at one setting on one refit; return the network, masks, epochs, loss, seconds."""
    arm = ARMS[arm_name]
    started = time.monotonic()
    refit = f412.refit_time(month=month)
    fold = inputs.build_fold(refit=refit, pair=pair, setting=setting, shuffle_u=arm.shuffle_u)
    prep = prepare(arm=arm, fold=fold, refit=refit)
    print(f"  masks: {json.dumps(prep.info.get('gates', {}))}", flush=True)
    if "labels" in prep.info:
        print(f"  labels: {json.dumps(prep.info['labels'])}", flush=True)
    model = f412.make_model(arm=f412.Arm(), fold=fold, refit=refit, seed=seed)  # ty: ignore[invalid-argument-type]
    epochs, val, shares = train(model=model, prep=prep, seed=seed, device=device)
    return model, prep, epochs, val, time.monotonic() - started, shares


def run_arms(
    *, arms: list[str], seed: int, device: str, months: list[str], only: list[str] | None
) -> None:
    """Fit the named arms and write each test-month forecast."""
    pair = inputs.ROTATIONS["r1"]
    for month in months:
        for base_name in arms:
            for suffix, run_names in ARM_RUNS.items():
                arm_name = f"{base_name}{suffix}"
                for run_name in run_names:
                    if only and run_name not in only:
                        continue
                    setting = SETTINGS[run_name]
                    target = OUT_DIR / "first" / arm_name / f"{setting.tag}_{month}_s{seed}.parquet"
                    if target.exists():
                        continue
                    model, prep, epochs, val, seconds, shares = fit_one(
                        arm_name=arm_name,
                        setting=setting,
                        month=month,
                        seed=seed,
                        device=device,
                        pair=pair,
                    )
                    frame = forecast_pair(
                        model=model, prep=prep, arm=ARMS[arm_name], month=month, device=device
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
                                "prior_gradient_share": shares,
                                "capacity_points": cap,
                                "dropped": prep.fold.n_dropped,
                                **prep.info,
                            }
                        )
                    )
                    print(
                        f"{arm_name} {setting.tag} {month} s{seed}: {epochs} epochs, val {val:.4f}, "
                        f"{seconds:.0f} s, prior share of capacity gradient "
                        f"{shares['end']['all']:.4f} (D5 {shares['end']['D5']:.4f}), "
                        f"dropped {sum(prep.fold.n_dropped.values())}",
                        flush=True,
                    )


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("era5", "run"))
    parser.add_argument("--arms", nargs="+", default=["ref"])
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these runs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--months", nargs="+", default=list(TEST_MONTHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "era5":
        print(inputs.write_era5())
    else:
        run_arms(
            arms=args.arms, seed=args.seed, device=args.device, months=args.months, only=args.only
        )


if __name__ == "__main__":
    main()
