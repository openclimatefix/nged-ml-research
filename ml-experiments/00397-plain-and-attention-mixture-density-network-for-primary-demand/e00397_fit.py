# ruff: noqa: E501
"""Fit and forecast the plain and attention mixture density networks of experiment 397.

Stages (each writes only under this experiment's private data folder):

- `grid`: the plain-network hyperparameter grid (ten settings, two seeds) on the validation block
  (train to 2025-03, validate 2025-04 to 2025-06, scored on the control member's own pass). It
  prints the pooled quantile score at days 1 to 10 per setting.
- `tokens`: the six token-width and dropout settings of the attention network, fitted as arm A0.
- `screen`: fit and forecast the named arms on the five screen folds.

Arms: `plain_own` (one plain network per Primary), `plain_pooled` (one plain network across the
Primaries with an embedding), `A0` to `A3` (group attention, pooled; `A2` with `--no-query-embedding`
is the ablation without the Primary embedding in the gate).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00397-plain-and-attention-mixture-density-network-for-primary-demand \
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00397-*/e00397_fit.py screen --scheme summary \
    --arms A0 A2 --seeds 0 1 2 --device cuda
```
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_models as models  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = 16_384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
GRAD_CLIP: Final[float] = 1.0
SCORE_CHUNK_ROWS: Final[int] = 4_096
GRID_SEEDS: Final[tuple[int, ...]] = (0, 1)
GRID_REFIT_MONTH: Final[str] = "2025-07"
"""The validation block's fold trains on rows up to 2025-03 and validates on 2025-04 to 2025-06."""
HEADLINE_MAX_HOURS: Final[float] = 240.0
COUNTABLE_EXCLUDED: Final[tuple[str, ...]] = ("D15", "D18")
OUT_DIR: Final[Path] = prep.WORK_DIR / "work"


@dataclass(frozen=True)
class Setting:
    """One hyperparameter setting."""

    name: str
    hidden: int = 64
    layers: int = 2
    components: int = 3
    lr: float = 5e-4
    weight_decay: float = 1e-3
    dropout: float = 0.0
    token: int = 32


GRID: Final[tuple[Setting, ...]] = (
    Setting("g01"),
    Setting("g02", hidden=32),
    Setting("g03", hidden=128),
    Setting("g04", layers=3),
    Setting("g05", components=2),
    Setting("g06", components=5),
    Setting("g07", lr=1e-3),
    Setting("g08", weight_decay=1e-2),
    Setting("g09", dropout=0.1),
    Setting("g10", hidden=128, layers=3, components=5),
)
"""Pre-registered: ten plain-network settings. `g01` is experiment 334's `MixtureMlp`."""


def token_settings(*, head: Setting, second: Setting) -> tuple[Setting, ...]:
    """Return the six attention settings: token 16 or 32 and dropout 0 or 0.1, then the second head."""
    return (
        replace(head, name="t01", token=16, dropout=0.0),
        replace(head, name="t02", token=16, dropout=0.1),
        replace(head, name="t03", token=32, dropout=0.0),
        replace(head, name="t04", token=32, dropout=0.1),
        replace(second, name="t05", token=32, dropout=0.0),
        replace(second, name="t06", token=32, dropout=0.1),
    )


def make_model(
    *,
    arm: str,
    setting: Setting,
    layout: models.Layout,
    n_primaries: int,
    n_inputs: int,
    primary_in_query: bool = True,
) -> torch.nn.Module:
    """Return an untrained network for an arm."""
    if arm in ("plain_own", "plain_pooled"):
        return models.PlainMdn(
            n_inputs=n_inputs,
            hidden=setting.hidden,
            layers=setting.layers,
            components=setting.components,
            dropout=setting.dropout,
            n_primaries=n_primaries if arm == "plain_pooled" else 0,
        )
    return models.GroupAttentionMdn(
        layout=layout,
        arm=arm,
        token=setting.token,
        hidden=setting.hidden,
        layers=setting.layers,
        components=setting.components,
        dropout=setting.dropout,
        n_primaries=n_primaries,
        primary_in_query=primary_in_query,
    )


def forward(*, model: torch.nn.Module, x: torch.Tensor, primary: torch.Tensor) -> torch.Tensor:
    """Return the raw mixture outputs of any network."""
    if isinstance(model, models.PlainMdn):
        return model(x, primary)
    out = model(x, primary)
    assert isinstance(out, torch.Tensor)
    return out


def train_network(
    *,
    model: torch.nn.Module,
    setting: Setting,
    x: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
    primary: torch.Tensor,
    is_val: torch.Tensor,
    seed: int,
) -> tuple[torch.nn.Module, int, float]:
    """Train with weighted negative log-likelihood and early stopping; return the best network."""
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=x.device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = models.mixture_nll(
                out=forward(model=model, x=x[b], primary=primary[b]),
                z=z[b],
                components=setting.components,
            )
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = float(
                torch.cat(
                    [
                        models.mixture_nll(
                            out=forward(model=model, x=x[b], primary=primary[b]),
                            z=z[b],
                            components=setting.components,
                        )
                        for b in val_idx.split(BATCH * 4)
                    ]
                ).mean()
            )
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
    return model, epochs, best


@dataclass
class Fitted:
    """A fitted network with what is needed to forecast with it."""

    model: torch.nn.Module
    setting: Setting
    arm: str
    seed: int
    epochs: int
    val_nll: float


def fit_arm(
    *,
    arm: str,
    setting: Setting,
    fold: prep.FoldTrain,
    seed: int,
    device: str,
    primary_in_query: bool = True,
) -> dict[int | None, Fitted]:
    """Fit an arm on a fold; `plain_own` returns one network per Primary, others one under None."""
    n_primaries = len(fold.target_mean)
    z_all = (fold.y - fold.target_mean[fold.primary]) / fold.target_std[fold.primary]
    t = {
        "x": torch.from_numpy(fold.x).to(device),
        "z": torch.from_numpy(z_all.astype(np.float32)).to(device),
        "w": torch.from_numpy(fold.weight).to(device),
        "p": torch.from_numpy(fold.primary).to(device),
        "v": torch.from_numpy(fold.is_val).to(device),
    }
    groups: list[int | None] = list(range(n_primaries)) if arm == "plain_own" else [None]
    out: dict[int | None, Fitted] = {}
    for g in groups:
        sel = (
            torch.nonzero(t["p"] == g).squeeze(1)
            if g is not None
            else torch.arange(len(t["x"]), device=device)
        )
        if len(sel) == 0 or not bool(t["v"][sel].any()):
            continue
        model = make_model(
            arm=arm,
            setting=setting,
            layout=fold.layout,
            n_primaries=n_primaries,
            n_inputs=fold.x.shape[1],
            primary_in_query=primary_in_query,
        ).to(device)
        model, epochs, val = train_network(
            model=model,
            setting=setting,
            x=t["x"][sel],
            z=t["z"][sel],
            w=t["w"][sel],
            primary=t["p"][sel],
            is_val=t["v"][sel],
            seed=seed + 1000 * (g or 0),
        )
        out[g] = Fitted(
            model=model, setting=setting, arm=arm, seed=seed, epochs=epochs, val_nll=val
        )
    return out


def predict_z_quantiles(
    *, fitted: Fitted, x: np.ndarray, primary: int, members: int, device: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return the standardised 11 quantiles per valid time, and the gate weights and token norms.

    `x` holds `members` consecutive rows per valid time. The weights and norms are the average
    over a valid time's members, or `None` for a network with no gate.
    """
    k = fitted.setting.components
    n_times = len(x) // members
    quantiles, weights, norms = [], [], []
    rows_per_chunk = max(1, SCORE_CHUNK_ROWS // members) * members
    with torch.no_grad():
        for start in range(0, len(x), rows_per_chunk):
            xb = torch.from_numpy(x[start : start + rows_per_chunk]).to(device)
            pb = torch.full((len(xb),), primary, dtype=torch.int64, device=device)
            details = isinstance(fitted.model, models.GroupAttentionMdn)
            if details:
                out, wts, nrm = fitted.model(xb, pb, details=True)  # ty: ignore[invalid-argument-type]
                weights.append(wts.reshape(-1, members, wts.shape[-1]).mean(1).cpu().numpy())
                norms.append(nrm.reshape(-1, members, nrm.shape[-1]).mean(1).cpu().numpy())
            else:
                out = forward(model=fitted.model, x=xb, primary=pb)
            log_w, mean, std = models.mixture_parts(out=out.double(), components=k)
            n = len(xb) // members
            log_w = (log_w.reshape(n, members * k)) - float(np.log(members))
            mixture = models.mixture_quantiles(
                log_w=log_w, mean=mean.reshape(n, members * k), std=std.reshape(n, members * k)
            )
            quantiles.append(mixture.cpu().numpy())
    assert sum(len(q) for q in quantiles) == n_times
    return (
        np.concatenate(quantiles),
        np.concatenate(weights) if weights else None,
        np.concatenate(norms) if norms else None,
    )


def forecast_month(
    *, fits: dict[int | None, Fitted], fold: prep.FoldTrain, scheme: str, month: str, device: str
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Return the test month's quantiles (fraction of capacity) and, for attention, the gate logs."""
    members = 51 if scheme == "member" else 1
    pieces, logs = [], []
    for origin in prep.test_origins(month=month):
        for row, keys, x in prep.run_inputs(
            origin=origin, scheme=scheme, standardiser=fold.standardiser
        ):
            p = int(row["primary"])
            fitted = fits.get(p, fits.get(None))
            if fitted is None:
                continue
            q, wts, nrm = predict_z_quantiles(
                fitted=fitted, x=x, primary=p, members=members, device=device
            )
            q = np.sort(q * fold.target_std[p] + fold.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
            if wts is not None and nrm is not None:
                logs.append(
                    keys.select(
                        "time_series_id", "power_fcst_init_time", "valid_time", "lead_hours"
                    )
                    .hstack(
                        pl.DataFrame(
                            wts.astype(np.float16), schema=[f"w_{g}" for g in models.GROUPS]
                        )
                    )
                    .hstack(
                        pl.DataFrame(
                            nrm.astype(np.float16), schema=[f"n_{g}" for g in models.GROUPS]
                        )
                    )
                )
    return pl.concat(pieces), (pl.concat(logs) if logs else None)


def save_fits(*, fits: dict[int | None, Fitted], fold: prep.FoldTrain, target: Path) -> None:
    """Save every network's weights, the standardiser, the target statistics, and the setting."""
    target.mkdir(parents=True, exist_ok=True)
    for g, f in fits.items():
        torch.save(f.model.state_dict(), target / f"{'pooled' if g is None else f'p{g:02d}'}.pt")
    (target / "meta.json").write_text(
        json.dumps(
            {
                "arm": next(iter(fits.values())).arm,
                "setting": asdict(next(iter(fits.values())).setting),
                "seed": next(iter(fits.values())).seed,
                "epochs": {str(g): f.epochs for g, f in fits.items()},
                "val_nll": {str(g): f.val_nll for g, f in fits.items()},
                "target_mean": fold.target_mean.tolist(),
                "target_std": fold.target_std.tolist(),
            }
        )
    )
    np.savez(
        target / "standardiser.npz",
        mean=fold.standardiser.mean,
        std=fold.standardiser.std,
        mask=fold.standardiser.mask,
    )


def control_pass_validation(
    *, fits: dict[int | None, Fitted], fold: prep.FoldTrain, scheme: str, device: str
) -> float:
    """Return the validation block's pinball score from the control member's (or summary) pass.

    The 51-member frames of the validation months are not stored, so the member scheme is scored
    with its single control-member pass: a fair ranking of settings, not a forecast score.
    """
    val = fold.is_val
    p = fold.primary
    countable = {
        r["primary"]
        for r in prep.primaries().iter_rows(named=True)
        if r["site"] not in COUNTABLE_EXCLUDED
    }
    losses = []
    for pr in sorted(countable):
        sel = np.nonzero(val & (p == pr) & (fold.lead <= HEADLINE_MAX_HOURS))[0]
        fitted = fits.get(pr, fits.get(None))
        if fitted is None or len(sel) == 0:
            continue
        q, _, _ = predict_z_quantiles(
            fitted=fitted, x=fold.x[sel], primary=pr, members=1, device=device
        )
        q = np.sort(q * fold.target_std[pr] + fold.target_mean[pr], axis=1)
        err = fold.y[sel, None] - q
        lv = np.array(models.LEVELS)
        losses.append(np.maximum(lv * err, (lv - 1) * err).mean() * 100)
    return float(np.mean(losses))


def run_grid(*, scheme: str, device: str, which: str, settings: tuple[Setting, ...]) -> None:
    """Fit each setting on the validation block and print its pooled score (lower is better)."""
    refit = prep.refit_of(month=GRID_REFIT_MONTH)
    fold = prep.build_train(scheme=scheme, refit=refit, validation=True)
    arm = "plain_own" if which == "plain" else "A0"
    print(f"validation fold: {len(fold.x):,} rows, {int(fold.is_val.sum()):,} validation rows")
    for setting in settings:
        scores, parameters = [], 0
        for seed in GRID_SEEDS:
            started = time.monotonic()
            fits = fit_arm(arm=arm, setting=setting, fold=fold, seed=seed, device=device)
            scores.append(
                control_pass_validation(fits=fits, fold=fold, scheme=scheme, device=device)
            )
            parameters = models.count_parameters(model=next(iter(fits.values())).model)
            print(
                f"{setting.name} seed {seed}: score {scores[-1]:.4f} "
                f"({time.monotonic() - started:.0f} s)",
                flush=True,
            )
        print(
            f"RESULT {setting.name} {asdict(setting)} parameters/network {parameters} "
            f"mean score {np.mean(scores):.4f}",
            flush=True,
        )


def run_screen(
    *,
    scheme: str,
    arms: list[str],
    setting: Setting,
    seeds: list[int],
    months: list[str],
    device: str,
    no_query_embedding: bool,
) -> None:
    """Fit each arm on each fold and seed, save the networks, forecast the test month."""
    for month in months:
        fold = prep.build_train(scheme=scheme, refit=prep.refit_of(month=month))
        print(f"fold {month}: {len(fold.x):,} rows", flush=True)
        for arm in arms:
            tag = f"{arm}_noqemb" if no_query_embedding and arm != "plain_own" else arm
            for seed in seeds:
                target = OUT_DIR / "seed_preds" / scheme / tag / f"{month}_s{seed}.parquet"
                if target.exists():
                    continue
                started = time.monotonic()
                fits = fit_arm(
                    arm=arm,
                    setting=setting,
                    fold=fold,
                    seed=seed,
                    device=device,
                    primary_in_query=not no_query_embedding,
                )
                save_fits(
                    fits=fits,
                    fold=fold,
                    target=OUT_DIR / "models" / scheme / tag / month / f"s{seed}",
                )
                q, log = forecast_month(
                    fits=fits, fold=fold, scheme=scheme, month=month, device=device
                )
                common.atomic_write(frame=q, target=target)
                if log is not None:
                    common.atomic_write(
                        frame=log, target=target.with_name(target.stem + "_gate.parquet")
                    )
                print(f"{tag} {month} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


def setting_by_name(*, name: str) -> Setting:
    """Return a grid setting by name."""
    return next(s for s in GRID if s.name == name)


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("grid", "tokens", "screen"))
    parser.add_argument("--scheme", choices=prep.SCHEMES, default="member")
    parser.add_argument("--arms", nargs="+", default=["plain_own"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--months", nargs="+", default=list(prep.SCREEN_MONTHS))
    parser.add_argument("--setting", default="g01", help="a grid setting name")
    parser.add_argument("--second", default="g02", help="the second-best head, for `tokens`")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-query-embedding", action="store_true")
    parser.add_argument("--token", type=int, default=None, help="override the token width")
    parser.add_argument("--dropout", type=float, default=None, help="override the dropout")
    args = parser.parse_args()
    if args.stage == "grid":
        run_grid(scheme=args.scheme, device=args.device, which="plain", settings=GRID)
    elif args.stage == "tokens":
        head, second = setting_by_name(name=args.setting), setting_by_name(name=args.second)
        run_grid(
            scheme=args.scheme,
            device=args.device,
            which="attention",
            settings=token_settings(head=head, second=second),
        )
    else:
        chosen = setting_by_name(name=args.setting)
        chosen = replace(
            chosen,
            token=args.token or chosen.token,
            dropout=chosen.dropout if args.dropout is None else args.dropout,
        )
        run_screen(
            scheme=args.scheme,
            arms=args.arms,
            setting=chosen,
            seeds=args.seeds,
            months=args.months,
            device=args.device,
            no_query_embedding=args.no_query_embedding,
        )


if __name__ == "__main__":
    main()
