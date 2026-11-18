"""Training loop, forecasting, and state files of the pooled network of experiment 707.

The functions are copied from experiment 397's `e00397_fit.py` (`Setting`, `train_network`,
`Fitted`, `predict_z_quantiles`, `forecast_month`, `save_fits`) and experiment 425's
`e00425_fit.py` (`atomic_save`), reduced to the pooled plain network of setting `g09`. Three
changes: `forecast_month` can append the two indicator inputs of `R_F+flags` (both 1 at every
scored run, which is after both dates); `save_fits` writes through a temporary folder and a rename;
and a state's calibration (standardiser and target statistics) is read back by `load_scale`, so
the gate can forecast from experiment 425's stored states without building a fold.

The state files have experiment 425's layout: `pooled.pt`, `meta.json`, and `standardiser.npz`
in `<arm>/<month>/s<seed>/`.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_models as models  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = 16_384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
GRAD_CLIP: Final[float] = 1.0
SCORE_CHUNK_ROWS: Final[int] = 4_096
N_FLAGS: Final[int] = 2
"""The two indicator inputs of `R_F+flags`."""


@dataclass(frozen=True)
class Setting:
    """One hyperparameter setting of experiment 397's grid."""

    name: str
    hidden: int = 64
    layers: int = 2
    components: int = 3
    lr: float = 5e-4
    weight_decay: float = 1e-3
    dropout: float = 0.0
    token: int = 32


G09: Final[Setting] = Setting(name="g09", dropout=0.1)
"""Experiment 397's setting `g09`, which experiment 425 uses: 64 wide, 2 layers, 3 components."""
FINE_TUNE_LR: Final[float] = 2e-4
"""The learning rate of every fine-tune that follows an earlier stage (experiment 418's arm P1)."""


@dataclass
class Fitted:
    """A fitted network with what is needed to forecast with it."""

    model: torch.nn.Module
    setting: Setting
    seed: int
    epochs: int
    val_nll: float


@dataclass(frozen=True)
class Scale:
    """The standardiser and target statistics of a fold: raw rows to inputs, outputs to capacity."""

    standardiser: prep.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray


def new_model(*, n_inputs: int, n_primaries: int, setting: Setting = G09) -> torch.nn.Module:
    """Return an untrained pooled plain network with an 8-wide Primary embedding."""
    return models.PlainMdn(
        n_inputs=n_inputs,
        hidden=setting.hidden,
        layers=setting.layers,
        components=setting.components,
        dropout=setting.dropout,
        n_primaries=n_primaries,
    )


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
    max_epochs: int = MAX_EPOCHS,
) -> tuple[int, float]:
    """Train in place with weighted negative log-likelihood and early stopping; return the result.

    The best network on the held-out rows is loaded back into `model`. Training runs at most
    `max_epochs` epochs (`MAX_EPOCHS` unless the smoke run lowers it).

    Returns:
        The number of epochs run and the best held-out mean negative log-likelihood.
    """
    torch.manual_seed(seed)
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=setting.lr, weight_decay=setting.weight_decay
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=x.device)]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            nll = models.mixture_nll(
                out=model(x[b], primary[b]), z=z[b], components=setting.components
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
                            out=model(x[b], primary[b]), z=z[b], components=setting.components
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
    if best_state is None:
        msg = "training produced no held-out score: the stage has no held-out rows"
        raise ValueError(msg)
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def predict_z_quantiles(*, fitted: Fitted, x: np.ndarray, primary: int, device: str) -> np.ndarray:
    """Return the standardised 11 quantiles of each row (one row per valid time)."""
    k = fitted.setting.components
    quantiles = []
    with torch.no_grad():
        for start in range(0, len(x), SCORE_CHUNK_ROWS):
            xb = torch.from_numpy(x[start : start + SCORE_CHUNK_ROWS]).to(device)
            pb = torch.full((len(xb),), primary, dtype=torch.int64, device=device)
            out = fitted.model(xb, pb)
            log_w, mean, std = models.mixture_parts(out=out.double(), components=k)
            mixture = models.mixture_quantiles(log_w=log_w, mean=mean, std=std)
            quantiles.append(mixture.cpu().numpy())
    return np.concatenate(quantiles)


def forecast_month(
    *, fitted: Fitted, scale: Scale, month: str, device: str, flags: bool = False
) -> pl.DataFrame:
    """Return the test month's raw quantiles (fraction of capacity), unwidened.

    Args:
        fitted: The pooled network.
        scale: Its standardiser and target statistics.
        month: The test month, such as `2026-01`.
        device: The torch device.
        flags: Whether the network has the two indicator inputs; they are 1 at every scored run.

    Returns:
        One row per series, origin, and valid time with the 11 `common.LEVEL_COLUMNS`.
    """
    pieces = []
    for origin in prep.test_origins(month=month):
        for row, keys, x in prep.run_inputs(origin=origin, standardiser=scale.standardiser):
            p = int(row["primary"])
            inputs = x
            if flags:
                inputs = np.concatenate([x, np.ones((len(x), N_FLAGS), dtype=np.float32)], axis=1)
            q = predict_z_quantiles(fitted=fitted, x=inputs, primary=p, device=device)
            q = np.sort(q * scale.target_std[p] + scale.target_mean[p], axis=1).astype(np.float32)
            pieces.append(
                keys.drop("lead_hours").hstack(
                    pl.DataFrame(q, schema=list(common.LEVEL_COLUMNS), orient="row")
                )
            )
    return pl.concat(pieces)


def atomic_save(*, state: dict, target: Path) -> None:
    """Write a state dict to a temporary file, then rename it into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(state, temporary)
    temporary.rename(target)


def save_fit(*, fitted: Fitted, scale: Scale, target: Path) -> None:
    """Save the network, standardiser, target statistics, and setting in experiment 425's layout.

    The files go to a temporary folder that is renamed into place, so a stopped job leaves no
    half-written state.
    """
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(fitted.model.state_dict(), temporary / "pooled.pt")
    (temporary / "meta.json").write_text(
        json.dumps(
            {
                "arm": "plain_pooled",
                "setting": asdict(fitted.setting),
                "seed": fitted.seed,
                "epochs": {"None": fitted.epochs},
                "val_nll": {"None": fitted.val_nll},
                "target_mean": scale.target_mean.tolist(),
                "target_std": scale.target_std.tolist(),
            }
        )
    )
    np.savez(
        temporary / "standardiser.npz",
        mean=scale.standardiser.mean,
        std=scale.standardiser.std,
        mask=scale.standardiser.mask,
    )
    if target.exists():
        shutil.rmtree(target)
    temporary.rename(target)


def load_scale(*, folder: Path) -> Scale:
    """Return the standardiser and target statistics stored beside a state."""
    meta = json.loads((folder / "meta.json").read_text())
    z = np.load(folder / "standardiser.npz")
    return Scale(
        standardiser=prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]),
        target_mean=np.array(meta["target_mean"]),
        target_std=np.array(meta["target_std"]),
    )


def load_fitted(*, folder: Path, device: str, n_inputs: int | None = None) -> Fitted:
    """Return the network stored in a state folder, on `device`, in evaluation mode.

    Args:
        folder: A `<arm>/<month>/s<seed>` folder with `pooled.pt` and `meta.json`.
        device: The torch device.
        n_inputs: The input width; the 46 inputs of `prep.feature_names` if not given.
    """
    meta = json.loads((folder / "meta.json").read_text())
    setting = Setting(**meta["setting"])
    width = prep.feature_names()[2].n_inputs if n_inputs is None else n_inputs
    model = new_model(n_inputs=width, n_primaries=len(meta["target_mean"]), setting=setting)
    model.load_state_dict(torch.load(folder / "pooled.pt", map_location=device))
    model.to(device).eval()
    return Fitted(
        model=model,
        setting=setting,
        seed=int(meta["seed"]),
        epochs=int(meta["epochs"]["None"]),
        val_nll=float(meta["val_nll"]["None"]),
    )
