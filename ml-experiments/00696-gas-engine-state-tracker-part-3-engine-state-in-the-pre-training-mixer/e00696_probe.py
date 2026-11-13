# ruff: noqa: E501, PT018, PERF401
"""Experiment 696: the encoder probe M1, a known-answer test at the host Primary.

M1 asks whether the weather encoder and the date-time encoder, pre-trained with the engine's true
state in the mixer (arm WST) or without it (arm W0), describe the host's own demand better. The probe
freezes an arm's pre-trained encoders and embedding, fits a fresh mixer with no state input and no lag
input to the host's reading *without* the injected engine, and scores the mean absolute error on
held-out weeks. It follows the linear probe A6-2 of experiment 570 with a small multilayer perceptron
in place of the linear layer.

**Rows.** The host's pre-training rows. Rows outside the held-out weeks train the mixer. The held-out
weeks are split by the parity of the week number of the target half-hour: even weeks decide early
stopping, odd weeks give the score, so no row decides both.
"""

from dataclasses import dataclass
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
from e00696_model import PROBE_WIDTH, encoder_features

WEEK: Final[int] = 336
"""Half-hours in a week."""
FEATURE_CHUNK: Final[int] = 200_000


@dataclass(frozen=True)
class ProbeSplit:
    """Row indices of the host's rows: training, early stopping, and scoring."""

    train: np.ndarray
    stop: np.ndarray
    score: np.ndarray


def split_rows(*, is_val: np.ndarray, target_idx: np.ndarray, is_host: np.ndarray) -> ProbeSplit:
    """Split the host's rows into training rows and held-out weeks for stopping and scoring."""
    host = np.flatnonzero(is_host)
    held = is_val[host]
    even = (target_idx[host] // WEEK) % 2 == 0
    return ProbeSplit(train=host[~held], stop=host[held & even], score=host[held & ~even])


def features_of(
    *, model: torch.nn.Module, x: torch.Tensor, primary: torch.Tensor, rows: np.ndarray
) -> torch.Tensor:
    """Return the frozen encoder features of the chosen rows, computed in chunks under no gradient."""
    model.eval()
    index = torch.from_numpy(rows).to(x.device)
    out = []
    with torch.no_grad():
        for chunk in index.split(FEATURE_CHUNK):
            out.append(encoder_features(model=model, x=x[chunk], primary=primary[chunk]))
    return torch.cat(out)


def fit_probe(
    *, features: torch.Tensor, target: torch.Tensor, n_train: int, n_stop: int, seed: int
) -> float:
    """Fit a fresh mixer on the features and return its mean absolute error on the scoring rows.

    Args:
        features: The frozen encoder features, rows ordered training, early stopping, scoring.
        target: The host's reading without the engine, in points of capacity, in the same order.
        n_train: Number of training rows at the front.
        n_stop: Number of early-stopping rows after them. The rest score.
        seed: The seed of the mixer and of the batches.

    Returns:
        The mean absolute error on the scoring rows, in points of capacity.
    """
    assert n_train > 0 and n_stop > 0 and len(features) > n_train + n_stop
    device = features.device
    torch.manual_seed(seed)
    train_y = target[:n_train]
    mean, std = train_y.mean(), train_y.std().clamp_min(1e-6)
    z = (target - mean) / std
    mixer = base.mlp(n_in=features.shape[1], hidden=PROBE_WIDTH, layers=2, n_out=1).to(device)
    optimiser = torch.optim.AdamW(mixer.parameters(), lr=c.PRETRAIN_LR, weight_decay=c.WEIGHT_DECAY)
    stop = slice(n_train, n_train + n_stop)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(c.MAX_EPOCHS):
        mixer.train()
        order = torch.randperm(n_train, device=device)[: max(1, int(c.ROW_FRACTION * n_train))]
        for start in range(0, len(order), c.BATCH):
            b = order[start : start + c.BATCH]
            loss = (mixer(features[b]).squeeze(1) - z[b]).abs().mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mixer.parameters(), c.GRAD_CLIP)
            optimiser.step()
        mixer.eval()
        with torch.no_grad():
            val = float((mixer(features[stop]).squeeze(1) - z[stop]).abs().mean())
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in mixer.state_dict().items()}
        else:
            stale += 1
            if stale >= c.PATIENCE:
                break
    assert best_state is not None
    mixer.load_state_dict(best_state)
    mixer.eval()
    with torch.no_grad():
        score = float(
            (mixer(features[n_train + n_stop :]).squeeze(1) - z[n_train + n_stop :]).abs().mean()
        )
    return score * float(std)


def probe_model(
    *, model: torch.nn.Module, x: torch.Tensor, primary: np.ndarray, raw_points: np.ndarray,
    is_val: np.ndarray, target_idx: np.ndarray, host_row: int, seed: int,
) -> dict[str, float | int]:  # fmt: skip
    """Run the probe on a pre-trained network and return the held-out error and the row counts.

    Args:
        model: The pre-trained network (stage `pre`), whose encoders are frozen by the probe.
        x: The pre-training input on the device, without state columns.
        primary: Each row's Primary (stack row).
        raw_points: Each row's engine-free reading, in points of capacity (no NaN on host rows).
        is_val: Each row's held-out-weeks flag.
        target_idx: Each row's target half-hour.
        host_row: The host's stack row.
        seed: The seed of the probe's mixer.

    Returns:
        `mae_points` (the held-out mean absolute error in points of capacity) and the three row counts.
    """
    split = split_rows(is_val=is_val, target_idx=target_idx, is_host=primary == host_row)
    rows = np.concatenate([split.train, split.stop, split.score])
    prim = torch.from_numpy(primary.astype(np.int64)).to(x.device)
    features = features_of(model=model, x=x, primary=prim, rows=rows)
    target = torch.from_numpy(raw_points[rows].astype(np.float32)).to(x.device)
    assert torch.isfinite(target).all(), "a host row has no reading without the engine"
    mae = fit_probe(
        features=features,
        target=target,
        n_train=len(split.train),
        n_stop=len(split.stop),
        seed=seed,
    )
    return {
        "mae_points": mae,
        "n_train": len(split.train),
        "n_stop": len(split.stop),
        "n_score": len(split.score),
    }
