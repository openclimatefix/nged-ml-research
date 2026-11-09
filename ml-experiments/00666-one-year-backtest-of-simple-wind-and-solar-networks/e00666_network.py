"""The three-component Gaussian mixture head and its training loop, shared by both fits.

`MixtureHead` and `mixture_quantiles` are copied from experiment 310 (`e00310_fit.py`) and
`fit_network` from experiments 659 (`e00659_wind_fit.py`, `e00659_solar_fit.py`): AdamW, learning
rate 2e-3, weight decay 1e-4, batch 2048, the epoch with the lowest stopping-window loss kept.
"""

from typing import Final

import e00666_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
from scipy.special import ndtr
from torch import nn  # ty: ignore[unresolved-import]

COMPONENTS: Final[int] = 3
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4


class MixtureHead(nn.Module):
    """Three-component Gaussian mixture on the residual from the mixture's centre."""

    def __init__(self, n_in: int) -> None:
        """Build two hidden layers of 64 units and the head of three weights, means, and spreads."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3 * COMPONENTS),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3))

    def parameters_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the log weights, means, and standard deviations of the three components."""
        out = self.net(x)
        return (
            torch.log_softmax(out[:, :COMPONENTS], dim=1),
            out[:, COMPONENTS : 2 * COMPONENTS],
            torch.exp(out[:, 2 * COMPONENTS :].clamp(-6.0, 1.0)),
        )

    def nll(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the mean negative log likelihood of `batch["actual"]` around `batch["mu"]`."""
        log_w, mean, std = self.parameters_of(batch["x"])
        z = (batch["actual"] - batch["mu"])[:, None]
        log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - 0.5 * float(np.log(2 * np.pi))
        return -torch.logsumexp(log_w + log_pdf, dim=1).mean()


def mixture_quantiles(
    *, model: MixtureHead, batch: dict[str, torch.Tensor], ceiling: float
) -> np.ndarray:
    """Return the mixture's 11 quantiles by bisection on its CDF, clamped to [0, ceiling]."""
    with torch.no_grad():
        log_w, mean, std = model.parameters_of(batch["x"])
    w, mean, std = (t.cpu().numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
    mu = batch["mu"].cpu().numpy().astype(np.float64)
    out = np.empty((len(mu), len(common.LEVELS)))
    for k, level in enumerate(common.LEVELS):
        lo, hi = np.full(len(mu), -2.0), np.full(len(mu), 2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
            below = cdf < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        out[:, k] = 0.5 * (lo + hi)
    return np.minimum(np.maximum(mu[:, None] + out, 0.0), ceiling)


def fit_network(
    *,
    x: np.ndarray,
    mu: np.ndarray,
    y: np.ndarray,
    is_fit: np.ndarray,
    is_stop: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    seed: int,
) -> tuple[MixtureHead, dict[str, torch.Tensor]]:
    """Train on the `is_fit` rows, keep the epoch with the lowest loss on the `is_stop` rows.

    Args:
        x: The raw (unstandardised) inputs, shape (rows, columns).
        mu: The mixture's centre per row.
        y: The target per row, as a fraction of the refit's capacity.
        is_fit: Boolean mask of the rows to train on.
        is_stop: Boolean mask of the stopping-window rows.
        mean: Per-column mean used to standardise `x`.
        std: Per-column standard deviation used to standardise `x`.
        epochs: Number of passes over the fit rows.
        seed: Seed of the weights and of the batch order.

    Returns:
        The network with the best epoch's weights loaded, and those weights.
    """
    torch.manual_seed(seed)
    x_t = torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    model = MixtureHead(x.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_i = torch.tensor(np.flatnonzero(is_fit), device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(is_stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        order = fit_i[torch.randperm(len(fit_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            i = order[start : start + BATCH]
            loss = model.nll({"x": x_t[i], "mu": mu_t[i], "actual": y_t[i]})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.nll({"x": x_t[stop_i], "mu": mu_t[stop_i], "actual": y_t[stop_i]}))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best_state


def predict_quantiles(
    *, model: MixtureHead, x: np.ndarray, mu: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Return the sorted (rows, 11) quantiles as fractions of the refit's capacity, capped at 1."""
    x_t = torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)
    result = np.empty((len(x), len(common.LEVELS)), dtype=np.float32)
    for start in range(0, len(x), 50_000):
        sl = slice(start, start + 50_000)
        batch = {"x": x_t[sl], "mu": torch.tensor(mu[sl], dtype=torch.float32, device=DEVICE)}
        result[sl] = mixture_quantiles(model=model, batch=batch, ceiling=1.0)
    return np.sort(result, axis=1)
