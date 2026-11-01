"""The lag-dropout S1 network of experiment 460 and its masking of lags.

The network is experiment 404's `PlainLagMdn` with the surprise at each lag as 3 extra columns (arm
S1). During training each of the 3 lags is masked independently with probability `LAG_DROPOUT`.
A masked lag has its value, age, and surprise set to 0 and its missing flag set to 1, so the
missing flag is the mask flag. At serving time `mask_lags` hides any subset of the lags the same
way, which gives the passes (all lags, each lag alone, no lags).
"""

from typing import Final

import e00404_models as models  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

LAG_DROPOUT: Final[float] = 0.3
"""Probability that each lag is masked independently, for each training row."""
N_SURPRISE: Final[int] = 3
LAG_COLUMNS: Final[int] = models.N_LAGS * models.LAG_FEATURES
"""The first 9 columns of the lag tensor: (value, missing flag, age) for each of the 3 lags."""
PASSES: Final[dict[str, tuple[int, ...]]] = {
    "all": (),
    "only0": (1, 2),
    "only1": (0, 2),
    "only2": (0, 1),
    "none": (0, 1, 2),
}
"""Each pass of one network, with the lags it hides."""


def mask_lags(*, lags: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Return the lag tensor (rows, 12) with the lags flagged in `hidden` (rows, 3) masked.

    Args:
        lags: Value, missing flag, and age for each of the 3 lags (columns 0 to 8), then the
            surprise at each lag (columns 9 to 11).
        hidden: Boolean, True where a lag is masked.

    Returns:
        A copy of `lags` with a masked lag's value, age, and surprise at 0 and its missing flag 1.
    """
    keep = (~hidden).float()
    view = lags[:, :LAG_COLUMNS].reshape(len(lags), models.N_LAGS, models.LAG_FEATURES)
    masked = torch.cat(
        [
            view[..., :1] * keep[..., None],
            torch.maximum(view[..., 1:2].clamp(0, 1), 1.0 - keep[..., None]),
            view[..., 2:] * keep[..., None],
        ],
        dim=-1,
    ).reshape(len(lags), LAG_COLUMNS)
    if lags.shape[1] == LAG_COLUMNS:
        return masked
    return torch.cat([masked, lags[:, LAG_COLUMNS:] * keep], dim=1)


class DropoutS1Mdn(models.PlainLagMdn):
    """Experiment 404's S1 network with random lag dropout that also hides the surprise."""

    def __init__(self, *, n_base: int, n_primaries: int) -> None:
        """Build the network with the surprise columns as the extra inputs."""
        super().__init__(
            n_base=n_base, n_primaries=n_primaries, lag_dropout=LAG_DROPOUT, n_extra=N_SURPRISE
        )

    def drop_lags(self, lags: torch.Tensor) -> torch.Tensor:
        """Mask each lag with probability `LAG_DROPOUT` (training only)."""
        if not self.training:
            return lags
        hidden = torch.rand(len(lags), models.N_LAGS, device=lags.device) < self.lag_dropout
        return mask_lags(lags=lags, hidden=hidden)
