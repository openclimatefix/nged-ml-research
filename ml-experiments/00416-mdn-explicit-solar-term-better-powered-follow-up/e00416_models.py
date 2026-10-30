"""Log-scale capacity priors for the explicit solar term (arms C3p and C3pa of experiment 416).

Experiment 405's `train` adds `model.penalty() / n_eff` to the weighted mean negative
log-likelihood of a batch, where `n_eff` is the training rows' summed weight (about 1.8 million).
The Gaussian prior of C3 therefore weighs against the whole training set's likelihood, and its
share of the log-capacity gradient is about 2.5% at the end of training. Here the penalty is
instead set against one batch's summed likelihood: it is multiplied by `n_eff / BATCH`, so that
`train`'s division by `n_eff` leaves `penalty / BATCH`.

- **C3p:** `0.5 * ((log c - log centre) / LOG_SD)^2` per capacity of a signed Primary.
- **C3pa:** `0.5 * (relu(log centre - log c) / LOG_SD)^2`. The capacity is free to rise above the
  prior centre and is penalised only for falling below it.
"""

from dataclasses import replace
from datetime import datetime
from typing import Any, Final

import e00405_fit as f5  # ty: ignore[unresolved-import]
import e00405_models as m5  # ty: ignore[unresolved-import]
import e00405_prep as p5  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

LOG_SD: Final[float] = 0.7
"""Standard deviation of the prior on the natural logarithm of the capacity."""
BATCH_ROWS: Final[int] = f5.BATCH
LOG_PRIOR_ARMS: Final[tuple[str, ...]] = ("C3p", "C3pa", "C3p-weak")
STRENGTH: Final[dict[str, float]] = {"C3p": 1.0, "C3pa": 1.0, "C3p-weak": 0.1}
"""Multiplier of the penalty (1.0 is `penalty / BATCH_ROWS`, about 100 times C3's strength)."""

# `LogPriorMdn.penalty` ignores `spec.prior`; the flag is False only to mark the base prior unused.
m5.SPECS["C3p"] = replace(m5.SPECS["C3"], prior=False)
m5.SPECS["C3pa"] = replace(m5.SPECS["C3"], prior=False)
m5.SPECS["C3p-weak"] = replace(m5.SPECS["C3"], prior=False)


class LogPriorMdn(m5.PvMdn):
    """`PvMdn` whose capacity prior acts on the log capacity, against one batch's likelihood."""

    def __init__(self, *, asymmetric: bool, n_eff: float, strength: float, **kwargs: Any) -> None:
        """Build the network; `n_eff` is the training rows' summed weight."""
        super().__init__(**kwargs)
        self.asymmetric = asymmetric
        self.penalty_weight = strength * n_eff / BATCH_ROWS

    def penalty(self) -> torch.Tensor:
        """Return the log-scale prior, multiplied by `n_eff / BATCH_ROWS` (see the module doc)."""
        mask = self.solar_mask[:, None]
        gap = self.log_c - torch.log(self.prior_centre)
        if self.asymmetric:
            gap = torch.relu(-gap)
        return self.penalty_weight * (0.5 * mask * (gap / LOG_SD) ** 2).sum()


def make_model(
    *, arm: str, fold: p5.Fold, table: pl.DataFrame, refit: datetime, seed: int
) -> m5.PvMdn:
    """Return an untrained network for an arm of this experiment."""
    if arm not in LOG_PRIOR_ARMS:
        return f5.make_model(arm=arm, fold=fold, table=table, refit=refit, seed=seed)
    torch.manual_seed(seed)
    n_eff = float(fold.weight[~fold.is_val].astype(np.float64).sum())
    return LogPriorMdn(
        asymmetric=arm == "C3pa",
        n_eff=n_eff,
        strength=STRENGTH[arm],
        spec=m5.SPECS[arm],
        n_inputs=fold.x.shape[1],
        n_primaries=len(table),
        **f5.arm_tables(arm=arm, fold=fold, table=table, refit=refit),  # ty: ignore[invalid-argument-type]
    )
