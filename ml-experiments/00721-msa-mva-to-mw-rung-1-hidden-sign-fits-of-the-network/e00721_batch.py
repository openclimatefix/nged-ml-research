# ruff: noqa: E501
"""Experiment 721: the batch source with the folded and signed histories, the sky gate, and the sign labels.

`SgSource` wraps experiment 716's `PvSource` (which wraps experiment 713's `Source`). The wrapped source
holds the target power (`power_z`). `SgSource` adds three (16, G) arrays on the same device:

- `plain_z`, the history that an arm reads by default (folded readings at the hidden pair and the
  MVA-only Primaries, and either the signed power or the folded reading at the visible signed Primaries);
- `fold_z`, the history with every Primary folded, which a sample of a supervised Primary reads when
its
  history rows are folded (probability 0.5 in training);
- `u_grid` and `sign_grid`, the shortwave likelihood at each grid stamp and the true sign (1 export,
  0 import, NaN unknown) of the supervised Primaries.

`assemble` calls the wrapped `PvSource.assemble` once with `power_in=plain_z`, then swaps in `fold_z`
for the samples whose history is folded. The two arrays share their NaN pattern, so the visibility
mask of 713's 27-hour rule is the same for both.
"""

from dataclasses import dataclass

import e00721_paths  # noqa: F401

# isort: split
import e00713_batch as bt  # ty: ignore[unresolved-import]
import e00713_model as mdl  # ty: ignore[unresolved-import]
import e00716_batch as pvb  # ty: ignore[unresolved-import]
import e00721_arms as arms
import numpy as np
import torch  # ty: ignore[unresolved-import]


@dataclass
class SgAssembled:
    """One assembled batch with the sign-head tensors.

    Attributes:
        inputs: The network's inputs.
        target: Standardised target, NaN where the token is not a target.
        target_idx: Grid index of each target token.
        hist_idx: Grid index of each history token.
        c_idx: The contract origin index of each sample.
        hist_sign: `(B, 14, 48)` 1 where the folded history token was an export, 0 an import, NaN unknown.
        target_u: `(B, 48)` the shortwave likelihood `u` at each target token (the page's `u < u_min` rule).
    """

    inputs: arms.SgInputs
    target: torch.Tensor
    target_idx: torch.Tensor
    hist_idx: torch.Tensor
    c_idx: torch.Tensor
    hist_sign: torch.Tensor
    target_u: torch.Tensor


def with_world_target(*, source: bt.Source, power_z: torch.Tensor) -> bt.Source:
    """Return experiment 713's source with the world's standardised power as its target and its history.

    The source that experiment 713's pipeline builds holds 713's own cleaned power, which at the
    hidden pair is the real signed reading. Every array of the world (the injection, the folding,
    the gap rows) is therefore swapped in here, and nothing else in the source changes.

    Args:
        source: 713's source (weather, date-time, day types, and ensemble table are kept).
        power_z: `(16, G)` the world's standardised target, on 713's grid, NaN where absent.

    Returns:
        A source with the same weather, date-time, day-type, and ensemble arrays and the new target.
    """
    assert power_z.shape == source.power_z.shape, (power_z.shape, source.power_z.shape)
    return bt.Source(
        power_z=power_z, dense_weather=source.dense_weather, dense_datetime=source.dense_datetime,
        day_type=source.day_type, ens=source.ens,
    )  # fmt: skip


def pad_to_grid(*, a: np.ndarray, n_grid: int) -> np.ndarray:
    """Return `a` with its last axis extended with NaN to `n_grid` steps (experiment 87's grid is shorter than 713's).

    Both grids start at 2019-09-01 00:00 with 30-minute steps, so index `i` is the same half-hour in both.
    """
    n = a.shape[-1]
    assert n <= n_grid, (n, n_grid)
    out = np.full((*a.shape[:-1], n_grid), np.nan, dtype=np.float64)
    out[..., :n] = a
    return out


class SgSource:
    """A `PvSource` plus the folded and signed history arrays of one arm."""

    def __init__(
        self, *, pv: pvb.PvSource, plain_z: torch.Tensor, fold_z: torch.Tensor, u_grid: torch.Tensor,
        sign_grid: torch.Tensor, always_folded: np.ndarray, supervised: np.ndarray,
    ) -> None:  # fmt: skip
        """Hold the wrapped source and the arrays.

        Args:
            pv: The wrapped source, whose `power_z` is the target.
            plain_z: `(16, G)` the history that the arm reads by default, in standardised units.
            fold_z: `(16, G)` the history with every Primary folded.
            u_grid: `(16, G)` the shortwave likelihood.
            sign_grid: `(16, G)` the true sign, NaN where unknown.
            always_folded: `(16,)` bool, the Primaries whose default history is a folded reading.
            supervised: `(16,)` bool, the Primaries whose sign labels train the head.
        """
        self.pv, self.plain_z, self.fold_z = pv, plain_z, fold_z
        self.u_grid, self.sign_grid = u_grid, sign_grid
        dev = pv.device
        self.always_folded = torch.as_tensor(
            np.asarray(always_folded), dtype=torch.bool, device=dev
        )
        self.supervised = torch.as_tensor(np.asarray(supervised), dtype=torch.bool, device=dev)
        self.device = dev

    def with_hist(self, *, plain_z: torch.Tensor, fold_z: torch.Tensor) -> SgSource:
        """Return the source with other history arrays (stage 3's synthetic shifts)."""
        return SgSource(
            pv=self.pv, plain_z=plain_z, fold_z=fold_z, u_grid=self.u_grid, sign_grid=self.sign_grid,
            always_folded=self.always_folded.cpu().numpy(), supervised=self.supervised.cpu().numpy(),
        )  # fmt: skip

    @property
    def ens(self):  # noqa: ANN201
        """Return the wrapped source's ensemble table."""
        return self.pv.ens

    @property
    def steps(self) -> torch.Tensor:
        """Return the 48 column indices."""
        return self.pv.steps

    @property
    def power_z(self) -> torch.Tensor:
        """Return the standardised target power."""
        return self.pv.power_z

    def assemble(
        self, *, primary: torch.Tensor, c_idx: torch.Tensor, lead_day: torch.Tensor,
        fold_hist: torch.Tensor | None = None, **kwargs,  # noqa: ANN003
    ) -> SgAssembled:  # fmt: skip
        """Return the batch.

        `fold_hist` is a `(B,)` bool of the samples whose history is folded (supervised Primaries only).
        """
        out = self.pv.assemble(
            primary=primary, c_idx=c_idx, lead_day=lead_day, power_in=self.plain_z, **kwargs
        )
        inp = out.inputs
        p3 = primary[:, None, None]
        hist_power = inp.hist_power
        folded_sample = self.always_folded[primary]
        if fold_hist is not None:
            fold_hist = fold_hist & self.supervised[primary]
            zf = self.fold_z[p3, out.hist_idx].clamp(-mdl.POWER_CLIP, mdl.POWER_CLIP)
            use = fold_hist[:, None, None] & inp.hist_visible
            hist_power = torch.where(use, zf, hist_power)
            folded_sample = folded_sample | fold_hist
        counted = folded_sample[:, None, None] & inp.hist_visible
        hist_u = torch.where(counted, self.u_grid[p3, out.hist_idx], torch.zeros_like(hist_power))
        labelled = counted & self.supervised[primary][:, None, None]
        hist_sign = torch.where(
            labelled, self.sign_grid[p3, out.hist_idx], torch.full_like(hist_power, float("nan"))
        )
        top = out.target_idx.clamp(0, self.u_grid.shape[1] - 1)
        target_u = self.u_grid[primary[:, None], top]
        values = dict(inp.__dict__)
        values.update(hist_power=hist_power)
        sg = arms.SgInputs(**values, hist_u=hist_u)
        return SgAssembled(
            inputs=sg, target=out.target, target_idx=out.target_idx, hist_idx=out.hist_idx,
            c_idx=out.c_idx, hist_sign=hist_sign, target_u=target_u,
        )  # fmt: skip
