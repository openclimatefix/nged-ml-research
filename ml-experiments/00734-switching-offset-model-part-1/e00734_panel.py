"""Inputs of the offset model of experiment 734: block means, noise units, and the shared term.

The shared term of a Primary at a block is the median, over the other Primaries that have a
reading, of each one's block residual minus its slow level, in that Primary's own noise units. The
loading of the shared term is a Huber fit on 2022 and 2023, and a per-Primary switch turns the
term on only where it lowers the noise unit by 5% or more.

The noise unit that scales the shared term is `sigma_raw`, the unit of the residual before the
shared term is removed. The noise unit of the offset model's grid is `sigma`, the same statistic of
the residual after the term is removed where the switch is on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from e00653_common import BLOCK, N_BLOCKS, SCORED_FROM, block_means  # ty: ignore[unresolved-import]
from e00653_core import freeze  # ty: ignore[unresolved-import]
from e00653_slow import (  # ty: ignore[unresolved-import]
    HUBER_K,
    MASK_UNITS,
    MASK_WIDEN,
    fit_slow,
    rolling_mean,
)

if TYPE_CHECKING:
    from e00660_ev34_detector import Rig  # ty: ignore[unresolved-import]

BLOCKS_PER_DAY: Final[int] = 4
MIN_OTHERS: Final[int] = 8
"""Fewest other Primaries with a reading for the shared term to be used at a block."""
SHARED_ON_GAIN: Final[float] = 0.05
"""The shared term is on at a Primary when it lowers the noise unit by this share or more."""
FOLD_UNITS: Final[float] = 3.0
"""An MVA block mean under this many noise units is reported as unknown."""
TUNE_BLOCKS: Final[int] = SCORED_FROM // BLOCK
"""Blocks of 2022 and 2023, the years every prior is tuned on."""
MIN_BLOCKS: Final[int] = 1000
ArmInput = Literal["cm", "shared"]
SigmaFrom = Literal["tune", "whole"]


@dataclass(frozen=True)
class BlockPanel:
    """Block-level inputs of the offset model for every Primary with a noise unit."""

    sites: tuple[str, ...]
    y: dict[str, np.ndarray]
    """Block means of the input residual, `(N_BLOCKS,)`, NaN where missing."""
    slow: dict[str, np.ndarray]
    """Experiment 653's masked slow level on blocks."""
    shared: dict[str, np.ndarray]
    """The shared term `switch * beta * sigma_raw * c` in megawatts (zeros where it is off)."""
    forecast: dict[str, np.ndarray]
    """Block means of the arm C forecast, megawatts."""
    sigma: dict[str, float]
    """The noise unit of the grid, megawatts."""
    scale: dict[str, np.ndarray]
    """The Student-t scale of each block of the day, `(4,)`, megawatts."""
    is_mva: dict[str, bool]
    unknown: dict[str, np.ndarray]
    """Boolean `(N_BLOCKS,)`: an MVA block mean under `FOLD_UNITS` noise units."""
    neighbours: dict[str, tuple[str, ...]]

    def observation(self, *, site: str) -> np.ndarray:
        """The model's `x`: input minus slow level minus shared term, NaN where unknown."""
        x = self.y[site] - self.slow[site] - self.shared[site]
        return np.where(self.unknown[site], np.nan, x)


@dataclass(frozen=True)
class Calibration:
    """Quantities frozen on the uninjected real record and reused in every world."""

    sigma: dict[str, float]
    sigma_raw: dict[str, float]
    loading: dict[str, float]
    switch: dict[str, bool]
    scale: dict[str, np.ndarray]


def raw_residual(*, rig: Rig, power: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Power minus the lag-free forecast, half-hourly, with no common mode removed."""
    return {s: power[s] - rig.forecast[s] for s in rig.p653.sites}


def common_mode_residual(*, rig: Rig, power: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The raw residual with experiment 653's common mode removed (the input of E1-J-sym)."""
    return rig.p653.mask_common_mode(residual=raw_residual(rig=rig, power=power))


def candidate_mask(*, y: np.ndarray, slow: np.ndarray, sigma: float) -> np.ndarray:
    """Blocks that experiment 653 would treat as event candidates: 3-day mean over 1.5 units."""
    rm = rolling_mean(r=y - slow, window=12)
    flag = np.abs(np.nan_to_num(rm)) > MASK_UNITS * sigma
    grown = flag.copy()
    for k in range(1, MASK_WIDEN + 1):
        grown[k:] |= flag[:-k]
        grown[:-k] |= flag[k:]
    return grown


def _mad_unit(*, r: np.ndarray) -> float:
    """1.4826 times the median absolute deviation of the finite values of `r`."""
    v = r[np.isfinite(r)]
    if len(v) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def shared_term(
    *, y: dict[str, np.ndarray], slow: dict[str, np.ndarray], sigma: dict[str, float]
) -> dict[str, np.ndarray]:
    """Median over the other Primaries of the block residual in noise units, per Primary.

    The Primary itself is left out of its own median. Where fewer than `MIN_OTHERS` other
    Primaries have a reading, or none does, the term is 0.

    Args:
        y: Block means of every Primary's input.
        slow: The slow level of every Primary on blocks.
        sigma: The noise unit of every Primary in megawatts.

    Returns:
        The shared term `c` of every Primary in noise units.
    """
    sites = list(y)
    z = np.stack([(y[s] - slow[s]) / sigma[s] for s in sites])
    out = {}
    for i, s in enumerate(sites):
        others = np.delete(z, i, axis=0)
        n_ok = np.isfinite(others).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(others, axis=0)
        out[s] = np.where(n_ok >= MIN_OTHERS, np.nan_to_num(med), 0.0)
    return out


def fit_loading(
    *, y: np.ndarray, slow: np.ndarray, c: np.ndarray, sigma: float, upto_block: int
) -> float:
    """Huber regression coefficient of `y - slow` on `sigma * c` over blocks before `upto_block`.

    Blocks in experiment 653's event-candidate mask get zero weight.

    Args:
        y: Block means of the input, megawatts.
        slow: The slow level in megawatts.
        c: The shared term in noise units.
        sigma: The noise unit that scales `c`, megawatts.
        upto_block: The fit uses blocks before this index.

    Returns:
        The loading `beta` (0 if too few blocks are usable).
    """
    r = (y - slow)[:upto_block]
    x = (sigma * c)[:upto_block]
    keep = (
        np.isfinite(r) & np.isfinite(x) & ~candidate_mask(y=y, slow=slow, sigma=sigma)[:upto_block]
    )
    if keep.sum() < MIN_BLOCKS // 2 or float(np.sum(x[keep] ** 2)) == 0.0:
        return 0.0
    r, x = r[keep], x[keep]
    beta = float(np.sum(x * r) / np.sum(x * x))
    for _ in range(30):
        resid = r - beta * x
        scale = max(_mad_unit(r=resid), 1e-9)
        w = np.minimum(1.0, HUBER_K * scale / np.maximum(np.abs(resid), 1e-12))
        beta = float(np.sum(w * x * r) / np.sum(w * x * x))
    return beta


def shared_switch(
    *,
    y: dict[str, np.ndarray],
    slow: dict[str, np.ndarray],
    c: dict[str, np.ndarray],
    sigma: dict[str, float],
    loading: dict[str, float],
    upto_block: int,
) -> dict[str, bool]:
    """Per Primary, whether the shared term lowers the noise unit by `SHARED_ON_GAIN` or more.

    The noise unit is 1.4826 times the median absolute deviation of `y - slow - beta * sigma * c`
    over blocks before `upto_block`.
    """
    out = {}
    for s in y:
        r = (y[s] - slow[s])[:upto_block]
        after = r - loading[s] * (sigma[s] * c[s])[:upto_block]
        before_unit, after_unit = _mad_unit(r=r), _mad_unit(r=after)
        out[s] = bool(
            np.isfinite(after_unit) and after_unit <= (1.0 - SHARED_ON_GAIN) * before_unit
        )
    return out


def block_scale(*, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Student-t scale per block of the day: 1.4826 times the MAD of `r` outside `mask`."""
    ok = np.where(mask, np.nan, r)
    out = np.zeros(BLOCKS_PER_DAY)
    for b in range(BLOCKS_PER_DAY):
        out[b] = _mad_unit(r=ok[b::BLOCKS_PER_DAY])
    return np.where(
        np.isfinite(out) & (out > 0), out, np.nanmedian(out[out > 0]) if (out > 0).any() else 1.0
    )


def _blocks(*, resid: dict[str, np.ndarray], sites: list[str]) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {s: block_means(x=resid[s]) for s in sites}


def calibrate(
    *, rig: Rig, power: dict[str, np.ndarray], arm_input: ArmInput, sigma_from: SigmaFrom
) -> Calibration:
    """Freeze the noise units, loadings, switches, and block-of-day scales of an input.

    Call it on the uninjected real power. With `sigma_from="tune"` the noise units and scales use
    2022 and 2023 only; with `"whole"` they use the whole record, as experiment 680's real run
    does. The loading and the switch always use 2022 and 2023.

    Args:
        rig: Experiment 660's rig.
        power: The uninjected real power of every Primary, half-hourly.
        arm_input: `"cm"` (experiment 653's common mode removed) or `"shared"` (raw residual, the
            fitted shared term removed at the observation).
        sigma_from: Which years the noise units and scales use.

    Returns:
        The frozen quantities, for the Primaries with a noise unit.
    """
    upto = SCORED_FROM if sigma_from == "tune" else None
    upto_block = TUNE_BLOCKS if sigma_from == "tune" else N_BLOCKS
    if arm_input == "cm":
        resid = common_mode_residual(rig=rig, power=power)
        fz = freeze(panel=rig.p653, residual=resid, upto=upto)
        sites = [s for s in rig.p653.sites if np.isfinite(fz.sigma[s])]
        y = _blocks(resid=resid, sites=sites)
        scale = {
            s: block_scale(
                r=(y[s] - fz.slow[s]["masked"])[:upto_block],
                mask=candidate_mask(y=y[s], slow=fz.slow[s]["masked"], sigma=fz.sigma[s])[
                    :upto_block
                ],
            )
            for s in sites
        }
        return Calibration(
            sigma={s: fz.sigma[s] for s in sites},
            sigma_raw={s: fz.sigma[s] for s in sites},
            loading=dict.fromkeys(sites, 0.0),
            switch=dict.fromkeys(sites, False),
            scale=scale,
        )
    raw = raw_residual(rig=rig, power=power)
    fz = freeze(panel=rig.p653, residual=raw, upto=upto)
    sites = [s for s in rig.p653.sites if np.isfinite(fz.sigma[s])]
    y = _blocks(resid=raw, sites=sites)
    slow = {s: fz.slow[s]["masked"] for s in sites}
    sigma_raw = {s: fz.sigma[s] for s in sites}
    c = shared_term(y=y, slow=slow, sigma=sigma_raw)
    loading = {
        s: fit_loading(y=y[s], slow=slow[s], c=c[s], sigma=sigma_raw[s], upto_block=TUNE_BLOCKS)
        for s in sites
    }
    switch = shared_switch(
        y=y, slow=slow, c=c, sigma=sigma_raw, loading=loading, upto_block=TUNE_BLOCKS
    )
    sigma, scale = {}, {}
    for s in sites:
        term = loading[s] * sigma_raw[s] * c[s] if switch[s] else np.zeros(N_BLOCKS)
        r = y[s] - slow[s] - term
        sigma[s] = _mad_unit(r=r[:upto_block])
        mask = candidate_mask(y=y[s], slow=slow[s], sigma=sigma_raw[s])[:upto_block]
        scale[s] = block_scale(r=r[:upto_block], mask=mask)
    return Calibration(
        sigma=sigma, sigma_raw=sigma_raw, loading=loading, switch=switch, scale=scale
    )


def build_blocks(
    *,
    rig: Rig,
    power: dict[str, np.ndarray],
    arm_input: ArmInput,
    loading: dict[str, float] | None,
    switch: dict[str, bool] | None,
    sigma_from: SigmaFrom,
    calibration: Calibration | None = None,
    resid: dict[str, np.ndarray] | None = None,
    n_blocks: int = N_BLOCKS,
) -> BlockPanel:
    """Block means, a refitted slow level, and the shared term of one world or the real record.

    The slow level is refitted on this world's input, as experiment 653's `decode_world` does. The
    noise unit, the loading, the switch, and the scales come from `calibration` (computed here
    from `power` when None, which is right for the uninjected real record only). Explicit
    `loading` and `switch` replace the calibration's.

    Args:
        rig: Experiment 660's rig.
        power: The power of this world, half-hourly.
        arm_input: `"cm"` or `"shared"`, as in `calibrate`.
        loading: Loadings that override the calibration's, or None.
        switch: Switches that override the calibration's, or None.
        sigma_from: Which years the noise units use when `calibration` is None.
        calibration: The frozen quantities of the uninjected real record.
        resid: A half-hourly input residual that replaces the one computed from `power` (the
            solar-growth and seasonal-bias worlds of experiment 653 are built on residuals).
        n_blocks: Cut every array to this many blocks (the tuning worlds hide the scored years).

    Returns:
        The `BlockPanel`.
    """
    cal = calibration or calibrate(rig=rig, power=power, arm_input=arm_input, sigma_from=sigma_from)
    sites = list(cal.sigma)
    if resid is None:
        resid = (
            common_mode_residual(rig=rig, power=power)
            if arm_input == "cm"
            else raw_residual(rig=rig, power=power)
        )
    y = {s: v[:n_blocks] for s, v in _blocks(resid=resid, sites=sites).items()}
    slow = {s: fit_slow(y=y[s], sigma=cal.sigma_raw[s], arm="masked") for s in sites}
    beta = loading or cal.loading
    on = switch or cal.switch
    if arm_input == "shared":
        c = shared_term(y=y, slow=slow, sigma=cal.sigma_raw)
        shared = {
            s: (beta[s] * cal.sigma_raw[s] * c[s] if on[s] else np.zeros(n_blocks)) for s in sites
        }
    else:
        shared = {s: np.zeros(n_blocks) for s in sites}
    obs = {s: v[:n_blocks] for s, v in _blocks(resid=power, sites=sites).items()}
    fc = {s: v[:n_blocks] for s, v in _blocks(resid=rig.forecast, sites=sites).items()}
    mva = {s: bool(rig.panel.is_mva[s]) for s in sites}
    with np.errstate(invalid="ignore"):
        unknown = {s: mva[s] & (obs[s] < FOLD_UNITS * cal.sigma[s]) for s in sites}
    nb = {s: tuple(n for n in rig.p653.neighbours[s] if n in sites) for s in sites}
    return BlockPanel(
        sites=tuple(sites),
        y=y,
        slow=slow,
        shared=shared,
        forecast=fc,
        sigma=dict(cal.sigma),
        scale={s: cal.scale[s] for s in sites},
        is_mva=mva,
        unknown=unknown,
        neighbours=nb,
    )
