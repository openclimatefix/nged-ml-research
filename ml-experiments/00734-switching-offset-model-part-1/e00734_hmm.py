"""The offset model of experiment 734: a hidden Markov model of a Primary's switched power.

The hidden state of one block is the product of four components: the switched offset `o` (41
levels, in noise units), its move flag `m` (0 settled, 1 moved up, 2 moved down), the permanent
level `l` (21 levels, steps of 2 noise units), and the demand-shaped gain `g` (5 levels). The
transition matrix factorises over `(o, m)`, `l`, and `g`, so one forward step is three small matrix
products over the array `alpha[(o, m), l, g]` of shape (123, 21, 5).

The observation of block `t` is `x[t]`, in megawatts. The predicted mean of the state `(o, l, g)` is
`sigma * (o + l) + g * f[t]`, with a Student-t density of 4 degrees of freedom and scale
`scale[t % 4]`. A NaN `x[t]` carries no emission term.

The joint index of `(o, m)` is `m * n_o + o_index`, so the emission, which does not depend on `m`,
is repeated three times along that axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Literal

import e00734_native as native
import numpy as np

N_O: Final[int] = 41
"""Offset levels, -20 to 20 noise units."""
N_M: Final[int] = 3
"""Move flag: 0 settled, 1 moved up, 2 moved down."""
N_L: Final[int] = 21
"""Permanent levels, -20 to 20 noise units in steps of 2."""
N_G: Final[int] = 5
"""Gain levels, -0.06 to 0.06 of the forecast demand."""
GAIN_LEVELS: Final[np.ndarray] = np.linspace(-0.06, 0.06, N_G)
NU: Final[float] = 4.0
"""Degrees of freedom of the Student-t noise."""
RHO_G: Final[float] = 0.02
"""Probability per block that the gain moves one level (split evenly up and down)."""
STAY_FLOOR: Final[float] = 0.5
"""Smallest stay probability of a settled row after a boost."""
BLOCKS_PER_DAY: Final[int] = 4
_LOG_T_NORM: Final[float] = (
    math.lgamma((NU + 1) / 2) - math.lgamma(NU / 2) - 0.5 * math.log(NU * math.pi)
)


@dataclass(frozen=True)
class Grid:
    """The state grid; the full grid is `FULL_GRID`, and a smaller grid exists for tests only."""

    o_levels: tuple[int, ...] = tuple(range(-20, 21))
    l_levels: tuple[int, ...] = tuple(range(-20, 21, 2))
    g_levels: tuple[float, ...] = tuple(float(g) for g in GAIN_LEVELS)

    @property
    def n_o(self) -> int:
        """Number of offset levels."""
        return len(self.o_levels)

    @property
    def n_l(self) -> int:
        """Number of permanent levels."""
        return len(self.l_levels)

    @property
    def n_g(self) -> int:
        """Number of gain levels."""
        return len(self.g_levels)

    @property
    def n_om(self) -> int:
        """Size of the joint `(o, m)` axis."""
        return N_M * self.n_o

    @property
    def zero(self) -> int:
        """Index of the offset level 0."""
        return self.o_levels.index(0)


FULL_GRID: Final[Grid] = Grid()


@dataclass(frozen=True)
class Priors:
    """Transition priors of one setting (pre-registered grid in the plan)."""

    rho_on: float
    rho_off: float
    rho_cont: float
    kappa: float = 0.0
    ramp_rule: bool = True
    """False prices a continuation at `rho_on` (arm H-nbr-noramp)."""


@dataclass(frozen=True)
class Posterior:
    """Smoothed quantities of one forward-backward pass."""

    o_marg: np.ndarray
    """(N_BLOCKS, n_o) float32 posterior of the offset."""
    l_jump: np.ndarray
    """(N_BLOCKS,) probability that the permanent level changes at the block."""
    up_jump: np.ndarray
    """(N_BLOCKS,) probability that the offset moves up at the block."""
    down_jump: np.ndarray
    """(N_BLOCKS,) probability that the offset moves down at the block."""
    loglik: float
    l_marg: np.ndarray = field(default_factory=lambda: np.zeros((0, N_L), dtype=np.float32))
    """(N_BLOCKS, n_l) float32 posterior of the permanent level."""
    n_floor: int = 0
    """How many blocks had a settled row rescaled by the stay floor."""


def _weights(*, distances: np.ndarray) -> np.ndarray:
    """Weights 1 / |distance| of a jump, zero for a distance of 0."""
    out = np.zeros(len(distances))
    nz = distances != 0
    out[nz] = 1.0 / np.abs(distances[nz])
    return out


@dataclass(frozen=True)
class _Base:
    """Unboosted transition matrix and the parts that a boost rescales."""

    full: np.ndarray
    fresh_up: np.ndarray
    fresh_down: np.ndarray
    level: np.ndarray
    settled: np.ndarray


@lru_cache(maxsize=64)
def _base(*, p: Priors, grid: Grid) -> _Base:  # noqa: PLR0912, PLR0915 - one transition table
    """Build the (n_om, n_om) transition matrix `full[from, to]` for a setting and a grid."""
    n_o, n_om = grid.n_o, grid.n_om
    levels = np.array(grid.o_levels)
    zero = grid.zero
    full = np.zeros((n_om, n_om))
    fresh_up = np.zeros((n_om, n_om))
    fresh_down = np.zeros((n_om, n_om))

    def idx(*, o: int, m: int) -> int:
        return m * n_o + o

    for m in range(N_M):
        for i in range(n_o):
            row = idx(o=i, m=m)
            lv = levels[i]
            settled_row = m == 0 or i == zero
            fresh = m == 0
            moves: dict[int, float] = {}
            if i == zero:
                w = _weights(distances=levels)
                w[zero] = 0.0
                w = w / w.sum()
                for k in range(n_o):
                    if k != zero:
                        moves[idx(o=k, m=1 if levels[k] > 0 else 2)] = p.rho_on * w[k]
                stay = idx(o=zero, m=0)
            else:
                dist = levels - lv
                w = _weights(distances=dist)
                w[zero] = 0.0
                w[i] = 0.0
                tot = w.sum()
                up_targets = [j for j in range(n_o) if j != i and levels[j] > lv]
                down_targets = [j for j in range(n_o) if j != i and levels[j] < lv]
                if settled_row:
                    if tot > 0:
                        for j in range(n_o):
                            if w[j] > 0:
                                new_m = 1 if levels[j] > lv else 2
                                moves[idx(o=j, m=new_m)] = p.rho_on * w[j] / tot
                    moves[idx(o=zero, m=0)] = p.rho_off
                else:
                    cont_dir = up_targets if m == 1 else down_targets
                    other_dir = down_targets if m == 1 else up_targets
                    cont_w = np.array([1.0 / abs(levels[j] - lv) for j in cont_dir])
                    cont_total = p.rho_cont if p.ramp_rule else p.rho_on
                    for j, cw in zip(
                        cont_dir, cont_w / cont_w.sum() if len(cont_dir) else [], strict=False
                    ):
                        target_m = m if j != zero else 0
                        moves[idx(o=j, m=target_m)] = cont_total * cw
                    for j in other_dir:
                        if j == zero:
                            moves[idx(o=zero, m=0)] = p.rho_off
                        elif tot > 0:
                            moves[idx(o=j, m=2 if m == 1 else 1)] = p.rho_on * w[j] / tot
                stay = idx(o=i, m=0)
            for target, prob in moves.items():
                full[row, target] += prob
                if fresh:
                    to_level = levels[target % n_o]
                    if to_level > lv:
                        fresh_up[row, target] += prob
                    elif to_level < lv:
                        fresh_down[row, target] += prob
            full[row, stay] += 1.0 - sum(moves.values())
    level = np.tile(levels, N_M)
    settled = np.zeros(n_om, dtype=bool)
    settled[:n_o] = True
    return _Base(full=full, fresh_up=fresh_up, fresh_down=fresh_down, level=level, settled=settled)


@lru_cache(maxsize=256)
def _boosted(*, p: Priors, grid: Grid, bu: float, bd: float) -> tuple[np.ndarray, bool]:
    """Transition matrix with fresh up-moves times `bu` and fresh down-moves times `bd`."""
    base = _base(p=p, grid=grid)
    if bu == 1.0 and bd == 1.0:
        return base.full, False
    t = base.full + (bu - 1.0) * base.fresh_up + (bd - 1.0) * base.fresh_down
    floored = False
    for r in np.flatnonzero(base.settled):
        off = t[r].copy()
        off[r] = 0.0
        moves = off.sum()
        if moves > 1.0 - STAY_FLOOR:
            t[r] = off * ((1.0 - STAY_FLOOR) / moves)
            t[r, r] = STAY_FLOOR
            floored = True
        else:
            t[r, r] = 1.0 - moves
    return t, floored


def offset_transition(
    *, p: Priors, boost_up: float = 1.0, boost_down: float = 1.0, grid: Grid = FULL_GRID
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint `(o, m)` transition matrix, split by what the offset does.

    A boost multiplies the probabilities of fresh moves, that is, moves out of a settled row
    (`m = 0`), and never a continuation. The stay probability absorbs the change. A settled row
    whose stay probability would fall under `STAY_FLOOR` has its moves rescaled to leave exactly
    that stay probability.

    Args:
        p: The transition priors.
        boost_up: Multiplier of every fresh move to a higher offset level.
        boost_down: Multiplier of every fresh move to a lower offset level.
        grid: The state grid.

    Returns:
        `(T_stay, T_up, T_down)`, each `(n_om, n_om)`, with `T_stay` holding the transitions that
        keep the offset level, `T_up` those to a higher level, and `T_down` those to a lower
        level. The three add up to the full transition matrix, whose rows sum to 1.
    """
    t, _ = _boosted(p=p, grid=grid, bu=round(boost_up, 9), bd=round(boost_down, 9))
    level = _base(p=p, grid=grid).level
    to_level = level[None, :]
    from_level = level[:, None]
    up = to_level > from_level
    down = to_level < from_level
    return np.where(~up & ~down, t, 0.0), np.where(up, t, 0.0), np.where(down, t, 0.0)


def level_transition(*, rho_l: float, grid: Grid = FULL_GRID) -> np.ndarray:
    """Permanent-level transition: a jump with probability `rho_l`, split by 1 / |j - i|."""
    n = grid.n_l
    out = np.zeros((n, n))
    for i in range(n):
        w = _weights(distances=np.arange(n) - i)
        if w.sum() > 0:
            out[i] = rho_l * w / w.sum()
        out[i, i] = 1.0 - out[i].sum()
    return out


def gain_transition(*, grid: Grid = FULL_GRID) -> np.ndarray:
    """Gain transition: one level up or down with probability `RHO_G / 2` each, reflecting."""
    n = grid.n_g
    out = np.zeros((n, n))
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                out[i, j] = RHO_G / 2
        out[i, i] = 1.0 - out[i].sum()
    return out


def log_emission(
    *, x: float, f: float, sigma: float, scale: float, grid: Grid = FULL_GRID
) -> np.ndarray:
    """Student-t log density of `x` in every state `(o, l, g)`, shape `(n_o, n_l, n_g)`.

    Args:
        x: The block observation in megawatts (NaN for a missing block).
        f: The forecast demand of the block in megawatts.
        sigma: The noise unit in megawatts.
        scale: The Student-t scale of the block of the day, in megawatts.
        grid: The state grid.

    Returns:
        The log density, zero everywhere where `x` is NaN.
    """
    shape = (grid.n_o, grid.n_l, grid.n_g)
    if not math.isfinite(x):
        return np.zeros(shape)
    o = np.array(grid.o_levels, dtype=float)[:, None, None]
    lvl = np.array(grid.l_levels, dtype=float)[None, :, None]
    g = np.array(grid.g_levels)[None, None, :]
    z = (x - sigma * (o + lvl) - g * f) / scale
    return _LOG_T_NORM - math.log(scale) - 0.5 * (NU + 1) * np.log1p(z * z / NU)


def _emission(
    *, x: np.ndarray, f: np.ndarray, sigma: float, scale: np.ndarray, t: int, grid: Grid
) -> tuple[np.ndarray, float]:
    """Emission of block `t` on the `(o, m)` axis, scaled to a maximum of 1, and the log maximum."""
    le = log_emission(
        x=float(x[t]),
        f=float(f[t]) if math.isfinite(f[t]) else 0.0,
        sigma=sigma,
        scale=float(scale[t % BLOCKS_PER_DAY]),
        grid=grid,
    )
    top = float(le.max())
    e = np.exp(le - top)
    return np.concatenate([e] * N_M, axis=0), top


def _initial(*, grid: Grid) -> np.ndarray:
    """Initial `alpha`: o = 0, m = 0, g in the middle level, l uniform."""
    a = np.zeros((grid.n_om, grid.n_l, grid.n_g))
    a[grid.zero, :, grid.n_g // 2] = 1.0 / grid.n_l
    return a


def _step_forward(
    *, a: np.ndarray, t_om: np.ndarray, lmat: np.ndarray, gmat: np.ndarray
) -> np.ndarray:
    """Apply the three factors of the transition to `alpha` (forward direction)."""
    a = np.tensordot(t_om.T, a, axes=(1, 0))
    a = np.swapaxes(np.tensordot(a, lmat, axes=(1, 0)), 1, 2)
    return a @ gmat


def _boost_at(*, boost: np.ndarray | None, t: int) -> float:
    return 1.0 if boost is None else float(boost[t])


def _matrix_at(
    *,
    p: Priors,
    grid: Grid,
    boost_up: np.ndarray | None,
    boost_down: np.ndarray | None,
    t: int,
) -> tuple[np.ndarray, bool]:
    """The `(o, m)` transition into block `t`, rebuilt from the boosts rather than stored.

    The forward and the backward pass both call this, so no per-block matrix is held in memory.
    """
    bu, bd = _boost_at(boost=boost_up, t=t), _boost_at(boost=boost_down, t=t)
    return _boosted(p=p, grid=grid, bu=round(bu, 9), bd=round(bd, 9))


def forward_backward(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None = None,
    boost_down: np.ndarray | None = None,
    grid: Grid = FULL_GRID,
    alpha_dtype: type = np.float32,
) -> Posterior:
    """Exact forward-backward inference over the whole record of one Primary.

    The forward messages are scaled probabilities, stored in 32-bit floats. Each block's emission
    is divided by its maximum before the exponential, and the log of that maximum is added back to
    the log likelihood, so a large residual cannot underflow every state to zero.

    Args:
        x: Block observations in megawatts (NaN where missing or flagged unknown), `(n_blocks,)`.
        f: Forecast demand block means in megawatts.
        sigma: The noise unit of the input in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.
        boost_up: Per-block multiplier of fresh up-moves into the block (None for 1).
        boost_down: Per-block multiplier of fresh down-moves into the block (None for 1).
        grid: The state grid.
        alpha_dtype: Storage type of the forward messages (64-bit only in the exactness tests).

    Returns:
        The `Posterior` with the offset and level marginals and the jump probabilities.
    """
    n = len(x)
    lmat = level_transition(rho_l=p.rho_on, grid=grid)
    gmat = gain_transition(grid=grid)
    lmat_off = lmat - np.diag(np.diag(lmat))
    level = _base(p=p, grid=grid).level
    up_mask = level[None, :] > level[:, None]
    down_mask = level[None, :] < level[:, None]
    alpha = np.zeros((n, grid.n_om, grid.n_l, grid.n_g), dtype=alpha_dtype)
    norm = np.zeros(n)
    loglik = 0.0
    n_floor = 0
    for t in range(n):
        e, top = _emission(x=x, f=f, sigma=sigma, scale=scale, t=t, grid=grid)
        if t == 0:
            a = _initial(grid=grid) * e
        else:
            t_om, floored = _matrix_at(
                p=p, grid=grid, boost_up=boost_up, boost_down=boost_down, t=t
            )
            n_floor += int(floored)
            a = (
                _step_forward(a=alpha[t - 1].astype(np.float64), t_om=t_om, lmat=lmat, gmat=gmat)
                * e
            )
        c = float(a.sum())
        norm[t] = c
        loglik += math.log(c) + top
        alpha[t] = (a / c).astype(alpha_dtype)
    o_marg = np.zeros((n, grid.n_o), dtype=np.float32)
    l_marg = np.zeros((n, grid.n_l), dtype=np.float32)
    l_jump = np.zeros(n)
    up_jump = np.zeros(n)
    down_jump = np.zeros(n)
    beta = np.ones((grid.n_om, grid.n_l, grid.n_g))
    for t in range(n - 1, -1, -1):
        a_t = alpha[t].astype(np.float64)
        gamma = a_t * beta
        gamma /= gamma.sum()
        by_om = gamma.sum(axis=(1, 2))
        o_marg[t] = by_om.reshape(N_M, grid.n_o).sum(axis=0)
        l_marg[t] = gamma.sum(axis=(0, 2))
        e, _ = _emission(x=x, f=f, sigma=sigma, scale=scale, t=t, grid=grid)
        w = e * beta / norm[t]
        # w2[(o, m), k, l] = sum over (k', l') of L[k, k'] G[l, l'] w[(o, m), k', l']
        w2 = np.tensordot(lmat, w, axes=(1, 1)).transpose(1, 0, 2) @ gmat.T
        w3 = np.tensordot(lmat_off, w, axes=(1, 1)).transpose(1, 0, 2) @ gmat.T
        if t == 0:
            break
        prev = alpha[t - 1].astype(np.float64)
        t_om, _ = _matrix_at(p=p, grid=grid, boost_up=boost_up, boost_down=boost_down, t=t)
        up_jump[t] = float(
            (prev * ((t_om * up_mask) @ w2.reshape(grid.n_om, -1)).reshape(w2.shape)).sum()
        )
        down_jump[t] = float(
            (prev * ((t_om * down_mask) @ w2.reshape(grid.n_om, -1)).reshape(w2.shape)).sum()
        )
        l_jump[t] = float((prev * (t_om @ w3.reshape(grid.n_om, -1)).reshape(w3.shape)).sum())
        beta = (t_om @ w2.reshape(grid.n_om, -1)).reshape(w2.shape)
    return Posterior(
        o_marg=o_marg,
        l_jump=l_jump,
        up_jump=up_jump,
        down_jump=down_jump,
        loglik=loglik,
        l_marg=l_marg,
        n_floor=n_floor,
    )


def _log(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(a)


def viterbi_reference(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None = None,
    boost_down: np.ndarray | None = None,
    grid: Grid = FULL_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The single most probable path in float64, one full-size temporary per step (the reference).

    The max-product step factorises like the sum-product step: the maximum over the `(o, m)`
    axis, then over the permanent level, then over the gain, each storing its argmax.

    Args:
        x: Block observations in megawatts (NaN where missing or flagged unknown).
        f: Forecast demand block means in megawatts.
        sigma: The noise unit of the input in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.
        boost_up: Per-block multiplier of fresh up-moves into the block (None for 1).
        boost_down: Per-block multiplier of fresh down-moves into the block (None for 1).
        grid: The state grid.

    Returns:
        The path of the offset level (noise units), of the permanent level (noise units), and of
        the gain index relative to the middle level, each `(n_blocks,)` integers.
    """
    n = len(x)
    log_l = _log(level_transition(rho_l=p.rho_on, grid=grid))
    log_g = _log(gain_transition(grid=grid))
    shape = (n, grid.n_om, grid.n_l, grid.n_g)
    bp_om = np.zeros(shape, dtype=np.int8)
    bp_l = np.zeros(shape, dtype=np.int8)
    bp_g = np.zeros(shape, dtype=np.int8)
    delta = _log(_initial(grid=grid))
    for t in range(n):
        le = log_emission(
            x=float(x[t]),
            f=float(f[t]) if math.isfinite(f[t]) else 0.0,
            sigma=sigma,
            scale=float(scale[t % BLOCKS_PER_DAY]),
            grid=grid,
        )
        emit = np.concatenate([le] * N_M, axis=0)
        if t > 0:
            bu, bd = _boost_at(boost=boost_up, t=t), _boost_at(boost=boost_down, t=t)
            t_om, _ = _boosted(p=p, grid=grid, bu=round(bu, 9), bd=round(bd, 9))
            cand = delta[:, None, :, :] + _log(t_om)[:, :, None, None]
            bp_om[t] = np.argmax(cand, axis=0)
            d1 = np.max(cand, axis=0)
            cand = d1[:, :, None, :] + log_l[None, :, :, None]
            bp_l[t] = np.argmax(cand, axis=1)
            d2 = np.max(cand, axis=1)
            cand = d2[:, :, :, None] + log_g[None, None, :, :]
            bp_g[t] = np.argmax(cand, axis=2)
            delta = np.max(cand, axis=2)
        delta = delta + emit
    o_path = np.zeros(n, dtype=int)
    l_path = np.zeros(n, dtype=int)
    g_path = np.zeros(n, dtype=int)
    i, k, g = np.unravel_index(int(np.argmax(delta)), delta.shape)
    for t in range(n - 1, -1, -1):
        o_path[t] = grid.o_levels[int(i) % grid.n_o]
        l_path[t] = grid.l_levels[int(k)]
        g_path[t] = int(g) - grid.n_g // 2
        if t == 0:
            break
        g_prev = int(bp_g[t, i, k, g])
        k_prev = int(bp_l[t, i, k, g_prev])
        i_prev = int(bp_om[t, i, k_prev, g_prev])
        i, k, g = i_prev, k_prev, g_prev
    return o_path, l_path, g_path


@lru_cache(maxsize=256)
def _log_transition32(*, p: Priors, grid: Grid, bu: float, bd: float) -> np.ndarray:
    """Transposed log of the boosted `(o, m)` matrix in float32, contiguous, cached per boost pair.

    Entry `[j, i]` is the log probability of moving from row `i` to row `j`, so that the maximum
    over the source row runs along the last, contiguous axis.
    """
    t_om, _ = _boosted(p=p, grid=grid, bu=bu, bd=bd)
    return np.ascontiguousarray(_log(t_om).T.astype(np.float32))


VITERBI_CHUNK: Final[int] = 21
"""Columns of the flattened `(l, g)` axis that one step of the Viterbi pass handles at a time."""


_Tables = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""The final scores and the three backpointer arrays of a Viterbi pass."""


def viterbi_numpy(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None = None,
    boost_down: np.ndarray | None = None,
    grid: Grid = FULL_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The single most probable path, in log space, with `int8` backpointers.

    The max-product step factorises like the sum-product step: the maximum over the `(o, m)`
    axis, then over the permanent level, then over the gain, each storing its argmax. The scores
    are held in float32 and shifted to a maximum of 0 at every block, so they stay small and
    float32 resolves the gaps between paths. The `(o, m)` step works on chunks of columns, so that
    its temporary array fits in the processor's cache, and reduces along a contiguous axis. The
    tests compare the decoded paths with those of `viterbi_reference` (float64, no chunks).

    Args:
        x: Block observations in megawatts (NaN where missing or flagged unknown).
        f: Forecast demand block means in megawatts.
        sigma: The noise unit of the input in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.
        boost_up: Per-block multiplier of fresh up-moves into the block (None for 1).
        boost_down: Per-block multiplier of fresh down-moves into the block (None for 1).
        grid: The state grid.

    Returns:
        The path of the offset level (noise units), of the permanent level (noise units), and of
        the gain index relative to the middle level, each `(n_blocks,)` integers.
    """
    delta, bp_om, bp_l, bp_g = _numpy_tables(
        x=x, f=f, sigma=sigma, scale=scale, p=p, boost_up=boost_up, boost_down=boost_down, grid=grid
    )
    return _backtrack(delta=delta, bp_om=bp_om, bp_l=bp_l, bp_g=bp_g, grid=grid)


def _numpy_tables(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None,
    boost_down: np.ndarray | None,
    grid: Grid,
) -> _Tables:
    """The final scores and the backpointers of `viterbi_numpy`, before the backtrack."""
    n = len(x)
    n_om, n_l, n_g = grid.n_om, grid.n_l, grid.n_g
    log_l_t = np.ascontiguousarray(_log(level_transition(rho_l=p.rho_on, grid=grid)).T)
    log_g_t = np.ascontiguousarray(_log(gain_transition(grid=grid)).T)
    log_l_t, log_g_t = log_l_t.astype(np.float32), log_g_t.astype(np.float32)
    bp_om = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    bp_l = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    bp_g = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    delta = _log(_initial(grid=grid)).astype(np.float32)
    cols = n_l * n_g
    for t in range(n):
        le = log_emission(
            x=float(x[t]),
            f=float(f[t]) if math.isfinite(f[t]) else 0.0,
            sigma=sigma,
            scale=float(scale[t % BLOCKS_PER_DAY]),
            grid=grid,
        )
        if t > 0:
            bu, bd = _boost_at(boost=boost_up, t=t), _boost_at(boost=boost_down, t=t)
            log_t = _log_transition32(p=p, grid=grid, bu=round(bu, 9), bd=round(bd, 9))
            flat = np.ascontiguousarray(delta.reshape(n_om, cols).T)
            d1 = np.empty((n_om, cols), dtype=np.float32)
            bp_flat = bp_om[t].reshape(n_om, cols)
            for c0 in range(0, cols, VITERBI_CHUNK):
                cs = slice(c0, c0 + VITERBI_CHUNK)
                cand = flat[cs, None, :] + log_t[None, :, :]
                best = np.argmax(cand, axis=2)
                bp_flat[:, cs] = best.T
                d1[:, cs] = np.take_along_axis(cand, best[..., None], axis=2)[..., 0].T
            d1 = np.ascontiguousarray(d1.reshape(n_om, n_l, n_g).transpose(0, 2, 1))
            cand = d1[:, :, None, :] + log_l_t[None, None, :, :]
            best = np.argmax(cand, axis=3)
            bp_l[t] = best.transpose(0, 2, 1)
            d2 = np.take_along_axis(cand, best[..., None], axis=3)[..., 0]
            d2 = np.ascontiguousarray(d2.transpose(0, 2, 1))
            cand = d2[:, :, None, :] + log_g_t[None, None, :, :]
            best = np.argmax(cand, axis=3)
            bp_g[t] = best
            delta = np.take_along_axis(cand, best[..., None], axis=3)[..., 0]
        delta = delta + np.concatenate([le] * N_M, axis=0).astype(np.float32)
        delta -= delta.max()
    return delta, bp_om, bp_l, bp_g


def _backtrack(
    *, delta: np.ndarray, bp_om: np.ndarray, bp_l: np.ndarray, bp_g: np.ndarray, grid: Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow the backpointers from the best final state to the paths of the three components."""
    n = len(bp_om)
    o_path = np.zeros(n, dtype=int)
    l_path = np.zeros(n, dtype=int)
    g_path = np.zeros(n, dtype=int)
    i, k, g = np.unravel_index(int(np.argmax(delta)), delta.shape)
    for t in range(n - 1, -1, -1):
        o_path[t] = grid.o_levels[int(i) % grid.n_o]
        l_path[t] = grid.l_levels[int(k)]
        g_path[t] = int(g) - grid.n_g // 2
        if t == 0:
            break
        g_prev = int(bp_g[t, i, k, g])
        k_prev = int(bp_l[t, i, k, g_prev])
        i_prev = int(bp_om[t, i, k_prev, g_prev])
        i, k, g = i_prev, k_prev, g_prev
    return o_path, l_path, g_path


VITERBI_NATIVE_CHUNK: Final[int] = 256
"""Blocks per call of the C kernel, so that the emission buffer stays small."""


def viterbi_native(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None = None,
    boost_down: np.ndarray | None = None,
    grid: Grid = FULL_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """`viterbi` with the three max-product steps in C, or `None` when the kernel is unavailable.

    The arithmetic is the NumPy path's: float32 additions, the first maximum wins, and the
    emission is the same NumPy expression. Only the loops over source states are compiled.

    Args:
        x: Block observations in megawatts (NaN where missing or flagged unknown).
        f: Forecast demand block means in megawatts.
        sigma: The noise unit in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.
        boost_up: Per-block multiplier of fresh up-moves into the block (None for 1).
        boost_down: Per-block multiplier of fresh down-moves into the block (None for 1).
        grid: The state grid (the kernel is built for `FULL_GRID`).

    Returns:
        The paths as `viterbi` returns them, or `None` when the library or the grid does not fit.
    """
    tables = _native_tables(
        x=x, f=f, sigma=sigma, scale=scale, p=p, boost_up=boost_up, boost_down=boost_down, grid=grid
    )
    if tables is None:
        return None
    delta, bp_om, bp_l, bp_g = tables
    return _backtrack(delta=delta, bp_om=bp_om, bp_l=bp_l, bp_g=bp_g, grid=grid)


def _native_tables(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None,
    boost_down: np.ndarray | None,
    grid: Grid,
) -> _Tables | None:
    """The final scores and the backpointers of `viterbi_native`, or `None` without the kernel."""
    lib = native.load()
    if lib is None or (grid.n_om, grid.n_l, grid.n_g, grid.n_o) != (123, 21, 5, 41):
        return None
    n = len(x)
    n_om, n_l, n_g = grid.n_om, grid.n_l, grid.n_g
    log_l_t = np.ascontiguousarray(_log(level_transition(rho_l=p.rho_on, grid=grid)).T)
    log_g_t = np.ascontiguousarray(_log(gain_transition(grid=grid)).T)
    log_l_t, log_g_t = log_l_t.astype(np.float32), log_g_t.astype(np.float32)
    bp_om = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    bp_l = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    bp_g = np.zeros((n, n_om, n_l, n_g), dtype=np.int8)
    delta = np.ascontiguousarray(_log(_initial(grid=grid)).astype(np.float32))
    for c0 in range(0, n, VITERBI_NATIVE_CHUNK):
        c1 = min(n, c0 + VITERBI_NATIVE_CHUNK)
        m = c1 - c0
        le = np.empty((m, grid.n_o, n_l, n_g), dtype=np.float32)
        keys: dict[tuple[float, float], int] = {}
        mats: list[np.ndarray] = []
        t_index = np.zeros(m, dtype=np.int32)
        for k, t in enumerate(range(c0, c1)):
            le[k] = log_emission(
                x=float(x[t]),
                f=float(f[t]) if math.isfinite(f[t]) else 0.0,
                sigma=sigma,
                scale=float(scale[t % BLOCKS_PER_DAY]),
                grid=grid,
            )
            if t > 0:
                bu, bd = _boost_at(boost=boost_up, t=t), _boost_at(boost=boost_down, t=t)
                key = (round(bu, 9), round(bd, 9))
                if key not in keys:
                    keys[key] = len(mats)
                    mats.append(_log_transition32(p=p, grid=grid, bu=key[0], bd=key[1]))
                t_index[k] = keys[key]
        if not mats:
            mats.append(np.zeros((n_om, n_om), dtype=np.float32))
        lib.viterbi_chunk(
            m,
            int(c0 > 0),
            np.ascontiguousarray(le),
            np.ascontiguousarray(np.stack(mats)),
            t_index,
            log_l_t,
            log_g_t,
            delta,
            bp_om[c0:c1],
            bp_l[c0:c1],
            bp_g[c0:c1],
        )
    return delta, bp_om, bp_l, bp_g


KernelType = Literal["auto", "c", "numpy"]
KERNELS_USED: Final[set[str]] = set()
"""The Viterbi kernels (`c` or `numpy`) that this process has decoded with, for the run metadata."""


def _check_finite(
    *, sigma: float, scale: np.ndarray, boosts: tuple[np.ndarray | None, ...]
) -> None:
    """Raise unless every input that can turn a score into NaN is finite (and positive for scales).

    A NaN score is resolved differently by `numpy.argmax` (the NaN wins) and by the C kernel (the
    NaN loses), so the two kernels would decode different paths.
    """
    scales = np.append(np.asarray(scale, dtype=float), sigma)
    if not (np.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError(f"sigma and scale must be finite and positive (got {sigma}, {scale})")
    for b in boosts:
        if b is not None and not np.isfinite(b).all():
            raise ValueError("boost_up and boost_down must be finite")


def viterbi(
    *,
    x: np.ndarray,
    f: np.ndarray,
    sigma: float,
    scale: np.ndarray,
    p: Priors,
    boost_up: np.ndarray | None = None,
    boost_down: np.ndarray | None = None,
    grid: Grid = FULL_GRID,
    kernel: KernelType = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The most probable path, by the C kernel when it builds and by NumPy otherwise.

    Both paths give identical decoded paths (the tests compare them with `viterbi_reference`).

    Args:
        x: Block observations in megawatts (NaN where missing or flagged unknown).
        f: Forecast demand block means in megawatts.
        sigma: The noise unit in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.
        boost_up: Per-block multiplier of fresh up-moves into the block (None for 1).
        boost_down: Per-block multiplier of fresh down-moves into the block (None for 1).
        grid: The state grid.
        kernel: `auto` uses C when it builds, `c` raises when it does not, `numpy` never uses C.

    Returns:
        The path of the offset level, of the permanent level, and of the gain index.

    Raises:
        RuntimeError: If `kernel` is `c` and the C kernel is unavailable.
        ValueError: If `sigma`, `scale`, or a boost is not finite, or a scale is not positive.
    """
    _check_finite(sigma=sigma, scale=scale, boosts=(boost_up, boost_down))
    kwargs = {
        "x": x,
        "f": f,
        "sigma": sigma,
        "scale": scale,
        "p": p,
        "boost_up": boost_up,
        "boost_down": boost_down,
        "grid": grid,
    }
    if kernel != "numpy":
        out = viterbi_native(**kwargs)
        if out is not None:
            KERNELS_USED.add("c")
            return out
        if kernel == "c":
            raise RuntimeError("the C Viterbi kernel is unavailable")
    KERNELS_USED.add("numpy")
    return viterbi_numpy(**kwargs)


def mirror_boost(
    *,
    site: str,
    pass1: dict[str, Posterior],
    neighbours: tuple[str, ...],
    unknown: dict[str, np.ndarray],
    kappa: float,
    window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Boosts of a Primary's fresh moves from the pass-1 opposite moves of its neighbours.

    `boost_up[t]` is `1 + kappa` times the largest pass-1 probability, over the neighbours and over
    the blocks within `window` of `t`, that a neighbour's offset moves down; `boost_down[t]` uses
    the neighbours' up-moves. A block that is `unknown` at the neighbour gives no evidence. The
    Primary's own pass 1 is never read.

    Args:
        site: The Primary the boosts are for.
        pass1: The pass-1 posterior of every Primary.
        neighbours: The neighbours of `site`.
        unknown: The `unknown` flag of every Primary, `(n_blocks,)` booleans.
        kappa: The evidence weight.
        window: Blocks either side of `t` that count.

    Returns:
        `(boost_up, boost_down)`, each `(n_blocks,)`.
    """
    n = len(pass1[site].up_jump)
    best_down = np.zeros(n)
    best_up = np.zeros(n)
    for nb in neighbours:
        if nb == site or nb not in pass1:
            continue
        ok = ~unknown[nb]
        dn = np.where(ok, pass1[nb].down_jump, 0.0)
        up = np.where(ok, pass1[nb].up_jump, 0.0)
        for shift in range(-window, window + 1):
            best_down = np.maximum(
                best_down, np.roll(dn, shift) * _valid_after_roll(n=n, shift=shift)
            )
            best_up = np.maximum(best_up, np.roll(up, shift) * _valid_after_roll(n=n, shift=shift))
    return 1.0 + kappa * best_down, 1.0 + kappa * best_up


def _valid_after_roll(*, n: int, shift: int) -> np.ndarray:
    """Zero where `np.roll` wrapped a value round the ends."""
    out = np.ones(n)
    if shift > 0:
        out[:shift] = 0.0
    elif shift < 0:
        out[shift:] = 0.0
    return out


def move_flags(*, o_path: np.ndarray) -> np.ndarray:
    """The move flag `m` implied by an offset path: 1 after an up-move, 2 after a down-move.

    A move to the level 0 is a return and leaves the flag at 0, as in the transition matrix.
    """
    m = np.zeros(len(o_path), dtype=int)
    d = np.diff(o_path)
    m[1:] = np.where((d > 0) & (o_path[1:] != 0), 1, np.where((d < 0) & (o_path[1:] != 0), 2, 0))
    return m


def decode(
    *, x: np.ndarray, f: np.ndarray, sigma: float, scale: np.ndarray, p: Priors
) -> tuple[Posterior, np.ndarray, np.ndarray]:
    """Forward-backward and Viterbi on one series with no neighbour evidence.

    Args:
        x: Block observations in megawatts.
        f: Forecast demand block means in megawatts.
        sigma: The noise unit in megawatts.
        scale: The Student-t scale of each block of the day in megawatts, `(4,)`.
        p: The transition priors.

    Returns:
        The posterior, the offset-level path, and the permanent-level path.
    """
    post = forward_backward(x=x, f=f, sigma=sigma, scale=scale, p=p)
    o_path, l_path, _ = viterbi(x=x, f=f, sigma=sigma, scale=scale, p=p)
    return post, o_path, l_path
