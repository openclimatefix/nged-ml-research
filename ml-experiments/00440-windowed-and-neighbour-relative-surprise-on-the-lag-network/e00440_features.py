# ruff: noqa: E501
"""Windowed and neighbour-relative surprise features for the lag network (experiment 440).

For each of a row's three lags (the newest reading, the same time yesterday, the same time last
week), this module computes statistics of the surprise series in a window of 12 hours either side of
the lag's time, cut at the newest usable reading. The surprise is the observed power minus the
lagless median of experiment 404's cross-fitted table, in units of the lagless 5%-95% band
half-width. The statistics are experiment 404's (`e00440_trust`, copied): the mean, the signed sum,
the standard deviation, the minimum, the maximum, the counts above the 95% and below the 5% lagless
quantile, and a triangular-tent mean. (Experiment 404 left out the median, because prefix sums and
sparse tables cannot give it; so does this module.)

- **S2 features** (24 per row): those 8 statistics for each lag.
- **S3 features** (54 per row): the S2 features, plus, per lag, the same 8 statistics averaged over
  the Primary's neighbours (weighted by the neighbour's effective capacity), the lag's surprise minus
  the weighted neighbour surprise at the same time, and the weighted sign agreement between the
  Primary's surprise and each neighbour's.

The sum, above-count, and below-count are divided by the 49 half-hours of the window and every
feature is clipped at +-10 (surprise at the lag itself at +-5).

A step added to the lags (the encroachment sweep) shifts the window's mean, sum, minimum, maximum,
and tent by the step in band units at the lag's time, leaves the standard deviation unchanged, and
leaves the two counts unchanged (an approximation, stated in the write-up).

```bash
uv run --no-sync python ml-experiments/00440-*/e00440_features.py selftest
```
"""

import json
from pathlib import Path
from typing import Final

import e00440_trust as trust  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

PAD: Final[int] = 1300
"""Unobserved half-hours between Primaries, so a window at a Primary's edge never reads another."""
HALF_WIDTH: Final[int] = 24
"""Half-width of the window in half-hours (12 hours)."""
WINDOW_STEPS: Final[float] = 2.0 * HALF_WIDTH + 1.0
FEATURE_CLIP: Final[float] = 10.0
SURPRISE_CLIP: Final[float] = 5.0
N_STATS: Final[int] = len(trust.STATS)
N_OWN_PER_LAG: Final[int] = N_STATS
N_NEIGHBOUR_PER_LAG: Final[int] = N_STATS + 2
N_S2: Final[int] = 3 * N_OWN_PER_LAG
N_S3: Final[int] = N_S2 + 3 * N_NEIGHBOUR_PER_LAG
RANDOM_NEIGHBOUR_SEED: Final[int] = 12_345


def neighbour_matrix(*, sites: list[str], path: Path) -> np.ndarray:
    """Return the (primaries, widest degree) index matrix of the adjacency file, -1 padded."""
    graph = json.loads(path.read_text())
    index = {s: i for i, s in enumerate(sites)}
    rows = [[index[n] for n in graph.get(s, []) if n in index] for s in sites]
    out = np.full((len(sites), max(1, *(len(r) for r in rows))), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def random_neighbours(*, matrix: np.ndarray, seed: int = RANDOM_NEIGHBOUR_SEED) -> np.ndarray:
    """Return a matrix in which each Primary keeps its degree but its neighbours are random others."""
    rng = np.random.default_rng(seed)
    out = np.full_like(matrix, -1)
    for p in range(matrix.shape[0]):
        degree = int((matrix[p] >= 0).sum())
        others = np.array([q for q in range(matrix.shape[0]) if q != p])
        out[p, :degree] = rng.choice(others, size=degree, replace=False)
    return out


class WorldStack:
    """One world's 16 surprise series stacked on a dense grid, in units of the lagless band."""

    def __init__(
        self, *, power: np.ndarray, median: np.ndarray, half: np.ndarray, device: str
    ) -> None:
        """Build the stack from power, lagless median, and band half-width, each (primaries, grid)."""
        primaries, grid = power.shape
        self.block = grid + 2 * PAD
        total = primaries * self.block
        observed = np.isfinite(power) & np.isfinite(median) & np.isfinite(half)
        actual, centre = np.full(total, np.nan), np.full(total, np.nan)
        p, g = np.nonzero(observed)
        pos = p * self.block + PAD + g
        actual[pos] = power[p, g] / half[p, g]
        centre[pos] = median[p, g] / half[p, g]
        self.series = trust.build_series(
            first=0, actual=actual, median=centre, q05=np.full(total, -1.0),
            q95=np.full(total, 1.0), device=device,
        )  # fmt: skip

    def flat(self, *, primary: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Return the index into the stacked series of a Primary's grid half-hour."""
        return primary * self.block + PAD + grid


def clean_stats(*, stats: torch.Tensor) -> torch.Tensor:
    """Scale the counts and the sum by the window length and clip, giving (rows, 8)."""
    out = stats.clone()
    for column in (
        trust.STATS.index("sum"),
        trust.STATS.index("above"),
        trust.STATS.index("below"),
    ):
        out[:, column] = out[:, column] / WINDOW_STEPS
    return out.clamp(-FEATURE_CLIP, FEATURE_CLIP)


def shift_stats(*, stats: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Return `stats` after a step of `shift` (rows,) band units over the whole window."""
    out = stats.clone()
    for name in ("mean", "min", "max", "tent"):
        out[:, trust.STATS.index(name)] += shift
    out[:, trust.STATS.index("sum")] += shift
    return out.clamp(-FEATURE_CLIP, FEATURE_CLIP)


def row_features(
    *, stack: WorldStack, primary: torch.Tensor, lag_grid: torch.Tensor, origin: torch.Tensor,
    shift: torch.Tensor, neighbours: torch.Tensor | None, weights: torch.Tensor,
) -> torch.Tensor:  # fmt: skip
    """Return the S2 features (rows, 24), or the S3 features (rows, 54) if `neighbours` is given.

    Args:
        stack: The world's stacked surprise series.
        primary: Primary index per row.
        lag_grid: Grid index of each lag's time (rows, 3).
        origin: Grid index of each row's forecast origin.
        shift: Step added to each lag, in band units at the lag's time (rows, 3); 0 for none.
        neighbours: (primaries, widest degree) neighbour matrix, -1 padded; None for S2.
        weights: Per-Primary weight of a neighbour (its effective capacity).
    """
    series = stack.series
    end = stack.flat(primary=primary, grid=origin - trust_newest_age())
    own_parts, extra_parts = [], []
    rows = neighbours[primary] if neighbours is not None else None
    for lag in range(3):
        grid = lag_grid[:, lag]
        centre = stack.flat(primary=primary, grid=grid)
        raw = trust.window_statistics(series=series, centre=centre, end=end, half_width=HALF_WIDTH)
        own = shift_stats(stats=clean_stats(stats=raw), shift=shift[:, lag])
        own_parts.append(own)
        if rows is None:
            continue
        z_own = (trust.single_step(series=series, at=centre)[:, 1] + shift[:, lag]).clamp(
            -SURPRISE_CLIP, SURPRISE_CLIP
        )
        total = torch.zeros(len(primary), N_STATS, device=primary.device)
        z_total = torch.zeros(len(primary), device=primary.device)
        agree = torch.zeros_like(z_total)
        weight = torch.zeros_like(z_total)
        for j in range(rows.shape[1]):
            who = rows[:, j]
            present = who >= 0
            safe = who.clamp(min=0)
            w = torch.where(present, weights[safe], torch.zeros_like(z_total))
            nb_centre = stack.flat(primary=safe, grid=grid)
            nb_end = stack.flat(primary=safe, grid=origin - trust_newest_age())
            nb = clean_stats(
                stats=trust.window_statistics(
                    series=series, centre=nb_centre, end=nb_end, half_width=HALF_WIDTH
                )
            )
            z_nb = trust.single_step(series=series, at=nb_centre)[:, 1].clamp(
                -SURPRISE_CLIP, SURPRISE_CLIP
            )
            total += w[:, None] * nb
            z_total += w * z_nb
            agree += w * torch.sign(z_own) * torch.sign(z_nb)
            weight += w
        scale = weight.clamp(min=1e-9)
        has = weight > 0
        relative = torch.where(has, z_own - z_total / scale, torch.zeros_like(z_own))
        extra_parts.append(
            torch.cat(
                [
                    total / scale[:, None],
                    relative.clamp(-SURPRISE_CLIP, SURPRISE_CLIP)[:, None],
                    torch.where(has, agree / scale, torch.zeros_like(agree))[:, None],
                ],
                dim=1,
            )
        )
    return torch.cat([*own_parts, *extra_parts], dim=1)


def trust_newest_age() -> int:
    """Return the half-hours between the origin and the newest usable reading (428's rule)."""
    import e00428_windows as win  # ty: ignore[unresolved-import]

    return win.NEWEST_AGE


def selftest() -> None:
    """Check the features against a brute-force computation on a small synthetic world."""
    import e00428_windows as win  # ty: ignore[unresolved-import]

    rng = np.random.default_rng(0)
    primaries, grid = 3, 3000
    power = rng.normal(size=(primaries, grid)) * 0.1 + 0.5
    median = np.full((primaries, grid), 0.5)
    half = np.full((primaries, grid), 0.1)
    power[0, 1500:1600] += 0.5
    stack = WorldStack(power=power, median=median, half=half, device="cpu")
    neighbours = torch.tensor([[1, 2], [0, -1], [0, -1]])
    weights = torch.tensor([1.0, 2.0, 3.0])
    origin = torch.tensor([2000, 2200, 2500])
    target = origin + 100
    primary = torch.tensor([0, 1, 0])
    lag_grid = win.lag_indices(origin=origin, target=target)
    zero = torch.zeros(3, 3)
    s2 = row_features(stack=stack, primary=primary, lag_grid=lag_grid, origin=origin, shift=zero,
                      neighbours=None, weights=weights)  # fmt: skip
    s3 = row_features(stack=stack, primary=primary, lag_grid=lag_grid, origin=origin, shift=zero,
                      neighbours=neighbours, weights=weights)  # fmt: skip
    assert s2.shape == (3, N_S2), s2.shape
    assert s3.shape == (3, N_S3), s3.shape
    assert torch.allclose(s3[:, :N_S2], s2)
    for r in range(3):
        for lag in range(3):
            c = int(lag_grid[r, lag])
            end = int(origin[r]) - win.NEWEST_AGE
            lo, hi = max(0, c - HALF_WIDTH), min(c + HALF_WIDTH, end)
            z = (power[int(primary[r]), lo : hi + 1] - 0.5) / 0.1
            got = s2[r, lag * N_STATS : (lag + 1) * N_STATS]
            assert abs(float(got[0]) - z.mean()) < 1e-3, ("mean", r, lag)
            assert abs(float(got[3]) - min(z.min(), FEATURE_CLIP)) < 1e-3, ("min", r, lag)
    # Primary 0 has +5 band units on 1500..1599; Primary 1 does not, so a row of Primary 1 at a
    # lag inside that span sees the neighbour mean rise while its own mean stays near 0.
    centre = torch.tensor([1550])
    one = row_features(
        stack=stack, primary=torch.tensor([1]), lag_grid=centre[:, None].repeat(1, 3),
        origin=torch.tensor([2400]), shift=torch.zeros(1, 3), neighbours=neighbours, weights=weights,
    )  # fmt: skip
    assert abs(float(one[0, 0])) < 1.0
    assert float(one[0, N_S2 + 0]) > 3.0, "the neighbour mean must show the +5 step"
    shifted = row_features(stack=stack, primary=primary, lag_grid=lag_grid, origin=origin,
                           shift=torch.full((3, 3), 2.0), neighbours=None, weights=weights)  # fmt: skip
    assert torch.allclose(shifted[:, 0] - s2[:, 0], torch.full((3,), 2.0), atol=1e-5)
    rw = random_neighbours(matrix=neighbours.numpy())
    assert ((rw >= 0).sum(1) == (neighbours.numpy() >= 0).sum(1)).all()
    assert all(p not in rw[p] for p in range(primaries))
    print("selftest passed")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["selftest"]:
        selftest()
