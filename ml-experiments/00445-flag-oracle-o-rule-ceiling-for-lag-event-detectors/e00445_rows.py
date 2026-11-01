"""Rows of experiment 404: lag values, surprises, and trust inputs recomputed for a world.

A world is one array of power over capacity, shape (16, grid), from experiment 428's injected worlds
(or the clean power). For a world and a scored month, `Stack` lays the 16 Primaries' surprise series
(the world's power minus the cross-fitted lagless median) end to end with unobserved padding between
them, so one `DenseSeries` serves every Primary and a window never reads the next Primary. A row's
three lag values come from `origin_lags` on the world's power, and its trust inputs are the window
statistics of `e00445_trust` at the three lags' valid times and at the 2 half-hours before them.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
    uv run --no-sync python ml-experiments/00445-*/e00445_rows.py selftest
```
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00445_trust as trust  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PAD: Final[int] = 1300
"""Unobserved half-hours between Primaries: a lag centre sits up to about 1000 half-hours before the
lagless table starts, and the longest window half-width is 168."""
SMOOTHING_STEPS: Final[int] = 3
N_PER_LAG: Final[int] = len(trust.HALF_WIDTHS) * len(trust.STATS) + 2 + 1
"""Window statistics, the single-step surprise and band score, and the neighbours' mean."""
N_TRUST: Final[int] = 3 * N_PER_LAG
NEIGHBOUR_PATH_PARTS: Final[tuple[str, ...]] = (
    "00383-load-shift-xgboost-oracle-first-part-1",
    "work",
    "neighbours_nged.json",
)


@dataclass(frozen=True)
class Stack:
    """The 16 Primaries' surprise series for one world and one lagless table.

    Attributes:
        series: The stacked `DenseSeries` (its `first` is 0).
        g0: Grid index of the first half-hour of each Primary's block.
        block: Length of each Primary's block, including the padding on both sides.
        neighbours: Row `p` lists the Primary indices adjacent to Primary `p`, -1 padded.
    """

    series: trust.DenseSeries
    g0: int
    block: int
    neighbours: np.ndarray

    def flat(self, *, primary: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Return the index into the stacked series of a Primary's grid half-hour."""
        return primary * self.block + PAD + (grid - self.g0)


def neighbour_matrix(*, sites: list[str], path: str) -> np.ndarray:
    """Return the (primaries, max neighbours) index matrix of the adjacency file, -1 padded."""
    graph = json.loads(Path(path).read_text())
    index = {s: i for i, s in enumerate(sites)}
    rows = [[index[n] for n in graph.get(s, []) if n in index] for s in sites]
    width = max(1, *(len(r) for r in rows))
    out = np.full((len(sites), width), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def build_stack(
    *, power: np.ndarray, table: pl.DataFrame, neighbours: np.ndarray, device: str
) -> Stack:
    """Build the stacked series from a world's power (16, grid) and a lagless table.

    The surprise at a half-hour is the world's power minus the table's median there, and the band
    is the table's q05 to q95. A half-hour with no table row or no power is unobserved.
    """
    g0, g1 = int(table["grid"].min()), int(table["grid"].max())  # ty: ignore[invalid-argument-type]
    n = g1 - g0 + 1
    block = n + 2 * PAD
    total = power.shape[0] * block
    actual = np.full(total, np.nan)
    median, q05, q95 = (np.full(total, np.nan) for _ in range(3))
    primary = table["primary"].to_numpy().astype(np.int64)
    pos = primary * block + PAD + (table["grid"].to_numpy().astype(np.int64) - g0)
    median[pos], q05[pos], q95[pos] = (
        table["q50"].to_numpy(),
        table["q05"].to_numpy(),
        table["q95"].to_numpy(),
    )
    actual[pos] = power[primary, table["grid"].to_numpy().astype(np.int64)]
    series = trust.build_series(
        first=0, actual=actual, median=median, q05=q05, q95=q95, device=device
    )
    return Stack(series=series, g0=g0, block=block, neighbours=neighbours)


def trust_inputs(
    *,
    stack: Stack,
    primary: torch.Tensor,
    lag_grid: torch.Tensor,
    origin: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """Return the trust inputs (rows, N_TRUST) with every lag's window centred `step` earlier.

    Args:
        stack: The world's stacked series.
        primary: Primary index per row.
        lag_grid: Grid index of each lag's valid time, shape (rows, 3).
        origin: Grid index of each row's forecast origin.
        step: Half-hours to move each lag's window centre earlier (0, 1, or 2).

    Returns:
        Per lag: the window statistics at each half-width, the single-step surprise and band
        score, and the mean over the Primary's neighbours of the 12-hour window mean.
    """
    end = stack.flat(primary=primary, grid=origin - common.NEWEST_AGE)
    neighbours = torch.from_numpy(stack.neighbours).to(primary.device)[primary]
    out = []
    for lag in range(3):
        centre = stack.flat(primary=primary, grid=lag_grid[:, lag] - step)
        parts = [
            trust.window_statistics(series=stack.series, centre=centre, end=end, half_width=h)
            for h in trust.HALF_WIDTHS
        ]
        parts.append(trust.single_step(series=stack.series, at=centre))
        total = torch.zeros(len(primary), device=primary.device)
        count = torch.zeros(len(primary), device=primary.device)
        for j in range(neighbours.shape[1]):
            who = neighbours[:, j]
            present = who >= 0
            nb_centre = stack.flat(primary=who.clamp(min=0), grid=lag_grid[:, lag] - step)
            nb_end = stack.flat(primary=who.clamp(min=0), grid=origin - common.NEWEST_AGE)
            mean = trust.window_statistics(
                series=stack.series,
                centre=nb_centre,
                end=nb_end,
                half_width=trust.NEIGHBOUR_WIDTH,
            )[:, 0]
            total += torch.where(present, mean, torch.zeros_like(mean))
            count += present.float()
        parts.append((total / count.clamp(min=1.0))[:, None])
        out.append(torch.cat(parts, dim=1))
    return torch.cat(out, dim=1)


def lag_grids(*, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the grid index of each row's 3 lags (newest, same time yesterday, last week)."""
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    day = target - common.STEPS_PER_DAY * np.ceil(gap / common.STEPS_PER_DAY).astype(np.int64)
    week = target - 7 * common.STEPS_PER_DAY * np.ceil(gap / (7 * common.STEPS_PER_DAY)).astype(
        np.int64
    )
    return np.stack([newest, day, week], axis=1)


def selftest() -> None:
    """Check that a window at one Primary's start does not read another Primary's series."""
    rng = np.random.default_rng(0)
    n, primaries = 3000, 3
    grid = np.arange(n) + 1000
    table = pl.DataFrame(
        {
            "primary": np.repeat(np.arange(primaries), n),
            "grid": np.tile(grid, primaries),
            "q50": np.zeros(n * primaries),
            "q05": np.full(n * primaries, -1.0),
            "q95": np.full(n * primaries, 1.0),
        }
    )
    power = np.zeros((primaries, 5000))
    power[0, 1000 : 1000 + n] = 5.0
    power[1, 1000 : 1000 + n] = 0.0
    power[2, 1000 : 1000 + n] = rng.normal(size=n)
    stack = build_stack(
        power=power, table=table, neighbours=np.array([[1], [0], [0]]), device="cpu"
    )
    rows = 4
    primary = torch.tensor([1, 1, 1, 0])
    origin = torch.tensor([1010, 2000, 3000, 2000])
    target = torch.tensor([1100, 2100, 3100, 2100])
    lags = torch.from_numpy(lag_grids(target=target.numpy(), origin=origin.numpy()))
    got = trust_inputs(stack=stack, primary=primary, lag_grid=lags, origin=origin, step=0)
    assert got.shape == (rows, N_TRUST), got.shape
    first_row = got[0].reshape(3, N_PER_LAG)
    assert torch.isfinite(got).all()
    # Primary 1 has zero surprise; primary 0 has +5. A Primary-1 row must show no +5 in its windows
    # (statistics 0..), only in its neighbour column.
    assert float(first_row[:, : N_PER_LAG - 1].abs().max()) < 1e-4
    assert abs(float(got[3].reshape(3, N_PER_LAG)[0, 0]) - 5.0) < 1e-4
    print("selftest passed")


def main() -> None:
    """Run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("selftest",))
    parser.parse_args()
    selftest()


if __name__ == "__main__":
    main()
