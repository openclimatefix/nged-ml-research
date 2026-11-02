"""Experiment 507: the non-learned repair, total-variation denoising of the 14-day residual.

For one window (Primary, origin) the residual is the reading minus the lab lagless median over the
672 half-hours ending at the newest usable reading (`origin - 7`). Total-variation denoising gives
a piecewise-constant level `u`. The normal level is the median of `u`. Where `u` departs from the
normal level by more than `tau` spreads, the repair subtracts that departure from the reading.
Unobserved half-hours are filled by the nearest observed residual before the solve and stay
unobserved in the output. The solver is FISTA on the dual problem, batched over windows on the GPU:

    minimise  0.5 * sum((y - u)^2) + lam * sum(|u[t+1] - u[t]|)   over u.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_tv.py selftest
```
"""

import argparse
from collections.abc import Callable
from typing import Final

import e00507_arms as arms  # ty: ignore[unresolved-import]
import e00507_generator as gen  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]

ITERATIONS: Final[int] = 2000
Reader = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Maps arrays of Primary and day indices to the readings (windows, 672) of their histories."""


def fill_nearest(*, x: np.ndarray) -> np.ndarray:
    """Fill NaN along axis 1 with the nearest earlier value, then the nearest later one."""
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(x.shape[1])[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.take_along_axis(x, idx, axis=1)
    back = np.where(ok, np.arange(x.shape[1])[None, :], x.shape[1] - 1)
    back = np.minimum.accumulate(back[:, ::-1], axis=1)[:, ::-1]
    out = np.where(np.isfinite(out), out, np.take_along_axis(x, back, axis=1))
    return np.nan_to_num(out, nan=0.0)


@torch.no_grad()
def tv_denoise(*, y: np.ndarray, lam: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Return the total-variation denoised level of each row of `y`; `lam` has one value per row."""
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    lt = torch.tensor(lam, dtype=torch.float32, device=device)[:, None]
    p = torch.zeros(yt.shape[0], yt.shape[1] - 1, device=device)
    q, t = p.clone(), 1.0
    tau = 0.25

    def dt(p: torch.Tensor) -> torch.Tensor:
        """Return D^T p."""
        out = torch.zeros(p.shape[0], p.shape[1] + 1, device=p.device)
        out[:, :-1] -= p
        out[:, 1:] += p
        return out

    for _ in range(ITERATIONS):
        u = yt - dt(q)
        grad = u[:, 1:] - u[:, :-1]
        p_new = torch.maximum(torch.minimum(q + tau * grad, lt), -lt)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        q = p_new + (t - 1) / t_new * (p_new - p)
        p, t = p_new, t_new
    return (yt - dt(p)).cpu().numpy()


def repair(
    *, reading: np.ndarray, median: np.ndarray, spread: np.ndarray, weight: float, tau: float,
    device: str = "cuda",
) -> np.ndarray:  # fmt: skip
    """Return the repaired readings (windows, 672) of the given histories.

    Args:
        reading: Readings (windows, 672), NaN where unobserved.
        median: The lagless median at the same half-hours.
        spread: The Primary's residual spread per window.
        weight: The total-variation weight in units of `spread`.
        tau: The departure, in spreads, beyond which the level is subtracted.
        device: Where to solve.
    """
    resid = reading - median
    observed = np.isfinite(resid)
    u = tv_denoise(y=fill_nearest(x=resid), lam=weight * spread, device=device)
    normal = np.median(u, axis=1, keepdims=True)
    depart = u - normal
    departure = np.where(np.abs(depart) > tau * spread[:, None], depart, 0.0)
    return np.where(observed, reading - departure, np.nan).astype(np.float32)


def window_positions(*, rows: arms.Rows) -> np.ndarray:
    """Return each row's three lag positions inside its 672-half-hour history."""
    newest = gen.newest_of(day=rows.day)
    pos = rows.lag_grid - (newest[:, None] - (gen.HISTORY - 1))
    assert pos.min() >= 0 and pos.max() < gen.HISTORY, "a lag lies outside the history"
    assert (rows.lag_grid <= newest[:, None]).all(), "a lag is newer than the newest usable reading"
    return pos


class TvLags:
    """A lag source whose readings are the total-variation repair of another source's histories."""

    def __init__(
        self, *, reader: Reader, med: np.ndarray, spread: np.ndarray, weight: float, tau: float,
        target_source: arms.LagSource, device: str = "cuda",
    ) -> None:  # fmt: skip
        """Hold the history reader, the lab median grid, and the repair settings."""
        self.reader, self.med, self.spread = reader, med, spread
        self.weight, self.tau, self.device = weight, tau, device
        self.target_source = target_source
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def repaired(self, *, p: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Return the repaired histories (windows, 672) for the given Primary and day arrays."""
        newest = gen.newest_of(day=d)
        g = newest[:, None] + np.arange(-(gen.HISTORY - 1), 1)[None, :]
        out = np.empty((len(p), gen.HISTORY), dtype=np.float32)
        for start in range(0, len(p), 4096):
            sl = slice(start, start + 4096)
            reading = self.reader(p[sl], d[sl])
            median = self.med[p[sl, None], np.clip(g[sl], 0, self.med.shape[1] - 1)]
            out[sl] = repair(
                reading=reading, median=median, spread=self.spread[p[sl]], weight=self.weight,
                tau=self.tau, device=self.device,
            )  # fmt: skip
        return out

    def values(self, *, rows: arms.Rows) -> np.ndarray:
        """Return the repaired lag readings of each row."""
        key = rows.primary * 100_000 + rows.day
        uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        windows = self.repaired(p=rows.primary[first], d=rows.day[first])
        pos = window_positions(rows=rows)
        return windows[inverse[:, None], pos]

    def target(self, *, rows: arms.Rows, clean: np.ndarray) -> np.ndarray:
        """Return the wrapped source's target."""
        return self.target_source.target(rows=rows, clean=clean)


def selftest() -> None:
    """Check the solver on a noisy piecewise-constant signal and the end-of-window rule."""
    rng = np.random.default_rng(0)
    true = np.zeros((4, gen.HISTORY), dtype=np.float32)
    true[:, 200:300] = 1.0
    true[2, 500:] = -0.8
    y = true + 0.05 * rng.normal(size=true.shape).astype(np.float32)
    u = tv_denoise(y=y, lam=np.full(4, 0.5))
    assert np.abs(u - true).mean() < 0.03, "total-variation solution is not close to the signal"
    jumps = (np.abs(np.diff(u, axis=1)) > 0.2).sum(axis=1)
    assert (jumps[:2] == 2).all() and jumps[2] == 3, jumps
    nan_y = y.copy()
    nan_y[0, 100:110] = np.nan
    fixed = fill_nearest(x=nan_y)
    assert np.isfinite(fixed).all()
    spread = np.full(4, 0.05)
    fixed_reading = repair(reading=y + 0.5, median=np.full_like(y, 0.5), spread=spread,
                           weight=10.0, tau=1.0)  # fmt: skip
    assert abs(float(np.median(fixed_reading[0, :150])) - 0.5) < 0.02
    assert abs(float(fixed_reading[0, 250]) - 0.5) < 0.1, "the step was not removed"
    print("selftest passed: solver recovers 2 and 3 jumps; the step is subtracted")


def main() -> None:
    """Run the self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["selftest"])
    parser.parse_args()
    selftest()


if __name__ == "__main__":
    main()
