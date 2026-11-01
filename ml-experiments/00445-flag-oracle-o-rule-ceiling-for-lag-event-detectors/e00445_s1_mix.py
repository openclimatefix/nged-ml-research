"""Average stored forecasts from their 11 quantiles, for heads without stored mixture parameters.

Experiment 404 stored only the final quantiles of the S1 arm. An equal-weight average of two such
forecasts is built here from each forecast's piecewise-linear cumulative distribution function
(CDF) through its 11 quantiles, extended linearly past q01 and q99 to CDF 0 and 1 over the width of
the outermost interval. The method is approximate, so `validate` first applies it to the stored
A0-inj and lagless-head quantiles and compares the result with the exact mixture, B-mix.

```bash
<PYTHONPATH as e00445_orule.py shows> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00445-*/e00445_s1_mix.py build --world 200
```
"""

import argparse
from typing import Final

import e00445_ceiling as ceiling  # ty: ignore[unresolved-import]
import e00445_orule as orule  # ty: ignore[unresolved-import]
import e00445_score as score  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99)
STEPS: Final[int] = 32
MONTH: Final[str] = "2026-06"
PAIRS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))


def knots(*, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CDF knots (rows, 13): the 11 quantiles plus two tail ends, and their levels."""
    low = q[:, :1] - (q[:, 1:2] - q[:, :1]).clamp(min=1e-6)
    high = q[:, -1:] + (q[:, -1:] - q[:, -2:-1]).clamp(min=1e-6)
    x = torch.cat([low, q, high], dim=1)
    level = torch.tensor([0.0, *LEVELS, 1.0], dtype=q.dtype)
    return x, level


def cdf_at(*, x_knots: torch.Tensor, level: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    """Return the piecewise-linear CDF of each row at `at` (rows, n)."""
    idx = (at[:, :, None] >= x_knots[:, None, :]).sum(-1).clamp(1, x_knots.shape[1] - 1)
    x0 = x_knots.gather(1, idx - 1)
    x1 = x_knots.gather(1, idx)
    f0, f1 = level[idx - 1], level[idx]
    frac = ((at - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
    return f0 + frac * (f1 - f0)


def average_forecasts(*, forecasts: list[np.ndarray]) -> np.ndarray:
    """Return the 11 quantiles of the equal-weight average of forecasts given as (rows, 11)."""
    out = []
    for start in range(0, len(forecasts[0]), 8192):
        sl = slice(start, start + 8192)
        qs = [torch.from_numpy(f[sl].astype(np.float64)) for f in forecasts]
        kn = [knots(q=q) for q in qs]
        lo = torch.stack([k[0][:, 0] for k in kn]).min(0).values[:, None].expand(-1, len(LEVELS))
        hi = torch.stack([k[0][:, -1] for k in kn]).max(0).values[:, None].expand(-1, len(LEVELS))
        lo, hi = lo.clone(), hi.clone()
        target = torch.tensor(LEVELS, dtype=torch.float64)
        for _ in range(STEPS):
            mid = 0.5 * (lo + hi)
            cdf = sum(cdf_at(x_knots=k[0], level=k[1], at=mid) for k in kn) / len(kn)
            below = cdf < target
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        out.append((0.5 * (lo + hi)).numpy())
    return np.concatenate(out)


def stored(*, arm: str, seed: int, world: int) -> pl.DataFrame:
    """Return a stored forecast frame of experiment 404 or this experiment."""
    return pl.read_parquet(ceiling.arm_path(arm=arm, month=MONTH, seed=seed, world=world))


def validate(*, world: int, rows: int = 60000) -> None:
    """Compare the approximate average of A0-inj and the lagless head with exact B-mix."""
    names = list(orule.LEVEL_NAMES)
    a = stored(arm="A0-inj", seed=0, world=world)[names].to_numpy()[:rows]
    b = stored(arm="LL-s0", seed=0, world=world)[names].to_numpy()[:rows]
    exact = stored(arm="B-mix", seed=0, world=world)[names].to_numpy()[:rows]
    approx = average_forecasts(forecasts=[a, b])
    frame = stored(arm="B-mix", seed=0, world=world).head(rows)
    err = np.abs(approx - exact)
    scores = {}
    for label, q in (("exact", exact), ("approx", approx)):
        scored = frame.with_columns(**{n: pl.Series(q[:, i]) for i, n in enumerate(names)})
        scores[label] = float(scored.select(score.pinball_points(frame=scored).mean()).item())
    print(
        f"validate w{world}: mean |approx - exact| {err.mean():.2e}, max {err.max():.2e}; "
        f"mean score exact {scores['exact']:.4f}, approx {scores['approx']:.4f}",
        flush=True,
    )


def build(*, world: int) -> None:
    """Write the three S1 pair averages for a world."""
    names = list(orule.LEVEL_NAMES)
    base_full = stored(arm="S1", seed=0, world=world)
    base = base_full.select([c for c in base_full.columns if c not in names])
    for i, j in PAIRS:
        a = stored(arm="S1", seed=i, world=world)
        b = stored(arm="S1", seed=j, world=world)
        for key in ("primary", "origin", "target"):
            assert np.array_equal(a[key].to_numpy(), b[key].to_numpy()), key
            assert np.array_equal(a[key].to_numpy(), base[key].to_numpy()), key
        q = average_forecasts(forecasts=[a[names].to_numpy(), b[names].to_numpy()])
        arm = f"E-s1-{i}{j}"
        frame = base.with_columns(
            **{n: pl.Series(q[:, k]).cast(pl.Float32) for k, n in enumerate(names)}
        )
        target = orule.prediction_path(arm=arm, month=MONTH, seed=0, world=world)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        print(f"wrote {arm} w{world}: {frame.height:,} rows", flush=True)


def main() -> None:
    """Validate the method and build the S1 pair averages for a world."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "validate":
        validate(world=args.world)
    else:
        build(world=args.world)


if __name__ == "__main__":
    main()
