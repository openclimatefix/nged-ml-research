"""Window statistics of experiment 404, copied unchanged into experiment 440 (see below).

Copied from `ml-experiments/00404-.../e00404_trust.py` (404 work folder name abbreviated) of
the draft pull request 58 (branch `mdn-lagless-calibration-trust`, copied 2026-10-31), because
experiment 404 has not merged. Original docstring:

Trust inputs of experiment 404, computed from dense surprise series by gathers.

A lag's trust inputs are statistics of the surprise series (the observed power minus the
cross-fitted lagless median) in windows centred on the lag's valid time. The windows depend on the
row only through the forecast origin, which truncates them at the newest usable observation, so the
statistics are not stored per row. Each Primary's surprise series sits on the dense half-hourly
grid with prefix sums and sparse tables, and a batch of rows gathers its windows on the GPU.

Statistics per window: the mean, the signed sum, the standard deviation, the minimum, the maximum,
the count above the lagless 95% quantile, the count below the 5% quantile, and a triangular-tent
mean (weight 1 at the lag's time falling to 0 at the window's half-width). The median named in the
brief cannot be read from prefix sums and sparse tables, so it is left out and the write-up says
so. The 84-hour window of the brief's week lag is applied to all three lags.

```bash
uv run --no-sync python ml-experiments/00404-*/e00404_trust.py selftest
```
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]

HALF_WIDTHS: Final[tuple[int, ...]] = (3, 6, 12, 24, 48, 168)
"""Half-widths in half-hours: 1.5, 3, 6, 12, 24, and 84 hours."""
STATS: Final[tuple[str, ...]] = (
    "mean", "sum", "std", "min", "max", "above", "below", "tent",
)  # fmt: skip
SCORE_CLIP: Final[float] = 20.0
NEIGHBOUR_WIDTH: Final[int] = 24
"""The neighbours' windowed mean uses the 12-hour half-width."""


@dataclass(frozen=True)
class DenseSeries:
    """One Primary's surprise series on the dense grid, with prefix sums and sparse tables.

    Attributes:
        first: Grid index of position 0 (the series covers `[first, first + n)`).
        s: Surprise (observed minus lagless median, fraction of capacity), 0 where unobserved.
        z: Surprise in units of the lagless band half-width, clipped at +-20.
        obs: 1 where observed, else 0.
        prefix: float64 prefix sums (n + 1 rows) of `obs`, `s`, `s^2`, above, below, `idx*obs`,
            `idx*s`, each already multiplied by `obs`.
        low: Sparse table of range minima of `s` (observed steps only), shape (levels, n).
        high: Sparse table of range maxima.
    """

    first: int
    s: torch.Tensor
    z: torch.Tensor
    obs: torch.Tensor
    prefix: torch.Tensor
    low: torch.Tensor
    high: torch.Tensor


def build_series(
    *, first: int, actual: np.ndarray, median: np.ndarray, q05: np.ndarray, q95: np.ndarray,
    device: str,
) -> DenseSeries:  # fmt: skip
    """Build a Primary's dense surprise series from the actual power and the lagless quantiles."""
    observed = np.isfinite(actual) & np.isfinite(median)
    s = np.where(observed, actual - median, 0.0)
    half = np.maximum((q95 - q05) / 2.0, 1e-3)
    z = np.clip(np.where(observed, s / half, 0.0), -SCORE_CLIP, SCORE_CLIP)
    above = (observed & (actual > q95)).astype(np.float64)
    below = (observed & (actual < q05)).astype(np.float64)
    obs = observed.astype(np.float64)
    idx = np.arange(len(actual), dtype=np.float64)
    columns = np.stack([obs, s, s * s, above, below, idx * obs, idx * s], axis=1)
    prefix = np.concatenate([np.zeros((1, columns.shape[1])), np.cumsum(columns, axis=0)])
    big = np.float32(3e38)
    low0 = np.where(observed, s, big).astype(np.float32)
    high0 = np.where(observed, s, -big).astype(np.float32)
    lows, highs = [low0], [high0]
    step = 1
    while 2 * step <= len(actual):
        lows.append(np.minimum(lows[-1][:-step], lows[-1][step:]))
        highs.append(np.maximum(highs[-1][:-step], highs[-1][step:]))
        step *= 2
    n = len(actual)
    pad = lambda a, fill: np.concatenate([a, np.full(n - len(a), fill, dtype=np.float32)])  # noqa: E731
    return DenseSeries(
        first=first,
        s=torch.tensor(s, dtype=torch.float32, device=device),
        z=torch.tensor(z, dtype=torch.float32, device=device),
        obs=torch.tensor(obs, dtype=torch.float32, device=device),
        prefix=torch.tensor(prefix, dtype=torch.float64, device=device),
        low=torch.tensor(np.stack([pad(a, big) for a in lows]), device=device),
        high=torch.tensor(np.stack([pad(a, -big) for a in highs]), device=device),
    )


def range_extreme(
    *, table: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, is_min: bool
) -> torch.Tensor:
    """Return the minimum (or maximum) over inclusive positions `lo..hi` from a sparse table."""
    length = (hi - lo + 1).clamp(min=1)
    level = torch.log2(length.to(torch.float32)).floor().to(torch.int64)
    span = 1 << level
    a = table[level, lo]
    b = table[level, (hi - span + 1).clamp(min=0)]
    return torch.minimum(a, b) if is_min else torch.maximum(a, b)


def window_statistics(
    *, series: DenseSeries, centre: torch.Tensor, end: torch.Tensor, half_width: int
) -> torch.Tensor:
    """Return the 8 statistics (rows, 8) of windows `[centre - h, min(centre + h, end)]`.

    Args:
        series: The Primary's dense series.
        centre: Grid index of each lag's valid time.
        end: Grid index of the newest usable observation (the window is cut there).
        half_width: Half-width `h` in half-hours.

    Returns:
        The statistics in `STATS` order. Windows with no observed step give zeros.
    """
    n = series.s.numel()
    c = (centre - series.first).clamp(0, n - 1)
    hi_all = (end - series.first).clamp(0, n - 1)
    lo = (c - half_width).clamp(min=0)
    hi = torch.minimum(c + half_width, hi_all).clamp(min=lo)
    p = series.prefix
    total = p[hi + 1] - p[lo]
    count = total[:, 0].clamp(min=1.0)
    ssum, sumsq = total[:, 1], total[:, 2]
    mean = ssum / count
    var = (sumsq / count - mean * mean).clamp(min=0.0)
    cf = c.to(torch.float64)
    h = float(half_width)
    left_hi = torch.minimum(c, hi)
    left = p[left_hi + 1] - p[lo]
    right = p[hi + 1] - p[left_hi + 1]
    # Tent weight: 1 - (c - i) / h on the left of the centre, 1 - (i - c) / h on the right.
    w_left_s = left[:, 1] * (1.0 - cf / h) + left[:, 6] / h
    w_right_s = right[:, 1] * (1.0 + cf / h) - right[:, 6] / h
    w_left_o = left[:, 0] * (1.0 - cf / h) + left[:, 5] / h
    w_right_o = right[:, 0] * (1.0 + cf / h) - right[:, 5] / h
    tent_weight = w_left_o + w_right_o
    tent = torch.where(
        tent_weight > 1e-6,
        (w_left_s + w_right_s) / tent_weight.clamp(min=1e-6),
        torch.zeros_like(tent_weight),
    )
    has = total[:, 0] > 0
    lo_val = range_extreme(table=series.low, lo=lo, hi=hi, is_min=True)
    hi_val = range_extreme(table=series.high, lo=lo, hi=hi, is_min=False)
    out = torch.stack(
        [
            mean, ssum, var.sqrt(), lo_val.to(torch.float64), hi_val.to(torch.float64),
            total[:, 3], total[:, 4], tent,
        ],
        dim=1,
    )  # fmt: skip
    return torch.where(has[:, None], out, torch.zeros_like(out)).to(torch.float32)


def single_step(*, series: DenseSeries, at: torch.Tensor) -> torch.Tensor:
    """Return the surprise (fraction of capacity) and the band score (rows, 2) at grid indices."""
    i = (at - series.first).clamp(0, series.s.numel() - 1)
    return torch.stack([series.s[i], series.z[i]], dim=1)


def selftest() -> None:
    """Compare every statistic with a brute-force numpy window on random data with gaps."""
    rng = np.random.default_rng(0)
    n = 4000
    actual = rng.normal(size=n)
    actual[rng.random(n) < 0.05] = np.nan
    median = np.zeros(n)
    q05, q95 = np.full(n, -1.0), np.full(n, 1.0)
    series = build_series(first=100, actual=actual, median=median, q05=q05, q95=q95, device="cpu")
    centre = rng.integers(400, n - 300, size=300) + 100
    end = centre + rng.integers(-50, 50, size=300)
    for h in HALF_WIDTHS:
        got = window_statistics(
            series=series,
            centre=torch.tensor(centre),
            end=torch.tensor(end),
            half_width=h,
        ).numpy()
        for r in range(0, 300, 7):
            c, e = centre[r] - 100, end[r] - 100
            lo, hi = max(0, c - h), max(0, c - h, min(c + h, e))
            idx = np.arange(lo, hi + 1)
            v = actual[idx]
            ok = np.isfinite(v)
            if not ok.any():
                continue
            vv, ii = v[ok], idx[ok]
            weight = np.maximum(0.0, 1.0 - np.abs(ii - c) / h)
            want = [
                vv.mean(), vv.sum(), vv.std(), vv.min(), vv.max(), (vv > 1).sum(), (vv < -1).sum(),
                (weight * vv).sum() / weight.sum() if weight.sum() > 0 else 0.0,
            ]  # fmt: skip
            np.testing.assert_allclose(got[r], want, rtol=1e-4, atol=1e-4, err_msg=f"h={h} r={r}")
    print("selftest passed")


if __name__ == "__main__":
    selftest()
