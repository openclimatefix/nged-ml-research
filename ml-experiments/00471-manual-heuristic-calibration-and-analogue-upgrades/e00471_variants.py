# ruff: noqa: E501, PLR0915, D107
"""Stage 1 and the level-adjustment arm of experiment 471: calibrated variants of the heuristic.

Every variant reads the stored analogue members (`e00471_common.load_members`) and the cleaned
actuals, and writes its 11 quantiles in MW to `work/predictions/<name>.parquet`. Fits are per
Primary and lead band (0 to 24, 24 to 72, 72 to 120, 120 to 168, 168 to 240 hours), refitted weekly
on a trailing window of `D` days of pairs whose valid time is at most `DELAY_HOURS` before the
refit origin, so nothing past the forecast origin enters a fit. All fitting is in fractions of
capacity. Rows with fewer than `MIN_MEMBERS` members take the faithful quantiles in every variant, and
the eight MVA-only series (demand cannot be negative) have every variant floored at zero.

* `E<D>`: Gaussian EMOS (Gneiting et al. 2005), mean a + b x analogue mean, spread
  exp(c) x (analogue standard deviation + eps) ** d, fitted by minimum Gaussian CRPS.
* `E<D>m13`, `E<D>m99`: the same distribution emitted as m equiprobable members at levels
  `(i - 0.5) / m`, then the 11 quantiles rebuilt from those members by linear interpolation.
* `MeanLS<D>`: members shifted by a least-squares line of the actual on the analogue mean (no
  spread change).
* `Q<D>`: every output level read from the members at the probability-integral-transform
  recalibrated level; `Tau95_<D>`: only q95 recalibrated.
* `Lvl<W>`: members shifted by the mean error (actual minus analogue mean) of the last `W` days.

```bash
source /tmp/e471_env.sh
uv run --no-sync python ml-experiments/00471-*/e00471_variants.py
```
"""

import time
from typing import Final

import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00448_refs as base  # ty: ignore[unresolved-import]
import e00471_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

T0: Final = np.datetime64("2025-01-01T09:00:00", "us")
DAY: Final = np.timedelta64(24, "h")
FIRST_ORIGIN: Final = np.datetime64("2025-09-25T09:00:00", "us")
EPS_SD: Final[float] = 0.002
MIN_PAIRS: Final[int] = 200
MIN_MEMBERS: Final[int] = 6
WINDOWS: Final[tuple[int, ...]] = (28, 56, 112)
LEVEL_WINDOWS: Final[tuple[int, ...]] = (7, 14, 28)
PIT_WINDOW: Final[int] = 56
OUT = common.WORK_DIR / "predictions"
Z: Final[np.ndarray] = norm.ppf(common.LEVEL_VALUES)


def gaussian_crps(*, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the closed-form CRPS of N(mu, sigma^2) at `y`."""
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def fit_emos(*, mean: np.ndarray, sd: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return (a, b, c, d) minimising the mean Gaussian CRPS of y given the analogue mean and spread."""
    log_sd = np.log(sd + EPS_SD)

    def loss(theta: np.ndarray) -> float:
        a, b, c, d = theta
        sigma = np.exp(c + d * log_sd)
        return float(gaussian_crps(y=y, mu=a + b * mean, sigma=sigma).mean())

    start = np.array([0.0, 1.0, np.log(max(np.median(np.abs(y - mean)), 1e-3) * 1.25), 0.0])
    bounds = [(-0.5, 0.5), (0.0, 2.0), (-9.0, 1.0), (0.0, 2.0)]
    return minimize(loss, start, method="L-BFGS-B", bounds=bounds).x


def pit_value(*, members: np.ndarray, n: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the level in [0, 1] at which the members' linear-interpolation quantile equals `y`."""
    out = np.empty(len(y))
    for i in range(len(y)):
        v = members[i, : n[i]]
        out[i] = np.interp(y[i], v, np.linspace(0.0, 1.0, n[i])) if n[i] > 1 else 0.5
    return out


def level_quantiles(*, members: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return, per row, the linear-interpolation quantile of the non-NaN members at that row's levels."""
    out = np.empty(len(members))
    for i in range(len(members)):
        v = members[i][~np.isnan(members[i])]
        out[i] = np.quantile(v, levels[i]) if len(v) else np.nan
    return out


class Series:
    """One Primary's rows in fractions of capacity, sorted by origin."""

    def __init__(self, *, table: pl.DataFrame, capacity: float, floor_at_zero: bool) -> None:
        t = table.sort("power_fcst_init_time", "valid_time")
        self.table = t
        self.members = t.select(common.MEMBER_COLUMNS).to_numpy() / capacity
        self.n = t["n"].to_numpy()
        self.capacity = capacity
        self.floor_at_zero = floor_at_zero
        self.actual = t["actual"].to_numpy() / capacity
        self.band = common.band_of(lead_hours=t["lead_hours"].to_numpy())
        origin = t["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
        valid = t["valid_time"].dt.replace_time_zone(None).to_numpy()
        self.origin = origin
        self.day = ((origin - T0) // DAY).astype(int)
        self.bin = ((valid - T0) // DAY).astype(int)
        with np.errstate(all="ignore"):
            self.mean = np.nanmean(self.members, axis=1)
            self.sd = np.nanstd(self.members, axis=1, ddof=1)
        self.usable = np.isfinite(self.actual) & (self.n >= MIN_MEMBERS)
        self.faithful = np.nanquantile(self.members, common.LEVEL_VALUES, axis=1).T
        self.targets = np.flatnonzero(self.origin >= FIRST_ORIGIN)

    def training_rows(self, *, refit_day: int, band: int, window: int) -> np.ndarray:
        """Return row indices of pairs usable by a fit made on `refit_day` (bins <= refit_day - 2)."""
        sel = self.usable & (self.band == band) & (self.bin <= refit_day - 2)
        sel &= self.bin >= refit_day - 2 - window + 1
        return np.flatnonzero(sel)


def emit(*, series: Series, name: str, quantiles: dict[int, np.ndarray], out: dict) -> None:
    """Store `quantiles` (rows -> (n, 11) fractions) as MW rows keyed for the scorer."""
    rows = np.array(sorted(quantiles))
    q = np.vstack([quantiles[r] for r in rows]) if len(rows) else np.empty((0, 11))
    thin = series.n[rows] < MIN_MEMBERS
    q[thin] = series.faithful[rows[thin]]
    if series.floor_at_zero:
        q = np.maximum(q, 0.0)
    frame = (
        series.table[rows.tolist()]
        .select(*common.KEYS, "lead_hours")
        .with_columns(
            *[pl.Series(k, q[:, i] * series.capacity) for i, k in enumerate(common.LEVELS)]
        )
    )
    out.setdefault(name, []).append(frame)


def run_series(*, series: Series, fallbacks: dict[str, int], out: dict) -> None:
    """Build every variant for one Primary."""
    days = series.day
    refit_of = days - (days % 7)
    for window in WINDOWS:
        for r in np.unique(refit_of[series.targets]):
            for band in range(5):
                rows = series.targets[
                    (refit_of[series.targets] == r) & (series.band[series.targets] == band)
                ]
                if not len(rows):
                    continue
                train = series.training_rows(refit_day=int(r), band=band, window=window)
                ok = len(train) >= MIN_PAIRS
                if not ok:
                    fallbacks[f"E{window}"] = fallbacks.get(f"E{window}", 0) + len(rows)
                    continue
                a, b, c, d = fit_emos(
                    mean=series.mean[train], sd=series.sd[train], y=series.actual[train]
                )
                use = rows[series.n[rows] >= 1]
                mu = a + b * series.mean[use]
                sigma = np.exp(c + d * np.log(series.sd[use] + EPS_SD))
                gauss = mu[:, None] + sigma[:, None] * Z[None, :]
                emit(
                    series=series,
                    name=f"E{window}",
                    quantiles=dict(zip(use, gauss, strict=True)),
                    out=out,
                )
                for m in (13, 99):
                    members = (np.arange(1, m + 1) - 0.5) / m
                    mem = mu[:, None] + sigma[:, None] * norm.ppf(members)[None, :]
                    q = np.quantile(mem, common.LEVEL_VALUES, axis=1).T
                    emit(
                        series=series,
                        name=f"E{window}m{m}",
                        quantiles=dict(zip(use, q, strict=True)),
                        out=out,
                    )
                slope, intercept = np.polyfit(series.mean[train], series.actual[train], 1)
                shift = intercept + (slope - 1.0) * series.mean[use]
                shifted = series.faithful[use] + shift[:, None]
                emit(
                    series=series,
                    name=f"MeanLS{window}",
                    quantiles=dict(zip(use, shifted, strict=True)),
                    out=out,
                )
    for r in np.unique(refit_of[series.targets]):
        for band in range(5):
            rows = series.targets[
                (refit_of[series.targets] == r) & (series.band[series.targets] == band)
            ]
            train = series.training_rows(refit_day=int(r), band=band, window=PIT_WINDOW)
            if not len(rows) or len(train) < MIN_PAIRS:
                continue
            u = pit_value(members=series.members[train], n=series.n[train], y=series.actual[train])
            tau = np.quantile(u, common.LEVEL_VALUES)
            fallbacks["pit_tau95_is_one"] = fallbacks.get("pit_tau95_is_one", 0) + int(
                tau[9] >= 1.0
            )
            fallbacks["pit_cells"] = fallbacks.get("pit_cells", 0) + 1
            use = rows[series.n[rows] >= 1]
            q = np.vstack(
                [
                    level_quantiles(members=series.members[use], levels=np.full(len(use), t))
                    for t in tau
                ]
            ).T
            emit(
                series=series,
                name=f"Q{PIT_WINDOW}",
                quantiles=dict(zip(use, q, strict=True)),
                out=out,
            )
            q95 = series.faithful[use].copy()
            q95[:, 9] = level_quantiles(
                members=series.members[use], levels=np.full(len(use), tau[9])
            )
            q95 = np.sort(q95, axis=1)
            emit(
                series=series,
                name=f"Tau95_{PIT_WINDOW}",
                quantiles=dict(zip(use, q95, strict=True)),
                out=out,
            )
    err = series.actual - series.mean
    n_bins = int(series.bin.max()) + 3
    for window in LEVEL_WINDOWS:
        total = np.zeros((n_bins, 5))
        count = np.zeros((n_bins, 5))
        ok = series.usable
        np.add.at(total, (series.bin[ok], series.band[ok]), err[ok])
        np.add.at(count, (series.bin[ok], series.band[ok]), 1.0)
        cs_total, cs_count = np.cumsum(total, axis=0), np.cumsum(count, axis=0)
        use = series.targets[series.n[series.targets] >= 1]
        hi = np.clip(days[use] - 2, 0, n_bins - 1)
        lo = np.clip(days[use] - 2 - window, 0, n_bins - 1)
        s = cs_total[hi, series.band[use]] - cs_total[lo, series.band[use]]
        c = cs_count[hi, series.band[use]] - cs_count[lo, series.band[use]]
        shift = np.where(c >= 50, s / np.maximum(c, 1), 0.0)
        fallbacks[f"Lvl{window}"] = fallbacks.get(f"Lvl{window}", 0) + int((c < 50).sum())
        emit(
            series=series,
            name=f"Lvl{window}",
            quantiles=dict(zip(use, series.faithful[use] + shift[:, None], strict=True)),  # ty: ignore[invalid-argument-type]
            out=out,
        )


def main() -> None:
    """Write every variant's quantiles."""
    series_table = adapter.series_table()
    mva_only = set(base.sites_table().filter(pl.col("group") == "MVA-only eight")["time_series_id"])
    members = common.load_members()
    actuals = common.cleaned_actuals(series=series_table)
    joined = members.join(actuals, on=["time_series_id", "valid_time"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[pl.DataFrame]] = {}
    fallbacks: dict[str, int] = {}
    for row in series_table.iter_rows(named=True):
        t0 = time.time()
        part = joined.filter(pl.col("time_series_id") == row["time_series_id"])
        run_series(
            series=Series(
                table=part,
                capacity=row["capacity_mw"],
                floor_at_zero=row["time_series_id"] in mva_only,
            ),
            fallbacks=fallbacks,
            out=out,
        )
        print(f"{row['label']}: {time.time() - t0:.0f} s")
    for name, parts in out.items():
        pl.concat(parts).write_parquet(OUT / f"{name}.parquet")
    print("fallback / diagnostic counts:", fallbacks)


if __name__ == "__main__":
    main()
