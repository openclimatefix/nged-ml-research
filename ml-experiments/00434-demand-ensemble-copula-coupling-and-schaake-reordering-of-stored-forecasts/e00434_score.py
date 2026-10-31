# ruff: noqa: E501
"""Re-order the 51 values of experiment 425's arm A into sample paths and score their coherence.

No fit. Reads experiment 425's stored arm A (seeds 0 to 4 averaged in quantile space, widened by
1.15 about the median), draws 51 values per Primary and half-hour at the levels k/52 by linear
interpolation of the 11 stored quantiles, and assigns the values to 51 paths in five ways:

- `ECC-T`: by the rank of each ensemble member's 2-metre temperature (coldest member, highest value);
- `ECC-SW`: by the rank of each member's shortwave radiation (darkest member, highest value);
- `SCH`: by the ranks of 51 past observed paths (Schaake shuffle);
- `IND`: by an independent random permutation per Primary and half-hour;
- `CO`: every path ranked alike (comonotone).

All five share the same 51 values at every Primary and half-hour, so the marginal score is the
same. The script writes, per month, the marginal rows and the coherence scores (energy and variogram
scores, exponent 0.5, in points of capacity) in three blocks: `spatial` (16 Primaries at each
half-hour), `temporal` (each Primary across leads), and `joint` (both, at coarser leads).

```bash
DATA_PATH_INTERNAL=... PYTHONPATH=<folders of experiments 87, 377, 381> \
  nice -n 19 uv run --no-sync python ml-experiments/00434-*/e00434_score.py --months 2025-07
```
"""

import argparse
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00381_score as s381  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.eval_protocol import metrics
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = (
    "00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
SEED_DIR_425: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00425-best-mdn-architecture-assembly-with-ablations"
    / "work"
    / "seed_preds"
    / "A"
)
FLAGGED_KEYS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00398-load-shift-event-rescore-without-flagged-zero-actuals"
    / "flagged_keys.parquet"
)
MONTHS: Final[tuple[str, ...]] = (
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
)  # fmt: skip
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
LEVELS: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
QUANTILE_LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
OUTER: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
WIDENING: Final[float] = 1.15
N_MEMBERS: Final[int] = 51
N_STEPS: Final[int] = 702
"""Half-hourly valid times per run: leads 0.5 to 351 hours."""
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
ARMS: Final[tuple[str, ...]] = ("ECC-T", "ECC-SW", "SCH", "IND", "CO")
WINDOWS: Final[dict[str, tuple[float, float]]] = {
    "days_1_10": (0.0, 240.0),
    "h12_36": (12.0, 36.0),
}
"""Lead windows in hours, open below and closed above."""
TEMPORAL_STRIDE: Final[dict[str, int]] = {"days_1_10": 4, "h12_36": 2}
JOINT_STRIDE: Final[dict[str, int]] = {"days_1_10": 12, "h12_36": 4}
SLICES: Final[dict[str, str | None]] = {
    "all": None,
    "daylight": "time_midday",
    "cold_nights": "cold_nights",
}
MIN_DIM: Final[dict[str, int]] = {"spatial": 8, "temporal": 10, "joint": 100}
SCHAAKE_FIRST: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
SCHAAKE_BUFFER: Final[timedelta] = timedelta(days=16)
SCHAAKE_MAX_MISSING: Final[float] = 0.05
VARIOGRAM_EXPONENT: Final[float] = 0.5
STEP: Final[timedelta] = timedelta(minutes=30)


# --- stored forecast and inputs ----------------------------------------------------------------


def stored_arm_a(*, month: str) -> pl.DataFrame:
    """Return arm A for a month: seeds averaged in quantile space, sorted, widened by 1.15."""
    frames = [pl.read_parquet(SEED_DIR_425 / f"{month}_s{seed}.parquet") for seed in SEEDS]
    keys = list(common.KEYS)
    for frame in frames[1:]:
        if frame.height != frames[0].height or not frame.select(keys).sort(keys).equals(
            frames[0].select(keys).sort(keys)
        ):
            msg = f"{month}: a seed differs in height or keys from seed 0"
            raise ValueError(msg)
    mean = pl.concat(frames).group_by(keys).agg(*[pl.col(c).mean() for c in LEVELS]).sort(keys)
    values = np.sort(mean.select(LEVELS).to_numpy().astype(np.float64), axis=1)
    mean = mean.with_columns(*[pl.Series(c, values[:, i]) for i, c in enumerate(LEVELS)])
    return mean.with_columns(
        *[(pl.col("q50") + WIDENING * (pl.col(c) - pl.col("q50"))).alias(c) for c in OUTER]
    )


def fifty_one_values(*, quantiles: np.ndarray) -> np.ndarray:
    """Return values at the levels k/52 by linear interpolation in the level, shape (..., 51).

    All 51 levels lie inside the stored q01 to q99, so nothing is extrapolated.
    """
    levels = np.arange(1, N_MEMBERS + 1) / (N_MEMBERS + 1)
    flat = quantiles.reshape(-1, quantiles.shape[-1])
    out = np.empty((flat.shape[0], N_MEMBERS))
    # Linear interpolation of 11 levels, vectorised: find the segment of each target level.
    index = np.clip(np.searchsorted(QUANTILE_LEVELS, levels, side="right") - 1, 0, 9)
    weight = (levels - QUANTILE_LEVELS[index]) / (
        QUANTILE_LEVELS[index + 1] - QUANTILE_LEVELS[index]
    )
    out = flat[:, index] * (1 - weight) + flat[:, index + 1] * weight
    return out.reshape(*quantiles.shape[:-1], N_MEMBERS)


def cleaned_context(*, series: pl.DataFrame) -> dict:
    """Return experiment 381's scoring context with the flagged exact-zero actuals removed."""
    ctx = s381.build_context(series=series)
    flagged = pl.read_parquet(FLAGGED_KEYS).select("time_series_id", "valid_time")
    before = ctx["actuals"].height
    ctx["actuals"] = ctx["actuals"].join(flagged, on=["time_series_id", "valid_time"], how="anti")
    print(f"scoring actuals dropped for a flagged exact zero: {before - ctx['actuals'].height}")
    return ctx


class Grid:
    """The actuals of the Primaries (points of capacity) on one half-hourly grid; NaN is missing."""

    def __init__(self, *, series: pl.DataFrame, actuals: pl.DataFrame) -> None:
        """Build the grid from the series table and the cleaned actuals (MW)."""
        self.start = datetime(2024, 4, 1, tzinfo=UTC)
        end = datetime(2026, 7, 1, tzinfo=UTC)
        self.n = int((end - self.start) / STEP)
        self.ids = series["time_series_id"].to_list()
        self.values = np.full((len(self.ids), self.n), np.nan)
        capacity = dict(zip(series["time_series_id"], series["capacity_mw"], strict=True))
        for p, sid in enumerate(self.ids):
            sub = actuals.filter(pl.col("time_series_id") == sid)
            idx = ((sub["valid_time"] - self.start).dt.total_minutes() // 30).to_numpy()
            ok = (idx >= 0) & (idx < self.n)
            self.values[p, idx[ok]] = sub["actual"].to_numpy()[ok] / capacity[sid] * 100.0

    def index(self, *, t: datetime) -> int:
        """Return the grid index of a half-hour."""
        return int((t - self.start) / STEP)

    def path(self, *, t0: datetime) -> np.ndarray:
        """Return the (P, 702) values from the half-hour after `t0`."""
        i = self.index(t=t0) + 1
        return self.values[:, i : i + N_STEPS]


def run_weather(*, init: datetime, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return member temperature and shortwave, each (P, 702, 51), for a run at 09:00 UTC."""
    frame = pl.read_parquet(
        common.demand_frame_path(run=init - PUBLICATION_DELAY),
        columns=["time_series_id", "valid_time", "ensemble_member", "temperature_2m", "shortwave"],
    ).filter(pl.col("time_series_id").is_in(ids))
    order = {sid: i for i, sid in enumerate(ids)}
    frame = (
        frame.with_columns(p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64))
        .sort("p", "valid_time", "ensemble_member")
        .filter(pl.col("valid_time") > init)
    )
    if frame.height != len(ids) * N_STEPS * N_MEMBERS:
        msg = f"{init}: weather frame has {frame.height} rows, expected {len(ids) * N_STEPS * N_MEMBERS}"
        raise ValueError(msg)
    shape = (len(ids), N_STEPS, N_MEMBERS)
    return (
        frame["temperature_2m"].to_numpy().reshape(shape).astype(np.float64),
        frame["shortwave"].to_numpy().reshape(shape).astype(np.float64),
    )


# --- orderings ---------------------------------------------------------------------------------


def ranks_by_key(*, key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return, along the last axis, each entry's rank (0 = smallest key), ties broken at random."""
    order = np.lexsort((rng.random(key.shape), key), axis=-1)
    return np.argsort(order, axis=-1)


def assign(*, values: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Return the paths (51, P, S): path j takes the value whose rank equals its member rank."""
    return np.take_along_axis(values, ranks, axis=-1).transpose(2, 0, 1)


def schaake_pool(*, init: datetime, grid: Grid) -> tuple[np.ndarray, int]:
    """Return 51 past observed paths (51, P, S) and the number of distinct dates that qualified.

    Candidate dates start at the run's time of day, from 2025-03-01 to 16 days before the run, so
    every path of up to 14.6 days ends before the run. A date is dropped if any Primary lacks more
    than 5% of its half-hours; gaps are filled by linear interpolation along the path. The 51 dates
    nearest in day of year are used, repeated cyclically if fewer qualify.
    """
    last = init - SCHAAKE_BUFFER
    candidates = []
    t0 = SCHAAKE_FIRST.replace(hour=init.hour, minute=init.minute)
    while t0 <= last:
        path = grid.path(t0=t0)
        if path.shape[1] == N_STEPS and np.isnan(path).mean(axis=1).max() <= SCHAAKE_MAX_MISSING:
            doy = abs((t0.timetuple().tm_yday - init.timetuple().tm_yday + 183) % 365 - 183)
            candidates.append((doy, t0, path))
        t0 += timedelta(days=1)
    if not candidates:
        msg = f"{init}: no Schaake date qualifies"
        raise ValueError(msg)
    candidates.sort(key=lambda c: (c[0], c[1]))
    chosen = [candidates[i % len(candidates)][2] for i in range(N_MEMBERS)]
    filled = []
    x = np.arange(N_STEPS)
    for path in chosen:
        out = path.copy()
        for p in range(out.shape[0]):
            bad = np.isnan(out[p])
            if bad.any():
                out[p, bad] = np.interp(x[bad], x[~bad], out[p, ~bad])
        filled.append(out)
    return np.stack(filled), len(candidates)


# --- scores --------------------------------------------------------------------------------------


def energy_score(*, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the energy score per batch item, divided by the square root of the dimension used.

    Args:
        x: Paths, shape (B, 51, d).
        y: Observed vector, shape (B, d), with any value where `mask` is false.
        mask: Components used, shape (B, d).
    """
    m = mask[:, None, :]
    to_obs = np.sqrt((((x - y[:, None, :]) * m) ** 2).sum(-1)).mean(-1)
    pair = np.sqrt((((x[:, :, None, :] - x[:, None, :, :]) * m[:, :, None, :]) ** 2).sum(-1))
    score = to_obs - 0.5 * pair.mean((-1, -2))
    return score / np.sqrt(mask.sum(-1))


def variogram_score(*, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the variogram score (exponent 0.5, equal weights) per batch item, per pair.

    Pairs with a missing component are dropped; the sum is divided by the number of pairs used.
    """
    p = VARIOGRAM_EXPONENT
    obs = np.abs(y[:, :, None] - y[:, None, :]) ** p
    sim = (np.abs(x[:, :, :, None] - x[:, :, None, :]) ** p).mean(1)
    pair_mask = (mask[:, :, None] & mask[:, None, :]) & ~np.eye(mask.shape[1], dtype=bool)[None]
    total = (((obs - sim) ** 2) * pair_mask).sum((-1, -2))
    return total / np.maximum(pair_mask.sum((-1, -2)), 1)


def score_blocks(
    *, x: np.ndarray, y: np.ndarray, mask: np.ndarray, chunk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the energy and variogram scores for every batch item, computed in chunks."""
    es, vs = [], []
    for a in range(0, x.shape[0], chunk):
        sl = slice(a, a + chunk)
        es.append(energy_score(x=x[sl], y=y[sl], mask=mask[sl]))
        vs.append(variogram_score(x=x[sl], y=y[sl], mask=mask[sl]))
    return np.concatenate(es), np.concatenate(vs)


def window_steps(*, window: str) -> np.ndarray:
    """Return the step indices (0 = lead 0.5 h) whose lead lies in the window."""
    low, high = WINDOWS[window]
    lead = (np.arange(N_STEPS) + 1) * 0.5
    return np.flatnonzero((lead > low) & (lead <= high))


def coherence_rows(
    *,
    paths: dict[str, np.ndarray],
    actual: np.ndarray,
    flags: dict[str, np.ndarray],
    labels: list[str],
    init: datetime,
) -> list[dict]:
    """Return one row per arm, block, window, slice (and Primary, for the temporal block)."""
    rows: list[dict] = []
    have = ~np.isnan(actual)
    for window in WINDOWS:
        steps = window_steps(window=window)
        for slice_name, flag_key in SLICES.items():
            flag = np.ones_like(have) if flag_key is None else flags[flag_key]
            # spatial: one vector of the Primaries per half-hour
            share = (flag[:, steps] & have[:, steps]).sum(0) / np.maximum(have[:, steps].sum(0), 1)
            keep = (have[:, steps].sum(0) >= MIN_DIM["spatial"]) & (
                (share >= 0.5) if flag_key is not None else True
            )
            if keep.any():
                sel = steps[keep]
                y = np.nan_to_num(actual[:, sel].T)
                mask = have[:, sel].T
                for arm, path in paths.items():
                    x = path[:, :, sel].transpose(2, 0, 1)
                    es, vs = score_blocks(x=x, y=y, mask=mask, chunk=60)
                    rows.append(
                        _row(
                            init,
                            "spatial",
                            window,
                            slice_name,
                            arm,
                            "all",
                            es.mean(),
                            vs.mean(),
                            int(keep.sum()),
                        )
                    )
            # temporal: each Primary across leads
            sub = steps[:: TEMPORAL_STRIDE[window]]
            tmask = have[:, sub] & (flag[:, sub] if flag_key is not None else True)
            ok = tmask.sum(1) >= MIN_DIM["temporal"] if window == "days_1_10" else tmask.sum(1) >= 6
            if ok.any():
                y = np.nan_to_num(actual[:, sub][ok])
                for arm, path in paths.items():
                    x = path[:, :, sub].transpose(1, 0, 2)[ok]
                    es, vs = score_blocks(x=x, y=y, mask=tmask[ok], chunk=4)
                    for k, p in enumerate(np.flatnonzero(ok)):
                        rows.append(
                            _row(
                                init,
                                "temporal",
                                window,
                                slice_name,
                                arm,
                                labels[p],
                                es[k],
                                vs[k],
                                int(tmask[p].sum()),
                            )
                        )
        # joint: both, coarser leads, no slice
        sub = steps[:: JOINT_STRIDE[window]]
        jm = have[:, sub].reshape(1, -1)
        if jm.sum() >= (MIN_DIM["joint"] if window == "days_1_10" else 60):
            y = np.nan_to_num(actual[:, sub]).reshape(1, -1)
            for arm, path in paths.items():
                x = path[:, :, sub].reshape(N_MEMBERS, -1)[None]
                es, vs = score_blocks(x=x, y=y, mask=jm, chunk=1)
                rows.append(
                    _row(init, "joint", window, "all", arm, "all", es[0], vs[0], int(jm.sum()))
                )
    return rows


def _row(
    init: datetime,
    block: str,
    window: str,
    slice_name: str,
    arm: str,
    label: str,
    es: float,
    vs: float,
    dim: int,
) -> dict:
    return {
        "power_fcst_init_time": init, "block": block, "window": window, "slice": slice_name,
        "arm": arm, "label": label, "energy": float(es), "variogram": float(vs), "n_dim": dim,
    }  # fmt: skip


def crps_of_sorted(*, values: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the CRPS of equally weighted sorted values (..., 51) against the actual (...)."""
    n = values.shape[-1]
    coeff = 2 * np.arange(1, n + 1) - n - 1
    spread = (coeff * values).sum(-1) * 2 / n**2
    return np.abs(values - actual[..., None]).mean(-1) - 0.5 * spread


# --- driver ---------------------------------------------------------------------------------------


def process_month(  # noqa: PLR0915
    *,
    month: str,
    series: pl.DataFrame,
    ctx: dict,
    grid: Grid,
    table: pl.DataFrame,
    max_inits: int | None = None,
) -> None:
    """Score one month's runs and write `coherence_<month>.parquet` and `marginal_<month>.parquet`."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_inits is None else f"_smoke{max_inits}"
    done = WORK_DIR / f"coherence_{month}{suffix}.parquet"
    if done.exists():
        print(f"{month}: exists")
        return
    stored = stored_arm_a(month=month)
    ids = series["time_series_id"].to_list()
    labels = series["label"].to_list()
    capacity = np.array(series["capacity_mw"].to_list())
    inits = stored["power_fcst_init_time"].unique().sort().to_list()[:max_inits]
    coherence: list[dict] = []
    marginal: list[pl.DataFrame] = []
    pool_sizes: list[dict] = []
    for init in inits:
        sub = stored.filter(pl.col("power_fcst_init_time") == init).sort(
            "time_series_id", "valid_time"
        )
        order = {sid: i for i, sid in enumerate(ids)}
        sub = sub.with_columns(
            p=pl.col("time_series_id").replace_strict(order, return_dtype=pl.Int64)
        ).sort("p", "valid_time")
        if sub.height != len(ids) * N_STEPS:
            msg = f"{init}: stored rows {sub.height}"
            raise ValueError(msg)
        quantiles = sub.select(LEVELS).to_numpy().reshape(len(ids), N_STEPS, len(LEVELS))
        values = fifty_one_values(quantiles=quantiles)  # points: stored in fractions of capacity
        values = values * 100.0
        valid = sub["valid_time"].to_numpy().reshape(len(ids), N_STEPS)[0]
        expected = np.datetime64(init.replace(tzinfo=None), "us") + (
            np.arange(1, N_STEPS + 1) * 30 * 60 * 1_000_000
        ).astype("timedelta64[us]")
        if not np.array_equal(valid.astype("datetime64[us]"), expected):
            msg = f"{init}: valid times are not the 702 half-hours after the run"
            raise ValueError(msg)
        t_member, sw_member = run_weather(init=init, ids=ids)
        rng = np.random.default_rng(
            int(hashlib.sha256(init.isoformat().encode()).hexdigest()[:8], 16)
        )
        pool, n_dates = schaake_pool(init=init, grid=grid)
        pool_sizes.append({"power_fcst_init_time": init, "schaake_dates": n_dates})
        identity = np.broadcast_to(np.arange(N_MEMBERS), values.shape)
        paths = {
            "ECC-T": assign(values=values, ranks=ranks_by_key(key=-t_member, rng=rng)),
            "ECC-SW": assign(values=values, ranks=ranks_by_key(key=-sw_member, rng=rng)),
            "SCH": assign(values=values, ranks=ranks_by_key(key=pool.transpose(1, 2, 0), rng=rng)),
            "IND": assign(values=values, ranks=rng.permuted(identity, axis=-1)),
            "CO": assign(values=values, ranks=identity),
        }
        i0 = grid.index(t=init) + 1
        actual = grid.values[:, i0 : i0 + N_STEPS].copy()
        tbl = table.filter(pl.col("power_fcst_init_time") == init)
        flags = {}
        for name in ("time_midday", "cold_nights"):
            arr = np.zeros((len(ids), N_STEPS), dtype=bool)
            part = tbl.filter(pl.col(name)).select("time_series_id", "valid_time")
            if part.height:
                pi = part["time_series_id"].replace_strict(order, return_dtype=pl.Int64).to_numpy()
                si = ((part["valid_time"] - init).dt.total_minutes() // 30).to_numpy() - 1
                arr[pi, si] = True
            flags[name] = arr
        # Score only the rows the shared slice table holds (the rows with a cleaned actual).
        scored = np.zeros((len(ids), N_STEPS), dtype=bool)
        pi = tbl["time_series_id"].replace_strict(order, return_dtype=pl.Int64).to_numpy()
        si = ((tbl["valid_time"] - init).dt.total_minutes() // 30).to_numpy() - 1
        scored[pi, si] = True
        actual[~scored] = np.nan
        coherence.extend(
            coherence_rows(paths=paths, actual=actual, flags=flags, labels=labels, init=init)
        )
        crps = crps_of_sorted(values=values, actual=np.nan_to_num(actual))
        lead = (np.arange(1, N_STEPS + 1) * 0.5)[None, :].repeat(len(ids), 0)
        marginal.append(
            pl.DataFrame(
                {
                    "time_series_id": np.repeat(np.array(ids), N_STEPS).astype(np.int32),
                    "label": np.repeat(np.array(labels), N_STEPS),
                    "power_fcst_init_time": [init] * (len(ids) * N_STEPS),
                    "lead_hours": lead.ravel(),
                    "scored": scored.ravel(),
                    "crps51": crps.ravel(),
                    "above_max": (actual > values[..., -1]).ravel(),
                    "below_min": (actual < values[..., 0]).ravel(),
                    "daylight": flags["time_midday"].ravel(),
                    "cold_night": flags["cold_nights"].ravel(),
                    "capacity_mw": np.repeat(capacity, N_STEPS),
                }
            ).filter(pl.col("scored"))
        )
        print(f"{month} {init:%Y-%m-%d}: {n_dates} Schaake dates", flush=True)
    pl.concat(marginal).write_parquet(WORK_DIR / f"marginal_{month}{suffix}.parquet")
    pl.DataFrame(pool_sizes).write_parquet(WORK_DIR / f"pool_{month}{suffix}.parquet")
    pl.DataFrame(coherence).write_parquet(done)


def main() -> None:
    """Score the requested months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--max-inits", type=int, default=None)
    args = parser.parse_args()
    series = adapter.series_table()
    ctx = cleaned_context(series=series)
    grid = Grid(series=series, actuals=ctx["actuals"])
    for month in args.months:
        stored = stored_arm_a(month=month)
        frame = adapter.forecast_frame(stored=stored, series=series, unit="fraction")
        table = metrics.shared_slice_table(
            forecasts={"A": frame}, actuals=ctx["actuals"], context=ctx["context"],
            events=ctx["events"], first_readings=ctx["first_readings"],
        )  # fmt: skip
        process_month(
            month=month, series=series, ctx=ctx, grid=grid, table=table, max_inits=args.max_inits
        )


if __name__ == "__main__":
    main()
