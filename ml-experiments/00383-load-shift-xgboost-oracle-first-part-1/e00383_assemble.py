"""Assemble experiment 383's arms from the stored refit predictions (no fits).

Reads `work/arm_preds/<arm>/<site>_s0_<refit>.parquet` for the arms `c1` and `lagless`, the stored
weekly member of experiment 87 (`n1h`, read only) as arm `d` once `e00383_check_d.py` has passed the
reproduction gate, and the event masks of the feature cache. Writes each arm to
`work/arms/<arm>.parquet` (keys, 11 quantiles as fractions of effective capacity, `lead_hours`) and
the table of contaminated-lag rows to `work/arms/contaminated.parquet`.

Arms written:

- `d`, `c1`, `lagless`: the raw mixtures.
- `c2`: `lagless` on the contaminated-lag rows (a lag inside an event), `d` elsewhere.
- `<arm>_w`: the arm with the fixed widening factor k = 1.28 on the contaminated-lag rows.
- `<arm>_wa`: the arm with the as-of widening factor on the contaminated-lag rows (see
  `as_of_factors`), for `d`, `c1`, and `c2`.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00383-* \
    uv run --no-sync python ml-experiments/00383-*/e00383_assemble.py
```
"""

from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00383_fit as fit
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = fit.WORK_DIR
ARM_DIR: Final[Path] = fit.ARM_DIR
OUT_DIR: Final[Path] = WORK_DIR / "arms"
STORED_87: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00087-bg2-best-guess-assembled-with-drop-one-ablations" / "work"
)
FIXED_K: Final[float] = 1.28
"""z(0.95) / z(0.90): takes a coverage of 0.80 to 0.90 for a Gaussian error. Tuning-adjacent."""
K_CAP: Final[float] = 2.0
K_GRID: Final[np.ndarray] = np.round(np.arange(1.0, K_CAP + 1e-9, 0.02), 2)
NOMINAL_BAND: Final[float] = 0.90
LEVELS: Final[tuple[str, ...]] = tuple(common.LEVEL_COLUMNS)
WEEKLY_REFITS: Final[tuple[datetime, ...]] = regimes.REGIMES["weekly"]


def read_arm(*, directory: Path, name: str, sites: list[str], stored: bool) -> pl.DataFrame:
    """Return an arm's per-run quantiles for the given sites, one row per key.

    A stored file of experiment 87 holds the origins of both regimes, so for `stored` only the
    origins that the weekly regime assigns to the file's refit are kept.
    """
    pieces = []
    for site in sites:
        for refit in WEEKLY_REFITS:
            path = fit_job_path(directory=directory, name=name, site=site, refit=refit)
            if not path.exists():
                continue
            frame = pl.read_parquet(path)
            if stored:
                origins = [
                    t
                    for t in frame["power_fcst_init_time"].unique().to_list()
                    if regimes.chosen_refit(refits=WEEKLY_REFITS, origin=t) == refit
                ]
                frame = frame.filter(pl.col("power_fcst_init_time").is_in(origins))
            pieces.append(frame)
    if not pieces:
        msg = f"no files for arm {name} in {directory}"
        raise ValueError(msg)
    return (
        pl.concat(pieces)
        .drop("seed")
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
    )


def fit_job_path(*, directory: Path, name: str, site: str, refit: datetime) -> Path:
    """Return the file of one refit's predictions for one series (experiment 87's naming)."""
    return directory / name / f"{site}_s0_{refit:%Y%m%dT%H%M}.parquet"


def contaminated_flags(*, frame: pl.DataFrame, masks: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the frame's keys with `lag_in_event` (any of the three lags inside an event)."""
    series = common.series_table().select("time_series_id", "site")
    label_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    pieces = []
    for series_id, part in frame.select(*common.KEYS).group_by(
        "time_series_id", maintain_order=True
    ):
        label = label_of[series_id[0]]
        target = np.asarray(common.grid_index(times=part["valid_time"]))
        origin = np.asarray(common.grid_index(times=part["power_fcst_init_time"]))
        flags = fit.lag_flags(mask=masks[label], target=target, origin=origin)
        pieces.append(part.with_columns(lag_in_event=pl.Series(flags.any(axis=1))))
    return pl.concat(pieces)


def widen(
    *, frame: pl.DataFrame, rows: pl.Series | pl.Expr, factor: float | pl.Expr
) -> pl.DataFrame:
    """Scale each quantile's distance from the median by `factor` where `rows` is true."""
    median = pl.col("q50")
    return frame.with_columns(
        pl.when(rows).then(median + (pl.col(c) - median) * factor).otherwise(pl.col(c)).alias(c)
        for c in LEVELS
    )


def band_coverage(*, frame: pl.DataFrame, actual_fraction: pl.Series, factor: float) -> float:
    """Return the share of readings inside the 5% to 95% band widened by `factor`."""
    median = frame["q50"].to_numpy()
    lower = median + (frame["q05"].to_numpy() - median) * factor
    upper = median + (frame["q95"].to_numpy() - median) * factor
    actual = actual_fraction.to_numpy()
    return float(np.mean((actual >= lower) & (actual <= upper)))


def as_of_factors(
    *, frame: pl.DataFrame, nulled: pl.Series, actual_fraction: pl.Series
) -> dict[str, float]:
    """Return the as-of factor of each origin month, from the nulled rows of earlier months.

    A month's factor is the smallest k on `K_GRID` whose 5% to 95% band covers at least
    `NOMINAL_BAND` of the readings on the nulled rows of all earlier origin months, capped at
    `K_CAP`, and `FIXED_K` while no earlier month has any nulled row. The nulled rows of a month
    are used only after the month has ended, so the factor uses no reading that is newer than the
    month's first forecast origin (the readings are the actuals up to the end of the earlier month
    plus the longest lead, which the caller restricts to by month of valid time).
    """
    month = frame["power_fcst_init_time"].dt.strftime("%Y-%m")
    valid_month = frame["valid_time"].dt.strftime("%Y-%m")
    factors: dict[str, float] = {}
    for current in sorted(month.unique().to_list()):
        known = nulled & (valid_month < current) & actual_fraction.is_finite()
        if known.sum() == 0:
            factors[current] = FIXED_K
            continue
        part, truth = frame.filter(known), actual_fraction.filter(known)
        factors[current] = K_CAP
        for k in K_GRID:
            if band_coverage(frame=part, actual_fraction=truth, factor=float(k)) >= NOMINAL_BAND:
                factors[current] = float(k)
                break
    return factors


def write_arm(*, name: str, frame: pl.DataFrame) -> None:
    """Write one arm."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT_DIR / f"{name}.parquet")
    print(f"wrote {name}: {frame.height:,} rows", flush=True)


def main() -> None:
    """Assemble and write every arm."""
    series = common.series_table()
    sites = series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    masks = fit.event_masks()
    d = read_arm(directory=STORED_87 / "anchor_preds", name="n1h", sites=sites, stored=True)
    c1 = read_arm(directory=ARM_DIR, name="c1", sites=sites, stored=False)
    lagless = read_arm(directory=ARM_DIR, name="lagless", sites=sites, stored=False)
    keys = list(common.KEYS)
    shared = d.select(keys).join(c1.select(keys), on=keys).join(lagless.select(keys), on=keys)
    d, c1, lagless = (f.join(shared, on=keys, how="semi").sort(keys) for f in (d, c1, lagless))
    flags = contaminated_flags(frame=d, masks=masks).sort(keys)
    if not (d.select(keys).equals(c1.select(keys)) and d.select(keys).equals(flags.select(keys))):
        msg = "arm keys differ after the join"
        raise ValueError(msg)
    nulled = flags["lag_in_event"]
    c2 = d.with_columns(
        pl.when(nulled).then(lagless[c]).otherwise(pl.col(c)).alias(c) for c in LEVELS
    )
    if not c2.filter(~nulled).select(LEVELS).equals(d.filter(~nulled).select(LEVELS)):
        msg = "c2 differs from d outside the contaminated-lag rows"
        raise ValueError(msg)
    write_arm(name="contaminated", frame=flags)
    arms = {"d": d, "c1": c1, "lagless": lagless, "c2": c2}
    actual_fraction = _actual_fraction(frame=d)
    for name, frame in arms.items():
        write_arm(name=name, frame=frame)
        if name == "lagless":
            continue
        write_arm(name=f"{name}_w", frame=widen(frame=frame, rows=nulled, factor=FIXED_K))
        factors = as_of_factors(frame=frame, nulled=nulled, actual_fraction=actual_fraction)
        print(name, "as-of k by origin month:", factors, flush=True)
        month = frame["power_fcst_init_time"].dt.strftime("%Y-%m")
        k = month.replace_strict(factors, return_dtype=pl.Float64)
        write_arm(name=f"{name}_wa", frame=widen(frame=frame, rows=nulled, factor=k))


def _actual_fraction(*, frame: pl.DataFrame) -> pl.Series:
    """Return the observed power over effective capacity at each key of `frame`."""
    series = common.series_table()
    arrays = common.load_series_arrays(series=series)
    site_of = dict(zip(series["time_series_id"], series["site"], strict=True))
    out = np.full(frame.height, np.nan)
    ids = frame["time_series_id"].to_numpy()
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    for series_id in np.unique(ids):
        rows = np.flatnonzero(ids == series_id)
        out[rows] = common.take(values=arrays[site_of[series_id]].power, index=target[rows])
    return pl.Series("actual_fraction", out)


if __name__ == "__main__":
    main()
