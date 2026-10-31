# Copied unchanged from experiment 397 (PR 50, branch mdn-attention-demand).
"""Inputs, rows, and folds for the demand networks of experiment 397.

The rows are experiment 87's control-member training rows (power, derived calendar columns, the
three origin lags) at the 16 Primaries. Two weather schemes feed the networks:

- `member`: the seven weather columns of one ensemble member. The network trains on the control
  member and is applied to each of the 51 members at forecast time.
- `summary`: the ensemble mean and the ensemble standard deviation of each weather column,
  computed once per run, so the network is applied once per run.

Commands (run with `PYTHONPATH` set as `e00397_fit.py` shows):

- `summaries`: write the ensemble mean and standard deviation of every run (the stored 51-member
  frames from 2025-07-01; the archive before that, because only the control member is stored for
  the training months).
- `raw`: write each Primary's raw input matrix for a scheme.
- `check`: compare the archive-built summaries of 2025-07-01 with the stored frame's.
"""

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from e00397_models import GROUPS, Layout  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00397-plain-and-attention-mixture-density-network-for-primary-demand"
)
SUMMARY_DIR: Final[Path] = WORK_DIR / "work" / "ens_summary"
RAW_DIR: Final[Path] = WORK_DIR / "work" / "raw"
SCHEMES: Final[tuple[str, ...]] = ("member", "summary")
WEATHER: Final[tuple[str, ...]] = fd.WEATHER
FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
INPUT_CLIP_SD: Final[float] = 5.0
WINDOW_DAYS: Final[int] = 730
"""A fold trains on the 24 months before its refit."""
HELD_WEEK_PERIOD: Final[int] = 12
"""One week in every 12 is held out for early stopping."""
LEAD_SCALE_HOURS: Final[float] = 336.0
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
"""The four months of `common.md` plus December, which holds Christmas."""
VALIDATION_FIRST: Final[datetime] = datetime(2025, 4, 1, tzinfo=UTC)
VALIDATION_LAST: Final[datetime] = datetime(2025, 6, 30, tzinfo=UTC)
GROUP_NAMES: Final[tuple[str, ...]] = (*GROUPS, "lead")


def primaries() -> pl.DataFrame:
    """Return the 16 Primary series of experiment 87's table, sorted by label number."""
    series = common.series_table().filter(pl.col("substation_type") == "Primary")
    return series.with_row_index("primary")


def run_times() -> list[datetime]:
    """Return the daily 00:00 UTC run times of the summaries."""
    return [FIRST_RUN + timedelta(days=i) for i in range((LAST_RUN - FIRST_RUN).days + 1)]


# --- ensemble summaries ---------------------------------------------------------------------


def summarise(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the mean and (population) standard deviation over members of each weather column."""
    return frame.group_by("time_series_id", "power_fcst_init_time", "valid_time").agg(
        *[pl.col(w).mean().alias(f"mean_{w}") for w in WEATHER],
        *[pl.col(w).std(ddof=0).alias(f"std_{w}") for w in WEATHER],
    )


def summary_of_run(*, run: datetime, targets: pl.DataFrame) -> pl.DataFrame:
    """Return one run's summaries for the Primaries, from the stored frame or from the archive."""
    stored = common.demand_frame_path(run=run)
    if stored.exists():
        frame = pl.read_parquet(stored).filter(
            pl.col("time_series_id").is_in(targets["time_series_id"].to_list())
        )
    else:
        import e00048_build_frames as build  # ty: ignore[unresolved-import]

        frame, _ = build._half_hourly_rows(run=run, targets=targets)
    return summarise(frame=frame).sort("time_series_id", "valid_time")


def write_summary(*, run: datetime) -> str:
    """Write one run's summary file if absent; return a progress line."""
    target = SUMMARY_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{run:%Y-%m-%d}: exists"
    started = time.monotonic()
    targets = primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    table = summary_of_run(run=run, targets=targets).cast(
        dict.fromkeys([f"{k}_{w}" for k in ("mean", "std") for w in WEATHER], pl.Float32)
    )
    common.atomic_write(frame=table, target=target)
    return f"{run:%Y-%m-%d}: {table.height:,} rows, {time.monotonic() - started:.0f} s"


def load_summary(*, series_id: int) -> pl.DataFrame:
    """Return one series' summaries over all runs."""
    return (
        pl.scan_parquet(SUMMARY_DIR / "init=*.parquet")
        .filter(pl.col("time_series_id") == series_id)
        .collect()
    )


# --- feature matrix -------------------------------------------------------------------------


def feature_names(*, scheme: str) -> tuple[list[str], np.ndarray, Layout]:
    """Return the input column names, the mask of columns to standardise, and the layout."""
    weather = (
        [f"w_{w}" for w in WEATHER]
        if scheme == "member"
        else [f"mean_{w}" for w in WEATHER] + [f"std_{w}" for w in WEATHER]
    )
    groups = {
        "weather": weather,
        "solar": ["cos_zenith", "sin_azimuth", "cos_azimuth"],
        "calendar": [
            "sin_hour",
            "cos_hour",
            "sin_hour_2",
            "cos_hour_2",
            *[f"dow_{d}" for d in range(1, 8)],
            "covid",
        ],
        "holidays": [
            "christmas_inside",
            "christmas_days",
            "christmas_tent",
            "bank_holiday",
            "nearest_days",
            "nearest_inside",
        ],
        "lags": [
            "lag_origin",
            "lag_day",
            "lag_week",
            "lag_origin_missing",
            "lag_day_missing",
            "lag_week_missing",
            "lag_origin_age",
            "lag_day_age",
            "lag_week_age",
        ],
        "lead": ["lead", "lead_sqrt"],
    }
    names: list[str] = []
    slices: dict[str, slice] = {}
    for g in GROUP_NAMES:
        slices[g] = slice(len(names), len(names) + len(groups[g]))
        names += groups[g]
    standardise = np.array(
        [
            n.startswith(("w_", "mean_", "std_", "christmas_days", "nearest_days", "lag_"))
            and not n.endswith(("_missing", "_age"))
            for n in names
        ]
    )
    return names, standardise, Layout(slices=slices, n_inputs=len(names))


def lag_ages_hours(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the age in hours of the three origin lags at each lead, shape (rows, 3)."""
    gap = np.rint(lead_hours * 2.0) + common.NEWEST_AGE
    day = 24.0 * np.ceil(gap / common.STEPS_PER_DAY)
    week = 168.0 * np.ceil(gap / (7 * common.STEPS_PER_DAY))
    newest = gap / 2.0
    return np.stack([newest, day, week], axis=1)


def feature_matrix(*, frame: pl.DataFrame, scheme: str) -> np.ndarray:
    """Return the raw (unstandardised) input matrix of a frame, float32, columns of `feature_names`.

    Weather precipitation goes through `log1p`. Lags that are missing are NaN and are flagged here
    and filled after standardisation. Every weather column must be finite.
    """
    names, _, _ = feature_names(scheme=scheme)
    n = frame.height
    cols: dict[str, np.ndarray] = {}
    if scheme == "member":
        for w in WEATHER:
            v = frame[w].to_numpy().astype(np.float64)
            cols[f"w_{w}"] = np.log1p(np.clip(v, 0, None)) if w == "precipitation" else v
    else:
        for k in ("mean", "std"):
            for w in WEATHER:
                v = frame[f"{k}_{w}"].to_numpy().astype(np.float64)
                cols[f"{k}_{w}"] = np.log1p(np.clip(v, 0, None)) if w == "precipitation" else v
    weather_columns = [c for c in cols if c.startswith(("w_", "mean_", "std_"))]
    if not all(np.isfinite(cols[c]).all() for c in weather_columns):
        msg = "a weather input is not finite; add an is-missing flag for it"
        raise ValueError(msg)
    zenith = np.radians(frame["solar_zenith"].to_numpy().astype(np.float64))
    azimuth = np.radians(frame["solar_azimuth"].to_numpy().astype(np.float64))
    cols["cos_zenith"], cols["sin_azimuth"], cols["cos_azimuth"] = (
        np.cos(zenith),
        np.sin(azimuth),
        np.cos(azimuth),
    )
    hour = frame["hour_local"].to_numpy().astype(np.float64)
    for k, cycles in (("", 1.0), ("_2", 2.0)):
        cols[f"sin_hour{k}"] = np.sin(2 * np.pi * cycles * hour / 24.0)
        cols[f"cos_hour{k}"] = np.cos(2 * np.pi * cycles * hour / 24.0)
    dow = frame["day_of_week_local"].to_numpy().astype(np.int64)
    for d in range(1, 8):
        cols[f"dow_{d}"] = (dow == d).astype(np.float64)
    cols["covid"] = frame["covid_indicator"].to_numpy().astype(np.float64)
    christmas = frame["days_to_christmas"].to_numpy().astype(np.float64)
    inside = np.isfinite(christmas)
    scaled = np.where(inside, christmas / common.CHRISTMAS_CLIP_DAYS, 0.0)
    cols["christmas_inside"] = inside.astype(np.float64)
    cols["christmas_days"] = scaled
    cols["christmas_tent"] = np.where(inside, 1.0 - np.abs(scaled), 0.0)
    nearest = frame["days_to_nearest_holiday"].to_numpy().astype(np.float64)
    cols["bank_holiday"] = (nearest == 0).astype(np.float64)
    cols["nearest_days"] = nearest / common.HOLIDAY_CLIP_DAYS
    cols["nearest_inside"] = (np.abs(nearest) < common.HOLIDAY_CLIP_DAYS).astype(np.float64)
    lag_values = np.stack([frame[c].to_numpy().astype(np.float64) for c in common.LAG_COLUMNS], 1)
    missing = ~np.isfinite(lag_values)
    ages = lag_ages_hours(lead_hours=frame["lead_hours"].to_numpy().astype(np.float64))
    for i, tag in enumerate(("origin", "day", "week")):
        cols[f"lag_{tag}"] = lag_values[:, i]
        cols[f"lag_{tag}_missing"] = missing[:, i].astype(np.float64)
        cols[f"lag_{tag}_age"] = ages[:, i] / LEAD_SCALE_HOURS
    lead = frame["lead_hours"].to_numpy().astype(np.float64) / LEAD_SCALE_HOURS
    cols["lead"], cols["lead_sqrt"] = lead, np.sqrt(np.clip(lead, 0, None))
    out = np.empty((n, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        out[:, j] = cols[name]
    return out


@dataclass(frozen=True)
class Standardiser:
    """Training mean and standard deviation of the standardised columns, applied unchanged later."""

    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray

    @classmethod
    def fit(cls, *, x: np.ndarray, mask: np.ndarray) -> Standardiser:
        """Fit on the training rows (NaN-aware, for the lag columns)."""
        mean = np.where(mask, np.nanmean(x, axis=0), 0.0)
        std = np.where(mask, np.nanstd(x, axis=0), 1.0)
        return cls(mean=mean, std=np.maximum(std, 1e-6), mask=mask)

    def apply(self, *, x: np.ndarray) -> np.ndarray:
        """Standardise, clip at five standard deviations, and fill NaN with 0."""
        out = (x - self.mean) / self.std
        out = np.clip(out, -INPUT_CLIP_SD, INPUT_CLIP_SD)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)


# --- per-Primary raw matrices ---------------------------------------------------------------


def series_frame(*, site: str, scheme: str) -> pl.DataFrame:
    """Return a Primary's control-member rows with the scheme's weather columns."""
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    rows = fd.training_rows(row=row, arrays=arrays[site], frozen=False, a7=False)
    if scheme == "summary":
        keys = ["power_fcst_init_time", "valid_time"]
        summary = load_summary(series_id=row["time_series_id"]).drop("time_series_id")
        rows = rows.join(summary, on=keys, how="inner", validate="m:1", maintain_order="left")
    return rows.with_columns(capacity=pl.lit(float(row["effective_capacity_mw"])))


def write_raw(*, site: str, scheme: str) -> str:
    """Write a Primary's raw matrix, target, and keys for a scheme."""
    target = RAW_DIR / scheme / f"{site}.npz"
    if target.exists():
        return f"{scheme} {site}: exists"
    started = time.monotonic()
    rows = series_frame(site=site, scheme=scheme)
    x = feature_matrix(frame=rows, scheme=scheme)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(
        temporary,
        x=x,
        y=(rows["power"] / rows["capacity"]).to_numpy().astype(np.float32),
        valid=rows["valid_time"].dt.epoch("us").to_numpy(),
        origin=rows["power_fcst_init_time"].dt.epoch("us").to_numpy(),
        lead=rows["lead_hours"].to_numpy().astype(np.float32),
    )
    temporary.rename(target)
    return f"{scheme} {site}: {len(x):,} rows, {time.monotonic() - started:.0f} s"


@dataclass
class FoldTrain:
    """The standardised training rows of one fold, ready for a network."""

    x: np.ndarray
    y: np.ndarray
    primary: np.ndarray
    weight: np.ndarray
    is_val: np.ndarray
    lead: np.ndarray
    origin_day: np.ndarray
    standardiser: Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    layout: Layout


def _us(*, t: datetime) -> int:
    """Return a UTC datetime as microseconds since the epoch."""
    return int(t.timestamp() * 1_000_000)


def held_out_weeks(*, origin_day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return masks of early-stopping rows and of purged rows, by run origin day.

    One week in every 12 is held out. A training run is purged when it begins up to 14 days before
    a held-out week (its leads reach into the week) or up to 7 days after it (its newest lags lie
    in the week).
    """
    week = origin_day // 7
    phase = week % HELD_WEEK_PERIOD
    held = phase == HELD_WEEK_PERIOD - 1
    purged = np.isin(phase, [HELD_WEEK_PERIOD - 3, HELD_WEEK_PERIOD - 2, 0])
    return held, purged


def build_train(*, scheme: str, refit: datetime, validation: bool = False) -> FoldTrain:
    """Return the standardised training rows of a fold.

    A fold sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit (experiment 87's rule), within the 24 months before the refit.
    With `validation`, the held-out rows are the validation block (runs from 2025-04-01) instead of
    every twelfth week, and the block's runs and the 14 days before them are not trained on.
    """
    _, mask, layout = feature_names(scheme=scheme)
    table = primaries()
    parts = []
    for row in table.iter_rows(named=True):
        z = np.load(RAW_DIR / scheme / f"{row['site']}.npz")
        keep = (
            (z["valid"] <= _us(t=refit - regimes.REFIT_CUTOFF_AGE))
            & (z["origin"] <= _us(t=refit))
            & (z["valid"] > _us(t=refit - timedelta(days=WINDOW_DAYS)))
        )
        if not keep.any():
            continue
        parts.append(
            {
                "x": z["x"][keep],
                "y": z["y"][keep],
                "origin": z["origin"][keep],
                "valid": z["valid"][keep],
                "lead": z["lead"][keep],
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
            }
        )
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    origin_day = ((cat["origin"] - _us(t=FIRST_RUN)) // (86_400 * 1_000_000)).astype(np.int64)
    if validation:
        first_day = (VALIDATION_FIRST - FIRST_RUN).days
        held = origin_day >= first_day
        purged = (origin_day >= first_day - 14) & ~held
    else:
        held, purged = held_out_weeks(origin_day=origin_day)
    train = ~held & ~purged
    standardiser = Standardiser.fit(x=cat["x"][train], mask=mask)
    age_days = (_us(t=refit) - cat["valid"]) / (86_400 * 1_000_000)
    weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
    n_primaries = len(table)
    target_mean, target_std = np.zeros(n_primaries), np.ones(n_primaries)
    for p in range(n_primaries):
        sel = train & (cat["primary"] == p)
        if sel.any():
            target_mean[p] = cat["y"][sel].mean()
            target_std[p] = max(float(cat["y"][sel].std()), 1e-3)
    keep = train | held
    x_kept = standardiser.apply(x=cat["x"][keep])
    del cat["x"]
    return FoldTrain(
        x=x_kept,
        y=cat["y"][keep],
        primary=cat["primary"][keep],
        weight=weight[keep].astype(np.float32),
        is_val=held[keep],
        lead=cat["lead"][keep],
        origin_day=origin_day[keep],
        standardiser=standardiser,
        target_mean=target_mean,
        target_std=target_std,
        layout=layout,
    )


def test_origins(*, month: str) -> list[datetime]:
    """Return the daily `power_fcst_init_time` of the runs whose origin falls in a month."""
    start = datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)
    stop = (start + timedelta(days=32)).replace(day=1)
    return [o for o in common.all_origins() if start <= o < stop]


def refit_of(*, month: str) -> datetime:
    """Return a fold's refit time: 00:00 UTC on the first day of its test month."""
    return datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC)


def run_inputs(
    *, origin: datetime, scheme: str, standardiser: Standardiser
) -> list[tuple[dict, pl.DataFrame, np.ndarray]]:
    """Return, for one forecast run, each Primary's keys, 51-member or summary rows, and inputs.

    Each item is (series row, keys frame, standardised inputs). For `member`, the inputs hold the
    51 members of each valid time in consecutive rows (valid time major). For `summary` they hold
    one row per valid time.
    """
    _series, arrays = fd._series_and_arrays()
    frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
    out = []
    for row in primaries().iter_rows(named=True):
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=arrays[row["site"]],
            a7=False,
        )
        if scheme == "summary":
            summary = pl.read_parquet(
                SUMMARY_DIR / f"init={origin - common.PUBLICATION_DELAY:%Y-%m-%d}.parquet"
            ).filter(pl.col("time_series_id") == row["time_series_id"])
            first = prepared.filter(pl.col("ensemble_member") == pl.col("ensemble_member").min())
            prepared = first.join(
                summary.drop("time_series_id"),
                on=["power_fcst_init_time", "valid_time"],
                how="inner",
                validate="1:1",
                maintain_order="left",
            )
        keys = prepared.select(*common.KEYS, "lead_hours").unique(
            subset=list(common.KEYS), maintain_order=True
        )
        out.append(
            (
                row,
                keys,
                standardiser.apply(x=feature_matrix(frame=prepared, scheme=scheme)),
            )
        )
    return out


def check_archive_summary() -> str:
    """Compare the archive-built summaries of the 2025-07-01 run with the stored frame's."""
    run = datetime(2025, 7, 1, tzinfo=UTC)
    targets = primaries().select("time_series_id", "h3_res_5", "latitude", "longitude")
    import e00048_build_frames as build  # ty: ignore[unresolved-import]

    rows, _ = build._half_hourly_rows(run=run, targets=targets)
    built = summarise(frame=rows).sort("time_series_id", "valid_time")
    stored = summarise(
        frame=pl.read_parquet(common.demand_frame_path(run=run)).filter(
            pl.col("time_series_id").is_in(targets["time_series_id"].to_list())
        )
    ).sort("time_series_id", "valid_time")
    joined = built.join(stored, on=["time_series_id", "valid_time"], suffix="_stored")
    worst = 0.0
    for k in ("mean", "std"):
        for w in WEATHER:
            built_values = joined[f"{k}_{w}"].to_numpy().astype(np.float64)
            stored_values = joined[f"{k}_{w}_stored"].to_numpy().astype(np.float64)
            gap = np.abs(built_values - stored_values) / (np.abs(stored_values) + 1e-6)
            worst = max(worst, float(gap.max()))
    return f"{joined.height:,} shared rows, largest relative gap {worst:.2e}"


def main() -> None:
    """Build the summaries or the raw matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summaries", "raw", "check"))
    parser.add_argument("--scheme", choices=SCHEMES, default="member")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="3 runs, or 2 Primaries")
    args = parser.parse_args()
    if args.command == "check":
        print(check_archive_summary())
    elif args.command == "summaries":
        runs = run_times()[:3] if args.smoke else run_times()
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for line in pool.map(_write_summary_kw, runs):
                print(line, flush=True)
    else:
        sites = primaries()["site"].to_list()[: 2 if args.smoke else None]
        for site in sites:
            print(write_raw(site=site, scheme=args.scheme), flush=True)


def _write_summary_kw(run: datetime) -> str:
    """Return `write_summary` for a run, as a picklable function of one argument."""
    return write_summary(run=run)


if __name__ == "__main__":
    main()
